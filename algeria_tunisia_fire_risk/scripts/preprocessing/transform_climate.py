#!/usr/bin/env python3
"""Transform monthly climate NetCDF into tabular feature sets.

Input expected: CLIMATE/processed/climate_2024.nc (or DATA_CLEANED/CLIMATE/climate_2024_clean.nc)
Variables typically: prec, tmax, tmin with dimensions (time, y, x) or (time, lat, lon).

Modes:
  monthly-wide: produce one row per (lat, lon) with 12 columns per variable (e.g., prec_01 .. prec_12)
  stats: aggregate each variable across months into mean, min, max, std (4 per variable => 12 total)
  seasonal: aggregate months into climatological seasons (DJF, MAM, JJA, SON) using mean (4 per variable => 12 total)
  long: keep long format (lat, lon, month, variable columns) for later temporal joins (e.g., fire acquisition date)

Additional derived features (optional flags):
  --add-derived: adds t_range = tmax_mean - tmin_mean (for stats/seasonal), prec_total (sum), and dryness_index = tmax_mean / (prec_mean+1)

Cleaning / preprocessing:
  - Missing months for a pixel: impute by variable mean over available months.
  - Outlier capping: IQR winsorization per aggregated column (--cap-iqr flag).
  - Precip log1p transform optionally (--log-prec) applied after capping.

Outputs:
  Parquet + CSV written to output directory with names:
    climate_features_<mode>.parquet / .csv
  EDA summary JSON (if --eda) climate_features_<mode>_summary.json

Usage examples:
  python scripts/transform_climate.py --mode stats --eda --cap-iqr --log-prec --add-derived
  python scripts/transform_climate.py --mode seasonal --include-dec --positive-year 2024

"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parents[1]
DEFAULT_IN = [
    BASE / "CLIMATE" / "processed" / "climate_2024.nc",
    BASE / "DATA_CLEANED" / "CLIMATE" / "climate_2024_clean.nc",
    BASE / "DATA_CLEANED" / "CLIMATE" / "climate_2024.nc",
]

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

def winsorize_iqr(vals: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, int, float, float]:
    finite = vals[np.isfinite(vals)]
    if finite.size < 10:
        return vals, 0, np.nan, np.nan
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return vals, 0, np.nan, np.nan
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    mask = (vals < lo) | (vals > hi)
    out_count = int(np.isfinite(vals[mask]).sum())
    capped = np.where(vals < lo, lo, np.where(vals > hi, hi, vals))
    return capped, out_count, lo, hi

def load_dataset(paths: List[Path]) -> xr.Dataset:
    for p in paths:
        if p.exists():
            try:
                return xr.open_dataset(p)
            except Exception:
                continue
    raise FileNotFoundError("No climate NetCDF found in expected locations.")

def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    # Ensure lon/lat coordinate names
    rename = {}
    if "x" in ds.coords and "lon" not in ds.coords:
        rename["x"] = "lon"
    if "y" in ds.coords and "lat" not in ds.coords:
        rename["y"] = "lat"
    if rename:
        ds = ds.rename(rename)
    return ds

def month_numbers(ds: xr.Dataset) -> np.ndarray:
    if "time" not in ds.dims:
        raise ValueError("Dataset missing time dimension for monthly transformation.")
    # Convert times to pandas DatetimeIndex
    times = pd.to_datetime(ds["time"].values)
    return times.month.values

def pivot_monthly(ds: xr.Dataset, vars_: List[str]) -> pd.DataFrame:
    """Vectorized monthly wide pivot: one row per spatial cell with 12 columns per variable."""
    months = month_numbers(ds)
    # Standardize dims names
    ydim = [d for d in ds.dims if d in ("lat","y")][0]
    xdim = [d for d in ds.dims if d in ("lon","x")][0]
    ny, nx = ds.dims[ydim], ds.dims[xdim]
    lat = ds["lat"].values if "lat" in ds.coords else ds[ydim].values
    lon = ds["lon"].values if "lon" in ds.coords else ds[xdim].values
    out = {"lat": np.repeat(lat, nx),
            "lon": np.tile(lon, ny)}
    for vi in vars_:
        data = ds[vi].values  # shape (time, y, x)
        # Impute NaNs per cell with its monthly mean if present
        if np.isnan(data).any():
            cell_mean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = cell_mean[inds[1], inds[2]]
        # For each month, flatten spatial (y,x)
        for i, m in enumerate(months):
            out[f"{vi}_{m:02d}"] = data[i].reshape(-1)
    df = pd.DataFrame(out)
    return df

def _impute_time_nan(data: np.ndarray, global_fill: float | None) -> tuple[np.ndarray, int]:
    """Impute NaNs per (y,x) cell using its time-mean; if entire cell is NaN use global_fill.
    Returns (imputed_data, n_cells_global_filled)
    data shape (time, y, x)
    """
    # Safe cell mean computation without warnings
    with np.errstate(invalid='ignore'):  # ignore warnings, we manually handle all-NaN
        cell_mean = np.nanmean(data, axis=0)
    all_nan_mask = np.isnan(cell_mean)
    inds = np.where(np.isnan(data))
    n_global = int(all_nan_mask.sum()) if global_fill is not None else 0
    if global_fill is not None and n_global:
        cell_mean_filled = cell_mean.copy()
        cell_mean_filled[all_nan_mask] = global_fill
    else:
        cell_mean_filled = cell_mean
    for t, y, x in zip(*inds):
        data[t, y, x] = cell_mean_filled[y, x]
    return data, n_global

def aggregate_stats_vectorized(ds: xr.Dataset, vars_: List[str], quantiles: bool=False, global_fill: bool=False, drop_allnan: bool=False, return_monthly: bool=False) -> tuple[pd.DataFrame, dict | None]:
    # return_monthly True returns monthly arrays for precip total sum later
    ydim = [d for d in ds.sizes if d in ("lat","y")][0]
    xdim = [d for d in ds.sizes if d in ("lon","x")][0]
    ny, nx = ds.sizes[ydim], ds.sizes[xdim]
    lat = ds["lat"].values if "lat" in ds.coords else ds[ydim].values
    lon = ds["lon"].values if "lon" in ds.coords else ds[xdim].values
    out = {"lat": np.repeat(lat, nx), "lon": np.tile(lon, ny)}
    monthly_store = {}
    for vi in vars_:
        data = ds[vi].values.astype(float)  # (time, y, x)
        global_mean = float(np.nanmean(data)) if global_fill and np.isfinite(np.nanmean(data)) else (0.0 if global_fill else None)
        data, _ = _impute_time_nan(data, global_mean)
        cell_all_nan_mask = np.all(~np.isfinite(data), axis=0)
        keep_idx = ~cell_all_nan_mask.reshape(-1) if drop_allnan else np.ones(ny*nx, dtype=bool)
        # compute stats guarded
        with np.errstate(invalid='ignore'):  # suppress warnings
            mean_arr = np.nanmean(data, axis=0)
            min_arr = np.nanmin(data, axis=0)
            max_arr = np.nanmax(data, axis=0)
            std_arr = np.nanstd(data, axis=0)
            out[f"{vi}_mean"] = mean_arr.reshape(-1)[keep_idx]
            out[f"{vi}_min"] = min_arr.reshape(-1)[keep_idx]
            out[f"{vi}_max"] = max_arr.reshape(-1)[keep_idx]
            out[f"{vi}_std"] = std_arr.reshape(-1)[keep_idx]
            if quantiles:
                out[f"{vi}_p10"] = np.nanpercentile(data, 10, axis=0).reshape(-1)[keep_idx]
                out[f"{vi}_p90"] = np.nanpercentile(data, 90, axis=0).reshape(-1)[keep_idx]
        if vi == vars_[0] and drop_allnan:
            base_lat = np.repeat(lat, nx)
            base_lon = np.tile(lon, ny)
            out["lat"] = base_lat[keep_idx]
            out["lon"] = base_lon[keep_idx]
        if return_monthly:
            monthly_store[vi] = data[:, :, :].reshape(data.shape[0], -1)[:, keep_idx]  # (time, cells_kept)
    return pd.DataFrame(out), (monthly_store if return_monthly else None)

def aggregate_seasonal_vectorized(ds: xr.Dataset, vars_: List[str], include_dec: bool, quantiles: bool=False, global_fill: bool=False, drop_allnan: bool=False) -> pd.DataFrame:
    months = month_numbers(ds)
    season_map = SEASONS.copy()
    if not include_dec:
        season_map["DJF"] = [1,2]
    ydim = [d for d in ds.sizes if d in ("lat","y")][0]
    xdim = [d for d in ds.sizes if d in ("lon","x")][0]
    ny, nx = ds.sizes[ydim], ds.sizes[xdim]
    lat = ds["lat"].values if "lat" in ds.coords else ds[ydim].values
    lon = ds["lon"].values if "lon" in ds.coords else ds[xdim].values
    out = {"lat": np.repeat(lat, nx), "lon": np.tile(lon, ny)}
    for vi in vars_:
        data = ds[vi].values.astype(float)
        global_mean = float(np.nanmean(data)) if global_fill else None
        data, n_global = _impute_time_nan(data, global_mean)
        cell_all_nan_mask = np.all(~np.isfinite(data), axis=0)
        if drop_allnan:
            keep_idx = ~cell_all_nan_mask.reshape(-1)
        else:
            keep_idx = np.ones(ny*nx, dtype=bool)
        for season, mlist in season_map.items():
            sel = [i for i, m in enumerate(months) if m in mlist]
            if sel:
                season_data = data[sel]
                arr_mean = np.nanmean(season_data, axis=0).reshape(-1)[keep_idx]
                out[f"{vi}_{season}"] = arr_mean
                if quantiles:
                    out[f"{vi}_{season}_p10"] = np.nanpercentile(season_data, 10, axis=0).reshape(-1)[keep_idx]
                    out[f"{vi}_{season}_p90"] = np.nanpercentile(season_data, 90, axis=0).reshape(-1)[keep_idx]
            else:
                out[f"{vi}_{season}"] = np.full(np.count_nonzero(keep_idx), np.nan)
        if vi == vars_[0] and drop_allnan:
            base_lat = np.repeat(lat, nx)
            base_lon = np.tile(lon, ny)
            out["lat"] = base_lat[keep_idx]
            out["lon"] = base_lon[keep_idx]
    return pd.DataFrame(out)

def add_derived(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df2 = df.copy()
    if mode in {"stats","seasonal"}:
        if mode == "seasonal" and all(c in df.columns for c in ["prec_DJF","prec_MAM","prec_JJA","prec_SON"]):
            df2["prec_mean"] = df2[["prec_DJF","prec_MAM","prec_JJA","prec_SON"]].mean(axis=1)
        if "tmax_mean" in df2 and "tmin_mean" in df2:
            df2["t_range"] = df2["tmax_mean"] - df2["tmin_mean"]
        # stats mode precipitation total approximation (mean * 12 months)
        if mode == "stats" and "prec_mean" in df2:
            df2["prec_total"] = df2["prec_mean"] * 12.0
        # dryness index
        if "tmax_mean" in df2 and "prec_mean" in df2:
            df2["dryness_index"] = df2["tmax_mean"] / (df2["prec_mean"] + 1.0)
        if mode == "seasonal" and {"tmax_JJA","tmax_DJF","prec_JJA","prec_DJF"}.issubset(df2.columns):
            df2["dryness_season_ratio"] = (df2["tmax_JJA"]/(df2["prec_JJA"]+1)) / (df2["tmax_DJF"]/(df2["prec_DJF"]+1))
    return df2

def cap_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str,float]]]:
    df2 = df.copy()
    report = {}
    for col in df2.columns:
        if col in ("lat","lon"):
            continue
        if pd.api.types.is_numeric_dtype(df2[col]):
            arr = df2[col].to_numpy(dtype=float)
            capped, n, lo, hi = winsorize_iqr(arr)
            if n:
                df2[col] = capped
                report[col] = {"capped": n, "lo": lo, "hi": hi}
    return df2, report

def log_transform_prec(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if col.startswith("prec_") and pd.api.types.is_numeric_dtype(df2[col]):
            df2[col] = np.log1p(np.clip(df2[col].to_numpy(dtype=float), a_min=0, a_max=None))
    return df2

def eda_summary(df: pd.DataFrame, out_path: Path) -> None:
    numeric = {c: df[c].describe().to_dict() for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    categorical = {c: df[c].value_counts().head(50).to_dict() for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
    summary = {"rows": len(df), "columns": list(df.columns), "numeric": numeric, "categorical": categorical}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Transform climate NetCDF into tabular feature sets")
    ap.add_argument("--mode", choices=["monthly-wide","stats","seasonal","long"], default="stats")
    ap.add_argument("--include-dec", action="store_true", help="Include December in DJF season (seasonal mode)")
    ap.add_argument("--cap-iqr", action="store_true", help="Apply IQR winsorization to numeric columns")
    ap.add_argument("--log-prec", action="store_true", help="Apply log1p transform to precipitation columns after capping")
    ap.add_argument("--add-derived", action="store_true", help="Add derived features (t_range, prec_total, dryness indices) for aggregated modes")
    ap.add_argument("--quantiles", action="store_true", help="Include p10/p90 quantile features for aggregated modes")
    ap.add_argument("--out", type=Path, default=BASE / "DATA_CLEANED" / "processed", help="Output directory")
    ap.add_argument("--eda", action="store_true", help="Write EDA summary JSON")
    ap.add_argument("--drop-allnan", action="store_true", help="Drop spatial cells with all months NaN for every variable")
    ap.add_argument("--global-fill", action="store_true", help="Fill all-NaN cells with global mean before aggregation")
    ap.add_argument("--prec-total-sum", action="store_true", help="Compute prec_total as sum of monthly precipitation (stats mode) instead of mean*12")
    args = ap.parse_args()

    ds = load_dataset(DEFAULT_IN)
    ds = normalize_coords(ds)
    vars_ = [v for v in ds.data_vars if v.lower() in {"prec","tmax","tmin"} or v in {"prec","tmax","tmin"}]
    if not vars_:
        vars_ = list(ds.data_vars)

    if args.mode == "monthly-wide":
        df = pivot_monthly(ds, vars_)
    elif args.mode == "stats":
        # request monthly if we might compute sum
        df_stats, monthly_store = aggregate_stats_vectorized(ds, vars_, quantiles=args.quantiles, global_fill=args.global_fill, drop_allnan=args.drop_allnan, return_monthly=args.prec_total_sum)
        df = df_stats
    elif args.mode == "seasonal":
        df = aggregate_seasonal_vectorized(ds, vars_, include_dec=args.include_dec, quantiles=args.quantiles, global_fill=args.global_fill, drop_allnan=args.drop_allnan)
    elif args.mode == "long":
        months = month_numbers(ds)
        # Build long format directly
        rows = []
        lon = ds["lon"].values if "lon" in ds.coords else ds["x"].values
        lat = ds["lat"].values if "lat" in ds.coords else ds["y"].values
        for ti, m in enumerate(months):
            for iy, la in enumerate(lat):
                for ix, lo in enumerate(lon):
                    row = {"lat": float(la), "lon": float(lo), "month": int(m)}
                    for vi in vars_:
                        row[vi] = float(ds[vi][ti, iy, ix].values)
                    rows.append(row)
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Unsupported mode")

    # Derived features
    if args.add_derived:
        df = add_derived(df, args.mode)
        # Override prec_total with monthly sum if flag set and stats mode
        if args.mode == 'stats' and args.prec_total_sum and monthly_store and 'prec_mean' in df:
            if 'prec' in monthly_store:
                # sum monthly precipitation after imputation and any global fill
                prec_sum = np.sum(monthly_store['prec'], axis=0)
                # monthly_store['prec'] shape (12, n_cells) matches df rows order
                df['prec_total'] = prec_sum

    # Capping
    cap_report = {}
    if args.cap_iqr:
        df, cap_report = cap_outliers(df)

    if args.log_prec:
        df = log_transform_prec(df)

    # Final cleaning: drop rows with any NaN in essential coords, fill remaining NaNs with column mean
    df = df.dropna(subset=["lat","lon"])  # ensure coords valid
    for col in df.columns:
        if col not in ("lat","lon","month") and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)

    # Write outputs
    args.out.mkdir(parents=True, exist_ok=True)
    base_name = f"climate_features_{args.mode.replace('-', '_')}"
    pq = args.out / f"{base_name}.parquet"
    csv = args.out / f"{base_name}.csv"
    try:
        df.to_parquet(pq, index=False)
    except Exception as e:
        print(f"[WARN] Parquet write failed: {e}")
    df.to_csv(csv, index=False)

    if args.eda:
        eda_summary(df, args.out / f"{base_name}_summary.json")
        if cap_report:
            (args.out / f"{base_name}_capping.json").write_text(json.dumps(cap_report, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "mode": args.mode,
        "rows": len(df),
        "columns": len(df.columns),
        "parquet": str(pq),
        "csv": str(csv),
        "capped_cols": list(cap_report.keys()),
        "derived": args.add_derived,
        "quantiles": args.quantiles,
        "dropped_allnan": args.drop_allnan,
        "global_fill": args.global_fill,
        "prec_total_mode": ("sum" if args.prec_total_sum and args.mode=='stats' else "mean*12" if args.mode=='stats' and 'prec_total' in df.columns else None),
    }, indent=2))

if __name__ == "__main__":
    main()
