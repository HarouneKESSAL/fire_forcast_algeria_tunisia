#!/usr/bin/env python3
"""Transform elevation raster(s) into tabular features with EDA, cleaning, and derived terrain metrics.

Inputs (searched in priority order):
  1. DATA_CLEANED/elevation/elevation_clipped_clean.tif
  2. ELEVATION/be15_grd/be15_grd/processed/elevation_clipped.tif
  3. ELEVATION/processed/elevation_clipped.tif (if exists)
  4. DATA_CLEANED/elevation/elevation_clipped.tif
Optional companion rasters (if present in same folder): slope_deg.tif, aspect_deg.tif

Outputs:
  DATA_CLEANED/processed/elevation_features.parquet / .csv
  elevation_features_summary.json (EDA) + elevation_features_capping.json (outlier report if capping applied)

Processing steps:
  - Load elevation as DataArray via rioxarray.
  - Optional slope/aspect load.
  - Impute NaNs: fill with local neighborhood mean (3x3) then global median if still NaN (unless --no-impute).
  - Outlier capping: IQR winsorization (--cap-iqr) for elevation and slope (aspect circular handled separately).
  - Derived metrics:
       slope_rad (from slope_deg if present)
       cos_aspect, sin_aspect (aspect encoding if aspect present)
       ruggedness = local std deviation (3x3 window) of elevation (--ruggedness)
       relative_elev = elevation - global_median
       elevation_z = (elevation - mean) / std (standard score) if --zscore
  - Optional log transform of positive elevation (--log-elev) using log1p.
  - EDA summary of numeric columns.
  - Spatial export as rows (lat, lon, elevation, ...).

CLI examples:
  python scripts/transform_elevation.py --cap-iqr --ruggedness --zscore --eda
  python scripts/transform_elevation.py --no-impute --log-elev --eda

"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import rioxarray as rxr
except Exception:
    rxr = None
import xarray as xr

BASE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE / "DATA_CLEANED" / "processed"

CANDIDATES = [
    BASE / "DATA_CLEANED" / "elevation" / "elevation_clipped_clean.tif",
    BASE / "ELEVATION" / "be15_grd" / "be15_grd" / "processed" / "elevation_clipped.tif",
    BASE / "ELEVATION" / "processed" / "elevation_clipped.tif",
    BASE / "DATA_CLEANED" / "elevation" / "elevation_clipped.tif",
]

NEIGHBOR_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def load_elevation() -> xr.DataArray:
    if rxr is None:
        raise RuntimeError("rioxarray not installed; cannot process elevation.")
    for p in CANDIDATES:
        if p.exists():
            da = rxr.open_rasterio(p, masked=True).squeeze()
            da.name = "elevation"
            return da
    raise FileNotFoundError("No elevation raster found in expected locations.")

def try_load_companion(name: str, base_path: Path) -> xr.DataArray | None:
    p = base_path.parent / f"{name}.tif"
    if p.exists() and rxr is not None:
        try:
            da = rxr.open_rasterio(p, masked=True).squeeze()
            da.name = name
            return da
        except Exception:
            return None
    return None

def local_mean_impute(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    nan_mask = ~np.isfinite(out)
    if not nan_mask.any():
        return out
    h, w = out.shape
    for y in range(h):
        for x in range(w):
            if nan_mask[y, x]:
                vals = []
                for dy, dx in NEIGHBOR_OFFSETS:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w and np.isfinite(out[yy, xx]):
                        vals.append(out[yy, xx])
                if vals:
                    out[y, x] = float(np.mean(vals))
    return out

def winsorize_iqr(arr: np.ndarray, factor: float = 1.5) -> tuple[np.ndarray,int,float,float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 8:
        return arr, 0, np.nan, np.nan
    q1, q3 = np.percentile(finite, [25,75])
    iqr = q3 - q1
    if iqr <= 0:
        return arr, 0, np.nan, np.nan
    lo = q1 - factor*iqr
    hi = q3 + factor*iqr
    mask = (arr < lo) | (arr > hi)
    clipped = np.where(arr < lo, lo, np.where(arr > hi, hi, arr))
    return clipped, int(np.isfinite(arr[mask]).sum()), lo, hi

def add_ruggedness(elev: np.ndarray) -> np.ndarray:
    h, w = elev.shape
    out = np.full_like(elev, np.nan, dtype=float)
    for y in range(h):
        for x in range(w):
            vals = []
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w and np.isfinite(elev[yy, xx]):
                        vals.append(elev[yy, xx])
            if len(vals) >= 3:
                out[y,x] = float(np.std(vals))
    return out

def eda_summary(df: pd.DataFrame, out_path: Path) -> None:
    numeric = {c: df[c].describe().to_dict() for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    summary = {"rows": len(df), "columns": list(df.columns), "numeric": numeric}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Transform elevation raster into feature table")
    ap.add_argument("--cap-iqr", action="store_true", help="Apply IQR winsorization to elevation and slope")
    ap.add_argument("--ruggedness", action="store_true", help="Compute local terrain ruggedness (std of 3x3 window)")
    ap.add_argument("--zscore", action="store_true", help="Add elevation_z standardized feature")
    ap.add_argument("--log-elev", action="store_true", help="Apply log1p to positive elevation values")
    ap.add_argument("--no-impute", action="store_true", help="Skip NaN imputation steps")
    ap.add_argument("--eda", action="store_true", help="Write EDA summary JSON")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    ap.add_argument("--grid-stride", type=int, default=1, help="Stride over grid to downsample spatial resolution (e.g. 4 keeps every 4th pixel)")
    ap.add_argument("--sample-frac", type=float, default=None, help="Random sample fraction after stride (0<frac<=1)")
    ap.add_argument("--max-rows", type=int, default=None, help="Cap number of output rows after sampling")
    ap.add_argument("--parquet-compression", type=str, default="snappy", help="Parquet compression codec (snappy|gzip|zstd)")
    ap.add_argument("--approx-eda", action="store_true", help="Use sampling for EDA instead of full dataset")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--sample-points", type=Path, default=None, help="CSV/Parquet file with lat/lon (or latitude/longitude) to sample elevation features at those points only")
    ap.add_argument("--derive-slope-aspect", action="store_true", help="Derive slope & aspect from elevation if not provided")
    ap.add_argument("--tpi-window", type=int, default=5, help="Window size (odd) for Topographic Position Index (TPI) computation")
    ap.add_argument("--hillshade", action="store_true", help="Compute hillshade from elevation (az=315, alt=45 deg)")
    ap.add_argument("--slope-bins", action="store_true", help="Add categorical slope class feature (0:0-5,1:5-15,2:15-30,3:>30 deg)")
    args = ap.parse_args()

    elev_da = load_elevation()
    base_src = elev_da.encoding.get("source", None)
    try:
        base_path = Path(str(base_src)) if base_src else CANDIDATES[0]
    except Exception:
        base_path = CANDIDATES[0]
    slope_da = try_load_companion("slope_deg", base_path)
    aspect_da = try_load_companion("aspect_deg", base_path)

    elev = elev_da.values.astype(float)
    slope = slope_da.values.astype(float) if slope_da is not None else None
    aspect = aspect_da.values.astype(float) if aspect_da is not None else None

    # Imputation
    if not args.no_impute:
        elev = local_mean_impute(elev)
        if np.isnan(elev).any():
            gm = float(np.nanmedian(elev[np.isfinite(elev)])) if np.isfinite(elev).any() else 0.0
            elev = np.where(np.isfinite(elev), elev, gm)
        if slope is not None:
            slope = local_mean_impute(slope)
            if np.isnan(slope).any():
                sm = float(np.nanmedian(slope[np.isfinite(slope)])) if np.isfinite(slope).any() else 0.0
                slope = np.where(np.isfinite(slope), slope, sm)
        if aspect is not None:
            # Aspect circular: leave NaNs or fill with 0 (north) if isolated
            amask = ~np.isfinite(aspect)
            if amask.any():
                aspect = np.where(amask, 0.0, aspect)

    cap_report = {}
    if args.cap_iqr:
        elev_c, n, lo, hi = winsorize_iqr(elev)
        if n:
            cap_report['elevation'] = {"capped": n, "lo": lo, "hi": hi}
            elev = elev_c
        if slope is not None:
            slope_c, sn, slo, shi = winsorize_iqr(slope)
            if sn:
                cap_report['slope_deg'] = {"capped": sn, "lo": slo, "hi": shi}
                slope = slope_c

    # Grid sampling
    h, w = elev.shape
    stride = max(1, args.grid_stride)
    if stride > 1:
        elev_stride = elev[::stride, ::stride]
        lat_full = elev_da["lat"].values if "lat" in elev_da.coords else elev_da["y"].values
        lon_full = elev_da["lon"].values if "lon" in elev_da.coords else elev_da["x"].values
        lat_used = lat_full[::stride]
        lon_used = lon_full[::stride]
    else:
        elev_stride = elev
        lat_used = elev_da["lat"].values if "lat" in elev_da.coords else elev_da["y"].values
        lon_used = elev_da["lon"].values if "lon" in elev_da.coords else elev_da["x"].values
    hs, ws = elev_stride.shape
    # Build index list for sampling
    total_cells = hs * ws
    rng = np.random.default_rng(args.seed)
    if args.sample_frac is not None:
        frac = min(1.0, max(0.0, args.sample_frac))
        sample_n = int(total_cells * frac)
    else:
        sample_n = total_cells
    if args.max_rows is not None:
        sample_n = min(sample_n, args.max_rows)
    # Select indices
    if sample_n < total_cells:
        flat_indices = rng.choice(total_cells, size=sample_n, replace=False)
    else:
        flat_indices = np.arange(total_cells)
    ys = (flat_indices // ws).astype(int)
    xs = (flat_indices % ws).astype(int)
    # Extract sampled elevation values
    elev_vals = elev_stride[ys, xs]
    rug_vals = None  # initialize
    # Ruggedness computation adapted for sampled cells
    if args.ruggedness:
        rug_vals = np.empty_like(elev_vals, dtype=float)
        for i, (yy, xx) in enumerate(zip(ys, xs)):
            oy = yy * stride
            ox = xx * stride
            vals = []
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    yy2 = oy + dy
                    xx2 = ox + dx
                    if 0 <= yy2 < h and 0 <= xx2 < w and np.isfinite(elev[yy2, xx2]):
                        vals.append(elev[yy2, xx2])
            rug_vals[i] = float(np.std(vals)) if len(vals) >= 3 else np.nan

    transform = getattr(elev_da, 'rio', None).transform() if hasattr(elev_da, 'rio') else None

    # Possibly derive slope/aspect if requested and absent
    if args.derive_slope_aspect and slope is None:
        slope, aspect = _derive_slope_aspect(elev, transform)
        slope_da = None; aspect_da = None  # mark derived

    # After building sampled indices (ys,xs) & before constructing features, allow point sampling override
    point_sample_mode = args.sample_points is not None and args.sample_points.exists()
    point_df = None
    if point_sample_mode:
        # Load points
        if args.sample_points.suffix.lower() in {'.parquet'}:
            point_df = pd.read_parquet(args.sample_points)
        else:
            point_df = pd.read_csv(args.sample_points)
        # Normalize column names
        cols_lower = {c.lower(): c for c in point_df.columns}
        lat_col = cols_lower.get('lat') or cols_lower.get('latitude')
        lon_col = cols_lower.get('lon') or cols_lower.get('longitude')
        if not lat_col or not lon_col:
            raise ValueError("Sample points file must contain lat/lon or latitude/longitude columns")
        pts_lat = pd.to_numeric(point_df[lat_col], errors='coerce').to_numpy()
        pts_lon = pd.to_numeric(point_df[lon_col], errors='coerce').to_numpy()
        # Build mapping function (nearest index along each axis). Lat_used, lon_used monotonic assumed.
        lat_array = lat_used
        lon_array = lon_used
        def nearest_idx(arr, v):
            # binary search via argmin absolute difference
            return int(np.argmin(np.abs(arr - v)))
        # Map points to indices
        m_ys = np.array([nearest_idx(lat_array, v) for v in pts_lat])
        m_xs = np.array([nearest_idx(lon_array, v) for v in pts_lon])
        # Extract values directly from full elevation (using stride mapping)
        elev_vals = elev_stride[m_ys, m_xs]
        ys, xs = m_ys, m_xs  # override indices with point mapping
        # If ruggedness requested recompute for mapped cells
        if args.ruggedness:
            rug_vals = np.empty_like(elev_vals, dtype=float)
            for i, (yy, xx) in enumerate(zip(ys, xs)):
                oy = yy * stride
                ox = xx * stride
                vals = []
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        yy2 = oy + dy
                        xx2 = ox + dx
                        if 0 <= yy2 < h and 0 <= xx2 < w and np.isfinite(elev[yy2, xx2]):
                            vals.append(elev[yy2, xx2])
                rug_vals[i] = float(np.std(vals)) if len(vals) >= 3 else np.nan
        # Adjust sample metadata reporting
        sample_n = len(ys)
        flat_indices = None  # no random sampling

    # Derived after sampling
    features = {}
    features['lat'] = lat_used[ys]
    features['lon'] = lon_used[xs]
    features['elevation'] = elev_vals
    if slope is not None:
        # Downsample slope to stride grid and sample
        slope_stride = slope[::stride, ::stride]
        slope_vals = slope_stride[ys, xs]
        features['slope_deg'] = slope_vals
        features['slope_rad'] = np.deg2rad(features['slope_deg'])
    if aspect is not None:
        aspect_stride = aspect[::stride, ::stride]
        aspect_vals = aspect_stride[ys, xs]
        arad = np.deg2rad(aspect_vals)
        features['aspect_deg'] = aspect_vals
        features['aspect_cos'] = np.cos(arad)
        features['aspect_sin'] = np.sin(arad)
    elev_med = float(np.median(features['elevation'])) if np.isfinite(features['elevation']).any() else 0.0
    features['relative_elev'] = features['elevation'] - elev_med
    if args.zscore:
        emean = float(np.mean(features['elevation'])) if np.isfinite(features['elevation']).any() else 0.0
        estd = float(np.std(features['elevation'])) or 1.0
        features['elevation_z'] = (features['elevation'] - emean) / estd
    if args.ruggedness and rug_vals is not None:
        features['ruggedness'] = rug_vals
    if args.tpi_window and args.tpi_window >=3 and args.tpi_window % 2 == 1 and not point_sample_mode:
        # Compute TPI on full (or stride) grid then sample
        tpi_full = _compute_tpi(elev_stride, args.tpi_window)
        tpi_vals = tpi_full[ys, xs]
    else:
        tpi_vals = None
    if args.hillshade and not point_sample_mode:
        hs_full = _hillshade(elev_stride, transform)
        hillshade_vals = hs_full[ys, xs]
    else:
        hillshade_vals = None
    if tpi_vals is not None:
        features['tpi'] = tpi_vals
    if hillshade_vals is not None:
        features['hillshade'] = hillshade_vals
    if args.slope_bins and 'slope_deg' in features:
        sd = features['slope_deg']
        bins = np.digitize(sd, [5,15,30], right=False)
        features['slope_class'] = bins  # 0..3
    df = pd.DataFrame(features)

    # Final cleaning: ensure numeric columns no remaining NaN (fill with column median)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)

    args.out.mkdir(parents=True, exist_ok=True)
    pq = args.out / "elevation_features.parquet"
    csv = args.out / "elevation_features.csv"
    try:
        df.to_parquet(pq, index=False, compression=args.parquet_compression)
    except Exception as e:
        print(f"[WARN] Parquet write failed: {e}")
    df.to_csv(csv, index=False)

    if args.eda:
        if args.approx_eda and len(df) > 500000:
            sample_eda = df.sample(min(50000, len(df)), random_state=args.seed)
        else:
            sample_eda = df
        eda_summary(sample_eda, args.out / "elevation_features_summary.json")
        if cap_report:
            (args.out / "elevation_features_capping.json").write_text(json.dumps(cap_report, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "rows": len(df),
        "columns": len(df.columns),
        "parquet": str(pq),
        "csv": str(csv),
        "capped": cap_report,
        "derived": {
            "slope": slope is not None,
            "aspect": aspect is not None,
            "ruggedness": args.ruggedness,
            "zscore": args.zscore,
            "log_elev": args.log_elev,
        },
        "stride": stride,
        "sample_frac": args.sample_frac,
        "max_rows": args.max_rows,
        "approx_eda": args.approx_eda,
        "reduction_factor": round((h*w)/max(1,len(df)),2),
        "point_sample_mode": point_sample_mode,
        "tpi_window": args.tpi_window if tpi_vals is not None else None,
        "hillshade": args.hillshade if hillshade_vals is not None else False,
        "slope_bins": args.slope_bins,
        "derived_slope_aspect": args.derive_slope_aspect,
    }, indent=2))

def _derive_slope_aspect(elev_arr: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    # Approximate slope & aspect from elevation using central differences; assumes geographic or projected units.
    # Derivation: slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2)), aspect: arctan2(dz/dy, -dz/dx)
    # Pixel size from transform (a: x pixel size, e: y pixel size signed)
    dx = abs(getattr(transform, 'a', 1.0) if transform else 1.0)
    dy = abs(getattr(transform, 'e', 1.0) if transform else 1.0)
    dzdx = np.gradient(elev_arr, axis=1) / dx
    dzdy = np.gradient(elev_arr, axis=0) / dy
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180.0/np.pi)
    aspect = np.arctan2(dzdy, -dzdx) * (180.0/np.pi)
    aspect = np.where(aspect < 0, 90.0 - aspect, 360.0 - aspect + 90.0)  # normalize to 0-360
    return slope, aspect

def _compute_tpi(elev_arr: np.ndarray, w: int) -> np.ndarray:
    if w < 3 or w % 2 == 0:
        return np.full_like(elev_arr, np.nan, dtype=float)
    pad = w//2
    padded = np.pad(elev_arr, pad, mode='edge')
    out = np.full_like(elev_arr, np.nan, dtype=float)
    for y in range(elev_arr.shape[0]):
        for x in range(elev_arr.shape[1]):
            block = padded[y:y+w, x:x+w]
            out[y,x] = elev_arr[y,x] - np.mean(block)
    return out

def _hillshade(elev_arr: np.ndarray, transform, az_deg=315, alt_deg=45) -> np.ndarray:
    # Simple hillshade using Horn's method derivatives
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    dx = abs(getattr(transform, 'a', 1.0) if transform else 1.0)
    dy = abs(getattr(transform, 'e', 1.0) if transform else 1.0)
    dzdx = np.gradient(elev_arr, axis=1) / dx
    dzdy = np.gradient(elev_arr, axis=0) / dy
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(dzdy, -dzdx)
    hs = (np.sin(alt) * np.sin(slope_rad) + np.cos(alt) * np.cos(slope_rad) * np.cos(az - aspect_rad))
    return np.clip(hs, 0, 1)

if __name__ == "__main__":
    main()
