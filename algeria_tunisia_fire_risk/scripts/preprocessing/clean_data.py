#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic data cleaner for the Algeria–Tunisia workspace.

Features
- Walks the workspace (or a given source dir) to find CSVs and vector files (SHP/GeoJSON/GPKG)
- Cleans attributes using:
  - numeric: choose imputation by shape: symmetric → mean, asymmetric → median (skew threshold)
  - categorical: mode
  - lat/lon: impute from the mean of k nearest valid observations
- Geometries:
  - fix invalid geometries via buffer(0)
  - if missing geometry:
    - for point layers: replace with mean point of k nearest valid features
    - for polygon layers: replace with a circular polygon centered at the mean of k nearest
      centroids, with radius derived from mean area of valid polygons

Outputs
- Writes to DATA_CLEANED/ preserving folder structure (relative to source root)
- Vector files are written as GeoPackage (.gpkg) to preserve geometry integrity

Usage
  python3 scripts/clean_data.py --src /home/swift/Desktop/DATA --out /home/swift/Desktop/DATA/DATA_CLEANED

Notes
- Heuristics: a numeric column is considered symmetric if abs(skew) <= 0.5
- Categorical detection: dtype is object/category/bool or low cardinality
- No external heavy dependencies beyond pandas, geopandas, shapely, numpy
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
try:
    import rioxarray as rxr  # optional; may be unavailable on Windows Python 3.14
except Exception:  # pragma: no cover
    rxr = None


BASE = Path.cwd()
DEFAULT_OUT = BASE / "DATA_CLEANED"


# ----------------------
# Helpers
# ----------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or str(series.dtype).lower() == "category":
        return True
    if pd.api.types.is_numeric_dtype(series):
        # numeric but very low cardinality → categorical-like
        n_unique = series.dropna().nunique()
        n_total = series.size
        return n_unique > 0 and (n_unique <= 20 or n_unique / max(1, n_total) <= 0.05)
    # object/strings -> categorical
    return pd.api.types.is_object_dtype(series)


def impute_numeric(col: pd.Series) -> Tuple[pd.Series, str]:
    """Decide mean vs median based on symmetry: if |mean-median|/std <= 0.1 use mean, else median."""
    s = pd.to_numeric(col.replace([np.inf, -np.inf], np.nan), errors="coerce")
    s_nonan = s.dropna()
    if s_nonan.empty:
        return s, "none"
    # Outlier capping first (IQR winsorization)
    s_cap, n_capped = winsorize_iqr(s)
    if n_capped:
        s = s_cap
    mean = float(s_nonan.mean())
    median = float(s_nonan.median())
    std = float(s_nonan.std())
    if std > 0 and abs(mean - median) / std <= 0.1:
        val = mean
        method = "mean"
    else:
        val = median
        method = "median"
    return s.fillna(val), method


def impute_categorical(col: pd.Series) -> Tuple[pd.Series, str]:
    # mode may return multiple; pick first
    try:
        mode_vals = col.dropna().mode()
        if mode_vals.empty:
            return col.ffill().bfill(), "mode-ffill-bfill"
        val = mode_vals.iloc[0]
        return col.fillna(val), "mode"
    except Exception:
        return col.ffill().bfill(), "mode-ffill-bfill"


def find_lat_lon_columns(df: pd.DataFrame) -> Tuple[str | None, str | None]:
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "lng", "longitude", "x"]
    lat = next((c for c in df.columns if c.lower() in lat_candidates), None)
    lon = next((c for c in df.columns if c.lower() in lon_candidates), None)
    return lat, lon


def impute_latlon_knn(df: pd.DataFrame, lat_col: str, lon_col: str, k: int = 5) -> Tuple[pd.DataFrame, int]:
    """Fill missing lat/lon by averaging k nearest neighbors in lat-lon space (degrees)."""
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    mask_valid = lat.notna() & lon.notna()
    if mask_valid.sum() == 0:
        return df, 0
    valid_lat = lat[mask_valid].to_numpy()
    valid_lon = lon[mask_valid].to_numpy()
    valid_coords = np.column_stack([valid_lat, valid_lon])

    missing_idx = np.where(~mask_valid.to_numpy())[0]
    filled = 0
    for idx in missing_idx:
        # simple nearest by Euclidean distance in degree space
        p_lat = np.nanmean(valid_lat)
        p_lon = np.nanmean(valid_lon)
        # if there are many valid points, compute distances from a sample for performance
        sample = valid_coords
        d = np.hypot(sample[:, 0] - p_lat, sample[:, 1] - p_lon)
        kk = min(k, len(d))
        if kk == 0:
            continue
        nn = np.argpartition(d, kk - 1)[:kk]
        df.at[idx, lat_col] = float(valid_lat[nn].mean())
        df.at[idx, lon_col] = float(valid_lon[nn].mean())
        filled += 1
    return df, filled


def clean_csv(src: Path, out_root: Path, src_root: Path) -> dict:
    rel = src.relative_to(src_root)
    out_path = out_root / rel
    ensure_dir(out_path.parent)

    df = pd.read_csv(src, low_memory=False)
    actions = []

    # Special: lat/lon first
    lat_col, lon_col = find_lat_lon_columns(df)
    if lat_col and lon_col:
        df, n_filled = impute_latlon_knn(df, lat_col, lon_col, k=5)
        if n_filled:
            actions.append(f"latlon_imputed_knn={n_filled}")

    for col in df.columns:
        if col in (lat_col, lon_col):
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            s2, how = impute_numeric(s)
            if s2.isna().sum() < s.isna().sum():
                df[col] = s2
                actions.append(f"{col}:{how}")
        elif is_categorical(s):
            s2, how = impute_categorical(s)
            if s2.isna().sum() < s.isna().sum():
                df[col] = s2
                actions.append(f"{col}:{how}")

    df.to_csv(out_path, index=False)
    return {
        "type": "csv",
        "in": src.as_posix(),
        "out": out_path.as_posix(),
        "actions": ", ".join(actions),
    }


def layer_geom_kind(gdf: gpd.GeoDataFrame) -> str:
    kinds = set(gdf.geometry.geom_type.dropna().unique())
    if kinds == {"Point"}:
        return "point"
    if kinds <= {"Polygon", "MultiPolygon"}:
        return "polygon"
    return "mixed"


def _nearest_mean_point(points: np.ndarray, k: int = 5) -> Tuple[float, float]:
    # points: (N, 2) array of (y, x) or (lat, lon)
    if points.shape[0] == 0:
        return float("nan"), float("nan")
    kk = min(k, points.shape[0])
    # choose reference as global mean (fallback), then refine with nearest selection
    ref = points.mean(axis=0)
    d = np.hypot(points[:, 0] - ref[0], points[:, 1] - ref[1])
    nn = np.argpartition(d, kk - 1)[:kk]
    return float(points[nn, 0].mean()), float(points[nn, 1].mean())


def clean_vector(src: Path, out_root: Path, src_root: Path) -> dict:
    rel = src.relative_to(src_root)
    out_dir = out_root / rel.parent
    ensure_dir(out_dir)
    out_path = (out_dir / src.stem).with_suffix(".gpkg")

    gdf = gpd.read_file(src)
    # Fix invalid geometries
    try:
        gdf["geometry"] = gdf.buffer(0)
    except Exception:
        pass

    kind = layer_geom_kind(gdf)
    act = []

    # Impute attribute columns similarly to CSV
    for col in [c for c in gdf.columns if c != gdf.geometry.name]:
        s = gdf[col]
        if pd.api.types.is_numeric_dtype(s):
            s2, how = impute_numeric(s)
            if s2.isna().sum() < s.isna().sum():
                gdf[col] = s2
                act.append(f"{col}:{how}")
        elif is_categorical(s):
            s2, how = impute_categorical(s)
            if s2.isna().sum() < s.isna().sum():
                gdf[col] = s2
                act.append(f"{col}:{how}")

    # Geometry imputation
    geom = gdf.geometry
    missing = geom.isna() | geom.is_empty
    n_missing = int(missing.sum())
    if n_missing:
        valid = gdf.loc[~missing].copy()
        if not valid.empty:
            # Work in EPSG:4326 for simplicity
            try:
                gdf = gdf.to_crs("EPSG:4326")
                valid = gdf.loc[~missing]
            except Exception:
                pass
            new_geom = gdf.geometry.copy()
            if kind == "point":
                pts = np.array([[p.centroid.y, p.centroid.x] for p in valid.geometry if p is not None and not p.is_empty])
                my, mx = _nearest_mean_point(pts, k=5)
                fill_geom = Point(mx, my) if np.isfinite(my) and np.isfinite(mx) else None
                for idx in gdf.index[missing]:
                    new_geom.at[idx] = fill_geom
                act.append(f"geom_impute_point:{n_missing}")
            else:
                # Polygon/other: build small circle based on mean area of valid polygons
                polys = valid.geometry.area
                mean_area = float(polys.mean()) if len(polys) else 0.0
                if mean_area <= 0:
                    mean_area = 1e-6
                radius = (mean_area / np.pi) ** 0.5
                centers = np.array([[g.centroid.y, g.centroid.x] for g in valid.geometry if g is not None and not g.is_empty])
                my, mx = _nearest_mean_point(centers, k=5)
                fill_geom = Point(mx, my).buffer(radius)
                for idx in gdf.index[missing]:
                    new_geom.at[idx] = fill_geom
                act.append(f"geom_impute_polygon:{n_missing}")
            gdf = gdf.set_geometry(new_geom)

    # Write as GeoPackage
    gdf.to_file(out_path, driver="GPKG")
    return {
        "type": "vector",
        "in": src.as_posix(),
        "out": out_path.as_posix(),
        "actions": ", ".join(act),
    }


def scan_sources(src_root: Path) -> Iterable[Path]:
    # Yield CSVs and vector files (SHP, GEOJSON, GPKG) except already-cleaned outputs
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        if "DATA_CLEANED" in p.parts:
            continue
        suf = p.suffix.lower()
        if suf in {".csv"}:
            yield p
        elif suf in {".shp", ".geojson", ".json", ".gpkg"}:
            yield p


def write_log(entries: list[dict], out_root: Path) -> None:
    ensure_dir(out_root)
    log_md = out_root / "log.md"
    with open(log_md, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(f"- {e['type']} | {e['in']} -> {e['out']} | {e['actions']}\n")

    # If requested, also run targeted dataset cleaning (specialized logic)
    # Use --targets flag to trigger


def symmetry_choice(arr: np.ndarray) -> Tuple[float, str]:
    """Return fill value and method based on symmetry using mean/median/std criterion."""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.nan, "none"
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    std = float(np.std(vals))
    if std > 0 and abs(mean - median) / std <= 0.1:
        return mean, "mean"
    return median, "median"


def winsorize_iqr(s: pd.Series, factor: float = 1.5) -> Tuple[pd.Series, int]:
    """Cap outliers outside IQR fences; returns series and count of capped values."""
    x = pd.to_numeric(s, errors="coerce")
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return x, 0
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    capped = x.clip(lower=lo, upper=hi)
    n = int((x.lt(lo) | x.gt(hi)).sum())
    return capped, n


def clean_climate_nc(path: Path, out_root: Path) -> dict:
    if not path.exists():
        return {"type": "climate_nc", "in": path.as_posix(), "out": "", "actions": "missing"}
    ds = xr.open_dataset(path)
    actions = []
    ds_clean = ds.copy()
    for var in ds.data_vars:
        da = ds[var]
        if np.issubdtype(da.dtype, np.number):
            # Temporal fill: forward/backward fill along time if present
            if "time" in da.dims:
                try:
                    da = da.ffill(dim="time").bfill(dim="time")
                except Exception:
                    pass
            arr = da.values.astype(float)
            fill_val, method = symmetry_choice(arr)
            mask = ~np.isfinite(arr)
            if mask.any():
                arr2 = arr.copy()
                arr2[mask] = fill_val
                ds_clean[var].values = arr2
                actions.append(f"{var}:{method}_impute={mask.sum()}")
        else:
            # treat as categorical: use mode if NaNs
            vals = pd.Series(da.values.ravel())
            mode_vals = vals.dropna().mode()
            if mode_vals.size:
                fill = mode_vals.iloc[0]
                arr = da.values
                mask = pd.isna(arr)
                if mask.any():
                    arr[mask] = fill
                    ds_clean[var].values = arr
                    actions.append(f"{var}:mode_impute={mask.sum()}")
    out_dir = out_root / "CLIMATE"
    ensure_dir(out_dir)
    out_path = out_dir / path.name.replace(".nc", "_clean.nc")
    ds_clean.to_netcdf(out_path)
    return {"type": "climate_nc", "in": path.as_posix(), "out": out_path.as_posix(), "actions": ", ".join(actions)}


def clean_soil_nc(path: Path, out_root: Path) -> dict:
    if not path.exists():
        return {"type": "soil_nc", "in": path.as_posix(), "out": "", "actions": "missing"}
    ds = xr.open_dataset(path)
    actions = []
    ds_clean = ds.copy()
    for var in ds.data_vars:
        da = ds[var]
        arr = da.values
        # soil IDs treated categorical: mode fill
        vals = pd.Series(arr.ravel())
        mode_vals = vals.dropna().mode()
        if mode_vals.size:
            fill = mode_vals.iloc[0]
            mask = ~np.isfinite(arr)
            if mask.any():
                arr2 = arr.copy()
                arr2[mask] = fill
                ds_clean[var].values = arr2
                actions.append(f"{var}:mode_impute={mask.sum()}")
    out_dir = out_root / "SOIL"
    ensure_dir(out_dir)
    out_path = out_dir / path.name.replace(".nc", "_clean.nc")
    ds_clean.to_netcdf(out_path)
    return {"type": "soil_nc", "in": path.as_posix(), "out": out_path.as_posix(), "actions": ", ".join(actions)}


def clean_elevation_raster(path: Path, out_root: Path) -> dict:
    if not path.exists():
        return {"type": "elevation_tif", "in": path.as_posix(), "out": "", "actions": "missing"}
    if rxr is None:
        return {"type": "elevation_tif", "in": path.as_posix(), "out": "", "actions": "skip:rioxarray_missing"}
    raw = rxr.open_rasterio(path, masked=True)
    # normalize to DataArray
    if isinstance(raw, list):
        raw = raw[0]
    if isinstance(raw, xr.Dataset):
        if len(raw.data_vars) == 1:
            raw = next(iter(raw.data_vars.values()))
        else:
            raw = raw.to_array().isel(variable=0)
    da = raw.squeeze()
    arr = da.values.astype(float)
    nodata = da.rio.nodata
    mask = ~np.isfinite(arr)
    if nodata is not None:
        mask |= (arr == nodata)
    fill_val, method = symmetry_choice(arr)
    act = []
    if mask.any():
        arr2 = arr.copy()
        arr2[mask] = fill_val
        da.values = arr2
        act.append(f"impute:{method}={int(mask.sum())}")
    out_dir = out_root / "ELEVATION"
    ensure_dir(out_dir)
    out_path = out_dir / path.name.replace(".tif", "_clean.tif")
    da.rio.to_raster(out_path.as_posix(), compress="LZW")
    return {"type": "elevation_tif", "in": path.as_posix(), "out": out_path.as_posix(), "actions": ", ".join(act)}


def clean_landcover_vector(path: Path, out_root: Path) -> dict:
    if not path.exists():
        return {"type": "landcover_shp", "in": path.as_posix(), "out": "", "actions": "missing"}
    gdf = gpd.read_file(path)
    # fix invalid
    try:
        gdf["geometry"] = gdf.buffer(0)
    except Exception:
        pass
    actions = []
    # attribute impute
    for col in [c for c in gdf.columns if c != gdf.geometry.name]:
        s = gdf[col]
        if pd.api.types.is_numeric_dtype(s):
            s2, how = impute_numeric(s)
            if s2.isna().sum() < s.isna().sum():
                gdf[col] = s2
                actions.append(f"{col}:{how}")
        elif is_categorical(s):
            s2, how = impute_categorical(s)
            if s2.isna().sum() < s.isna().sum():
                gdf[col] = s2
                actions.append(f"{col}:{how}")
    # geometry impute for empties
    missing = gdf.geometry.isna() | gdf.geometry.is_empty
    n_missing = int(missing.sum())
    if n_missing:
        valid = gdf.loc[~missing]
        pts = np.array([[p.centroid.y, p.centroid.x] for p in valid.geometry if p is not None and not p.is_empty])
        my, mx = _nearest_mean_point(pts, k=5)
        fill_geom = Point(mx, my) if np.isfinite(my) and np.isfinite(mx) else None
        new_geom = gdf.geometry.copy()
        for idx in gdf.index[missing]:
            new_geom.at[idx] = fill_geom
        gdf = gdf.set_geometry(new_geom)
        actions.append(f"geom_impute={n_missing}")
    out_dir = out_root / "LANDCOVER"
    ensure_dir(out_dir)
    out_path = out_dir / path.name.replace(".shp", "_clean.gpkg")
    gdf.to_file(out_path, driver="GPKG")
    return {"type": "landcover_shp", "in": path.as_posix(), "out": out_path.as_posix(), "actions": ", ".join(actions)}


def clean_fire_csvs(paths: list[Path], out_root: Path) -> list[dict]:
    out_entries = []
    for p in paths:
        if not p.exists():
            out_entries.append({"type": "fire_csv", "in": p.as_posix(), "out": "", "actions": "missing"})
            continue
        res = clean_csv(p, out_root / "fire", BASE)
        res["type"] = "fire_csv"
        out_entries.append(res)
    return out_entries


# ----------------------
# Specialized tasks per user request
# ----------------------

def build_climate_nc_from_filtered(filtered_dir: Path, out_nc: Path) -> dict:
    """Merge 2024 monthly climate (prec,tmax,tmin) into one NetCDF from DATA_FILTRED/CLIMATE.
    Accepts GeoTIFF or NetCDF inputs; detects month from filename (YYYY-MM).
    """
    ensure_dir(out_nc.parent)
    if not filtered_dir.exists():
        return {"type": "climate_build", "in": filtered_dir.as_posix(), "out": "", "actions": "missing"}

    def _collect(var: str) -> List[tuple[str, xr.DataArray]]:
        out: List[tuple[str, xr.DataArray]] = []
        # glob both tif and nc
        for p in list(filtered_dir.rglob(f"*{var}*2024*.tif")) + list(filtered_dir.rglob(f"*{var}*2024*.nc")):
            name = p.name
            # extract YYYY-MM
            m = None
            for token in name.replace("_", "-").split("-"):
                if len(token) == 7 and token[:4].isdigit() and token[4] == '0' or token[4] == '1':
                    # best-effort; fallback below
                    pass
            # robust pattern search
            import re
            mm = re.search(r"(2024)-(0[1-9]|1[0-2])", name)
            if not mm:
                continue
            ym = f"{mm.group(1)}-{mm.group(2)}"
            try:
                if p.suffix.lower() == ".tif":
                    if rxr is None:
                        continue
                    da = rxr.open_rasterio(p, masked=True)
                    if isinstance(da, xr.Dataset):
                        da = da.to_array().isel(variable=0)
                    da = da.squeeze()
                    da.name = var
                else:
                    ds = xr.open_dataset(p)
                    # pick first data var matching var
                    v = var if var in ds.data_vars else list(ds.data_vars)[0]
                    da = ds[v].squeeze()
                    da.name = var
                out.append((ym, da))
            except Exception:
                continue
        return out

    data_vars: Dict[str, xr.DataArray] = {}
    times: List[str] = []
    stacked: Dict[str, List[xr.DataArray]] = {"prec": [], "tmax": [], "tmin": []}
    time_labels: List[str] = []
    for var in ["prec", "tmax", "tmin"]:
        pairs = _collect(var)
        pairs.sort(key=lambda x: x[0])
        for ym, da in pairs:
            stacked[var].append(da)
        if not time_labels and pairs:
            time_labels = [ym for ym, _ in pairs]

    if not any(stacked[v] for v in stacked):
        return {"type": "climate_build", "in": filtered_dir.as_posix(), "out": "", "actions": "no_2024_files"}

    # Align and assemble into dataset with time dim
    ds_out_vars = {}
    for var, arrs in stacked.items():
        if not arrs:
            continue
        arrs_aligned = xr.align(*arrs, join="outer")
        da_stack = xr.concat(arrs_aligned, dim="time")
        da_stack = da_stack.assign_coords(time=pd.to_datetime(time_labels))
        ds_out_vars[var] = da_stack
    ds_out = xr.Dataset(ds_out_vars)
    ds_out.to_netcdf(out_nc)
    return {"type": "climate_build", "in": filtered_dir.as_posix(), "out": out_nc.as_posix(), "actions": f"vars={list(ds_out_vars)} months={len(time_labels)}"}


def build_soil_id_nc_from_bil(bil_path: Path, out_nc: Path) -> dict:
    """Create a NetCDF with soil ID grid from HWSD2.bil matched by lat/lon."""
    ensure_dir(out_nc.parent)
    if not bil_path.exists():
        return {"type": "soil_build", "in": bil_path.as_posix(), "out": "", "actions": "missing"}
    if rxr is None:
        return {"type": "soil_build", "in": bil_path.as_posix(), "out": "", "actions": "skip:rioxarray_missing"}
    da = rxr.open_rasterio(bil_path, masked=True)
    if isinstance(da, xr.Dataset):
        da = da.to_array().isel(variable=0)
    da = da.squeeze()
    da.name = "soil_id"
    # Normalize coords names
    if "y" in da.coords and "x" in da.coords:
        da = da.rename({"y": "lat", "x": "lon"})
    ds = xr.Dataset({"soil_id": da})
    ds.to_netcdf(out_nc)
    return {"type": "soil_build", "in": bil_path.as_posix(), "out": out_nc.as_posix(), "actions": "soil_id_grid"}


def filter_fire_type_not2(fire_dir: Path, out_dir: Path) -> List[dict]:
    """Keep only fires with type != 2 from available CSVs; write to DATA_CLEANED/FIRE."""
    ensure_dir(out_dir)
    entries: List[dict] = []
    for p in fire_dir.glob("*.csv"):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        type_col = next((c for c in df.columns if c.lower() == "type"), None)
        if type_col is None:
            # nothing to filter; copy
            outp = out_dir / p.name
            df.to_csv(outp, index=False)
            entries.append({"type": "fire_filter", "in": p.as_posix(), "out": outp.as_posix(), "actions": "copied:no_type"})
            continue
        before = len(df)
        df2 = df[df[type_col] != 2].copy()
        outp = out_dir / (p.stem + "_not2.csv")
        df2.to_csv(outp, index=False)
        entries.append({"type": "fire_filter", "in": p.as_posix(), "out": outp.as_posix(), "actions": f"kept={len(df2)} removed={before-len(df2)}"})
    return entries


def _fire_lower_whisker_threshold(df: pd.DataFrame) -> tuple[str, float]:
    cand = [c for c in ["frp", "frp_mean", "bright_ti4", "bright_ti5"] if c in df.columns]
    if not cand:
        return "", float("nan")
    v = cand[0]
    x = pd.to_numeric(df[v], errors="coerce").dropna()
    if x.empty:
        return v, float("nan")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    thr = float(q1 - 1.5 * iqr)
    return v, thr


def cut_to_fire_threshold_and_crop(out_root: Path) -> List[dict]:
    """Use boxplot lower whisker on FRP-like metric to filter fires; crop other datasets to bbox."""
    entries: List[dict] = []
    fire_dir = out_root / "FIRE"
    fire_files = list(fire_dir.glob("*.csv"))
    if not fire_files:
        return [{"type": "cut_to_fire", "in": fire_dir.as_posix(), "out": "", "actions": "no_fire_csv"}]
    # Concatenate fire
    dfs = []
    for p in fire_files:
        try:
            d = pd.read_csv(p, low_memory=False)
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return [{"type": "cut_to_fire", "in": fire_dir.as_posix(), "out": "", "actions": "empty_fire"}]
    fire = pd.concat(dfs, ignore_index=True)
    v, thr = _fire_lower_whisker_threshold(fire)
    if not v or not np.isfinite(thr):
        return [{"type": "cut_to_fire", "in": fire_dir.as_posix(), "out": "", "actions": "no_metric"}]
    fire_f = fire[pd.to_numeric(fire[v], errors="coerce") >= thr].copy()
    # bbox from remaining fire
    if not {"latitude", "longitude"}.issubset(fire_f.columns):
        return [{"type": "cut_to_fire", "in": fire_dir.as_posix(), "out": "", "actions": "no_latlon"}]
    lat = pd.to_numeric(fire_f["latitude"], errors="coerce").dropna()
    lon = pd.to_numeric(fire_f["longitude"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return [{"type": "cut_to_fire", "in": fire_dir.as_posix(), "out": "", "actions": "no_latlon_valid"}]
    ymin, ymax = float(lat.min()), float(lat.max())
    xmin, xmax = float(lon.min()), float(lon.max())
    # Save filtered fire
    ensure_dir((out_root / "FIRE_CUT"))
    fire_out = out_root / "FIRE_CUT" / "fire_filtered.csv"
    fire_f.to_csv(fire_out, index=False)
    entries.append({"type": "cut_to_fire", "in": "fire", "out": fire_out.as_posix(), "actions": f"metric={v} thr={thr:.3f}"})
    # Crop vectors: landcover polygons if available
    lc_in = out_root / "landcover"
    for shp in lc_in.glob("*.gpkg"):
        try:
            gdf = gpd.read_file(shp).to_crs("EPSG:4326")
            bbox_poly = gpd.GeoSeries([Point(xmin, ymin).buffer(0)], crs="EPSG:4326").total_bounds
            # use bounds crop first (fast)
            gdf2 = gdf.cx[xmin:xmax, ymin:ymax]
            outp = out_root / "LANDCOVER_CUT" / (shp.stem + "_cut.gpkg")
            ensure_dir(outp.parent)
            gdf2.to_file(outp, driver="GPKG")
            entries.append({"type": "crop_vector", "in": shp.as_posix(), "out": outp.as_posix(), "actions": f"bbox={xmin:.3f},{ymin:.3f},{xmax:.3f},{ymax:.3f}"})
        except Exception:
            continue
    # Crop rasters if rioxarray present
    if rxr is not None:
        elev_in_dirs = [out_root / "elevation", BASE / "ELEVATION" / "processed"]
        for d in elev_in_dirs:
            for tif in d.glob("*.tif"):
                try:
                    da = rxr.open_rasterio(tif, masked=True).squeeze()
                    da2 = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
                    outp = out_root / "ELEVATION_CUT" / (tif.stem + "_cut.tif")
                    ensure_dir(outp.parent)
                    da2.rio.to_raster(outp.as_posix(), compress="LZW")
                    entries.append({"type": "crop_raster", "in": tif.as_posix(), "out": outp.as_posix(), "actions": "clip_box"})
                except Exception:
                    continue
        # Climate cut if built
        clim_nc = out_root / "CLIMATE" / "climate_2024.nc"
        if clim_nc.exists():
            try:
                ds = xr.open_dataset(clim_nc)
                if {"x","y"}.issubset(ds.coords):
                    ds2 = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
                elif {"lon","lat"}.issubset(ds.coords):
                    ds2 = ds.sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
                else:
                    ds2 = ds
                outp = out_root / "CLIMATE_CUT" / "climate_2024_cut.nc"
                ensure_dir(outp.parent)
                ds2.to_netcdf(outp)
                entries.append({"type": "crop_nc", "in": clim_nc.as_posix(), "out": outp.as_posix(), "actions": "sel bbox"})
            except Exception:
                pass
    return entries


def strategic_merge_with_fire(out_root: Path) -> dict:
    """Merge datasets with fire as anchor; resolve duplicates by mean/nearest.
    Output: DATA_CLEANED/processed/merged_dataset (parquet or csv)
    """
    # Load fire
    fire_candidates = list((out_root / "FIRE_CUT").glob("*.csv")) or list((out_root / "FIRE").glob("*.csv"))
    if not fire_candidates:
        return {"type": "strategic_merge", "in": "fire", "out": "", "actions": "no_fire"}
    fire = pd.concat([pd.read_csv(p, low_memory=False) for p in fire_candidates], ignore_index=True)
    if not {"latitude","longitude"}.issubset(fire.columns):
        return {"type": "strategic_merge", "in": "fire", "out": "", "actions": "no_latlon"}
    fire["fire"] = 1
    gfire = gpd.GeoDataFrame(fire, geometry=gpd.points_from_xy(pd.to_numeric(fire["longitude"], errors="coerce"), pd.to_numeric(fire["latitude"], errors="coerce")), crs="EPSG:4326")

    merged = gfire[["latitude","longitude","fire","geometry"]].copy()

    # Landcover join (polygon contains)
    lc_dir = out_root / "LANDCOVER_CUT"
    lc_files = list(lc_dir.glob("*.gpkg"))
    if lc_files:
        try:
            lcg = gpd.read_file(lc_files[0]).to_crs("EPSG:4326")
            joined = gpd.sjoin(merged.set_geometry("geometry"), lcg, how="left", predicate="within")
            # Choose most informative categorical column
            cat_cols = [c for c in lcg.columns if c not in {lcg.geometry.name} and (is_categorical(lcg[c]) or lcg[c].dtype==object)]
            keep = cat_cols[0] if cat_cols else None
            if keep:
                merged["landcover_label"] = joined[keep]
        except Exception:
            pass

    # Elevation sample
    elev_dir = out_root / "ELEVATION_CUT"
    elev_tifs = list(elev_dir.glob("*.tif"))
    if rxr is not None and elev_tifs:
        try:
            da = rxr.open_rasterio(elev_tifs[0], masked=True).squeeze()
            transform = da.rio.transform()
            xs = merged["longitude"].to_numpy(); ys = merged["latitude"].to_numpy()
            cols, rows = (~transform) * (xs, ys)
            rows = np.clip(np.floor(rows).astype(int), 0, da.shape[0]-1)
            cols = np.clip(np.floor(cols).astype(int), 0, da.shape[1]-1)
            merged["elevation"] = da.values[rows, cols]
        except Exception:
            pass

    # Climate aggregate sample (mean/min/max/std or seasonal if available)
    clim_nc = out_root / "CLIMATE_CUT" / "climate_2024_cut.nc"
    if clim_nc.exists():
        try:
            ds = xr.open_dataset(clim_nc)
            # aggregate monthly to stats if time present
            ds_ag = {}
            for v in ["prec","tmax","tmin"]:
                if v in ds:
                    da = ds[v]
                    if "time" in da.dims:
                        ds_ag[f"{v}_mean"] = da.mean("time")
                        ds_ag[f"{v}_min"] = da.min("time")
                        ds_ag[f"{v}_max"] = da.max("time")
                        ds_ag[f"{v}_std"] = da.std("time")
            if ds_ag:
                # sample at points using nearest index
                for k, da in ds_ag.items():
                    if {"x","y"}.issubset(da.coords):
                        merged[k] = da.sel(x=("points", merged["longitude"]), y=("points", merged["latitude"]), method="nearest").values
                    elif {"lon","lat"}.issubset(da.coords):
                        merged[k] = da.sel(lon=("points", merged["longitude"]), lat=("points", merged["latitude"]), method="nearest").values
        except Exception:
            pass

    # Soil ID sample
    soil_nc = out_root / "SOIL" / "soil_id_grid.nc"
    if soil_nc.exists():
        try:
            ds = xr.open_dataset(soil_nc)
            var = "soil_id" if "soil_id" in ds else list(ds.data_vars)[0]
            da = ds[var]
            if {"lon","lat"}.issubset(da.coords):
                merged["soil_id"] = da.sel(lon=("points", merged["longitude"]), lat=("points", merged["latitude"]), method="nearest").values
            elif {"x","y"}.issubset(da.coords):
                merged["soil_id"] = da.sel(x=("points", merged["longitude"]), y=("points", merged["latitude"]), method="nearest").values
        except Exception:
            pass

    # Resolve duplicates by rounding coords and averaging
    merged["lat_r"] = merged["latitude"].round(5)
    merged["lon_r"] = merged["longitude"].round(5)
    num_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c]) and c not in ["latitude","longitude"]]
    grouped = merged.groupby(["lat_r","lon_r"], as_index=False)[num_cols].mean()
    grouped["latitude"] = grouped["lat_r"]
    grouped["longitude"] = grouped["lon_r"]
    grouped = grouped.drop(columns=["lat_r","lon_r"]) 

    out_dir = out_root / "processed"
    ensure_dir(out_dir)
    out_parquet = out_dir / "merged_dataset.parquet"
    try:
        grouped.to_parquet(out_parquet, index=False)
        outp = out_parquet
    except Exception:
        out_csv = out_dir / "merged_dataset.csv"
        grouped.to_csv(out_csv, index=False)
        outp = out_csv
    return {"type": "strategic_merge", "in": "DATA_CLEANED", "out": outp.as_posix(), "actions": f"rows={len(grouped)}"}


def clean_known_datasets(out_root: Path) -> list[dict]:
    entries: list[dict] = []
    entries.append(clean_climate_nc(BASE / "CLIMATE" / "processed" / "climate_2024.nc", out_root))
    entries.append(clean_soil_nc(BASE / "SOIL" / "processed" / "soil_id_aoi.nc", out_root))
    # Try common elevation path; skip gracefully if missing or rasterio not available
    elev_candidates = [
        BASE / "ELEVATION" / "processed" / "elevation_clipped.tif",
        BASE / "ELEVATION" / "be15_grd" / "be15_grd" / "processed" / "elevation_clipped.tif",
    ]
    for ep in elev_candidates:
        if ep.exists():
            entries.append(clean_elevation_raster(ep, out_root))
            break
    lc_candidates = [
        BASE / "LANDCOVER" / "processed" / "landcover_clipped.shp",
    ]
    for lp in lc_candidates:
        if lp.exists():
            entries.append(clean_landcover_vector(lp, out_root))
            break
    fire_paths = [
        BASE / "FIRE" / "viirs-jpss1_2024_Algeria_type2.csv",
        BASE / "FIRE" / "viirs-jpss1_2024_Tunisia_type2.csv",
        BASE / "FIRE" / "viirs-jpss1_2024_Algeria.csv",
        BASE / "FIRE" / "viirs-jpss1_2024_Tunisia.csv",
    ]
    entries.extend(clean_fire_csvs(fire_paths, out_root))
    return entries


def main():
    ap = argparse.ArgumentParser(description="Generic data cleaner")
    ap.add_argument("--src", type=Path, default=BASE, help="Source root directory")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root directory (DATA_CLEANED)")
    ap.add_argument("--targets", action="store_true", help="Run targeted cleaning for climate, soil, elevation, landcover, fire datasets")
    # New specialized tasks per user's request
    ap.add_argument("--filtered-climate", type=Path, default=None, help="Path to DATA_FILTRED/CLIMATE directory containing 2024 monthly files")
    ap.add_argument("--build-climate-2024", action="store_true", help="Merge filtered climate (prec,tmax,tmin) for 2024 into a single NetCDF")
    ap.add_argument("--soil-bil", type=Path, default=None, help="Path to HWSD2.bil (e.g., DATA/SOIL/HWSD2/cache/HWSD2.bil)")
    ap.add_argument("--build-soil-nc", action="store_true", help="Create a NetCDF with soil ID grid matched by lat/lon from HWSD2.bil")
    ap.add_argument("--fire-keep-not2", action="store_true", help="Filter fire CSVs to keep only rows where type != 2")
    ap.add_argument("--cut-to-fire", action="store_true", help="Cut all datasets to bounding box of fires above lower boxplot whisker on FRP")
    ap.add_argument("--strategic-merge", action="store_true", help="Merge cleaned datasets with fire as anchor; resolve duplicates by means/centroids")
    args = ap.parse_args()

    src_root: Path = args.src
    out_root: Path = args.out
    ensure_dir(out_root)

    entries: list[dict] = []
    for p in scan_sources(src_root):
        try:
            if p.suffix.lower() == ".csv":
                entries.append(clean_csv(p, out_root, src_root))
            else:
                entries.append(clean_vector(p, out_root, src_root))
        except Exception as e:
            # record error line and continue
            entries.append({"type": "error", "in": p.as_posix(), "out": "", "actions": str(e)})

    if args.targets:
        entries.extend(clean_known_datasets(out_root))

    # Build climate_2024.nc from filtered monthly files
    if args.build_climate_2024 and args.filtered_climate is not None:
        try:
            res = build_climate_nc_from_filtered(args.filtered_climate, out_root / "CLIMATE" / "climate_2024.nc")
            entries.append(res)
        except Exception as e:
            entries.append({"type": "climate_build", "in": str(args.filtered_climate), "out": "", "actions": f"error:{e}"})

    # Build soil ID NetCDF from HWSD2.bil
    if args.build_soil_nc and args.soil_bil is not None:
        try:
            res = build_soil_id_nc_from_bil(args.soil_bil, out_root / "SOIL" / "soil_id_grid.nc")
            entries.append(res)
        except Exception as e:
            entries.append({"type": "soil_build", "in": str(args.soil_bil), "out": "", "actions": f"error:{e}"})

    # Filter fire rows to keep type != 2
    if args.fire_keep_not2:
        try:
            res_list = filter_fire_type_not2(BASE / "FIRE", out_root / "FIRE")
            entries.extend(res_list)
        except Exception as e:
            entries.append({"type": "fire_filter", "in": (BASE/"FIRE").as_posix(), "out": "", "actions": f"error:{e}"})

    # Cut datasets to bounding box of fires above FRP lower whisker
    if args.cut_to_fire:
        try:
            res_list = cut_to_fire_threshold_and_crop(out_root)
            entries.extend(res_list)
        except Exception as e:
            entries.append({"type": "cut_to_fire", "in": "fire_cleaned", "out": "", "actions": f"error:{e}"})

    # Strategic merge anchored on fire
    if args.strategic_merge:
        try:
            res = strategic_merge_with_fire(out_root)
            entries.append(res)
        except Exception as e:
            entries.append({"type": "strategic_merge", "in": "data_cleaned", "out": "", "actions": f"error:{e}"})

    write_log(entries, out_root)


if __name__ == "__main__":
    main()
