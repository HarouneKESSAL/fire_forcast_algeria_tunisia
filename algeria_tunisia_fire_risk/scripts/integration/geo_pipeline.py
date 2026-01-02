#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Geospatial Processing Pipeline for Algeria–Tunisia AOI
- CLIMATE: clip → reproject_match(template) → unit checks → COG + stats
- ELEVATION: clip → reproject_match → slope/aspect → COG + stats
- LANDCOVER: rasterize vector or reproject raster (nearest) → COG + codebook
- SOIL: standardize CSVs to snake_case, rasterize if spatial layers provided
- FIRE: monthly point density rasters to template grid

Notes
- AOI: DATA/shapefiles/algeria_tunisia.shp (dissolve if needed)
- Template: DATA/CLIMATE/mean_tmin_2020_2024.tif
- Resampling: continuous=bilinear, categorical=nearest; preserve nodata
- COG export: gdal_translate -of COG -co COMPRESS=LZW -co OVERVIEWS=IGNORE_EXISTING
"""

from __future__ import annotations
import os
import io
import json
import math
import argparse
import subprocess
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import get_config, get_base
from scripts.logging_setup import setup_logging

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import CRS as PJCRS, Transformer

# --- Config & Paths (absolute) ---
CFG = get_config()
setup_logging(level=CFG.get("log", {}).get("level", "INFO"), fmt=CFG.get("log", {}).get("format"))
logger = logging.getLogger("pipeline")

BASE = get_base()
AOI_SHP = BASE / "shapefiles" / "algeria_tunisia.shp"
# Primary template target (may not exist if climate stats not yet built)
TEMPLATE = BASE / "CLIMATE" / "mean_tmin_2020_2024.tif"
# Fallback candidates (first existing will be used)
TEMPLATE_CANDIDATES = [
    TEMPLATE,
    BASE / "ELEVATION" / "be15_grd" / "processed" / "elevation_clipped.tif",
    BASE / "LANDCOVER" / "processed" / "landcover_raster.tif",
    BASE / "CLIMATE" / "processed" / "wc2.1_cruts4.09_5m_prec_2020-01_clipped.tif",
]

CLIMATE_DIR = BASE / "CLIMATE"
CLIMATE_OUT = CLIMATE_DIR / "processed"
ELEV_DIR = BASE / "ELEVATION" / "be15_grd" / "be15_grd"
ELEV_OUT = BASE / "ELEVATION" / "be15_grd" / "processed"
LANDCOVER_DIR = BASE / "LANDCOVER"
LANDCOVER_OUT = LANDCOVER_DIR / "processed"
SOIL_DIR = BASE / "SOIL"
SOIL_OUT = SOIL_DIR / "processed"
FIRE_DIR = BASE / "FIRE"

META_ROOT = BASE / "processed" / "_metadata"
META_ROOT.mkdir(parents=True, exist_ok=True)
CLIMATE_OUT.mkdir(parents=True, exist_ok=True)
ELEV_OUT.mkdir(parents=True, exist_ok=True)
LANDCOVER_OUT.mkdir(parents=True, exist_ok=True)
(SOIL_OUT / "cleaned").mkdir(parents=True, exist_ok=True)


# ----------------------
# Utility functions
# ----------------------

def load_aoi() -> gpd.GeoDataFrame:
    """Load AOI and dissolve to single geometry in EPSG:4326."""
    aoi = gpd.read_file(AOI_SHP)
    if aoi.empty:
        raise FileNotFoundError(f"AOI shapefile empty or missing: {AOI_SHP}")
    aoi = aoi.to_crs("EPSG:4326")
    if len(aoi) > 1:
        aoi = aoi.dissolve().reset_index(drop=True)
    # fix potential topology errors
    aoi["geometry"] = aoi.buffer(0)
    return aoi


def open_template() -> xr.DataArray:
    for candidate in TEMPLATE_CANDIDATES:
        if candidate.exists():
            da = _ensure_dataarray(rxr.open_rasterio(candidate, masked=True)).squeeze()
            if da.rio.crs is None:
                raise ValueError(f"Template missing CRS: {candidate}")
            return da
    raise FileNotFoundError("No template raster found among candidates: " + ", ".join(c.as_posix() for c in TEMPLATE_CANDIDATES))


def _ensure_dataarray(raster) -> xr.DataArray:
    """Normalize rioxarray.open_rasterio outputs to DataArray."""
    if isinstance(raster, list):
        raster = raster[0]
    if isinstance(raster, xr.Dataset):
        if len(raster.data_vars) == 1:
            raster = next(iter(raster.data_vars.values()))
        else:
            raster = raster.to_array().isel(variable=0)
    if not isinstance(raster, xr.DataArray):
        raise TypeError(f"Unsupported raster type: {type(raster)}")
    return raster


def clip_to_aoi(da: xr.DataArray, aoi: gpd.GeoDataFrame) -> xr.DataArray:
    geoms = [aoi.geometry.union_all()]
    clipped = da.rio.clip(geoms, aoi.crs, drop=True)
    return clipped


def to_template(da: xr.DataArray, tpl: xr.DataArray, resampling: str = "bilinear") -> xr.DataArray:
    """Reproject to match template grid using rioxarray.reproject_match.
    resampling: 'nearest' or 'bilinear'
    """
    from rasterio.enums import Resampling
    res = Resampling.bilinear if resampling == "bilinear" else Resampling.nearest
    return da.rio.reproject_match(tpl, resampling=res)


def _np_stats(arr: np.ndarray, nodata: Optional[float]) -> Dict[str, float]:
    mask = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        mask &= (arr != nodata)
    mask &= np.isfinite(arr)
    valid = arr[mask]
    res = {
        "count": float(valid.size),
        "min": float(np.nanmin(valid)) if valid.size else float("nan"),
        "mean": float(np.nanmean(valid)) if valid.size else float("nan"),
        "max": float(np.nanmax(valid)) if valid.size else float("nan"),
        "std": float(np.nanstd(valid)) if valid.size else float("nan"),
        "pct_nodata": float(100.0 * (arr.size - valid.size) / arr.size) if arr.size else 0.0,
    }
    return res

# ----------------------
# Data cleaning utilities (missing + outliers)
# ----------------------

def _detect_symmetry(arr: np.ndarray) -> str:
    """Detect distribution symmetry using skewness.
    Returns: 'symmetric' if |skewness| < 0.5, else 'asymmetric'
    """
    from scipy import stats
    finite = arr[np.isfinite(arr)]
    if finite.size < 10:
        return "symmetric"
    skew = stats.skew(finite)
    return "symmetric" if abs(skew) < 0.5 else "asymmetric"


def _is_categorical(arr: np.ndarray, max_unique_ratio: float = 0.05, max_unique_count: int = 20) -> bool:
    """Detect if array is categorical based on unique value ratio and integer dtype.
    
    Returns True only if:
    - Values are integer-like AND
    - Unique ratio < 5% (very few unique values) OR unique count <= 20
    - Data size is reasonable (> 20 values to avoid false positives on small arrays)
    """
    finite = arr[np.isfinite(arr)]
    if finite.size < 20:  # Too small to reliably detect
        return False
    unique_vals = np.unique(finite)
    unique_ratio = len(unique_vals) / finite.size
    # Check if values are effectively integers
    is_int_like = np.allclose(finite, np.rint(finite), rtol=0, atol=1e-9)
    # Categorical if: integer-like AND (very few unique relative to size OR small number of unique values)
    return is_int_like and (unique_ratio < max_unique_ratio or len(unique_vals) <= max_unique_count)


def _spatial_mean_fill(arr: np.ndarray, mask_outliers: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Replace outliers with mean of spatial neighbors using convolution.
    Args:
        arr: 2D array with values
        mask_outliers: boolean mask where True indicates outlier
        kernel_size: size of neighborhood window (must be odd)
    Returns: array with outliers replaced by local spatial mean
    """
    from scipy.ndimage import generic_filter
    
    result = arr.copy()
    if arr.ndim != 2:
        return result
    
    def local_mean(values):
        """Compute mean of valid (non-outlier) neighbors."""
        center_idx = len(values) // 2
        # Exclude center pixel and use only valid neighbors
        neighbors = np.concatenate([values[:center_idx], values[center_idx+1:]])
        valid = neighbors[np.isfinite(neighbors)]
        return np.mean(valid) if valid.size > 0 else values[center_idx]
    
    # Create a copy where outliers are marked
    temp = arr.copy()
    
    # Apply local mean only to outlier locations
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask_outliers[i, j]:
                # Extract neighborhood
                i_start = max(0, i - kernel_size // 2)
                i_end = min(arr.shape[0], i + kernel_size // 2 + 1)
                j_start = max(0, j - kernel_size // 2)
                j_end = min(arr.shape[1], j + kernel_size // 2 + 1)
                
                neighborhood = arr[i_start:i_end, j_start:j_end]
                neighbor_mask = mask_outliers[i_start:i_end, j_start:j_end]
                
                # Get valid (non-outlier) neighbors
                valid_neighbors = neighborhood[~neighbor_mask & np.isfinite(neighborhood)]
                
                if valid_neighbors.size > 0:
                    result[i, j] = np.mean(valid_neighbors)
    
    return result


def sanitize_dataarray(
    da: xr.DataArray,
    layer_name: str,
    max_fill_pct: float = 5.0,
    use_smart_imputation: bool = True,
    outlier_iqr: bool = True,
    clip_percentiles: Optional[Tuple[float, float]] = None,
) -> Tuple[xr.DataArray, List[str]]:
    """Clean raster values with smart outlier imputation.
    
    Smart Imputation Strategy:
      1. Detect data characteristics (symmetric/asymmetric, categorical, spatial)
      2. Identify outliers using IQR method (Q1-1.5*IQR, Q3+1.5*IQR)
      3. Replace outliers based on data type:
         - Symmetric continuous: mean
         - Asymmetric continuous: median
         - Categorical: mode
         - Geospatial (2D): mean of nearest valid neighbors
      4. Fill remaining missing values if < max_fill_pct
    
    Args:
        da: Input DataArray
        layer_name: Layer identifier for logging
        max_fill_pct: Max percentage of missing values to fill
        use_smart_imputation: If True, use smart strategy; else use simple capping
        outlier_iqr: Use IQR method for outlier detection
        clip_percentiles: Alternative percentile-based capping
    
    Returns:
        (cleaned DataArray, notes list)
    """
    notes: List[str] = []
    arr = da.values.astype(float, copy=True)
    original_shape = arr.shape
    nodata = da.rio.nodata
    
    # Step 1: Identify initial missing/invalid values
    mask_invalid = ~np.isfinite(arr)
    if nodata is not None:
        mask_invalid |= (arr == nodata)
    
    invalid_pct = 100.0 * mask_invalid.sum() / arr.size if arr.size else 0.0
    
    if use_smart_imputation and arr.size > 0:
        # Step 2: Detect data characteristics
        finite = arr[~mask_invalid]
        
        if finite.size < 10:
            notes.append("insufficient_data_for_smart_imputation")
            use_smart_imputation = False
        else:
            is_categorical = _is_categorical(finite)
            symmetry = _detect_symmetry(finite) if not is_categorical else "n/a"
            is_spatial = (arr.ndim == 2 and arr.shape[0] > 3 and arr.shape[1] > 3)
            
            notes.append(f"data_profile: categorical={is_categorical}, symmetry={symmetry}, spatial={is_spatial}")
            
            # Step 3: Identify outliers using IQR (skip for categorical data)
            outlier_mask = np.zeros(arr.shape, dtype=bool)
            
            if outlier_iqr and not is_categorical:
                q1, q3 = np.nanpercentile(finite, [25, 75])
                iqr = q3 - q1
                
                if iqr > 0:
                    lo = q1 - 1.5 * iqr
                    hi = q3 + 1.5 * iqr
                    outlier_mask = (arr < lo) | (arr > hi)
                    outlier_mask &= ~mask_invalid  # Only mark valid values as outliers
                    
                    outlier_count = outlier_mask.sum()
                    outlier_pct = 100.0 * outlier_count / arr.size if arr.size else 0.0
                    
                    if outlier_count > 0:
                        notes.append(f"outliers_detected: {outlier_count} ({outlier_pct:.2f}%) outside [{lo:.3g}, {hi:.3g}]")
                        
                        # Step 4: Smart imputation based on data type
                        # Priority: Spatial > Categorical > Distribution-based
                        if is_spatial and not is_categorical:
                            # Use spatial mean of neighbors for geospatial continuous data
                            arr = _spatial_mean_fill(arr, outlier_mask, kernel_size=3)
                            notes.append(f"imputed_spatial: neighbor_mean (3x3 kernel)")
                        
                        elif is_categorical and not is_spatial:
                            # Use mode for pure categorical data (non-spatial)
                            from scipy import stats
                            mode_val = stats.mode(finite, keepdims=False)[0]
                            arr[outlier_mask] = mode_val
                            notes.append(f"imputed_categorical: mode={mode_val:.3g}")
                        
                        elif symmetry == "symmetric":
                            # Use mean for symmetric distributions
                            mean_val = float(np.nanmean(finite[(finite >= lo) & (finite <= hi)]))
                            arr[outlier_mask] = mean_val
                            notes.append(f"imputed_symmetric: mean={mean_val:.3g}")
                        
                        else:  # asymmetric or categorical+spatial
                            # Use median for asymmetric distributions
                            median_val = float(np.nanmedian(finite[(finite >= lo) & (finite <= hi)]))
                            arr[outlier_mask] = median_val
                            notes.append(f"imputed_asymmetric: median={median_val:.3g}")
            
            # Step 5: Fill remaining missing values if below threshold
            mask_still_invalid = ~np.isfinite(arr)
            if nodata is not None:
                mask_still_invalid |= (arr == nodata)
            
            remaining_invalid_pct = 100.0 * mask_still_invalid.sum() / arr.size if arr.size else 0.0
            
            if mask_still_invalid.any() and remaining_invalid_pct <= max_fill_pct:
                valid_after_outlier = arr[~mask_still_invalid]
                if valid_after_outlier.size:
                    if is_categorical:
                        from scipy import stats
                        fill_val = stats.mode(valid_after_outlier, keepdims=False)[0]
                        notes.append(f"filled_missing: {remaining_invalid_pct:.2f}% with mode={fill_val:.3g}")
                    elif symmetry == "symmetric":
                        fill_val = float(np.nanmean(valid_after_outlier))
                        notes.append(f"filled_missing: {remaining_invalid_pct:.2f}% with mean={fill_val:.3g}")
                    else:
                        fill_val = float(np.nanmedian(valid_after_outlier))
                        notes.append(f"filled_missing: {remaining_invalid_pct:.2f}% with median={fill_val:.3g}")
                    arr[mask_still_invalid] = fill_val
            elif mask_still_invalid.any():
                notes.append(f"missing_retained: {remaining_invalid_pct:.2f}% (exceeds {max_fill_pct}%)")
    
    else:
        # Fallback to original simple approach
        if mask_invalid.any() and invalid_pct <= max_fill_pct:
            valid = arr[~mask_invalid]
            if valid.size:
                med = float(np.nanmedian(valid))
                arr[mask_invalid] = med
                notes.append(f"filled_missing_simple: {invalid_pct:.2f}% median={med:.3g}")
        elif mask_invalid.any():
            notes.append(f"missing_retained: {invalid_pct:.2f}% (exceeds {max_fill_pct}%)")
        
        # Simple outlier capping
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if clip_percentiles is not None:
                p_low, p_high = clip_percentiles
                lo, hi = np.nanpercentile(finite, [p_low, p_high])
                before_min, before_max = float(np.nanmin(arr)), float(np.nanmax(arr))
                arr = np.clip(arr, lo, hi)
                notes.append(f"cap_percentiles[{p_low},{p_high}] ({before_min:.3g},{before_max:.3g})→({lo:.3g},{hi:.3g})")
            elif outlier_iqr:
                q1, q3 = np.nanpercentile(finite, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    lo = q1 - 1.5 * iqr
                    hi = q3 + 1.5 * iqr
                    before_min, before_max = float(np.nanmin(arr)), float(np.nanmax(arr))
                    clipped = np.clip(arr, lo, hi)
                    if (clipped != arr).any():
                        arr = clipped
                        notes.append(f"cap_iqr_simple: ({before_min:.3g},{before_max:.3g})→({lo:.3g},{hi:.3g})")
    
    # Reassign dtype
    try:
        if np.issubdtype(da.dtype, np.integer):
            arr = np.rint(arr).astype(da.dtype)
        da.values = arr
    except Exception:
        pass
    
    return da, notes


def compute_stats(da: xr.DataArray) -> Dict[str, float]:
    nodata = da.rio.nodata
    arr = da.squeeze().values.astype(float, copy=False)
    return _np_stats(arr, nodata)


def _write_temp_gtiff(da: xr.DataArray, tmp_path: Path) -> None:
    # Preserve nodata, compress LZW to intermediate; COG later via gdal
    da.rio.to_raster(tmp_path.as_posix(), compress="LZW")


def write_cog(in_path: Path, out_path: Path) -> None:
    """Write Cloud Optimized GeoTIFF using gdal_translate if available, else copy."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if gdal_translate is available
    try:
        result = subprocess.run(["gdal_translate", "--version"], 
                              capture_output=True, check=False)
        if result.returncode != 0:
            raise FileNotFoundError("gdal_translate not available")
    except FileNotFoundError:
        # Fallback: just copy the file (already compressed as LZW)
        logger.warning("gdal_translate not found, copying GeoTIFF without COG optimization")
        import shutil
        shutil.copy(in_path, out_path)
        return
    
    # Use gdal_translate for COG
    cmd = [
        "gdal_translate", in_path.as_posix(), out_path.as_posix(),
        "-of", "COG",
        "-co", "COMPRESS=LZW",
        "-co", "OVERVIEWS=IGNORE_EXISTING",
        "-co", "BLOCKSIZE=512",
    ]
    subprocess.run(cmd, check=True)


def save_stats_json(stats: Dict[str, float], raster_path: Path) -> None:
    jpath = raster_path.with_suffix(raster_path.suffix + ".stats.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def save_histogram_png(da: xr.DataArray, out_path: Path, bins: int = 100) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = da.squeeze().values
        nodata = da.rio.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(arr, bins=bins, color="#3b82f6", alpha=0.8)
        ax.set_title(out_path.name)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".hist.png"))
        plt.close(fig)
    except Exception:
        pass


def log_metadata(entry: Dict[str, object]) -> None:
    dd_csv = META_ROOT / "data_dictionary.csv"
    df = pd.DataFrame([entry])
    if dd_csv.exists():
        df_existing = pd.read_csv(dd_csv)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(dd_csv, index=False)

    # Log line in markdown
    log_md = META_ROOT / "log.md"
    line = f"- {datetime.now(timezone.utc).isoformat()}Z | {entry.get('layer')} → {entry.get('path')} | {entry.get('notes','')}\n"
    with open(log_md, "a", encoding="utf-8") as f:
        f.write(line)

    # STAC-like per-layer metadata JSON
    try:
        layer_id = str(entry.get("layer"))
        meta_json = META_ROOT / f"{layer_id}.metadata.json"
        stac_obj = {
            "stac_version": CFG.get("metadata", {}).get("stac_version", "1.0.0"),
            "type": "Feature",
            "id": layer_id,
            "properties": {
                "created": datetime.now(timezone.utc).isoformat() + "Z",
                "notes": entry.get("notes", ""),
                "units": entry.get("units"),
                "layer_type": entry.get("type"),
            },
            "assets": {
                "data": {
                    "href": entry.get("path"),
                    "type": "application/octet-stream",
                }
            },
            "extensions": [],
            "license": CFG.get("metadata", {}).get("license", "CC-BY-4.0"),
        }
        with open(meta_json, "w", encoding="utf-8") as mf:
            json.dump(stac_obj, mf, indent=2)
    except Exception as e:
        logger.warning("Failed writing STAC metadata for %s: %s", entry.get("layer"), e)


def ensure_units(da: xr.DataArray, layer_name: str) -> Tuple[xr.DataArray, str]:
    """Best-effort units check and convert.
    - tmin/tmax: expect °C; if looks like Kelvin (> 150), convert K→°C
    - prec: expect mm; assume already mm
    Returns (possibly converted DataArray, units string)
    """
    name = layer_name.lower()
    arr = da.values
    units = "unknown"
    if any(k in name for k in ["tmin", "tmax", "temp"]):
        units = "degC"
        sample = np.nanmedian(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else np.nan
        if sample > 150:  # likely Kelvin
            da = da - 273.15
            units = "degC (from K)"
    elif "prec" in name or "ppt" in name or "rain" in name:
        units = "mm"
    return da, units


# ----------------------
# CLIMATE
# ----------------------

def process_climate() -> None:
    aoi = load_aoi()
    tpl = open_template()

    # Find climate TIFs in subdirectories (exclude processed folder)
    # Filter for 2024 data only
    tif_paths = []
    for subdir in ["wc2.1_cruts4.09_5m_prec_2020-2024", "wc2.1_cruts4.09_5m_tmax_2020-2024", "wc2.1_cruts4.09_5m_tmin_2020-2024"]:
        subdir_path = CLIMATE_DIR / subdir
        if subdir_path.exists():
            # Only include files with "2024" in the filename
            tif_paths.extend([p for p in subdir_path.glob("*.tif") if "2024" in p.stem])
    
    if not tif_paths:
        logger.warning("No climate .tif files found for 2024 in subdirectories")
        return

    for p in tif_paths:
        try:
            da = _ensure_dataarray(rxr.open_rasterio(p, masked=True)).squeeze()
            da, units = ensure_units(da, p.stem)
            da = clip_to_aoi(da, aoi)
            da = to_template(da, tpl, resampling="bilinear")

            # Smart outlier imputation
            use_smart = CFG.get("cleaning", {}).get("use_smart_imputation", True)
            da, notes = sanitize_dataarray(da, p.stem, use_smart_imputation=use_smart)

            # Write COG
            tmp = CLIMATE_OUT / f"{p.stem}_tmp.tif"
            out = CLIMATE_OUT / f"{p.stem}_clipped.tif"
            _write_temp_gtiff(da, tmp)
            write_cog(tmp, out)
            tmp.unlink(missing_ok=True)

            stats_after = compute_stats(da)
            save_stats_json(stats_after, out)
            save_histogram_png(da, out)

            entry = {
                "layer": p.stem,
                "path": out.as_posix(),
                "type": "continuous",
                "CRS": str(tpl.rio.crs),
                "res": str(tpl.rio.resolution()),
                "nodata": da.rio.nodata,
                "units": units,
                "transform": str(tpl.rio.transform()),
                "source": p.as_posix(),
                "date": datetime.now(timezone.utc).isoformat() + "Z",
                "notes": "; ".join(notes),
            }
            log_metadata(entry)
            logger.info("CLIMATE processed %s", out.name)
        except Exception as e:
            logger.warning("Climate processing failed for %s: %s", p.name, e)


# ----------------------
# ELEVATION
# ----------------------

def _open_esri_grid(folder: Path) -> xr.DataArray:
    # Typical first band file name in Esri grid
    cand = folder / "w001001.adf"
    if not cand.exists():
        # try to find any *.adf with raster band
        files = list(folder.glob("*.adf"))
        if not files:
            raise FileNotFoundError(f"No .adf found in {folder}")
        cand = files[0]
    with rasterio.open(cand) as src:
        arr = src.read(1)
        da = xr.DataArray(arr, dims=("y", "x"))
        da = da.rio.write_crs(src.crs)
        da = da.rio.write_transform(src.transform)
        da = da.rio.write_nodata(src.nodata)
    return da


def _slope_aspect(da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    # Compute slope/aspect in degrees using Horn's method approximation via numpy gradients
    transform: Affine = da.rio.transform()
    xres = abs(transform.a)
    yres = abs(transform.e)
    arr = da.values.astype(float)
    nodata = da.rio.nodata
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= (arr != nodata)
    # simple padding to compute gradients
    arr2 = np.where(mask, arr, np.nan)
    gy, gx = np.gradient(arr2, yres, xres)
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect = np.degrees(np.arctan2(gy, -gx))
    aspect = np.where(np.isnan(aspect), np.nan, (aspect + 360) % 360)
    slope = np.degrees(slope_rad)
    slope_da = xr.DataArray(slope, dims=da.dims).rio.write_crs(da.rio.crs).rio.write_transform(transform).rio.write_nodata(np.nan)
    aspect_da = xr.DataArray(aspect, dims=da.dims).rio.write_crs(da.rio.crs).rio.write_transform(transform).rio.write_nodata(np.nan)
    return slope_da, aspect_da


def process_elevation() -> None:
    aoi = load_aoi()
    tpl = open_template()
    try:
        dem = _open_esri_grid(ELEV_DIR)
        dem = clip_to_aoi(dem, aoi)
        dem = to_template(dem, tpl, resampling="bilinear")
        
        use_smart = CFG.get("cleaning", {}).get("use_smart_imputation", True)
        dem, dem_notes = sanitize_dataarray(dem, "elevation", use_smart_imputation=use_smart)

        tmp = ELEV_OUT / "elevation_tmp.tif"
        out = ELEV_OUT / "elevation_clipped.tif"
        _write_temp_gtiff(dem, tmp)
        write_cog(tmp, out)
        tmp.unlink(missing_ok=True)
        save_stats_json(compute_stats(dem), out)
        save_histogram_png(dem, out)

        slope, aspect = _slope_aspect(dem)
        slope, slope_notes = sanitize_dataarray(slope, "slope_deg", use_smart_imputation=use_smart)
        aspect, aspect_notes = sanitize_dataarray(aspect, "aspect_deg", use_smart_imputation=use_smart)
        for name, da in [("slope_deg", slope), ("aspect_deg", aspect)]:
            ttmp = ELEV_OUT / f"{name}_tmp.tif"
            tout = ELEV_OUT / f"{name}.tif"
            _write_temp_gtiff(da, ttmp)
            write_cog(ttmp, tout)
            ttmp.unlink(missing_ok=True)
            save_stats_json(compute_stats(da), tout)
            save_histogram_png(da, tout)
        for name, da in [("elevation", dem), ("slope_deg", slope), ("aspect_deg", aspect)]:
            log_metadata({
                "layer": name,
                "path": (ELEV_OUT / ("elevation_clipped.tif" if name=="elevation" else f"{name}.tif")).as_posix(),
                "type": "continuous",
                "CRS": str(tpl.rio.crs),
                "res": str(tpl.rio.resolution()),
                "nodata": da.rio.nodata,
                "units": "m" if name=="elevation" else "deg",
                "transform": str(tpl.rio.transform()),
                "source": str(ELEV_DIR),
                "date": datetime.now(timezone.utc).isoformat() + "Z",
                "notes": "; ".join(dem_notes if name=="elevation" else (slope_notes if name=="slope_deg" else aspect_notes)),
            })
        logger.info("ELEVATION processed")
    except Exception as e:
        logger.warning("Elevation processing failed: %s", e)


# ----------------------
# LANDCOVER
# ----------------------

def _pick_class_field(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "class", "CLASS", "code", "CODE", "LC_TYPE", "landcover", "land_class", "VALUE", "value"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback to first non-geometry
    for c in gdf.columns:
        if c != gdf.geometry.name:
            return c
    raise ValueError("No attribute column found for landcover codes")


def rasterize_vector(shp_path: Path, tpl: xr.DataArray, out_path: Path) -> Tuple[Path, pd.DataFrame]:
    gdf = gpd.read_file(shp_path).to_crs(tpl.rio.crs)
    field = _pick_class_field(gdf)
    vals = gdf[field]
    # factorize to integer codes
    codes, uniques = pd.factorize(vals.fillna("NA"))
    gdf["_code"] = codes.astype(np.int32)

    transform = tpl.rio.transform()
    out_shape = (tpl.rio.height, tpl.rio.width)
    shapes = list(zip(gdf.geometry, gdf["_code"]))

    arr = rasterize(shapes=shapes, out_shape=out_shape, transform=transform, fill=0, dtype="int32")
    da = xr.DataArray(arr, dims=("y", "x"))
    da = da.rio.write_crs(tpl.rio.crs).rio.write_transform(transform).rio.write_nodata(0)

    tmp = out_path.with_suffix(".tmp.tif")
    _write_temp_gtiff(da, tmp)
    write_cog(tmp, out_path)
    tmp.unlink(missing_ok=True)

    codebook = pd.DataFrame({"code": range(len(uniques)), "label": uniques})
    return out_path, codebook


def process_landcover() -> None:
    tpl = open_template()
    aoi = load_aoi()

    # Prefer processed vector if present
    shp = LANDCOVER_OUT / "landcover_clipped.shp"
    codebook_path = LANDCOVER_OUT / "landcover_codebook.csv"
    codebook_json = LANDCOVER_OUT / "landcover_codebook.json"

    try:
        if shp.exists():
            out = LANDCOVER_OUT / "landcover_raster.tif"
            out_path, codebook = rasterize_vector(shp, tpl, out)
            codebook.to_csv(codebook_path, index=False)
            try:
                # Write JSON mapping code->label using explicit iteration
                mapping = {}
                for _, rec in codebook.iterrows():
                    try:
                        mapping[int(rec["code"])] = str(rec["label"])
                    except Exception:
                        continue
                codebook_json.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
            except Exception:
                logger.warning("Failed writing landcover_codebook.json")
            da = _ensure_dataarray(rxr.open_rasterio(out_path, masked=True)).squeeze()
            # Landcover is categorical - smart imputation will use mode
            use_smart = CFG.get("cleaning", {}).get("use_smart_imputation", True)
            da, lc_notes = sanitize_dataarray(da, "landcover", use_smart_imputation=use_smart, outlier_iqr=False)
            save_stats_json(compute_stats(da), out_path)
            save_histogram_png(da, out_path)
            log_metadata({
                "layer": "landcover",
                "path": out_path.as_posix(),
                "type": "categorical",
                "CRS": str(tpl.rio.crs),
                "res": str(tpl.rio.resolution()),
                "nodata": 0,
                "units": "class_code",
                "transform": str(tpl.rio.transform()),
                "source": shp.as_posix(),
                "date": datetime.now(timezone.utc).isoformat() + "Z",
            })
            logger.info("LANDCOVER rasterized from vector")
        else:
            # Otherwise find any rasters under LANDCOVER
            rasters = list(LANDCOVER_DIR.glob("**/*.tif"))
            for p in rasters:
                da = _ensure_dataarray(rxr.open_rasterio(p, masked=True)).squeeze()
                da = clip_to_aoi(da, aoi)
                da = to_template(da, tpl, resampling="nearest")
                out = LANDCOVER_OUT / f"{p.stem}_clipped.tif"
                tmp = out.with_suffix(".tmp.tif")
                _write_temp_gtiff(da, tmp)
                write_cog(tmp, out)
                tmp.unlink(missing_ok=True)
                save_stats_json(compute_stats(da), out)
                save_histogram_png(da, out)
                log_metadata({
                    "layer": p.stem,
                    "path": out.as_posix(),
                    "type": "categorical",
                    "CRS": str(tpl.rio.crs),
                    "res": str(tpl.rio.resolution()),
                    "nodata": da.rio.nodata,
                    "units": "class_code",
                    "transform": str(tpl.rio.transform()),
                    "source": p.as_posix(),
                    "date": datetime.now(timezone.utc).isoformat() + "Z",
                })
            logger.info("LANDCOVER rasters processed")
    except Exception as e:
        logger.warning("Landcover processing failed: %s", e)


# ----------------------
# SOIL (CSV standardization; rasterization if applicable)
# ----------------------

def snake_case(s: str) -> str:
    return (
        s.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
        .replace("(", "").replace(")", "").replace("%", "pct")
        .lower()
    )


def process_soil_clean_csvs() -> None:
    cleaned_dir = SOIL_OUT / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted((SOIL_OUT).glob("HWSD2_LAYERS_D*.csv"))
    if not csvs:
        print("No HWSD2_LAYERS_D*.csv found in", SOIL_OUT)
        return
    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
            df.columns = [snake_case(c) for c in df.columns]
            # Numeric columns small NA fill
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            for c in num_cols:
                na_pct = df[c].isna().mean() * 100
                if 0 < na_pct <= 5:
                    df[c] = df[c].fillna(df[c].median())
            out_csv = cleaned_dir / (csv.stem + "_clean.csv")
            df.to_csv(out_csv, index=False)
            log_metadata({
                "layer": out_csv.stem,
                "path": out_csv.as_posix(),
                "type": "table",
                "CRS": "n/a",
                "res": "n/a",
                "nodata": "n/a",
                "units": "various",
                "transform": "n/a",
                "source": csv.as_posix(),
                "date": datetime.now(timezone.utc).isoformat() + "Z",
            })
            logger.info("SOIL cleaned %s", out_csv.name)
        except Exception as e:
            logger.warning("Soil CSV clean failed for %s: %s", csv.name, e)


# ----------------------
# FIRE monthly point density
# ----------------------

def process_fire_2024() -> None:
    aoi = load_aoi()
    tpl = open_template()

    csvs = sorted(FIRE_DIR.glob("viirs-jpss1_2024_*.csv"))
    if not csvs:
        print("No FIRE CSVs found in", FIRE_DIR)
        return

    # Build per-month rasters
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
            # enforce types
            if "latitude" in df.columns and "longitude" in df.columns:
                df = df.dropna(subset=["latitude", "longitude"]).copy()
                df["lat"] = df["latitude"].astype(float)
                df["lon"] = df["longitude"].astype(float)
            else:
                # try generic
                df["lat"] = df.filter(regex="lat", axis=1).iloc[:, 0].astype(float)
                df["lon"] = df.filter(regex="lon|lng", axis=1).iloc[:, 0].astype(float)

            # datetime parse (UTC)
            ts_col = None
            for c in ["acq_time", "acq_date", "datetime", "time", "timestamp"]:
                if c in df.columns:
                    ts_col = c
                    break
            if ts_col is None:
                # fabricate month as unknown; skip
                df["month"] = 1
            else:
                df["dt"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
                df["month"] = df["dt"].dt.month.fillna(1).astype(int)

            # clip to AOI bbox using shapely within
            g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
            g = g.to_crs(aoi.crs or "EPSG:4326")
            g = g[g.within(aoi.geometry.union_all())].copy()
            if g.empty:
                print("⚠️ No FIRE points within AOI for", csv.name)
                continue

            # Bin to template grid
            transform = tpl.rio.transform()
            height, width = tpl.rio.height, tpl.rio.width
            nodata = 0

            for m in sorted(g["month"].unique()):
                gm = g[g["month"] == m]
                counts = np.zeros((height, width), dtype=np.int32)
                # convert lon/lat to row/col
                cols, rows = ~transform * (gm["lon"].values, gm["lat"].values)
                rows = np.clip(np.floor(rows).astype(int), 0, height - 1)
                cols = np.clip(np.floor(cols).astype(int), 0, width - 1)
                for r, c in zip(rows, cols):
                    counts[r, c] += 1

                da = xr.DataArray(counts, dims=("y", "x")).rio.write_crs(tpl.rio.crs).rio.write_transform(transform).rio.write_nodata(nodata)
                out = FIRE_DIR / f"viirs_count_2024{m:02d}.tif"
                tmp = out.with_suffix(".tmp.tif")
                _write_temp_gtiff(da, tmp)
                write_cog(tmp, out)
                tmp.unlink(missing_ok=True)
                save_stats_json(compute_stats(da), out)
                save_histogram_png(da, out)
                log_metadata({
                    "layer": f"viirs_count_2024{m:02d}",
                    "path": out.as_posix(),
                    "type": "count",
                    "CRS": str(tpl.rio.crs),
                    "res": str(tpl.rio.resolution()),
                    "nodata": nodata,
                    "units": "count",
                    "transform": str(tpl.rio.transform()),
                    "source": csv.as_posix(),
                    "date": datetime.now(timezone.utc).isoformat() + "Z",
                })
                logger.info("FIRE processed month %s (%s)", m, csv.name)
        except Exception as e:
            logger.warning("FIRE processing failed for %s: %s", csv.name, e)


# ----------------------
# NetCDF utilities & Fire filtering
# ----------------------

def _month_from_name(name: str) -> Optional[pd.Timestamp]:
    """Extract YYYY-MM from a filename like *_2024-03_*.tif; return Timestamp at first of month."""
    import re
    m = re.search(r"(19|20)\d{2}-\d{2}", name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(0) + "-01")
    except Exception:
        return None


def build_climate_nc_2024(out_path: Path | None = None) -> Path:
    """Merge 2024 climate (prec, tmax, tmin) from CLIMATE/processed into a single NetCDF.
    Assumes files named like *prec_2024-MM_clipped.tif, etc.
    """
    out_path = out_path or (CLIMATE_OUT / "climate_2024.nc")
    var_patterns = {
        "prec": "*prec_2024-??_clipped.tif",
        "tmax": "*tmax_2024-??_clipped.tif",
        "tmin": "*tmin_2024-??_clipped.tif",
    }

    stacks: dict[str, list[tuple[pd.Timestamp, xr.DataArray]]] = {k: [] for k in var_patterns}
    for var, pat in var_patterns.items():
        for p in sorted(CLIMATE_OUT.glob(pat)):
            ts = _month_from_name(p.name)
            if ts is None or ts.year != 2024:
                continue
            da = _ensure_dataarray(rxr.open_rasterio(p, masked=True)).squeeze()
            stacks[var].append((ts, da))

    # Build Dataset with aligned dims
    ds_vars = {}
    coords_y = None
    coords_x = None
    times_all = None
    for var, items in stacks.items():
        if not items:
            continue
        items.sort(key=lambda t: t[0])
        times = [t for t, _ in items]
        data = [np.asarray(da.values) for _, da in items]
        arr = np.stack(data, axis=0)  # (time, y, x)
        if coords_y is None or coords_x is None:
            # attempt to get coords
            da0 = items[0][1]
            y = da0.coords.get("y", None)
            x = da0.coords.get("x", None)
            if y is not None and x is not None:
                coords_y = np.asarray(y.values)
                coords_x = np.asarray(x.values)
        ds_vars[var] = (("time", "y", "x"), arr)
        times_all = times  # last assigned; vars share same months ideally

    if not ds_vars:
        raise RuntimeError("No 2024 climate rasters found in CLIMATE/processed")

    ds = xr.Dataset(ds_vars)
    if times_all is not None:
        ds = ds.assign_coords(time=("time", pd.DatetimeIndex(times_all)))
    if coords_y is not None and coords_x is not None:
        ds = ds.assign_coords(y=("y", coords_y), x=("x", coords_x))

    # attrs
    if "prec" in ds:
        ds["prec"].attrs.update(units="mm", long_name="Monthly precipitation")
    if "tmax" in ds:
        ds["tmax"].attrs.update(units="degC", long_name="Monthly max temperature")
    if "tmin" in ds:
        ds["tmin"].attrs.update(units="degC", long_name="Monthly min temperature")
    ds.attrs.update(title="Climate 2024 (prec, tmax, tmin)", source=str(CLIMATE_OUT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    log_metadata({
        "layer": "climate_2024_nc",
        "path": out_path.as_posix(),
        "type": "netcdf",
        "CRS": "same as rasters",
        "res": "same as rasters",
        "nodata": "see raster layers",
        "units": "mixed",
        "transform": "grid-aligned",
        "source": CLIMATE_OUT.as_posix(),
        "date": datetime.now(timezone.utc).isoformat() + "Z",
        "notes": "Merged monthly 2024 prec/tmax/tmin",
    })
    return out_path


def build_soil_id_nc(out_path: Path | None = None) -> Path:
    """Create a NetCDF with soil unit IDs from HWSD2.bil for the AOI with lat/lon coordinates.
    Variable: soil_id (int32) with dims (lat, lon)
    """
    out_path = out_path or (SOIL_OUT / "soil_id_aoi.nc")
    aoi = load_aoi()

    # Try two common locations
    candidates = [
        SOIL_DIR / "HWSD2_RASTER" / "HWSD2.bil",
        SOIL_DIR / "HWSD2" / "data" / "HWSD2_RASTER" / "HWSD2.bil",
    ]
    src_path = next((p for p in candidates if p.exists()), None)
    if src_path is None:
        raise FileNotFoundError("HWSD2.bil not found in expected locations")

    da = _ensure_dataarray(rxr.open_rasterio(src_path, masked=True)).squeeze()
    # Ensure CRS known
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    # Clip to AOI
    da = clip_to_aoi(da, aoi).squeeze()
    # Reproject to WGS84 to ensure lat/lon axis
    if str(da.rio.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        from rasterio.enums import Resampling
        da = da.rio.reproject("EPSG:4326", resampling=Resampling.nearest)

    # Build DataArray with (lat, lon)
    lat = da.coords.get("y")
    lon = da.coords.get("x")
    soil = xr.DataArray(np.asarray(da.values, dtype=np.int32), dims=("lat", "lon"))
    if lat is not None and lon is not None:
        soil = soil.assign_coords(lat=lat.values, lon=lon.values)

    ds = xr.Dataset({"soil_id": soil})
    ds["soil_id"].attrs.update(long_name="HWSD2 SMU ID")
    ds.attrs.update(title="HWSD2 SMU IDs (AOI)", source=src_path.as_posix(), crs="EPSG:4326")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    log_metadata({
        "layer": "soil_id_aoi_nc",
        "path": out_path.as_posix(),
        "type": "netcdf",
        "CRS": "EPSG:4326",
        "res": "native",
        "nodata": da.rio.nodata,
        "units": "SMU code",
        "transform": str(da.rio.transform()),
        "source": src_path.as_posix(),
        "date": datetime.now(timezone.utc).isoformat() + "Z",
        "notes": "Clipped to AOI with lat/lon coords",
    })
    return out_path


def build_soil_id_with_attributes_nc(out_path: Path | None = None, csv_path: Path | None = None) -> Path:
    """Extended soil NetCDF including attribute table.
    Produces a dataset with:
      - soil_id(lat, lon): raster grid of HWSD2_SMU_IDs clipped to AOI
      - One-dimensional attribute variables keyed by soil_id dimension, sourced from the CSV
    The CSV is expected at SOIL/processed/HWSD2_LAYERS_D1.csv unless overridden.
    """
    out_path = out_path or (SOIL_OUT / "soil_id_with_attrs_aoi.nc")
    csv_path = csv_path or (SOIL_OUT / "HWSD2_LAYERS_D1.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Soil attributes CSV not found: {csv_path}")

    # Build base grid using existing function
    grid_path = build_soil_id_nc()  # ensures soil_id_aoi.nc exists
    grid_ds = xr.open_dataset(grid_path)
    # Rename raster variable to avoid name collision with attribute dimension
    grid_ds = grid_ds.rename({"soil_id": "soil_id_grid"})
    soil_grid = grid_ds["soil_id_grid"]

    # Read attribute table
    attr_df = pd.read_csv(csv_path, low_memory=False)
    # Ensure expected ID column present; prefer HWSD2_SMU_ID as the soil_id key
    key_col = None
    for c in attr_df.columns:
        if c.lower() in {"hwsd2_smu_id", "soil_id", "smu_id"}:
            key_col = c
            break
    if key_col is None:
        raise ValueError("Could not find soil ID key column (HWSD2_SMU_ID) in attributes CSV")

    # Restrict attribute table to soil IDs actually present in grid (excluding nodata)
    present_ids = np.unique(soil_grid.values)
    nodata_val = np.iinfo(np.int32).min  # matches -2147483648 earlier
    present_ids = present_ids[present_ids != nodata_val]
    attr_df = attr_df[attr_df[key_col].isin(present_ids)].copy()

    # Rename key column to soil_id and snake_case other columns for consistency
    attr_df = attr_df.rename(columns={key_col: "soil_id"})
    def _sc(s: str) -> str:
        return (
            s.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
            .replace("(", "").replace(")", "").replace("%", "pct").lower()
        )
    attr_df.columns = ["soil_id" if c == "soil_id" else _sc(c) for c in attr_df.columns]

    # Set index to soil_id for alignment
    attr_df = attr_df.set_index("soil_id").sort_index()
    # Convert to xarray (1D variables along soil_id)
    table_ds = xr.Dataset()
    soil_id_coord = xr.DataArray(attr_df.index.values.astype(np.int32), dims=("soil_id",))
    table_ds = table_ds.assign_coords(soil_id=soil_id_coord)

    for col in attr_df.columns:
        series = attr_df[col]
        # Attempt to coerce to numeric where feasible
        if series.dtype == object:
            try:
                series = pd.to_numeric(series, errors="coerce")
            except Exception:
                pass
        data = series.values
        # Skip extremely large text columns to avoid ballooning NetCDF size
        # Use len(data) for compatibility with pandas ExtensionArray values
        if series.dtype == object and len(data) and max(len(str(v)) for v in data[:100]) > 50:
            continue
        arr = xr.DataArray(data, dims=("soil_id",))
        table_ds[col] = arr

    # Combine grid + table via merge retaining separate dimensions
    # Build combined dataset explicitly to avoid merge ambiguity on 'soil_id'
    combined = xr.Dataset({"soil_id_grid": soil_grid})
    combined = combined.assign_coords(lat=soil_grid["lat"], lon=soil_grid["lon"], soil_id=soil_id_coord)
    for var in table_ds.data_vars:
        combined[var] = table_ds[var]
    combined.attrs.update(
        title="HWSD2 Soil IDs and Attributes (AOI)",
        source=f"Raster: {grid_path.as_posix()} | Attributes: {csv_path.as_posix()}",
        crs="EPSG:4326",
        notes="Two-dimensional soil_id_grid(lat, lon) plus one-dimensional attribute variables keyed by soil_id coordinate"
    )

    combined.to_netcdf(out_path)
    log_metadata({
        "layer": "soil_id_with_attrs_aoi_nc",
        "path": out_path.as_posix(),
        "type": "netcdf",
        "CRS": "EPSG:4326",
        "res": "native",
        "nodata": str(nodata_val),
        "units": "mixed",
        "transform": str(soil_grid.rio.transform()) if hasattr(soil_grid, 'rio') else "unknown",
        "source": f"{grid_path.as_posix()} | {csv_path.as_posix()}",
        "date": datetime.now(timezone.utc).isoformat() + "Z",
        "notes": "Added attribute table along soil_id dimension"
    })
    return out_path


def fire_keep_type2(target_type: int = 2) -> None:
    """Filter VIIRS CSVs to keep only rows with a specific type code.

    Args:
        target_type: VIIRS type value to retain (e.g., 2 for vegetation, 0 for unknown).
    Outputs new files named *_type{target_type}.csv alongside originals.
    """
    csvs = sorted(FIRE_DIR.glob("viirs-jpss1_2024_*.csv"))
    if not csvs:
        print("No FIRE CSVs found in", FIRE_DIR)
        return
    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
            # find 'type' column case-insensitively
            type_col = next((c for c in df.columns if c.lower() == "type"), None)
            if not type_col:
                print("⚠️ No 'type' column in", csv.name)
                continue
            df2 = df[df[type_col] == target_type].copy()
            # Outlier capping for FRP if present
            if "frp" in df2.columns:
                frp = pd.to_numeric(df2["frp"], errors="coerce")
                finite = frp[np.isfinite(frp)]
                if finite.size:
                    q1, q3 = np.nanpercentile(finite, [25, 75])
                    iqr = q3 - q1
                    lo = q1 - 1.5 * iqr
                    hi = q3 + 1.5 * iqr
                    df2["frp"] = frp.clip(lo, hi)
            out = csv.with_name(f"{csv.stem}_type{target_type}.csv")
            df2.to_csv(out, index=False)
            log_metadata({
                "layer": out.stem,
                "path": out.as_posix(),
                "type": "table",
                "CRS": "n/a",
                "res": "n/a",
                "nodata": "n/a",
                "units": "n/a",
                "transform": "n/a",
                "source": csv.as_posix(),
                "date": datetime.now(timezone.utc).isoformat() + "Z",
                "notes": f"Filtered to type=={target_type}",
            })
            logger.info("FIRE filtered type%s %s (%d/%d)", target_type, out.name, len(df2), len(df))
        except Exception as e:
            logger.warning("FIRE filtering failed for %s: %s", csv.name, e)


# ----------------------
# Fire-informed spatial cropping & environmental merge
# ----------------------

def _fire_min_lat(csvs: List[Path]) -> Optional[float]:
    vals = []
    for p in csvs:
        try:
            df = pd.read_csv(p, usecols=["latitude"], low_memory=False)
            lat_series = pd.to_numeric(df["latitude"], errors="coerce").dropna()
            lat_list = lat_series.tolist()
            if lat_list:
                vals.append(min(lat_list))
        except Exception:
            pass
    if not vals:
        return None
    return float(min(vals))


def crop_raster_to_fire_min_lat(da: xr.DataArray, min_lat: float) -> xr.DataArray:
    y = da.coords.get("y") or da.coords.get("lat")
    if y is None:
        return da
    y_vals = y.values
    mask = y_vals >= min_lat
    if not mask.any():
        return da
    # Determine ordering
    if y_vals[0] > y_vals[-1]:  # descending
        return da.sel(y=slice(y_vals[0], max(min_lat, y_vals[-1])))
    else:
        return da.sel(y=slice(min_lat, y_vals[-1]))


def fire_crop_all_layers() -> None:
    csvs = sorted(FIRE_DIR.glob("viirs-jpss1_2024_*type2.csv")) or sorted(FIRE_DIR.glob("viirs-jpss1_2024_*.csv"))
    min_lat = _fire_min_lat(csvs)
    if min_lat is None:
        print("⚠️ Could not determine southernmost fire latitude.")
        return
    logger.info("Southernmost fire latitude = %.4f (cropping)", min_lat)
    # Climate NetCDF
    climate_nc = CLIMATE_OUT / "climate_2024.nc"
    if climate_nc.exists():
        ds = xr.open_dataset(climate_nc)
        if "y" in ds.dims:
            y_vals = ds["y"].values
            mask = y_vals >= min_lat
            if mask.any():
                ds_c = ds.sel(y=y_vals[mask])
                ds_c.to_netcdf(CLIMATE_OUT / "climate_2024_cropped.nc")
                logger.info("Cropped climate_2024.nc")
    # Elevation
    elev_path = ELEV_OUT / "elevation_clipped.tif"
    if elev_path.exists():
        dem = _ensure_dataarray(rxr.open_rasterio(elev_path, masked=True)).squeeze()
        dem_c = crop_raster_to_fire_min_lat(dem, min_lat)
        tmp = ELEV_OUT / "elevation_cropped_tmp.tif"
        out = ELEV_OUT / "elevation_cropped.tif"
        _write_temp_gtiff(dem_c, tmp)
        write_cog(tmp, out)
        tmp.unlink(missing_ok=True)
    logger.info("Cropped elevation raster")
    # Landcover
    lc_path = LANDCOVER_OUT / "landcover_raster.tif"
    if lc_path.exists():
        lc = _ensure_dataarray(rxr.open_rasterio(lc_path, masked=True)).squeeze()
        lc_c = crop_raster_to_fire_min_lat(lc, min_lat)
        tmp = LANDCOVER_OUT / "landcover_cropped_tmp.tif"
        out = LANDCOVER_OUT / "landcover_cropped.tif"
        _write_temp_gtiff(lc_c, tmp)
        write_cog(tmp, out)
        tmp.unlink(missing_ok=True)
    logger.info("Cropped landcover raster")
    # Soil ID
    soil_nc = SOIL_OUT / "soil_id_aoi.nc"
    if soil_nc.exists():
        sds = xr.open_dataset(soil_nc)
        if "lat" in sds.dims:
            lat_vals = sds["lat"].values
            mask = lat_vals >= min_lat
            if mask.any():
                sds_c = sds.sel(lat=lat_vals[mask])
                sds_c.to_netcdf(SOIL_OUT / "soil_id_aoi_cropped.nc")
                logger.info("Cropped soil_id_aoi.nc")


def fire_environment_merge(out_path: Path | None = None) -> Path:
    out_path = out_path or (FIRE_DIR / "fire_environment_2024.parquet")
    csvs = sorted(FIRE_DIR.glob("viirs-jpss1_2024_*type2.csv")) or sorted(FIRE_DIR.glob("viirs-jpss1_2024_*.csv"))
    if not csvs:
        raise FileNotFoundError("No FIRE CSVs found for merge")
    frames = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        tcol = next((c for c in df.columns if c.lower() == "type"), None)
        if tcol:
            df = df[df[tcol] == 2].copy()
        if {"latitude", "longitude"}.issubset(df.columns):
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"])
        else:
            continue
        df["country"] = "Algeria" if "Algeria" in p.name else ("Tunisia" if "Tunisia" in p.name else "Unknown")
        if "acq_date" in df.columns:
            df["dt"] = pd.to_datetime(df["acq_date"], errors="coerce")
            df["month"] = df["dt"].dt.strftime("%Y-%m")
        else:
            df["month"] = "unknown"
        frames.append(df)
    fire_df = pd.concat(frames, ignore_index=True)

    tpl = open_template()
    transform = tpl.rio.transform()
    height, width = tpl.rio.height, tpl.rio.width
    cols, rows = ~transform * (fire_df["longitude"].values, fire_df["latitude"].values)
    rows = np.clip(np.floor(rows).astype(int), 0, height - 1)
    cols = np.clip(np.floor(cols).astype(int), 0, width - 1)

    # Load env layers
    climate_nc = CLIMATE_OUT / "climate_2024.nc"
    climate_ds = xr.open_dataset(climate_nc) if climate_nc.exists() else None
    elev_path = ELEV_OUT / "elevation_clipped.tif"
    elev = _ensure_dataarray(rxr.open_rasterio(elev_path, masked=True)).squeeze() if elev_path.exists() else None
    slope = _ensure_dataarray(rxr.open_rasterio(ELEV_OUT / "slope_deg.tif", masked=True)).squeeze() if (ELEV_OUT / "slope_deg.tif").exists() else None
    aspect = _ensure_dataarray(rxr.open_rasterio(ELEV_OUT / "aspect_deg.tif", masked=True)).squeeze() if (ELEV_OUT / "aspect_deg.tif").exists() else None
    lc_path = LANDCOVER_OUT / "landcover_raster.tif"
    landcover = _ensure_dataarray(rxr.open_rasterio(lc_path, masked=True)).squeeze() if lc_path.exists() else None
    # Soil grid and optional attributes
    soil_nc = SOIL_OUT / "soil_id_aoi.nc"
    soil_ds = xr.open_dataset(soil_nc) if soil_nc.exists() else None
    soil_full_nc = SOIL_OUT / "soil_id_with_attrs_aoi.nc"
    soil_full_ds = xr.open_dataset(soil_full_nc) if soil_full_nc.exists() else None

    def sample(da: Optional[xr.DataArray]) -> np.ndarray:
        if da is None:
            return np.full(rows.shape, np.nan)
        return da.values[rows, cols]

    fire_df["elevation"] = sample(elev)
    fire_df["slope_deg"] = sample(slope)
    fire_df["aspect_deg"] = sample(aspect)
    fire_df["landcover_code"] = sample(landcover)
    # Sample soil_id via nearest lat/lon if grid present
    if soil_ds is not None and {"lat","lon"}.issubset(soil_ds.dims) and "soil_id" in soil_ds.data_vars:
        try:
            lat_da = xr.DataArray(fire_df["latitude"], dims=["points"])
            lon_da = xr.DataArray(fire_df["longitude"], dims=["points"])
            sampled = soil_ds["soil_id"].sel(lat=lat_da, lon=lon_da, method="nearest").values
            fire_df["soil_id"] = sampled
        except Exception:
            fire_df["soil_id"] = np.nan
    else:
        fire_df["soil_id"] = np.nan
    # Attach soil attributes if full dataset exists (one-dimensional variables keyed by soil_id)
    if soil_full_ds is not None and "soil_id" in soil_full_ds.data_vars:
        # Build attribute DataFrame
        attr_vars = [v for v in soil_full_ds.data_vars if v != "soil_id_grid" and v != "soil_id"]
        if attr_vars:
            # Some variables may have dimension soil_id only
            attrs = {}
            for v in attr_vars:
                var = soil_full_ds[v]
                if {"soil_id"}.issubset(var.dims):
                    try:
                        attrs[v] = var.to_pandas()
                    except Exception:
                        pass
            if attrs:
                attr_df = pd.DataFrame(attrs)
                # Join by sampled soil_id (if soil_id column numeric)
                if "soil_id" in fire_df.columns and fire_df["soil_id"].notna().any():
                    merge_df = fire_df.merge(attr_df.reset_index(), how="left", left_on="soil_id", right_on="soil_id")
                    fire_df = merge_df
    # Landcover label mapping (optional)
    lc_codebook = LANDCOVER_OUT / "landcover_codebook.json"
    if lc_codebook.exists():
        try:
            import json as _json
            mapping = _json.loads(lc_codebook.read_text())
            if "landcover_code" in fire_df.columns:
                fire_df["landcover_label"] = fire_df["landcover_code"].map(mapping)
        except Exception:
            pass

    if climate_ds is not None:
        month_map = {pd.to_datetime(str(t)).strftime("%Y-%m"): i for i, t in enumerate(climate_ds.time.values)}
        idxs = fire_df["month"].map(month_map).fillna(-1).astype(int).values
        for var in ["prec", "tmax", "tmin"]:
            if var in climate_ds.data_vars:
                data = climate_ds[var].values
                vals = [data[m, r, c] if 0 <= m < data.shape[0] else np.nan for m, r, c in zip(idxs, rows, cols)]
                fire_df[var] = vals

    # Aggregate duplicates
    group_cols = ["latitude", "longitude", "month", "country"]
    num_cols = [c for c in fire_df.columns if pd.api.types.is_numeric_dtype(fire_df[c])]
    agg = {c: "mean" for c in num_cols}
    for c in fire_df.columns:
        if c not in agg and c not in group_cols:
            agg[c] = "first"
    merged = fire_df.groupby(group_cols, as_index=False).agg(agg)
    # FRP stats if FRP column present
    if "frp" in fire_df.columns:
        frp_series = pd.to_numeric(fire_df["frp"], errors="coerce")
        merged["frp_mean"] = fire_df.groupby(group_cols)["frp"].mean().values
        merged["frp_max"] = fire_df.groupby(group_cols)["frp"].max().values

    # Write output (parquet preferred, fallback to CSV)
    try:
        merged.to_parquet(out_path, index=False)
        out_written = out_path
    except Exception:
        out_csv = out_path.with_suffix(".csv")
        merged.to_csv(out_csv, index=False)
        out_written = out_csv

    log_metadata({
        "layer": "fire_environment_2024",
        "path": out_written.as_posix(),
        "type": "table",
        "CRS": "EPSG:4326",
        "res": "point",
        "nodata": "n/a",
        "units": "mixed",
        "transform": "n/a",
        "source": FIRE_DIR.as_posix(),
        "date": datetime.now(timezone.utc).isoformat() + "Z",
        "notes": "Environmental sampling at fire points aggregated by mean",
    })
    logger.info("Fire-environment merge written %s rows=%d", out_written.name, len(merged))
    return out_written


# ----------------------
# Negatives generation and final merged dataset
# ----------------------

def _load_fire_points_type2() -> gpd.GeoDataFrame:
    csvs = sorted(FIRE_DIR.glob("viirs-jpss1_2024_*type2.csv")) or sorted(FIRE_DIR.glob("viirs-jpss1_2024_*.csv"))
    if not csvs:
        raise FileNotFoundError("No FIRE CSVs found")
    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p, low_memory=False)
            # type==2 if present
            tcol = next((c for c in df.columns if c.lower() == "type"), None)
            if tcol in df.columns:
                df = df[df[tcol] == 2].copy()
            # coords
            if {"latitude","longitude"}.issubset(df.columns):
                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                df = df.dropna(subset=["latitude","longitude"]).copy()
            else:
                continue
            frames.append(df[["latitude","longitude"]])
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No valid fire points loaded")
    all_df = pd.concat(frames, ignore_index=True)
    g = gpd.GeoDataFrame(all_df, geometry=gpd.points_from_xy(all_df["longitude"], all_df["latitude"]), crs="EPSG:4326")
    # clip to AOI
    aoi = load_aoi()
    g = g.to_crs(aoi.crs or "EPSG:4326")
    g = g[g.within(aoi.geometry.union_all())].copy()
    return g.reset_index(drop=True)


def generate_non_fire_points(
    out_path: Path | None = None,
    fire_ratio: float = 0.8,
    min_dist_km: float = 3.0,
    max_candidates: int = 500000,
    random_seed: int = 42,
) -> Path:
    """Generate spatially dispersed non-fire points within AOI, away from fire points.
    - fire_ratio: desired fraction of fire in final dataset (0.7..0.9)
    - min_dist_km: minimum distance to any fire point and between negatives
    - Writes Parquet with columns: latitude, longitude, fire=0
    """
    if not (0.5 < fire_ratio < 0.99):
        raise ValueError("fire_ratio should be within (0.5, 0.99)")
    rng = np.random.default_rng(random_seed)
    aoi = load_aoi()
    fires = _load_fire_points_type2()
    n_fire = len(fires)
    if n_fire == 0:
        raise RuntimeError("No fire points found inside AOI")
    # desired negatives count so that fire/(fire+neg)=fire_ratio
    n_neg = int(round(n_fire * (1 - fire_ratio) / fire_ratio))
    n_neg = max(n_neg, 1)

    # Work in metric CRS for buffering and distances
    # Use World Mercator (EPSG:3395) as robust default for km-scale ops
    metric_crs = PJCRS.from_epsg(3395)
    aoi_m = aoi.to_crs(metric_crs)
    fires_m = fires.to_crs(metric_crs)
    aoi_poly = aoi_m.geometry.unary_union

    # Exclusion buffer around fires
    buf = float(min_dist_km * 1000.0)
    try:
        fire_buffer = fires_m.buffer(buf)
        fire_exclusion = unary_union(list(fire_buffer.geometry)) if hasattr(fire_buffer, "geometry") else unary_union(fire_buffer)
    except Exception:
        # Fallback: single union then buffer
        fire_exclusion = unary_union(list(fires_m.geometry)).buffer(buf)

    # Allowed region = AOI minus buffered fires
    allowed = aoi_poly.difference(fire_exclusion)
    if allowed.is_empty:
        # If exclusion too aggressive, reduce buffer by half and retry quickly
        buf *= 0.5
        fire_exclusion = unary_union(list(fires_m.geometry)).buffer(buf)
        allowed = aoi_poly.difference(fire_exclusion)
        if allowed.is_empty:
            allowed = aoi_poly

    # Random rejection sampling with minimum distance enforcement
    # Build spatial index of accepted negatives and existing fires for quick checks
    accepted: List[Tuple[float, float]] = []
    # transformers for quick lon/lat <-> metric
    to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    to_geo = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    min_dist = buf  # already in meters
    # bbox in metric
    minx, miny, maxx, maxy = allowed.bounds

    # Pre-pack fire points as arrays for distance checks
    fire_xy = np.array([[geom.x, geom.y] for geom in fires_m.geometry])

    def too_close_to_fires(xy: np.ndarray) -> bool:
        if fire_xy.size == 0:
            return False
        # cheap prune by bounding box window of ~min_dist
        dx = np.abs(fire_xy[:, 0] - xy[0])
        dy = np.abs(fire_xy[:, 1] - xy[1])
        mask = (dx <= min_dist) & (dy <= min_dist)
        if not mask.any():
            return False
        d2 = (fire_xy[mask][:, 0] - xy[0]) ** 2 + (fire_xy[mask][:, 1] - xy[1]) ** 2
        return np.any(d2 <= (min_dist ** 2))

    accepted_xy: List[Tuple[float, float]] = []
    tries = 0
    target = n_neg
    while len(accepted_xy) < target and tries < max_candidates:
        tries += 1
        rx = rng.uniform(minx, maxx)
        ry = rng.uniform(miny, maxy)
        pt = Point(rx, ry)
        if not allowed.contains(pt):
            continue
        xy = np.array([rx, ry])
        if too_close_to_fires(xy):
            continue
        # Check min distance to previously accepted negatives (local window)
        ok = True
        if accepted_xy:
            arr = np.array(accepted_xy)
            dx = np.abs(arr[:, 0] - rx)
            dy = np.abs(arr[:, 1] - ry)
            mask = (dx <= min_dist) & (dy <= min_dist)
            if mask.any():
                d2 = (arr[mask][:, 0] - rx) ** 2 + (arr[mask][:, 1] - ry) ** 2
                if np.any(d2 <= (min_dist ** 2)):
                    ok = False
        if not ok:
            continue
        accepted_xy.append((rx, ry))

    if len(accepted_xy) < target:
        logger.warning("Generated %d negatives (target %d). Consider reducing min_dist_km.", len(accepted_xy), target)

    # Back to lon/lat
    longs, lats = [], []
    for rx, ry in accepted_xy:
        lon, lat = to_geo.transform(rx, ry)
        longs.append(lon)
        lats.append(lat)
    out_df = pd.DataFrame({"latitude": lats, "longitude": longs, "fire": 0})
    out_path = out_path or (FIRE_DIR / "non_fire_points_2024.parquet")
    try:
        out_df.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".csv")
        out_df.to_csv(out_path, index=False)
    log_metadata({
        "layer": out_path.stem,
        "path": out_path.as_posix(),
        "type": "points",
        "CRS": "EPSG:4326",
        "res": f"min_dist_km={min_dist_km}",
        "nodata": "n/a",
        "units": "n/a",
        "transform": "n/a",
        "source": "random-with-exclusion",
        "date": datetime.now(timezone.utc).isoformat() + "Z",
        "notes": f"negatives={len(out_df)} fire_ratio={fire_ratio}",
    })
    logger.info("Non-fire points written %s (%d)", out_path.name, len(out_df))
    return out_path


def _climate_aggregates_for_points(ds: xr.Dataset, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """Given climate ds (time,y,x) and point indices, compute per-variable aggregates.
    Returns a DataFrame with columns like prec_mean, prec_min, prec_max, prec_std, etc.
    """
    feats = {}
    for var in ["prec", "tmax", "tmin"]:
        if var not in ds:
            continue
        data = ds[var].values  # (time, y, x)
        # sample each time slice
        # shape: (n_points, n_time)
        vals = np.vstack([data[t, rows, cols] for t in range(data.shape[0])]).T
        feats[f"{var}_mean"] = np.nanmean(vals, axis=1)
        feats[f"{var}_min"] = np.nanmin(vals, axis=1)
        feats[f"{var}_max"] = np.nanmax(vals, axis=1)
        feats[f"{var}_std"] = np.nanstd(vals, axis=1)
    return pd.DataFrame(feats)


def build_final_merged_dataset(
    out_path: Path | None = None,
    fire_ratio: float = 0.8,
    min_dist_km: float = 3.0,
    climate_mode: str = "stats",
) -> Path:
    """Create labeled dataset combining fire points (1) and generated non-fire points (0),
    enriched with elevation, slope, aspect, landcover, soil (id+attrs), and aggregated climate features.
    """
    out_path = out_path or (BASE / "processed" / "merged_dataset.parquet")

    # Ensure climate NetCDF exists
    climate_nc = CLIMATE_OUT / "climate_2024.nc"
    if not climate_nc.exists():
        build_climate_nc_2024()
    climate_ds = xr.open_dataset(climate_nc) if climate_nc.exists() else None

    # Fire points
    fires = _load_fire_points_type2().copy()
    fires["fire"] = 1

    # Negatives file or generate
    neg_path = FIRE_DIR / "non_fire_points_2024.parquet"
    if not neg_path.exists():
        generate_non_fire_points(out_path=neg_path, fire_ratio=fire_ratio, min_dist_km=min_dist_km)
    try:
        neg_df = pd.read_parquet(neg_path)
    except Exception:
        neg_df = pd.read_csv(neg_path.with_suffix(".csv"))
    neg_gdf = gpd.GeoDataFrame(neg_df, geometry=gpd.points_from_xy(neg_df["longitude"], neg_df["latitude"]), crs="EPSG:4326")

    # Combine and sample environmental layers via template indexing
    tpl = open_template()
    transform = tpl.rio.transform()

    def _rows_cols(df_like: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cols, rows = ~transform * (df_like["longitude"].values, df_like["latitude"].values)
        rows = np.clip(np.floor(rows).astype(int), 0, tpl.rio.height - 1)
        cols = np.clip(np.floor(cols).astype(int), 0, tpl.rio.width - 1)
        return rows, cols

    # Load static layers
    elev = _ensure_dataarray(rxr.open_rasterio(ELEV_OUT / "elevation_clipped.tif", masked=True)).squeeze() if (ELEV_OUT / "elevation_clipped.tif").exists() else None
    slope = _ensure_dataarray(rxr.open_rasterio(ELEV_OUT / "slope_deg.tif", masked=True)).squeeze() if (ELEV_OUT / "slope_deg.tif").exists() else None
    aspect = _ensure_dataarray(rxr.open_rasterio(ELEV_OUT / "aspect_deg.tif", masked=True)).squeeze() if (ELEV_OUT / "aspect_deg.tif").exists() else None
    lc = _ensure_dataarray(rxr.open_rasterio(LANDCOVER_OUT / "landcover_raster.tif", masked=True)).squeeze() if (LANDCOVER_OUT / "landcover_raster.tif").exists() else None
    soil_ds = xr.open_dataset(SOIL_OUT / "soil_id_aoi.nc") if (SOIL_OUT / "soil_id_aoi.nc").exists() else None
    soil_full_ds = xr.open_dataset(SOIL_OUT / "soil_id_with_attrs_aoi.nc") if (SOIL_OUT / "soil_id_with_attrs_aoi.nc").exists() else None

    def _climate_seasonal_for_points(ds: xr.Dataset, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
        """Compute seasonal means DJF/MAM/JJA/SON for each variable; 4 columns per feature."""
        # Build month index for available times
        times = pd.to_datetime(ds.time.values)
        months = np.array([t.month for t in times]) if times.size else np.array([], dtype=int)
        season_defs = {
            "DJF": {12, 1, 2},
            "MAM": {3, 4, 5},
            "JJA": {6, 7, 8},
            "SON": {9, 10, 11},
        }
        feats: dict[str, np.ndarray] = {}
        for var in ["prec", "tmax", "tmin"]:
            if var not in ds:
                continue
            data = ds[var].values  # (time, y, x)
            # sample each time slice -> (n_points, n_time)
            vals = np.vstack([data[t, rows, cols] for t in range(data.shape[0])]).T if data.size else np.empty((rows.shape[0], 0))
            for s_name, s_months in season_defs.items():
                if months.size:
                    idx = [i for i, m in enumerate(months) if m in s_months]
                else:
                    idx = []
                if idx:
                    feats[f"{var}_{s_name}"] = np.nanmean(vals[:, idx], axis=1)
                else:
                    feats[f"{var}_{s_name}"] = np.full(rows.shape, np.nan)
        return pd.DataFrame(feats)

    def _sample_static(df_like: pd.DataFrame) -> pd.DataFrame:
        rows, cols = _rows_cols(df_like)
        out = {}
        if elev is not None:
            out["elevation"] = elev.values[rows, cols]
        if slope is not None:
            out["slope_deg"] = slope.values[rows, cols]
        if aspect is not None:
            out["aspect_deg"] = aspect.values[rows, cols]
        if lc is not None:
            out["landcover_code"] = lc.values[rows, cols]
        if climate_ds is not None:
            if climate_mode == "seasonal":
                out.update(_climate_seasonal_for_points(climate_ds, rows, cols))
            else:
                out.update(_climate_aggregates_for_points(climate_ds, rows, cols))
        # soil id nearest
        if soil_ds is not None and {"lat","lon"}.issubset(soil_ds.dims) and "soil_id" in soil_ds.data_vars:
            try:
                lat_da = xr.DataArray(df_like["latitude"], dims=["points"])  
                lon_da = xr.DataArray(df_like["longitude"], dims=["points"]) 
                out["soil_id"] = soil_ds["soil_id"].sel(lat=lat_da, lon=lon_da, method="nearest").values
            except Exception:
                out["soil_id"] = np.nan
        return pd.DataFrame(out)

    # Fire features
    fire_df = fires[["latitude","longitude","fire"]].copy()
    fire_feats = _sample_static(fire_df)
    fire_df = pd.concat([fire_df.reset_index(drop=True), fire_feats.reset_index(drop=True)], axis=1)

    # Negatives features
    neg_df2 = neg_gdf[["latitude","longitude"]].copy()
    neg_df2["fire"] = 0
    neg_feats = _sample_static(neg_df2)
    neg_df2 = pd.concat([neg_df2.reset_index(drop=True), neg_feats.reset_index(drop=True)], axis=1)

    merged = pd.concat([fire_df, neg_df2], ignore_index=True)

    # Landcover labels if mapping exists
    codebook = LANDCOVER_OUT / "landcover_codebook.json"
    if codebook.exists() and "landcover_code" in merged.columns:
        try:
            mapping = json.loads(codebook.read_text())
            merged["landcover_label"] = merged["landcover_code"].map(lambda v: mapping.get(str(int(v)), None) if pd.notna(v) else None)
        except Exception:
            pass

    # Attach soil attributes by join on soil_id if available
    if soil_full_ds is not None and "soil_id" in merged.columns and merged["soil_id"].notna().any():
        try:
            attr_vars = [v for v in soil_full_ds.data_vars if v not in ("soil_id_grid", "soil_id")]
            attrs = {}
            for v in attr_vars:
                var = soil_full_ds[v]
                if {"soil_id"}.issubset(var.dims):
                    attrs[v] = var.to_pandas()
            if attrs:
                attr_df = pd.DataFrame(attrs).reset_index()  # includes soil_id
                merged = merged.merge(attr_df, how="left", on="soil_id")
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged.to_parquet(out_path, index=False)
        written = out_path
    except Exception:
        written = out_path.with_suffix(".csv")
        merged.to_csv(written, index=False)

    log_metadata({
        "layer": "merged_fire_nonfire_2024",
        "path": written.as_posix(),
        "type": "table",
        "CRS": "EPSG:4326",
        "res": "point",
        "nodata": "n/a",
        "units": "mixed",
        "transform": "n/a",
        "source": "FIRE + CLIMATE + ELEV + LC + SOIL",
        "date": datetime.now(timezone.utc).isoformat() + "Z",
        "notes": f"fire_ratio={fire_ratio}; climate_mode={climate_mode}",
    })
    logger.info("Final merged dataset written %s rows=%d", written.name, len(merged))
    return written


# ----------------------
# CLI
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Algeria–Tunisia geospatial pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("climate")
    sub.add_parser("elevation")
    sub.add_parser("landcover")
    sub.add_parser("soil")
    sub.add_parser("fire")
    sub.add_parser("all")
    sub.add_parser("climate-nc-2024")
    sub.add_parser("soil-nc")
    sub.add_parser("soil-nc-full")
    ft = sub.add_parser("fire-type2")
    ft.add_argument("--type", type=int, default=0, help="VIIRS type value to retain (default=0)")
    sub.add_parser("fire-crop")
    sub.add_parser("fire-env-merge")
    sub.add_parser("model-eval")
    nf = sub.add_parser("generate-negatives")
    nf.add_argument("--fire-ratio", type=float, default=0.8)
    nf.add_argument("--min-dist-km", type=float, default=3.0)
    fm = sub.add_parser("final-merge")
    fm.add_argument("--fire-ratio", type=float, default=0.8)
    fm.add_argument("--min-dist-km", type=float, default=3.0)
    fm.add_argument("--climate-mode", choices=["stats", "seasonal"], default="stats")

    args = parser.parse_args()

    if args.cmd == "climate":
        process_climate()
    elif args.cmd == "elevation":
        process_elevation()
    elif args.cmd == "landcover":
        process_landcover()
    elif args.cmd == "soil":
        process_soil_clean_csvs()
    elif args.cmd == "fire":
        process_fire_2024()
    elif args.cmd == "all":
        process_climate()
        process_elevation()
        process_landcover()
        process_soil_clean_csvs()
        process_fire_2024()
    elif args.cmd == "climate-nc-2024":
        build_climate_nc_2024()
    elif args.cmd == "soil-nc":
        build_soil_id_nc()
    elif args.cmd == "soil-nc-full":
        build_soil_id_with_attributes_nc()
    elif args.cmd == "fire-type2":
        fire_keep_type2(target_type=args.type)
    elif args.cmd == "fire-crop":
        fire_crop_all_layers()
    elif args.cmd == "fire-env-merge":
        fire_environment_merge()
    elif args.cmd == "model-eval":
        from scripts.model_pipeline import evaluate_models, save_results
        res = evaluate_models()
        p = save_results(res)
        print("Model evaluation saved to", p)
    elif args.cmd == "generate-negatives":
        generate_non_fire_points(fire_ratio=args.fire_ratio, min_dist_km=args.min_dist_km)
    elif args.cmd == "final-merge":
        build_final_merged_dataset(fire_ratio=args.fire_ratio, min_dist_km=args.min_dist_km, climate_mode=args.climate_mode)


if __name__ == "__main__":
    main()
