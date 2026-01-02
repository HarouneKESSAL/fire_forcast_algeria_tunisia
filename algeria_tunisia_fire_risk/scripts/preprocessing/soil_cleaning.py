# scripts/soil_cleaning.py
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr

try:
    from scipy.ndimage import median_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------
# Defaults (edit if you like)
# ----------------------------
DEFAULT_SENTINELS = (-9, -7, -9999)

# Valid ranges per variable (file stem). Any missing stem → "default".
# Put your HWSD2 stems here as you learn them.
DEFAULT_VALID_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # chemistry / texture examples (adjust/extend as needed)
    "BSAT": (0, 100),       # base saturation (%)
    "CLYPPT": (0, 100),     # clay %
    "SLTPPT": (0, 100),     # silt %
    "SNDPPT": (0, 100),     # sand %
    "OC": (0, None),        # organic carbon >= 0
    "CN_RATIO": (1, 40),    # typical C/N ratio
    "default": (None, None),
}


@dataclass
class CleanerConfig:
    sentinels: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_SENTINELS)
    valid_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=lambda: DEFAULT_VALID_RANGES.copy()
    )
    winsor_pcts: Optional[Tuple[float, float]] = (1.0, 99.0)  # None to disable
    interp_limit: int = 2                                     # fill tiny gaps (px)
    median_kernel: Optional[int] = 3                          # None/0 to disable
    template_path: Optional[str] = None                       # raster to align to
    # write options
    compress: str = "DEFLATE"
    predictor: int = 2
    overviews: str = "AUTO"

    @staticmethod
    def from_json(path: Optional[str]) -> "CleanerConfig":
        if not path:
            return CleanerConfig()
        cfg = CleanerConfig()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in data.items():
            setattr(cfg, k, v)
        return cfg


# ----------------------------
# Helpers
# ----------------------------
def _valid_range_for(stem: str, valid_ranges: Dict[str, Tuple[Optional[float], Optional[float]]]):
    stem_up = stem.upper()
    for key, rng in valid_ranges.items():
        if key != "default" and key.upper() == stem_up:
            return rng
    return valid_ranges.get("default", (None, None))


def _open_da(path: str, sentinels: Tuple[float, ...]) -> xr.DataArray:
    da = rxr.open_rasterio(path, masked=True).squeeze()
    arr = da.values.astype("float32")
    # Normalize nodata/sentinels
    nd = da.rio.nodata
    if nd is not None:
        arr[arr == float(nd)] = np.nan
    for s in sentinels:
        arr[arr == s] = np.nan
    da = xr.DataArray(arr, dims=da.dims, coords=da.coords, attrs=da.attrs)
    da.rio.write_nodata(np.nan, inplace=True)
    return da


def _clean_continuous(da: xr.DataArray, stem: str, cfg: CleanerConfig) -> xr.DataArray:
    arr = da.astype("float32").values

    # 1) enshrine NaN sentinels (already done in _open_da, but keep idempotent)
    for s in cfg.sentinels:
        arr[arr == s] = np.nan

    # 2) clamp to valid range
    lo, hi = _valid_range_for(stem, cfg.valid_ranges)
    if lo is not None:
        arr[arr < lo] = np.nan
    if hi is not None:
        arr[arr > hi] = np.nan

    # 3) winsorize
    if cfg.winsor_pcts is not None:
        valid = np.isfinite(arr)
        if valid.any():
            p1, p2 = np.nanpercentile(arr, cfg.winsor_pcts)
            arr[valid] = np.clip(arr[valid], p1, p2)

    # 4) de-speckle (optional)
    if cfg.median_kernel and cfg.median_kernel >= 3 and cfg.median_kernel % 2 == 1 and _HAS_SCIPY:
        fill_val = np.nanpercentile(arr, 50) if np.isfinite(arr).any() else 0.0
        tmp = np.where(np.isfinite(arr), arr, fill_val)
        tmp = median_filter(tmp, size=cfg.median_kernel, mode="nearest")
        arr = np.where(np.isfinite(arr), tmp, np.nan)

    # 5) fill tiny gaps
    da2 = xr.DataArray(arr, dims=da.dims, coords=da.coords, attrs=da.attrs)
    for dim in [d for d in da2.dims if d in ("y", "x")]:
        da2 = da2.interpolate_na(dim=dim, method="nearest", limit=cfg.interp_limit)

    da2.rio.write_nodata(np.nan, inplace=True)
    return da2


def _reproject_and_crop(
    da: xr.DataArray,
    aoi_gdf: gpd.GeoDataFrame,
    template_path: Optional[str],
    resampling: str = "bilinear",
) -> xr.DataArray:
    if template_path:
        tmpl = rxr.open_rasterio(template_path, masked=True)
        da = da.rio.reproject_match(tmpl, resampling=resampling)
    # Clip and **shrink** extent to AOI so coverage totals reflect AOI only
    aoi_in = aoi_gdf.to_crs(da.rio.crs)
    da = da.rio.clip(aoi_in.geometry, aoi_in.crs, drop=True)
    return da


def _prepare_for_write(da: xr.DataArray) -> xr.DataArray:
    """Replace NaN with numeric NODATA for reliable GeoTIFF/COG writing."""
    out = da.copy()
    nodata_val = out.rio.nodata
    if nodata_val is None or (isinstance(nodata_val, float) and math.isnan(nodata_val)):
        nodata_val = -9999.0
        out.rio.write_nodata(nodata_val, inplace=True)
    out.values = np.where(np.isfinite(out.values), out.values, nodata_val).astype("float32")
    return out


def _cog_translate(tmp_tif: Path, out_path: Path, compress: str, predictor: int, overviews: str):
    try:
        subprocess.run(
            [
                "gdal_translate", "-of", "COG",
                "-co", f"COMPRESS={compress}",
                "-co", f"PREDICTOR={predictor}",
                "-co", f"OVERVIEWS={overviews}",
                tmp_tif.as_posix(), out_path.as_posix(),
            ],
            check=True,
            capture_output=True,
        )
        tmp_tif.unlink(missing_ok=True)
    except Exception:
        # Fallback: keep the GTiff
        out_path.unlink(missing_ok=True)
        tmp_tif.rename(out_path)


def write_cog(da: xr.DataArray, out_path: str, cfg: CleanerConfig):
    out = _prepare_for_write(da)
    tmp = Path(out_path).with_suffix(".tmp.tif")
    out.rio.to_raster(tmp.as_posix(), compress=cfg.compress, BIGTIFF="IF_SAFER")
    _cog_translate(tmp, Path(out_path), cfg.compress, cfg.predictor, cfg.overviews)


def clean_soil_file(
    src_path: str,
    out_path: str,
    aoi_gdf: gpd.GeoDataFrame,
    cfg: CleanerConfig,
    *,
    is_categorical: bool = False,
):
    """
    Full aggressive cleaning for a single file.

    - continuous: sentinel→NaN, clamp to valid range, winsorize, optional median filter,
                  fill tiny gaps, reproject+crop, write COG
    - categorical: nearest resample, crop, write COG (no value transforms)
    """
    stem = Path(src_path).stem
    da = _open_da(src_path, cfg.sentinels)

    if is_categorical:
        da = _reproject_and_crop(da, aoi_gdf, cfg.template_path, resampling="nearest")
        write_cog(da, out_path, cfg)
        return

    da = _clean_continuous(da, stem=stem, cfg=cfg)
    da = _reproject_and_crop(da, aoi_gdf, cfg.template_path, resampling="bilinear")
    write_cog(da, out_path, cfg)


__all__ = [
    "CleanerConfig",
    "clean_soil_file",
]
