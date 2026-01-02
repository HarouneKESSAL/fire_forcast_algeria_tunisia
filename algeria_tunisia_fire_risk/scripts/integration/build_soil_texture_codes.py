#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build categorical rasters (codes) + legends for HWSD2 D1 texture attributes.

Creates:
  SOIL/processed/rasters_d1/soil_texture_usda_d1_codes.tif
  SOIL/processed/rasters_d1/soil_texture_usda_d1_legend.json
  SOIL/processed/rasters_d1/soil_texture_soter_d1_codes.tif
  SOIL/processed/rasters_d1/soil_texture_soter_d1_legend.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import rasterio

# --- CONFIG (adjust if your tree differs) ---
BASE = Path("/home/swift/Desktop/DATA")
D1CSV = BASE / "SOIL" / "processed" / "HWSD2_LAYERS_D1.csv"          # has SMU_ID + TEXTURE_*
SMU_RASTER = BASE / "SOIL" / "HWSD2_RASTER" / "HWSD2.bil"            # integer SMU IDs
AOI_SHP = BASE / "shapefiles" / "algeria_tunisia.shp"
OUT_DIR = BASE / "SOIL" / "processed" / "rasters_d1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# which texture columns to export as categorical rasters
TEXTURE_COLS = {
    "TEXTURE_USDA": "soil_texture_usda_d1",
    "TEXTURE_SOTER": "soil_texture_soter_d1",
}

# tile size must be multiples of 16
BLOCK_SIZE = 256


def open_and_clip_smu():
    """Open the SMU raster (HWSD2.bil) and clip to AOI."""
    aoi = gpd.read_file(AOI_SHP)
    if len(aoi) > 1:
        aoi = aoi.dissolve().reset_index(drop=True)
    aoi = aoi.to_crs("EPSG:4326")

    da = rxr.open_rasterio(SMU_RASTER, masked=True)

    # some HWSD2.bil files miss CRS; force WGS84 if needed
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=True)

    clipped = da.rio.clip(aoi.to_crs(da.rio.crs).geometry.values, aoi.crs, drop=True)
    # squeeze to 2D array
    mu = clipped.squeeze().values.astype("float64")  # SMU IDs as floats
    mu_nodata = ~np.isfinite(mu) | (mu <= 0)
    return clipped, mu, mu_nodata


def load_d1_table():
    """Load D1 CSV and make sure SMU_ID exists and is numeric."""
    df = pd.read_csv(D1CSV, low_memory=False)
    if "SMU_ID" not in df.columns:
        raise SystemExit("❌ SMU_ID column not found in D1 CSV.")
    df["SMU_ID"] = pd.to_numeric(df["SMU_ID"], errors="coerce").astype("Int64")
    # some attributes might be non-numeric; that’s fine for factorizing
    return df


def export_texture(df, mu_clip, mu, mu_nodata, attr_col, out_base):
    """
    Build *_codes.tif + *_legend.json for a categorical attribute (e.g., TEXTURE_USDA).
    Codes: start at 1 (0 reserved for nodata).
    Legend: {code: name}
    """
    if attr_col not in df.columns:
        print(f"⚠️ {attr_col} not in CSV; skipping.")
        return

    # Pick one texture value per SMU_ID (first non-null if duplicates)
    sub = df[["SMU_ID", attr_col]].dropna(subset=["SMU_ID"]).copy()
    sub[attr_col] = sub[attr_col].astype(str)
    # if multiple rows per SMU_ID, keep the first non-null
    sub = sub.dropna(subset=[attr_col]).drop_duplicates(subset=["SMU_ID"], keep="first")

    # Factorize -> integer codes starting at 0, then shift to start at 1
    codes, uniques = pd.factorize(sub[attr_col], sort=True)  # code 0..K-1
    codes = codes.astype("int32") + 1                         # shift: 1..K
    legend = {int(code): str(name) for code, name in zip(codes, uniques)}

    # Build SMU_ID -> code series
    code_map = pd.Series(codes, index=sub["SMU_ID"].astype("int64"))

    # Map SMU IDs in the raster to codes; nodata -> 0
    mu_int = np.nan_to_num(mu, nan=0.0).astype("int64")
    flat = pd.Series(mu_int.ravel())
    mapped_flat = flat.map(code_map).fillna(0).astype("uint16").to_numpy()
    codes_arr = mapped_flat.reshape(mu.shape)
    codes_arr[mu_nodata] = 0  # keep 0 as nodata

    # Prepare a clean GeoTIFF profile based on the clipped raster
    height, width = codes_arr.shape
    transform = rasterio.transform.from_bounds(
        *mu_clip.rio.bounds(), width, height
    )
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "height": height,
        "width": width,
        "crs": mu_clip.rio.crs.to_string(),
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "nodata": 0,
    }

    # Write codes raster
    codes_tif = OUT_DIR / f"{out_base}_codes.tif"
    with rasterio.open(codes_tif, "w", **profile) as dst:
        dst.write(codes_arr, 1)
        # overviews help performance
        try:
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")
        except Exception:
            pass

    # Write legend JSON
    legend_json = OUT_DIR / f"{out_base}_legend.json"
    legend_json.write_text(json.dumps(legend, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote {codes_tif.name}  and  {legend_json.name}  | classes: {len(legend)}")


def main():
    print("📖 Loading D1 table…")
    df = load_d1_table()

    print("🗺️ Opening & clipping SMU raster…")
    mu_clip, mu, mu_nodata = open_and_clip_smu()

    for col, out_base in TEXTURE_COLS.items():
        print(f"🖨️ Exporting: {col}")
        export_texture(df, mu_clip, mu, mu_nodata, col, out_base)

    print(f"\n🎯 Done. Files saved in: {OUT_DIR}\n"
          f"   - *_codes.tif (categorical codes, nodata=0)\n"
          f"   - *_legend.json (code → label)\n")


if __name__ == "__main__":
    main()
