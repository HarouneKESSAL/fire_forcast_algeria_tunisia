#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
⚡ HWSD2 — Fast Raster Generation (AOI Only)
---------------------------------------------
- Clips HWSD2.bil raster to Algeria & Tunisia shapefile once
- Uses the clipped raster as a template
- Builds rasters for all soil layers (D1–D7)
- Much faster and lighter on memory (~50x faster)
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.mask import mask
from pathlib import Path

# === CONFIG ===
BASE = Path("/home/swift/Desktop/DATA")
SOIL_BASE = BASE / "SOIL"
CLEAN_DIR = SOIL_BASE / "processed" / "cleaned"
OUT_BASE = SOIL_BASE / "processed" / "rasters_all_aoi"
OUT_BASE.mkdir(exist_ok=True)

AOI_SHP = BASE / "shapefiles" / "algeria_tunisia.shp"
RASTER_GLOBAL = SOIL_BASE / "HWSD2_RASTER" / "HWSD2.bil"
RASTER_CLIPPED = SOIL_BASE / "HWSD2_RASTER" / "HWSD2_AOI.tif"

ATTRIBUTES = [
    "COARSE", "SAND", "SILT", "CLAY", "BULK", "ORG_CARBON", "PH_WATER",
    "TOTAL_N", "CN_RATIO", "CEC_SOIL", "TEB", "BSAT", "ESP", "GYPSUM"
]

# -----------------------------------------------------------
# STEP 1 — CLIP GLOBAL RASTER TO AOI (if not already done)
# -----------------------------------------------------------
def clip_global_to_aoi():
    if RASTER_CLIPPED.exists():
        print("✅ AOI raster already exists:", RASTER_CLIPPED)
        return RASTER_CLIPPED

    print("✂️ Clipping HWSD2.bil to AOI...")
    aoi = gpd.read_file(AOI_SHP).to_crs("EPSG:4326")

    # --- Fix: dissolve & unify AOI ---
    if len(aoi) > 1:
        aoi = aoi.dissolve().reset_index(drop=True)
    aoi["geometry"] = aoi.buffer(0)

    # ✅ Corrected line:
    geom_list = [aoi.geometry.unary_union]

    da = rxr.open_rasterio(RASTER_GLOBAL, masked=True)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")

    clipped = da.rio.clip(geom_list, aoi.crs, drop=True)
    clipped.rio.to_raster(RASTER_CLIPPED, compress="LZW")

    print("✅ Clipped raster saved to:", RASTER_CLIPPED)
    return RASTER_CLIPPED


# -----------------------------------------------------------
# STEP 2 — MAKE RASTERS FOR EACH LAYER (AOI only)
# -----------------------------------------------------------
def make_rasters_for_layer(layer_csv, raster_template):
    layer_name = Path(layer_csv).stem.replace("HWSD2_LAYERS_", "").replace("_clean", "")
    print(f"\n🧩 Generating AOI rasters for {layer_name} ...")

    out_dir = OUT_BASE / f"rasters_{layer_name.lower()}"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(layer_csv)
    if "HWSD2_SMU_ID" not in df.columns:
        print(f"⚠️ Skipping {layer_name}: no SMU_ID column.")
        return
    smu_to_value = df.set_index("HWSD2_SMU_ID")

    with rasterio.open(raster_template) as src:
        mu_data = src.read(1)
        profile = src.profile

    for attr in ATTRIBUTES:
        if attr not in smu_to_value.columns:
            print(f"⚠️ {attr} not found in {layer_name}, skipping.")
            continue

        arr = np.full(mu_data.shape, np.nan, dtype=np.float32)
        mapping = smu_to_value[attr].to_dict()

        # Vectorized replacement (fast)
        mask_valid = np.isin(mu_data, list(mapping.keys()))
        valid_pixels = mu_data[mask_valid]
        arr[mask_valid] = np.vectorize(mapping.get)(valid_pixels)

        out_path = out_dir / f"{attr}.tif"
        profile.update(dtype="float32", compress="LZW", driver="GTiff")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)

        coverage = np.isfinite(arr).mean() * 100
        print(f"🗺️ {attr}: coverage {coverage:.2f}% → {out_path.name}")

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
def main():
    raster_aoi = clip_global_to_aoi()

    csvs = sorted(CLEAN_DIR.glob("HWSD2_LAYERS_D*_clean.csv"))
    if not csvs:
        print("❌ No cleaned CSVs found. Run clean_hwsd2_all_layers.py first.")
        return

    for csv in csvs:
        make_rasters_for_layer(csv, raster_aoi)

    print("\n✅ All AOI rasters created successfully! 🎉")

if __name__ == "__main__":
    main()
