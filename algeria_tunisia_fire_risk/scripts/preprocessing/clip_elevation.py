#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clip the Elevation dataset (ESRI Grid .adf) to Algeria + Tunisia boundaries.
"""

import os
import geopandas as gpd
import rioxarray as rxr

# === CONFIG ===
BASE_DIR = "/home/swift/Desktop/DATA"
ELEV_DIR = os.path.join(BASE_DIR, "ELEVATION/be15_grd/be15_grd")
OUTPUT_DIR = os.path.join(ELEV_DIR, "processed")
SHAPE_PATH = os.path.join(BASE_DIR, "shapefiles/algeria_tunisia.shp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🌍 Loading study area shapefile...")
study_area = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")

# === Locate .adf raster ===
adf_path = os.path.join(ELEV_DIR, "w001000.adf")
if not os.path.exists(adf_path):
    raise FileNotFoundError(f"❌ Elevation file not found: {adf_path}")

print(f"🗺️ Loading elevation raster from: {adf_path}")
elevation = rxr.open_rasterio(adf_path, masked=True)

# === Clip raster ===
print("✂️ Clipping to Algeria + Tunisia...")
clipped = elevation.rio.clip(study_area.geometry, study_area.crs, drop=True)

# === Save clipped raster ===
output_path = os.path.join(OUTPUT_DIR, "elevation_clipped.tif")
clipped.rio.to_raster(output_path)
print(f"✅ Clipped elevation saved to: {output_path}")
print("🎯 Elevation clipping completed successfully!"   )