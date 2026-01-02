#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import geopandas as gpd
import rioxarray as rxr

BASE_DIR = Path("/home/swift/Desktop/DATA")
CLIMATE_DIR = BASE_DIR / "CLIMATE"
OUT_DIR = CLIMATE_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_PATH = BASE_DIR / "shapefiles" / "algeria_tunisia.shp"

print("🗺️ Loading Algeria + Tunisia shapefile...")
study = gpd.read_file(SHAPE_PATH)

# Dissolve to a single polygon if not already
if len(study) > 1:
    study = study.dissolve().reset_index(drop=True)

# Fix potential topology issues and ensure a valid geometry column
study = study.set_geometry(study.geometry.buffer(0))
if study.crs is None:
    study = study.set_crs("EPSG:4326")  # fallback if missing

# Collect all .tif files (skip processed)
tif_files = []
for root, _, files in os.walk(CLIMATE_DIR):
    if str(OUT_DIR) in root:
        continue
    for f in files:
        if f.lower().endswith(".tif"):
            tif_files.append(Path(root) / f)

if not tif_files:
    print("⚠️ No .tif files found in CLIMATE directory.")
    raise SystemExit

print(f"🌦️ Found {len(tif_files)} raster files to clip.\n")

for tif_path in tif_files:
    out_path = OUT_DIR / (tif_path.stem + "_clipped.tif")
    print(f"🧩 Clipping: {tif_path.name}")

    try:
        da = rxr.open_rasterio(tif_path, masked=True)

        # Ensure raster has a CRS
        if da.rio.crs is None:
            # WorldClim/GMTED are typically EPSG:4326; set if missing
            da = da.rio.write_crs("EPSG:4326", inplace=True)

        # Reproject study boundary to raster CRS (robust)
        study_in_raster_crs = study.to_crs(da.rio.crs)

        # Clip (set all_touched as you prefer: False (default) or True)
        clipped = da.rio.clip(
            study_in_raster_crs.geometry.values,
            study_in_raster_crs.crs,
            drop=True,
            all_touched=False,
        )

        # Save compressed GeoTIFF
        clipped.rio.to_raster(
            out_path,
            compress="LZW",
            tiled=True,
            dtype=str(clipped.dtype),
        )
        print(f"✅ Saved → {out_path}\n")

    except Exception as e:
        print(f"❌ Error clipping {tif_path.name}: {e}\n")

print("🎯 All climate rasters clipped successfully!")
