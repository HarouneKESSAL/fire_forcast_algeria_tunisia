#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone exporter: write each land-cover class to its own GeoJSON
without Streamlit or Leafmap (avoids truncation & memory issues).
"""

import os
import hashlib
from pathlib import Path
import geopandas as gpd

# === CONFIG ===
LANDCOVER_PATH = "/home/swift/Desktop/DATA/LANDCOVER/processed/landcover_clipped.shp"
OUTPUT_DIR = Path("/home/swift/Desktop/DATA/static/geojson")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🌍 Loading shapefile…")
gdf = gpd.read_file(LANDCOVER_PATH)
print(f"✅ Loaded {len(gdf):,} polygons")

# === Simplify to make files lighter ===
print("⚙️ Simplifying geometries (this may take a few minutes)…")
gdf["geometry"] = gdf["geometry"].simplify(0.002, preserve_topology=True)

# === Map readable class name ===
def map_lcc_name(code):
    s = str(code)
    if s.startswith("1"): return "Urban / Artificial"
    if s.startswith("2"): return "Cropland"
    if s.startswith("3"): return "Grassland"
    if s.startswith("4"): return "Shrubland"
    if s.startswith("5"): return "Forest"
    if s.startswith("6"): return "Bare / Sparse Vegetation"
    if s.startswith("7"): return "Snow / Ice"
    if s.startswith("8"): return "Water Body"
    if s.startswith("9"): return "Wetlands"
    return "Other / Unknown"

gdf["LCC_NAME"] = gdf["LCCCODE"].apply(map_lcc_name)

# === Safe write function ===
def safe_write_geojson(subset, path):
    tmp = f"{path}.tmp"
    subset.to_file(tmp, driver="GeoJSON")
    os.replace(tmp, path)
    print(f"✅ Wrote {len(subset):,} features → {path}")

# === Export each class ===
for cls in sorted(gdf["LCC_NAME"].unique()):
    subset = gdf[gdf["LCC_NAME"] == cls]
    if subset.empty:
        continue
    print(f"✏️ Exporting {cls} ({len(subset):,} polygons)…")

    h = hashlib.md5(cls.encode()).hexdigest()[:8]
    fname = f"landcover_{h}.geojson"
    fpath = OUTPUT_DIR / fname

    safe_write_geojson(subset, fpath.as_posix())

print("\n🎯 All classes exported successfully!")
print(f"📂 Files saved to: {OUTPUT_DIR}")
