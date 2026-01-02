#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clip & merge Land Cover shapefiles (Algeria + Tunisia)
Handles:
- Geometry column naming differences
- Case sensitivity
- Mixed geometry types
- AREA overflow fields
"""

import os
import geopandas as gpd
import pandas as pd

# === CONFIG ===
BASE_DIR = "/home/swift/Desktop/DATA"
LANDCOVER_DIR = os.path.join(BASE_DIR, "LANDCOVER")
OUTPUT_DIR = os.path.join(LANDCOVER_DIR, "processed")
SHAPE_PATH = os.path.join(BASE_DIR, "shapefiles/algeria_tunisia.shp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🌍 Loading study area shapefile...")
study_area = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")


def load_and_prepare(path):
    """Load shapefile safely, preserving geometry."""
    print(f"🗺️ Loading {os.path.basename(path)} ...")
    gdf = gpd.read_file(path)

    # 🔧 Detect and set geometry column properly
    if not hasattr(gdf, "geometry") or gdf.geometry is None:
        geom_candidates = [
            c for c in gdf.columns
            if "geom" in c.lower() or gdf[c].apply(lambda x: hasattr(x, "geom_type") if pd.notnull(x) else False).any()
        ]
        if geom_candidates:
            gdf = gdf.set_geometry(geom_candidates[0])
        else:
            raise ValueError(f"❌ No valid geometry found in {path}")

    # 🔠 Normalize non-geometry columns only
    non_geom_cols = [c for c in gdf.columns if c != gdf.geometry.name]
    gdf.rename(columns={c: c.strip().upper() for c in non_geom_cols}, inplace=True)

    # 🧭 CRS
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        gdf.set_crs("EPSG:4326", inplace=True)

    # 🧱 Keep only polygons
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf


# === LOAD BOTH ===
dza = load_and_prepare(os.path.join(LANDCOVER_DIR, "geonetwork_landcover_DZA_gc_adg", "dza_gc_adg.shp"))
tun = load_and_prepare(os.path.join(LANDCOVER_DIR, "geonetwork_landcover_TUN_gc_adg", "tun_gc_adg.shp"))

# === ALIGN FIELDS ===
geom_name = dza.geometry.name  # Usually 'geometry'
print(f"📐 Active geometry column: {geom_name}")

# Keep only common non-geometry columns
common_cols = [c for c in dza.columns if c in tun.columns and c != geom_name]
dza = dza[common_cols + [geom_name]]
tun = tun[common_cols + [geom_name]]

# Drop any duplicate geometry columns just in case
for gdf in [dza, tun]:
    dup_cols = [c for c in gdf.columns if c.lower() == geom_name.lower() and c != geom_name]
    if dup_cols:
        print(f"🧹 Dropping duplicate geometry columns: {dup_cols}")
        gdf.drop(columns=dup_cols, inplace=True)

# === MERGE ===
print("🧩 Merging layers...")
merged = pd.concat([dza, tun], ignore_index=True)

# Force geometry after merge (to avoid multiple geometry fields)
merged = gpd.GeoDataFrame(merged, geometry=geom_name, crs="EPSG:4326")

# === CLIP ===
print("✂️ Clipping to study area...")
clipped = gpd.clip(merged, study_area)

# 🧼 Clean geometries and filter polygons only
print("🧹 Cleaning geometries...")
clipped["geometry"] = clipped["geometry"].buffer(0)

# Some features become lines during clipping; drop them
non_poly = ~clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
if non_poly.any():
    n = non_poly.sum()
    print(f"⚠️ Dropping {n} non-polygon geometries (lines/broken shapes)")
    clipped = clipped[~non_poly].copy()

# Fix invalid polygons if any
clipped = clipped[clipped.is_valid].copy()

# === DROP BIG FIELDS ===
drop_cols = [c for c in clipped.columns if "AREA" in c.upper()]
if drop_cols:
    print(f"🧽 Dropping large numeric fields: {drop_cols}")
    clipped.drop(columns=drop_cols, inplace=True)

# === SAVE ===
output_path = os.path.join(OUTPUT_DIR, "landcover_clipped.shp")
clipped.to_file(output_path)
print(f"✅ Saved merged & clipped shapefile → {output_path}")
print("🎯 Land Cover clipping and merging completed successfully!")
