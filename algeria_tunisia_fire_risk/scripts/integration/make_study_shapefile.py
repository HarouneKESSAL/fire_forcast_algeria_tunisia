import geopandas as gpd
import pandas as pd
import os

# === Input paths (updated) ===
ALG_PATH = os.path.expanduser("/home/swift/Desktop/DATA/LANDCOVER/geonetwork_landcover_DZA_gc_adg/dza_gc_adg.shp")
TUN_PATH = os.path.expanduser("/home/swift/Desktop/DATA/LANDCOVER/geonetwork_landcover_tun_gc_adg/tun_gc_adg.shp")

# === Output directory ===
OUT_DIR = os.path.expanduser("~/Desktop/DATA/shapefiles")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "algeria_tunisia.shp")

print("📂 Loading Algeria and Tunisia shapefiles...")
alg = gpd.read_file(ALG_PATH)
tun = gpd.read_file(TUN_PATH)

print("🧩 Merging Algeria + Tunisia...")
study_area = gpd.GeoDataFrame(pd.concat([alg, tun], ignore_index=True))

# Dissolve into one polygon for clipping
study_area = study_area.dissolve().reset_index(drop=True)

# Ensure WGS84 projection
study_area = study_area.to_crs("EPSG:4326")

# Save merged shapefile
study_area.to_file(OUT_PATH)
print(f"✅ Saved merged shapefile → {OUT_PATH}")
