import os
import pandas as pd
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS

# Paths
CSV_PATH = "/run/media/swift/MISO_EFI/DATA/SOIL/processed/HWSD2_LAYERS_D1.csv"
REF_RASTER = "/run/media/swift/MISO_EFI/DATA/SOIL/HWSD2_RASTER/HWSD2.bil"
OUT_DIR = "/run/media/swift/MISO_EFI/DATA/SOIL/processed/from_csv_tifs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Read reference raster for grid and geotransform
with rasterio.open(REF_RASTER) as ref:
    ref_profile = ref.profile.copy()
    ref_data = ref.read(1)
    ref_shape = ref_data.shape
    ref_transform = ref.transform
    ref_crs = ref.crs

# 2. Read CSV
# Assume there is a column like 'MU_GLOBAL' or similar that matches the raster values
csv = pd.read_csv(CSV_PATH)

# Find the mapping column
possible_keys = ["MU_GLOBAL", "mu_global", "GRIDCODE", "gridcode", "ID", "id"]
key_col = None
for k in possible_keys:
    if k in csv.columns:
        key_col = k
        break
if key_col is None:
    raise ValueError("No mapping key column found in CSV.")

# 3. For each attribute column, create a raster
attr_cols = [c for c in csv.columns if c != key_col]

# Build a lookup: mapping unit -> row of attributes
csv_lookup = csv.set_index(key_col)

for attr in attr_cols:
    arr = np.full(ref_shape, np.nan, dtype=np.float32)
    # For each unique mapping unit in the raster, fill with the attribute value
    for mu, val in csv_lookup[attr].items():
        mask = (ref_data == mu)
        arr[mask] = val
    # Save as GeoTIFF
    out_path = os.path.join(OUT_DIR, f"{attr}.tif")
    out_profile = ref_profile.copy()
    out_profile.update({"dtype": "float32", "count": 1, "driver": "GTiff", "nodata": np.nan})
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr, 1)
    print(f"Saved {out_path}")

print("All attribute rasters written to:", OUT_DIR)
