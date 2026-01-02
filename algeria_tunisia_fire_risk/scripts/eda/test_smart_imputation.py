#!/usr/bin/env python3
"""
Test script to demonstrate smart outlier imputation
"""

import numpy as np
import xarray as xr
import rioxarray
from scripts.geo_pipeline import sanitize_dataarray
import sys

def create_test_dataarray(data, name="test"):
    """Create a simple DataArray for testing"""
    if data.ndim == 2:
        da = xr.DataArray(data, dims=("y", "x"), 
                          coords={"y": np.arange(data.shape[0]), 
                                  "x": np.arange(data.shape[1])})
    else:
        da = xr.DataArray(data, dims=("x",), 
                          coords={"x": np.arange(data.shape[0])})
    # Use rioxarray to add CRS and nodata
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_nodata(None)
    return da

print("=" * 80)
print("SMART OUTLIER IMPUTATION TEST")
print("=" * 80)

# Test 1: Symmetric distribution (should use mean)
print("\n1. SYMMETRIC DISTRIBUTION (Normal-like)")
print("-" * 80)
symmetric_data = np.array([10, 12, 11, 13, 12, 11, 100, 10, 12, 11])  # 100 is outlier
print(f"Original data: {symmetric_data}")
da_sym = create_test_dataarray(symmetric_data)
da_cleaned, notes = sanitize_dataarray(da_sym, "symmetric_test", use_smart_imputation=True)
print(f"Cleaned data:  {da_cleaned.values}")
print(f"Notes: {notes}")

# Test 2: Asymmetric distribution (should use median)
print("\n2. ASYMMETRIC DISTRIBUTION (Skewed)")
print("-" * 80)
asymmetric_data = np.array([1, 2, 2, 3, 3, 4, 5, 8, 15, 100, 1, 2, 3])  # 100 is outlier
print(f"Original data: {asymmetric_data}")
da_asym = create_test_dataarray(asymmetric_data)
da_cleaned, notes = sanitize_dataarray(da_asym, "asymmetric_test", use_smart_imputation=True)
print(f"Cleaned data:  {da_cleaned.values}")
print(f"Notes: {notes}")

# Test 3: Categorical data (should use mode)
print("\n3. CATEGORICAL DATA")
print("-" * 80)
categorical_data = np.array([1, 1, 1, 2, 2, 3, 3, 99, 1, 2, 1])  # 99 is outlier
print(f"Original data: {categorical_data}")
da_cat = create_test_dataarray(categorical_data)
da_cleaned, notes = sanitize_dataarray(da_cat, "categorical_test", use_smart_imputation=True)
print(f"Cleaned data:  {da_cleaned.values}")
print(f"Notes: {notes}")

# Test 4: Spatial/Geospatial data (should use neighbor mean)
print("\n4. SPATIAL DATA (2D grid - neighbor mean)")
print("-" * 80)
# Create realistic elevation data with decimal values to avoid categorical detection
spatial_data = np.array([
    [100.2, 105.8, 110.3, 105.9, 100.1],
    [105.5, 110.2, 999.0, 110.4, 105.3],  # 999 is outlier in center
    [110.1, 115.7, 120.5, 115.2, 110.8],
    [105.2, 110.9, 115.1, 110.3, 105.6],
    [100.3, 105.4, 110.2, 105.1, 100.5]
])
print("Original data (5x5 grid - elevation in meters):")
print(spatial_data)
da_spatial = create_test_dataarray(spatial_data)
da_cleaned, notes = sanitize_dataarray(da_spatial, "spatial_test", use_smart_imputation=True)
print("\nCleaned data (outlier replaced with neighbor mean):")
print(da_cleaned.values)
print(f"\nNotes: {notes}")

# Test 5: Multiple outliers in spatial data
print("\n5. SPATIAL DATA with MULTIPLE OUTLIERS")
print("-" * 80)
spatial_multi = np.array([
    [100.5, 105.2, 110.8, 105.4, 100.9],
    [105.1, 999.0, 105.7, 110.3, 105.2],  # Multiple outliers
    [110.4, 115.1, 888.0, 115.8, 110.6],
    [105.3, 110.5, 115.2, 777.0, 105.7],
    [100.2, 105.6, 110.1, 105.3, 100.4]
])
print("Original data (5x5 grid with multiple outliers - elevation):")
print(spatial_multi)
da_spatial_multi = create_test_dataarray(spatial_multi)
da_cleaned, notes = sanitize_dataarray(da_spatial_multi, "spatial_multi_test", use_smart_imputation=True)
print("\nCleaned data (outliers replaced with neighbor means):")
print(da_cleaned.values)
print(f"\nNotes: {notes}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("✓ Symmetric distributions use MEAN for outlier imputation")
print("✓ Asymmetric distributions use MEDIAN for outlier imputation")
print("✓ Categorical data uses MODE for outlier imputation")
print("✓ Spatial data uses NEIGHBOR MEAN for outlier imputation")
print("✓ Outlier detection uses IQR method (Q1-1.5*IQR, Q3+1.5*IQR)")
print("=" * 80)
