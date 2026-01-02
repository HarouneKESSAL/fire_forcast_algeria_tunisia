import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio as rio
from rasterio.plot import show

# ============================================================================
# 1. LINKING COLUMN ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("LINKING COLUMN ANALYSIS")
print("="*60)
print(f"\nAll columns in df_d1: {df_d1.columns.tolist()}")

# Find ID columns
id_cols = [col for col in df_d1.columns if 'ID' in col.upper()]
print(f"\nColumns containing 'ID': {id_cols}")

# Check uniqueness of ID columns
for col in id_cols:
    unique_count = df_d1[col].nunique()
    total_count = len(df_d1)
    print(f"  {col}: {unique_count:,} unique values out of {total_count:,} ({unique_count/total_count*100:.2f}%)")

# Show sample of ID columns with soil properties
if id_cols:
    print(f"\nSample data with ID columns:")
    display_cols = id_cols[:2] + ['COARSE', 'SAND', 'SILT', 'CLAY']
    # Select only columns that exist
    existing_display_cols = [col for col in display_cols if col in df_d1.columns]
    if existing_display_cols:
        print(df_d1[existing_display_cols].head(10))
    else:
        print("No display columns available")

print("="*60)

# ============================================================================
# 2. LOAD RASTER DATA (if HWSD_BIL variable exists)
# ============================================================================
try:
    # Check if HWSD_BIL variable exists
    if 'HWSD_BIL' in locals() or 'HWSD_BIL' in globals():
        with rio.open(str(HWSD_BIL)) as src:
            raster_data = src.read()  # Read all bands
            raster_profile = src.profile
            raster_bounds = src.bounds
            raster_crs = src.crs
            
            print('✓ Raster loaded successfully!')
            print(f'  Number of bands: {raster_data.shape[0]}')
            print(f'  Raster shape: {raster_data.shape[1]} x {raster_data.shape[2]} pixels')
            print(f'  Bounds: {raster_bounds}')
            print(f'  CRS: {raster_crs}')
            print(f'  Memory: {raster_data.nbytes / (1024**2):.2f} MB')
    else:
        print("⚠ HWSD_BIL variable not defined - skipping raster loading")
        raster_data = None
        
except Exception as e:
    print(f'Error loading raster: {e}')
    raster_data = None

# ============================================================================
# 3. DATASET CONTENT SUMMARY
# ============================================================================
print("=" * 60)
print("HWSD2 DATASET SUMMARY")
print("=" * 60)
print(f"\nTabular Data (MDB):")
print(f"  Records loaded: {df_soil.shape[0]:,}")
print(f"  Features: {df_soil.shape[1]}")
print(f"  Physical properties: {sum(1 for col in df_soil.columns if col in ['COARSE', 'SAND', 'SILT', 'CLAY', 'TEXTURE_USDA', 'TEXTURE_SOTER', 'BULK', 'REF_BULK'])} features")
print(f"  Chemical properties: {sum(1 for col in df_soil.columns if col in ['ORG_CARBON', 'PH_WATER', 'TOTAL_N', 'CN_RATIO', 'CEC_SOIL', 'CEC_CLAY', 'CEC_EFF', 'TEB', 'BSAT', 'ALUM_SAT', 'ESP', 'TCARBON_EQ', 'GYPSUM', 'ELEC_COND'])} features")
print(f"  Data layer: D1 (topsoil/surface layer)")

if raster_data is not None:
    print(f"\nSpatial Data (Raster):")
    print(f"  Dimensions: {raster_data.shape[1]} x {raster_data.shape[2]} pixels")
    print(f"  Total pixels: {raster_data.shape[1] * raster_data.shape[2]:,}")
    print(f"  Memory: {raster_data.nbytes / (1024**2):.2f} MB")
else:
    print(f"\nSpatial Data (Raster): Not loaded")

print("\n" + "=" * 60)

# ============================================================================
# 4. QUICK DATA ASSESSMENT
# ============================================================================
print("\nQUICK DATA ASSESSMENT")
print("=" * 60)

print("\n1. Data Types:")
print(df_soil.dtypes)

print("\n2. First 5 rows:")
print(df_soil.head())

print("\n3. Missing Values:")
missing_summary = df_soil.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
if len(missing_cols) > 0:
    print("Columns with missing values:")
    for col, count in missing_cols.items():
        percentage = (count / len(df_soil)) * 100
        print(f"  {col}: {count:,} missing ({percentage:.2f}%)")
else:
    print("  No missing values detected")

print("\n4. Sentinel Values Check:")
sentinels = [-9, -1, -99, -999, -9999]
sentinel_found = False
numeric_cols = df_soil.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    sentinel_counts = {}
    for sentinel in sentinels:
        count = (df_soil[col] == sentinel).sum()
        if count > 0:
            sentinel_counts[sentinel] = count
    
    if sentinel_counts:
        print(f"  {col}:")
        for sentinel, count in sentinel_counts.items():
            percentage = (count / len(df_soil)) * 100
            print(f"    {sentinel}: {count:,} values ({percentage:.2f}%)")
        sentinel_found = True

if not sentinel_found:
    print("  No sentinel values detected in common sentinel list")

print("\n✓ Initial data loading and assessment complete!")
print("=" * 60)

# ============================================================================
# 5. BIVARIATE ANALYSIS - CORRELATION MATRIX
# ============================================================================
print("\n" + "="*60)
print("BIVARIATE ANALYSIS - CORRELATION MATRIX")
print("="*60)

# Prepare data: replace sentinel values with NaN for correlation
df_soil_clean = df_soil.copy()
sentinels_all = [-9, -1, -99, -999, -9999, -99999]

for col in df_soil_clean.select_dtypes(include=[np.number]).columns:
    # Replace sentinel values with NaN
    df_soil_clean[col] = df_soil_clean[col].replace(sentinels_all, np.nan)

numeric_df = df_soil_clean.select_dtypes(include=[np.number])

# Check if we have enough numeric columns
if len(numeric_df.columns) < 2:
    print("⚠ Not enough numeric columns for correlation analysis")
else:
    # Correlation matrix
    print("\n1. Correlation Matrix (Pearson)")
    
    # Calculate correlation, handling NaN values
    corr_matrix = numeric_df.corr(min_periods=10)  # Require at least 10 non-NaN pairs
    
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                          center=0, linewidths=0.5, cbar_kws={'label': 'Correlation'}, 
                          vmin=-1, vmax=1, square=True)
    
    plt.title('Correlation Matrix - Soil Properties (Layer D1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Find top correlated pairs
    print("\n2. Top Correlated Pairs:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val, abs(corr_val)))
    
    if corr_pairs:
        corr_pairs.sort(key=lambda x: x[3], reverse=True)
        
        print("\nTop 10 positively correlated pairs:")
        pos_pairs = [(col1, col2, corr) for col1, col2, corr, _ in corr_pairs if corr > 0.5][:10]
        for idx, (col1, col2, corr_val) in enumerate(pos_pairs, 1):
            print(f"  {idx}. {col1} vs {col2}: {corr_val:.3f}")
        
        print("\nTop 10 negatively correlated pairs:")
        neg_pairs = [(col1, col2, corr) for col1, col2, corr, _ in corr_pairs if corr < -0.5][:10]
        for idx, (col1, col2, corr_val) in enumerate(neg_pairs, 1):
            print(f"  {idx}. {col1} vs {col2}: {corr_val:.3f}")
        
        print("\n3. Most correlated features (absolute):")
        feature_corr_sums = {}
        for col in corr_matrix.columns:
            # Sum absolute correlations for each feature (excluding self-correlation)
            corr_sum = corr_matrix[col].abs().sum() - 1  # Subtract self-correlation (1.0)
            feature_corr_sums[col] = corr_sum
        
        sorted_features = sorted(feature_corr_sums.items(), key=lambda x: x[1], reverse=True)
        print("\nFeatures with highest overall correlation:")
        for idx, (feature, corr_sum) in enumerate(sorted_features[:10], 1):
            avg_corr = corr_sum / (len(corr_matrix.columns) - 1)
            print(f"  {idx}. {feature}: total abs correlation = {corr_sum:.2f}, average = {avg_corr:.3f}")
    else:
        print("⚠ No valid correlation pairs found - check data quality and missing values")
    
    print("\n✓ Correlation analysis complete!")
    print("="*60)

# ============================================================================
# 6. ADDITIONAL ANALYSIS SUGGESTIONS
# ============================================================================
print("\n" + "="*60)
print("RECOMMENDED NEXT STEPS")
print("="*60)

print("\n1. Data Cleaning Tasks:")
print("   - Handle sentinel values (-9, -1, -99, etc.) appropriately")
print("   - Consider imputation for missing values")
print("   - Check for outliers in key soil properties")

print("\n2. Soil-Specific Analyses:")
print("   - Soil texture classification (sand-silt-clay triangle)")
print("   - pH categories (acidic/neutral/alkaline)")
print("   - Organic carbon categories")
print("   - Cation Exchange Capacity (CEC) interpretation")

print("\n3. Spatial Analysis (if raster available):")
print("   - Link tabular data to raster pixels")
print("   - Create maps of key soil properties")
print("   - Spatial interpolation if needed")

print("\n4. Statistical Analysis:")
print("   - Principal Component Analysis (PCA) for dimensionality reduction")
print("   - Cluster analysis to identify soil types")
print("   - Regression analysis for relationships between properties")

print("\n" + "="*60)