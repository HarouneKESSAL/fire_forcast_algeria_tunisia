import pandas as pd
import numpy as np
from pathlib import Path

def analyze_merged_dataset(file_path):
    """Analyze the merged dataset and display comprehensive information"""
    
    print("=" * 80)
    print("MERGED DATASET ANALYSIS")
    print("=" * 80)
    
    # Load the data
    df = pd.read_parquet(file_path)
    
    # Basic information
    print(f"\n📊 BASIC INFORMATION:")
    print(f"File: {file_path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\n📝 DATA TYPES:")
    dtypes_summary = df.dtypes.value_counts()
    for dtype, count in dtypes_summary.items():
        print(f"  {dtype}: {count} columns")
    
    # Fire label analysis
    print(f"\n🔥 FIRE LABEL ANALYSIS:")
    if 'fire' in df.columns:
        fire_counts = df['fire'].value_counts(dropna=False)
        total = len(df)
        print(f"  Total records: {total:,}")
        print(f"  Fire (1): {fire_counts.get(1, 0):,} ({fire_counts.get(1, 0)/total*100:.2f}%)")
        print(f"  Non-Fire (0): {fire_counts.get(0, 0):,} ({fire_counts.get(0, 0)/total*100:.2f}%)")
        print(f"  Missing: {fire_counts.get(pd.NA, 0):,}")
    else:
        print("  ❌ No 'fire' column found!")
    
    # Key columns analysis
    print(f"\n🎯 KEY COLUMNS ANALYSIS:")
    key_columns = ['latitude', 'longitude', 'confidence', 'country', 'elevation', 'soil_id', 'lcc_label']
    for col in key_columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isna().sum()
            print(f"  {col}: {unique_count} unique values, {missing_count} missing")
    
    # Source breakdown
    if 'source' in df.columns:
        print(f"\n📁 SOURCE BREAKDOWN:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} records ({count/total*100:.2f}%)")
    
    # Environmental variables summary
    print(f"\n🌿 ENVIRONMENTAL VARIABLES SUMMARY:")
    env_columns = ['elevation', 'coarse', 'sand', 'silt', 'clay', 'org_carbon']
    for col in env_columns:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f"  {col}: min={valid_data.min():.2f}, max={valid_data.max():.2f}, "
                          f"mean={valid_data.mean():.2f}, missing={df[col].isna().sum()}")
    
    # Spatial extent
    print(f"\n🗺️ SPATIAL EXTENT:")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_valid = df['latitude'].dropna()
        lon_valid = df['longitude'].dropna()
        if len(lat_valid) > 0 and len(lon_valid) > 0:
            print(f"  Latitude: {lat_valid.min():.4f} to {lat_valid.max():.4f}")
            print(f"  Longitude: {lon_valid.min():.4f} to {lon_valid.max():.4f}")
    
    # Column categories
    print(f"\n📋 COLUMN CATEGORIES:")
    column_categories = {
        'Spatial': [col for col in df.columns if 'lat' in col.lower() or 'lon' in col.lower()],
        'Fire Related': [col for col in df.columns if 'fire' in col.lower() or 'frp' in col.lower() or 'bright' in col.lower()],
        'Soil': [col for col in df.columns if 'soil' in col.lower() or 'clay' in col.lower() or 'sand' in col.lower() or 'silt' in col.lower()],
        'Climate': [col for col in df.columns if 'prec' in col.lower() or 'tmax' in col.lower() or 'tmin' in col.lower()],
        'Land Cover': [col for col in df.columns if 'lcc' in col.lower() or 'land' in col.lower()],
        'Topography': [col for col in df.columns if 'elev' in col.lower() or 'slope' in col.lower() or 'aspect' in col.lower()],
    }
    
    for category, columns in column_categories.items():
        if columns:
            print(f"  {category} ({len(columns)}): {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
    
    # Missing values analysis
    print(f"\n❌ MISSING VALUES ANALYSIS:")
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(high_missing) > 0:
        print("  Columns with missing values (top 10):")
        for col, missing_count in high_missing.head(10).items():
            missing_pct = (missing_count / len(df)) * 100
            print(f"    {col}: {missing_count:,} ({missing_pct:.1f}%)")
    else:
        print("  No missing values found!")
    
    # Sample data
    print(f"\n👀 SAMPLE DATA (first 3 rows):")
    print(df.head(3).to_string(max_cols=10))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Path to your merged dataset
    file_path = Path('C:/Users/anis/Desktop/DATA/DATA_CLEANED/processed/merged_dataset.parquet')
    
    if file_path.exists():
        analyze_merged_dataset(file_path)
    else:
        print(f"❌ File not found: {file_path}")
        print("Available files in directory:")
        for f in file_path.parent.glob("*.parquet"):
            print(f"  - {f.name}")