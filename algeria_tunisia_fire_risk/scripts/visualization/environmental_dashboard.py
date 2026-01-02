#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌍 Unified Environmental Dashboard — Algeria & Tunisia
-----------------------------------------------------
- Before vs After viewer (preview + stats + AOI & total coverage + deltas)
- Cleaning & preprocessing summary (from metadata + heuristics)
- Recent actions / problems from processing log

Datasets:
 - 🧪 Soil
 - 🌦️ Climate
 - 🏔️ Elevation
 - 🪴 Land Cover
 - 🔥 Fire
"""

# ---- Core & plotting ----
import io
import os
import json
import warnings
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal

# Set consistent plotting style
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
warnings.filterwarnings('ignore')

import pydeck as pdk

# ---- Raster utilities ----
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ---- App ----
import streamlit as st
import importlib

# ==========================
# CONFIG — adjust if needed
# ==========================
BASE = "/run/media/swift/MISO_EFI/DATA"
SHAPE_PATH = os.path.join(BASE, "shapefiles/algeria_tunisia.shp")

SOIL_RASTERS_ROOT = os.path.join(BASE, "SOIL/processed")           # processed soil rasters live here
SOIL_ORIG_FALLBACK = os.path.join(BASE, "SOIL/HWSD2_RASTER")       # original soil rasters (fallback)

CLIMATE_DIR = os.path.join(BASE, "CLIMATE")
CLIMATE_PROCESSED = os.path.join(CLIMATE_DIR, "processed")

ELEV_DIR = os.path.join(BASE, "ELEVATION/be15_grd/be15_grd")
ELEV_PROCESSED = os.path.join(ELEV_DIR, "processed")
ELEV_TMP = "/run/media/swift/MISO_EFI/DATA/ELEVATION/be15_grd/processed/elevation_tmp.tif"

LANDCOVER_DIR = os.path.join(BASE, "LANDCOVER")
LANDCOVER_PROC = os.path.join(LANDCOVER_DIR, "processed")

FIRE_DIR = os.path.join(BASE, "FIRE")

META_ROOT = os.path.join(BASE, "processed/_metadata")

# ==========================
# FEATURE NAME MAPPING
# ==========================
# Dictionary to map file patterns to human-readable feature names
FEATURE_NAME_MAP = {
    # Soil features
    'bulk_density': 'Bulk Density',
    'clay': 'Clay Content',
    'sand': 'Sand Content',
    'silt': 'Silt Content',
    'ph': 'pH',
    'organic_carbon': 'Organic Carbon',
    'nitrogen': 'Nitrogen',
    'cec': 'Cation Exchange Capacity',
    'porosity': 'Porosity',
    'water_capacity': 'Water Holding Capacity',
    'soil_depth': 'Soil Depth',
    'texture': 'Soil Texture',
    
    # Climate features
    'precip': 'Precipitation',
    'precipitation': 'Precipitation',
    'temp': 'Temperature',
    'temperature': 'Temperature',
    'tmax': 'Maximum Temperature',
    'tmin': 'Minimum Temperature',
    'tmean': 'Mean Temperature',
    'rh': 'Relative Humidity',
    'humidity': 'Relative Humidity',
    'wind': 'Wind Speed',
    'solar': 'Solar Radiation',
    'evap': 'Evaporation',
    'evapotranspiration': 'Evapotranspiration',
    'pet': 'Potential Evapotranspiration',
    'aridity': 'Aridity Index',
    
    # Elevation features
    'elevation': 'Elevation',
    'dem': 'Digital Elevation Model',
    'slope': 'Slope',
    'aspect': 'Aspect',
    'roughness': 'Surface Roughness',
    
    # Fire features
    'frp': 'Fire Radiative Power',
    'brightness': 'Brightness Temperature',
    'bright_ti4': 'Brightness (I-4 Band)',
    'bright_ti5': 'Brightness (I-5 Band)',
    'confidence': 'Fire Detection Confidence',
    
    # Land cover features
    'landcover': 'Land Cover',
    'lulc': 'Land Use/Land Cover',
    'vegetation': 'Vegetation Cover',
    'forest': 'Forest Cover',
    'urban': 'Urban Area',
    'water': 'Water Bodies',
    'bare': 'Bare Soil',
    'agriculture': 'Agricultural Land',
}

# Unit mapping for different features
FEATURE_UNITS = {
    'Bulk Density': 'g/cm³',
    'Clay Content': '%',
    'Sand Content': '%',
    'Silt Content': '%',
    'pH': 'pH units',
    'Organic Carbon': '%',
    'Nitrogen': '%',
    'Cation Exchange Capacity': 'cmol+/kg',
    'Porosity': '%',
    'Water Holding Capacity': 'mm',
    'Soil Depth': 'cm',
    'Precipitation': 'mm',
    'Temperature': '°C',
    'Maximum Temperature': '°C',
    'Minimum Temperature': '°C',
    'Mean Temperature': '°C',
    'Relative Humidity': '%',
    'Wind Speed': 'm/s',
    'Solar Radiation': 'W/m²',
    'Evaporation': 'mm',
    'Evapotranspiration': 'mm',
    'Potential Evapotranspiration': 'mm',
    'Aridity Index': 'unitless',
    'Elevation': 'm',
    'Slope': '°',
    'Aspect': '°',
    'Surface Roughness': 'm',
    'Fire Radiative Power': 'MW',
    'Brightness Temperature': 'K',
    'Brightness (I-4 Band)': 'K',
    'Brightness (I-5 Band)': 'K',
    'Fire Detection Confidence': '%',
}

def get_feature_name(file_path):
    """
    Extract human-readable feature name from file path.
    Returns (feature_name, unit) tuple.
    """
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Remove common suffixes
    suffixes_to_remove = ['_clipped', '_processed', '_algeria', '_tunisia', '_aoi', 
                         '_masked', '_resampled', '_algeria_tunisia']
    for suffix in suffixes_to_remove:
        base_name = base_name.replace(suffix, '')
    
    # Remove depth indicators like _d1, _d2, etc.
    base_name = re.sub(r'_d\d+', '', base_name)
    
    # Convert to lowercase for matching
    base_lower = base_name.lower()
    
    # Try to match with known feature names
    for pattern, feature_name in FEATURE_NAME_MAP.items():
        if pattern in base_lower:
            # Get unit if available
            unit = FEATURE_UNITS.get(feature_name, '')
            return feature_name, unit
    
    # If no match found, clean up the filename
    # Replace underscores with spaces and capitalize
    feature_name = base_name.replace('_', ' ').title()
    
    # Remove common prefixes
    prefixes_to_remove = ['Soil ', 'Climate ', 'Elevation ', 'Fire ', 'Landcover ']
    for prefix in prefixes_to_remove:
        if feature_name.startswith(prefix):
            feature_name = feature_name[len(prefix):]
    
    # Default unit
    unit = ''
    return feature_name, unit

# ==========================
# PAGE
# ==========================
st.set_page_config(page_title="Algeria–Tunisia Environmental Dashboard", layout="wide")
st.title("Algeria–Tunisia Environmental Data Explorer")
st.caption("Explore Soil, Climate, Elevation, Land Cover, and Fire datasets — with before/after, cleaning summary, and issues.")

# ==========================
# Helpers (AOI, metadata, logs)
# ==========================
@st.cache_data
def load_aoi():
    """Try shared loader; fallback to shapefile."""
    try:
        import sys
        sys.path.append(os.path.join(BASE, "scripts"))
        gp = importlib.import_module("geo_pipeline")
        return gp.load_aoi()
    except Exception:
        return gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")

AOI = load_aoi()

@st.cache_data
def _load_metadata_df() -> pd.DataFrame:
    dd = Path(META_ROOT) / "data_dictionary.csv"
    if dd.exists():
        try:
            return pd.read_csv(dd)
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data
def _load_log_lines() -> list[str]:
    log = Path(META_ROOT) / "log.md"
    if log.exists():
        try:
            return log.read_text(encoding="utf-8").splitlines()
        except Exception:
            pass
    return []

def _ensure_dataarray(raster) -> xr.DataArray:
    """Normalize rioxarray outputs to DataArray for safe previewing."""
    if isinstance(raster, list):
        raster = raster[0]
    if isinstance(raster, xr.Dataset):
        if len(raster.data_vars) == 1:
            raster = next(iter(raster.data_vars.values()))
        else:
            raster = raster.to_array().isel(variable=0)
    if not isinstance(raster, xr.DataArray):
        raise TypeError(f"Unsupported raster type: {type(raster)}")
    return raster

# ==========================
# Enhanced Statistical Analysis Functions
# ==========================
@st.cache_data
def compute_raster_statistics(raster_path: str, _aoi=None) -> dict:
    """Calculate comprehensive descriptive statistics for raster data."""
    try:
        da = _ensure_dataarray(rxr.open_rasterio(raster_path, masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)
        
        # Handle nodata and sentinel values
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        
        # Common sentinel values
        sentinel_values = [-9999, -999, -99, -9, 9999, 999, 99, 9]
        for s in sentinel_values:
            arr = np.where(arr == s, np.nan, arr)
        
        finite_vals = arr[np.isfinite(arr)]
        
        if finite_vals.size == 0:
            return {"error": "No finite values in raster"}
        
        # Calculate comprehensive statistics
        stats_dict = {
            "count": int(finite_vals.size),
            "mean": float(np.nanmean(finite_vals)),
            "std": float(np.nanstd(finite_vals)),
            "min": float(np.nanmin(finite_vals)),
            "max": float(np.nanmax(finite_vals)),
            "median": float(np.nanmedian(finite_vals)),
            "q25": float(np.nanpercentile(finite_vals, 25)),
            "q75": float(np.nanpercentile(finite_vals, 75)),
            "iqr": float(np.nanpercentile(finite_vals, 75) - np.nanpercentile(finite_vals, 25)),
            "coverage_pct": float(np.isfinite(arr).mean() * 100),
            "shape": arr.shape
        }
        
        # Add skewness and kurtosis if possible
        if finite_vals.size > 10:
            try:
                stats_dict["skewness"] = float(stats.skew(finite_vals))
                stats_dict["kurtosis"] = float(stats.kurtosis(finite_vals))
            except:
                stats_dict["skewness"] = None
                stats_dict["kurtosis"] = None
        
        return stats_dict
    except Exception as e:
        return {"error": str(e)}

@st.cache_data
def analyze_spatial_distribution(raster_path: str, _aoi=None):
    """Analyze spatial distribution patterns."""
    try:
        da = _ensure_dataarray(rxr.open_rasterio(raster_path, masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)
        
        # Handle nodata
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        
        sentinel_values = [-9999, -999, -99, -9, 9999, 999, 99, 9]
        for s in sentinel_values:
            arr = np.where(arr == s, np.nan, arr)
        
        finite_vals = arr[np.isfinite(arr)]
        
        if finite_vals.size == 0:
            return {"error": "No finite values"}
        
        results = {}
        
        # Calculate spatial gradients
        if arr.shape[0] > 10 and arr.shape[1] > 10:
            # North-South gradient (difference between top and bottom)
            if arr.shape[0] > 100:
                top_section = arr[:50, :].flatten()
                bottom_section = arr[-50:, :].flatten()
                top_section = top_section[np.isfinite(top_section)]
                bottom_section = bottom_section[np.isfinite(bottom_section)]
                if len(top_section) > 10 and len(bottom_section) > 10:
                    results["north_south_gradient"] = float(np.mean(bottom_section) - np.mean(top_section))
            
            # East-West gradient
            if arr.shape[1] > 100:
                left_section = arr[:, :50].flatten()
                right_section = arr[:, -50:].flatten()
                left_section = left_section[np.isfinite(left_section)]
                right_section = right_section[np.isfinite(right_section)]
                if len(left_section) > 10 and len(right_section) > 10:
                    results["east_west_gradient"] = float(np.mean(right_section) - np.mean(left_section))
        
        # For elevation data, calculate terrain metrics
        feature_name, _ = get_feature_name(raster_path)
        if "elevation" in feature_name.lower() or "dem" in feature_name.lower():
            try:
                # Calculate slope (simplified)
                if arr.shape[0] > 100 and arr.shape[1] > 100:
                    sample_arr = arr[::10, ::10]
                    dx, dy = np.gradient(sample_arr)
                    slope = np.sqrt(dx**2 + dy**2)
                    slope = slope[np.isfinite(slope)]
                    if slope.size > 0:
                        results["mean_slope"] = float(np.nanmean(slope))
                        results["max_slope"] = float(np.nanmax(slope))
            except:
                pass
        
        return results
    except Exception as e:
        return {"error": str(e)}

@st.cache_data
def create_univariate_plots(raster_path: str, feature_name: str = None):
    """Generate univariate analysis plots with proper feature names and 100-step intervals for elevation."""
    try:
        da = _ensure_dataarray(rxr.open_rasterio(raster_path, masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)
        
        # Handle nodata
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        
        sentinel_values = [-9999, -999, -99, -9, 9999, 999, 99, 9]
        for s in sentinel_values:
            arr = np.where(arr == s, np.nan, arr)
        
        finite_vals = arr[np.isfinite(arr)]
        
        if finite_vals.size == 0:
            return None
        
        # Get feature name and unit
        if not feature_name:
            feature_name, unit = get_feature_name(raster_path)
        else:
            _, unit = get_feature_name(raster_path)
        
        # For elevation plots, calculate tick locations with 100-step intervals
        is_elevation = "elevation" in feature_name.lower() or "dem" in feature_name.lower()
        
        if is_elevation:
            min_val = np.nanmin(finite_vals)
            max_val = np.nanmax(finite_vals)
            
            # Ensure min is at least 0
            min_val = max(0, min_val)
            
            # Round down to nearest 100 for min, round up for max
            min_rounded = int(np.floor(min_val / 100.0)) * 100
            max_rounded = int(np.ceil(max_val / 100.0)) * 100
            
            # Ensure we have at least 2 ticks
            if max_rounded <= min_rounded:
                max_rounded = min_rounded + 100
            
            # Create tick locations every 100 units
            tick_locations = np.arange(min_rounded, max_rounded + 100, 100)
            
            # Create bins for histogram (aligned with 100-step intervals)
            num_bins = min(50, len(tick_locations) - 1)
            bins = np.linspace(min_rounded, max_rounded, num_bins + 1)
            
            # Define plot range for all plots
            plot_min = min_rounded
            plot_max = max_rounded
        else:
            # For non-elevation data, use regular bins
            bins = 50
            tick_locations = None
            plot_min = finite_vals.min()
            plot_max = finite_vals.max()
        
        xlabel = f'{feature_name} ({unit})' if unit else feature_name
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram
        axes[0, 0].hist(finite_vals, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{feature_name} Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Set x-axis limits and ticks for elevation plots
        if is_elevation:
            axes[0, 0].set_xlim(plot_min, plot_max)
            if len(tick_locations) <= 20:
                axes[0, 0].set_xticks(tick_locations)
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Box plot
        bp = axes[0, 1].boxplot(finite_vals, vert=False, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        axes[0, 1].set_xlabel(xlabel)
        axes[0, 1].set_title(f'{feature_name} Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Set x-axis limits and ticks for elevation plots
        if is_elevation:
            axes[0, 1].set_xlim(plot_min, plot_max)
            if len(tick_locations) <= 20:
                axes[0, 1].set_xticks(tick_locations)
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. KDE Plot
        axes[1, 0].hist(finite_vals, bins=bins, density=True, alpha=0.5, 
                        color='lightblue', edgecolor='black', label='Histogram')
        try:
            kde = stats.gaussian_kde(finite_vals)
            
            # Use the same range for all elevation plots
            x_range = np.linspace(plot_min, plot_max, 200)
            kde_vals = kde(x_range)
            
            # Only plot KDE within valid range
            valid_mask = (x_range >= finite_vals.min()) & (x_range <= finite_vals.max())
            if np.any(valid_mask):
                axes[1, 0].plot(x_range[valid_mask], kde_vals[valid_mask], 'r-', linewidth=2, label='KDE')
        except Exception:
            pass
        axes[1, 0].set_xlabel(xlabel)
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'{feature_name} Density Estimate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Set x-axis limits and ticks for elevation plots
        if is_elevation:
            axes[1, 0].set_xlim(plot_min, plot_max)
            if len(tick_locations) <= 20:
                axes[1, 0].set_xticks(tick_locations)
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. QQ Plot (normality check)
        stats.probplot(finite_vals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Ordered Values')
        axes[1, 1].set_title(f'{feature_name} Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Univariate Analysis: {feature_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Univariate analysis failed: {e}")
        return None
    
@st.cache_data
def analyze_multiple_rasters(raster_paths: list, feature_names: list = None):
    """Perform bivariate analysis between multiple rasters with feature names and diagnostics."""
    if len(raster_paths) < 2:
        return None
    
    try:
        # Load and prepare data with diagnostics
        data_arrays = []
        all_feature_names = []
        diagnostics = []  # Track data quality
        
        for i, path in enumerate(raster_paths[:5]):  # Limit to 5 for performance
            try:
                da = _ensure_dataarray(rxr.open_rasterio(path, masked=True)).squeeze()
                arr = np.asarray(da.values, dtype=float)
                
                # Handle nodata
                nodata = da.rio.nodata
                if nodata is not None:
                    arr = np.where(arr == float(nodata), np.nan, arr)
                
                sentinel_values = [-9999, -999, -99, -9, 9999, 999, 99, 9]
                for s in sentinel_values:
                    arr = np.where(arr == s, np.nan, arr)
                
                # Flatten and remove NaNs
                flat = arr.flatten()
                finite_mask = np.isfinite(flat)
                finite_count = np.sum(finite_mask)
                
                # Diagnostic info
                diag_info = {
                    'feature': os.path.basename(path),
                    'total_pixels': len(flat),
                    'finite_pixels': finite_count,
                    'finite_percent': (finite_count / len(flat)) * 100 if len(flat) > 0 else 0,
                    'mean': np.nanmean(arr) if finite_count > 0 else None,
                    'std': np.nanstd(arr) if finite_count > 0 else None
                }
                diagnostics.append(diag_info)
                
                if finite_count > 1000:  # Require at least 1000 valid points
                    flat_finite = flat[finite_mask]
                    
                    # Sample for performance if too large
                    if flat_finite.size > 10000:
                        # Use stratified sampling to preserve distribution
                        sample_size = min(10000, flat_finite.size)
                        # Ensure we get a good spread of values
                        if flat_finite.size > 100000:
                            # For very large arrays, use systematic sampling
                            step = flat_finite.size // 10000
                            flat_finite = flat_finite[::step][:10000]
                        else:
                            # Random sampling with fixed seed for reproducibility
                            np.random.seed(42)
                            flat_finite = np.random.choice(flat_finite, sample_size, replace=False)
                    
                    data_arrays.append(flat_finite)
                    
                    # Get feature name
                    if feature_names and i < len(feature_names):
                        feature_name = feature_names[i]
                    else:
                        feature_name, _ = get_feature_name(path)
                    all_feature_names.append(feature_name[:20])  # Truncate for display
                else:
                    st.warning(f"Insufficient data in {os.path.basename(path)}: {finite_count} valid pixels")
                    
            except Exception as e:
                st.warning(f"Could not load {path}: {e}")
        
        if len(data_arrays) < 2:
            st.error("Need at least 2 rasters with sufficient data for correlation analysis")
            return None
        
        # Display diagnostics
        with st.expander("📊 Data Quality Diagnostics"):
            diag_df = pd.DataFrame(diagnostics)
            st.dataframe(diag_df)
            
            # Check for potential issues
            issues = []
            for i, diag in enumerate(diagnostics):
                if diag['finite_percent'] < 10:
                    issues.append(f"{diag['feature']}: Low data coverage ({diag['finite_percent']:.1f}%)")
                if diag['finite_pixels'] < 1000:
                    issues.append(f"{diag['feature']}: Insufficient valid points ({diag['finite_pixels']})")
            
            if issues:
                st.warning("Potential data quality issues detected:")
                for issue in issues:
                    st.write(f"• {issue}")
        
        # Align arrays to same valid positions (spatial alignment)
        # This is CRITICAL for soil data comparison
        st.info("🔍 Aligning raster data spatially...")
        
        # Find common valid positions across all rasters
        common_data = []
        common_names = []
        
        # For soil data, we need to ensure spatial alignment
        # Let's check if rasters have the same dimensions
        dimensions_info = []
        for i, path in enumerate(raster_paths[:min(5, len(raster_paths))]):
            try:
                da = _ensure_dataarray(rxr.open_rasterio(path, masked=True)).squeeze()
                dimensions_info.append({
                    'feature': all_feature_names[i] if i < len(all_feature_names) else f"Raster_{i}",
                    'shape': da.shape,
                    'crs': str(da.rio.crs) if hasattr(da.rio, 'crs') else 'Unknown'
                })
            except:
                pass
        
        if dimensions_info:
            with st.expander("📐 Raster Dimensions"):
                dim_df = pd.DataFrame(dimensions_info)
                st.dataframe(dim_df)
                
                # Check for dimension mismatches
                unique_shapes = set(str(info['shape']) for info in dimensions_info)
                if len(unique_shapes) > 1:
                    st.error("⚠️ Rasters have different dimensions! Spatial alignment may be incorrect.")
                    st.info("Correlations may be misleading due to different raster extents/resolutions.")
        
        # Calculate correlation using only spatially aligned pixels
        # This is more complex but more accurate
        try:
            # Method 1: Simple pixel-by-pixel correlation (if all rasters have same shape)
            min_length = min([len(arr) for arr in data_arrays])
            aligned_arrays = [arr[:min_length] for arr in data_arrays]
            
            # Remove any arrays that are constant (zero variance)
            valid_arrays = []
            valid_names = []
            for i, arr in enumerate(aligned_arrays):
                if np.std(arr) > 0.001:  # Small threshold for near-constant arrays
                    valid_arrays.append(arr)
                    valid_names.append(all_feature_names[i])
                else:
                    st.warning(f"Skipping {all_feature_names[i]}: Near-constant values (std={np.std(arr):.6f})")
            
            if len(valid_arrays) < 2:
                st.error("Need at least 2 rasters with variable data for correlation analysis")
                return None
            
            # Create data matrix
            data_matrix = np.column_stack(valid_arrays)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data_matrix, rowvar=False)
            
            # Check for perfect correlations
            perfect_corrs = []
            for i in range(len(valid_names)):
                for j in range(i+1, len(valid_names)):
                    if abs(corr_matrix[i, j]) > 0.999:
                        perfect_corrs.append((valid_names[i], valid_names[j], corr_matrix[i, j]))
            
            if perfect_corrs:
                st.warning(f"Found {len(perfect_corrs)} potentially spurious perfect correlations:")
                for feat1, feat2, corr in perfect_corrs[:5]:  # Show first 5
                    st.write(f"• {feat1} vs {feat2}: r = {corr:.4f}")
                
                # Suggest checking data
                st.info("""
                **Possible causes of perfect correlations:**
                1. **Identical rasters** (same data file loaded multiple times)
                2. **Constant values** in sampled regions
                3. **All NaN values** aligned in same positions
                4. **Data preprocessing issues** (clipping/resampling created identical outputs)
                
                **Suggestions:**
                - Check if you're selecting the same raster multiple times
                - Verify data preprocessing steps
                - Try different soil depth layers
                """)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. Correlation heatmap
            im = axes[0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0].set_xticks(range(len(valid_names)))
            axes[0].set_yticks(range(len(valid_names)))
            axes[0].set_xticklabels(valid_names, rotation=45, ha='right', fontsize=9)
            axes[0].set_yticklabels(valid_names, fontsize=9)
            axes[0].set_title('Correlation Matrix', fontweight='bold')
            plt.colorbar(im, ax=axes[0], shrink=0.8)
            
            # Add correlation values (skip perfect 1.0s)
            for i in range(len(valid_names)):
                for j in range(len(valid_names)):
                    corr_val = corr_matrix[i, j]
                    # Only display if not exactly 1.0 (diagonal) or very close
                    if i == j:
                        axes[0].text(j, i, '1.00', ha='center', va='center', 
                                   color='black', fontsize=8, fontweight='bold')
                    elif abs(corr_val) < 0.999:
                        axes[0].text(j, i, f'{corr_val:.2f}', 
                                   ha='center', va='center', 
                                   color='white' if abs(corr_val) > 0.5 else 'black',
                                   fontsize=8)
                    else:
                        axes[0].text(j, i, f'{corr_val:.2f}', 
                                   ha='center', va='center', 
                                   color='red', fontsize=8, fontweight='bold')
            
            # 2. Scatter plot for first two variables
            if len(valid_names) >= 2:
                # Get units for labels
                feature1_name, unit1 = get_feature_name(raster_paths[0])
                feature2_name, unit2 = get_feature_name(raster_paths[1])
                
                xlabel = f'{feature1_name} ({unit1})' if unit1 else feature1_name
                ylabel = f'{feature2_name} ({unit2})' if unit2 else feature2_name
                
                # Sample points for scatter plot (too many points slow down rendering)
                if len(data_matrix) > 5000:
                    sample_idx = np.random.choice(len(data_matrix), 5000, replace=False)
                    scatter_data = data_matrix[sample_idx]
                else:
                    scatter_data = data_matrix
                
                scatter = axes[1].scatter(scatter_data[:, 0], scatter_data[:, 1], 
                                         alpha=0.3, s=10, c='blue')
                axes[1].set_xlabel(xlabel, fontsize=10)
                axes[1].set_ylabel(ylabel, fontsize=10)
                axes[1].set_title(f'{feature1_name} vs {feature2_name}', fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                # Add trend line only if correlation is meaningful
                if len(data_matrix[:, 0]) > 10 and len(data_matrix[:, 1]) > 10:
                    try:
                        # Check if we have enough variation
                        if np.std(data_matrix[:, 0]) > 0.001 and np.std(data_matrix[:, 1]) > 0.001:
                            z = np.polyfit(data_matrix[:, 0], data_matrix[:, 1], 1)
                            p = np.poly1d(z)
                            r_squared = np.corrcoef(data_matrix[:, 0], data_matrix[:, 1])[0,1]**2
                            
                            # Only add trend line if R² is meaningful
                            if abs(r_squared) > 0.01:  # More than 1% explained variance
                                axes[1].plot(data_matrix[:, 0], p(data_matrix[:, 0]), 
                                            "r--", alpha=0.8, linewidth=2,
                                            label=f'R² = {r_squared:.3f}')
                                axes[1].legend()
                    except:
                        pass
                
                # Add data point count
                axes[1].text(0.02, 0.98, f'n = {len(data_matrix):,}', 
                           transform=axes[1].transAxes, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.suptitle('Bivariate Analysis - Soil Attributes', fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error in correlation calculation: {e}")
            return None
        
    except Exception as e:
        print(f"Bivariate analysis failed: {e}")
        return None
    


def check_soil_data_quality(selected_dir):
    """Check soil data quality in a directory."""
    tifs = sorted([f for f in os.listdir(selected_dir) if f.endswith(".tif")])
    
    diagnostics = []
    for tif in tifs[:10]:  # Check first 10 files
        path = os.path.join(selected_dir, tif)
        try:
            da = _ensure_dataarray(rxr.open_rasterio(path, masked=True)).squeeze()
            arr = np.asarray(da.values, dtype=float)
            
            # Handle nodata
            nodata = da.rio.nodata
            if nodata is not None:
                arr = np.where(arr == float(nodata), np.nan, arr)
            
            sentinel_values = [-9999, -999, -99, -9, 9999, 999, 99, 9]
            for s in sentinel_values:
                arr = np.where(arr == s, np.nan, arr)
            
            finite_vals = arr[np.isfinite(arr)]
            
            diagnostics.append({
                'file': tif,
                'shape': arr.shape,
                'total_pixels': arr.size,
                'valid_pixels': finite_vals.size,
                'valid_percent': (finite_vals.size / arr.size) * 100,
                'mean': np.mean(finite_vals) if finite_vals.size > 0 else None,
                'std': np.std(finite_vals) if finite_vals.size > 0 else None,
                'min': np.min(finite_vals) if finite_vals.size > 0 else None,
                'max': np.max(finite_vals) if finite_vals.size > 0 else None,
                'constant': np.std(finite_vals) < 0.001 if finite_vals.size > 0 else True
            })
        except Exception as e:
            diagnostics.append({
                'file': tif,
                'error': str(e)
            })
    
    return pd.DataFrame(diagnostics)

@st.cache_data
def temporal_analysis_elevation(elevation_path: str):
    """Perform temporal analysis on elevation (change detection if multiple timestamps)."""
    try:
        da = _ensure_dataarray(rxr.open_rasterio(elevation_path, masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)
        
        # Handle nodata
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        
        sentinel_values = [-9999, -999, -99, -9, 9999, 999, 99, 9]
        for s in sentinel_values:
            arr = np.where(arr == s, np.nan, arr)
        
        finite_vals = arr[np.isfinite(arr)]
        
        if finite_vals.size == 0:
            return None
        
        # Get feature name
        feature_name, unit = get_feature_name(elevation_path)
        ylabel = f'{feature_name} ({unit})' if unit else feature_name
        
        # Calculate min, max and round to nearest 100
        min_val = np.nanmin(finite_vals)
        max_val = np.nanmax(finite_vals)
        
        # Round down to nearest 100 for min, round up for max
        min_rounded = max(0, int(np.floor(min_val / 100.0)) * 100)
        max_rounded = int(np.ceil(max_val / 100.0)) * 100
        
        # Create tick locations every 100 units
        tick_locations = np.arange(min_rounded, max_rounded + 100, 100)
        
        # Create elevation profiles
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. North-South profile (middle column)
        if arr.shape[1] > 10:
            mid_col = arr.shape[1] // 2
            ns_profile = arr[:, mid_col-10:mid_col+10].mean(axis=1)
            ns_profile = ns_profile[np.isfinite(ns_profile)]
            if len(ns_profile) > 0:
                axes[0, 0].plot(range(len(ns_profile)), ns_profile, 'b-', linewidth=2)
                axes[0, 0].fill_between(range(len(ns_profile)), ns_profile, alpha=0.3, color='blue')
                axes[0, 0].set_xlabel('Row Index (North-South)')
                axes[0, 0].set_ylabel(ylabel)
                axes[0, 0].set_title('North-South Profile')
                axes[0, 0].grid(True, alpha=0.3)
                # Set y-axis ticks
                axes[0, 0].set_yticks(tick_locations)
                axes[0, 0].tick_params(axis='y', rotation=0)
        
        # 2. East-West profile (middle row)
        if arr.shape[0] > 10:
            mid_row = arr.shape[0] // 2
            ew_profile = arr[mid_row-10:mid_row+10, :].mean(axis=0)
            ew_profile = ew_profile[np.isfinite(ew_profile)]
            if len(ew_profile) > 0:
                axes[0, 1].plot(range(len(ew_profile)), ew_profile, 'g-', linewidth=2)
                axes[0, 1].fill_between(range(len(ew_profile)), ew_profile, alpha=0.3, color='green')
                axes[0, 1].set_xlabel('Column Index (East-West)')
                axes[0, 1].set_ylabel(ylabel)
                axes[0, 1].set_title('East-West Profile')
                axes[0, 1].grid(True, alpha=0.3)
                # Set y-axis ticks
                axes[0, 1].set_yticks(tick_locations)
                axes[0, 1].tick_params(axis='y', rotation=0)
        
        # 3. Elevation histogram by quadrant
        h, w = arr.shape
        quadrants = [
            ("NW", arr[:h//2, :w//2]),
            ("NE", arr[:h//2, w//2:]),
            ("SW", arr[h//2:, :w//2]),
            ("SE", arr[h//2:, w//2:])
        ]
        
        quad_data = []
        quad_labels = []
        for label, quad_arr in quadrants:
            quad_flat = quad_arr.flatten()
            quad_finite = quad_flat[np.isfinite(quad_flat)]
            if len(quad_finite) > 0:
                quad_data.append(quad_finite)
                quad_labels.append(label)
        
        if quad_data:
            bp = axes[1, 0].boxplot(quad_data, labels=quad_labels, patch_artist=True)
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            axes[1, 0].set_ylabel(ylabel)
            axes[1, 0].set_title('Distribution by Quadrant')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            # Set y-axis ticks
            axes[1, 0].set_yticks(tick_locations)
            axes[1, 0].tick_params(axis='y', rotation=0)
        
        # 4. Cumulative distribution
        axes[1, 1].hist(finite_vals, bins=50, density=True, cumulative=True, 
                       histtype='step', linewidth=2, color='purple')
        axes[1, 1].set_xlabel(ylabel)
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        # Set x-axis ticks
        axes[1, 1].set_xticks(tick_locations)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{feature_name} Spatial Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Elevation temporal analysis failed: {e}")
        return None

@st.cache_data
def fire_correlation_analysis(df_clean: pd.DataFrame):
    """Perform correlation analysis on fire dataset with proper feature names."""
    try:
        # Select numeric columns only
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that are likely IDs or not meaningful for correlation
        exclude_keywords = ['id', 'index', 'code', '_id']
        numeric_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in exclude_keywords)]
        
        if len(numeric_cols) < 2:
            return None
        
        # Create mapping from column names to feature names
        feature_name_map = {}
        for col in numeric_cols:
            # Try to map column name to feature name
            col_lower = col.lower()
            mapped = False
            for pattern, feature_name in FEATURE_NAME_MAP.items():
                if pattern in col_lower:
                    feature_name_map[col] = feature_name
                    mapped = True
                    break
            
            if not mapped:
                # Clean up column name
                feature_name = col.replace('_', ' ').title()
                feature_name_map[col] = feature_name
        
        # Calculate correlation matrix
        corr_matrix = df_clean[numeric_cols].corr()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. Correlation heatmap
        # Use feature names for labels
        feature_names = [feature_name_map[col] for col in numeric_cols]
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, square=True, ax=axes[0], cbar_kws={"shrink": 0.8},
                   annot_kws={"size": 9}, xticklabels=feature_names, yticklabels=feature_names)
        axes[0].set_title('Fire Dataset Correlation Matrix', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # 2. Top correlations
        # Flatten correlation matrix and get top pairs
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_pairs.append({
                    'var1': numeric_cols[i],
                    'var2': numeric_cols[j],
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_pairs = corr_pairs[:10]  # Top 10
        
        if top_pairs:
            # Create bar chart of top correlations
            pairs_labels = [f"{p['feature1'][:15]}\nvs\n{p['feature2'][:15]}" for p in top_pairs]
            corr_values = [p['correlation'] for p in top_pairs]
            colors = ['red' if v > 0 else 'blue' for v in corr_values]
            
            bars = axes[1].barh(range(len(top_pairs)), corr_values, color=colors, alpha=0.7)
            axes[1].set_yticks(range(len(top_pairs)))
            axes[1].set_yticklabels(pairs_labels, fontsize=9)
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].set_title('Top 10 Feature Correlations', fontweight='bold')
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # Add correlation values on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width >= 0:
                    ha = 'left' if width < 0.1 else 'center'
                    x_pos = width + 0.02 if width < 0.1 else width/2
                else:
                    ha = 'right'
                    x_pos = width - 0.02
                
                axes[1].text(x_pos, bar.get_y() + bar.get_height()/2,
                           f'{corr_values[i]:.3f}', 
                           ha=ha, va='center', 
                           color='white' if abs(width) > 0.3 else 'black',
                           fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Fire correlation analysis failed: {e}")
        return None

# ==========================
# Stats / Coverage
# ==========================
SENTINELS = (-9, -7, -9999)

def _raster_stats(path: str, aoi_gdf: gpd.GeoDataFrame | None = None) -> dict:
    """Return min/mean/max/std + coverage (AOI and total) + shape."""
    try:
        da = _ensure_dataarray(rxr.open_rasterio(path, masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)

        # mark nodata
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        # ignore sentinels (common in HWSD & friends)
        for s in SENTINELS:
            arr = np.where(arr == s, np.nan, arr)

        finite_mask = np.isfinite(arr)
        coverage_total = float(finite_mask.mean() * 100) if arr.size else 0.0

        coverage_aoi = None
        if aoi_gdf is not None:
            try:
                aoi_in = aoi_gdf.to_crs(da.rio.crs)
                mask = rasterize(
                    ((geom, 1) for geom in aoi_in.geometry),
                    out_shape=arr.shape,
                    transform=da.rio.transform(),
                    fill=0,
                    dtype="uint8",
                )
                inside = mask == 1
                if inside.any():
                    coverage_aoi = float(np.isfinite(arr[inside]).mean() * 100)
                else:
                    coverage_aoi = 0.0
            except Exception:
                coverage_aoi = None

        stats = {
            "path": path,
            "min": float(np.nanmin(arr)) if np.isfinite(arr).any() else None,
            "mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else None,
            "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else None,
            "std": float(np.nanstd(arr)) if np.isfinite(arr).any() else None,
            "coverage_pct_total": coverage_total,
            "coverage_pct_aoi": coverage_aoi,
            "shape": tuple(arr.shape),
        }
        return stats
    except Exception as e:
        return {"path": path, "error": str(e)}

def _fmt_stat(s: dict | None, key: str):
    if s is None or key not in s or s[key] is None:
        return "—"
    v = s[key]
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

def _cov_text(s: dict | None) -> str:
    """Prefer AOI coverage, and also show total."""
    if s is None:
        return "—"
    aoi = s.get("coverage_pct_aoi")
    tot = s.get("coverage_pct_total")
    if aoi is not None:
        if tot is not None:
            return f"{aoi:.2f}% (AOI) — {tot:.2f}% total"
        return f"{aoi:.2f}% (AOI)"
    return f"{tot:.2f}% total" if tot is not None else "—"

# ==========================
# Visualization
# ==========================

# LCCS Mapping (move this to module level so it can be used in multiple places)
LCCS_NAMES = {
    0: "No Data",
    10: "Cropland, rainfed",
    11: "Herbaceous cover",
    12: "Tree or shrub cover",
    14: "Rainfed Cropland",
    20: "Cropland, irrigated",
    30: "Mosaic cropland",
    40: "Mosaic natural vegetation",
    50: "Tree cover, broadleaved, evergreen",
    60: "Tree cover, broadleaved, deciduous",
    61: "Tree cover, broadleaved, deciduous, closed",
    62: "Tree cover, broadleaved, deciduous, open",
    70: "Tree cover, needleleaved, evergreen",
    80: "Tree cover, needleleaved, deciduous",
    90: "Tree cover, mixed",
    100: "Mosaic tree and shrub",
    110: "Mosaic herbaceous",
    120: "Shrubland",
    121: "Shrubland evergreen",
    122: "Shrubland deciduous",
    130: "Grassland",
    140: "Lichens and mosses",
    150: "Sparse vegetation",
    160: "Tree cover, flooded, fresh",
    170: "Tree cover, flooded, saline",
    180: "Shrub/herbaceous, flooded",
    190: "Urban areas",
    200: "Bare areas",
    201: "Consolidated bare areas",
    202: "Unconsolidated bare areas",
    210: "Water bodies",
    220: "Permanent snow and ice",
    # Specific LCCS Grid Codes identified in your data
    6001: "Consolidated Bare Area",
    6004: "Unconsolidated Bare Area",
    6020: "Salt Flats / Hardpans (Sebkha)",
    11498: "Rainfed Herbaceous Crop",
    20058: "Sparse Vegetation (<15%)",
    21450: "Shrubland (Thicket)",
    21518: "Shrubland (Deciduous)"
}

def _preview_raster(raster_path: str, title: str, categorical: bool = False) -> None:
    """Safe preview with percentile stretch and AOI overlay. Adds legend for categorical data."""
    try:
        da = _ensure_dataarray(rxr.open_rasterio(raster_path, masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        for s in SENTINELS:
            arr = np.where(arr == s, np.nan, arr)

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            st.caption("No finite values to preview.")
            return

        if categorical or str(da.dtype).startswith("int") or str(da.dtype).startswith("uint"):
            vmin = vmax = None
            cm = "tab20"
        else:
            vmin, vmax = np.nanpercentile(finite, [2, 98])
            cm = "viridis"

        # Get feature name for title
        feature_name, unit = get_feature_name(raster_path)
        plot_title = f"{feature_name} ({unit})" if unit else feature_name
        
        if title:
            plot_title = f"{title}: {plot_title}"

        # Use constrained_layout instead of tight_layout for better legend handling
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        # Attempt to get extent from the DataArray for correct georeferencing
        extent = None
        try: 
            b = da.rio.bounds()
            extent = (b[0], b[2], b[1], b[3])
        except Exception:
            pass

        if vmin is None or vmax is None:
            im = ax.imshow(arr, cmap=cm, extent=extent)
        else:
            im = ax.imshow(arr, cmap=cm, vmin=float(vmin), vmax=float(vmax), extent=extent)

        try:
            if AOI is not None:
                if hasattr(da, "rio") and da.rio.crs:
                    aoi_plot = AOI.to_crs(da.rio.crs)
                else:
                    aoi_plot = AOI
                aoi_plot.boundary.plot(ax=ax, color="red", linewidth=0.7)
        except Exception: 
            pass

        ax.set_title(plot_title, fontweight='bold')

        # For categorical data, create a custom legend instead of colorbar
        if categorical:
            # Get unique values present in the data
            unique_vals = np.unique(finite[np.isfinite(finite)]).astype(int)
            
            # FIX: Use matplotlib.colormaps instead of deprecated plt.cm.get_cmap
            cmap = matplotlib.colormaps[cm]
            
            # Normalize values for color mapping
            norm = plt.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())
            
            # Create legend patches
            from matplotlib.patches import Patch
            legend_patches = []
            for val in sorted(unique_vals):
                color = cmap(norm(val))
                label = LCCS_NAMES.get(val, f"Code {val}")
                legend_patches.append(Patch(facecolor=color, label=f"{val}:  {label}"))
            
            # Add legend outside the plot with smaller font for many classes
            if len(legend_patches) <= 10:
                ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
                         fontsize=8, frameon=True, title="Land Cover Classes")
            else:
                # If too many classes, use smaller font and possibly two columns
                ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
                         fontsize=6, frameon=True, title="Classes", ncol=1 if len(legend_patches) <= 20 else 2)
        else:
            cbar = fig.colorbar(im, ax=ax)
            if unit:
                cbar.set_label(unit)

        buf = io.BytesIO()
        # Remove tight_layout call - using constrained_layout=True instead
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        st.image(buf.getvalue(), caption=plot_title, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        feature_name, _ = get_feature_name(raster_path)
        st.caption(f"Preview failed for {feature_name}: {e}")

# ==========================
# Metadata → cleaning badges
# ==========================
def _load_meta_row(proc_path: str | None, layer_hint: str | None = None):
    df = _load_metadata_df()
    if df.empty:
        return None
    if proc_path and "path" in df.columns:
        hit = df[df["path"] == proc_path]
        if not hit.empty:
            return hit.iloc[0]
    if layer_hint and "layer" in df.columns:
        hit = df[df["layer"].astype(str) == str(layer_hint)]
        if not hit.empty:
            return hit.iloc[0]
    return None

def _derive_original_path(proc_path: str, layer_hint: str | None, fallback_folder: str) -> str | None:
    row = _load_meta_row(proc_path, layer_hint)
    if row is not None and "source" in row.index and pd.notna(row["source"]):
        src = str(row["source"])
        if src and os.path.exists(src):
            return src
    stem = os.path.basename(proc_path).replace("_clipped", "")
    cand = os.path.join(fallback_folder, stem)
    if os.path.exists(cand):
        return cand
    return None

def _parse_cleaning_actions(row, filename: str, lyr_type: str | None) -> list[str]:
    actions = []
    for key in ["transform", "transforms", "processing", "actions"]:
        if row is not None and key in row.index and pd.notna(row[key]):
            actions.append(str(row[key]))

    if row is not None:
        for k in ["CRS", "res", "nodata", "units", "type"]:
            if k in row.index and pd.notna(row[k]):
                actions.append(f"{k}: {row[k]}")
        if "notes" in row.index and pd.notna(row["notes"]):
            actions.append(f"notes: {row['notes']}")

    n = filename.lower()
    if "clipped" in n:
        actions.append("Clipped to AOI (drop=True)")
    if lyr_type in ("categorical", "class", "count"):
        actions.append("Resampled: nearest")
    else:
        actions.append("Resampled: bilinear")

    seen, clean = set(), []
    for a in actions:
        if a.strip() and a not in seen:
            clean.append(a)
            seen.add(a)
    return clean

def _show_cleaning_and_issues(layer_name: str, proc_path: str, lyr_type: str | None):
    row = _load_meta_row(proc_path, layer_name)
    actions = _parse_cleaning_actions(row, os.path.basename(proc_path), lyr_type)

    st.subheader("🧹 Cleaning & Preprocessing")
    if actions:
        st.markdown(" ".join([
            f"<span style='padding:4px 8px;border-radius:9999px;background:#eef5ef;border:1px solid #c6e9d2;color:#065f46;margin-right:6px;display:inline-block'>{a}</span>"
            for a in actions
        ]), unsafe_allow_html=True)
    else:
        st.info("No explicit actions found in metadata; showing inferred defaults.")

    # Logs
    lines = _load_log_lines()
    key = layer_name or os.path.basename(proc_path)
    related = [ln for ln in lines if key in ln]
    problems = [ln for ln in related if ("ERROR" in ln or "WARN" in ln)]
    if problems:
        st.subheader("⚠️ Problems Encountered")
        st.code("\n".join(problems[-12:]))
    elif related:
        st.subheader("📒 Recent Actions")
        st.code("\n".join(related[-12:]))

# ==========================
# Before / After block
# ==========================
def _before_after_block(proc_path: str,
                        layer_hint: str,
                        fallback_src_folder: str,
                        categorical: bool = False,
                        force_before_path: str | None = None):
    if not os.path.exists(proc_path):
        st.error(f"Processed file not found: {proc_path}")
        return

    src_path = force_before_path or _derive_original_path(proc_path, layer_hint, fallback_src_folder)

    # Get feature name for display
    feature_name, unit = get_feature_name(proc_path)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Before (original) - {feature_name}**")
        if src_path and os.path.exists(src_path):
            _preview_raster(src_path, title="Original", categorical=categorical)
            s_before = _raster_stats(src_path, AOI)
        else:
            st.info("Original source not found in metadata; showing processed only.")
            s_before = None

    with c2:
        st.markdown(f"**After (processed) - {feature_name}**")
        _preview_raster(proc_path, title="Processed", categorical=categorical)
        s_after = _raster_stats(proc_path, AOI)

    # Stats table
    st.subheader(f"📊 Stats — Before vs After: {feature_name}")
    table = {
        "Metric": ["min", "mean", "max", "std", "coverage", "shape"],
        "Before": [
            _fmt_stat(s_before, "min"),
            _fmt_stat(s_before, "mean"),
            _fmt_stat(s_before, "max"),
            _fmt_stat(s_before, "std"),
            _cov_text(s_before),
            _fmt_stat(s_before, "shape"),
        ],
        "After": [
            _fmt_stat(s_after, "min"),
            _fmt_stat(s_after, "mean"),
            _fmt_stat(s_after, "max"),
            _fmt_stat(s_after, "std"),
            _cov_text(s_after),
            _fmt_stat(s_after, "shape"),
        ],
    }
    st.table(pd.DataFrame(table))

    # Coverage deltas
    if s_before and s_after:
        a_b = s_before.get("coverage_pct_aoi")
        a_a = s_after.get("coverage_pct_aoi")
        t_b = s_before.get("coverage_pct_total")
        t_a = s_after.get("coverage_pct_total")
        if a_b is not None and a_a is not None:
            st.caption(f"Δ Coverage (AOI): **{a_a - a_b:+.2f}%** (After − Before)")
        if t_b is not None and t_a is not None:
            st.caption(f"Δ Coverage (total): **{t_a - t_b:+.2f}%** (After − Before)")

# ==========================
# Sidebar Nav
# ==========================
page = st.sidebar.radio("Choose dataset to explore:", ["Soil", "Climate", "Elevation", "Land Cover", "Fire"])

# ---------------------------------------------------------------
# 1) SOIL
# ---------------------------------------------------------------
if page == "Soil":
    st.header("Soil Attributes — Before/After & Cleaning")

    # detect depth folders automatically (rasters_d1, rasters_d2, …)
    depth_dirs = []
    root = Path(SOIL_RASTERS_ROOT)
    if root.exists():
        for p in root.rglob("rasters_*"):
            if p.is_dir():
                depth_dirs.append(str(p))
    depth_dirs = sorted(depth_dirs)
    if not depth_dirs:
        st.error("No soil raster directories found under SOIL/processed/.")
        st.stop()

    selected_dir = st.selectbox("Select soil raster folder:", depth_dirs, index=0)
    tifs = sorted([f for f in os.listdir(selected_dir) if f.endswith(".tif")])
    if not tifs:
        st.error("No .tif files found in the selected folder.")
        st.stop()

    # Display files with feature names instead of filenames
    file_options = []
    feature_names = []
    for tif in tifs:
        proc_path = os.path.join(selected_dir, tif)
        feature_name, unit = get_feature_name(proc_path)
        display_name = f"{feature_name} ({unit})" if unit else feature_name
        file_options.append(tif)
        feature_names.append(display_name)
    
    # Create mapping for display
    file_display_map = {display: file for file, display in zip(tifs, feature_names)}
    
    selected_display = st.selectbox("Select soil attribute:", feature_names, index=0)
    selected_attr = file_options[feature_names.index(selected_display)]
    
    proc_path = os.path.join(selected_dir, selected_attr)
    feature_name, unit = get_feature_name(proc_path)
    
    st.subheader(f"Analysis: {feature_name} ({unit})" if unit else f"Analysis: {feature_name}")

    # Before/After Block
    _before_after_block(proc_path, feature_name, fallback_src_folder=SOIL_ORIG_FALLBACK, categorical=False)

    # Cleaning Info
    row = _load_meta_row(proc_path, feature_name)
    lyr_type = None
    if row is not None and "type" in row.index and pd.notna(row["type"]):
        lyr_type = str(row["type"]).lower()
    _show_cleaning_and_issues(feature_name, proc_path, lyr_type)
    
    # ============================================
    # SOIL DATA ANALYSIS
    # ============================================
    st.markdown("---")
    st.header(f"📊 {feature_name} Analysis")
    
    # Get all soil rasters for selected depth
    soil_rasters = [os.path.join(selected_dir, f) for f in tifs]
    
    # Create tabs for different analyses
    soil_tab1, soil_tab2, soil_tab3, soil_tab4 = st.tabs([
        "📈 Descriptive Stats", "🗺️ Spatial Distribution", 
        "📊 Univariate Analysis", "🔗 Bivariate Analysis"
    ])
    
    with soil_tab1:
        st.subheader(f"1. Descriptive Statistics: {feature_name}")
        with st.spinner("Calculating statistics..."):
            soil_stats = compute_raster_statistics(proc_path, AOI)
        
        if "error" not in soil_stats:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{soil_stats.get('mean', 'N/A'):.4f} {unit}")
                st.metric("Min", f"{soil_stats.get('min', 'N/A'):.4f} {unit}")
            with col2:
                st.metric("Std Dev", f"{soil_stats.get('std', 'N/A'):.4f} {unit}")
                st.metric("Max", f"{soil_stats.get('max', 'N/A'):.4f} {unit}")
            with col3:
                st.metric("Median", f"{soil_stats.get('median', 'N/A'):.4f} {unit}")
                st.metric("IQR", f"{soil_stats.get('iqr', 'N/A'):.4f} {unit}")
            with col4:
                st.metric("Coverage", f"{soil_stats.get('coverage_pct', 'N/A'):.2f}%")
                st.metric("Data Points", f"{soil_stats.get('count', 'N/A'):,}")
            
            # Detailed statistics table
            st.subheader("Detailed Statistics")
            stats_df = pd.DataFrame.from_dict(soil_stats, orient='index', columns=['Value'])
            # Remove error key if present
            stats_df = stats_df[~stats_df.index.str.contains('error')]
            # Add units to relevant metrics
            unit_metrics = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr']
            for metric in unit_metrics:
                if metric in stats_df.index:
                    stats_df.loc[metric, 'Value'] = f"{float(stats_df.loc[metric, 'Value']):.4f} {unit}"
            st.dataframe(stats_df)
        else:
            st.error(f"Error calculating statistics: {soil_stats.get('error')}")
    
    with soil_tab2:
        st.subheader(f"2. Spatial Distribution: {feature_name}")
        with st.spinner("Analyzing spatial patterns..."):
            spatial_stats = analyze_spatial_distribution(proc_path, AOI)
        
        if spatial_stats and "error" not in spatial_stats:
            # Display spatial metrics
            col1, col2 = st.columns(2)
            
            for i, (key, value) in enumerate(spatial_stats.items()):
                if value is not None and "error" not in str(value):
                    display_key = key.replace("_", " ").title()
                    if i % 2 == 0:
                        with col1:
                            st.metric(
                                display_key, 
                                f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                            )
                    else:
                        with col2:
                            st.metric(
                                display_key, 
                                f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                            )
            
            # Add spatial visualization
            try:
                da = _ensure_dataarray(rxr.open_rasterio(proc_path, masked=True)).squeeze()
                arr = np.asarray(da.values, dtype=float)
                
                # Handle nodata
                nodata = da.rio.nodata
                if nodata is not None:
                    arr = np.where(arr == float(nodata), np.nan, arr)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(arr, cmap='YlOrBr', alpha=0.8)
                ax.set_title(f'Spatial Distribution: {feature_name}')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f'{unit}' if unit else 'Value')
                st.pyplot(fig)
                plt.close(fig)
            except:
                pass
        else:
            st.warning("Could not compute spatial distribution statistics")
    
    with soil_tab3:
        st.subheader(f"3. Univariate Analysis: {feature_name}")
        with st.spinner("Generating univariate plots..."):
            univariate_fig = create_univariate_plots(proc_path, feature_name)
        
        if univariate_fig:
            st.pyplot(univariate_fig)
            plt.close(univariate_fig)
        else:
            st.warning("Could not generate univariate analysis plots")
    
    with soil_tab4:
        st.subheader(f"4. Bivariate Analysis: {feature_name}")
        st.markdown("Compare multiple soil attributes to analyze relationships")
        if st.button("🔍 Check Soil Data Quality"):
            with st.spinner("Analyzing soil data quality..."):
                quality_df = check_soil_data_quality(selected_dir)
                st.dataframe(quality_df)
            
            # Flag potential issues
            issues = []
            for _, row in quality_df.iterrows():
                if 'error' not in row:
                    if row['valid_percent'] < 50:
                        issues.append(f"{row['file']}: Low coverage ({row['valid_percent']:.1f}%)")
                    if row['constant']:
                        issues.append(f"{row['file']}: Constant or near-constant values")
            
            if issues:
                st.warning("Data quality issues found:")
                for issue in issues:
                    st.write(f"• {issue}")
        # Allow user to select multiple soil attributes with feature names
        file_feature_map = {}
        display_options = []
        for tif in tifs:
            file_path = os.path.join(selected_dir, tif)
            feat_name, feat_unit = get_feature_name(file_path)
            display_name = f"{feat_name} ({feat_unit})" if feat_unit else feat_name
            file_feature_map[display_name] = tif
            display_options.append(display_name)
        
        selected_displays = st.multiselect(
            "Select soil attributes for comparison (max 5):",
            display_options,
            default=[selected_display] if len(display_options) > 1 else display_options[:min(2, len(display_options))]
        )
        
        if len(selected_displays) >= 2:
            selected_files = [file_feature_map[disp] for disp in selected_displays]
            selected_paths = [os.path.join(selected_dir, var) for var in selected_files]
            
            # Get feature names for display
            selected_features = []
            for disp in selected_displays:
                # Extract feature name from display (remove unit in parentheses)
                if " (" in disp:
                    feat_name = disp.split(" (")[0]
                else:
                    feat_name = disp
                selected_features.append(feat_name)
            
            with st.spinner("Performing bivariate analysis..."):
                bivariate_fig = analyze_multiple_rasters(selected_paths, selected_features)
            
            if bivariate_fig:
                st.pyplot(bivariate_fig)
                plt.close(bivariate_fig)
                
                # Add interpretation
                st.info("""
                **Interpretation Guide:**
                - **Correlation values** range from -1 (perfect negative) to +1 (perfect positive)
                - **R² value** in scatter plot shows how well the linear model fits the data
                - Strong correlations (|r| > 0.7) suggest related soil properties
                """)
            else:
                st.warning("Could not perform bivariate analysis with selected attributes")
        else:
            st.info("Select at least 2 soil attributes to compare")

# ---------------------------------------------------------------
# 2) CLIMATE
# ---------------------------------------------------------------
elif page == "Climate":
    st.header("Climate — Before/After & Cleaning")

    tif_files = sorted([f for f in os.listdir(CLIMATE_PROCESSED) if f.endswith(".tif")])
    if not tif_files:
        st.warning("No climate rasters found in CLIMATE/processed.")
        st.stop()

    # Display climate files with feature names
    climate_options = []
    climate_features = []
    for tif in tif_files:
        proc_path = os.path.join(CLIMATE_PROCESSED, tif)
        feature_name, unit = get_feature_name(proc_path)
        display_name = f"{feature_name} ({unit})" if unit else feature_name
        climate_options.append(tif)
        climate_features.append(display_name)
    
    selected_display = st.selectbox("Select a climate variable:", climate_features, index=0)
    selected_idx = climate_features.index(selected_display)
    selected_file = climate_options[selected_idx]
    
    proc_path = os.path.join(CLIMATE_PROCESSED, selected_file)
    feature_name, unit = get_feature_name(proc_path)
    
    st.subheader(f"Analysis: {feature_name} ({unit})" if unit else f"Analysis: {feature_name}")

    # Before/After Block
    _before_after_block(proc_path, feature_name, fallback_src_folder=CLIMATE_DIR, categorical=False)

    # Cleaning Info
    row = _load_meta_row(proc_path, feature_name)
    lyr_type = "continuous"
    if row is not None and "type" in row.index and pd.notna(row["type"]):
        lyr_type = str(row["type"]).lower()
    _show_cleaning_and_issues(feature_name, proc_path, lyr_type)
    
    # ============================================
    # CLIMATE DATA ANALYSIS
    # ============================================
    st.markdown("---")
    st.header(f"🌡️ {feature_name} Analysis")
    
    # Get all climate rasters for potential multi-variable analysis
    climate_tifs = [f for f in os.listdir(CLIMATE_PROCESSED) if f.endswith(".tif")]
    
    # Create tabs for different analyses
    climate_tab1, climate_tab2, climate_tab3, climate_tab4 = st.tabs([
        "📈 Descriptive Stats", "🗺️ Spatial Distribution", 
        "📊 Univariate Analysis", "🔗 Bivariate Analysis"
    ])
    
    with climate_tab1:
        st.subheader(f"1. Descriptive Statistics: {feature_name}")
        with st.spinner("Calculating climate statistics..."):
            climate_stats = compute_raster_statistics(proc_path, AOI)
        
        if "error" not in climate_stats:
            # Display metrics in a nice format
            cols = st.columns(4)
            metric_pairs = [
                ("Mean", "mean"), ("Std Dev", "std"),
                ("Min", "min"), ("Max", "max"),
                ("Median", "median"), ("IQR", "iqr"),
                ("Coverage", "coverage_pct"), ("Data Points", "count")
            ]
            
            for i, (label, key) in enumerate(metric_pairs):
                if key in climate_stats and climate_stats[key] is not None:
                    value = climate_stats[key]
                    if key == 'coverage_pct':
                        display_value = f"{value:.2f}%"
                    elif key == 'count':
                        display_value = f"{value:,}"
                    elif isinstance(value, float):
                        display_value = f"{value:.4f} {unit}"
                    else:
                        display_value = str(value)
                    
                    with cols[i % 4]:
                        st.metric(label, display_value)
            
            # Additional statistics
            if 'skewness' in climate_stats and climate_stats['skewness'] is not None:
                st.subheader("Distribution Shape")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skewness", f"{climate_stats['skewness']:.4f}")
                with col2:
                    if 'kurtosis' in climate_stats:
                        st.metric("Kurtosis", f"{climate_stats.get('kurtosis', 'N/A'):.4f}")
        else:
            st.error(f"Error calculating statistics: {climate_stats.get('error')}")
    
    with climate_tab2:
        st.subheader(f"2. Spatial Distribution: {feature_name}")
        with st.spinner("Analyzing spatial patterns..."):
            spatial_climate = analyze_spatial_distribution(proc_path, AOI)
        
        if spatial_climate and "error" not in spatial_climate:
            # Create metrics for spatial analysis
            st.markdown("**Spatial Gradients**")
            col1, col2, col3 = st.columns(3)
            
            gradients_found = False
            if "north_south_gradient" in spatial_climate and spatial_climate["north_south_gradient"] is not None:
                with col1:
                    st.metric("N-S Gradient", f"{spatial_climate['north_south_gradient']:.4f} {unit}/unit")
                    gradients_found = True
            
            if "east_west_gradient" in spatial_climate and spatial_climate["east_west_gradient"] is not None:
                with col2:
                    st.metric("E-W Gradient", f"{spatial_climate['east_west_gradient']:.4f} {unit}/unit")
                    gradients_found = True
            
            # Additional terrain metrics for elevation if present
            if "mean_slope" in spatial_climate:
                with col3:
                    st.metric("Mean Slope", f"{spatial_climate.get('mean_slope', 'N/A'):.4f}°")
                    gradients_found = True
            
            if not gradients_found:
                st.info("No significant spatial gradients detected")
            
            # Create a simple spatial visualization
            try:
                da = _ensure_dataarray(rxr.open_rasterio(proc_path, masked=True)).squeeze()
                arr = np.asarray(da.values, dtype=float)
                
                # Handle nodata
                nodata = da.rio.nodata
                if nodata is not None:
                    arr = np.where(arr == float(nodata), np.nan, arr)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(arr, cmap='RdYlBu_r', alpha=0.9)
                ax.set_title(f'Spatial Distribution: {feature_name}')
                ax.set_xlabel('East-West')
                ax.set_ylabel('North-South')
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f'{unit}' if unit else feature_name)
                st.pyplot(fig)
                plt.close(fig)
            except:
                pass
        else:
            st.warning("Could not compute spatial distribution statistics")
    
    with climate_tab3:
        st.subheader(f"3. Univariate Analysis: {feature_name}")
        with st.spinner("Generating univariate plots..."):
            uni_fig = create_univariate_plots(proc_path, feature_name)
        
        if uni_fig:
            st.pyplot(uni_fig)
            plt.close(uni_fig)
            
            # Interpretation
            st.info("""
            **Univariate Analysis Interpretation:**
            - **Histogram**: Shows frequency distribution of climate values
            - **Box Plot**: Displays median, quartiles, and potential outliers
            - **KDE Plot**: Smooth probability density estimate
            - **Q-Q Plot**: Checks if data follows normal distribution (points should follow the red line)
            """)
        else:
            st.warning("Could not generate univariate analysis plots")
    
    with climate_tab4:
        st.subheader(f"4. Bivariate Analysis: {feature_name}")
        st.markdown("Compare multiple climate variables to analyze relationships")
        
        # Allow user to select multiple climate variables with feature names
        climate_file_map = {}
        climate_display_options = []
        for tif in climate_tifs:
            file_path = os.path.join(CLIMATE_PROCESSED, tif)
            feat_name, feat_unit = get_feature_name(file_path)
            display_name = f"{feat_name} ({feat_unit})" if feat_unit else feat_name
            climate_file_map[display_name] = tif
            climate_display_options.append(display_name)
        
        selected_climate_displays = st.multiselect(
            "Select climate variables for comparison (max 5):",
            climate_display_options,
            default=climate_display_options[:min(3, len(climate_display_options))]
        )
        
        if len(selected_climate_displays) >= 2:
            selected_files = [climate_file_map[disp] for disp in selected_climate_displays]
            climate_paths = [os.path.join(CLIMATE_PROCESSED, f) for f in selected_files]
            
            # Get feature names for display
            selected_climate_features = []
            for disp in selected_climate_displays:
                # Extract feature name from display (remove unit in parentheses)
                if " (" in disp:
                    feat_name = disp.split(" (")[0]
                else:
                    feat_name = disp
                selected_climate_features.append(feat_name)
            
            with st.spinner("Performing bivariate analysis..."):
                biv_fig = analyze_multiple_rasters(climate_paths, selected_climate_features)
            
            if biv_fig:
                st.pyplot(biv_fig)
                plt.close(biv_fig)
                
                # Add climate-specific interpretation
                st.info("""
                **Climate Correlation Insights:**
                - **High positive correlation**: Variables that change together (e.g., temperature and evaporation)
                - **High negative correlation**: Inverse relationships (e.g., temperature and humidity in some regions)
                - **Near-zero correlation**: Independent climate processes
                """)
            else:
                st.warning("Could not perform bivariate analysis with selected variables")
        else:
            st.info("Select at least 2 climate variables to compare")

# ---------------------------------------------------------------
# 3) ELEVATION
# ---------------------------------------------------------------
elif page == "Elevation":
    # Check for the specific file mentioned in your request
    elev_tmp_exists = os.path.exists(ELEV_TMP)
    proc_path = ELEV_TMP if elev_tmp_exists else os.path.join(ELEV_PROCESSED, "elevation_clipped.tif")
    
    if not os.path.exists(proc_path):
        st.error(f"Processed elevation raster not found at {proc_path}")
        st.stop()
    
    # Get feature name
    feature_name, unit = get_feature_name(proc_path)
    
    st.header(f"{feature_name} — Before/After & Cleaning")

    # Before/After Block
    _before_after_block(proc_path, feature_name, fallback_src_folder=ELEV_DIR, categorical=False)
    
    # Cleaning Info
    _show_cleaning_and_issues(feature_name, proc_path, "continuous")
    
    # ============================================
    # ELEVATION DATA ANALYSIS
    # ============================================
    st.markdown("---")
    st.header(f"⛰️ {feature_name} Analysis")
    
    # Create tabs for different analyses
    elev_tab1, elev_tab2, elev_tab3, elev_tab4 = st.tabs([
        "📈 Descriptive Stats", "🗺️ Spatial Distribution", 
        "📊 Univariate Analysis", "📐 Spatial Profiles"
    ])
    
    with elev_tab1:
        st.subheader(f"1. Descriptive Statistics: {feature_name}")
        with st.spinner("Calculating elevation statistics..."):
            elev_stats = compute_raster_statistics(proc_path, AOI)
        
        if "error" not in elev_stats:
            # Display elevation metrics
            st.markdown(f"**{feature_name} Summary**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{elev_stats.get('mean', 'N/A'):.2f} {unit}")
                st.metric("Minimum", f"{elev_stats.get('min', 'N/A'):.2f} {unit}")
            with col2:
                st.metric("Std Deviation", f"{elev_stats.get('std', 'N/A'):.2f} {unit}")
                st.metric("Maximum", f"{elev_stats.get('max', 'N/A'):.2f} {unit}")
            with col3:
                st.metric("Median", f"{elev_stats.get('median', 'N/A'):.2f} {unit}")
                st.metric("IQR", f"{elev_stats.get('iqr', 'N/A'):.2f} {unit}")
            with col4:
                st.metric("Coverage", f"{elev_stats.get('coverage_pct', 'N/A'):.2f}%")
                st.metric("Data Points", f"{elev_stats.get('count', 'N/A'):,}")
            
            # Elevation ranges
            st.subheader(f"{feature_name} Ranges")
            if all(k in elev_stats for k in ['min', 'max', 'mean']):
                elevation_range = elev_stats['max'] - elev_stats['min']
                st.metric("Total Range", f"{elevation_range:.2f} {unit}")
                
                # Classification
                low_elev = elev_stats['min'] + (elevation_range * 0.33)
                mid_elev = elev_stats['min'] + (elevation_range * 0.66)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Low Elevation", f"< {low_elev:.0f} {unit}")
                with col2:
                    st.metric("Medium Elevation", f"{low_elev:.0f} - {mid_elev:.0f} {unit}")
                with col3:
                    st.metric("High Elevation", f"> {mid_elev:.0f} {unit}")
        else:
            st.error(f"Error calculating statistics: {elev_stats.get('error')}")
    
    with elev_tab2:
        st.subheader(f"2. Spatial Distribution: {feature_name}")
        with st.spinner("Analyzing terrain patterns..."):
            elev_spatial = analyze_spatial_distribution(proc_path, AOI)
        
    if elev_spatial and "error" not in elev_spatial:
        # Display terrain metrics
        st.markdown("**Terrain Characteristics**")
        
        # Create metrics in columns
        metrics_to_show = []
        if "north_south_gradient" in elev_spatial and elev_spatial["north_south_gradient"] is not None:
            metrics_to_show.append(("N-S Gradient", elev_spatial["north_south_gradient"], f"{unit}/unit"))
        
        if "east_west_gradient" in elev_spatial and elev_spatial["east_west_gradient"] is not None:
            metrics_to_show.append(("E-W Gradient", elev_spatial["east_west_gradient"], f"{unit}/unit"))
        
        if "mean_slope" in elev_spatial:
            metrics_to_show.append(("Mean Slope", elev_spatial["mean_slope"], "°"))
        
        if "max_slope" in elev_spatial:
            metrics_to_show.append(("Max Slope", elev_spatial["max_slope"], "°"))
        
        # Display in columns
        cols = st.columns(min(4, len(metrics_to_show)))
        for i, (label, value, unit_display) in enumerate(metrics_to_show):
            with cols[i % len(cols)]:
                st.metric(label, f"{value:.4f} {unit_display}")
        
        if not metrics_to_show:
            st.info("Limited terrain metrics available.")
        
        # Terrain visualization
        try:
            da = _ensure_dataarray(rxr.open_rasterio(proc_path, masked=True)).squeeze()
            arr = np.asarray(da.values, dtype=float)
            
            # Handle nodata
            nodata = da.rio.nodata
            if nodata is not None:
                arr = np.where(arr == float(nodata), np.nan, arr)
            
            # Calculate min and max for colorbar ticks
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size > 0:
                min_val = np.nanmin(finite_vals)
                max_val = np.nanmax(finite_vals)
                
                # Round down to nearest 100 for min, round up for max
                min_rounded = max(0, int(np.floor(min_val / 100.0)) * 100)
                max_rounded = int(np.ceil(max_val / 100.0)) * 100
                
                # Create tick locations every 100 units
                tick_locations = np.arange(min_rounded, max_rounded + 100, 100)
            
            # Create terrain visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(arr, cmap='terrain', alpha=0.9)
            ax.set_title(f'{feature_name} Map')
            ax.set_xlabel('East-West')
            ax.set_ylabel('North-South')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{unit}' if unit else feature_name)
            
            # Set colorbar ticks for elevation plots
            if finite_vals.size > 0 and len(tick_locations) <= 20:
                cbar.set_ticks(tick_locations)
            
            st.pyplot(fig)
            plt.close(fig)
        except:
            pass
    else:
        st.warning("Could not compute spatial distribution statistics")
    with elev_tab3:
        st.subheader(f"3. Univariate Analysis: {feature_name}")
        with st.spinner("Generating elevation distribution plots..."):
            elev_uni = create_univariate_plots(proc_path, feature_name)
        
        if elev_uni:
            st.pyplot(elev_uni)
            plt.close(elev_uni)
            
            # Elevation-specific interpretation
            st.info("""
            **Distribution Interpretation:**
            - **Histogram**: Shows frequency of different elevation ranges
            - **Box Plot**: Identifies median elevation and elevation spread
            - **KDE Plot**: Smooth elevation probability density
            - **Q-Q Plot**: Tests if elevation follows normal distribution (often not for mountainous regions)
            """)
        else:
            st.warning("Could not generate univariate analysis plots")
    
    with elev_tab4:
        st.subheader(f"4. Spatial Profiles: {feature_name}")
        with st.spinner("Generating elevation profiles..."):
            temp_fig = temporal_analysis_elevation(proc_path)
        
        if temp_fig:
            st.pyplot(temp_fig)
            plt.close(temp_fig)
            
            # Interpretation of profiles
            st.info("""
            **Spatial Profile Interpretation:**
            - **N-S Profile**: Shows elevation changes from North to South
            - **E-W Profile**: Shows elevation changes from East to West
            - **Quadrant Box Plot**: Compares elevation distributions in different map regions
            - **CDF**: Shows what percentage of area is below certain elevation
            """)
            
            # Additional elevation metrics
            try:
                da = _ensure_dataarray(rxr.open_rasterio(proc_path, masked=True)).squeeze()
                arr = np.asarray(da.values, dtype=float)
                
                # Handle nodata
                nodata = da.rio.nodata
                if nodata is not None:
                    arr = np.where(arr == float(nodata), np.nan, arr)
                
                finite_vals = arr[np.isfinite(arr)]
                
                if finite_vals.size > 0:
                    # Calculate area above certain thresholds
                    mean_elev = np.nanmean(finite_vals)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        above_mean = np.sum(finite_vals > mean_elev)
                        pct_above = (above_mean / len(finite_vals)) * 100
                        st.metric("Area Above Mean", f"{pct_above:.1f}%")
                    
                    with col2:
                        # Calculate ruggedness (simplified)
                        if arr.shape[0] > 100 and arr.shape[1] > 100:
                            sample = arr[::20, ::20]
                            sample = sample[np.isfinite(sample)]
                            if len(sample) > 0:
                                ruggedness = np.std(sample)
                                st.metric("Ruggedness Index", f"{ruggedness:.2f} {unit}")
                    
                    with col3:
                        # Elevation zones
                        low_count = np.sum(finite_vals < 500)
                        mid_count = np.sum((finite_vals >= 500) & (finite_vals < 1500))
                        high_count = np.sum(finite_vals >= 1500)
                        
                        if low_count + mid_count + high_count > 0:
                            st.metric("High Elev (>1500m)", f"{(high_count/len(finite_vals)*100):.1f}%")
            except:
                pass
        else:
            st.warning("Could not generate elevation profiles")

# ---------------------------------------------------------------
# 4) LAND COVER
# ---------------------------------------------------------------
elif page == "Land Cover":
    st.header("🌿 Land Cover — Vector/Shapefile Visualization")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🗺️ Map Visualization", "📊 Statistics & Analysis", "ℹ️ Metadata & Cleaning"])
    
    with tab1:
        # Find available vector files
        vector_files = []
        raster_files = []
        
        if os.path.exists(LANDCOVER_PROC):
            # Find shapefiles
            for item in os.listdir(LANDCOVER_PROC):
                if item.endswith(".shp"):
                    base_name = os.path.splitext(item)[0]
                    required_files = [f"{base_name}{ext}" for ext in ['.shp', '.shx', '.dbf', '.prj']]
                    if all(os.path.exists(os.path.join(LANDCOVER_PROC, f)) for f in required_files):
                        vector_files.append({
                            "name": item,
                            "type": "shapefile",
                            "path": os.path.join(LANDCOVER_PROC, item),
                            "base": base_name
                        })
                
                # Find geopackage
                elif item.endswith(".gpkg"):
                    vector_files.append({
                        "name": item,
                        "type": "geopackage",
                        "path": os.path.join(LANDCOVER_PROC, item),
                        "base": os.path.splitext(item)[0]
                    })
                
                # Find raster
                elif item.endswith(".tif"):
                    raster_files.append({
                        "name": item,
                        "type": "raster",
                        "path": os.path.join(LANDCOVER_PROC, item),
                        "base": os.path.splitext(item)[0]
                    })
        
        if not vector_files and not raster_files:
            st.error("❌ No land cover files found in processed directory.")
            st.info(f"Checked directory: {LANDCOVER_PROC}")
            st.stop()
        
        # File selector
        st.subheader("Select Land Cover Data Source")
        
        # If there are vector files, show them first
        if vector_files:
            file_options = []
            file_labels = []
            
            for vf in vector_files:
                file_options.append(vf)
                # Get feature name for display
                feature_name, _ = get_feature_name(vf['path'])
                file_labels.append(f"{feature_name} ({vf['type']})")
            
            # If there are raster files, add them to options
            for rf in raster_files:
                file_options.append(rf)
                feature_name, _ = get_feature_name(rf['path'])
                file_labels.append(f"{feature_name} ({rf['type']})")
            
            selected_idx = st.selectbox(
                "Choose file:",
                range(len(file_options)),
                format_func=lambda i: file_labels[i]
            )
            
            selected_file = file_options[selected_idx]
            feature_name, _ = get_feature_name(selected_file['path'])
            
        else:
            # Only raster files
            selected_file = raster_files[0]
            feature_name, _ = get_feature_name(selected_file['path'])
            st.info(f"Only raster file available: {feature_name}")
        
        # Process based on file type
        if selected_file['type'] in ['shapefile', 'geopackage']:
            st.markdown(f"### 📁 {feature_name}")
            
            # Load vector data
            with st.spinner(f"Loading {selected_file['type']}..."):
                try:
                    gdf = gpd.read_file(selected_file['path'])
                    
                    # Display basic information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Features", len(gdf))
                    with col2:
                        st.metric("CRS", str(gdf.crs)[:30] + "..." if len(str(gdf.crs)) > 30 else str(gdf.crs))
                    with col3:
                        st.metric("Columns", len(gdf.columns))
                    
                    # Try to identify classification column
                    potential_class_cols = []
                    for col in gdf.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in ['code', 'class', 'type', 'lulc', 'lc', 'value', 'category']):
                            potential_class_cols.append(col)
                        elif col_lower in ['dn', 'gridcode', 'id']:
                            potential_class_cols.append(col)
                    
                    # If there are multiple columns, let the user choose
                    if potential_class_cols:
                        if len(potential_class_cols) > 1:
                            class_col = st.selectbox(
                                "Select land cover class column:",
                                potential_class_cols,
                                help="Choose the column containing land cover classification codes"
                            )
                        else:
                            class_col = potential_class_cols[0]
                            st.info(f"Using column '{class_col}' for classification")
                    else:
                        # Try numeric columns
                        numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            class_col = st.selectbox(
                                "Select numeric column for classification:",
                                numeric_cols,
                                help="No obvious class column found. Select a numeric column."
                            )
                        else:
                            st.error("No suitable classification column found!")
                            st.write("Available columns:", list(gdf.columns))
                            class_col = None
                    
                    if class_col:
                        # Ensure numeric type
                        gdf[class_col] = pd.to_numeric(gdf[class_col], errors='coerce')
                        valid_gdf = gdf.dropna(subset=[class_col])
                        
                        if len(valid_gdf) == 0:
                            st.error("No valid classification values found!")
                        else:
                            # Get unique values
                            unique_codes = sorted(valid_gdf[class_col].unique())
                            unique_codes = [int(code) for code in unique_codes if not np.isnan(code)]
                            
                            st.success(f"Found {len(unique_codes)} unique land cover classes")
                            
                            # Create map
                            st.subheader(f"🗺️ {feature_name} Map")
                            
                            # Create color map
                            if len(unique_codes) <= 20:
                                cmap_name = "tab20"
                            elif len(unique_codes) <= 40:
                                cmap_name = "tab20c"
                            else:
                                cmap_name = "viridis"
                            
                            cmap = matplotlib.colormaps[cmap_name]
                            
                            # Set figure size
                            fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
                            
                            # Draw each category
                            legend_patches = []
                            color_mapping = {}
                            
                            for i, code in enumerate(unique_codes):
                                if len(unique_codes) <= 20:
                                    color = cmap(i / max(len(unique_codes) - 1, 1))
                                else:
                                    color = cmap(i / len(unique_codes))
                                
                                color_mapping[code] = color
                                
                                subset = valid_gdf[valid_gdf[class_col] == code]
                                if not subset.empty:
                                    subset.plot(ax=ax, color=color, alpha=0.8, linewidth=0.5, edgecolor='black')
                                
                                # Create legend handles
                                class_name = LCCS_NAMES.get(code, f"Class {code}")
                                legend_patches.append(matplotlib.patches.Patch(
                                    facecolor=color, 
                                    edgecolor='black',
                                    label=f"{code}: {class_name}"
                                ))
                            
                            # Add AOI boundary
                            try:
                                aoi_reproj = AOI.to_crs(gdf.crs)
                                aoi_reproj.boundary.plot(
                                    ax=ax, 
                                    color='red', 
                                    linewidth=2, 
                                    linestyle='--',
                                    label="Study Area Boundary"
                                )
                            except Exception as e:
                                st.warning(f"Could not overlay AOI: {e}")
                            
                            # Set map properties
                            ax.set_title(f"{feature_name} - Classification", fontsize=16, fontweight='bold')
                            ax.set_xlabel("Easting")
                            ax.set_ylabel("Northing")
                            
                            # Add grid
                            ax.grid(True, alpha=0.3, linestyle='--')
                            
                            # Add scale and north arrow text
                            ax.text(0.02, 0.02, "N", transform=ax.transAxes, 
                                   fontsize=14, fontweight='bold', color='black',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                            
                            # Add legend
                            if len(legend_patches) <= 15:
                                # Internal legend
                                ax.legend(handles=legend_patches, loc='upper right', 
                                         fontsize=9, framealpha=0.9, title="Land Cover Classes")
                            else:
                                # External legend (display separately)
                                st.info(f"Too many classes ({len(legend_patches)}) for inline legend. See legend below.")
                                
                                # Create separate legend figure
                                fig_legend, ax_legend = plt.subplots(figsize=(8, min(20, len(legend_patches) * 0.5)))
                                ax_legend.axis('off')
                                
                                # Display in columns
                                ncol = 1 if len(legend_patches) <= 30 else 2
                                legend = ax_legend.legend(
                                    handles=legend_patches, 
                                    loc='center', 
                                    fontsize=8, 
                                    ncol=ncol,
                                    frameon=True,
                                    title="Land Cover Classes (Click to expand)"
                                )
                                legend.get_title().set_fontsize(10)
                                plt.tight_layout()
                                st.pyplot(fig_legend)
                                plt.close(fig_legend)
                            
                            # Display map
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Interactive map view
                            st.subheader("📍 Interactive Map View")
                            
                            # Convert to geojson for interactive map
                            if len(valid_gdf) < 10000:  # Limit data volume
                                # Simplify geometry for faster display
                                try:
                                    simplified_gdf = valid_gdf.copy()
                                    if valid_gdf.crs and valid_gdf.crs.is_projected:
                                        # If projected coordinate system, convert to WGS84 for map display
                                        simplified_gdf = simplified_gdf.to_crs("EPSG:4326")
                                    
                                    # Add color and class name columns
                                    simplified_gdf['color'] = simplified_gdf[class_col].map(color_mapping)
                                    simplified_gdf['class_name'] = simplified_gdf[class_col].apply(
                                        lambda x: LCCS_NAMES.get(int(x), f"Class {int(x)}")
                                    )
                                    
                                    # Create interactive map
                                    st.map(
                                        simplified_gdf,
                                        latitude='centroid_lat' if 'centroid_lat' in simplified_gdf.columns else None,
                                        longitude='centroid_lon' if 'centroid_lon' in simplified_gdf.columns else None,
                                        color='color',
                                        size=100
                                    )
                                    
                                    with st.expander("View Sample Data on Map"):
                                        st.dataframe(simplified_gdf[[class_col, 'class_name', 'geometry']].head(10))
                                        
                                except Exception as e:
                                    st.warning(f"Could not create interactive map: {e}")
                            else:
                                st.info("Data too large for interactive map. Showing sample instead.")
                                sample_gdf = valid_gdf.sample(min(1000, len(valid_gdf)))
                                try:
                                    sample_gdf = sample_gdf.to_crs("EPSG:4326")
                                    st.map(sample_gdf)
                                except Exception as e:
                                    st.warning(f"Could not create sample map: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading {selected_file['type']}: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        else:
            # Process raster file
            st.markdown(f"### 🗺️ {feature_name}")
            _before_after_block(selected_file['path'], feature_name, 
                              fallback_src_folder=LANDCOVER_DIR, categorical=True)
    
    with tab2:
        if 'feature_name' in locals():
            st.header(f"📊 {feature_name} Statistics & Analysis")
        else:
            st.header("📊 Land Cover Statistics & Analysis")
        
        if 'gdf' in locals() and 'class_col' in locals() and class_col:
            # Calculate statistics
            st.subheader("Class Distribution")
            
            # 1. Basic statistics
            class_counts = valid_gdf[class_col].value_counts().reset_index()
            class_counts.columns = ['Code', 'Feature_Count']
            class_counts['Code'] = class_counts['Code'].astype(int)
            class_counts['Class_Name'] = class_counts['Code'].apply(
                lambda x: LCCS_NAMES.get(x, f"Class {x}")
            )
            class_counts = class_counts.sort_values('Code')
            
            # Display table
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Classes", len(class_counts))
                st.metric("Total Features", len(valid_gdf))
            
            with col2:
                avg_features = len(valid_gdf) / len(class_counts) if len(class_counts) > 0 else 0
                st.metric("Avg Features per Class", f"{avg_features:.1f}")
                
                dominant_class = class_counts.iloc[0]['Class_Name'] if len(class_counts) > 0 else "N/A"
                st.metric("Dominant Class", dominant_class[:20])
            
            # 2. Area calculation (if possible)
            st.subheader("Area Statistics")
            
            try:
                if valid_gdf.crs and valid_gdf.crs.is_projected:
                    # Calculate area
                    valid_gdf_copy = valid_gdf.copy()
                    valid_gdf_copy['area_sqkm'] = valid_gdf_copy.geometry.area / 1e6
                    
                    area_stats = valid_gdf_copy.groupby(class_col).agg({
                        'area_sqkm': ['sum', 'mean', 'min', 'max', 'count']
                    }).round(2)
                    
                    area_stats.columns = ['Total_Area_sqkm', 'Mean_Area_sqkm', 
                                         'Min_Area_sqkm', 'Max_Area_sqkm', 'Polygon_Count']
                    area_stats = area_stats.reset_index()
                    area_stats['Code'] = area_stats[class_col].astype(int)
                    area_stats['Class_Name'] = area_stats['Code'].apply(
                        lambda x: LCCS_NAMES.get(x, f"Class {x}")
                    )
                    
                    # Sort by area
                    area_stats = area_stats.sort_values('Total_Area_sqkm', ascending=False)
                    
                    # Display area table
                    st.dataframe(area_stats[['Code', 'Class_Name', 'Total_Area_sqkm', 
                                            'Polygon_Count', 'Mean_Area_sqkm']])
                    
                    # Area distribution chart
                    fig_area, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Pie chart (top 10 categories)
                    top_n = min(10, len(area_stats))
                    top_classes = area_stats.head(top_n)
                    other_area = area_stats['Total_Area_sqkm'].iloc[top_n:].sum()
                    
                    if other_area > 0:
                        pie_data = pd.concat([
                            pd.Series(top_classes['Total_Area_sqkm'].values, 
                                     index=top_classes['Class_Name'].apply(lambda x: x[:20])),
                            pd.Series([other_area], index=['Other'])
                        ])
                    else:
                        pie_data = pd.Series(top_classes['Total_Area_sqkm'].values,
                                           index=top_classes['Class_Name'].apply(lambda x: x[:20]))
                    
                    wedges, texts, autotexts = ax1.pie(pie_data, autopct='%1.1f%%', 
                                                      startangle=90, pctdistance=0.85)
                    ax1.set_title(f'Top {top_n} Classes by Area', fontweight='bold')
                    
                    # Beautify percentage text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    # Bar chart
                    bars = ax2.barh(range(len(top_classes)), top_classes['Total_Area_sqkm'])
                    ax2.set_yticks(range(len(top_classes)))
                    ax2.set_yticklabels(top_classes['Class_Name'].apply(lambda x: x[:25]))
                    ax2.set_xlabel('Area (sq km)')
                    ax2.set_title(f'Top {top_n} Classes Area Distribution')
                    ax2.invert_yaxis()
                    
                    # Add values on bars
                    for i, (bar, area) in enumerate(zip(bars, top_classes['Total_Area_sqkm'])):
                        ax2.text(area, bar.get_y() + bar.get_height()/2, 
                                f'{area:,.1f}', va='center', ha='left', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig_area)
                    plt.close(fig_area)
                    
                    # Download data
                    csv = area_stats.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Area Statistics (CSV)",
                        data=csv,
                        file_name=f"landcover_area_stats_{feature_name}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("⚠️ CRS is not projected. Cannot calculate accurate area.")
                    st.info("Showing feature count distribution instead.")
                    
                    # Feature count distribution chart
                    fig_count, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(range(len(class_counts)), class_counts['Feature_Count'])
                    ax.set_yticks(range(len(class_counts)))
                    ax.set_yticklabels(class_counts['Class_Name'].apply(lambda x: x[:30]))
                    ax.set_xlabel('Number of Features')
                    ax.set_title('Land Cover Class Distribution (Feature Count)')
                    ax.invert_yaxis()
                    
                    # Add value labels
                    for i, (bar, count) in enumerate(zip(bars, class_counts['Feature_Count'])):
                        ax.text(count, bar.get_y() + bar.get_height()/2, 
                               f'{count:,}', va='center', ha='left', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig_count)
                    plt.close(fig_count)
                    
            except Exception as e:
                st.error(f"Error calculating area statistics: {e}")
            
            # 3. Spatial analysis
            st.subheader("Spatial Analysis")
            
            # Create sub-tabs
            spatial_tab1, spatial_tab2 = st.tabs(["Geometric Properties", "Spatial Pattern"])
            
            with spatial_tab1:
                try:
                    # Calculate geometric properties
                    valid_gdf_copy = valid_gdf.copy()
                    valid_gdf_copy['area'] = valid_gdf_copy.geometry.area
                    valid_gdf_copy['perimeter'] = valid_gdf_copy.geometry.length
                    
                    # Calculate compactness (4π * area / perimeter²)
                    valid_gdf_copy['compactness'] = 4 * np.pi * valid_gdf_copy['area'] / (valid_gdf_copy['perimeter'] ** 2)
                    
                    # Group statistics
                    geom_stats = valid_gdf_copy.groupby(class_col).agg({
                        'area': ['mean', 'std', 'min', 'max'],
                        'compactness': ['mean', 'std']
                    }).round(4)
                    
                    geom_stats.columns = ['Mean_Area', 'Std_Area', 'Min_Area', 'Max_Area',
                                         'Mean_Compactness', 'Std_Compactness']
                    geom_stats = geom_stats.reset_index()
                    geom_stats['Code'] = geom_stats[class_col].astype(int)
                    geom_stats['Class_Name'] = geom_stats['Code'].apply(
                        lambda x: LCCS_NAMES.get(x, f"Class {x}")
                    )
                    
                    st.dataframe(geom_stats[['Code', 'Class_Name', 'Mean_Area', 'Std_Area',
                                            'Mean_Compactness', 'Std_Compactness']])
                    
                except Exception as e:
                    st.warning(f"Could not compute geometric properties: {e}")
            
            with spatial_tab2:
                st.info("Spatial pattern analysis would require additional computations.")
                st.write("Potential analyses:")
                st.write("• Nearest neighbor distance between polygons")
                st.write("• Spatial autocorrelation (Moran's I)")
                st.write("• Fragmentation indices")
                st.write("• Edge density calculations")
        
        else:
            st.info("Load data in the 'Map Visualization' tab first to see statistics.")
    
    with tab3:
        if 'feature_name' in locals():
            st.header(f"ℹ️ {feature_name} Metadata & Cleaning")
        else:
            st.header("ℹ️ Metadata & Cleaning Information")
        
        # Display cleaning information
        if 'selected_file' in locals():
            _show_cleaning_and_issues(feature_name, selected_file['path'], "categorical")
        else:
            st.info("No file selected yet. Please select a file in the 'Map Visualization' tab.")
        
        # Display LCCS reference table
        st.subheader("📚 LCCS (Land Cover Classification System) Reference")
        
        # Display full or partial LCCS table
        show_full_lccs = st.checkbox("Show full LCCS reference table", value=False)
        
        if show_full_lccs:
            lccs_df = pd.DataFrame(list(LCCS_NAMES.items()), columns=['Code', 'Description'])
            lccs_df = lccs_df.sort_values('Code')
            st.dataframe(lccs_df, height=400)
        else:
            # Only show common categories
            common_codes = [10, 11, 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                          110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
            common_lccs = {code: LCCS_NAMES.get(code, f"Code {code}") for code in common_codes}
            common_df = pd.DataFrame(list(common_lccs.items()), columns=['Code', 'Description'])
            st.dataframe(common_df)
        
        # Data quality information
        st.subheader("🔍 Data Quality Checks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'gdf' in locals() and 'class_col' in locals():
                null_count = gdf[class_col].isnull().sum() if class_col else 0
                st.metric("Null Values", null_count)
        
        with col2:
            if 'valid_gdf' in locals():
                duplicate_geom = valid_gdf.geometry.duplicated().sum()
                st.metric("Duplicate Geometries", duplicate_geom)
        
        with col3:
            if 'valid_gdf' in locals():
                invalid_geom = (~valid_gdf.geometry.is_valid).sum()
                st.metric("Invalid Geometries", invalid_geom)
        
        # Display raw data
        with st.expander("View Raw Data Structure"):
            if 'gdf' in locals():
                st.write("**Columns and Data Types:**")
                col_info = pd.DataFrame({
                    'Column': gdf.columns,
                    'Type': [str(gdf[col].dtype) for col in gdf.columns],
                    'Non-Null Count': [gdf[col].notnull().sum() for col in gdf.columns]
                })
                st.dataframe(col_info)
                
                st.write("**Sample Data (first 10 rows):**")
                st.dataframe(gdf.head(10))

# ---------------------------------------------------------------
# 5) FIRE
# ---------------------------------------------------------------
elif page == "Fire":
    feature_name = "Fire Detections"
    st.header(f"{feature_name} (VIIRS 2024) — Raw vs Processed")

    raw_files = {
        "Algeria": os.path.join(FIRE_DIR, "viirs-jpss1_2024_Algeria.csv"),
        "Tunisia": os.path.join(FIRE_DIR, "viirs-jpss1_2024_Tunisia.csv"),
    }
    dfs = []
    for label, p in raw_files.items():
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["source"] = label
            dfs.append(df)
    if not dfs:
        st.error("No fire CSVs found.")
        st.stop()
    raw_df = pd.concat(dfs, ignore_index=True)

    # --- Prepare Cleaned Data (Moved up for mapping) ---
    n_raw = len(raw_df)
    cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in raw_df.columns]
    df_dedup = raw_df.drop_duplicates(subset=cols) if cols else raw_df.drop_duplicates(subset=["latitude", "longitude"]) \
                if {"latitude", "longitude"}.issubset(raw_df.columns) else raw_df.copy()

    if {"latitude", "longitude"}.issubset(df_dedup.columns):
        bbox = AOI.total_bounds  # minx,miny,maxx,maxy
        lon = pd.to_numeric(df_dedup["longitude"], errors="coerce")
        lat = pd.to_numeric(df_dedup["latitude"], errors="coerce")
        in_box = (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])
        df_clean = df_dedup.loc[in_box].copy()
    else:
        df_clean = df_dedup.copy()

    # --- Map of Fire Points (Replaces Before/After) ---
    st.subheader(f"📍 {feature_name} Locations (VIIRS 2024)")
    if not df_clean.empty:
        # Assign colors based on fire type if available
        # VIIRS Types: 0=Vegetation, 1=Volcano, 2=Static, 3=Offshore
        if "type" in df_clean.columns:
            # Map types to hex colors
            color_map = {
                0: "#FF4B4B", # Red (Vegetation)
                1: "#FFA500", # Orange (Volcano)
                2: "#FFD700", # Gold (Static)
                3: "#1E90FF"  # Blue (Offshore)
            }
            
            # Ensure type is numeric for mapping
            # Robust color mapping function
            def _get_fire_color(val):
                try:
                    # Handle float/int/str conversion safely
                    v = int(float(val))
                    return color_map.get(v, "#808080")
                except Exception:
                    return "#808080"

            # Apply color mapping
            df_clean["color"] = df_clean["type"].apply(_get_fire_color)

            # Convert hex colors to RGB list for pydeck
            def hex_to_rgb(h):
                h = h.lstrip('#')
                return [int(h[i:i+2], 16) for i in (0, 2, 4)]

            df_clean["color_rgb"] = df_clean["color"].apply(hex_to_rgb)
            
            # Define view state centered on AOI
            # Calculate centroid of AOI
            # Use total_bounds to get center if centroid fails or is complex
            minx, miny, maxx, maxy = AOI.total_bounds
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            
            view_state = pdk.ViewState(
                latitude=center_y,
                longitude=center_x,
                zoom=5,
                pitch=0,
            )

            # Layer 1: AOI Boundaries (Algeria & Tunisia)
            # Convert AOI to GeoJSON for pydeck
            aoi_json = json.loads(AOI.to_json())
            aoi_layer = pdk.Layer(
                "GeoJsonLayer",
                aoi_json,
                pickable=False,
                stroked=True,
                filled=False,
                line_width_min_pixels=2,
                get_line_color=[0, 0, 0, 200], # Black borders
            )

            # Layer 2: Fire Points
            fire_layer = pdk.Layer(
                "ScatterplotLayer",
                df_clean,
                get_position=["longitude", "latitude"],
                get_color="color_rgb",
                get_radius=2000,  # Larger radius for visibility
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_min_pixels=3,
                radius_max_pixels=10,
            )

            # Render Deck
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[aoi_layer, fire_layer],
                tooltip=True
            ))
            
            # Legend
            st.markdown(
                """
                <div style="display: flex; flex-wrap: wrap; gap: 15px; font-size: 0.85em; margin-top: -10px; margin-bottom: 15px; padding: 8px; background-color: #ffff; border-radius: 5px; color: black;">
                    <div><span style="color:#FF4B4B; font-weight:bold">●</span> Vegetation Fire (0)</div>
                    <div><span style="color:#FFA500; font-weight:bold">●</span> Volcano (1)</div>
                    <div><span style="color:#FFD700; font-weight:bold">●</span> Static Land Source (2)</div>
                    <div><span style="color:#1E90FF; font-weight:bold">●</span> Offshore (3)</div>
                    <div><span style="color:#808080; font-weight:bold">●</span> Other/Unknown</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.map(df_clean[["latitude", "longitude"]])
    else:
        st.warning("No fire points found in AOI.")

    # Processed monthly density rasters (if present) - show cleaning info only
    proc_months = sorted([f for f in os.listdir(FIRE_DIR) if f.startswith("viirs_count_") and f.endswith(".tif")])
    if proc_months:
        example = os.path.join(FIRE_DIR, proc_months[0])
        example_feature, _ = get_feature_name(example)
        _show_cleaning_and_issues(example_feature, example, "count")

    # Quick raw vs dedup+bbox counts (transparency)
    st.subheader("📊 Raw vs Cleaned Data")

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Raw Detections", f"{n_raw:,}")
    with m2:
        st.metric("Cleaned Detections (AOI)", f"{len(df_clean):,}")
    with m3:
        pct_kept = (len(df_clean) / n_raw * 100) if n_raw > 0 else 0
        st.metric("Retention Rate", f"{pct_kept:.1f}%")

    # Visualization of Data Reduction
    fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
    comp_data = pd.DataFrame({
        "Stage": ["Raw", "Cleaned"],
        "Count": [n_raw, len(df_clean)]
    })
    sns.barplot(data=comp_data, y="Stage", x="Count", palette="viridis", ax=ax_comp)
    ax_comp.set_title("Data Reduction Pipeline")
    ax_comp.set_xlabel("Number of Fire Detections")
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # --- FIRE DATA ANALYSIS ---
    if not df_clean.empty:
        st.markdown("---")
        st.header(f"🔥 {feature_name} Analysis")

        # 1. Descriptive Statistics
        st.subheader("1. Descriptive Statistics")
        
        # Numerical stats
        st.markdown("**Numerical Attributes**")
        num_cols = [c for c in ["bright_ti4", "bright_ti5", "frp"] if c in df_clean.columns]
        if num_cols:
            # Get feature names for display
            stats_df = df_clean[num_cols].describe().T
            stats_df.index = [get_feature_name(col)[0] for col in num_cols]
            st.dataframe(stats_df)

        # Categorical stats
        st.markdown("**Categorical Attributes**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if "confidence" in df_clean.columns:
                st.caption("Confidence Level")
                conf_counts = df_clean["confidence"].value_counts()
                st.dataframe(conf_counts)
        with c2:
            if "daynight" in df_clean.columns:
                st.caption("Day/Night")
                dn_counts = df_clean["daynight"].value_counts()
                st.dataframe(dn_counts)
        with c3:
            if "type" in df_clean.columns:
                st.caption("Fire Type")
                type_counts = df_clean["type"].value_counts()
                # Map type codes to names
                type_names = {
                    0: "Vegetation Fire",
                    1: "Volcano",
                    2: "Static Land Source",
                    3: "Offshore"
                }
                type_counts.index = [type_names.get(i, f"Type {i}") for i in type_counts.index]
                st.dataframe(type_counts)

        # 2. Spatial Distribution
        st.subheader("2. Spatial Distribution")
        st.markdown("Fire locations within AOI:")
        
        if "size" not in df_clean.columns:
            df_clean["size"] = 300

        if "color" in df_clean.columns:
            st.map(df_clean, latitude="latitude", longitude="longitude", color="color", size="size")
        else:
            st.map(df_clean, latitude="latitude", longitude="longitude", size="size")

        # 3. Univariate Analysis
        st.subheader("3. Univariate Analysis")
        
        u1, u2 = st.columns(2)
        with u1:
            if "frp" in df_clean.columns:
                feature_name, unit = get_feature_name("frp")
                st.markdown(f"**{feature_name} Distribution (log scale)**")
                fig_frp, ax_frp = plt.subplots(figsize=(6, 4))
                valid_frp = df_clean[df_clean["frp"] > 0]
                if not valid_frp.empty:
                    sns.histplot(data=valid_frp, x="frp", log_scale=True, kde=True, color="orange", ax=ax_frp)
                    ax_frp.set_xlabel(f"{feature_name} ({unit})")
                    ax_frp.set_title(f"{feature_name} Distribution")
                    st.pyplot(fig_frp)
                else:
                    st.info("No valid FRP data.")
                plt.close(fig_frp)

        with u2:
            if "bright_ti4" in df_clean.columns:
                feature_name, unit = get_feature_name("bright_ti4")
                st.markdown(f"**{feature_name}**")
                fig_bt, ax_bt = plt.subplots(figsize=(6, 4))
                sns.histplot(data=df_clean, x="bright_ti4", kde=True, color="red", ax=ax_bt)
                ax_bt.set_xlabel(f"{feature_name} ({unit})")
                ax_bt.set_title(f"{feature_name} Distribution")
                st.pyplot(fig_bt)
                plt.close(fig_bt)

        # 4. Bivariate Analysis
        st.subheader("4. Bivariate Analysis")
        
        b1, b2 = st.columns(2)
        with b1:
            if "bright_ti4" in df_clean.columns and "frp" in df_clean.columns:
                feat1, unit1 = get_feature_name("bright_ti4")
                feat2, unit2 = get_feature_name("frp")
                st.markdown(f"**{feat1} vs. {feat2}**")
                fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df_clean, x="bright_ti4", y="frp", alpha=0.3, s=15, color="firebrick", ax=ax_sc)
                ax_sc.set_yscale("log")
                ax_sc.set_xlabel(f"{feat1} ({unit1})")
                ax_sc.set_ylabel(f"{feat2} ({unit2})")
                ax_sc.set_title(f"{feat1} vs {feat2}")
                st.pyplot(fig_sc)
                plt.close(fig_sc)
            
        with b2:
            if "frp" in df_clean.columns and "daynight" in df_clean.columns:
                feat_name, unit = get_feature_name("frp")
                st.markdown(f"**{feat_name} by Day/Night**")
                fig_box, ax_box = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df_clean, x="daynight", y="frp", showfliers=False, palette="Set2", ax=ax_box)
                ax_box.set_ylabel(f"{feat_name} ({unit}) - outliers hidden")
                ax_box.set_title(f"{feat_name} Distribution by Time")
                st.pyplot(fig_box)
                plt.close(fig_box)

        # 5. Temporal Analysis
        st.subheader("5. Temporal Analysis")
        
        if "acq_date" in df_clean.columns:
            try:
                df_temp = df_clean.copy()
                df_temp["acq_date"] = pd.to_datetime(df_temp["acq_date"])
                
                # Daily
                daily_counts = df_temp.groupby("acq_date").size().reset_index(name="count")
                
                st.markdown("**Daily Fire Counts**")
                fig_daily, ax_daily = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=daily_counts, x="acq_date", y="count", color="tab:red", ax=ax_daily)
                ax_daily.set_xlabel("Date")
                ax_daily.set_ylabel("Number of Fires")
                ax_daily.set_title("Daily Fire Frequency")
                st.pyplot(fig_daily)
                plt.close(fig_daily)
                
                # Monthly
                st.markdown("**Monthly Fire Counts**")
                monthly_counts = df_temp.set_index("acq_date").resample("ME").size().reset_index(name="count")
                
                fig_monthly, ax_monthly = plt.subplots(figsize=(10, 4))
                sns.barplot(data=monthly_counts, x="acq_date", y="count", color="tab:orange", ax=ax_monthly)
                # Format x-axis for monthly
                ax_monthly.set_xticklabels([d.strftime('%Y-%m') for d in monthly_counts["acq_date"]], rotation=45)
                ax_monthly.set_xlabel("Month")
                ax_monthly.set_ylabel("Number of Fires")
                ax_monthly.set_title("Monthly Fire Frequency")
                st.pyplot(fig_monthly)
                plt.close(fig_monthly)
                
            except Exception as e:
                st.error(f"Error in temporal analysis: {e}")

        # 6. Correlation Analysis
        st.subheader("6. Correlation Analysis")
        
        # Check if we have enough data
        if len(df_clean.select_dtypes(include=[np.number]).columns) >= 2:
            with st.spinner("Calculating correlations..."):
                corr_fig = fire_correlation_analysis(df_clean)
            
            if corr_fig:
                st.pyplot(corr_fig)
                plt.close(corr_fig)
                
                # Add interpretation
                st.info("""
                **Correlation Interpretation:**
                - **Positive correlation (red)**: Variables increase together
                - **Negative correlation (blue)**: Variables have inverse relationship
                - **Near zero (white)**: No linear relationship
                """)
            else:
                st.warning("Could not generate correlation analysis")
            
            # Additional correlation details
            st.subheader("Detailed Correlation Analysis")
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            
            # Let user select variables for detailed scatter
            if len(numeric_cols) >= 2:
                # Create display names for selection
                display_options = []
                col_mapping = {}
                for col in numeric_cols:
                    feat_name, unit = get_feature_name(col)
                    display_name = f"{feat_name} ({unit})" if unit else feat_name
                    display_options.append(display_name)
                    col_mapping[display_name] = col
                
                col1, col2 = st.columns(2)
                with col1:
                    x_display = st.selectbox("X Variable", display_options, index=0)
                    x_var = col_mapping[x_display]
                with col2:
                    # Don't allow same variable selection
                    y_options = [opt for opt in display_options if opt != x_display]
                    y_display = st.selectbox("Y Variable", y_options, index=min(1, len(y_options)-1))
                    y_var = col_mapping[y_display]
                
                # Create detailed scatter
                if x_var != y_var:
                    x_feat, x_unit = get_feature_name(x_var)
                    y_feat, y_unit = get_feature_name(y_var)
                    
                    fig_scatter, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(df_clean[x_var], df_clean[y_var], alpha=0.3, s=10, color='steelblue')
                    ax.set_xlabel(f"{x_feat} ({x_unit})" if x_unit else x_feat)
                    ax.set_ylabel(f"{y_feat} ({y_unit})" if y_unit else y_feat)
                    ax.set_title(f'{x_feat} vs {y_feat}')
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate and display correlation
                    correlation = df_clean[[x_var, y_var]].corr().iloc[0, 1]
                    ax.annotate(f'Correlation: {correlation:.3f}', 
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    # Add trend line
                    if len(df_clean) > 2:
                        try:
                            mask = df_clean[[x_var, y_var]].notna().all(axis=1)
                            if mask.sum() > 2:
                                x_vals = df_clean.loc[mask, x_var].values
                                y_vals = df_clean.loc[mask, y_var].values
                                z = np.polyfit(x_vals, y_vals, 1)
                                p = np.poly1d(z)
                                ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)
                        except:
                            pass
                    
                    st.pyplot(fig_scatter)
                    plt.close(fig_scatter)
        else:
            st.info("Not enough numerical columns for correlation analysis")

# ==========================
# NOTE for pipeline cropping
# ==========================
st.caption(
    "If you want **total coverage ~ 100%** as well, reprocess rasters with a *true crop* to AOI "
    "(e.g., `da.rio.clip(AOI.geometry, AOI.crs, drop=True)` or `rasterio.mask(..., crop=True)`), "
    "so the output extent matches the AOI instead of a global grid."
)