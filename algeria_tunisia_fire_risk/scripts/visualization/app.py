#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit EDA app for Algeria–Tunisia datasets.

Loads files from visualization/data:
 - climate_2024.nc
 - soil_id_with_attrs_aoi.nc
 - elevation_clipped.tif
 - landcover_clipped.(shp,shx,dbf,prj,cpg,cpg)
Also browses cleaned / merged outputs under DATA_CLEANED.

Key fixes:
 - Robust fire label normalization (normalize_fire_column)
 - Dynamic label detection (no hard-coded [0,1])
 - Optional debug panel (toggle in sidebar)
 - Prevent accidental filtering out non-fire rows
 - Treat low-cardinality numeric codes (<=20 unique) as categorical, avoiding unintended numeric range filtering
 - Safer selection of focus variable (defaults to elevation if present)
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import rioxarray as rxr  # requires rasterio
except ImportError:
    rxr = None

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats

try:
    from scipy.spatial import cKDTree  # for distance validation
except Exception:
    cKDTree = None

import pydeck as pdk  # (planned usage)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
FIRE_DIR = BASE / "FIRE"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLEANED_DIR = BASE / "DATA_CLEANED"

def locate_merged_dataset() -> Path | None:
    # Prioritize CSV to avoid corrupted parquet issues
    candidates = [
        CLEANED_DIR / "processed" / "merged_dataset.csv",
        BASE / "processed" / "merged_dataset.csv",
        Path.cwd() / "processed" / "merged_dataset.csv",
        CLEANED_DIR / "processed" / "merged_dataset.parquet",
        BASE / "processed" / "merged_dataset.parquet",
        Path.cwd() / "processed" / "merged_dataset.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@st.cache_data
def load_cached_merged_dataset(path_str: str, mtime_ns: int) -> pd.DataFrame:
    """Load merged dataset with cache invalidation tied to file path + mtime."""
    p = Path(path_str)
    
    if p.suffix == ".parquet":
        # Try multiple engines/methods for parquet files
        try:
            # First try with pyarrow (default)
            return pd.read_parquet(p, engine='pyarrow')
        except OSError as e:
            if "Repetition level histogram" in str(e) or "mismatch" in str(e):
                # Try with fastparquet as fallback
                try:
                    return pd.read_parquet(p, engine='fastparquet')
                except Exception:
                    pass
                # Try reading with pyarrow but with legacy settings
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(p, use_legacy_dataset=True)
                    return table.to_pandas()
                except Exception:
                    pass
            # If all parquet methods fail, check for CSV fallback
            csv_fallback = p.with_suffix('.csv')
            if csv_fallback.exists():
                st.warning(f"Parquet file corrupted, falling back to CSV: {csv_fallback}")
                return pd.read_csv(csv_fallback)
            raise OSError(
                f"Failed to read parquet file: {p}\n"
                f"The file may be corrupted. Try regenerating it or converting to CSV.\n"
                f"Original error: {e}"
            )
    else:
        return pd.read_csv(p)

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AT EDA", layout="wide")
st.title("Algeria–Tunisia EDA Console")
st.caption("Quick exploration of climate, soil, elevation, land cover, and merged datasets")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dataarray(raster) -> xr.DataArray:
    """Normalize rioxarray/xarray open outputs to a DataArray."""
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

@st.cache_data
def load_climate() -> xr.Dataset | None:
    p = DATA_DIR / "climate_2024.nc"
    if p.exists():
        try:
            return xr.open_dataset(p)
        except Exception as e:
            st.warning(f"Failed to open climate NetCDF: {e}")
    return None

@st.cache_data
def load_soil() -> xr.Dataset | None:
    p = DATA_DIR / "soil_id_with_attrs_aoi.nc"
    if p.exists():
        try:
            return xr.open_dataset(p)
        except Exception as e:
            st.warning(f"Failed to open soil NetCDF: {e}")
    return None

@st.cache_data
def load_elevation() -> xr.DataArray | None:
    p = DATA_DIR / "elevation_clipped.tif"
    if p.exists() and rxr is not None:
        try:
            return _ensure_dataarray(rxr.open_rasterio(p, masked=True)).squeeze()
        except Exception as e:
            st.warning(f"Failed to open elevation raster: {e}")
    return None

@st.cache_data
def load_landcover() -> gpd.GeoDataFrame | None:
    shp = DATA_DIR / "landcover_clipped.shp"
    if shp.exists():
        try:
            return gpd.read_file(shp)
        except Exception as e:
            st.warning(f"Failed to open landcover shapefile: {e}")
    return None

SENTINELS = (-9, -7, -9999)

def cap_iqr(arr: np.ndarray, factor: float = 1.5) -> tuple[np.ndarray, int, float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 5:
        return arr, 0, float('nan'), float('nan')
    q1, q3 = np.nanpercentile(finite, [25, 75])
    iqr = q3 - q1
    if iqr <= 0 or not np.isfinite(iqr):
        return arr, 0, float('nan'), float('nan')
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    mask = (arr < lo) | (arr > hi)
    out_count = int(np.isfinite(arr[mask]).sum())
    clipped = np.where(arr < lo, lo, np.where(arr > hi, hi, arr))
    return clipped, out_count, lo, hi

def preview_array(arr: np.ndarray, title: str, categorical: bool = False):
    finite = np.isfinite(arr)
    if not finite.any():
        st.info("No finite values to display.")
        return
    if categorical:
        cmap = "tab20"
        vmin = vmax = None
    else:
        valid = arr[finite]
        vmin, vmax = np.nanpercentile(valid, [2, 98])
        cmap = "viridis"
    fig, ax = plt.subplots(figsize=(7, 5))
    if categorical:
        im = ax.imshow(arr, cmap=cmap)
    else:
        vmin_val = float(vmin) if vmin is not None else float(np.nanmin(arr))
        vmax_val = float(vmax) if vmax is not None else float(np.nanmax(arr))
        im = ax.imshow(arr, cmap=cmap, vmin=vmin_val, vmax=vmax_val)
    ax.set_title(title)
    if not categorical:
        fig.colorbar(im, ax=ax)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    st.image(buf.getvalue(), caption=title, use_column_width=True)
    plt.close(fig)

def uniq(seq):
    return list(dict.fromkeys(seq))

def normalize_fire_column(df: pd.DataFrame, col='fire') -> pd.Series:
    """
    Robustly map fire-like column values to Int64 0/1.
    Accepts numeric, boolean, or common textual encodings.
    Unknown -> <NA>.
    """
    if col not in df.columns:
        raise KeyError(f"{col} not in DataFrame")
    s = df[col]

    if pd.api.types.is_bool_dtype(s):
        return s.astype('Int64').replace({True: 1, False: 0})

    # Attempt numeric conversion
    num = pd.to_numeric(s, errors='coerce')

    mapping = {
        '1': 1, '0': 0,
        'true': 1, 'false': 0, 't': 1, 'f': 0,
        'yes': 1, 'no': 0, 'y': 1, 'n': 0,
        'fire': 1, 'non-fire': 0, 'nonfire': 0, 'non_fire': 0, 'non fire': 0
    }

    def map_val(orig, num_val):
        if pd.isna(orig) and pd.isna(num_val):
            return pd.NA
        if not pd.isna(num_val):
            iv = int(num_val)
            if iv in (0, 1):
                return iv
        s_orig = str(orig).strip().lower()
        if s_orig in mapping:
            return mapping[s_orig]
        return pd.NA

    out = [map_val(o, n) for o, n in zip(s.values, num.values)]
    return pd.Series(out, index=s.index, dtype="Int64")

def is_low_cardinality_numeric(df: pd.DataFrame, col: str, threshold: int = 20) -> bool:
    if col not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    uniq_vals = pd.Series(df[col].dropna().unique())
    return len(uniq_vals) <= threshold

# ---------------------------------------------------------------------------
# EDA Helper Functions - Comprehensive Analysis
# ---------------------------------------------------------------------------
import seaborn as sns

def compute_skewness_kurtosis(series: pd.Series) -> dict:
    """Compute skewness, kurtosis and interpret distribution shape."""
    arr = pd.to_numeric(series, errors='coerce').dropna()
    if len(arr) < 3:
        return {"skewness": np.nan, "kurtosis": np.nan, "interpretation": "Insufficient data"}
    
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))
    
    # Interpret skewness
    if abs(skew) < 0.5:
        skew_interp = "approximately symmetric"
    elif skew > 0:
        skew_interp = f"positively skewed (right-tailed, skew={skew:.2f})"
    else:
        skew_interp = f"negatively skewed (left-tailed, skew={skew:.2f})"
    
    # Interpret kurtosis (excess kurtosis, normal=0)
    if abs(kurt) < 1:
        kurt_interp = "mesokurtic (normal-like tails)"
    elif kurt > 0:
        kurt_interp = f"leptokurtic (heavy tails, kurt={kurt:.2f})"
    else:
        kurt_interp = f"platykurtic (light tails, kurt={kurt:.2f})"
    
    return {
        "skewness": skew, 
        "kurtosis": kurt, 
        "skew_interpretation": skew_interp,
        "kurt_interpretation": kurt_interp
    }

def detect_outliers_summary(series: pd.Series, method: str = 'iqr') -> dict:
    """Detect outliers and return summary statistics."""
    arr = pd.to_numeric(series, errors='coerce').dropna().astype(np.float64)
    if len(arr) < 4:
        return {"outlier_count": 0, "outlier_pct": 0, "lower_bound": np.nan, "upper_bound": np.nan}
    
    if method == 'iqr':
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    elif method == 'zscore':
        mean, std = np.mean(arr), np.std(arr)
        lower = mean - 3 * std
        upper = mean + 3 * std
    else:
        lower, upper = np.percentile(arr, [1, 99])
    
    outliers = (arr < lower) | (arr > upper)
    return {
        "outlier_count": int(outliers.sum()),
        "outlier_pct": 100 * outliers.sum() / len(arr),
        "lower_bound": float(lower),
        "upper_bound": float(upper)
    }

def plot_distribution_analysis(df: pd.DataFrame, col: str, target_col: str = 'fire', 
                               color: str = "#3b82f6", figsize: tuple = (12, 4)):
    """Create comprehensive distribution plots for a numeric column."""
    arr = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(arr) < 10:
        return None
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # 1. Histogram with KDE
    axes[0].hist(arr, bins=50, density=True, alpha=0.7, color=color, edgecolor='white')
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr)
        x_range = np.linspace(arr.min(), arr.max(), 200)
        axes[0].plot(x_range, kde(x_range), color='darkred', linewidth=2, label='KDE')
        axes[0].legend()
    except Exception:
        pass
    axes[0].set_title(f'Distribution of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Density')
    
    # 2. Box plot
    axes[1].boxplot(arr, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6))
    axes[1].set_title(f'Box Plot - {col}')
    axes[1].set_ylabel(col)
    
    # 3. QQ Plot
    qq_data = stats.probplot(arr, dist="norm")
    osm, osr = qq_data[0]
    slope, intercept = qq_data[1][0], qq_data[1][1]
    axes[2].scatter(osm, osr, s=10, alpha=0.6, color=color)
    axes[2].plot(osm, slope * osm + intercept, color='red', linewidth=1.5)
    axes[2].set_title(f'Q-Q Plot - {col}')
    axes[2].set_xlabel('Theoretical Quantiles')
    axes[2].set_ylabel('Sample Quantiles')
    
    # 4. Distribution by target (if binary target exists)
    if target_col in df.columns:
        target = df[target_col].dropna()
        common_idx = arr.index.intersection(target.index)
        if len(common_idx) > 10:
            df_plot = pd.DataFrame({col: arr.loc[common_idx], target_col: target.loc[common_idx].astype(str)})
            for label, group in df_plot.groupby(target_col):
                axes[3].hist(group[col], bins=30, alpha=0.5, label=f'{target_col}={label}', density=True)
            axes[3].legend()
            axes[3].set_title(f'{col} by {target_col}')
            axes[3].set_xlabel(col)
        else:
            axes[3].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axes[3].set_title(f'{col} by {target_col}')
    else:
        axes[3].text(0.5, 0.5, 'No target column', ha='center', va='center')
    
    plt.tight_layout()
    return fig

def plot_bivariate_numeric(df: pd.DataFrame, x_col: str, y_col: str, 
                           hue_col: str = None, sample_n: int = 5000, figsize: tuple = (10, 4)):
    """Create scatter plot and regression line for two numeric variables."""
    x = pd.to_numeric(df[x_col], errors='coerce')
    y = pd.to_numeric(df[y_col], errors='coerce')
    
    valid_idx = x.notna() & y.notna()
    if valid_idx.sum() < 10:
        return None
    
    df_plot = pd.DataFrame({x_col: x[valid_idx], y_col: y[valid_idx]})
    if hue_col and hue_col in df.columns:
        df_plot[hue_col] = df.loc[valid_idx.index[valid_idx], hue_col].values
    
    if len(df_plot) > sample_n:
        df_plot = df_plot.sample(sample_n, random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    if hue_col and hue_col in df_plot.columns:
        for label in df_plot[hue_col].unique():
            mask = df_plot[hue_col] == label
            color = 'red' if str(label) == '1' else 'blue'
            axes[0].scatter(df_plot.loc[mask, x_col], df_plot.loc[mask, y_col], 
                           alpha=0.4, s=8, label=f'{hue_col}={label}', c=color)
        axes[0].legend()
    else:
        axes[0].scatter(df_plot[x_col], df_plot[y_col], alpha=0.4, s=8, c='#3b82f6')
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_title(f'{y_col} vs {x_col}')
    
    # Add trend line
    try:
        z = np.polyfit(df_plot[x_col], df_plot[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_plot[x_col].min(), df_plot[x_col].max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
    except Exception:
        pass
    
    # Hexbin for density
    hb = axes[1].hexbin(df_plot[x_col], df_plot[y_col], gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)
    axes[1].set_title(f'Density: {y_col} vs {x_col}')
    plt.colorbar(hb, ax=axes[1], label='Count')
    
    plt.tight_layout()
    return fig

def plot_categorical_vs_target(df: pd.DataFrame, cat_col: str, target_col: str = 'fire',
                               max_categories: int = 15, figsize: tuple = (12, 4)):
    """Create box plot and violin plot for categorical vs numeric target."""
    if cat_col not in df.columns or target_col not in df.columns:
        return None
    
    df_plot = df[[cat_col, target_col]].dropna().copy()
    df_plot[target_col] = pd.to_numeric(df_plot[target_col], errors='coerce')
    df_plot = df_plot.dropna()
    
    if len(df_plot) < 10:
        return None
    
    # Limit categories
    top_cats = df_plot[cat_col].value_counts().head(max_categories).index
    df_plot = df_plot[df_plot[cat_col].isin(top_cats)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    categories = sorted(df_plot[cat_col].unique())
    data_by_cat = [df_plot[df_plot[cat_col] == c][target_col].values for c in categories]
    bp = axes[0].boxplot(data_by_cat, labels=categories, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3b82f6')
        patch.set_alpha(0.6)
    axes[0].set_xlabel(cat_col)
    axes[0].set_ylabel(target_col)
    axes[0].set_title(f'{target_col} by {cat_col}')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Violin plot
    parts = axes[1].violinplot(data_by_cat, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#10b981')
        pc.set_alpha(0.6)
    axes[1].set_xticks(range(1, len(categories) + 1))
    axes[1].set_xticklabels(categories, rotation=45)
    axes[1].set_xlabel(cat_col)
    axes[1].set_ylabel(target_col)
    axes[1].set_title(f'{target_col} Distribution by {cat_col}')
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, cols: list = None, figsize: tuple = (12, 10),
                             method: str = 'pearson'):
    """Create correlation heatmap for numeric columns."""
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return None
    
    # Limit to reasonable number of columns
    if len(cols) > 25:
        cols = cols[:25]
    
    corr_matrix = df[cols].corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, 
                annot=True if len(cols) <= 12 else False, fmt='.2f',
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f'Correlation Heatmap ({method.capitalize()})')
    plt.tight_layout()
    return fig

def identify_multicollinearity(df: pd.DataFrame, cols: list = None, threshold: float = 0.8):
    """Identify highly correlated feature pairs."""
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return []
    
    corr_matrix = df[cols].corr().abs()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)

def plot_pairplot(df: pd.DataFrame, cols: list, hue_col: str = None, 
                  sample_n: int = 1000, figsize_per_plot: float = 2.5):
    """Create pair plot for selected columns."""
    if len(cols) < 2 or len(cols) > 6:
        return None
    
    df_plot = df[cols].copy()
    if hue_col and hue_col in df.columns:
        df_plot[hue_col] = df[hue_col].astype(str)
    
    df_plot = df_plot.dropna()
    if len(df_plot) > sample_n:
        df_plot = df_plot.sample(sample_n, random_state=42)
    
    if len(df_plot) < 10:
        return None
    
    if hue_col and hue_col in df_plot.columns:
        g = sns.pairplot(df_plot, hue=hue_col, diag_kind='kde', 
                        plot_kws={'alpha': 0.5, 's': 15},
                        palette={'0': 'blue', '1': 'red'})
    else:
        g = sns.pairplot(df_plot, diag_kind='kde', 
                        plot_kws={'alpha': 0.5, 's': 15, 'color': '#3b82f6'})
    
    g.fig.suptitle('Pair Plot Analysis', y=1.02)
    return g.fig

def generate_statistical_summary(df: pd.DataFrame, col: str, target_col: str = 'fire') -> str:
    """Generate text interpretation of statistical findings."""
    arr = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(arr) < 10:
        return "Insufficient data for analysis."
    
    summary_parts = []
    
    # Distribution shape
    shape = compute_skewness_kurtosis(arr)
    summary_parts.append(f"**Distribution Shape**: {col} is {shape['skew_interpretation']} and {shape['kurt_interpretation']}.")
    
    # Outliers
    outliers = detect_outliers_summary(arr, 'iqr')
    if outliers['outlier_pct'] > 5:
        summary_parts.append(f"**Outliers**: {outliers['outlier_count']:,} values ({outliers['outlier_pct']:.1f}%) detected outside IQR bounds [{outliers['lower_bound']:.2f}, {outliers['upper_bound']:.2f}]. Consider winsorization or transformation.")
    else:
        summary_parts.append(f"**Outliers**: Low outlier presence ({outliers['outlier_pct']:.1f}%).")
    
    # Relationship with target
    if target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors='coerce')
        common = arr.index.intersection(target.dropna().index)
        if len(common) > 10:
            corr = arr.loc[common].corr(target.loc[common])
            if abs(corr) > 0.3:
                direction = "positive" if corr > 0 else "negative"
                summary_parts.append(f"**Target Correlation**: {direction} correlation with {target_col} (r={corr:.3f}), suggesting this feature may be predictive.")
            else:
                summary_parts.append(f"**Target Correlation**: Weak correlation with {target_col} (r={corr:.3f}).")
    
    # Normality test
    if len(arr) >= 20 and len(arr) <= 5000:
        try:
            stat, p_value = stats.shapiro(arr.sample(min(5000, len(arr)), random_state=42))
            if p_value < 0.05:
                summary_parts.append(f"**Normality**: Data is NOT normally distributed (Shapiro-Wilk p={p_value:.4f}). Consider transformations for parametric methods.")
            else:
                summary_parts.append(f"**Normality**: Data appears approximately normal (Shapiro-Wilk p={p_value:.4f}).")
        except Exception:
            pass
    
    return " ".join(summary_parts)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
AVAILABLE_PAGES = [
    "Climate", "Soil", "Land Cover", "Fire", "Fire + Environment",
    "Merged Dataset", "Cleaned Outputs", "Outliers"
]
if rxr is not None:
    AVAILABLE_PAGES.insert(2, "Elevation")
    AVAILABLE_PAGES.append("Fire Overlay")
AVAILABLE_PAGES.insert(AVAILABLE_PAGES.index("Fire") + 1, "Fire Classification")

page = st.sidebar.radio("Dataset", AVAILABLE_PAGES)

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Panel", value=False)

# ---------------------------------------------------------------------------
# Page: Climate
# ---------------------------------------------------------------------------
if page == "Climate":
    ds = load_climate()
    if ds is None:
        st.error("climate_2024.nc not found in data folder.")
        st.stop()
    var = st.selectbox("Variable", list(ds.data_vars), index=0)
    times = list(ds.time.dt.strftime("%Y-%m").values) if "time" in ds.dims else []
    if times:
        idx = st.slider("Month", 0, len(times) - 1, 0)
        st.caption(f"Selected: {times[idx]}")
        da = ds[var].isel(time=idx)
    else:
        da = ds[var]
    arr = np.asarray(da.values, dtype=float)
    if hasattr(da, "rio"):
        nd = da.rio.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
    for s in SENTINELS:
        arr = np.where(arr == s, np.nan, arr)

    with st.expander("Outlier & Transform Options"):
        c1, c2, c3 = st.columns(3)
        apply_iqr = c1.checkbox("Cap IQR outliers", False)
        iqr_factor = c2.slider("IQR factor", 0.5, 3.0, 1.5, 0.5)
        log_prec = c3.checkbox("log1p transform (prec only)", False)

    arr_plot = arr.copy()
    out_count = 0
    lo_thr = hi_thr = None
    if apply_iqr:
        arr_plot, out_count, lo_thr, hi_thr = cap_iqr(arr_plot, factor=iqr_factor)
    if log_prec and var.startswith("prec"):
        arr_plot = np.log1p(np.where(arr_plot < 0, 0, arr_plot))
        st.caption("Applied log1p transform to precipitation.")
    if apply_iqr and out_count:
        pct = 100.0 * out_count / max(1, np.isfinite(arr).sum())
        st.caption(f"IQR capping: {out_count} pixels ({pct:.2f}%) clipped to [{lo_thr:.3g}, {hi_thr:.3g}]")

    st.subheader("Summary stats")
    finite = np.isfinite(arr_plot)
    if finite.any():
        st.table(pd.DataFrame({
            "metric": ["min", "mean", "max", "std"],
            "value": [float(np.nanmin(arr_plot)), float(np.nanmean(arr_plot)),
                      float(np.nanmax(arr_plot)), float(np.nanstd(arr_plot))]
        }))
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(arr_plot[finite].ravel(), bins=50, color="#3b82f6", alpha=0.85)
            ax.set_title(f"Histogram — {var}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.boxplot(arr_plot[finite].ravel(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#3b82f6", alpha=0.6))
            ax.set_title(f"Boxplot — {var}")
            ax.set_ylabel("Value")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    preview_array(arr_plot, f"{var} map", categorical=False)
    
    # ==========================================================================
    # CLIMATE COMPREHENSIVE EDA
    # ==========================================================================
    st.markdown("---")
    st.header("📊 Climate Data Analysis")
    
    with st.expander("📈 Distribution & Statistical Analysis", expanded=True):
        flat_data = arr_plot[finite].ravel()
        
        if len(flat_data) > 0:
            # Distribution metrics
            clim_col1, clim_col2, clim_col3, clim_col4 = st.columns(4)
            
            skew_val = float(stats.skew(flat_data))
            kurt_val = float(stats.kurtosis(flat_data))
            
            clim_col1.metric("Skewness", f"{skew_val:.3f}")
            clim_col2.metric("Kurtosis", f"{kurt_val:.3f}")
            
            q1, q3 = np.percentile(flat_data, [25, 75])
            iqr = q3 - q1
            outlier_mask = (flat_data < q1 - 1.5*iqr) | (flat_data > q3 + 1.5*iqr)
            outlier_pct = 100 * outlier_mask.sum() / len(flat_data)
            
            clim_col3.metric("Outliers", f"{outlier_mask.sum():,}")
            clim_col4.metric("Outlier %", f"{outlier_pct:.1f}%")
            
            # Interpretation
            if abs(skew_val) < 0.5:
                skew_interp = "approximately symmetric"
            elif skew_val > 0:
                skew_interp = "positively skewed (right-tailed)"
            else:
                skew_interp = "negatively skewed (left-tailed)"
            
            st.info(f"**Distribution Interpretation**: {var} data is {skew_interp} with {'heavy' if kurt_val > 1 else 'normal'} tails. "
                   f"{'Consider log transformation for modeling.' if skew_val > 1 and var.startswith('prec') else ''}")
            
            # QQ Plot and KDE
            qq_col1, qq_col2 = st.columns(2)
            
            with qq_col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                qq_data = stats.probplot(flat_data, dist="norm")
                osm, osr = qq_data[0]
                slope, intercept = qq_data[1][0], qq_data[1][1]
                ax.scatter(osm, osr, s=10, alpha=0.5, color="#3b82f6")
                ax.plot(osm, slope * osm + intercept, color='red', linewidth=1.5)
                ax.set_title(f'Q-Q Plot — {var}')
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            with qq_col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(flat_data, bins=50, density=True, alpha=0.7, color="#3b82f6", edgecolor='white')
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(flat_data)
                    x_range = np.linspace(flat_data.min(), flat_data.max(), 200)
                    ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
                    ax.legend()
                except Exception:
                    pass
                ax.set_title(f'Distribution with KDE — {var}')
                ax.set_xlabel(var)
                ax.set_ylabel('Density')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    
    # Multi-variable comparison if multiple climate variables exist
    if len(ds.data_vars) > 1:
        with st.expander("🔗 Multi-Variable Comparison", expanded=False):
            st.subheader("Compare Climate Variables")
            
            compare_vars = st.multiselect(
                "Select variables to compare",
                options=list(ds.data_vars),
                default=list(ds.data_vars)[:min(3, len(ds.data_vars))],
                key="climate_compare"
            )
            
            if compare_vars and len(compare_vars) >= 2:
                # Extract data for comparison
                compare_data = {}
                for v in compare_vars:
                    if times:
                        v_arr = np.asarray(ds[v].isel(time=idx).values, dtype=float)
                    else:
                        v_arr = np.asarray(ds[v].values, dtype=float)
                    v_finite = np.isfinite(v_arr)
                    compare_data[v] = v_arr[v_finite].ravel()
                
                # Box plot comparison
                fig, ax = plt.subplots(figsize=(10, 5))
                data_list = [compare_data[v] for v in compare_vars]
                bp = ax.boxplot(data_list, labels=compare_vars, patch_artist=True)
                colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#6366f1']
                for patch, color in zip(bp['boxes'], colors[:len(compare_vars)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax.set_ylabel('Value')
                ax.set_title('Climate Variable Comparison')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Correlation if same shape
                if all(len(compare_data[v]) == len(compare_data[compare_vars[0]]) for v in compare_vars):
                    st.subheader("Variable Correlations")
                    corr_df = pd.DataFrame(compare_data)
                    corr_matrix = corr_df.corr()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                               center=0, ax=ax, square=True)
                    ax.set_title('Climate Variable Correlations')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Scatter & Pair Plots
            st.subheader("Scatter & Pair Plots")
            sp_tab1, sp_tab2 = st.tabs(["Scatter Plot", "Pair Plot"])
            
            with sp_tab1:
                if len(compare_vars) >= 2:
                    c_x = st.selectbox("X Axis", compare_vars, index=0, key="clim_scatter_x")
                    c_y = st.selectbox("Y Axis", compare_vars, index=1, key="clim_scatter_y")
                    
                    if c_x and c_y:
                        # Get data
                        if times:
                            x_arr = np.asarray(ds[c_x].isel(time=idx).values, dtype=float).ravel()
                            y_arr = np.asarray(ds[c_y].isel(time=idx).values, dtype=float).ravel()
                        else:
                            x_arr = np.asarray(ds[c_x].values, dtype=float).ravel()
                            y_arr = np.asarray(ds[c_y].values, dtype=float).ravel()
                        
                        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
                        if mask.sum() > 0:
                            x_valid = x_arr[mask]
                            y_valid = y_arr[mask]
                            
                            # Sample if too large
                            if len(x_valid) > 5000:
                                idxs = np.random.choice(len(x_valid), 5000, replace=False)
                                x_plot = x_valid[idxs]
                                y_plot = y_valid[idxs]
                            else:
                                x_plot = x_valid
                                y_plot = y_valid
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(x_plot, y_plot, alpha=0.5, s=10, c='#3b82f6')
                            
                            # Trend line
                            try:
                                z = np.polyfit(x_plot, y_plot, 1)
                                p = np.poly1d(z)
                                x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
                                ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend (r={np.corrcoef(x_valid, y_valid)[0,1]:.3f})')
                                ax.legend()
                            except Exception:
                                pass
                                
                            ax.set_xlabel(c_x)
                            ax.set_ylabel(c_y)
                            ax.set_title(f'{c_x} vs {c_y}')
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.warning("No valid overlapping data points.")
                else:
                    st.info("Select at least 2 variables above to enable scatter plots.")
            
            with sp_tab2:
                if len(compare_vars) >= 2:
                    if st.button("Generate Pair Plot", key="clim_pairplot_btn"):
                        with st.spinner("Generating pair plot..."):
                            # Build DF
                            pp_data = {}
                            for v in compare_vars:
                                if times:
                                    arr = np.asarray(ds[v].isel(time=idx).values, dtype=float).ravel()
                                else:
                                    arr = np.asarray(ds[v].values, dtype=float).ravel()
                                pp_data[v] = arr
                            
                            pp_df = pd.DataFrame(pp_data).dropna()
                            
                            if len(pp_df) > 2000:
                                pp_df = pp_df.sample(2000, random_state=42)
                            
                            if not pp_df.empty:
                                fig = sns.pairplot(pp_df, diag_kind='kde', 
                                                  plot_kws={'alpha': 0.5, 's': 10, 'color': '#3b82f6'})
                                st.pyplot(fig.fig)
                            else:
                                st.warning("Not enough valid data for pair plot.")
                else:
                    st.info("Select at least 2 variables above to enable pair plots.")

# ---------------------------------------------------------------------------
# Page: Soil
# ---------------------------------------------------------------------------
elif page == "Soil":
    ds = load_soil()
    if ds is None:
        st.error("soil_id_with_attrs_aoi.nc not found in data folder.")
        st.stop()
    if "soil_id_grid" in ds:
        grid = ds["soil_id_grid"].values.astype(float)
        grid[grid == -2147483648] = np.nan
        preview_array(grid, "Soil ID grid (SMU)", categorical=True)

    st.subheader("Attribute table")
    
    # Only show the 19 specified soil features (after removing highly correlated pairs)
    # Dropped: bulk (r=0.91 with ph_water), ref_bulk (r=0.90 with silt), teb (r=0.99 with cec_eff)
    SOIL_FEATURES_TO_SHOW = {
        'coarse', 'sand', 'silt', 'clay', 'texture_usda', 'texture_soter',
        'org_carbon', 'ph_water', 'total_n', 'cn_ratio',
        'cec_soil', 'cec_clay', 'cec_eff', 'bsat', 'alum_sat', 'esp',
        'tcarbon_eq', 'gypsum', 'elec_cond'
    }
    
    all_attr_vars = [str(v) for v in ds.data_vars if v != "soil_id_grid"]
    # Filter to only show specified features (case-insensitive)
    attr_vars = [v for v in all_attr_vars if v.lower() in SOIL_FEATURES_TO_SHOW]
    
    if not attr_vars:
        st.warning("No matching soil attributes found. Showing all available.")
        attr_vars = all_attr_vars
    
    attr = st.selectbox("Attribute", sorted(attr_vars), index=0)
    series = ds[attr].to_pandas()

    with st.expander("Outlier Options"):
        sc1, sc2 = st.columns(2)
        soil_cap = sc1.checkbox("Cap IQR outliers", False)
        soil_factor = sc2.slider("IQR factor", 0.5, 3.0, 1.5, 0.5)

    arr_vals = pd.to_numeric(series, errors="coerce").to_numpy()
    arr_show = arr_vals.copy()
    soil_out = 0
    slo = shi = None
    if soil_cap:
        arr_show, soil_out, slo, shi = cap_iqr(arr_show, factor=soil_factor)
    st.write(series.describe(include='all'))
    if soil_cap and soil_out:
        pct_s = 100.0 * soil_out / max(1, np.isfinite(arr_vals).sum())
        st.caption(f"IQR capping: {soil_out} values ({pct_s:.2f}%) clipped to [{slo:.3g}, {shi:.3g}]")

    if pd.api.types.is_numeric_dtype(series):
        arr_clean = arr_show[np.isfinite(arr_show)]
        if arr_clean.size:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(arr_clean, bins=50, color="#10b981", alpha=0.85)
                ax.set_title(f"Histogram — {attr}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.boxplot(arr_clean, vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#10b981", alpha=0.6))
                ax.set_title(f"Boxplot — {attr}")
                ax.set_ylabel("Value")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    
    # ==========================================================================
    # SOIL COMPREHENSIVE EDA
    # ==========================================================================
    st.markdown("---")
    st.header("📊 Soil Data Analysis")
    
    with st.expander("📈 Distribution & Statistical Analysis", expanded=True):
        if pd.api.types.is_numeric_dtype(series):
            arr_clean = arr_show[np.isfinite(arr_show)]
            if arr_clean.size > 0:
                # Distribution metrics
                soil_col1, soil_col2, soil_col3, soil_col4 = st.columns(4)
                
                skew_val = float(stats.skew(arr_clean))
                kurt_val = float(stats.kurtosis(arr_clean))
                
                soil_col1.metric("Skewness", f"{skew_val:.3f}")
                soil_col2.metric("Kurtosis", f"{kurt_val:.3f}")
                
                q1, q3 = np.percentile(arr_clean, [25, 75])
                iqr = q3 - q1
                outlier_mask = (arr_clean < q1 - 1.5*iqr) | (arr_clean > q3 + 1.5*iqr)
                outlier_pct = 100 * outlier_mask.sum() / len(arr_clean)
                
                soil_col3.metric("Outliers", f"{outlier_mask.sum():,}")
                soil_col4.metric("Outlier %", f"{outlier_pct:.1f}%")
                
                # Interpretation
                if abs(skew_val) < 0.5:
                    skew_interp = "approximately symmetric"
                elif skew_val > 0:
                    skew_interp = "positively skewed (right-tailed)"
                else:
                    skew_interp = "negatively skewed (left-tailed)"
                
                st.info(f"**Distribution Interpretation**: {attr} is {skew_interp}. "
                       f"{'High outlier presence may require winsorization.' if outlier_pct > 5 else 'Outlier presence is acceptable.'}")
                
                # QQ Plot
                qq_col1, qq_col2 = st.columns(2)
                
                with qq_col1:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    qq_data = stats.probplot(arr_clean, dist="norm")
                    osm, osr = qq_data[0]
                    slope, intercept = qq_data[1][0], qq_data[1][1]
                    ax.scatter(osm, osr, s=10, alpha=0.5, color="#10b981")
                    ax.plot(osm, slope * osm + intercept, color='red', linewidth=1.5)
                    ax.set_title(f'Q-Q Plot — {attr}')
                    ax.set_xlabel('Theoretical Quantiles')
                    ax.set_ylabel('Sample Quantiles')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                with qq_col2:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.hist(arr_clean, bins=50, density=True, alpha=0.7, color="#10b981", edgecolor='white')
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(arr_clean)
                        x_range = np.linspace(arr_clean.min(), arr_clean.max(), 200)
                        ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
                        ax.legend()
                    except Exception:
                        pass
                    ax.set_title(f'Distribution with KDE — {attr}')
                    ax.set_xlabel(attr)
                    ax.set_ylabel('Density')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
    
    # Multi-attribute comparison
    if len(attr_vars) > 1:
        with st.expander("🔗 Multi-Attribute Correlation Analysis", expanded=False):
            st.subheader("Soil Attribute Correlations")
            
            compare_attrs = st.multiselect(
                "Select attributes to compare",
                options=sorted(attr_vars),
                default=sorted(attr_vars)[:min(6, len(attr_vars))],
                key="soil_compare"
            )
            
            if compare_attrs and len(compare_attrs) >= 2:
                # Build dataframe for correlations
                soil_df = pd.DataFrame()
                for a in compare_attrs:
                    try:
                        soil_df[a] = pd.to_numeric(ds[a].to_pandas(), errors='coerce')
                    except Exception:
                        pass
                
                if len(soil_df.columns) >= 2:
                    # Correlation heatmap
                    corr_matrix = soil_df.corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                               cmap='RdYlGn', center=0, ax=ax, square=True,
                               linewidths=0.5, cbar_kws={"shrink": 0.8})
                    ax.set_title('Soil Attribute Correlations')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Identify high correlations
                    high_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) >= 0.9:
                                high_corr.append({
                                    'Attribute 1': corr_matrix.columns[i],
                                    'Attribute 2': corr_matrix.columns[j],
                                    'Correlation': corr_matrix.iloc[i, j]
                                })
                    
                    if high_corr:
                        st.warning(f"⚠️ Found {len(high_corr)} highly correlated attribute pairs (|r| ≥ 0.7):")
                        st.dataframe(pd.DataFrame(high_corr).round(3), use_container_width=True)
                    
                    # Scatter & Pair Plots
                    st.subheader("Scatter & Pair Plots")
                    sp_tab1, sp_tab2 = st.tabs(["Scatter Plot", "Pair Plot"])
                    
                    with sp_tab1:
                        if len(compare_attrs) >= 2:
                            s_x = st.selectbox("X Axis", compare_attrs, index=0, key="soil_scatter_x")
                            s_y = st.selectbox("Y Axis", compare_attrs, index=1, key="soil_scatter_y")
                            
                            if s_x and s_y and s_x in soil_df.columns and s_y in soil_df.columns:
                                x_vals = soil_df[s_x].dropna()
                                y_vals = soil_df[s_y].dropna()
                                common_idx = x_vals.index.intersection(y_vals.index)
                                
                                if len(common_idx) > 0:
                                    # Sample if too large
                                    if len(common_idx) > 5000:
                                        sample_idx = np.random.choice(common_idx, 5000, replace=False)
                                    else:
                                        sample_idx = common_idx
                                    
                                    x_plot = x_vals.loc[sample_idx]
                                    y_plot = y_vals.loc[sample_idx]
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.scatter(x_plot, y_plot, alpha=0.5, s=10, c='#10b981')
                                    
                                    # Trend line
                                    try:
                                        z = np.polyfit(x_plot, y_plot, 1)
                                        p = np.poly1d(z)
                                        x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
                                        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend (r={x_plot.corr(y_plot):.3f})')
                                        ax.legend()
                                    except Exception:
                                        pass
                                        
                                    ax.set_xlabel(s_x)
                                    ax.set_ylabel(s_y)
                                    ax.set_title(f'{s_x} vs {s_y}')
                                    fig.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.warning("No overlapping data for selected attributes.")
                        else:
                            st.info("Select at least 2 attributes above.")
                    
                    with sp_tab2:
                        if len(compare_attrs) <= 6:
                            if st.button("Generate Pair Plot", key="soil_pairplot_btn"):
                                with st.spinner("Generating pair plot..."):
                                    sample_df = soil_df.dropna().sample(min(1000, len(soil_df)), random_state=42)
                                    if len(sample_df) > 10:
                                        g = sns.pairplot(sample_df, diag_kind='kde', 
                                                       plot_kws={'alpha': 0.5, 's': 15, 'color': '#10b981'})
                                        g.fig.suptitle('Soil Attributes Pair Plot', y=1.02)
                                        st.pyplot(g.fig)
                                        plt.close(g.fig)
                        else:
                            st.info("Select 6 or fewer attributes for pair plot.")

# ---------------------------------------------------------------------------
# Page: Elevation
# ---------------------------------------------------------------------------
elif page == "Elevation":
    if rxr is None:
        st.error("Elevation disabled: rasterio/rioxarray not installed.")
        st.stop()
    da = load_elevation()
    if da is None:
        st.error("elevation_clipped.tif not found.")
        st.stop()
    arr = np.asarray(da.values, dtype=float)
    nd = da.rio.nodata
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    
    # Statistics summary
    finite = np.isfinite(arr)
    if finite.any():
        flat = arr[finite].ravel()
        
        st.subheader("Elevation Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min (m)", f"{np.nanmin(flat):.1f}")
        col2.metric("Mean (m)", f"{np.nanmean(flat):.1f}")
        col3.metric("Max (m)", f"{np.nanmax(flat):.1f}")
        col4.metric("Std (m)", f"{np.nanstd(flat):.1f}")
        
        # Histogram of elevation values
        st.subheader("Elevation Distribution")
        hist_col1, hist_col2 = st.columns(2)
        
        with hist_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(flat, bins=50, color="#3b82f6", alpha=0.85, edgecolor='white')
            ax.set_xlabel("Elevation (m)")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Elevation Values")
            ax.axvline(np.nanmean(flat), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.nanmean(flat):.1f}m')
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with hist_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(flat, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#3b82f6", alpha=0.6))
            ax.set_ylabel("Elevation (m)")
            ax.set_title("Boxplot of Elevation Values")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Percentile information
        with st.expander("Elevation Percentiles"):
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            pct_values = np.nanpercentile(flat, percentiles)
            pct_df = pd.DataFrame({
                "Percentile": [f"P{p}" for p in percentiles],
                "Elevation (m)": [f"{v:.1f}" for v in pct_values]
            })
            st.table(pct_df)
    
    # Spatial preview
    preview_array(arr, "Elevation (m)", categorical=False)

# ---------------------------------------------------------------------------
# Page: Land Cover
# ---------------------------------------------------------------------------
elif page == "Land Cover":
    gdf = load_landcover()
    if gdf is None:
        st.error("landcover_clipped.shp not found.")
        st.stop()
    st.write("Records:", len(gdf))
    col = None
    for c in ["class", "CLASS", "code", "CODE", "LC_TYPE", "landcover",
              "land_class", "VALUE", "value", "gridcode"]:
        if c in gdf.columns:
            col = c
            break
    if col:
        counts = gdf[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        st.dataframe(counts)
        
        # Bar chart of land cover class counts
        st.subheader("Land Cover Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Limit to top 15 classes if there are many
        plot_counts = counts.head(15) if len(counts) > 15 else counts
        
        bars = ax.barh(plot_counts[col].astype(str), plot_counts["count"], color="#10b981", alpha=0.85)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_title(f"Land Cover Classes Distribution (n={len(gdf):,})")
        
        # Add count labels on bars
        for bar, count in zip(bars, plot_counts["count"]):
            ax.text(bar.get_width() + max(plot_counts["count"]) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=9)
        
        ax.invert_yaxis()  # Largest at top
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        if len(counts) > 15:
            st.caption(f"Showing top 15 of {len(counts)} classes.")
    
    # Map visualization with sampling for large datasets
    try:
        g = gdf.to_crs("EPSG:4326").copy()
        
        # Filter out invalid/empty geometries
        g = g[~g.geometry.is_empty & g.geometry.is_valid]
        
        if len(g) == 0:
            st.warning("No valid geometries found for mapping.")
        else:
            # Sample if too many polygons
            max_map_points = 5000
            if len(g) > max_map_points:
                st.info(f"Sampling {max_map_points:,} of {len(g):,} polygons for map display.")
                g = g.sample(max_map_points, random_state=42)
            
            # Extract centroids with warning suppression
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cents = g.geometry.centroid
            
            mdf = pd.DataFrame({
                "latitude": cents.y.values, 
                "longitude": cents.x.values
            }).dropna()
            
            if len(mdf) > 0:
                st.map(mdf)
            else:
                st.warning("No valid centroids could be extracted.")
    except Exception as e:
        st.info(f"Could not map polygons: {str(e)[:100]}")

# ---------------------------------------------------------------------------
# Page: Fire (raw type 2)
# ---------------------------------------------------------------------------
elif page == "Fire":
    st.header("Fire (VIIRS 2024) — Type 2 points")

    @st.cache_data
    def load_fire_df() -> pd.DataFrame:
        files = [
            FIRE_DIR / "viirs-jpss1_2024_Algeria_type2.csv",
            FIRE_DIR / "viirs-jpss1_2024_Tunisia_type2.csv",
            FIRE_DIR / "viirs-jpss1_2024_Algeria.csv",
            FIRE_DIR / "viirs-jpss1_2024_Tunisia.csv",
        ]
        dfs = []
        for fp in files:
            if not fp.exists():
                continue
            try:
                df = pd.read_csv(fp, low_memory=False)
                df["country"] = "Algeria" if "Algeria" in fp.name else "Tunisia"
                type_col = next((c for c in df.columns if c.lower() == "type"), None)
                if type_col:
                    df = df[df[type_col] == 2].copy()
                # Latitude/longitude coercion
                if {"latitude", "longitude"}.issubset(df.columns):
                    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                else:
                    latc = df.filter(regex="lat", axis=1).columns
                    lonc = df.filter(regex="lon|lng", axis=1).columns
                    if len(latc) and len(lonc):
                        df["latitude"] = pd.to_numeric(df[latc[0]], errors="coerce")
                        df["longitude"] = pd.to_numeric(df[lonc[0]], errors="coerce")
                df = df.dropna(subset=["latitude", "longitude"]).copy()
                if "acq_date" in df.columns:
                    if "acq_time" in df.columns:
                        def _parse_dt(row):
                            t = str(row.get("acq_time", "")).zfill(4)
                            try:
                                hh, mm = int(t[:2]), int(t[2:])
                                return pd.to_datetime(row["acq_date"]) + pd.Timedelta(hours=hh, minutes=mm)
                            except Exception:
                                return pd.to_datetime(row["acq_date"], errors="coerce")
                        df["dt"] = df.apply(_parse_dt, axis=1)
                    else:
                        df["dt"] = pd.to_datetime(df["acq_date"], errors="coerce")
                    df["month"] = df["dt"].dt.strftime("%Y-%m")
                else:
                    df["month"] = "unknown"
                dfs.append(df)
            except Exception:
                continue
        if not dfs:
            return pd.DataFrame()
        out = pd.concat(dfs, ignore_index=True)
        subset_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in out.columns]
        if subset_cols:
            out = out.drop_duplicates(subset=subset_cols)
        return out

    df = load_fire_df()
    if df.empty:
        st.error("No FIRE CSVs found.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    countries = sorted(df["country"].dropna().unique()) if "country" in df.columns else []
    sel_c = c1.multiselect("Country", countries, default=countries)
    months = sorted(df["month"].dropna().unique()) if "month" in df.columns else []
    sel_m = c2.multiselect("Month", months, default=months)
    cand = [c for c in ["frp", "bright_ti4", "bright_ti5"] if c in df.columns]
    hist_var = c3.selectbox("Histogram variable", cand, index=0) if cand else None

    df2 = df.copy()
    if sel_c:
        df2 = df2[df2["country"].isin(sel_c)]
    if sel_m:
        df2 = df2[df2["month"].isin(sel_m)]

    st.subheader("Counts")
    left, right = st.columns(2)
    with left:
        st.metric("Rows", f"{len(df2):,}")
        if "month" in df2.columns and not df2.empty:
            st.bar_chart(df2.groupby("month").size())
    with right:
        if hist_var and not df2.empty:
            vals = pd.to_numeric(df2[hist_var], errors="coerce").dropna()
            if not vals.empty:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(vals.values, bins=60, color="#ef4444", alpha=0.85)
                ax.set_title(f"Histogram — {hist_var}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Map")
    if {"latitude", "longitude"}.issubset(df2.columns) and not df2.empty:
        n_max = 20000
        mdf = df2.sample(n_max, random_state=0) if len(df2) > n_max else df2
        st.map(mdf[["latitude", "longitude"]])
    else:
        st.info("No valid latitude/longitude columns to map.")

# ---------------------------------------------------------------------------
# Page: Fire + Environment
# ---------------------------------------------------------------------------
elif page == "Fire + Environment":
    st.header("Fire points enriched with environmental sampling")

    @st.cache_data
    def load_fire_env() -> pd.DataFrame:
        parquet = FIRE_DIR / "fire_environment_2024.parquet"
        csv_alt = FIRE_DIR / "fire_environment_2024.csv"
        if parquet.exists():
            try:
                return pd.read_parquet(parquet)
            except Exception:
                pass
        if csv_alt.exists():
            try:
                return pd.read_csv(csv_alt)
            except Exception:
                pass
        return pd.DataFrame()

    fedf = load_fire_env()
    if fedf.empty:
        st.error("fire_environment_2024 parquet/csv not found. Run pipeline fire-env-merge.")
        st.stop()

    for col in ["latitude", "longitude", "prec", "tmax", "tmin", "elevation",
                "slope_deg", "aspect_deg", "frp_mean", "frp_max"]:
        if col in fedf.columns:
            fedf[col] = pd.to_numeric(fedf[col], errors="coerce")
    if "month" in fedf.columns:
        fedf["month"] = fedf["month"].astype(str)

    fl1, fl2, fl3, fl4 = st.columns(4)
    countries = sorted(fedf["country"].dropna().unique()) if "country" in fedf.columns else []
    sel_c = fl1.multiselect("Country", countries, default=countries)
    months = sorted(fedf["month"].dropna().unique()) if "month" in fedf.columns else []
    sel_m = fl2.multiselect("Month", months, default=months)
    lc_vals = sorted(fedf["landcover_label"].dropna().unique()) if "landcover_label" in fedf.columns else []
    sel_lc = fl3.multiselect("Landcover", lc_vals, default=lc_vals[:10] if len(lc_vals) > 10 else lc_vals)
    color_var = fl4.selectbox("Color by", [c for c in ["frp_mean", "prec", "tmax", "elevation", "landcover_label"]
                                           if c in fedf.columns], index=0)

    df_env = fedf.copy()
    if sel_c:
        df_env = df_env[df_env["country"].isin(sel_c)]
    if sel_m:
        df_env = df_env[df_env["month"].isin(sel_m)]
    if sel_lc and "landcover_label" in df_env.columns:
        df_env = df_env[df_env["landcover_label"].isin(sel_lc)]

    st.subheader("Overview")
    st.metric("Rows", f"{len(df_env):,}")
    if "frp_mean" in df_env.columns and len(df_env):
        st.metric("Mean FRP", f"{pd.to_numeric(df_env['frp_mean'], errors='coerce').mean():.2f}")
    if "prec" in df_env.columns and len(df_env):
        st.metric("Mean Precip", f"{pd.to_numeric(df_env['prec'], errors='coerce').mean():.2f}")

    if color_var in df_env.columns and pd.api.types.is_numeric_dtype(df_env[color_var]):
        vals = pd.to_numeric(df_env[color_var], errors="coerce").dropna()
        if not vals.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(vals.values, bins=60, color="#6366f1", alpha=0.85)
                ax.set_title(f"Histogram — {color_var}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.boxplot(vals.values, vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#6366f1", alpha=0.6))
                ax.set_title(f"Boxplot — {color_var}")
                ax.set_ylabel("Value")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            col3, col4 = st.columns(2)
            with col3:
                qq_data = stats.probplot(vals.values, dist="norm")
                fig, ax2 = plt.subplots(figsize=(5, 3))
                osm, osr = qq_data[0]
                slope, intercept = qq_data[1][0], qq_data[1][1]
                ax2.scatter(osm, osr, s=10, alpha=0.6, color="#6366f1")
                ax2.plot(osm, slope * osm + intercept, color="#374151", linewidth=1)
                ax2.set_title(f"QQ-plot — {color_var}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with col4:
                if "elevation" in df_env.columns and color_var != "elevation":
                    x_vals = pd.to_numeric(df_env["elevation"], errors="coerce").dropna()
                    y_vals = pd.to_numeric(df_env[color_var], errors="coerce").dropna()
                    common_idx = x_vals.index.intersection(y_vals.index)
                    if len(common_idx) > 1:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        sample_n = min(5000, len(common_idx))
                        sample_idx = common_idx if len(common_idx) <= sample_n else np.random.choice(common_idx, sample_n, replace=False)
                        ax.scatter(x_vals.loc[sample_idx], y_vals.loc[sample_idx], alpha=0.4, s=8, color="#6366f1")
                        ax.set_xlabel("Elevation (m)")
                        ax.set_ylabel(color_var)
                        ax.set_title(f"{color_var} vs Elevation")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Not enough data for scatter plot")
                else:
                    st.info("Scatter plot requires elevation and a different variable")

    st.subheader("Sampled Data Table")
    show_cols = [c for c in ["latitude", "longitude", "country", "month", "landcover_label",
                             "soil_id", "prec", "tmax", "tmin", "elevation",
                             "slope_deg", "aspect_deg", "frp_mean", "frp_max"] if c in df_env.columns]
    st.dataframe(df_env[show_cols].head(500))
    st.caption("Showing first 500 rows")

    csv_bytes = df_env[show_cols].to_csv(index=False).encode()
    st.download_button("Download filtered CSV", data=csv_bytes,
                       file_name="fire_environment_filtered.csv", mime="text/csv")

    st.subheader("Map")
    if {"latitude", "longitude"}.issubset(df_env.columns) and not df_env.empty:
        sample_n = 15000
        df_map = df_env.sample(sample_n, random_state=0) if len(df_env) > sample_n else df_env
        st.map(df_map[["latitude", "longitude"]])
    else:
        st.info("No points to map.")
    
    # ==========================================================================
    # FIRE + ENVIRONMENT COMPREHENSIVE EDA
    # ==========================================================================
    st.markdown("---")
    st.header("📊 Comprehensive Fire-Environment Analysis")
    
    # Identify numeric columns for analysis
    env_numeric_cols = [c for c in df_env.columns 
                        if pd.api.types.is_numeric_dtype(df_env[c]) 
                        and c not in ('latitude', 'longitude')]
    env_categorical_cols = [c for c in df_env.columns 
                            if df_env[c].dtype == 'object' or c in ('month', 'country', 'landcover_label')]
    
    with st.expander("📈 1. Distribution Analysis (Univariate)", expanded=True):
        st.markdown("""
        **Purpose**: Analyze the distribution of fire-related and environmental variables 
        to understand data characteristics and identify potential issues.
        """)
        
        dist_features = st.multiselect(
            "Select features for distribution analysis",
            options=sorted(env_numeric_cols),
            default=sorted([c for c in ['frp_mean', 'prec', 'tmax', 'elevation'] if c in env_numeric_cols])[:4],
            key="fire_env_dist"
        )
        
        if dist_features:
            for feat in dist_features:
                st.markdown(f"#### {feat}")
                
                arr = pd.to_numeric(df_env[feat], errors='coerce').dropna()
                if len(arr) > 0:
                    # Stats
                    fe_col1, fe_col2, fe_col3, fe_col4 = st.columns(4)
                    
                    skew_val = float(stats.skew(arr))
                    kurt_val = float(stats.kurtosis(arr))
                    q1, q3 = np.percentile(arr, [25, 75])
                    iqr = q3 - q1
                    outlier_mask = (arr < q1 - 1.5*iqr) | (arr > q3 + 1.5*iqr)
                    outlier_pct = 100 * outlier_mask.sum() / len(arr)
                    
                    fe_col1.metric("Skewness", f"{skew_val:.3f}")
                    fe_col2.metric("Kurtosis", f"{kurt_val:.3f}")
                    fe_col3.metric("Outliers", f"{outlier_mask.sum():,}")
                    fe_col4.metric("Outlier %", f"{outlier_pct:.1f}%")
                    
                    # Plots
                    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                    
                    # Histogram with KDE
                    axes[0].hist(arr, bins=50, density=True, alpha=0.7, color='#6366f1', edgecolor='white')
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(arr)
                        x_range = np.linspace(arr.min(), arr.max(), 200)
                        axes[0].plot(x_range, kde(x_range), color='red', linewidth=2)
                    except Exception:
                        pass
                    axes[0].set_title(f'Distribution — {feat}')
                    axes[0].set_xlabel(feat)
                    
                    # Box plot
                    axes[1].boxplot(arr, vert=True, patch_artist=True,
                                   boxprops=dict(facecolor='#6366f1', alpha=0.6))
                    axes[1].set_title(f'Box Plot — {feat}')
                    
                    # QQ plot
                    qq_data = stats.probplot(arr, dist="norm")
                    osm, osr = qq_data[0]
                    slope, intercept = qq_data[1][0], qq_data[1][1]
                    axes[2].scatter(osm, osr, s=10, alpha=0.5, color='#6366f1')
                    axes[2].plot(osm, slope * osm + intercept, color='red', linewidth=1.5)
                    axes[2].set_title(f'Q-Q Plot — {feat}')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Interpretation
                    if abs(skew_val) < 0.5:
                        skew_interp = "approximately symmetric"
                    elif skew_val > 0:
                        skew_interp = "positively skewed"
                    else:
                        skew_interp = "negatively skewed"
                    
                    st.info(f"**Interpretation**: {feat} is {skew_interp}. "
                           f"{'Consider log transformation.' if skew_val > 1 else ''} "
                           f"{'High outlier presence may need handling.' if outlier_pct > 5 else ''}")
                
                st.markdown("---")
    
    with st.expander("🔗 2. Relationship Mapping (Bivariate)", expanded=False):
        st.markdown("""
        **Purpose**: Examine relationships between environmental variables and fire intensity (FRP).
        """)
        
        # Numeric scatter plots
        if 'frp_mean' in df_env.columns:
            st.subheader("Environmental Variables vs Fire Intensity (FRP)")
            
            scatter_vars = st.multiselect(
                "Select variables to plot against FRP",
                options=[c for c in env_numeric_cols if c not in ('frp_mean', 'frp_max')],
                default=[c for c in ['prec', 'tmax', 'elevation', 'slope_deg'] if c in env_numeric_cols][:3],
                key="fire_env_scatter"
            )
            
            if scatter_vars:
                for var in scatter_vars:
                    st.markdown(f"#### {var} vs FRP")
                    
                    x = pd.to_numeric(df_env[var], errors='coerce')
                    y = pd.to_numeric(df_env['frp_mean'], errors='coerce')
                    valid = x.notna() & y.notna()
                    
                    if valid.sum() > 10:
                        x_v, y_v = x[valid], y[valid]
                        
                        # Sample for plotting
                        if len(x_v) > 5000:
                            sample_idx = np.random.choice(len(x_v), 5000, replace=False)
                            x_plot = x_v.iloc[sample_idx]
                            y_plot = y_v.iloc[sample_idx]
                        else:
                            x_plot, y_plot = x_v, y_v
                        
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Scatter
                        axes[0].scatter(x_plot, y_plot, alpha=0.4, s=8, c='#6366f1')
                        try:
                            z = np.polyfit(x_plot, y_plot, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
                            axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label='Trend')
                            axes[0].legend()
                        except Exception:
                            pass
                        axes[0].set_xlabel(var)
                        axes[0].set_ylabel('FRP Mean')
                        axes[0].set_title(f'FRP vs {var}')
                        
                        # Hexbin density
                        hb = axes[1].hexbin(x_plot, y_plot, gridsize=30, cmap='YlOrRd', mincnt=1)
                        axes[1].set_xlabel(var)
                        axes[1].set_ylabel('FRP Mean')
                        axes[1].set_title(f'Density: FRP vs {var}')
                        plt.colorbar(hb, ax=axes[1], label='Count')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Correlation
                        corr = x_v.corr(y_v)
                        st.caption(f"**Pearson Correlation**: r = {corr:.4f}")
        
        # Categorical analysis
        st.subheader("Fire Intensity by Category")
        
        if env_categorical_cols and 'frp_mean' in df_env.columns:
            cat_var = st.selectbox(
                "Select categorical variable",
                options=[c for c in env_categorical_cols if c in df_env.columns],
                key="fire_env_cat"
            )
            
            if cat_var:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                top_cats = df_env[cat_var].value_counts().head(12).index
                df_cat = df_env[df_env[cat_var].isin(top_cats)].copy()
                
                # Box plot by category
                categories = sorted(df_cat[cat_var].unique())
                data_by_cat = [pd.to_numeric(df_cat[df_cat[cat_var] == c]['frp_mean'], errors='coerce').dropna().values 
                              for c in categories]
                bp = axes[0].boxplot(data_by_cat, labels=categories, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('#ef4444')
                    patch.set_alpha(0.6)
                axes[0].set_xlabel(cat_var)
                axes[0].set_ylabel('FRP Mean')
                axes[0].set_title(f'Fire Intensity by {cat_var}')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Mean FRP by category
                mean_frp = df_cat.groupby(cat_var)['frp_mean'].mean().sort_values(ascending=False)
                axes[1].barh(mean_frp.index, mean_frp.values, color='#ef4444', alpha=0.7)
                axes[1].set_xlabel('Mean FRP')
                axes[1].set_ylabel(cat_var)
                axes[1].set_title(f'Mean Fire Intensity by {cat_var}')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    
    with st.expander("🔥 3. Multivariate Correlation Analysis", expanded=False):
        st.markdown("""
        **Purpose**: Identify multicollinearity and complex relationships between fire and environmental variables.
        """)
        
        corr_vars = st.multiselect(
            "Select variables for correlation analysis",
            options=sorted(env_numeric_cols),
            default=sorted(env_numeric_cols)[:min(10, len(env_numeric_cols))],
            key="fire_env_corr"
        )
        
        if corr_vars and len(corr_vars) >= 2:
            # Build correlation matrix
            corr_df = df_env[corr_vars].apply(pd.to_numeric, errors='coerce')
            corr_matrix = corr_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, annot=True if len(corr_vars) <= 10 else False, 
                       fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax, square=True,
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax.set_title('Fire-Environment Correlation Heatmap')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # High correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) >= 0.7:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                st.warning(f"⚠️ Found {len(high_corr)} highly correlated variable pairs (|r| ≥ 0.7):")
                st.dataframe(pd.DataFrame(high_corr).round(3), use_container_width=True)
            
            # Pair Plot
            st.subheader("Pair Plot Analysis")
            if len(corr_vars) >= 2 and len(corr_vars) <= 6:
                if st.button("Generate Pair Plot", key="fire_env_pairplot_btn"):
                    with st.spinner("Generating pair plot..."):
                        pp_df = df_env[corr_vars].apply(pd.to_numeric, errors='coerce').dropna()
                        if len(pp_df) > 1500:
                            pp_df = pp_df.sample(1500, random_state=42)
                        
                        if not pp_df.empty:
                            # Add fire intensity class if available for hue
                            if 'frp_mean' in df_env.columns:
                                frp = pd.to_numeric(df_env.loc[pp_df.index, 'frp_mean'], errors='coerce')
                                pp_df['Fire Intensity'] = pd.qcut(frp, q=3, labels=['Low', 'Medium', 'High'])
                                hue = 'Fire Intensity'
                            else:
                                hue = None
                                
                            fig = sns.pairplot(pp_df, hue=hue, diag_kind='kde', 
                                              plot_kws={'alpha': 0.5, 's': 15},
                                              palette='YlOrRd' if hue else None)
                            st.pyplot(fig.fig)
                        else:
                            st.warning("Not enough valid data for pair plot.")
            elif len(corr_vars) > 6:
                st.info("Select 6 or fewer variables for pair plot analysis.")
            else:
                st.info("Select at least 2 variables for pair plot analysis.")
    
    with st.expander("📋 4. Statistical Summary & Insights", expanded=False):
        st.markdown("""
        **Purpose**: Consolidated insights from the fire-environment analysis.
        """)
        
        st.subheader("Key Statistics by Category")
        
        if 'landcover_label' in df_env.columns and 'frp_mean' in df_env.columns:
            st.markdown("**Fire Intensity by Land Cover**")
            lc_stats = df_env.groupby('landcover_label').agg({
                'frp_mean': ['count', 'mean', 'std', 'max']
            }).round(2)
            lc_stats.columns = ['Fire Count', 'Mean FRP', 'Std FRP', 'Max FRP']
            lc_stats = lc_stats.sort_values('Mean FRP', ascending=False)
            st.dataframe(lc_stats, use_container_width=True)
        
        if 'month' in df_env.columns and 'frp_mean' in df_env.columns:
            st.markdown("**Fire Intensity by Month**")
            month_stats = df_env.groupby('month').agg({
                'frp_mean': ['count', 'mean', 'std']
            }).round(2)
            month_stats.columns = ['Fire Count', 'Mean FRP', 'Std FRP']
            st.dataframe(month_stats, use_container_width=True)
            
            # Temporal trend
            fig, ax = plt.subplots(figsize=(10, 4))
            month_counts = df_env.groupby('month').size()
            ax.bar(month_counts.index, month_counts.values, color='#ef4444', alpha=0.7)
            ax.set_xlabel('Month')
            ax.set_ylabel('Fire Count')
            ax.set_title('Temporal Distribution of Fires')
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Summary insights
        st.subheader("📌 Key Insights")
        
        insights = []
        
        # Peak fire month
        if 'month' in df_env.columns:
            peak_month = df_env['month'].value_counts().idxmax()
            insights.append(f"**Peak Fire Month**: {peak_month} with {df_env['month'].value_counts().max():,} fires")
        
        # Highest risk land cover
        if 'landcover_label' in df_env.columns and 'frp_mean' in df_env.columns:
            lc_mean = df_env.groupby('landcover_label')['frp_mean'].mean()
            high_risk_lc = lc_mean.idxmax()
            insights.append(f"**Highest Intensity Land Cover**: {high_risk_lc} (Mean FRP: {lc_mean.max():.2f})")
        
        # Elevation relationship
        if 'elevation' in df_env.columns and 'frp_mean' in df_env.columns:
            elev_corr = df_env['elevation'].corr(df_env['frp_mean'])
            if abs(elev_corr) > 0.2:
                direction = "decreases" if elev_corr < 0 else "increases"
                insights.append(f"**Elevation Effect**: Fire intensity {direction} with elevation (r={elev_corr:.3f})")
        
        for insight in insights:
            st.markdown(f"- {insight}")

# ---------------------------------------------------------------------------
# Page: Fire Overlay
# ---------------------------------------------------------------------------
elif page == "Fire Overlay":
    if rxr is None:
        st.error("Fire Overlay disabled: rasterio/rioxarray not installed.")
        st.stop()
    st.header("Fire Points Overlaid on Raster")

    @st.cache_data
    def load_fire_env() -> pd.DataFrame:
        parquet = FIRE_DIR / "fire_environment_2024.parquet"
        if parquet.exists():
            try:
                return pd.read_parquet(parquet)
            except Exception:
                pass
        # fallback raw fire
        files = [
            FIRE_DIR / "viirs-jpss1_2024_Algeria_type2.csv",
            FIRE_DIR / "viirs-jpss1_2024_Tunisia_type2.csv",
            FIRE_DIR / "viirs-jpss1_2024_Algeria.csv",
            FIRE_DIR / "viirs-jpss1_2024_Tunisia.csv",
        ]
        dfs = []
        for fp in files:
            if not fp.exists():
                continue
            try:
                d = pd.read_csv(fp, low_memory=False)
                if {"latitude", "longitude"}.issubset(d.columns):
                    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
                    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
                    d = d.dropna(subset=["latitude", "longitude"])
                if "type" in d.columns:
                    d = d[d["type"] == 2]
                d["country"] = "Algeria" if "Algeria" in fp.name else "Tunisia"
                if "acq_date" in d.columns:
                    d["dt"] = pd.to_datetime(d["acq_date"], errors="coerce")
                    d["month"] = d["dt"].dt.strftime("%Y-%m")
                dfs.append(d)
            except Exception:
                pass
        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            return out
        return pd.DataFrame()

    fire = load_fire_env()
    if fire.empty:
        st.error("No fire data available.")
        st.stop()

    bg_type = st.selectbox("Background Raster", ["Elevation", "Climate"], index=0)
    climate_ds = load_climate() if bg_type == "Climate" else None
    elev_da = load_elevation() if bg_type == "Elevation" else None

    clim_var = None
    clim_month_idx = None
    if bg_type == "Climate" and climate_ds is not None:
        clim_var = st.selectbox("Climate Variable", list(climate_ds.data_vars), index=0)
        months = list(climate_ds.time.dt.strftime("%Y-%m").values) if "time" in climate_ds.dims else []
        if months:
            clim_month_idx = st.slider("Month", 0, len(months) - 1, 0)
            st.caption(f"Selected month: {months[clim_month_idx]}")

    col1, col2 = st.columns(2)
    countries = sorted(fire["country"].dropna().unique()) if "country" in fire.columns else []
    sel_c = col1.multiselect("Country", countries, default=countries)
    months_fire = sorted(fire["month"].dropna().unique()) if "month" in fire.columns else []
    sel_m = col2.multiselect("Fire Month", months_fire, default=months_fire)

    fdf = fire.copy()
    if sel_c:
        fdf = fdf[fdf["country"].isin(sel_c)]
    if sel_m and "month" in fdf.columns:
        fdf = fdf[fdf["month"].isin(sel_m)]

    raster_arr = None
    extent = None
    title = ""
    if bg_type == "Elevation" and elev_da is not None:
        raster_arr = np.asarray(elev_da.values, dtype=float)
        nd = elev_da.rio.nodata
        if nd is not None:
            raster_arr = np.where(raster_arr == nd, np.nan, raster_arr)
        if "x" in elev_da.coords and "y" in elev_da.coords:
            xs = elev_da["x"].values
            ys = elev_da["y"].values
            extent = [xs.min(), xs.max(), ys.min(), ys.max()]
        title = "Elevation (m)"
    elif bg_type == "Climate" and climate_ds is not None and clim_var is not None:
        da = climate_ds[clim_var]
        if clim_month_idx is not None and "time" in da.dims:
            da = da.isel(time=clim_month_idx)
        raster_arr = np.asarray(da.values, dtype=float)
        if hasattr(da, "rio"):
            nodata = da.rio.nodata
            if nodata is not None:
                raster_arr = np.where(raster_arr == nodata, np.nan, raster_arr)
        if "x" in da.coords and "y" in da.coords:
            xs = da["x"].values
            ys = da["y"].values
            extent = [xs.min(), xs.max(), ys.min(), ys.max()]
        title = f"{clim_var}"

    if raster_arr is None:
        st.warning("Raster not available for selected background.")
        st.stop()

    if extent is not None and {"latitude", "longitude"}.issubset(fdf.columns):
        xmin, xmax, ymin, ymax = extent
        fdf = fdf[(fdf["longitude"] >= xmin) & (fdf["longitude"] <= xmax) &
                  (fdf["latitude"] >= ymin) & (fdf["latitude"] <= ymax)]

    max_pts = 25000
    fdf_plot = fdf.sample(max_pts, random_state=0) if len(fdf) > max_pts else fdf

    fig, ax = plt.subplots(figsize=(8, 6))
    finite = np.isfinite(raster_arr)
    if finite.any():
        vmin, vmax = np.nanpercentile(raster_arr[finite], [2, 98])
    else:
        vmin, vmax = None, None
    im = ax.imshow(raster_arr,
                   cmap="terrain" if bg_type == "Elevation" else "viridis",
                   vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(f"{title} with Fire Points")

    if extent is not None and {"latitude", "longitude"}.issubset(fdf_plot.columns):
        xmin, xmax, ymin, ymax = extent
        w = raster_arr.shape[1]
        h = raster_arr.shape[0]
        xpix = (fdf_plot["longitude"].values - xmin) / (xmax - xmin) * w
        ypix = (ymax - fdf_plot["latitude"].values) / (ymax - ymin) * h
        ax.scatter(xpix, ypix, s=6, c="red", alpha=0.5, linewidths=0)
    else:
        ax.scatter(fdf_plot["longitude"], fdf_plot["latitude"], s=6, c="red", alpha=0.5, linewidths=0)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, shrink=0.7, label=title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Overlay is static; pydeck can be added for interactive layers.")

# ---------------------------------------------------------------------------
# Page: Cleaned Outputs
# ---------------------------------------------------------------------------
elif page == "Cleaned Outputs":
    st.header("DATA_CLEANED — Outputs Browser")

    clim_clean = next((p for p in [
        CLEANED_DIR / "CLIMATE" / "climate_2024_clean.nc",
        CLEANED_DIR / "CLIMATE" / "climate_2024.nc",
        CLEANED_DIR / "climate" / "climate_2024_clean.nc",
        CLEANED_DIR / "climate" / "climate_2024.nc",
    ] if p.exists()), None)
    clim_cut = CLEANED_DIR / "CLIMATE_CUT" / "climate_2024_cut.nc"
    soil_nc = CLEANED_DIR / "SOIL" / "soil_id_grid.nc"
    fire_dir = CLEANED_DIR / "FIRE"
    fire_not2 = []
    if fire_dir.exists():
        # More forgiving patterns
        for pat in ["*_not2.csv", "*not2*.csv"]:
            fire_not2 = list(fire_dir.glob(pat))
            if fire_not2:
                break
    fire_cut = CLEANED_DIR / "FIRE_CUT" / "fire_filtered.csv"
    merged_pq = CLEANED_DIR / "processed" / "merged_dataset.parquet"
    merged_csv = CLEANED_DIR / "processed" / "merged_dataset.csv"

    st.subheader("Index")
    st.write({
        "climate_2024_clean.nc": bool(clim_clean),
        "climate_2024_cut.nc": clim_cut.exists(),
        "soil_id_grid.nc": soil_nc.exists(),
        "fire not2 matches": len(fire_not2),
        "fire_filtered.csv": fire_cut.exists(),
        "merged_dataset": merged_pq.exists() or merged_csv.exists(),
    })

    tab1, tab2, tab3, tab4 = st.tabs(["Climate", "Soil", "Fire (not2/cut)", "Merged"])

    with tab1:
        st.markdown("**Climate (2024)**")
        ds = None
        candidate = clim_clean if clim_clean and clim_clean.exists() else clim_cut if clim_cut.exists() else None
        if candidate:
            try:
                ds = xr.open_dataset(candidate)
                st.caption(f"Loaded: {candidate}")
            except Exception as e:
                st.warning(f"Failed to open {candidate.name}: {e}")
        if ds is None:
            st.info("No cleaned climate NetCDF found.")
        else:
            var = st.selectbox("Variable", list(ds.data_vars), index=0)
            da = ds[var]
            if isinstance(da, xr.DataArray) and 'time' in da.dims:
                try:
                    times = [pd.to_datetime(t).strftime("%Y-%m") for t in da.time.values]
                except Exception:
                    times = []
                if times:
                    tidx = st.slider("Month", 0, len(times) - 1, 0)
                    da = da.isel(time=tidx)
                    st.caption(f"Selected: {times[tidx]}")
            arr = np.asarray(getattr(da, 'values', []), dtype=float)
            finite = np.isfinite(arr)
            if finite.any():
                st.table(pd.DataFrame({
                    "metric": ["min", "mean", "max", "std"],
                    "value": [float(np.nanmin(arr)), float(np.nanmean(arr)),
                              float(np.nanmax(arr)), float(np.nanstd(arr))]
                }))
            st.caption("Map preview omitted (see Climate page).")

    with tab2:
        st.markdown("**Soil ID Grid**")
        if not soil_nc.exists():
            st.info("No soil_id_grid.nc found.")
        else:
            try:
                ds = xr.open_dataset(soil_nc)
                var = "soil_id" if "soil_id" in ds else list(ds.data_vars)[0]
                da = ds[var]
                st.write("Dimensions:", dict(da.sizes))
                st.write("Dtype:", str(da.dtype))
            except Exception as e:
                st.warning(f"Failed to open soil NetCDF: {e}")

    with tab3:
        st.markdown("**Fire — type != 2**")
        if fire_not2:
            fp = st.selectbox("File", [p.name for p in fire_not2], index=0)
            sel = next(p for p in fire_not2 if p.name == fp)
            try:
                df = pd.read_csv(sel, low_memory=False)
                st.metric("Rows", f"{len(df):,}")
                cand = [c for c in ["frp", "frp_mean", "bright_ti4", "bright_ti5"] if c in df.columns]
                if cand:
                    vals = pd.to_numeric(df[cand[0]], errors="coerce").dropna()
                    if not vals.empty:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.hist(vals.values, bins=60, color="#ef4444", alpha=0.85)
                        ax.set_title(f"Histogram — {cand[0]}")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                if {"latitude", "longitude"}.issubset(df.columns) and len(df):
                    m = df.sample(min(15000, len(df)), random_state=0)
                    st.map(m[["latitude", "longitude"]])
            except Exception as e:
                st.warning(f"Failed to load {fp}: {e}")
        else:
            st.info("No not2 fire files found.")
        st.markdown("**Fire — filtered by lower whisker**")
        if fire_cut.exists():
            try:
                df = pd.read_csv(fire_cut, low_memory=False)
                st.metric("Rows", f"{len(df):,}")
                if {"latitude", "longitude"}.issubset(df.columns) and len(df):
                    m = df.sample(min(15000, len(df)), random_state=0)
                    st.map(m[["latitude", "longitude"]])
            except Exception as e:
                st.warning(f"Failed to open fire_filtered.csv: {e}")
        else:
            st.info("No fire_filtered.csv found.")

    with tab4:
        st.markdown("**Merged Dataset**")
        if merged_pq.exists() or merged_csv.exists():
            st.success("Use the 'Merged Dataset' page for exploration.")
        else:
            st.info("No merged dataset found.")

# ---------------------------------------------------------------------------
# Page: Merged Dataset
# ---------------------------------------------------------------------------
elif page == "Merged Dataset":
    st.header("Final Merged Fire / Non-Fire Dataset")

    def load_merged() -> tuple[pd.DataFrame, Path | None]:
        path = locate_merged_dataset()
        if path is None:
            return pd.DataFrame(), None
        df = load_cached_merged_dataset(path.as_posix(), int(path.stat().st_mtime_ns))
        return df, path

    dfm, merged_path = load_merged()
    if merged_path is not None:
        st.caption(f"Loaded merged dataset from: {merged_path}")
    if dfm.empty:
        st.error("Merged dataset not found. Run geo_pipeline.py final-merge first.")
        st.stop()

    # Preserve original fire counts before any normalization / filtering for debugging
    if 'fire' in dfm.columns:
        _orig_fire = dfm['fire'].copy()
        _orig_counts = _orig_fire.value_counts(dropna=False)
    else:
        _orig_fire = pd.Series([], dtype='Int64')
        _orig_counts = pd.Series([], dtype='int')

    # Normalize 'fire' column robustly
    if 'fire' not in dfm.columns:
        st.error("No 'fire' column found in merged dataset.")
        st.stop()
    dfm['fire'] = normalize_fire_column(dfm, 'fire')

    fire_uniques = sorted([v for v in dfm['fire'].dropna().unique()])
    if len(fire_uniques) == 1 and fire_uniques[0] == 1:
        if 'source' in dfm.columns:
            neg_sources = {"random_batch", "band", "random", "even_grid"}
            mask_neg = dfm['source'].astype(str).str.lower().isin(neg_sources)
            if mask_neg.any():
                dfm.loc[mask_neg, 'fire'] = 0
                st.warning("Inferred non-fire rows from 'source' column; fire label repaired.")
        else:
            st.warning("All rows labeled fire=1; no negatives detected. Check sampling pipeline.")

    # Fallback repair: if original had zeros but after normalization none remain, restore them
    if 0 in _orig_counts.index and (dfm['fire'] == 0).sum() == 0:
        zero_mask = _orig_fire.astype(str).isin(['0']) | (_orig_fire == 0)
        if zero_mask.any():
            dfm.loc[zero_mask, 'fire'] = 0
            st.warning("Restored non-fire labels from original raw fire column.")

    if debug_mode:
        with st.sidebar.expander("Debug: Original fire counts", expanded=False):
            st.write(_orig_counts)

    if debug_mode:
        with st.sidebar.expander("Debug: Initial fire counts", expanded=True):
            st.write(dfm['fire'].value_counts(dropna=False))

    # Coerce numeric-like columns
    obj_cols = [c for c in dfm.columns if dfm[c].dtype == object]
    if obj_cols:
        dfm[obj_cols] = dfm[obj_cols].replace({'None': np.nan, 'none': np.nan, '': np.nan})

    likely_numeric_prefixes = ("prec_", "tmax_", "tmin_", "elevation", "slope", "aspect",
                               "frp_", "coarse", "clay", "silt", "sand", "org_carbon",
                               "total_n", "soil_id", "bright_ti", "ruggedness", "bulk", "ref_bulk")
    for c in dfm.columns:
        if c.startswith(likely_numeric_prefixes) or c in ("latitude", "longitude"):
            try:
                dfm[c] = pd.to_numeric(dfm[c], errors='coerce')
            except Exception:
                pass

    if debug_mode:
        with st.expander("Raw Data Debug Info", expanded=False):
            st.write("Shape:", dfm.shape)
            st.write("Columns:", list(dfm.columns))
            st.write("Fire unique:", sorted(dfm['fire'].dropna().unique()))
            st.write(dfm['fire'].value_counts(dropna=False))

    # Confidence categorical handling (standardize encodings but no filtering)
    conf_col = 'confidence' if 'confidence' in dfm.columns else None
    if conf_col:
        unique_conf = set(str(v).strip().lower() for v in dfm[conf_col].dropna().unique())
        if unique_conf.issubset({'l', 'n', 'h', 'low', 'normal', 'high', 'none'}):
            dfm[conf_col] = dfm[conf_col].astype(str).str.strip().str.lower().replace(
                {'low': 'l', 'normal': 'n', 'high': 'h'}
            )

    # Candidate numeric columns excluding coordinates & fire (for stats only)
    numeric_candidates = [c for c in dfm.columns
                          if pd.api.types.is_numeric_dtype(dfm[c])
                          and c not in ("latitude", "longitude", "fire")]

    numeric_candidates = [c for c in numeric_candidates if not is_low_cardinality_numeric(dfm, c)]
    sorted_numeric = sorted(numeric_candidates)
    default_focus_idx = 0
    if "elevation" in sorted_numeric:
        default_focus_idx = sorted_numeric.index("elevation")
    focus_var = st.selectbox("Focus variable", sorted_numeric, index=default_focus_idx) if sorted_numeric else None

    # No filters: always work on the full dataset for metrics and previews
    fdf = dfm.copy()

    # Metrics
    st.subheader("Class Distribution")
    counts = fdf['fire'].value_counts(dropna=False)
    fire_count = int(counts.get(1, 0))
    nonfire_count = int(counts.get(0, 0))
    total = fire_count + nonfire_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Fire (1)", f"{fire_count:,}")
    m2.metric("Non-Fire (0)", f"{nonfire_count:,}")
    m3.metric("Fire Ratio", f"{(fire_count / total):.2%}" if total else "0.00%")
    if total:
        st.progress(fire_count / total)

    with st.expander("Detailed Counts"):
        st.write(counts)

    if 'source' in fdf.columns:
        src_counts = fdf['source'].value_counts()
        st.caption("Source breakdown: " + ", ".join([f"{s}:{c}" for s, c in src_counts.items()]))

    target_fire_ratio = 0.30
    if total:
        ratio_fire_local = fire_count / total
        diff_ratio = ratio_fire_local - target_fire_ratio
        st.caption(f"Target fire ratio: {target_fire_ratio:.2%} | Actual: {ratio_fire_local:.2%} | Delta: {diff_ratio:+.2%}")

    st.subheader("Table & Map")

    preferred = ["latitude", "longitude", "fire", "confidence", "source", "country", "soil_id",
                 "elevation", "coarse", "sand", "silt", "clay", "org_carbon", "TEXTURE_USDA", "lcc_label"]
    default_cols = [c for c in preferred if c in fdf.columns]
    cols = st.multiselect("Columns to display", options=list(fdf.columns), default=default_cols)

    tf1, tf2 = st.columns([2, 1])
    page_rows = tf1.slider("Rows per page", 100, 5000, 1000, 100)
    page_idx = tf2.number_input("Page", min_value=1, value=1, step=1)

    view_df = fdf[cols] if cols else fdf.copy()

    n = len(view_df)
    start = max(0, (int(page_idx) - 1) * int(page_rows))
    end = min(n, start + int(page_rows))
    if start >= n:
        start = 0
        end = min(n, int(page_rows))
    st.caption(f"Showing rows {start + 1:,}–{end:,} of {n:,}")

    if n:
        st.dataframe(view_df.iloc[start:end], use_container_width=True, height=420)
        csv_bytes = view_df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV (selected columns)",
            data=csv_bytes,
            file_name="merged_dataset_selected.csv",
            mime="text/csv",
        )

    st.subheader("Map")
    if {"latitude", "longitude"}.issubset(fdf.columns) and not fdf.empty:
        
        # Map controls
        map_col1, map_col2, map_col3 = st.columns(3)
        with map_col1:
            sample_size = st.slider("Sample size for map", 500, 20000, 5000, 500)
        with map_col2:
            show_layer = st.radio("Show", ["Both", "Fire Only", "Non-Fire Only"], horizontal=True)
        with map_col3:
            point_size = st.slider("Point size", 1, 10, 4)
        
        # Stratified sampling to ensure both classes are represented
        map_df = fdf.copy()
        
        # Robust fire value detection
        def is_fire(val):
            if pd.isna(val):
                return False
            if isinstance(val, bool):
                return val
            try:
                return int(float(val)) == 1
            except (ValueError, TypeError):
                return str(val).strip().lower() in ('1', 'true', 'yes', 'fire')
        
        map_df['is_fire'] = map_df['fire'].apply(is_fire)
        
        # Separate fire and non-fire
        fire_df = map_df[map_df['is_fire']].copy()
        nonfire_df = map_df[~map_df['is_fire']].copy()
        
        n_fire = len(fire_df)
        n_nonfire = len(nonfire_df)
        
        # Calculate actual ratio and sample proportionally (preserve 80/20 or actual ratio)
        total_available = n_fire + n_nonfire
        if total_available > 0:
            actual_fire_ratio = n_fire / total_available
        else:
            actual_fire_ratio = 0.8  # default
        
        # Sample proportionally to actual data ratio
        n_fire_sample = int(sample_size * actual_fire_ratio)
        n_nonfire_sample = sample_size - n_fire_sample
        
        # Ensure at least some of each class if they exist
        if n_fire > 0 and n_fire_sample < 100:
            n_fire_sample = min(100, n_fire)
        if n_nonfire > 0 and n_nonfire_sample < 100:
            n_nonfire_sample = min(100, n_nonfire)
        
        sampled_fire = fire_df.sample(min(n_fire_sample, n_fire), random_state=0) if n_fire > 0 else pd.DataFrame()
        sampled_nonfire = nonfire_df.sample(min(n_nonfire_sample, n_nonfire), random_state=0) if n_nonfire > 0 else pd.DataFrame()
        
        # Calculate map center from all data
        center_lat = float(map_df['latitude'].mean())
        center_lon = float(map_df['longitude'].mean())
        
        # Debug info
        if debug_mode:
            with st.expander("Map Debug Info"):
                st.write(f"Total fire points: {n_fire:,}")
                st.write(f"Total non-fire points: {n_nonfire:,}")
                st.write(f"Fire ratio: {actual_fire_ratio*100:.1f}%")
                st.write(f"Sampled fire: {len(sampled_fire):,} ({100*len(sampled_fire)/(len(sampled_fire)+len(sampled_nonfire)):.1f}%)")
                st.write(f"Sampled non-fire: {len(sampled_nonfire):,} ({100*len(sampled_nonfire)/(len(sampled_fire)+len(sampled_nonfire)):.1f}%)")
                st.write("Original fire values:", fdf['fire'].value_counts(dropna=False).head(10).to_dict())
        
        # Create layers
        layers = []
        
        # Non-fire layer (blue) - render first so fire points appear on top
        if show_layer in ["Both", "Non-Fire Only"] and len(sampled_nonfire) > 0:
            nonfire_data = sampled_nonfire[['latitude', 'longitude']].copy()
            nonfire_data['label'] = 'Non-Fire'
            
            nonfire_layer = pdk.Layer(
                "ScatterplotLayer",
                data=nonfire_data,
                get_position=["longitude", "latitude"],
                get_color=[59, 130, 246, 200],  # Blue
                get_radius=point_size * 800,
                radius_min_pixels=point_size,
                radius_max_pixels=point_size * 3,
                pickable=True,
            )
            layers.append(nonfire_layer)
        
        # Fire layer (red) - render on top
        if show_layer in ["Both", "Fire Only"] and len(sampled_fire) > 0:
            fire_data = sampled_fire[['latitude', 'longitude']].copy()
            fire_data['label'] = 'Fire'
            
            fire_layer = pdk.Layer(
                "ScatterplotLayer",
                data=fire_data,
                get_position=["longitude", "latitude"],
                get_color=[239, 68, 68, 220],  # Red
                get_radius=point_size * 800,
                radius_min_pixels=point_size,
                radius_max_pixels=point_size * 3,
                pickable=True,
            )
            layers.append(fire_layer)
        
        if layers:
            # Create view state
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=5,
                pitch=0,
            )
            
            # Create deck
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={"text": "{label}\nLat: {latitude:.4f}\nLon: {longitude:.4f}"},
                map_style="mapbox://styles/mapbox/light-v10",
            )
            
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning("No data to display on map.")
        
        # Legend and stats
        st.markdown("---")
        legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
        with legend_col1:
            st.markdown("🔴 **Fire**")
            st.caption(f"Showing: {len(sampled_fire):,}")
            st.caption(f"Total: {n_fire:,}")
        with legend_col2:
            st.markdown("🔵 **Non-Fire**")
            st.caption(f"Showing: {len(sampled_nonfire):,}")
            st.caption(f"Total: {n_nonfire:,}")
        with legend_col3:
            if n_fire + n_nonfire > 0:
                st.metric("Fire %", f"{100*n_fire/(n_fire+n_nonfire):.1f}%")
        with legend_col4:
            if n_fire + n_nonfire > 0:
                st.metric("Non-Fire %", f"{100*n_nonfire/(n_fire+n_nonfire):.1f}%")
        
        # Side-by-side static maps for clearer comparison
        with st.expander("📊 Side-by-Side Comparison Maps"):
            static_col1, static_col2 = st.columns(2)
            
            with static_col1:
                st.markdown("**🔴 Fire Points**")
                if len(sampled_fire) > 0:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(sampled_fire['longitude'], sampled_fire['latitude'], 
                              c='red', s=3, alpha=0.6, label=f'Fire ({len(sampled_fire):,})')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.set_title('Fire Detections')
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No fire points")
            
            with static_col2:
                st.markdown("**🔵 Non-Fire Points**")
                if len(sampled_nonfire) > 0:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(sampled_nonfire['longitude'], sampled_nonfire['latitude'], 
                              c='blue', s=3, alpha=0.6, label=f'Non-Fire ({len(sampled_nonfire):,})')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.set_title('Non-Fire Samples')
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No non-fire points")
    else:
        st.info("No spatial coordinates available for mapping.")

    st.subheader("Statistics")
    if focus_var and focus_var in fdf.columns and pd.api.types.is_numeric_dtype(fdf[focus_var]):
        arr = pd.to_numeric(fdf[focus_var], errors='coerce').dropna()
        if not arr.empty:
            stats_df = pd.DataFrame({
                'count': [int(arr.count())],
                'mean': [float(arr.mean())],
                'std': [float(arr.std())],
                'min': [float(arr.min())],
                '25%': [float(arr.quantile(0.25))],
                '50%': [float(arr.median())],
                '75%': [float(arr.quantile(0.75))],
                'max': [float(arr.max())]
            })
            st.table(stats_df)
        else:
            st.info("No numeric values for selected focus variable.")
    else:
        st.info("Select a numeric focus variable to see statistics.")

    # ==========================================================================
    # COMPREHENSIVE EDA SECTION
    # ==========================================================================
    st.markdown("---")
    st.header("📊 Comprehensive Exploratory Data Analysis")
    
    # Identify column types
    numeric_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c]) 
                    and c not in ('latitude', 'longitude', 'fire')]
    categorical_cols = [c for c in fdf.columns if fdf[c].dtype == 'object' 
                        or is_low_cardinality_numeric(fdf, c)]
    
    # ==========================================================================
    # 1. DISTRIBUTION ANALYSIS (UNIVARIATE)
    # ==========================================================================
    with st.expander("📈 1. Distribution Analysis (Univariate)", expanded=True):
        st.markdown("""
        **Purpose**: Analyze the distribution of the target variable and key features to identify 
        skewness, outliers, and overall data shape that may impact modeling decisions.
        """)
        
        # Target variable distribution
        st.subheader("Target Variable: Fire")
        if 'fire' in fdf.columns:
            fire_counts = fdf['fire'].value_counts(dropna=False)
            
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#3b82f6', '#ef4444']
                bars = ax.bar(['Non-Fire (0)', 'Fire (1)'], 
                             [fire_counts.get(0, 0), fire_counts.get(1, 0)],
                             color=colors, edgecolor='white', linewidth=2)
                ax.set_ylabel('Count')
                ax.set_title('Target Variable Distribution')
                for bar, count in zip(bars, [fire_counts.get(0, 0), fire_counts.get(1, 0)]):
                    ax.annotate(f'{count:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=11, fontweight='bold')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            with dist_col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sizes = [fire_counts.get(0, 0), fire_counts.get(1, 0)]
                labels = [f'Non-Fire\n({sizes[0]:,})', f'Fire\n({sizes[1]:,})']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                      startangle=90, explode=(0.02, 0.02))
                ax.set_title('Class Balance')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Class imbalance interpretation
            total = sum(sizes)
            ratio = sizes[1] / total if total > 0 else 0
            if ratio < 0.2 or ratio > 0.8:
                st.warning(f"⚠️ **Class Imbalance Detected**: Fire ratio is {ratio:.1%}. Consider using stratified sampling, SMOTE, or class weights in modeling.")
            else:
                st.success(f"✅ **Balanced Classes**: Fire ratio is {ratio:.1%}, which is suitable for standard modeling approaches.")
        
        st.markdown("---")
        st.subheader("Key Feature Distributions")
        
        # Select features for distribution analysis
        key_features = st.multiselect(
            "Select features for distribution analysis",
            options=sorted(numeric_cols),
            default=sorted(numeric_cols)[:min(5, len(numeric_cols))],
            key="dist_features"
        )
        
        if key_features:
            for feat in key_features:
                st.markdown(f"#### {feat}")
                
                fig = plot_distribution_analysis(fdf, feat, 'fire', color='#6366f1')
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Statistical summary
                summary = generate_statistical_summary(fdf, feat, 'fire')
                st.info(summary)
                
                # Outlier detection summary
                outliers = detect_outliers_summary(fdf[feat], 'iqr')
                shape = compute_skewness_kurtosis(fdf[feat])
                
                eda_col1, eda_col2, eda_col3, eda_col4 = st.columns(4)
                eda_col1.metric("Skewness", f"{shape['skewness']:.3f}" if not np.isnan(shape['skewness']) else "N/A")
                eda_col2.metric("Kurtosis", f"{shape['kurtosis']:.3f}" if not np.isnan(shape['kurtosis']) else "N/A")
                eda_col3.metric("Outliers", f"{outliers['outlier_count']:,}")
                eda_col4.metric("Outlier %", f"{outliers['outlier_pct']:.1f}%")
                
                st.markdown("---")
    
    # ==========================================================================
    # 2. RELATIONSHIP MAPPING (BIVARIATE)
    # ==========================================================================
    with st.expander("🔗 2. Relationship Mapping (Bivariate)", expanded=False):
        st.markdown("""
        **Purpose**: Examine relationships between features and the target variable.
        - **Numeric features**: Scatter plots to identify linear/non-linear trends
        - **Categorical features**: Box/violin plots to show target distribution across categories
        """)
        
        biv_tab1, biv_tab2 = st.tabs(["Numeric vs Target", "Categorical vs Target"])
        
        with biv_tab1:
            st.subheader("Numeric Features vs Fire Occurrence")
            
            # Select numeric features for scatter analysis
            scatter_features = st.multiselect(
                "Select numeric features for scatter analysis",
                options=sorted(numeric_cols),
                default=sorted(numeric_cols)[:min(4, len(numeric_cols))],
                key="scatter_features"
            )
            
            if scatter_features and 'fire' in fdf.columns:
                for feat in scatter_features:
                    st.markdown(f"#### {feat} vs Fire")
                    
                    fig = plot_bivariate_numeric(fdf, feat, 'fire', hue_col='fire', sample_n=3000)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Compute correlation and statistical test
                    feat_vals = pd.to_numeric(fdf[feat], errors='coerce').dropna()
                    fire_vals = pd.to_numeric(fdf['fire'], errors='coerce')
                    common_idx = feat_vals.index.intersection(fire_vals.dropna().index)
                    
                    if len(common_idx) > 10:
                        corr = feat_vals.loc[common_idx].corr(fire_vals.loc[common_idx])
                        
                        # Point-biserial correlation test
                        try:
                            stat, p_val = stats.pointbiserialr(fire_vals.loc[common_idx], feat_vals.loc[common_idx])
                            st.caption(f"**Point-Biserial Correlation**: r={stat:.4f}, p-value={p_val:.4e}")
                            if p_val < 0.05:
                                st.success(f"✅ Statistically significant relationship (p < 0.05)")
                            else:
                                st.warning(f"⚠️ No statistically significant relationship (p >= 0.05)")
                        except Exception:
                            st.caption(f"**Pearson Correlation**: r={corr:.4f}")
                    
                    # Compare distributions by fire class
                    fire_1_vals = fdf.loc[fdf['fire'] == 1, feat].dropna()
                    fire_0_vals = fdf.loc[fdf['fire'] == 0, feat].dropna()
                    
                    if len(fire_1_vals) > 5 and len(fire_0_vals) > 5:
                        try:
                            t_stat, t_pval = stats.ttest_ind(fire_1_vals, fire_0_vals)
                            st.caption(f"**T-test** (Fire vs Non-Fire): t={t_stat:.3f}, p-value={t_pval:.4e}")
                        except Exception:
                            pass
                    
                    st.markdown("---")
        
        with biv_tab2:
            st.subheader("Categorical Features vs Fire Occurrence")
            
            if categorical_cols:
                cat_feature = st.selectbox(
                    "Select categorical feature",
                    options=sorted(categorical_cols),
                    key="cat_feature"
                )
                
                if cat_feature and 'fire' in fdf.columns:
                    st.markdown(f"#### Fire Distribution by {cat_feature}")
                    
                    # Count plot by category and fire status
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Get top categories
                    top_cats = fdf[cat_feature].value_counts().head(15).index
                    df_cat = fdf[fdf[cat_feature].isin(top_cats)].copy()
                    
                    # Stacked bar chart
                    cross_tab = pd.crosstab(df_cat[cat_feature], df_cat['fire'], normalize='index') * 100
                    cross_tab.plot(kind='bar', stacked=True, ax=axes[0], 
                                   color=['#3b82f6', '#ef4444'], edgecolor='white')
                    axes[0].set_xlabel(cat_feature)
                    axes[0].set_ylabel('Percentage')
                    axes[0].set_title(f'Fire % by {cat_feature}')
                    axes[0].legend(['Non-Fire', 'Fire'], loc='upper right')
                    axes[0].tick_params(axis='x', rotation=45)
                    
                    # Count by category
                    count_by_cat = df_cat.groupby([cat_feature, 'fire']).size().unstack(fill_value=0)
                    count_by_cat.plot(kind='bar', ax=axes[1], 
                                     color=['#3b82f6', '#ef4444'], edgecolor='white')
                    axes[1].set_xlabel(cat_feature)
                    axes[1].set_ylabel('Count')
                    axes[1].set_title(f'Sample Count by {cat_feature}')
                    axes[1].legend(['Non-Fire', 'Fire'], loc='upper right')
                    axes[1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Chi-square test
                    try:
                        contingency = pd.crosstab(df_cat[cat_feature], df_cat['fire'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        st.caption(f"**Chi-Square Test**: χ²={chi2:.2f}, df={dof}, p-value={p_val:.4e}")
                        if p_val < 0.05:
                            st.success(f"✅ Significant association between {cat_feature} and fire occurrence (p < 0.05)")
                        else:
                            st.info(f"ℹ️ No significant association between {cat_feature} and fire occurrence (p >= 0.05)")
                    except Exception as e:
                        st.caption(f"Chi-square test could not be performed: {e}")
            else:
                st.info("No categorical columns found in the dataset.")
    
    # ==========================================================================
    # 3. COMPLEX INTERACTIONS (MULTIVARIATE)
    # ==========================================================================
    with st.expander("🔥 3. Complex Interactions (Multivariate)", expanded=False):
        st.markdown("""
        **Purpose**: Identify multicollinearity and complex interactions between multiple variables.
        - **Correlation Heatmap**: Identify highly correlated features (multicollinearity)
        - **Pair Plots**: See how relationships change when a third variable is introduced
        """)
        
        multi_tab1, multi_tab2, multi_tab3 = st.tabs(["Correlation Heatmap", "Multicollinearity", "Pair Plots"])
        
        with multi_tab1:
            st.subheader("Feature Correlation Heatmap")
            
            corr_features = st.multiselect(
                "Select features for correlation analysis",
                options=sorted(numeric_cols),
                default=sorted(numeric_cols)[:min(15, len(numeric_cols))],
                key="corr_features"
            )
            
            corr_method = st.radio("Correlation method", ["pearson", "spearman", "kendall"], horizontal=True)
            
            if corr_features and len(corr_features) >= 2:
                fig = plot_correlation_heatmap(fdf, corr_features, figsize=(12, 10), method=corr_method)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Unable to generate correlation heatmap.")
            else:
                st.info("Select at least 2 features for correlation analysis.")
        
        with multi_tab2:
            st.subheader("Multicollinearity Detection")
            
            corr_threshold = st.slider("Correlation threshold", 0.5, 0.95, 0.8, 0.05)
            
            if numeric_cols:
                high_corr = identify_multicollinearity(fdf, numeric_cols[:30], threshold=corr_threshold)
                
                if high_corr:
                    st.warning(f"⚠️ Found {len(high_corr)} highly correlated feature pairs (|r| ≥ {corr_threshold}):")
                    
                    corr_df = pd.DataFrame(high_corr)
                    corr_df['correlation'] = corr_df['correlation'].round(3)
                    st.dataframe(corr_df, use_container_width=True)
                    
                    st.markdown("""
                    **Recommendations**:
                    - Consider removing one feature from each highly correlated pair
                    - Use PCA or other dimensionality reduction techniques
                    - Apply regularization (L1/L2) in modeling to handle multicollinearity
                    """)
                else:
                    st.success(f"✅ No highly correlated feature pairs found (threshold: |r| ≥ {corr_threshold})")
        
        with multi_tab3:
            st.subheader("Pair Plots")
            st.caption("Visualize pairwise relationships with fire status as hue.")
            
            pair_features = st.multiselect(
                "Select features for pair plot (max 5)",
                options=sorted(numeric_cols),
                default=sorted(numeric_cols)[:min(4, len(numeric_cols))],
                max_selections=5,
                key="pair_features"
            )
            
            if pair_features and len(pair_features) >= 2:
                with st.spinner("Generating pair plot (this may take a moment)..."):
                    fig = plot_pairplot(fdf, pair_features, hue_col='fire', sample_n=1500)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Unable to generate pair plot. Try selecting different features or reducing sample size.")
            else:
                st.info("Select 2-5 features for pair plot analysis.")
    
    # ==========================================================================
    # 4. STATISTICAL SUMMARY & INTERPRETATION
    # ==========================================================================
    with st.expander("📋 4. Statistical Summary & Recommendations", expanded=False):
        st.markdown("""
        **Purpose**: Provide a consolidated statistical summary and actionable recommendations 
        based on the EDA findings.
        """)
        
        st.subheader("Dataset Overview")
        
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        overview_col1.metric("Total Samples", f"{len(fdf):,}")
        overview_col2.metric("Total Features", f"{len(fdf.columns):,}")
        overview_col3.metric("Numeric Features", f"{len(numeric_cols):,}")
        
        # Missing values summary
        st.subheader("Missing Values Summary")
        missing = fdf.isnull().sum()
        missing_pct = (missing / len(fdf) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df.head(20), use_container_width=True)
            
            high_missing = missing_df[missing_df['Missing %'] > 20]
            if len(high_missing) > 0:
                st.warning(f"⚠️ {len(high_missing)} columns have >20% missing values. Consider imputation or removal.")
        else:
            st.success("✅ No missing values in the dataset!")
        
        # Feature statistics table
        st.subheader("Feature Statistics")
        
        if numeric_cols:
            stats_data = []
            for col in numeric_cols[:20]:  # Limit to 20 columns
                arr = pd.to_numeric(fdf[col], errors='coerce').dropna()
                if len(arr) > 0:
                    shape = compute_skewness_kurtosis(arr)
                    outliers = detect_outliers_summary(arr, 'iqr')
                    
                    # Correlation with fire
                    fire_vals = pd.to_numeric(fdf['fire'], errors='coerce')
                    common_idx = arr.index.intersection(fire_vals.dropna().index)
                    corr_fire = arr.loc[common_idx].corr(fire_vals.loc[common_idx]) if len(common_idx) > 10 else np.nan
                    
                    stats_data.append({
                        'Feature': col,
                        'Mean': f"{arr.mean():.2f}",
                        'Std': f"{arr.std():.2f}",
                        'Skewness': f"{shape['skewness']:.2f}" if not np.isnan(shape['skewness']) else 'N/A',
                        'Outlier %': f"{outliers['outlier_pct']:.1f}%",
                        'Corr w/ Fire': f"{corr_fire:.3f}" if not np.isnan(corr_fire) else 'N/A'
                    })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        # Key recommendations
        st.subheader("📌 Key Recommendations")
        
        recommendations = []
        
        # Class imbalance check
        fire_ratio = fire_count / total if total > 0 else 0
        if fire_ratio < 0.3 or fire_ratio > 0.7:
            recommendations.append("**Class Imbalance**: Use stratified sampling, SMOTE oversampling, or adjust class weights in models.")
        
        # High skewness features
        skewed_features = []
        for col in numeric_cols[:15]:
            shape = compute_skewness_kurtosis(fdf[col])
            if abs(shape['skewness']) > 1:
                skewed_features.append(col)
        if skewed_features:
            recommendations.append(f"**Skewed Features**: {', '.join(skewed_features[:5])} show high skewness. Consider log or Box-Cox transformations.")
        
        # High outlier features
        outlier_features = []
        for col in numeric_cols[:15]:
            outliers = detect_outliers_summary(fdf[col], 'iqr')
            if outliers['outlier_pct'] > 10:
                outlier_features.append(col)
        if outlier_features:
            recommendations.append(f"**High Outliers**: {', '.join(outlier_features[:5])} have >10% outliers. Consider winsorization or robust scaling.")
        
        # Multicollinearity
        high_corr = identify_multicollinearity(fdf, numeric_cols[:20], threshold=0.85)
        if high_corr:
            recommendations.append(f"**Multicollinearity**: {len(high_corr)} feature pairs have |r| > 0.85. Consider feature selection or PCA.")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.success("✅ No major data quality issues detected. Dataset appears ready for modeling!")
        
        # Suggested modeling approaches
        st.subheader("🎯 Suggested Modeling Approaches")
        st.markdown("""
        Based on the dataset characteristics:
        
        1. **Random Forest / Gradient Boosting**: Robust to outliers, handles non-linear relationships, provides feature importance
        2. **Logistic Regression with Regularization**: Good baseline, interpretable coefficients, use L1 for feature selection
        3. **XGBoost / LightGBM**: Handle imbalanced data well with `scale_pos_weight`, efficient for large datasets
        4. **Spatial Cross-Validation**: Recommended due to geographic nature of fire data to avoid spatial autocorrelation leakage
        """)

# ---------------------------------------------------------------------------
# Page: Fire Classification
# ---------------------------------------------------------------------------
elif page == "Fire Classification":
    st.header("Fire Classification Analysis")
    st.caption("Train and evaluate machine learning models to predict fire occurrence.")

    # Check for sklearn
    try:
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report,
            roc_curve, precision_recall_curve, average_precision_score
        )
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        st.error("scikit-learn is required for this page. Install with: `pip install scikit-learn`")
        st.stop()

    # Load merged dataset
    def load_merged_for_classification() -> tuple[pd.DataFrame, Path | None]:
        path = locate_merged_dataset()
        if path is None:
            return pd.DataFrame(), None
        df = load_cached_merged_dataset(path.as_posix(), int(path.stat().st_mtime_ns))
        return df, path

    df_class, class_path = load_merged_for_classification()
    if df_class.empty:
        st.error("Merged dataset not found. Run geo_pipeline.py final-merge first.")
        st.stop()

    st.caption(f"Loaded: {class_path}")

    # Normalize fire column
    if 'fire' not in df_class.columns:
        st.error("No 'fire' column found in dataset.")
        st.stop()

    df_class['fire'] = normalize_fire_column(df_class, 'fire')

    # Check class distribution
    fire_counts = df_class['fire'].value_counts(dropna=False)
    n_fire = int(fire_counts.get(1, 0))
    n_nonfire = int(fire_counts.get(0, 0))

    if n_fire == 0 or n_nonfire == 0:
        st.error(f"Need both fire and non-fire samples. Found: fire={n_fire}, non-fire={n_nonfire}")
        st.stop()

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Samples", f"{len(df_class):,}")
    col_info2.metric("Fire (1)", f"{n_fire:,}")
    col_info3.metric("Non-Fire (0)", f"{n_nonfire:,}")

    # Identify feature columns
    exclude_cols = {'fire', 'latitude', 'longitude', 'country', 'source', 'confidence',
                    'acq_date', 'acq_time', 'dt', 'month', 'daynight', 'satellite',
                    'instrument', 'version', 'bright_t31', 'type', 'scan', 'track'}

    # Get numeric columns suitable for features
    feature_candidates = []
    for c in df_class.columns:
        if c.lower() in exclude_cols or c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_class[c]):
            non_null = df_class[c].notna().sum()
            if non_null > len(df_class) * 0.5:  # At least 50% non-null
                feature_candidates.append(c)

    if not feature_candidates:
        st.error("No suitable numeric feature columns found.")
        st.stop()

    # Sidebar controls
    st.sidebar.subheader("Model Settings")

    selected_features = st.sidebar.multiselect(
        "Select Features",
        options=sorted(feature_candidates),
        default=sorted(feature_candidates)[:min(10, len(feature_candidates))]
    )

    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        index=0
    )

    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 1000, 42)

    # Model-specific parameters
    if model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100, 10)
        max_depth = st.sidebar.slider("Max Depth", 2, 30, 10, 1)
        class_weight = st.sidebar.selectbox("Class Weight", ["balanced", "None"])
    elif model_type == "Gradient Boosting":
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
        max_depth = st.sidebar.slider("Max Depth", 2, 15, 5, 1)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    else:  # Logistic Regression
        C_param = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
        class_weight = st.sidebar.selectbox("Class Weight", ["balanced", "None"])

    balance_classes = st.sidebar.checkbox("Undersample Majority Class", value=False)

    if not selected_features:
        st.info("Select at least one feature to train a model.")
        st.stop()

    # Prepare data
    st.subheader("Data Preparation")

    # Create feature matrix
    df_model = df_class[selected_features + ['fire']].copy()

    # Convert to numeric and handle missing values
    for col in selected_features:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

    # Drop rows with missing target
    df_model = df_model.dropna(subset=['fire'])

    # Handle missing features - fill with median
    missing_before = df_model[selected_features].isna().sum().sum()
    for col in selected_features:
        if df_model[col].isna().any():
            median_val = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_val)

    st.write(f"**Samples after cleaning:** {len(df_model):,}")
    if missing_before > 0:
        st.caption(f"Filled {missing_before:,} missing feature values with median.")

    X = df_model[selected_features].values.astype(np.float64)
    y = df_model['fire'].values.astype(int)

    # Balance classes if requested
    if balance_classes:
        n_minority = min(np.sum(y == 0), np.sum(y == 1))
        idx_fire = np.where(y == 1)[0]
        idx_nonfire = np.where(y == 0)[0]

        np.random.seed(random_state)
        if len(idx_fire) > n_minority:
            idx_fire = np.random.choice(idx_fire, n_minority, replace=False)
        if len(idx_nonfire) > n_minority:
            idx_nonfire = np.random.choice(idx_nonfire, n_minority, replace=False)

        balanced_idx = np.concatenate([idx_fire, idx_nonfire])
        X = X[balanced_idx]
        y = y[balanced_idx]
        st.caption(f"Balanced dataset: {len(y):,} samples ({np.sum(y==1):,} fire, {np.sum(y==0):,} non-fire)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    col_split1, col_split2 = st.columns(2)
    col_split1.write(f"**Training set:** {len(X_train):,} samples")
    col_split2.write(f"**Test set:** {len(X_test):,} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    st.subheader("Model Training")

    train_button = st.button("🚀 Train Model", type="primary")

    if train_button:
        with st.spinner("Training model..."):
            # Create model
            if model_type == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight=class_weight if class_weight != "None" else None,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif model_type == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state
                )
            else:  # Logistic Regression
                model = LogisticRegression(
                    C=C_param,
                    class_weight=class_weight if class_weight != "None" else None,
                    random_state=random_state,
                    max_iter=1000,
                    n_jobs=-1
                )

            # Fit model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Store in session state
            st.session_state['fire_model'] = model
            st.session_state['fire_scaler'] = scaler
            st.session_state['fire_features'] = selected_features
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['y_pred_proba'] = y_pred_proba

        st.success("Model trained successfully!")

    # Display results if model exists
    if 'fire_model' in st.session_state:
        model = st.session_state['fire_model']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        y_pred_proba = st.session_state['y_pred_proba']

        st.subheader("Model Evaluation")

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        met1, met2, met3 = st.columns(3)
        met1.metric("Accuracy", f"{accuracy:.3f}")
        met2.metric("Precision", f"{precision:.3f}")
        met3.metric("Recall", f"{recall:.3f}")

        met4, met5, met6 = st.columns(3)
        met4.metric("F1 Score", f"{f1:.3f}")
        met5.metric("ROC AUC", f"{roc_auc:.3f}")
        met6.metric("Avg Precision", f"{avg_precision:.3f}")

        # Confusion Matrix and ROC Curve
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-Fire", "Fire"])
            ax.set_yticklabels(["Non-Fire", "Fire"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=text_color, fontsize=12)

            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_roc:
            st.markdown("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Precision-Recall Curve
        col_pr, col_fi = st.columns(2)

        with col_pr:
            st.markdown("**Precision-Recall Curve**")
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(recall_curve, precision_curve, color="#10b981", lw=2,
                    label=f"AP = {avg_precision:.3f}")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="upper right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_fi:
            st.markdown("**Feature Importance**")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': st.session_state['fire_features'],
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

                fig, ax = plt.subplots(figsize=(5, 4))
                y_pos = range(len(feat_imp))
                ax.barh(y_pos, feat_imp['Importance'].values, color="#6366f1", alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feat_imp['Feature'].values, fontsize=8)
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif hasattr(model, 'coef_'):
                coefs = np.abs(model.coef_[0])
                feat_imp = pd.DataFrame({
                    'Feature': st.session_state['fire_features'],
                    'Coefficient': coefs
                }).sort_values('Coefficient', ascending=True)

                fig, ax = plt.subplots(figsize=(5, 4))
                y_pos = range(len(feat_imp))
                ax.barh(y_pos, feat_imp['Coefficient'].values, color="#6366f1", alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feat_imp['Feature'].values, fontsize=8)
                ax.set_xlabel("|Coefficient|")
                ax.set_title("Feature Coefficients (Absolute)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Feature importance not available for this model.")

        # Classification Report
        with st.expander("Detailed Classification Report"):
            report = classification_report(y_test, y_pred, target_names=["Non-Fire", "Fire"])
            st.code(report)

        # Cross-validation
        st.subheader("Cross-Validation")
        run_cv = st.button("Run 5-Fold Cross-Validation")

        if run_cv:
            with st.spinner("Running cross-validation..."):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

                st.write(f"**ROC AUC Scores:** {', '.join([f'{s:.3f}' for s in cv_scores])}")
                st.write(f"**Mean:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(1, 6), cv_scores, color="#3b82f6", alpha=0.8)
                ax.axhline(cv_scores.mean(), color="#ef4444", linestyle="--", label=f"Mean: {cv_scores.mean():.3f}")
                ax.set_xlabel("Fold")
                ax.set_ylabel("ROC AUC")
                ax.set_title("Cross-Validation Scores")
                ax.set_ylim([0, 1])
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Prediction on new data
        st.subheader("Make Predictions")
        st.caption("Enter values for each feature to predict fire probability.")

        with st.expander("Predict on Custom Input"):
            input_values = {}
            input_cols = st.columns(3)
            for i, feat in enumerate(st.session_state['fire_features']):
                col_idx = i % 3
                with input_cols[col_idx]:
                    # Get feature statistics for default/range
                    feat_col = df_model[feat] if feat in df_model.columns else df_class[feat]
                    feat_min = float(feat_col.min())
                    feat_max = float(feat_col.max())
                    feat_mean = float(feat_col.mean())
                    input_values[feat] = st.number_input(
                        feat,
                        min_value=feat_min,
                        max_value=feat_max,
                        value=feat_mean,
                        key=f"input_{feat}"
                    )

            if st.button("Predict"):
                input_array = np.array([[input_values[f] for f in st.session_state['fire_features']]])
                input_scaled = st.session_state['fire_scaler'].transform(input_array)
                pred_proba = model.predict_proba(input_scaled)[0]
                pred_class = model.predict(input_scaled)[0]

                st.write("---")
                res1, res2 = st.columns(2)
                res1.metric("Predicted Class", "🔥 Fire" if pred_class == 1 else "✅ Non-Fire")
                res2.metric("Fire Probability", f"{pred_proba[1]:.1%}")

                # Probability bar
                fig, ax = plt.subplots(figsize=(6, 1))
                ax.barh([0], [pred_proba[0]], color="#10b981", label="Non-Fire")
                ax.barh([0], [pred_proba[1]], left=[pred_proba[0]], color="#ef4444", label="Fire")
                ax.set_xlim([0, 1])
                ax.set_yticks([])
                ax.set_xlabel("Probability")
                ax.legend(loc="upper right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# ---------------------------------------------------------------------------
# Page: Outliers
# ---------------------------------------------------------------------------
elif page == "Outliers":
    st.header("Outlier Detection")
    st.caption("Detect and analyze outliers in the merged dataset using multiple methods.")

    # Load merged dataset
    def load_merged_for_outliers() -> tuple[pd.DataFrame, Path | None]:
        path = locate_merged_dataset()
        if path is None:
            return pd.DataFrame(), None
        df = load_cached_merged_dataset(path.as_posix(), int(path.stat().st_mtime_ns))
        return df, path

    df_outliers, outlier_path = load_merged_for_outliers()
    if df_outliers.empty:
        st.error("Merged dataset not found. Run geo_pipeline.py final-merge first.")
        st.stop()

    st.caption(f"Loaded: {outlier_path}")

    # Identify numeric columns suitable for outlier detection
    numeric_cols = [c for c in df_outliers.columns
                    if pd.api.types.is_numeric_dtype(df_outliers[c])
                    and c not in ("latitude", "longitude", "fire", "soil_id")
                    and df_outliers[c].notna().sum() > 10]

    if not numeric_cols:
        st.warning("No suitable numeric columns found for outlier detection.")
        st.stop()

    # Sidebar controls
    st.sidebar.subheader("Outlier Settings")
    selected_cols = st.sidebar.multiselect(
        "Columns to analyze",
        options=sorted(numeric_cols),
        default=sorted(numeric_cols)[:5] if len(numeric_cols) >= 5 else sorted(numeric_cols)
    )

    method = st.sidebar.selectbox(
        "Detection Method",
        ["IQR (Interquartile Range)", "Z-Score", "Modified Z-Score (MAD)", "Percentile"],
        index=0
    )

    # Method-specific parameters
    if method == "IQR (Interquartile Range)":
        iqr_factor = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
    elif method == "Z-Score":
        z_threshold = st.sidebar.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
    elif method == "Modified Z-Score (MAD)":
        mad_threshold = st.sidebar.slider("MAD Threshold", 2.0, 5.0, 3.5, 0.1)
    elif method == "Percentile":
        lower_pct = st.sidebar.slider("Lower Percentile", 0.0, 10.0, 1.0, 0.5)
        upper_pct = st.sidebar.slider("Upper Percentile", 90.0, 100.0, 99.0, 0.5)

    if not selected_cols:
        st.info("Select at least one column to analyze.")
        st.stop()

    # Outlier detection functions
    def detect_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method."""
        # Ensure float64 to avoid boolean/integer issues with quantile
        s = series.astype(np.float64)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return (s < lower) | (s > upper)

    def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        s = series.astype(np.float64)
        mean = s.mean()
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=series.index)
        z_scores = np.abs((s - mean) / std)
        return z_scores > threshold

    def detect_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Detect outliers using Modified Z-score (MAD) method."""
        s = series.astype(np.float64)
        median = s.median()
        mad = np.median(np.abs(s - median))
        if mad == 0:
            return pd.Series(False, index=series.index)
        modified_z = 0.6745 * (s - median) / mad
        return np.abs(modified_z) > threshold

    def detect_percentile(series: pd.Series, lower: float = 1.0, upper: float = 99.0) -> pd.Series:
        """Detect outliers using percentile method."""
        s = series.astype(np.float64)
        lower_bound = s.quantile(lower / 100)
        upper_bound = s.quantile(upper / 100)
        return (s < lower_bound) | (s > upper_bound)

    # Run outlier detection
    st.subheader("Outlier Analysis Results")

    outlier_summary = []
    outlier_masks = {}

    for col in selected_cols:
        series = pd.to_numeric(df_outliers[col], errors='coerce').dropna().astype(np.float64)
        if series.empty:
            continue

        if method == "IQR (Interquartile Range)":
            mask = detect_iqr(series, iqr_factor)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            bounds = (q1 - iqr_factor * iqr, q3 + iqr_factor * iqr)
        elif method == "Z-Score":
            mask = detect_zscore(series, z_threshold)
            bounds = (series.mean() - z_threshold * series.std(),
                      series.mean() + z_threshold * series.std())
        elif method == "Modified Z-Score (MAD)":
            mask = detect_mad(series, mad_threshold)
            median = series.median()
            mad = np.median(np.abs(series - median))
            bounds = (median - mad_threshold * mad / 0.6745,
                      median + mad_threshold * mad / 0.6745)
        elif method == "Percentile":
            mask = detect_percentile(series, lower_pct, upper_pct)
            bounds = (series.quantile(lower_pct / 100), series.quantile(upper_pct / 100))

        n_outliers = mask.sum()
        pct_outliers = 100.0 * n_outliers / len(series)

        outlier_masks[col] = mask
        outlier_summary.append({
            "Column": col,
            "Total": len(series),
            "Outliers": int(n_outliers),
            "% Outliers": f"{pct_outliers:.2f}%",
            "Lower Bound": f"{bounds[0]:.4g}",
            "Upper Bound": f"{bounds[1]:.4g}",
            "Min": f"{series.min():.4g}",
            "Max": f"{series.max():.4g}",
        })

    # Display summary table
    if outlier_summary:
        summary_df = pd.DataFrame(outlier_summary)
        st.dataframe(summary_df, use_container_width=True)

        # Total outliers across all columns
        total_outlier_rows = pd.Series(False, index=df_outliers.index)
        for col, mask in outlier_masks.items():
            total_outlier_rows.loc[mask.index] |= mask

        st.metric("Rows with any outlier", f"{total_outlier_rows.sum():,} / {len(df_outliers):,}")

    # Visualization
    st.subheader("Visualizations")

    vis_col = st.selectbox("Select column for detailed view", selected_cols, index=0)

    if vis_col:
        series = pd.to_numeric(df_outliers[vis_col], errors='coerce').dropna()
        mask = outlier_masks.get(vis_col, pd.Series(False, index=series.index))

        col1, col2 = st.columns(2)

        with col1:
            # Histogram with outliers highlighted
            fig, ax = plt.subplots(figsize=(6, 4))
            inliers = series[~mask]
            outliers = series[mask]

            # Determine bins from full data
            bins = np.histogram_bin_edges(series.values, bins=50)

            ax.hist(inliers.values, bins=bins, color="#3b82f6", alpha=0.7, label=f"Inliers ({len(inliers):,})")
            if len(outliers) > 0:
                ax.hist(outliers.values, bins=bins, color="#ef4444", alpha=0.8, label=f"Outliers ({len(outliers):,})")
            ax.set_title(f"Distribution — {vis_col}")
            ax.set_xlabel(vis_col)
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Box plot
            fig, ax = plt.subplots(figsize=(6, 4))
            bp = ax.boxplot(series.values, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="#3b82f6", alpha=0.6),
                            flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=4, alpha=0.5))
            ax.set_title(f"Boxplot — {vis_col}")
            ax.set_ylabel(vis_col)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Scatter plot: outliers vs index (to see distribution)
        st.markdown("**Outlier Distribution by Index**")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(series[~mask].index, series[~mask].values, s=2, alpha=0.3, c="#3b82f6", label="Inliers")
        if mask.sum() > 0:
            ax.scatter(series[mask].index, series[mask].values, s=10, alpha=0.8, c="#ef4444", label="Outliers")
        ax.set_xlabel("Row Index")
        ax.set_ylabel(vis_col)
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Statistics comparison
        st.markdown("**Statistics: Inliers vs Outliers**")
        if mask.sum() > 0:
            stats_compare = pd.DataFrame({
                "Metric": ["Count", "Mean", "Std", "Min", "Median", "Max"],
                "All Data": [
                    len(series),
                    f"{series.mean():.4g}",
                    f"{series.std():.4g}",
                    f"{series.min():.4g}",
                    f"{series.median():.4g}",
                    f"{series.max():.4g}",
                ],
                "Inliers": [
                    len(inliers),
                    f"{inliers.mean():.4g}",
                    f"{inliers.std():.4g}",
                    f"{inliers.min():.4g}",
                    f"{inliers.median():.4g}",
                    f"{inliers.max():.4g}",
                ],
                "Outliers": [
                    len(outliers),
                    f"{outliers.mean():.4g}" if len(outliers) > 0 else "N/A",
                    f"{outliers.std():.4g}" if len(outliers) > 1 else "N/A",
                    f"{outliers.min():.4g}" if len(outliers) > 0 else "N/A",
                    f"{outliers.median():.4g}" if len(outliers) > 0 else "N/A",
                    f"{outliers.max():.4g}" if len(outliers) > 0 else "N/A",
                ],
            })
            st.table(stats_compare)
        else:
            st.info("No outliers detected for this column with current settings.")

    # Correlation heatmap of outlier counts
    if len(selected_cols) > 1:
        st.subheader("Outlier Co-occurrence")
        st.caption("Which columns tend to have outliers in the same rows?")

        # Build binary outlier matrix
        outlier_binary = pd.DataFrame(index=df_outliers.index)
        for col in selected_cols:
            if col in outlier_masks:
                mask = outlier_masks[col]
                outlier_binary[col] = False
                outlier_binary.loc[mask.index, col] = mask.values

        # Compute co-occurrence matrix
        cooccur = outlier_binary.astype(int).T.dot(outlier_binary.astype(int))

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cooccur.values, cmap="YlOrRd")
        ax.set_xticks(range(len(cooccur.columns)))
        ax.set_yticks(range(len(cooccur.index)))
        ax.set_xticklabels(cooccur.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cooccur.index, fontsize=8)
        ax.set_title("Outlier Co-occurrence Matrix")
        fig.colorbar(im, ax=ax, label="Shared Outlier Rows")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Export outliers
    st.subheader("Export Outliers")
    if outlier_masks:
        # Combine all outlier masks
        combined_mask = pd.Series(False, index=df_outliers.index)
        for col, mask in outlier_masks.items():
            combined_mask.loc[mask.index] |= mask

        outlier_rows = df_outliers.loc[combined_mask]

        if not outlier_rows.empty:
            st.write(f"Found **{len(outlier_rows):,}** rows with outliers.")

            # Add column indicating which columns had outliers
            outlier_cols_list = []
            for idx in outlier_rows.index:
                cols_with_outliers = [c for c, m in outlier_masks.items() if idx in m.index and m.loc[idx]]
                outlier_cols_list.append(", ".join(cols_with_outliers))
            outlier_rows = outlier_rows.copy()
            outlier_rows["outlier_columns"] = outlier_cols_list

            st.dataframe(outlier_rows.head(100), use_container_width=True)
            st.caption("Showing first 100 outlier rows.")

            csv_bytes = outlier_rows.to_csv(index=False).encode()
            st.download_button(
                "Download Outlier Rows (CSV)",
                data=csv_bytes,
                file_name="outlier_rows.csv",
                mime="text/csv",
            )

            # Option to download cleaned dataset (without outliers)
            clean_rows = df_outliers.loc[~combined_mask]
            st.write(f"Clean dataset (outliers removed): **{len(clean_rows):,}** rows")
            clean_csv = clean_rows.to_csv(index=False).encode()
            st.download_button(
                "Download Cleaned Dataset (CSV)",
                data=clean_csv,
                file_name="merged_dataset_no_outliers.csv",
                mime="text/csv",
            )
        else:
            st.success("No outliers detected with current settings.")

# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
else:
    st.info(f"Page '{page}' is not yet implemented.")