#!/usr/bin/env python3
"""Sequential environmental dataset merger for Fire classification.

Starts from a fire (and non-fire) point dataset then enriches it in order:
  1. Climate features (stats or seasonal) --> add columns
  2. Elevation features (raster sample or nearest table rows)
  3. Soil features
  4. Landcover (polygon containment or nearest centroid)
Each step uses the result of the previous merge as the base for the next, enabling
progressive enrichment and optional intermediate saves.

Spatial Resolution Reconciliation Strategies:
  - nearest: Take single nearest point in source dataset (KDTree)
  - knn: Take mean of numeric columns over k nearest neighbors; categorical columns use mode
  - raster: Sample raster/NetCDF at point coordinates (climate/elevation only)
  - polygon: Point-in-polygon join (landcover shapefile)
  - centroid-nearest: Use polygon centroids (landcover) with nearest or knn

CLI Examples:
  python scripts/merge_env.py --auto --save-csv --global-fill --fill-strategy median
  python scripts/merge_env.py --fire-path DATA_CLEANED/processed/fire_with_negatives.parquet \
      --climate-method nearest --soil-method knn --soil-k 3 --elev-method raster --landcover-method polygon --summary

Outputs:
  DATA_CLEANED/processed/merged_dataset.parquet (default)
  DATA_CLEANED/processed/merged_dataset.csv (if --save-csv)
  DATA_CLEANED/processed/merged_dataset_summary.json (if --summary)

Assumptions:
  - Fire dataset has latitude/longitude columns and a binary label column 'fire' (1=fire,0=non-fire).
  - Climate/soil/elevation feature tables contain lat/lon columns OR x/y convertible.
  - Landcover shapefile in EPSG:4326 or convertible.

Performance Notes:
  - Elevation feature table may be very large; automatic subsampling invoked if rows > threshold unless raster sampling selected.
  - KDTree built in float32 for memory efficiency.

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd

try:
    import xarray as xr
    import rioxarray as rxr  # optional for raster sampling
except ImportError:
    xr = None
    rxr = None

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None


    class Point:  # minimal stub
        def __init__(self, x, y):
            self.x = x
            self.y = y

try:
    from scipy.spatial import cKDTree as _CKDTree
except ImportError:
    _CKDTree = None


# --------------- Constants -----------------

# Soil features to keep - reduced set after removing highly correlated pairs:
# Removed: teb (r=0.99 with cec_eff), ref_bulk (r=0.90 with silt), 
#          bulk (r=0.91 with ph_water - keep ph_water as more interpretable)
# Kept texture_usda over individual sand/silt/clay for dimensionality reduction
SOIL_FEATURES_TO_KEEP = {
    'coarse', 'sand', 'silt', 'clay', 'texture_usda', 'texture_soter',
    'org_carbon', 'ph_water', 'total_n', 'cn_ratio',
    'cec_soil', 'cec_clay', 'cec_eff', 'bsat', 'alum_sat', 'esp',
    'tcarbon_eq', 'gypsum', 'elec_cond'
}
# Dropped: 'bulk', 'ref_bulk', 'teb' (highly correlated with other features)


# --------------- Utility -----------------

def robust_read_file(path: Path) -> pd.DataFrame:
    """Load parquet or CSV with fallback for corrupted parquet files."""
    if path.suffix == '.parquet':
        try:
            return pd.read_parquet(path, engine='pyarrow')
        except OSError as e:
            if "Repetition level histogram" in str(e) or "corrupted" in str(e).lower():
                print(f"Warning: {path.name} corrupted, trying fastparquet...")
                try:
                    return pd.read_parquet(path, engine='fastparquet')
                except Exception:
                    pass
                # Try CSV fallback
                csv_path = path.with_suffix('.csv')
                if csv_path.exists():
                    print(f"Loading CSV fallback: {csv_path}")
                    return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


def filter_soil_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Filter soil dataframe to keep only the specified features + lat/lon/soil_id."""
    # Always keep coordinate and ID columns
    keep_always = {'lat', 'lon', 'latitude', 'longitude', 'soil_id'}
    
    # Build list of columns to keep (case-insensitive matching)
    cols_to_keep = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in keep_always:
            cols_to_keep.append(col)
        elif col_lower in SOIL_FEATURES_TO_KEEP:
            cols_to_keep.append(col)
    
    # Remove duplicates while preserving order
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    
    dropped = set(df.columns) - set(cols_to_keep)
    if dropped:
        print(f"Filtering soil columns: keeping {len(cols_to_keep)}, dropping {len(dropped)} "
              f"(e.g., {list(dropped)[:5]}...)")
    
    return df[cols_to_keep].copy()


def _coerce_latlon(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = None
    lon_col = None
    for cand in ['latitude', 'lat']:
        if cand in cols_lower:
            lat_col = cols_lower[cand]
            break
    for cand in ['longitude', 'lon', 'lng']:
        if cand in cols_lower:
            lon_col = cols_lower[cand]
            break
    if lat_col is None or lon_col is None:
        raise ValueError("DataFrame missing latitude/longitude columns")
    out = df.copy()
    out['latitude'] = pd.to_numeric(out[lat_col], errors='coerce')
    out['longitude'] = pd.to_numeric(out[lon_col], errors='coerce')
    return out.dropna(subset=['latitude', 'longitude']).copy()


def _build_kdtree(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.vstack([df['latitude'].to_numpy(dtype=np.float32), df['longitude'].to_numpy(dtype=np.float32)]).T
    return arr, np.arange(arr.shape[0])


def _nearest(source_coords: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    t_coords = np.vstack([target_lat, target_lon]).T.astype(np.float32)
    idx_out = np.empty(t_coords.shape[0], dtype=np.int64)
    dist_out = np.empty(t_coords.shape[0], dtype=np.float32)
    for i, (la, lo) in enumerate(t_coords):
        d2 = (source_coords[:, 0] - la) ** 2 + (source_coords[:, 1] - lo) ** 2
        j = int(np.argmin(d2))
        idx_out[i] = j
        dist_out[i] = np.sqrt(d2[j])
    return idx_out, dist_out


def _knn_indices(source_coords: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray, k: int) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    k = max(1, k)
    t_coords = np.vstack([target_lat, target_lon]).T.astype(np.float32)
    out_idx: List[np.ndarray] = []
    out_dist: List[np.ndarray] = []
    for (la, lo) in t_coords:
        d2 = (source_coords[:, 0] - la) ** 2 + (source_coords[:, 1] - lo) ** 2
        sel = np.argpartition(d2, k - 1)[:k]
        out_idx.append(sel)
        out_dist.append(np.sqrt(d2[sel]))
    return out_idx, out_dist


def _aggregate_knn(df: pd.DataFrame, idx_list: List[np.ndarray], dist_list: List[np.ndarray], weighting: str) -> Dict[
    str, List]:
    numeric_cols = [c for c in df.columns if
                    pd.api.types.is_numeric_dtype(df[c]) and c not in ['latitude', 'longitude']]
    cat_cols = [c for c in df.columns if
                not pd.api.types.is_numeric_dtype(df[c]) and c not in ['latitude', 'longitude']]
    res: Dict[str, List] = {c: [] for c in numeric_cols + cat_cols}
    res['__knn_valid_count'] = []
    res['__knn_mean_dist'] = []
    for idx, dists in zip(idx_list, dist_list):
        subset = df.iloc[idx]
        weights = None
        wsum = None
        if weighting == 'idw':
            weights = 1.0 / (dists + 1e-6)
            wsum = float(weights.sum())
            if wsum <= 0:
                weights = None
        num_valid = 0
        for c in numeric_cols:
            vals = pd.to_numeric(subset[c], errors='coerce')
            finite = vals[np.isfinite(vals)]
            if finite.size:
                num_valid += finite.size
                if isinstance(weights, np.ndarray) and weights.size and weights.sum() > 0:
                    w_local = weights[np.isfinite(vals)]
                    if isinstance(w_local, np.ndarray) and w_local.size and w_local.sum() > 0:
                        res[c].append(float(np.dot(finite, w_local / w_local.sum())))
                    else:
                        res[c].append(float(np.nanmean(finite)))
                else:
                    res[c].append(float(np.nanmean(finite)))
            else:
                res[c].append(np.nan)
        for c in cat_cols:
            vals = subset[c].dropna().astype(str)
            res[c].append(vals.value_counts().idxmax() if not vals.empty else None)
        res['__knn_valid_count'].append(int(num_valid))
        res['__knn_mean_dist'].append(float(dists.mean()) if dists.size else np.nan)
    return res


def _raster_sample(raster_path: Path, lat: np.ndarray, lon: np.ndarray) -> Dict[str, List]:
    if xr is None or rxr is None:
        raise RuntimeError("xarray/rioxarray not available for raster sampling")
    da = rxr.open_rasterio(raster_path, masked=True).squeeze()
    # Ensure CRS EPSG:4326
    try:
        if str(da.rio.crs) not in ['EPSG:4326', 'CRS.from_epsg(4326)']:
            da = da.rio.reproject('EPSG:4326')
    except Exception:
        pass
    # Build grid coordinate arrays if present
    xs = da['x'].values if 'x' in da.coords else None
    ys = da['y'].values if 'y' in da.coords else None
    arr = da.values.astype(float)
    if xs is None or ys is None:
        raise ValueError("Raster missing x/y coordinates for sampling")
    # Map lon to x index, lat to y index (nearest)
    x_idx = np.array([int(np.argmin(np.abs(xs - lo))) for lo in lon])
    y_idx = np.array([int(np.argmin(np.abs(ys - la))) for la in lat])
    sampled = arr[y_idx, x_idx]
    return {da.name if da.name else 'raster_val': sampled.tolist()}


def _netcdf_sample(nc_path: Path, var_names: List[str], lat: np.ndarray, lon: np.ndarray,
                   month: Optional[str] = None) -> Dict[str, List]:
    if xr is None:
        raise RuntimeError("xarray not available for NetCDF sampling")
    ds = xr.open_dataset(nc_path)
    # Ensure lat/lon coordinate names
    lat_name = 'lat' if 'lat' in ds.coords else ('y' if 'y' in ds.coords else None)
    lon_name = 'lon' if 'lon' in ds.coords else ('x' if 'x' in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise ValueError("NetCDF missing lat/lon or y/x coordinates")
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    y_idx = np.array([int(np.argmin(np.abs(lat_vals - la))) for la in lat])
    x_idx = np.array([int(np.argmin(np.abs(lon_vals - lo))) for lo in lon])
    out: Dict[str, List] = {}
    for v in var_names:
        if v not in ds.data_vars:
            continue
        da = ds[v]
        if 'time' in da.dims and month is not None:
            # match month string YYYY-MM if possible
            times = pd.to_datetime(ds['time'].values)
            sel_idx = 0
            for i, t in enumerate(times):
                t_val = pd.to_datetime(t)
                if month is not None and t_val.strftime('%Y-%m') == month:
                    sel_idx = i
                    break
            da = da.isel(time=sel_idx)
        arr = da.values
        sampled = arr[y_idx, x_idx]
        out[v] = sampled.tolist()
    return out


def _point_in_polygon(points_df: pd.DataFrame, poly_gdf: gpd.GeoDataFrame, attr_cols: List[str], method: str) -> Dict[
    str, List]:
    if gpd is None:
        raise RuntimeError("geopandas not available for polygon operations")
    # Build Point geometries
    pts_gdf = gpd.GeoDataFrame(points_df[['latitude', 'longitude']],
                               geometry=[Point(xy) for xy in zip(points_df['longitude'], points_df['latitude'])],
                               crs='EPSG:4326')
    poly_gdf = poly_gdf.to_crs('EPSG:4326')
    joined = gpd.sjoin(pts_gdf, poly_gdf, predicate='within', how='left')
    res: Dict[str, List] = {}
    for c in attr_cols:
        if c in joined.columns:
            res[c] = joined[c].tolist()
    return res


# --------------- Correlation-Based Feature Selection -----------------

def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.95, 
                           protected_cols: Optional[Set[str]] = None,
                           export_path: Optional[Path] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """Drop highly correlated features to reduce multicollinearity.
    
    For each pair with |correlation| >= threshold, drops the second feature
    (alphabetically) unless it's in the protected set.
    
    Args:
        df: DataFrame with numeric features
        threshold: Correlation threshold (default 0.95)
        protected_cols: Set of column names to never drop
        export_path: Optional path to export correlation pairs CSV
        
    Returns:
        Tuple of (filtered DataFrame, list of dropped pairs info)
    """
    if protected_cols is None:
        protected_cols = {'fire', 'latitude', 'longitude', 'lat', 'lon'}
    
    # Get numeric columns only
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return df, []
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if upper_tri[i, j] and corr_matrix.iloc[i, j] >= threshold:
                high_corr_pairs.append({
                    'feature_1': col1,
                    'feature_2': col2,
                    'correlation': round(corr_matrix.iloc[i, j], 3)
                })
    
    # Sort by correlation descending
    high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
    
    # Export if path provided
    if export_path and high_corr_pairs:
        export_df = pd.DataFrame(high_corr_pairs)
        export_df.to_csv(export_path, index=True)
        print(f"Exported {len(high_corr_pairs)} high correlation pairs to {export_path}")
    
    # Determine columns to drop
    to_drop = set()
    dropped_info = []
    
    for pair in high_corr_pairs:
        col1, col2 = pair['feature_1'], pair['feature_2']
        
        # Skip if both already marked for drop or both protected
        if col1 in to_drop or col2 in to_drop:
            continue
        if col1 in protected_cols and col2 in protected_cols:
            continue
            
        # Prefer dropping the one NOT in protected
        if col1 in protected_cols:
            drop_col = col2
            keep_col = col1
        elif col2 in protected_cols:
            drop_col = col1
            keep_col = col2
        else:
            # Both unprotected: drop alphabetically second one
            drop_col = max(col1, col2)
            keep_col = min(col1, col2)
        
        to_drop.add(drop_col)
        dropped_info.append({
            'dropped': drop_col,
            'kept': keep_col,
            'correlation': pair['correlation']
        })
    
    # Drop columns
    if to_drop:
        df_filtered = df.drop(columns=list(to_drop), errors='ignore')
        print(f"Dropped {len(to_drop)} highly correlated features (threshold={threshold}):")
        for info in dropped_info[:10]:  # Show first 10
            print(f"  - {info['dropped']} (r={info['correlation']:.3f} with {info['kept']})")
        if len(dropped_info) > 10:
            print(f"  ... and {len(dropped_info) - 10} more")
    else:
        df_filtered = df
        print(f"No features dropped (no pairs with |r| >= {threshold})")
    
    return df_filtered, dropped_info


# --------------- Merge Functions -----------------

def _reduce_source_df(df: pd.DataFrame, max_points: int, grid_deg: Optional[float]) -> Tuple[
    pd.DataFrame, Dict[str, int]]:
    """Reduce large source dataframe to avoid memory errors.
    Strategy:
      1. If grid_deg provided: snap lat/lon to grid and keep first row per cell.
      2. If still > max_points: random sample down to max_points.
    Returns reduced df and stats.
    """
    stats = {'original': len(df), 'grid_applied': 0, 'sampled': 0}
    out = df
    if grid_deg and grid_deg > 0:
        glat = (out['latitude'] / grid_deg).round().astype(int)
        glon = (out['longitude'] / grid_deg).round().astype(int)
        out = out.assign(_g_lat=glat, _g_lon=glon)
        out = out.sort_values(['_g_lat', '_g_lon']).drop_duplicates(['_g_lat', '_g_lon']).drop(
            columns=['_g_lat', '_g_lon'])
        stats['grid_applied'] = 1
    if len(out) > max_points:
        out = out.sample(max_points, random_state=42).reset_index(drop=True)
        stats['sampled'] = 1
    stats['final'] = len(out)
    return out, stats


def merge_step(base_df: pd.DataFrame, source_df: pd.DataFrame, method: str, k: int = 5, tag: str = '',
               raster_path: Optional[Path] = None, add_distance: bool = False, weighting: str = 'uniform',
               max_source_points: int = 800000, grid_deg: Optional[float] = None,
               steps_log: Optional[List[Dict]] = None,
               update_existing: bool = False, only_fill_missing: bool = False, force_no_prefix: bool = False) -> pd.DataFrame:
    """Spatial merge helper.

    New parameters:
      update_existing: if True, do not create new columns when name already exists; instead optionally fill missing.
      only_fill_missing: if True (requires update_existing), only assign values where target is NaN.
      force_no_prefix: if True, ignore tag for feature columns (still uses tag for distance metrics) to avoid duplicates.
    """
    method = method.lower()
    b = base_df.copy()
    source_df = _coerce_latlon(source_df)
    source_df = source_df.drop_duplicates(subset=['latitude', 'longitude']).reset_index(drop=True)
    # Reduce if large
    if len(source_df) > max_source_points or (grid_deg and grid_deg > 0):
        reduced_df, red_stats = _reduce_source_df(source_df, max_source_points, grid_deg)
        if steps_log is not None:
            steps_log.append({'stage': f'{tag}_reduction', **red_stats})
        source_df = reduced_df
    coords = np.vstack(
        [source_df['latitude'].to_numpy(dtype=np.float32), source_df['longitude'].to_numpy(dtype=np.float32)]).T
    lat = b['latitude'].to_numpy(dtype=np.float32)
    lon = b['longitude'].to_numpy(dtype=np.float32)

    def _assign(col_name: str, values: np.ndarray | List):
        # Helper to assign respecting update_existing / only_fill_missing
        if isinstance(values, list):
            arr = values
        else:
            arr = values.tolist() if hasattr(values, 'tolist') else list(values)
        if update_existing and col_name in b.columns:
            if only_fill_missing:
                mask = b[col_name].isna()
                # Ensure length matches
                if len(arr) == len(b):
                    b.loc[mask, col_name] = [arr[i] for i in range(len(b)) if mask.iloc[i]]
                else:
                    # Fallback: assign entire column if shape mismatch
                    b[col_name] = arr
            else:
                # Overwrite entirely
                if len(arr) == len(b):
                    b[col_name] = arr
        else:
            b[col_name] = arr

    def _feature_name(orig: str) -> str:
        if force_no_prefix:
            return orig
        if tag and orig not in ['fire'] and orig not in ['latitude', 'longitude']:
            return f"{tag}_{orig}"
        return orig

    if method in ['nearest', 'knn'] and _CKDTree is not None:
        tree = _CKDTree(coords)
        if method == 'nearest':
            dist, idx = tree.query(np.vstack([lat, lon]).T, k=1)
            sel = source_df.iloc[idx].reset_index(drop=True)
            for c in sel.columns:
                if c in ['latitude', 'longitude']:
                    continue
                new_name = _feature_name(c)
                # Collect values (must align with b length)
                vals = sel[c].values
                _assign(new_name, vals)
            if add_distance:
                dist_col = f'{tag}_dist' if tag else 'dist'
                _assign(dist_col, dist.astype(np.float32))
        else:  # knn
            dist, idx = tree.query(np.vstack([lat, lon]).T, k=k)
            if k == 1:
                dist = dist[:, None]; idx = idx[:, None]
            numeric_cols = [c for c in source_df.columns if pd.api.types.is_numeric_dtype(source_df[c]) and c not in ['latitude', 'longitude']]
            cat_cols = [c for c in source_df.columns if not pd.api.types.is_numeric_dtype(source_df[c]) and c not in ['latitude', 'longitude']]
            if weighting == 'idw':
                w = 1.0 / (dist + 1e-6)
                w_sum = w.sum(axis=1, keepdims=True)
                w_norm = np.where(w_sum > 0, w / w_sum, 0)
            else:
                w_norm = np.ones_like(dist) / dist.shape[1]
            knn_valid_counts = np.zeros(dist.shape[0], dtype=np.int32)
            neighbor_valid_any = np.zeros_like(dist, dtype=bool)
            for c in numeric_cols:
                vals = source_df[c].to_numpy()
                gathered = vals[idx].astype(float, copy=False)
                mask = np.isfinite(gathered)
                knn_valid_counts += mask.sum(axis=1).astype(np.int32)
                neighbor_valid_any |= mask
                gathered = np.where(mask, gathered, np.nan)
                weighted = np.nansum(gathered * w_norm, axis=1)
                new_name = _feature_name(c)
                _assign(new_name, weighted)
            for c in cat_cols:
                vals = source_df[c].astype(str).to_numpy()
                gathered_cat = vals[idx]
                modes = []
                for row in gathered_cat:
                    vc = pd.Series(row).value_counts()
                    modes.append(vc.index[0] if not vc.empty else None)
                new_name = _feature_name(c)
                _assign(new_name, modes)
            if add_distance:
                _assign(f'{tag}_knn_mean_dist' if tag else 'knn_mean_dist', dist.mean(axis=1))
                _assign(f'{tag}_knn_valid' if tag else 'knn_valid', knn_valid_counts.tolist())
                valid_neighbors = neighbor_valid_any.sum(axis=1).astype(int)
                _assign(f'{tag}_knn_valid_neighbors' if tag else 'knn_valid_neighbors', valid_neighbors.tolist())
                _assign(f'{tag}_knn_valid_ratio' if tag else 'knn_valid_ratio', (valid_neighbors / float(dist.shape[1])).tolist())
    elif method == 'nearest':  # brute force fallback
        sel_rows = []
        dists_rows = []
        for i in range(len(lat)):
            la = lat[i]; lo = lon[i]
            diff_lat = coords[:, 0] - la
            diff_lon = coords[:, 1] - lo
            d2 = diff_lat * diff_lat + diff_lon * diff_lon
            j = int(np.argmin(d2))
            sel_rows.append(j); dists_rows.append(np.sqrt(d2[j]))
        sel = source_df.iloc[sel_rows].reset_index(drop=True)
        for c in sel.columns:
            if c in ['latitude', 'longitude']:
                continue
            new_name = _feature_name(c)
            _assign(new_name, sel[c].values)
        if add_distance:
            _assign(f'{tag}_dist' if tag else 'dist', np.array(dists_rows, dtype=np.float32))
    elif method == 'knn':  # brute force knn fallback
        k = max(1, k)
        numeric_cols = [c for c in source_df.columns if pd.api.types.is_numeric_dtype(source_df[c]) and c not in ['latitude', 'longitude']]
        cat_cols = [c for c in source_df.columns if not pd.api.types.is_numeric_dtype(source_df[c]) and c not in ['latitude', 'longitude']]
        gathered_numeric = {c: [] for c in numeric_cols}; gathered_cat = {c: [] for c in cat_cols}
        mean_dists = []; valid_counts = []; valid_neighbors_counts = []
        for i in range(len(lat)):
            la = lat[i]; lo = lon[i]
            diff_lat = coords[:, 0] - la; diff_lon = coords[:, 1] - lo
            d2 = diff_lat * diff_lat + diff_lon * diff_lon
            if k < len(d2):
                nn_idx = np.argpartition(d2, k - 1)[:k]
            else:
                nn_idx = np.arange(len(d2))
            nn_d = np.sqrt(d2[nn_idx])
            mean_dists.append(float(nn_d.mean()))
            if weighting == 'idw':
                w = 1.0 / (nn_d + 1e-6); w /= w.sum() if w.sum() > 0 else 1.0
            else:
                w = np.ones_like(nn_d) / len(nn_d)
            vc_total = 0; neighbor_any = np.zeros_like(nn_d, dtype=bool)
            for c in numeric_cols:
                vals = pd.to_numeric(source_df.iloc[nn_idx][c], errors='coerce').to_numpy()
                mask = np.isfinite(vals)
                vc_total += int(mask.sum()); neighbor_any |= mask
                gathered_numeric[c].append(float(np.nansum(vals[mask] * w[mask]) if mask.any() else np.nan))
            for c in cat_cols:
                vals = source_df.iloc[nn_idx][c].dropna().astype(str)
                gathered_cat[c].append(vals.value_counts().index[0] if not vals.empty else None)
            valid_counts.append(vc_total); valid_neighbors_counts.append(int(neighbor_any.sum()))
        for c, arr in gathered_numeric.items():
            new_name = _feature_name(c); _assign(new_name, arr)
        for c, arr in gathered_cat.items():
            new_name = _feature_name(c); _assign(new_name, arr)
        if add_distance:
            _assign(f'{tag}_knn_mean_dist' if tag else 'knn_mean_dist', mean_dists)
            _assign(f'{tag}_knn_valid' if tag else 'knn_valid', valid_counts)
            _assign(f'{tag}_knn_valid_neighbors' if tag else 'knn_valid_neighbors', valid_neighbors_counts)
            _assign(f'{tag}_knn_valid_ratio' if tag else 'knn_valid_ratio', [vn / float(k) for vn in valid_neighbors_counts])
    elif method == 'raster':
        if raster_path is None:
            raise ValueError("raster_path required for raster method")
        samples = _raster_sample(raster_path, lat, lon)
        for c, vals in samples.items():
            new_name = _feature_name(c)
            _assign(new_name, vals)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return b


# --------------- Layer Coordinate Assignment Helper ---------------

def _assign_layer_coords(df: pd.DataFrame, layer: str, steps_log: Optional[List[Dict]] = None) -> None:
    """Ensure layer-specific coordinate columns are present and filled.
    layer: one of 'clim','elev','soil'.
    Creates columns if absent or fills missing values from canonical latitude/longitude.
    """
    mapping = {
        'clim': ('latitude_clim', 'longitude_clim'),
        'elev': ('latitude_elev', 'longitude_elev'),
        'soil': ('lat_soil', 'lon_soil'),
    }
    if layer not in mapping:
        return
    lat_col, lon_col = mapping[layer]
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return
    # Latitude
    if lat_col not in df.columns:
        df[lat_col] = df['latitude']
    else:
        mask_lat = df[lat_col].isna()
        if mask_lat.any():
            df.loc[mask_lat, lat_col] = df.loc[mask_lat, 'latitude']
    # Longitude
    if lon_col not in df.columns:
        df[lon_col] = df['longitude']
    else:
        mask_lon = df[lon_col].isna()
        if mask_lon.any():
            df.loc[mask_lon, lon_col] = df.loc[mask_lon, 'longitude']
    if steps_log is not None:
        steps_log.append({'stage': 'layer_coords', 'layer': layer, 'lat_col': lat_col, 'lon_col': lon_col})


# --------------- Auto-detect Paths -----------------

def auto_detect(base: Path) -> Dict[str, Optional[Path]]:
    cleaned = base / 'DATA_CLEANED' / 'processed'
    fire_candidates = [cleaned / 'fire_with_negatives.parquet', base / 'FIRE' / 'fire_environment_2024.parquet']
    fire_path = next((p for p in fire_candidates if p.exists()), None)
    climate_path = None
    cp = cleaned / 'climate_features_stats.parquet'
    if cp.exists():
        climate_path = cp
    elev_path = next((p for p in [cleaned / 'elevation_features.parquet'] if p.exists()), None)
    soil_path = next((p for p in [cleaned / 'soil_features.parquet'] if p.exists()), None)
    landcover_shp = next((p for p in [base / 'visualization' / 'data' / 'landcover_clipped.shp',
                                      base / 'LANDCOVER' / 'processed' / 'landcover_clipped.shp'] if p.exists()), None)
    landcover_features = next(
        (p for p in [base / 'DATA_CLEANED' / 'processed' / 'landcover_features.parquet'] if p.exists()), None)
    elevation_raster = next(
        (p for p in [base / 'ELEVATION' / 'be15_grd' / 'processed' / 'elevation_clipped.tif'] if p.exists()), None)
    climate_nc = None  # Add default value

    return {
        'fire': fire_path,
        'climate_stats': climate_path,
        'climate_nc': climate_nc,
        'elevation': elev_path,
        'elevation_raster': elevation_raster,
        'soil': soil_path,
        'landcover_shp': landcover_shp,
        'landcover_features': landcover_features,
    }


# --------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Merge fire dataset with environmental features sequentially")
    ap.add_argument('--fire-path', type=Path, default=None)
    ap.add_argument('--climate-stats', type=Path, default=None)
    ap.add_argument('--climate-nc', type=Path, default=None, help='Optional NetCDF for raster sampling')
    ap.add_argument('--elevation-features', type=Path, default=None)
    ap.add_argument('--elevation-raster', type=Path, default=None, help='Elevation raster for sampling')
    ap.add_argument('--soil-features', type=Path, default=None)
    ap.add_argument('--landcover-shp', type=Path, default=None)
    ap.add_argument('--landcover-features', type=Path, default=None,
                    help='Precomputed landcover features table (centroids + classes)')
    ap.add_argument('--output-dir', type=Path, default=None, help='Output directory for results')

    ap.add_argument('--climate-method', choices=['nearest', 'knn', 'raster'], default='nearest')
    ap.add_argument('--climate-k', type=int, default=3, help='k for knn climate method')
    ap.add_argument('--elev-method', choices=['nearest', 'knn', 'raster'], default='raster')
    ap.add_argument('--elev-k', type=int, default=3)
    ap.add_argument('--soil-method', choices=['nearest', 'knn'], default='nearest')
    ap.add_argument('--soil-k', type=int, default=3)
    ap.add_argument('--landcover-method', choices=['polygon', 'nearest'], default='polygon')
    ap.add_argument('--landcover-k', type=int, default=3)

    ap.add_argument('--max-elev-rows', type=int, default=3_000_000,
                    help='If elevation features exceed this, downsample for KDTree')
    ap.add_argument('--downsample-elev-frac', type=float, default=0.2,
                    help='Fraction for elevation downsample when large')
    ap.add_argument('--landcover-feat-method', choices=['nearest', 'knn'], default='nearest',
                    help='Merge method for landcover features table')
    ap.add_argument('--landcover-feat-k', type=int, default=5, help='k for knn when using landcover features table')
    ap.add_argument('--save-csv', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--global-fill', action='store_true')
    ap.add_argument('--fill-strategy', choices=['mean', 'median'], default='median')
    ap.add_argument('--auto', action='store_true', help='Auto-detect common paths under repository root')

    ap.add_argument('--limit', type=int, default=None, help='Optional limit on number of fire rows (for quick test)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--climate-month-col', type=str, default=None,
                    help='Column in fire dataset containing YYYY-MM month to guide climate sampling (raster method)')
    ap.add_argument('--add-distance', action='store_true',
                    help='Add distance metrics (degrees) for each nearest/knn merge step')
    ap.add_argument('--knn-weighting', choices=['uniform', 'idw'], default='uniform',
                    help='Weighting scheme for knn numeric aggregation (idw=inverse distance)')
    ap.add_argument('--max-knn-source-points', type=int, default=800000,
                    help='Maximum source points retained for nearest/knn merge (random downsample if exceeded)')
    ap.add_argument('--grid-reduction-deg', type=float, default=None,
                    help='Optional grid cell size (degrees) to aggregate and reduce large source datasets before knn/nearest')
    
    # Correlation-based feature selection
    ap.add_argument('--drop-correlated', action='store_true',
                    help='Drop highly correlated features to reduce multicollinearity')
    ap.add_argument('--corr-threshold', type=float, default=0.95,
                    help='Correlation threshold for dropping features (default: 0.95)')
    ap.add_argument('--export-correlations', type=Path, default=None,
                    help='Path to export high correlation pairs CSV')

    args = ap.parse_args()
    base = Path(__file__).resolve().parents[1]
    if args.output_dir is None:
        args.output_dir = base / 'DATA_CLEANED' / 'processed'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detected = auto_detect(base) if args.auto else {}

    fire_path = args.fire_path or detected.get('fire')
    if fire_path is None or not fire_path.exists():
        raise FileNotFoundError('Fire dataset not found. Provide --fire-path or use --auto with existing file.')

    # Load fire dataset using robust loader
    fire_df = robust_read_file(fire_path)
    fire_df = _coerce_latlon(fire_df)

    def _add_key(df: pd.DataFrame) -> pd.DataFrame:
        if 'latitude' in df.columns and 'longitude' in df.columns:
            if 'latitude_r' not in df.columns:
                df['latitude_r'] = df['latitude'].round(5)
            if 'longitude_r' not in df.columns:
                df['longitude_r'] = df['longitude'].round(5)
            if 'latlon_key' not in df.columns:
                df['latlon_key'] = df['latitude_r'].astype(str) + '_' + df['longitude_r'].astype(str)
        return df

    def exact_merge(base: pd.DataFrame, source: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        source = _add_key(source).drop_duplicates(subset=['latlon_key']) if 'latlon_key' in _add_key(
            source).columns else source
        base = _add_key(base)
        if 'latlon_key' not in base.columns or 'latlon_key' not in source.columns:
            return base, source, {'stage': f'{tag}_exact', 'skipped': True}
        # Restrict source columns to avoid duplicate coordinate rounding columns suffix handling
        common = base.merge(source, on='latlon_key', how='left', suffixes=('', f'_{tag}'))
        base_cols = set(base.columns)
        new_cols = [c for c in common.columns if
                    c not in base_cols and not c.endswith('_r') and c not in ['latitude_r', 'longitude_r',
                                                                              'latlon_key']]
        added_mask = pd.Series(False, index=base.index)
        for c in new_cols:
            added_mask |= common[c].notna()
        remain = common.loc[~added_mask, base.columns].copy()
        updated = common.loc[added_mask, :].copy()
        out = pd.concat([updated, remain], ignore_index=True)
        meta = {'stage': f'{tag}_exact', 'new_cols': len(new_cols), 'matched_rows': int(added_mask.sum()),
                'remaining_rows': int((~added_mask).sum())}
        # Return updated base with appended new columns
        return out, source, meta

    fire_df = _add_key(fire_df).drop_duplicates(subset=['latlon_key']).reset_index(drop=True)
    steps_log = []  # initialize before potential confidence derivation
    protected_columns: Set[str] = set()

    # Derive confidence classes if requested (now we only preserve existing)
    if 'confidence' in fire_df.columns:
        unique_conf = set(str(v).strip().lower() for v in fire_df['confidence'].dropna().unique())
        if unique_conf and unique_conf.issubset({'l', 'n', 'h'}):
            fire_df['confidence'] = fire_df['confidence'].astype(str).str.strip().str.lower()
            fire_df['confidence_class'] = fire_df['confidence']
            steps_log.append({'stage': 'confidence_preserved', 'unique': sorted(unique_conf)})
        else:
            # If numeric confidence, keep as is; do not derive class
            if pd.api.types.is_numeric_dtype(fire_df['confidence']) or all(
                    x.replace('.', '', 1).isdigit() for x in fire_df['confidence'].dropna().astype(str)):
                steps_log.append({'stage': 'confidence_numeric_retained'})
            else:
                steps_log.append({'stage': 'confidence_mixed_retained'})

    merged = fire_df.copy()

    # Climate merge
    climate_path = args.climate_stats or detected.get('climate_stats')
    if climate_path and climate_path.exists():
        clim_df = robust_read_file(climate_path)
        clim_df = _coerce_latlon(clim_df)
        climate_feature_cols = [
            c for c in clim_df.columns
            if c.lower() not in {'latitude', 'longitude', 'lat', 'lon'}
        ]
        protected_columns.update(climate_feature_cols)
        merged, clim_df, meta = exact_merge(merged, clim_df, 'clim')
        steps_log.append(meta)
        # For remaining unmatched rows apply spatial method
        if meta['remaining_rows'] > 0:
            method = args.climate_method
            month_val = None
            if method == 'raster' and args.climate_month_col and args.climate_month_col in merged.columns:
                # Use most frequent month among fire points as representative (reduce multi-time sampling cost)
                month_counts = merged[args.climate_month_col].dropna().astype(str).value_counts()
                if not month_counts.empty:
                    month_val = month_counts.index[0]
            if method == 'raster':
                nc_path = args.climate_nc or detected.get('climate_nc')
                if nc_path and nc_path.exists():
                    if xr is None:
                        raise RuntimeError('xarray not installed for raster climate sampling')
                    ds = xr.open_dataset(nc_path)
                    var_names = list(ds.data_vars)
                    samples = _netcdf_sample(nc_path, var_names, merged['latitude'].to_numpy(),
                                             merged['longitude'].to_numpy(), month=month_val)
                    for k, v in samples.items():
                        # fill existing if present
                        if k in merged.columns:
                            mask = merged[k].isna(); merged.loc[mask, k] = [v[i] for i in range(len(v)) if mask.iloc[i]]
                        else:
                            merged[f'{k}'] = v
                        protected_columns.add(k)
                    if args.add_distance:
                        merged['clim_dist'] = np.nan
                    steps_log.append({'stage': 'climate_nc', 'method': 'raster', 'vars': len(samples), 'month_used': month_val})
                else:
                    merged = merge_step(merged, clim_df, 'nearest', k=args.climate_k, tag='clim', add_distance=args.add_distance,
                                        weighting=args.knn_weighting, max_source_points=args.max_knn_source_points,
                                        grid_deg=args.grid_reduction_deg, steps_log=steps_log,
                                        update_existing=True, only_fill_missing=True, force_no_prefix=True)
                    steps_log.append({'stage': 'climate_stats', 'method': 'nearest_fallback_fill'})
            else:
                merged = merge_step(merged, clim_df, method, k=args.climate_k, tag='clim', add_distance=args.add_distance,
                                    weighting=args.knn_weighting, max_source_points=args.max_knn_source_points,
                                    grid_deg=args.grid_reduction_deg, steps_log=steps_log,
                                    update_existing=True, only_fill_missing=True, force_no_prefix=True)
                steps_log.append({'stage': 'climate_stats', 'method': f'{method}_fill'})
    else:
        steps_log.append({'stage': 'climate_stats', 'skipped': True})
    # Assign climate layer coordinates regardless of merge success (still useful for consistency)
    _assign_layer_coords(merged, 'clim', steps_log)

    # Elevation merge
    elev_features = args.elevation_features or detected.get('elevation')
    elev_raster = args.elevation_raster or detected.get('elevation_raster')
    if args.elev_method == 'raster' and elev_raster and elev_raster.exists():
        try:
            samples = _raster_sample(elev_raster, merged['latitude'].to_numpy(), merged['longitude'].to_numpy())
            key = next(iter(samples))
            # Assign or fill existing elevation
            if 'elevation' in merged.columns:
                mask = merged['elevation'].isna(); merged.loc[mask, 'elevation'] = [samples[key][i] for i in range(len(samples[key])) if mask.iloc[i]]
            else:
                merged['elevation'] = samples[key]
            steps_log.append({'stage': 'elevation_raster', 'method': 'raster'})
        except Exception as e:
            steps_log.append({'stage': 'elevation_raster', 'error': str(e)})
    elif elev_features and elev_features.exists():
        elev_df = robust_read_file(elev_features)
        elev_df = _coerce_latlon(elev_df)
        merged, elev_df, meta = exact_merge(merged, elev_df, 'elev')
        steps_log.append(meta)
        if meta['remaining_rows'] > 0:
            if len(elev_df) > args.max_elev_rows and args.elev_method in ['nearest', 'knn', 'raster']:
                frac = args.downsample_elev_frac
                elev_df = elev_df.sample(int(len(elev_df) * frac), random_state=args.seed).reset_index(drop=True)
                steps_log.append({'stage': 'elevation_features', 'downsampled': True, 'frac': frac})
            effective_method = args.elev_method
            if effective_method == 'raster':
                effective_method = 'nearest'
                steps_log.append({'stage': 'elevation_features', 'fallback': 'raster→nearest'})
            merged = merge_step(merged, elev_df, effective_method, k=args.elev_k, tag='elev', add_distance=args.add_distance,
                                weighting=args.knn_weighting, max_source_points=args.max_knn_source_points,
                                grid_deg=args.grid_reduction_deg, steps_log=steps_log,
                                update_existing=True, only_fill_missing=True, force_no_prefix=True)
            steps_log.append({'stage': 'elevation_features', 'method': f'{effective_method}_fill'})
    else:
        steps_log.append({'stage': 'elevation', 'skipped': True})
    _assign_layer_coords(merged, 'elev', steps_log)

    # Soil merge
    soil_path = args.soil_features or detected.get('soil')
    if soil_path and soil_path.exists():
        soil_df = robust_read_file(soil_path)
        soil_df = filter_soil_columns(soil_df)  # Keep only specified soil features
        soil_df = _coerce_latlon(soil_df)
        merged, soil_df, meta = exact_merge(merged, soil_df, 'soil')
        steps_log.append(meta)
        if meta['remaining_rows'] > 0:
            merged = merge_step(merged, soil_df, args.soil_method, k=args.soil_k, tag='soil', add_distance=args.add_distance,
                                weighting=args.knn_weighting, max_source_points=args.max_knn_source_points,
                                grid_deg=args.grid_reduction_deg, steps_log=steps_log,
                                update_existing=True, only_fill_missing=True, force_no_prefix=True)
            steps_log.append({'stage': 'soil_features', 'method': f'{args.soil_method}_fill'})
    else:
        steps_log.append({'stage': 'soil', 'skipped': True})
    _assign_layer_coords(merged, 'soil', steps_log)

    # Landcover merge
    lc_path = args.landcover_shp or detected.get('landcover_shp')
    lc_feat_path = args.landcover_features or detected.get('landcover_features')

    if lc_path and lc_path.exists() and gpd is not None:
        try:
            lc_gdf = gpd.read_file(lc_path)
            lc_gdf = lc_gdf.to_crs('EPSG:4326')
            lc_centroids = lc_gdf.geometry.centroid
            cent_df = pd.DataFrame({'latitude': lc_centroids.y, 'longitude': lc_centroids.x})
            for c in lc_gdf.columns:
                if c != 'geometry':
                    cent_df[c] = lc_gdf[c].values
            merged, cent_df, meta = exact_merge(merged, cent_df, 'lc')
            steps_log.append(meta)
            if meta['remaining_rows'] > 0 and args.landcover_method == 'polygon':
                attr_cols = [c for c in lc_gdf.columns if c != 'geometry']
                lc_vals = _point_in_polygon(merged, lc_gdf, attr_cols, 'polygon')
                for c, v in lc_vals.items():
                    if c in merged.columns:
                        mask = merged[c].isna(); merged.loc[mask, c] = [v[i] for i in range(len(v)) if mask.iloc[i]]
                    else:
                        merged[f'{c}'] = v
                steps_log.append({'stage': 'landcover', 'method': 'polygon_fill', 'cols': len(attr_cols)})
            elif meta['remaining_rows'] > 0:
                merged = merge_step(merged, cent_df, 'nearest', k=args.landcover_k, tag='lc', add_distance=args.add_distance,
                                    weighting=args.knn_weighting, max_source_points=args.max_knn_source_points,
                                    grid_deg=args.grid_reduction_deg, steps_log=steps_log,
                                    update_existing=True, only_fill_missing=True, force_no_prefix=True)
                steps_log.append({'stage': 'landcover', 'method': 'nearest_fallback_fill'})
        except Exception as e:
            steps_log.append({'stage': 'landcover', 'error': str(e)})
    else:
        steps_log.append({'stage': 'landcover', 'skipped': True})

    # Landcover features table merge (runs after shapefile attempt to allow both)
    if lc_feat_path and lc_feat_path.exists():
        try:
            lc_feat_df = robust_read_file(lc_feat_path)
            if ('latitude' not in lc_feat_df.columns or lc_feat_df['latitude'].isna().all()) and lc_path and lc_path.exists() and gpd is not None:
                try:
                    _g = gpd.read_file(lc_path).to_crs('EPSG:4326')
                    cents = _g.geometry.centroid
                    lc_feat_df['latitude'] = cents.y; lc_feat_df['longitude'] = cents.x
                    steps_log.append({'stage': 'landcover_features', 'centroid_recomputed': True, 'rows': len(lc_feat_df)})
                except Exception as e:
                    steps_log.append({'stage': 'landcover_features', 'centroid_recompute_error': str(e)})
            lc_feat_df = _coerce_latlon(lc_feat_df)
            merged, lc_feat_df, meta = exact_merge(merged, lc_feat_df, 'lcfeat')
            steps_log.append(meta)
            if meta.get('remaining_rows', 0) > 0:
                method = args.landcover_feat_method; k_val = args.landcover_feat_k
                merged = merge_step(merged, lc_feat_df, method, k=k_val, tag='lcfeat', add_distance=args.add_distance,
                                    weighting=args.knn_weighting, max_source_points=args.max_knn_source_points,
                                    grid_deg=args.grid_reduction_deg, steps_log=steps_log,
                                    update_existing=True, only_fill_missing=True, force_no_prefix=True)
                steps_log.append({'stage': 'landcover_features', 'method': f'{method}_fill', 'k': k_val})
        except Exception as e:
            steps_log.append({'stage': 'landcover_features', 'error': str(e)})
    else:
        steps_log.append({'stage': 'landcover_features', 'skipped': True})

    # Global fill
    if args.global_fill:
        numeric_cols = [c for c in merged.columns if pd.api.types.is_numeric_dtype(merged[c])]
        for c in numeric_cols:
            if merged[c].isna().any():
                if args.fill_strategy == 'median':
                    fill_val = merged[c].median()
                else:
                    fill_val = merged[c].mean()
                merged[c] = merged[c].fillna(fill_val)
        steps_log.append({'stage': 'global_fill', 'strategy': args.fill_strategy, 'cols_filled': len(numeric_cols)})

    # Prune statistical / auxiliary columns (user request)
    essential = {'latitude','longitude','fire','bright_ti4','bright_ti5','frp','confidence','daynight','type','country','acq_datetime'}
    prune_tokens = ['max','min','std','p10','p90','mean','range','total','index','dist','knn','_r','key','class']
    drop_cols = []
    for col in merged.columns:
        if col in essential:
            continue
        if col in protected_columns:
            continue
        token_hit = any(tok in col.lower() for tok in prune_tokens)
        if token_hit:
            drop_cols.append(col)
    if drop_cols:
        merged.drop(columns=drop_cols, inplace=True, errors='ignore')
        steps_log.append({
            'stage': 'prune_columns',
            'dropped': len(drop_cols),
            'tokens': prune_tokens,
            'protected': sorted(protected_columns)
        })

    # Apply limit if specified
    if args.limit and len(merged) > args.limit:
        merged = merged.sample(args.limit, random_state=args.seed).reset_index(drop=True)
        steps_log.append({'stage': 'limit_applied', 'n_rows': args.limit})

    # Drop highly correlated features
    if args.drop_correlated:
        protected = {'fire', 'latitude', 'longitude', 'lat', 'lon', 'bright_ti4', 'bright_ti5', 
                     'frp', 'confidence', 'country', 'source', 'acq_datetime', 'acq_date', 'acq_time'}
        export_path = args.export_correlations
        if export_path is None:
            # Auto-generate export path with timestamp
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%dT%H-%M')
            export_path = args.output_dir / f'{ts}_export.csv'
        
        merged, dropped_info = drop_highly_correlated(
            merged, 
            threshold=args.corr_threshold,
            protected_cols=protected,
            export_path=export_path
        )
        steps_log.append({
            'stage': 'drop_correlated',
            'threshold': args.corr_threshold,
            'features_dropped': len(dropped_info),
            'dropped_details': dropped_info[:20]  # Store first 20 for summary
        })

    # Save results
    output_path = args.output_dir / 'merged_dataset.parquet'
    merged.to_parquet(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")

    if args.save_csv:
        csv_path = args.output_dir / 'merged_dataset.csv'
        merged.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

    if args.summary:
        summary = {
            'total_rows': len(merged),
            'columns': list(merged.columns),
            'steps': steps_log,
            'missing_counts': {col: int(merged[col].isna().sum()) for col in merged.columns}
        }
        summary_path = args.output_dir / 'merged_dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")

    print("Merge completed successfully!")


if __name__ == '__main__':
    main()
