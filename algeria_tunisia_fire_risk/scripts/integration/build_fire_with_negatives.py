#!/usr/bin/env python3
"""Build a fire / no-fire classification dataset from VIIRS fire CSVs.

Steps:
1. Load all 2024 VIIRS fire CSVs under FIRE/.
2. Standardize columns and parse acquisition datetime.
3. Filter by confidence and optional exclusion of type values.
4. Drop low-information / engineering-only columns (scan, track, version, instrument, satellite if constant).
5. Create positive class label fire=1.
6. Generate negative samples (fire=0) as random points across the AOI polygon (algeria_tunisia.shp),
   excluding points within a minimum distance of any fire point.
7. Adjust negative count so positive ratio falls within a target (e.g., 0.8 => 80% fire).
8. Produce combined Parquet + CSV outputs and an optional EDA summary JSON.

CLI:
  python -m scripts.build_fire_with_negatives --aoi shapefiles/algeria_tunisia.shp --confidence-min 50 --positive-ratio 0.8 --min-distance-km 1 --eda-summary

Dependencies: pandas, numpy, shapely, geopandas. (Optional: scipy for fast nearest-neighbor filtering).
If geopandas is not installed the script will abort gracefully.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random
import math

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon
except Exception:
    gpd = None
    Point = Polygon = MultiPolygon = None

try:
    from scipy.spatial import cKDTree  # optional acceleration
except Exception:
    cKDTree = None

try:
    from scipy.io import netcdf_file as NetCDFFile  # optional grid coordinate reader
except Exception:
    NetCDFFile = None

try:
    import pyogrio
except ImportError:
    pyogrio = None
try:
    from shapely import wkb as _shp_wkb
    from shapely.geometry import Polygon as ShpPolygon, MultiPolygon as ShpMultiPolygon
except Exception:
    _shp_wkb = None
    ShpPolygon = None
    ShpMultiPolygon = None

BASE = Path(__file__).resolve().parents[1]
FIRE_DIR = BASE / "FIRE"
DEFAULT_OUT = BASE / "DATA_CLEANED" / "processed"

VIIRS_FEATURE_MEANINGS: Dict[str, str] = {
    "latitude": "Center latitude of fire pixel (degrees, WGS84)",
    "longitude": "Center longitude of fire pixel (degrees, WGS84)",
    "bright_ti4": "Brightness temperature channel I4 (Kelvin)",
    "bright_ti5": "Brightness temperature channel I5 (Kelvin)",
    "scan": "Scan pixel width (degrees) — acquisition geometry, low predictive info",
    "track": "Track pixel height (degrees) — acquisition geometry",
    "acq_date": "Acquisition date (UTC)",
    "acq_time": "Acquisition time (UTC HHMM)",
    "satellite": "Satellite identifier (e.g., N, J)",
    "instrument": "Sensor instrument (VIIRS)",
    "confidence": "Fire confidence score (0–100 or categorical depending on provider)",
    "version": "Data/processing version string",
    "frp": "Fire Radiative Power (MW)",
    "daynight": "D or N indicating day/night detection context",
    "type": "Classification of source (0=unknown,1=presumed vegetation,2=other static source)",
}

DROP_COLUMNS_DEFAULT = ["scan", "track", "version", "instrument"]  # optionally drop satellite if constant
POTENTIAL_ID_COLUMNS = ["acq_date", "acq_time", "latitude", "longitude"]

NEG_NUMERIC_DEFAULTS = {
    "bright_ti4": 0.0,
    "bright_ti5": 0.0,
    "frp": 0.0,
    "confidence": 0.0,
    "filter_relaxed": 0.0,
    "type": 0.0,
}

NEG_CATEGORICAL_DEFAULTS = {
    "daynight": "U",
    "country": "AOI",
    "confidence_normalized": "0",
}

NEG_DATETIME_DEFAULT = pd.Timestamp("2024-01-01T00:00:00Z")

def parse_fire_files(fire_dir: Path, pattern: str = "*.csv") -> List[Path]:
    """Return sorted list of source CSV files under FIRE/ matching a glob pattern."""
    return sorted([p for p in fire_dir.glob(pattern) if p.is_file()])

def load_and_standardize(paths: List[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        name = p.name.lower()
        if "alger" in name:
            df["country"] = "Algeria"
        elif "tunis" in name:
            df["country"] = "Tunisia"
        else:
            df["country"] = None
        # Infer type from filename if not present
        if "type2" in name and "type" not in df.columns:
            df["type"] = 2
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    fire = pd.concat(dfs, ignore_index=True)
    # parse datetime
    if {"acq_date", "acq_time"}.issubset(fire.columns):
        def _dt(row):
            try:
                t = str(row["acq_time"]).zfill(4)
                hh, mm = int(t[:2]), int(t[2:])
                return pd.to_datetime(row["acq_date"], errors="coerce") + pd.Timedelta(hours=hh, minutes=mm)
            except Exception:
                return pd.to_datetime(row["acq_date"], errors="coerce")
        fire["acq_datetime"] = fire.apply(_dt, axis=1)
    elif "acq_date" in fire.columns:
        fire["acq_datetime"] = pd.to_datetime(fire["acq_date"], errors="coerce")
    # coerce numerics except confidence categorical codes
    for col in ["latitude","longitude","bright_ti4","bright_ti5","frp"]:
        if col in fire.columns:
            fire[col] = pd.to_numeric(fire[col], errors="coerce")
    if "confidence" in fire.columns:
        # Detect if confidence already categorical (n/l/h or N/L/H)
        unique_conf = set(str(v).strip().lower() for v in fire["confidence"].dropna().unique())
        if unique_conf and unique_conf.issubset({"n","l","h"}):
            fire["confidence"] = fire["confidence"].str.strip().str.lower()
            fire["confidence_normalized"] = fire["confidence"]  # duplicate explicit field
        else:
            # Attempt numeric conversion; keep original text in confidence_raw if conversion loses data
            raw_vals = fire["confidence"].astype(str)
            num_vals = pd.to_numeric(fire["confidence"], errors="coerce")
            fire["confidence_raw"] = raw_vals
            fire["confidence_numeric"] = num_vals
            # Do NOT derive categories automatically; leave for downstream if desired
    # drop rows with invalid lat/lon
    if {"latitude","longitude"}.issubset(fire.columns):
        fire = fire.dropna(subset=["latitude","longitude"]).copy()
    # remove duplicates by lat/lon/datetime if available
    subset_cols = [c for c in ["latitude","longitude","acq_datetime"] if c in fire.columns]
    if subset_cols:
        fire = fire.drop_duplicates(subset=subset_cols)
    return fire

def filter_fire(df: pd.DataFrame, conf_min: int, exclude_types: List[int]) -> pd.DataFrame:
    out = df.copy()
    if "confidence" in out.columns:
        # Only apply numeric filter if confidence_numeric exists (original categorical should be preserved)
        if "confidence_numeric" in out.columns:
            out = out[pd.to_numeric(out["confidence_numeric"], errors="coerce") >= conf_min]
        else:
            # If categorical confidence, we keep all unless user sets conf_min > 0, which we ignore here
            pass
    if "type" in out.columns and exclude_types:
        out = out[~out["type"].isin(exclude_types)]
    return out

def drop_useless_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    to_drop = [c for c in DROP_COLUMNS_DEFAULT if c in df.columns]
    # drop satellite if single unique value
    if "satellite" in df.columns and df["satellite"].nunique() <= 1:
        to_drop.append("satellite")
    # keep acq_date/time only if needed; we already have acq_datetime
    if "acq_datetime" in df.columns:
        for c in ["acq_date","acq_time"]:
            if c in df.columns:
                to_drop.append(c)
    cleaned = df.drop(columns=[c for c in to_drop if c in df.columns]).copy()
    return cleaned, to_drop

def eda_summary(df: pd.DataFrame, out_path: Path) -> None:
    """Write a lightweight EDA summary with JSON-safe keys/values.
    - Numeric columns: describe() stats (keys cast to str, NaNs dropped).
    - Datetime columns: min/max/count/nunique as strings.
    - Other columns: top 50 value counts with keys cast to str.
    """
    numeric: Dict[str, Dict[str, float | int | str]] = {}
    categorical: Dict[str, Dict[str, int | str]] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            stats = s.describe().to_dict()
            # Filter out non-serializable keys and cast
            clean_stats = {}
            for k, v in stats.items():
                k_str = str(k)
                if isinstance(v, (np.floating, float)):
                    v_out = float(v)
                elif isinstance(v, (np.integer, int)):
                    v_out = int(v)
                else:
                    v_out = v if v is not None else None
                clean_stats[k_str] = v_out
            numeric[c] = clean_stats
        elif pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                categorical[c] = {
                    "count": int(s.count()),
                    "min": str(s.min()),
                    "max": str(s.max()),
                    "nunique": int(s.nunique()),
                }
            else:
                categorical[c] = {"count": 0, "min": None, "max": None, "nunique": 0}
        else:
            vc = s.value_counts(dropna=True).head(50)
            cat_map = {str(k): int(v) for k, v in vc.items()}
            categorical[c] = cat_map
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric": numeric,
        "categorical": categorical,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def _nearest_neighbor_stats(latitudes: np.ndarray, longitudes: np.ndarray) -> dict:
    if len(latitudes) < 2:
        return {"count": int(len(latitudes)), "min_km": None, "median_km": None, "mean_km": None}
    coords = np.vstack([latitudes, longitudes]).T
    # brute force distance matrix in manageable chunks
    n = len(coords)
    min_dists = np.empty(n, dtype=float)
    for i in range(n):
        la, lo = coords[i]
        dlat = np.radians(coords[:,0]-la)
        dlon = np.radians(coords[:,1]-lo)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(la))*np.cos(np.radians(coords[:,0]))*np.sin(dlon/2)**2
        d = 2*6371.0*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d[i] = np.inf
        min_dists[i] = d.min()
    finite = min_dists[np.isfinite(min_dists)]
    return {
        "count": int(n),
        "min_km": float(finite.min()) if finite.size else None,
        "median_km": float(np.median(finite)) if finite.size else None,
        "mean_km": float(finite.mean()) if finite.size else None
    }

def sample_negative_points(aoi_path: Path, pos_df: pd.DataFrame, target_positive_ratio: float,
                           min_distance_km: float, max_negatives: int | None = None, seed: int = 42,
                           override_neg_count: int | None = None, attempt_mult: int = 200) -> pd.DataFrame:
    # Determine desired negative count
    P = len(pos_df)
    if P == 0:
        raise RuntimeError("No positive fire samples to base ratio on.")
    if override_neg_count is not None and override_neg_count > 0:
        N = int(override_neg_count)
    else:
        r = target_positive_ratio
        N = int(round(P * (1 / r - 1)))
    if max_negatives is not None:
        N = min(N, max_negatives)
    N = max(N, 1)
    # If geopandas available use AOI polygon rejection sampling, else fallback bounding-box of positives
    if gpd is None:
        if {'latitude','longitude'}.issubset(pos_df.columns):
            miny, maxy = pos_df['latitude'].min(), pos_df['latitude'].max()
            minx, maxx = pos_df['longitude'].min(), pos_df['longitude'].max()
        else:
            raise RuntimeError('Missing latitude/longitude for fallback negative sampling.')
        rng = random.Random(seed)
        pos_points = pos_df[['latitude','longitude']].to_numpy()
        neg_rows = []
        attempts = 0
        max_attempts = N * attempt_mult
        while len(neg_rows) < N and attempts < max_attempts:
            attempts += 1
            lon = rng.uniform(minx, maxx)
            lat = rng.uniform(miny, maxy)
            # compute nearest haversine distance to any fire point (approx: use vectorized subset)
            dlat = np.radians(pos_points[:,0]-lat)
            dlon = np.radians(pos_points[:,1]-lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(pos_points[:,0]))*np.sin(dlon/2)**2
            d = 2*6371.0*np.arctan2(np.sqrt(a), np.sqrt(1-a))
            if d.min() < min_distance_km:
                continue
            neg_rows.append((lat, lon))
        if len(neg_rows) < N:
            print(f"[WARN] Fallback negative sampling incomplete: requested {N}, got {len(neg_rows)} attempts={attempts}")
        neg_df = pd.DataFrame(neg_rows, columns=['latitude','longitude'])
        neg_df['fire'] = 0
        return neg_df
    # Geopandas path (unchanged logic except added override_neg_count)
    aoi = gpd.read_file(aoi_path)
    geom = getattr(aoi, 'union_all', aoi.unary_union)() if hasattr(aoi, 'union_all') else aoi.unary_union
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise RuntimeError("AOI geometry must be polygonal.")
    minx, miny, maxx, maxy = geom.bounds
    rng = random.Random(seed)
    pos_points = pos_df[["latitude","longitude"]].to_numpy() if {"latitude","longitude"}.issubset(pos_df.columns) else np.empty((0,2))
    tree = None
    if cKDTree is not None and pos_points.size:
        tree = cKDTree(pos_points[:, ::-1])
    neg_rows: List[Tuple[float,float]] = []
    attempts = 0
    max_attempts = N * attempt_mult
    while len(neg_rows) < N and attempts < max_attempts:
        attempts += 1
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        pt = Point(lon, lat)
        if not geom.contains(pt):
            continue
        if tree is not None:
            dist, _ = tree.query([[lon, lat]], k=1)
            # convert degree distance to km (rough)
            if dist[0] * 111.0 < min_distance_km:
                continue
        elif pos_points.size:
            dlat = np.radians(pos_points[:,0]-lat)
            dlon = np.radians(pos_points[:,1]-lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(pos_points[:,0]))*np.sin(dlon/2)**2
            d = 2*6371.0*np.arctan2(np.sqrt(a), np.sqrt(1-a))
            if d.min() < min_distance_km:
                continue
        neg_rows.append((lat, lon))
    if len(neg_rows) < N:
        print(f"[WARN] Negative sampling incomplete: requested {N}, got {len(neg_rows)} attempts={attempts}")
    neg_df = pd.DataFrame(neg_rows, columns=["latitude","longitude"])
    neg_df["fire"] = 0
    return neg_df

def sample_negative_points_polygons(aoi_path: Path, landcover_path: Path | None, pos_df: pd.DataFrame, target_positive_ratio: float,
                                    max_negatives: int | None = None, seed: int = 42, min_distance_km: float = 0.5,
                                    max_distance_km: float | None = None) -> pd.DataFrame:
    """Generate non-fire negatives from polygons that contain no fire points.
    Improvements:
      - Correct fire polygon detection using spatial join and polygon indices hit by fire points.
      - Remove invalid/empty geometries.
      - Compute centroids in projected CRS (EPSG:3857) then reproject to WGS84 for accuracy.
      - Exclude centroids within min_distance_km of any fire point.
      - Supplement with interior random points if centroids insufficient.
    """
    if gpd is None:
        raise RuntimeError("geopandas/shapely not installed; cannot polygon-sample negatives.")
    rng = random.Random(seed)

    # Load polygons (landcover preferred; fallback AOI)
    if landcover_path and landcover_path.exists():
        polys_gdf = gpd.read_file(landcover_path)
    else:
        polys_gdf = gpd.read_file(aoi_path)

    # Clean geometry
    polys_gdf = polys_gdf[polys_gdf.geometry.notna()].copy()
    try:
        polys_gdf["geometry"] = polys_gdf.buffer(0)
    except Exception:
        pass
    polys_gdf = polys_gdf[~polys_gdf.geometry.is_empty]
    if polys_gdf.empty:
        raise RuntimeError("No valid polygons available for negative sampling.")

    # Fire points GeoDataFrame
    if not {"latitude","longitude"}.issubset(pos_df.columns):
        raise RuntimeError("Positive fire DataFrame missing latitude/longitude columns.")
    g_fire = gpd.GeoDataFrame(pos_df, geometry=gpd.points_from_xy(pos_df["longitude"], pos_df["latitude"]), crs="EPSG:4326")

    # Ensure polygon CRS is WGS84 then project for centroid accuracy
    try:
        if polys_gdf.crs is None or polys_gdf.crs.to_epsg() != 4326:
            polys_gdf = polys_gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    # Spatial join to find polygons containing fire points
    try:
        hits = gpd.sjoin(polys_gdf, g_fire, how="inner", predicate="contains")
        fire_poly_ids = set(hits.index)
        candidate = polys_gdf.loc[~polys_gdf.index.isin(fire_poly_ids)].copy()
    except Exception:
        candidate = polys_gdf.copy()

    if candidate.empty:
        raise RuntimeError("No candidate polygons without fire detections after spatial join.")

    # Project to Web Mercator for centroid computation
    try:
        candidate_proj = candidate.to_crs("EPSG:3857")
    except Exception:
        candidate_proj = candidate  # fallback

    centroids_proj = candidate_proj.geometry.centroid
    # Reproject centroids back to WGS84
    try:
        centroids_wgs = gpd.GeoSeries(centroids_proj, crs=candidate_proj.crs).to_crs("EPSG:4326")
    except Exception:
        centroids_wgs = centroids_proj  # fallback

    # Positive points array for distance filtering
    pos_points = pos_df[["latitude","longitude"]].to_numpy()
    def haversine(lat1, lon1, lat2, lon2):
        R=6371.0
        dlat=np.radians(lat2-lat1); dlon=np.radians(lon2-lon1)
        a=np.sin(dlat/2)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
        return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def too_close_or_outside_band(lat, lon):
        """Return True if a candidate point is too close (< min_distance_km) to any fire
        or outside the optional upper bound (> max_distance_km). Handles empty pos_points."""
        if pos_points.size == 0:
            return False
        # quick coarse prefilter in degrees (approx)
        deg_thresh = max(min_distance_km / 111.0 / 4.0, 0.0001)
        ddeg = np.sqrt(((pos_points[:,0]-lat)**2 + (pos_points[:,1]-lon)**2))
        if ddeg.min() > deg_thresh and (max_distance_km is None or ddeg.min() < (max_distance_km/111.0)*4.0):
            # likely safe (not too close); still do full haversine check on small subset below
            pass
        # precise haversine for nearest few points to reduce cost
        # find N closest candidates by simple squared-degree metric
        idx_sorted = np.argsort(ddeg)[: min(500, len(ddeg))]
        for idx in idx_sorted:
            plat, plon = pos_points[idx]
            dkm = haversine(lat, lon, plat, plon)
            if dkm < min_distance_km:
                return True
            if max_distance_km is not None and dkm > max_distance_km:
                # if the closest fire is already beyond max_distance_km, then point is outside band
                return True
        return False

    # Compute required negatives
    P = len(pos_df)
    r = target_positive_ratio
    N = int(round(P * (1 / r - 1)))
    if max_negatives is not None:
        N = min(N, max_negatives)
    N = max(N, 1)

    neg_rows: List[Tuple[float,float]] = []
    # Add centroid negatives first
    for geom in centroids_wgs:
        if geom is None or geom.is_empty:
            continue
        lat = geom.y if hasattr(geom, 'y') else None
        lon = geom.x if hasattr(geom, 'x') else None
        if lat is None or lon is None:
            continue
        if too_close_or_outside_band(lat, lon):
            continue
        neg_rows.append((lat, lon))
        if len(neg_rows) >= N:
            break

    # If more needed, rejection sample inside candidate polygons (use projected for uniformity, then transform)
    if len(neg_rows) < N:
        deficit = N - len(neg_rows)
        attempts = 0
        max_attempts = deficit * 300
        # Prepare list of polygons (projected) and their inverse transform if needed
        polys_for_sampling = candidate_proj.geometry
        while deficit > 0 and attempts < max_attempts:
            attempts += 1
            poly = polys_for_sampling.iloc[rng.randrange(len(polys_for_sampling))]
            if poly is None or poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            # Uniform sample bounding box until inside polygon
            for _ in range(10):  # small inner loop
                x = rng.uniform(minx, maxx)
                y = rng.uniform(miny, maxy)
                pt = Point(x, y)
                if poly.contains(pt):
                    # Reproject point to WGS84 if necessary
                    try:
                        pt_wgs = gpd.GeoSeries([pt], crs=candidate_proj.crs).to_crs("EPSG:4326").iloc[0]
                    except Exception:
                        pt_wgs = pt
                    lat, lon = pt_wgs.y, pt_wgs.x
                    if too_close_or_outside_band(lat, lon):
                        continue
                    neg_rows.append((lat, lon))
                    deficit -= 1
                    break
        if deficit > 0:
            print(f"[WARN] Polygon sampling incomplete: requested {N}, got {N - deficit}; attempts={attempts}")

    neg_df = pd.DataFrame(neg_rows, columns=["latitude","longitude"])
    neg_df["fire"] = 0
    return neg_df

def sample_negative_points_polygons_grid(shp_path: Path, pos_df: pd.DataFrame, target_positive_ratio: float,
                                         min_distance_km: float, override_neg_count: int | None, grid_res_deg: float,
                                         max_negatives: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Generate negative points from a regular grid clipped to polygons that contain NO fire points.
    Requires pyogrio + shapely. Falls back to random if missing.
    Steps:
      1. Read polygons (no attributes needed).
      2. Identify polygons with fire points (point-in-polygon test) and exclude them.
      3. For remaining polygons, create a grid at resolution grid_res_deg inside their bounds.
      4. Keep points inside polygon and farther than min_distance_km from any fire point.
      5. Stop when reaching required negative count.
    """
    if pyogrio is None or _shp_wkb is None:
        raise RuntimeError("pyogrio + shapely required for polygons-grid method; install via pip or conda.")
    if grid_res_deg <= 0:
        raise ValueError("grid_res_deg must be > 0")
    fire_points = pos_df[['latitude','longitude']].to_numpy()
    P = len(pos_df)
    if P == 0:
        raise RuntimeError("No positive fire samples to base ratio on.")
    if override_neg_count is not None and override_neg_count > 0:
        N_target = int(override_neg_count)
    else:
        r = target_positive_ratio
        N_target = int(round(P * (1/r - 1)))
    if max_negatives is not None:
        N_target = min(N_target, max_negatives)
    N_target = max(N_target, 1)
    # Read geometries
    gdf = pyogrio.read_dataframe(shp_path, columns=[])
    if gdf.empty:
        raise RuntimeError("No polygons read from shapefile")
    polys = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        try:
            s = _shp_wkb.loads(bytes(geom)) if hasattr(geom, 'data') else _shp_wkb.loads(geom.wkb)
        except Exception:
            continue
        if isinstance(s, (ShpPolygon, ShpMultiPolygon)) and not s.is_empty:
            polys.append(s)
    if not polys:
        raise RuntimeError("No valid polygon geometries parsed")
    # Helper for point inside any polygon (return index)
    def _poly_index(lat, lon):
        pt = (lon, lat)
        for i, poly in enumerate(polys):
            try:
                if poly.contains(ShpPolygon([(lon,lat),(lon,lat),(lon,lat)])):
                    return i
            except Exception:
                # Fallback precise point test using bounds
                if poly.bounds[0] <= lon <= poly.bounds[2] and poly.bounds[1] <= lat <= poly.bounds[3] and poly.contains(ShpPolygon([(lon,lat),(lon,lat),(lon,lat)])):
                    return i
        return -1
    # Build simple bounding box index for faster tests
    poly_bounds = [p.bounds for p in polys]
    # Tag polygons that contain any fire point by bounding box prefilter + contains check
    fire_poly_ids = set()
    for lat, lon in fire_points:
        for idx, (minx,miny,maxx,maxy) in enumerate(poly_bounds):
            if not (miny <= lat <= maxy and minx <= lon <= maxx):
                continue
            # Accurate check using point-in-polygon
            try:
                if polys[idx].contains(ShpPolygon([(lon,lat),(lon,lat),(lon,lat)])):
                    fire_poly_ids.add(idx)
            except Exception:
                pass
    candidate_polys = [p for i,p in enumerate(polys) if i not in fire_poly_ids]
    if not candidate_polys:
        raise RuntimeError("No candidate polygons without fire detections for grid sampling.")
    rng = random.Random(seed)
    # Distance filter helpers
    def haversine(lat1, lon1, lat2, lon2):
        R=6371.0
        dlat=np.radians(lat2-lat1); dlon=np.radians(lon2-lon1)
        a=np.sin(dlat/2)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
        return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    def too_close(lat, lon):
        # Quick coarse filter using bounding boxes: skip expensive distance if far
        # Here we still compute distances for accuracy; can optimize later
        dlat = np.radians(fire_points[:,0]-lat)
        dlon = np.radians(fire_points[:,1]-lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(fire_points[:,0]))*np.sin(dlon/2)**2
        d = 2*6371.0*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return d.min() < min_distance_km
    neg_rows = []
    for poly in candidate_polys:
        if len(neg_rows) >= N_target:
            break
        minx, miny, maxx, maxy = poly.bounds
        xs = np.arange(minx, maxx + 1e-9, grid_res_deg)
        ys = np.arange(miny, maxy + 1e-9, grid_res_deg)
        # Randomize order to spatially diversify early termination
        xs_list = list(xs); ys_list = list(ys)
        rng.shuffle(xs_list); rng.shuffle(ys_list)
        for y in ys_list:
            for x in xs_list:
                if len(neg_rows) >= N_target:
                    break
                # inside test
                try:
                    # create a tiny triangle polygon as point proxy for shapely contains (point geometry alternative)
                    ptri = ShpPolygon([(x,y),(x,y),(x,y)])
                    inside = poly.contains(ptri)
                except Exception:
                    inside = False
                if not inside:
                    continue
                if too_close(y, x):
                    continue
                neg_rows.append((y, x))
            if len(neg_rows) >= N_target:
                break
    if len(neg_rows) < N_target:
        print(f"[WARN] polygons-grid sampling incomplete: requested {N_target}, got {len(neg_rows)}")
    neg_df = pd.DataFrame(neg_rows, columns=['latitude','longitude'])
    neg_df['fire'] = 0
    neg_df['source'] = 'polygons_grid'
    return neg_df

def sample_negative_points_even_grid(pos_df: pd.DataFrame, fire_ratio: float, min_distance_km: float, override_neg_count: int | None = None) -> pd.DataFrame:
    if {'latitude','longitude'}.issubset(pos_df.columns):
        miny, maxy = pos_df['latitude'].min(), pos_df['latitude'].max()
        minx, maxx = pos_df['longitude'].min(), pos_df['longitude'].max()
    else:
        raise RuntimeError('Missing latitude/longitude for even grid negative sampling.')
    P = len(pos_df)
    if override_neg_count and override_neg_count > 0:
        N = override_neg_count
    else:
        N = int(round(P * (1/fire_ratio - 1)))
    N = max(N, 1)
    area_deg2 = (maxx-minx)*(maxy-miny)
    side = math.sqrt(area_deg2 / N)
    side = max(side, 0.02)  # clamp to avoid extremely fine grid (≈2km)
    xs = np.arange(minx, maxx+1e-9, side)
    ys = np.arange(miny, maxy+1e-9, side)
    fire_points = pos_df[['latitude','longitude']].to_numpy()
    # Optional KDTree for speed
    kdt = None
    try:
        if cKDTree is not None and fire_points.shape[0] > 0:
            kdt = cKDTree(fire_points[:, ::-1])  # lon, lat
    except Exception:
        kdt = None
    def too_close(lat, lon):
        if kdt is not None:
            dist_deg, _ = kdt.query([[lon, lat]], k=1)
            return dist_deg[0] * 111.0 < min_distance_km
        dlat = np.radians(fire_points[:,0]-lat)
        dlon = np.radians(fire_points[:,1]-lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(fire_points[:,0]))*np.sin(dlon/2)**2
        d = 2*6371.0*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return d.min() < min_distance_km
    neg_rows = []
    total_cells = len(xs)*len(ys)
    scanned = 0
    for y in ys:
        if len(neg_rows) >= N:
            break
        for x in xs:
            scanned += 1
            if len(neg_rows) >= N:
                break
            if too_close(y, x):
                continue
            neg_rows.append((y, x))
            if len(neg_rows) % 50000 == 0:
                print(f"[INFO] even_grid collected {len(neg_rows)} negatives (target {N}) scanned_cells={scanned}/{total_cells}")
        # Early exit heuristic if remaining potential cells insufficient
        remaining_cells = (len(ys) - (np.where(ys==y)[0][0]+1)) * len(xs)
        potential_total = len(neg_rows) + remaining_cells
        if potential_total < N:
            break
    deficit = N - len(neg_rows)
    if deficit > 0:
        rng = random.Random(42)
        attempts = 0
        max_attempts = deficit * 50
        while deficit > 0 and attempts < max_attempts:
            attempts += 1
            lon = rng.uniform(minx, maxx); lat = rng.uniform(miny, maxy)
            if too_close(lat, lon):
                continue
            neg_rows.append((lat, lon)); deficit -= 1
        if deficit > 0:
            print(f"[WARN] even_grid incomplete: needed {N}, got {len(neg_rows)} after random fill")
    neg_df = pd.DataFrame(neg_rows, columns=['latitude','longitude'])
    neg_df['fire'] = 0
    neg_df['source'] = 'even_grid'
    return neg_df

def _haversine_bulk(lat1, lon1, lat2_arr, lon2_arr):
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2_arr))*np.sin(dlon/2.0)**2
    return 2*6371.0*np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def load_grid_coordinates(nc_path: Path, x_var: str, y_var: str,
                          mask_var: str | None = None,
                          mask_nodata: float | None = None) -> np.ndarray:
    """Load latitude/longitude coordinates from a NetCDF grid."""
    if NetCDFFile is None:
        raise RuntimeError("scipy.io.netcdf_file is required for --grid-nc usage")
    if not nc_path.exists():
        raise FileNotFoundError(f"Grid NetCDF not found: {nc_path}")
    ds = NetCDFFile(str(nc_path), mode="r")
    try:
        if x_var not in ds.variables or y_var not in ds.variables:
            raise KeyError(f"Variables '{x_var}'/'{y_var}' not found in {nc_path}")
        xs = np.array(ds.variables[x_var][:], dtype=float)
        ys = np.array(ds.variables[y_var][:], dtype=float)
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("grid coordinate variables must be 1-D")
        lon_grid, lat_grid = np.meshgrid(xs, ys)
        valid = np.ones(lon_grid.shape, dtype=bool)
        if mask_var:
            if mask_var not in ds.variables:
                raise KeyError(f"Mask variable '{mask_var}' not found in {nc_path}")
            mask_data = np.array(ds.variables[mask_var][:])
            mask_data = np.squeeze(mask_data)
            while mask_data.ndim > 2:
                mask_data = mask_data[0]
            if mask_data.shape != valid.shape:
                raise ValueError("Mask variable shape does not match y/x grid")
            valid &= np.isfinite(mask_data)
            if mask_nodata is not None:
                valid &= (mask_data != mask_nodata)
        coords = np.column_stack([lat_grid[valid], lon_grid[valid]])
        return coords.astype(float)
    finally:
        ds.close()


def sample_negative_points_from_grid(grid_coords: np.ndarray, pos_df: pd.DataFrame, fire_ratio: float,
                                     min_distance_km: float, max_distance_km: float | None = None,
                                     seed: int = 42, target_override: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Select negative samples from a fixed coordinate grid (soil/climate)."""
    if grid_coords.size == 0:
        raise RuntimeError("Grid coordinate list is empty")
    if {"latitude","longitude"}.issubset(pos_df.columns):
        fire_points = pos_df[["latitude","longitude"]].to_numpy()
    else:
        raise RuntimeError("Positive dataframe missing latitude/longitude")
    if len(pos_df) == 0:
        raise RuntimeError("No positive fire samples available for grid sampling")
    if target_override is not None and target_override > 0:
        target_neg = int(target_override)
    else:
        target_neg = int(round(len(pos_df) * (1 / fire_ratio - 1)))
    target_neg = max(target_neg, 1)
    target_neg = max(target_neg, 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid_coords))
    # KDTree for nearest distance checks
    kdt = None
    if cKDTree is not None and fire_points.shape[0] > 0:
        try:
            kdt = cKDTree(fire_points[:, ::-1])
        except Exception:
            kdt = None
    fire_keys: set[str] = set()
    if "latlon_key" in pos_df.columns:
        fire_keys = set(pos_df["latlon_key"].astype(str))
    selected: list[tuple[float,float]] = []
    seen_keys: set[str] = set()
    processed = 0
    def nearest_km(lat: float, lon: float) -> float:
        if kdt is not None:
            dist_deg, _ = kdt.query([[lon, lat]], k=1)
            return float(dist_deg[0] * 111.0)
        dists = _haversine_bulk(lat, lon, fire_points[:,0], fire_points[:,1])
        return float(dists.min())
    for idx in order:
        processed += 1
        lat, lon = float(grid_coords[idx, 0]), float(grid_coords[idx, 1])
        if np.isnan(lat) or np.isnan(lon):
            continue
        key = f"{round(lat,5)}_{round(lon,5)}"
        if key in fire_keys or key in seen_keys:
            continue
        d_km = nearest_km(lat, lon)
        if d_km < min_distance_km:
            continue
        if max_distance_km is not None and max_distance_km > 0 and d_km > max_distance_km:
            continue
        selected.append((lat, lon))
        seen_keys.add(key)
        if len(selected) >= target_neg:
            break
    neg_df = pd.DataFrame(selected, columns=["latitude","longitude"])
    neg_df["fire"] = 0
    neg_df["source"] = "grid"
    metrics = {
        "source": "grid",
        "target_negatives": target_neg,
        "accepted_negatives": len(selected),
        "available_candidates": int(len(grid_coords)),
        "scanned_candidates": int(processed),
        "completed": len(selected) >= target_neg,
    }
    return neg_df, metrics


def fill_negative_fire_metadata(df: pd.DataFrame) -> Dict[str, int]:
    """Fill fire-specific columns for synthetic negatives so downstream steps see valid values."""
    if "fire" not in df.columns:
        return {}
    neg_mask = df["fire"] == 0
    if not neg_mask.any():
        return {}
    fill_counts: Dict[str, int] = {}
    for col, val in NEG_NUMERIC_DEFAULTS.items():
        if col in df.columns:
            missing = int(df.loc[neg_mask, col].isna().sum())
            if missing:
                df.loc[neg_mask, col] = df.loc[neg_mask, col].fillna(val)
                fill_counts[col] = missing
    for col, val in NEG_CATEGORICAL_DEFAULTS.items():
        if col in df.columns:
            missing = int(df.loc[neg_mask, col].isna().sum())
            if missing:
                df.loc[neg_mask, col] = df.loc[neg_mask, col].fillna(val)
                fill_counts[col] = missing
    if "acq_datetime" in df.columns:
        missing = int(df.loc[neg_mask, "acq_datetime"].isna().sum())
        if missing:
            if pd.api.types.is_datetime64_any_dtype(df["acq_datetime"]):
                df.loc[neg_mask, "acq_datetime"] = df.loc[neg_mask, "acq_datetime"].fillna(NEG_DATETIME_DEFAULT)
            else:
                df.loc[neg_mask, "acq_datetime"] = df.loc[neg_mask, "acq_datetime"].fillna(str(NEG_DATETIME_DEFAULT))
            fill_counts["acq_datetime"] = missing
    return fill_counts

# Fast random batch sampler

def sample_negative_points_random_batch(pos_df: pd.DataFrame, fire_ratio: float, min_distance_km: float,
                                        bbox: tuple[float,float,float,float], batch_size: int, max_seconds: int,
                                        neg_cap: int | None = None, seed: int = 42,
                                        max_distance_km: float | None = None,
                                        adapt: bool = True) -> tuple[pd.DataFrame, dict]:
    """Vectorized random batch negative sampling with bounded distance constraints.
    Accept candidates whose nearest fire distance d satisfies min_distance_km <= d <= max_distance_km.
    If adapt=True and acceptance rate is low, max_distance_km is widened progressively.
    Returns negative dataframe and performance metrics dict.
    """
    minx, miny, maxx, maxy = bbox
    fire_points = pos_df[['latitude','longitude']].to_numpy()
    P = len(pos_df)
    target_neg = int(round(P * (1/fire_ratio - 1)))
    if neg_cap is not None:
        target_neg = min(target_neg, neg_cap)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.utcnow()
    accepted_lat: list[float] = []
    accepted_lon: list[float] = []
    attempts = 0
    batches = 0
    widen_events: list[dict] = []
    relax_events: list[dict] = []
    # KDTree for nearest distances (in degrees)
    kdt = None
    if cKDTree is not None and fire_points.shape[0] > 0:
        try:
            kdt = cKDTree(fire_points[:, ::-1])  # lon, lat order
        except Exception:
            kdt = None
    # Distance thresholds in degrees
    min_d_km = min_distance_km
    max_d_km = max_distance_km if (max_distance_km and max_distance_km > min_d_km) else None
    state = {
        'min_deg': min_d_km / 111.0,
        'max_deg': (max_d_km / 111.0) if max_d_km else None,
        'min_base_deg': min_d_km / 111.0
    }
    def maybe_adapt(elapsed: float, accepted: int):
        if not adapt:
            return
        remaining = target_neg - accepted
        if remaining <= 0:
            return
        acceptance_rate = accepted / attempts if attempts else 0.0
        # Relax min distance
        if acceptance_rate < 0.005 and elapsed > max_seconds/4.0 and state['min_deg'] > 0.8 * state['min_base_deg']:
            new_min_deg = max(0.8 * state['min_base_deg'], state['min_deg'] * 0.9)
            relax_events.append({'elapsed': round(elapsed,1), 'old_min_km': state['min_deg']*111.0, 'new_min_km': new_min_deg*111.0})
            state['min_deg'] = new_min_deg
        # Widen max distance
        if state['max_deg'] is not None and acceptance_rate < 0.01 and elapsed > max_seconds/3.0:
            new_max_deg = state['max_deg'] * 1.4
            widen_events.append({'elapsed': round(elapsed,1), 'old_max_km': state['max_deg']*111.0, 'new_max_km': new_max_deg*111.0})
            state['max_deg'] = new_max_deg
    # Safety limits
    max_total_attempts = target_neg * 50  # generous cap
    # Main loop
    while len(accepted_lat) < target_neg:
        batches += 1
        elapsed = (pd.Timestamp.utcnow() - start).total_seconds()
        if elapsed > max_seconds or attempts > max_total_attempts:
            break
        cand_lon = rng.uniform(minx, maxx, size=batch_size)
        cand_lat = rng.uniform(miny, maxy, size=batch_size)
        attempts += batch_size
        if kdt is not None:
            dist_deg, _ = kdt.query(np.vstack([cand_lon, cand_lat]).T, k=1)
            keep_mask = dist_deg >= state['min_deg']
            if state['max_deg'] is not None:
                keep_mask &= dist_deg <= state['max_deg']
        else:
            keep_mask = np.ones(batch_size, dtype=bool)
            fp_sub = fire_points
            if fp_sub.shape[0] > 10000:
                idx_sub = rng.choice(fp_sub.shape[0], size=10000, replace=False)
                fp_sub = fp_sub[idx_sub]
            for i in range(batch_size):
                la = cand_lat[i]; lo = cand_lon[i]
                ddeg2 = (fp_sub[:,0]-la)**2 + (fp_sub[:,1]-lo)**2
                mindeg = math.sqrt(ddeg2.min())
                if mindeg < state['min_deg'] or (state['max_deg'] is not None and mindeg > state['max_deg']):
                    keep_mask[i] = False
        accepted_indices = np.where(keep_mask)[0]
        space_left = target_neg - len(accepted_lat)
        if accepted_indices.size > space_left:
            accepted_indices = accepted_indices[:space_left]
        if accepted_indices.size:
            accepted_lat.extend(cand_lat[accepted_indices].tolist())
            accepted_lon.extend(cand_lon[accepted_indices].tolist())
        # Log every batch (early visibility)
        acc_rate = len(accepted_lat)/attempts if attempts else 0.0
        print(f"[INFO] batch={batches} accepted={len(accepted_lat)} target={target_neg} attempts={attempts} acc_rate={acc_rate:.4f} elapsed={elapsed:.1f}s min_km={state['min_deg']*111.0:.2f} max_km={(state['max_deg']*111.0 if state['max_deg'] else None)}")
        maybe_adapt(elapsed, len(accepted_lat))
    elapsed_total = (pd.Timestamp.utcnow() - start).total_seconds()
    neg_df = pd.DataFrame({'latitude': accepted_lat, 'longitude': accepted_lon})
    neg_df['fire'] = 0
    neg_df['source'] = 'random_batch'
    completed = len(accepted_lat) >= target_neg
    metrics = {
        'target_negatives': target_neg,
        'accepted_negatives': len(accepted_lat),
        'attempts': attempts,
        'batches': batches,
        'elapsed_seconds': round(elapsed_total,2),
        'acceptance_rate': round(len(accepted_lat)/attempts,6) if attempts else 0.0,
        'completed': completed,
        'partial': not completed,
        'min_distance_km_final': round(state['min_deg']*111.0,3),
        'max_distance_km_final': round(state['max_deg']*111.0,3) if state['max_deg'] else None,
        'adaptive_widens': widen_events,
        'adaptive_relax': relax_events
    }
    return neg_df, metrics

def sample_negative_points_band(pos_df: pd.DataFrame, fire_ratio: float, min_km: float, max_km: float,
                                aoi_path: Path | None = None, seed: int = 42) -> pd.DataFrame:
    """Generate negatives in a strict distance band [min_km, max_km] around fire points.
    Strategy: For each fire point generate N_per_fire ring samples with random angle and random distance.
    Distance conversion: 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 km * cos(lat_rad).
    If AOI provided (shapefile), filter points outside polygon union. Ensures all negatives have
    distance >= min_km and <= max_km (subject to small numeric error).
    """
    if min_km <= 0 or max_km <= 0 or max_km <= min_km:
        raise ValueError("Invalid band distances; require 0 < min_km < max_km")
    if pos_df.empty:
        raise RuntimeError("No fire points available for band sampling")
    fire_points = pos_df[['latitude','longitude']].to_numpy()
    P = len(fire_points)
    target_neg = int(round(P * (1/fire_ratio - 1)))
    # per-fire negatives (ceil) may overshoot slightly
    neg_per_fire = max(1, math.ceil(target_neg / P))
    rng = np.random.default_rng(seed)
    lat_arr = fire_points[:,0]
    lon_arr = fire_points[:,1]
    # Prepare AOI geometry if provided
    aoi_geom = None
    if aoi_path and aoi_path.exists() and gpd is not None:
        try:
            gdf_aoi = gpd.read_file(aoi_path)
            aoi_geom = getattr(gdf_aoi, 'union_all', gdf_aoi.unary_union)()
            if not isinstance(aoi_geom, (Polygon, MultiPolygon)):
                aoi_geom = None
        except Exception:
            aoi_geom = None
    neg_rows = []
    # Vector loop per fire point
    for i in range(P):
        lat0 = float(lat_arr[i]); lon0 = float(lon_arr[i])
        lat_rad = math.radians(lat0)
        # sample distances & angles
        dists = rng.uniform(min_km, max_km, size=neg_per_fire)
        thetas = rng.uniform(0, 2*math.pi, size=neg_per_fire)
        # local cartesian offsets (km)
        dx = dists * np.cos(thetas)  # east-west km
        dy = dists * np.sin(thetas)  # north-south km
        # convert to degrees
        dlat_deg = dy / 111.0
        # avoid division by zero near poles (not relevant here but guard)
        cos_lat = max(1e-6, math.cos(lat_rad))
        dlon_deg = dx / (111.0 * cos_lat)
        lat_new = lat0 + dlat_deg
        lon_new = lon0 + dlon_deg
        if aoi_geom is not None:
            for la, lo in zip(lat_new, lon_new):
                pt = Point(lo, la)
                if not aoi_geom.contains(pt):
                    continue
                neg_rows.append((la, lo))
                if len(neg_rows) >= target_neg:
                    break
        else:
            for la, lo in zip(lat_new, lon_new):
                neg_rows.append((la, lo))
                if len(neg_rows) >= target_neg:
                    break
        if len(neg_rows) >= target_neg:
            break
    # Truncate overshoot
    if len(neg_rows) > target_neg:
        neg_rows = neg_rows[:target_neg]
    # Build DataFrame
    neg_df = pd.DataFrame(neg_rows, columns=['latitude','longitude'])
    neg_df['fire'] = 0
    neg_df['source'] = 'band'
    # Validate distances (optional) compute nearest fire distance for sample subset
    try:
        if cKDTree is not None and not neg_df.empty:
            tree = cKDTree(fire_points[:, ::-1])
            dist_deg, _ = tree.query(neg_df[['longitude','latitude']].to_numpy(), k=1)
            dist_km = dist_deg * 111.0
            # filter any accidental < min_km or > max_km due to projection approximations
            mask = (dist_km >= (min_km - 0.05)) & (dist_km <= (max_km + 0.05))
            neg_df = neg_df.loc[mask].reset_index(drop=True)
    except Exception:
        pass
    # If after filtering we lost rows, attempt simple refill via rejection sampling around all fires
    deficit = target_neg - len(neg_df)
    if deficit > 0:
        attempts = 0
        max_attempts = deficit * 50
        while len(neg_df) < target_neg and attempts < max_attempts:
            attempts += 1
            # pick random fire as center
            idx = rng.integers(0, P)
            lat0 = float(lat_arr[idx]); lon0 = float(lon_arr[idx])
            lat_rad = math.radians(lat0)
            d = rng.uniform(min_km, max_km)
            theta = rng.uniform(0, 2*math.pi)
            dx = d * math.cos(theta); dy = d * math.sin(theta)
            la = lat0 + dy/111.0
            lo = lon0 + dx/(111.0 * max(1e-6, math.cos(lat_rad)))
            if aoi_geom is not None and not aoi_geom.contains(Point(lo, la)):
                continue
            neg_df.loc[len(neg_df)] = [la, lo, 0, 'band']
    return neg_df

def build_dataset(conf_min: int, exclude_types: List[int], fire_ratio: float, min_distance_km: float,
                  aoi_path: Path, out_dir: Path, eda: bool, max_negatives: int | None,
                  neg_method: str, landcover_path: Path | None, even_neg: bool = False, neg_count: int | None = None,
                  batch_size: int = 50000, max_sampling_seconds: int = 180, neg_cap: int | None = None,
                  min_neg_dist_km: float | None = None, max_neg_dist_km: float | None = None,
                  fire_glob: str = "*.csv", grid_nc: Path | None = None,
                  grid_x_var: str = "x", grid_y_var: str = "y",
                  grid_mask_var: str | None = None, grid_mask_nodata: float | None = None) -> Dict[str,str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fire_paths = parse_fire_files(FIRE_DIR, fire_glob)
    if exclude_types:
        fire_paths = [p for p in fire_paths if not ("type2" in p.name.lower() and 2 in exclude_types)]
    if not fire_paths:
        return {"status":"no_fire_csv_after_exclusion", "pattern": fire_glob}
    fire_raw = load_and_standardize(fire_paths)
    if fire_raw.empty:
        return {"status":"empty_fire"}
    fire_filt = filter_fire(fire_raw, conf_min, exclude_types)
    if fire_filt.empty:
        fire_filt = fire_raw.copy(); fire_filt["filter_relaxed"] = 1
    else:
        fire_filt["filter_relaxed"] = 0
    fire_filt["fire"] = 1
    # Add coordinate rounding and key before negative sampling
    fire_filt["latitude_r"] = fire_filt["latitude"].round(5)
    fire_filt["longitude_r"] = fire_filt["longitude"].round(5)
    fire_filt["latlon_key"] = fire_filt["latitude_r"].astype(str) + "_" + fire_filt["longitude_r"].astype(str)
    fire_clean, dropped = drop_useless_columns(fire_filt)
    # Preserve rounding/key through cleaned df
    if "latitude_r" not in fire_clean.columns:
        fire_clean["latitude_r"] = fire_clean["latitude"].round(5)
    if "longitude_r" not in fire_clean.columns:
        fire_clean["longitude_r"] = fire_clean["longitude"].round(5)
    if "latlon_key" not in fire_clean.columns:
        fire_clean["latlon_key"] = fire_clean["latitude_r"].astype(str) + "_" + fire_clean["longitude_r"].astype(str)
    # Drop duplicate coordinate keys to avoid exploding merge size later
    fire_clean = fire_clean.drop_duplicates(subset=["latlon_key"]).reset_index(drop=True)
    essential = ["latitude","longitude","fire"]
    for col in essential:
        if col not in fire_clean.columns:
            return {"status":"missing_essential", "missing": col}
    pos_count = len(fire_clean)
    if pos_count == 0:
        return {"status":"no_positive_after_fallback"}
    batch_metrics = {}
    grid_coords = None
    if grid_nc is not None:
        try:
            grid_coords = load_grid_coordinates(grid_nc, grid_x_var, grid_y_var, mask_var=grid_mask_var, mask_nodata=grid_mask_nodata)
        except Exception as exc:
            return {"status": "grid_load_error", "error": str(exc), "grid_path": str(grid_nc)}
    effective_min = min_neg_dist_km if min_neg_dist_km is not None else min_distance_km
    method_used = neg_method
    # Negative sampling
    try:
        if grid_coords is not None:
            if neg_method != "grid":
                print(json.dumps({"info": "grid_nc_override", "requested_neg_method": neg_method, "used": "grid"}))
            method_used = "grid"
            target_override = int(round(pos_count * (1 / fire_ratio - 1)))
            if max_negatives is not None:
                target_override = min(target_override, max_negatives)
            if neg_cap is not None:
                target_override = min(target_override, neg_cap)
            neg_df, batch_metrics = sample_negative_points_from_grid(grid_coords, fire_clean, fire_ratio, effective_min, max_distance_km=max_neg_dist_km, target_override=target_override)
        elif neg_method == "polygons":
            neg_df = sample_negative_points_polygons(aoi_path, landcover_path, fire_clean, fire_ratio, max_negatives=max_negatives, min_distance_km=effective_min, max_distance_km=max_neg_dist_km)
        elif neg_method == "polygons-grid":
            poly_source = landcover_path if (landcover_path and landcover_path.exists()) else aoi_path
            neg_df = sample_negative_points_polygons_grid(poly_source, fire_clean, fire_ratio, effective_min, override_neg_count=neg_count, grid_res_deg=0.05, max_negatives=max_negatives, seed=42)
        elif neg_method == "random-batch":
            miny, maxy = fire_clean['latitude'].min(), fire_clean['latitude'].max()
            minx, maxx = fire_clean['longitude'].min(), fire_clean['longitude'].max()
            neg_df, batch_metrics = sample_negative_points_random_batch(
                fire_clean, fire_ratio, effective_min,
                (minx,miny,maxx,maxy), batch_size, max_sampling_seconds,
                neg_cap=neg_cap, seed=42,
                max_distance_km=max_neg_dist_km, adapt=True)
        elif neg_method == "band":
            if max_neg_dist_km is None:
                raise ValueError("band method requires --max-neg-dist-km > min distance")
            neg_df = sample_negative_points_band(fire_clean, fire_ratio, effective_min, max_neg_dist_km, aoi_path=aoi_path)
        else:
            if even_neg and gpd is None:
                neg_df = sample_negative_points_even_grid(fire_clean, fire_ratio, effective_min, override_neg_count=neg_count)
            else:
                neg_df = sample_negative_points(aoi_path, fire_clean, target_positive_ratio=fire_ratio, min_distance_km=effective_min, max_negatives=max_negatives, override_neg_count=neg_count, attempt_mult=400)
    except Exception as e:
        return {"status":"neg_sampling_error", "error": str(e), "method": neg_method}
    # Add rounded coords and key to negatives
    neg_df["latitude_r"] = neg_df["latitude"].round(5)
    neg_df["longitude_r"] = neg_df["longitude"].round(5)
    neg_df["latlon_key"] = neg_df["latitude_r"].astype(str) + "_" + neg_df["longitude_r"].astype(str)
    # Ensure uniqueness of negative keys (drop duplicates if any)
    neg_df = neg_df.drop_duplicates(subset=["latlon_key"]).reset_index(drop=True)
    combined = pd.concat([fire_clean[fire_clean.columns], neg_df], ignore_index=True)
    combined = combined.sample(len(combined), random_state=0).reset_index(drop=True)
    neg_fill_counts = fill_negative_fire_metadata(combined)
    pq_path = out_dir / ("fire_with_negatives_polygons.parquet" if neg_method == "polygons" else "fire_with_negatives.parquet")
    csv_path = out_dir / pq_path.with_suffix(".csv").name
    try:
        combined.to_parquet(pq_path, index=False)
    except Exception as e:
        print(f"[WARN] Parquet write failed: {e}")
    combined.to_csv(csv_path, index=False)
    summary_path = out_dir / ("fire_with_negatives_polygons_summary.json" if neg_method == "polygons" else "fire_with_negatives_summary.json")
    if eda:
        eda_summary(combined, summary_path)
    fire_ratio_out = combined["fire"].mean()
    nonfire_ratio_out = 1.0 - fire_ratio_out
    # Compute fire-fire nearest neighbor stats
    nn_stats = _nearest_neighbor_stats(fire_clean['latitude'].to_numpy(), fire_clean['longitude'].to_numpy())
    # Compute negative-fire distance metrics
    try:
        fp = fire_clean[['latitude','longitude']].to_numpy()
        np_neg = neg_df[['latitude','longitude']].to_numpy()
        if fp.size and np_neg.size:
            if cKDTree is not None:
                tree2 = cKDTree(fp[:, ::-1])
                dist_deg, _ = tree2.query(np_neg[:, ::-1], k=1)
                dist_km = dist_deg * 111.0
            else:
                dist_km = []
                for la, lo in np_neg:
                    dists = _haversine_bulk(la, lo, fp[:,0], fp[:,1])
                    dist_km.append(dists.min())
                dist_km = np.array(dist_km)
            neg_dist_metrics = {
                'neg_fire_min_km': float(np.min(dist_km)),
                'neg_fire_median_km': float(np.median(dist_km)),
                'neg_fire_mean_km': float(np.mean(dist_km))
            }
        else:
            neg_dist_metrics = {'neg_fire_min_km': None,'neg_fire_median_km': None,'neg_fire_mean_km': None}
    except Exception as e:
        neg_dist_metrics = {'neg_fire_distance_error': str(e)}
    return {
        "status":"ok" if batch_metrics.get('completed', True) else "partial",
        "rows": str(len(combined)),
        "positives": str(pos_count),
        "fire_ratio": f"{fire_ratio_out:.3f}",
        "nonfire_ratio": f"{nonfire_ratio_out:.3f}",
        "dropped": ",".join(dropped),
        "relaxed_filter": str(int(fire_filt['filter_relaxed'].iloc[0])) if 'filter_relaxed' in fire_filt else '0',
        "neg_method": method_used,
        "batch_metrics": batch_metrics if method_used in ('random-batch','grid') else {},
        "grid_source": str(grid_nc) if grid_coords is not None else "",
        "grid_candidates": str(len(grid_coords)) if grid_coords is not None else "",
        **neg_dist_metrics,
        "neg_fill_counts": neg_fill_counts,
        "parquet": pq_path.as_posix(),
        "csv": csv_path.as_posix(),
        "eda_summary": summary_path.as_posix() if eda else "",
        "fire_nn_min_km": nn_stats.get('min_km'),
        "fire_nn_median_km": nn_stats.get('median_km'),
        "fire_nn_mean_km": nn_stats.get('mean_km')
    }

def main():
    ap = argparse.ArgumentParser(description="Build fire/no-fire classification dataset with negative sampling")
    ap.add_argument("--aoi", type=Path, default=BASE / "shapefiles" / "algeria_tunisia.shp", help="AOI polygon shapefile")
    ap.add_argument("--confidence-min", type=int, default=0, help="Minimum confidence to keep fire pixels")
    ap.add_argument("--exclude-types", default="", help="Comma list of type codes to exclude (e.g. 2)")
    ap.add_argument("--fire-ratio", type=float, default=None, help="Desired fraction of fire points (positives) in final dataset (recommended 0.10-0.30). Overrides --positive-ratio if set.")
    ap.add_argument("--nonfire-ratio", type=float, default=None, help="Desired fraction of non-fire points; fire_ratio=1-nonfire_ratio if provided.")
    ap.add_argument("--even-negatives", action="store_true", help="Use grid-based even distribution for fallback (no geopandas) negative sampling.")
    ap.add_argument("--min-distance-km", type=float, default=1.0, help="Minimum distance (km) between negative samples and any fire point")
    ap.add_argument("--max-negatives", type=int, default=None, help="Optional cap on number of negative samples")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    ap.add_argument("--eda-summary", action="store_true", help="Write EDA summary JSON")
    ap.add_argument("--batch-size", type=int, default=50000, help="Candidate negatives per batch for random-batch method")
    ap.add_argument("--max-sampling-seconds", type=int, default=180, help="Time budget (seconds) for negative sampling random-batch")
    ap.add_argument("--neg-cap", type=int, default=None, help="Hard cap on negatives regardless of ratio (for random-batch/even grid)")
    ap.add_argument("--neg-method", choices=["random","polygons","polygons-grid","random-batch","band","grid"], default="random", help="Negative sampling strategy")
    ap.add_argument("--landcover", type=Path, default=BASE / "LANDCOVER" / "processed" / "landcover_clipped.shp", help="Landcover shapefile for polygon-based negatives (if exists)")
    ap.add_argument("--adjust-negative-ratio", action="store_true", help="After initial build, if positive ratio > target, generate additional negatives to meet target.")
    ap.add_argument("--neg-count", type=int, default=None, help="Explicit number of negative samples (overrides positive-ratio if set)")
    ap.add_argument("--neg-attempt-mult", type=int, default=200, help="Multiplier for max attempts in negative sampling (increase if using very large neg-count)")
    ap.add_argument("--polygon-grid-res", type=float, default=0.05, help="Grid resolution in degrees for polygons-grid method (~0.01 ≈ 1km)")
    ap.add_argument("--positive-ratio", type=float, default=None, help="(Legacy) Previously fire ratio or non-fire ratio depending on value; prefer --fire-ratio or --nonfire-ratio.")
    ap.add_argument("--min-neg-dist-km", type=float, default=1.0, help="Minimum distance (km) from any fire point for a negative sample (default 1.0)")
    ap.add_argument("--max-neg-dist-km", type=float, default=3.0, help="Maximum distance (km) from any fire point for a negative sample (default 3.0). Set <=0 to disable upper bound.")
    ap.add_argument("--grid-nc", type=Path, default=None, help="NetCDF (soil/climate) containing coordinate grid for negatives")
    ap.add_argument("--grid-x-var", default="x", help="Longitude variable name inside --grid-nc")
    ap.add_argument("--grid-y-var", default="y", help="Latitude variable name inside --grid-nc")
    ap.add_argument("--grid-mask-var", default=None, help="Optional mask variable (NaNs or == grid-mask-nodata are discarded)")
    ap.add_argument("--grid-mask-nodata", type=float, default=None, help="Value in mask variable treated as nodata")
    ap.add_argument("--fire-file-glob", default="*.csv", help="Glob relative to FIRE/ to choose fire CSVs (e.g. '*type0*.csv').")
    args = ap.parse_args()

    # Resolve fire ratio semantics
    legacy_ratio = args.positive_ratio
    fire_ratio = None
    if args.fire_ratio is not None and args.nonfire_ratio is not None:
        print(json.dumps({"status":"error","error":"Specify only one of --fire-ratio or --nonfire-ratio"}))
        return
    if args.fire_ratio is not None:
        fire_ratio = args.fire_ratio
    elif args.nonfire_ratio is not None:
        fire_ratio = 1.0 - args.nonfire_ratio
    else:
        # Legacy interpretation
        if legacy_ratio > 0.5:
            # Treat as non-fire ratio
            fire_ratio = 1.0 - legacy_ratio
            print(json.dumps({"info":"legacy_positive_ratio_interpreted_as_nonfire_ratio","legacy_value":legacy_ratio,"derived_fire_ratio":fire_ratio}, indent=2))
        else:
            fire_ratio = legacy_ratio
            print(json.dumps({"info":"legacy_positive_ratio_interpreted_as_fire_ratio","fire_ratio":fire_ratio}, indent=2))
    if fire_ratio <= 0 or fire_ratio >= 0.5:
        print(json.dumps({"warn":"fire_ratio_outside_recommended","fire_ratio":fire_ratio,"recommended":"0.10-0.30"}, indent=2))
    args.fire_ratio_effective = fire_ratio

    exclude_types = [int(x) for x in args.exclude_types.split(",") if x.strip()] if args.exclude_types else []
    # Normalize max distance: <=0 means no upper bound
    if args.max_neg_dist_km is not None and args.max_neg_dist_km <= 0:
        args.max_neg_dist_km = None
    res = build_dataset(args.confidence_min, exclude_types, args.fire_ratio_effective, args.min_distance_km, args.aoi, args.out, args.eda_summary, args.max_negatives, args.neg_method, args.landcover, even_neg=args.even_negatives, neg_count=args.neg_count, batch_size=args.batch_size, max_sampling_seconds=args.max_sampling_seconds, neg_cap=args.neg_cap, min_neg_dist_km=args.min_neg_dist_km, max_neg_dist_km=args.max_neg_dist_km, fire_glob=args.fire_file_glob, grid_nc=args.grid_nc, grid_x_var=args.grid_x_var, grid_y_var=args.grid_y_var, grid_mask_var=args.grid_mask_var, grid_mask_nodata=args.grid_mask_nodata)
    # Post-adjust negative ratio if requested
    if res.get('status') == 'ok' and args.adjust_negative_ratio:
        try:
            pq_path = Path(res['parquet'])
            df = pd.read_parquet(pq_path)
            P = int((df['fire'] == 1).sum()); N = int((df['fire'] == 0).sum())
            fire_ratio_actual = P / max(1, (P + N))
            target = args.fire_ratio_effective
            if fire_ratio_actual > target:
                # Need more negatives to reach desired fire ratio (target = fraction of fires)
                needed_N = int(round(P * (1.0 / target - 1.0)))
                extra = max(0, needed_N - N)
                if extra > 0:
                    add_neg = sample_negative_points(args.aoi, df[df['fire'] == 1], target_positive_ratio=target, min_distance_km=args.min_distance_km, max_negatives=extra, override_neg_count=None, attempt_mult=args.neg_attempt_mult)
                    df2 = pd.concat([df, add_neg], ignore_index=True)
                    df2 = df2.sample(len(df2), random_state=1).reset_index(drop=True)
                    try:
                        df2.to_parquet(pq_path, index=False)
                    except Exception:
                        pass
                    df2.to_csv(pq_path.with_suffix('.csv'), index=False)
                    new_ratio = (df2['fire'] == 1).sum() / max(1, len(df2))
                    res['adjusted_fire_ratio'] = f"{new_ratio:.3f}"
                    res['added_negatives'] = str(extra)
                else:
                    res['adjusted_fire_ratio'] = f"{fire_ratio_actual:.3f}"
                    res['added_negatives'] = '0'
            else:
                res['adjusted_fire_ratio'] = f"{fire_ratio_actual:.3f}"
                res['added_negatives'] = '0'
        except Exception as e:
            res['adjust_error'] = str(e)
    print(json.dumps(res, indent=2))
    return

if __name__ == "__main__":
    main()
