#!/usr/bin/env python3
"""Transform HWSD2 soil dataset into tabular per-pixel or point-sampled features.

Steps implemented:
  1. Load layer attribute table (preferred: processed/HWSD2_LAYERS_D1.csv else HWSD2_LAYERS.csv and filter LAYER==D1).
  2. Keep only specified feature columns for surface horizon (D1 ~ top 0-20cm):
        HWSD2_SMU_ID, COARSE, SAND, SILT, CLAY, TEXTURE_USDA, TEXTURE_SOTER,
        BULK, REF_BULK, ORG_CARBON, PH_WATER, TOTAL_N, CN_RATIO, CEC_SOIL,
        CEC_CLAY, CEC_EFF, TEB, BSAT, ALUM_SAT, ESP, TCARBON_EQ, GYPSUM, ELEC_COND
  3. Load soil id raster NetCDF (auto-detect among soil_id_with_attrs_aoi.nc, soil_id_aoi_cropped.nc, soil_id_aoi.nc).
  4. Map each pixel soil ID to its D1 attributes; optionally downsample (stride) or sample fraction / point file.
  5. Optional point sampling using a fire/no-fire dataset or any CSV/Parquet with lat/lon columns.
  6. Preprocessing: numeric coercion, percentage normalization (ensure fractions sum), derived features, outlier capping (IQR), one-hot textures.
  7. EDA summary (stats + top categories) and capping report.

CLI examples:
  python scripts/transform_soil.py --eda
  python scripts/transform_soil.py --stride 4 --sample-frac 0.05 --max-rows 200000 --eda --cap-iqr --one-hot-texture
  python scripts/transform_soil.py --sample-points DATA_CLEANED/processed/fire_with_negatives.parquet --eda --derived

Outputs:
  DATA_CLEANED/processed/soil_features.parquet
  DATA_CLEANED/processed/soil_features.csv
  soil_features_summary.json (+ soil_features_capping.json if capping)

Derived features (if --derived flag):
  fine_fraction = 100 - COARSE
  clay_to_silt_ratio = CLAY / (SILT+1)
  sand_to_clay_ratio = SAND / (CLAY+1)
  organic_bulk_density = (ORG_CARBON/100) * BULK (approx)
  base_saturation_ratio = BSAT / (CEC_SOIL+1)
  cec_eff_ratio = CEC_EFF / (CEC_SOIL+1)
  nutrient_index = (TOTAL_N + ORG_CARBON/100) / (BULK+1)

Assumptions:
  - Percent columns (COARSE,SAND,SILT,CLAY,BSAT,ALUM_SAT,ESP,GYPSUM) in 0-100 scale.
  - Soil ID field is HWSD2_SMU_ID in table and variable containing 'soil' or 'id' in NetCDF.

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parents[1]
PROC_SOIL = BASE / "SOIL" / "processed"
RAW_SOIL = BASE / "SOIL"
OUT_DIR_DEFAULT = BASE / "DATA_CLEANED" / "processed"

FEATURE_COLS = [
    "HWSD2_SMU_ID","COARSE","SAND","SILT","CLAY","TEXTURE_USDA","TEXTURE_SOTER","BULK","REF_BULK",
    "ORG_CARBON","PH_WATER","TOTAL_N","CN_RATIO","CEC_SOIL","CEC_CLAY","CEC_EFF","TEB","BSAT","ALUM_SAT",
    "ESP","TCARBON_EQ","GYPSUM","ELEC_COND"
]

NC_CANDIDATES = [
    PROC_SOIL / "soil_id_with_attrs_aoi.nc",
    PROC_SOIL / "soil_id_aoi_cropped.nc",
    PROC_SOIL / "soil_id_aoi.nc",
    BASE / "DATA_CLEANED" / "SOIL" / "soil_id_grid.nc",
]

def load_layer_table() -> pd.DataFrame:
    # prefer pre-filtered D1 file if present
    d1 = PROC_SOIL / "HWSD2_LAYERS_D1.csv"
    if d1.exists():
        df = pd.read_csv(d1)
    else:
        raw = RAW_SOIL / "HWSD2_LAYERS.csv"
        if not raw.exists():
            raise FileNotFoundError("HWSD2_LAYERS.csv not found.")
        # stream reading to reduce memory (optionally)
        df = pd.read_csv(raw, low_memory=False)
        # handle layer identifier; accept 'D1' or 1
        layer_col = None
        for c in df.columns:
            if c.upper() == 'LAYER':
                layer_col = c
                break
        if layer_col is None:
            raise ValueError("No LAYER column found in soil layers table.")
        df = df[df[layer_col].astype(str).str.upper().eq('D1')]
    return df

def subset_features(df: pd.DataFrame) -> pd.DataFrame:
    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing feature columns: {missing}")
    sub = df[avail].copy()
    return sub

def load_soil_id_grid() -> xr.Dataset:
    for p in NC_CANDIDATES:
        if p.exists():
            try:
                ds = xr.open_dataset(p)
                return ds
            except Exception:
                continue
    raise FileNotFoundError("No soil ID NetCDF found among candidates.")

def detect_id_var(ds: xr.Dataset) -> str:
    for name in list(ds.data_vars):
        nm = str(name)
        if any(tok in nm.lower() for tok in ["soil","id","smu"]):
            return nm
    for name in list(ds.coords):
        nm = str(name)
        if any(tok in nm.lower() for tok in ["soil","id","smu"]):
            return nm
    raise ValueError("Could not detect soil ID variable in NetCDF.")

def build_mapping(df: pd.DataFrame) -> Dict[int, Dict[str,float|str]]:
    id_col = 'HWSD2_SMU_ID'
    if id_col not in df.columns:
        # attempt case-insensitive match
        for c in df.columns:
            if c.lower() == 'hwsd2_smu_id':
                id_col = c
                break
        if id_col not in df.columns:
            raise ValueError("HWSD2_SMU_ID column not found in feature table.")
    df = df.drop_duplicates(subset=[id_col])
    mapping = {}
    for _, row in df.iterrows():
        sid = row[id_col]
        # only keep numeric/str relevant
        record = {}
        for c in df.columns:
            if c == id_col:
                continue
            val = row[c]
            if pd.isna(val):
                continue
            record[c] = val
        mapping[int(sid)] = record
    return mapping

def iqr_cap(arr: np.ndarray, factor: float = 1.5) -> tuple[np.ndarray,int,float,float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 12:
        return arr,0,np.nan,np.nan
    q1,q3 = np.percentile(finite,[25,75])
    iqr = q3-q1
    if iqr <= 0:
        return arr,0,np.nan,np.nan
    lo = q1 - factor*iqr
    hi = q3 + factor*iqr
    mask = (arr < lo) | (arr > hi)
    capped = np.where(arr < lo, lo, np.where(arr > hi, hi, arr))
    return capped, int(np.isfinite(arr[mask]).sum()), lo, hi

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived pedological ratios/features in-place on provided DataFrame."""
    if df is None:
        raise ValueError("add_derived_features received None")
    # Fine fraction
    if 'COARSE' in df.columns and pd.api.types.is_numeric_dtype(df['COARSE']):
        df['fine_fraction'] = 100.0 - pd.to_numeric(df['COARSE'], errors='coerce')
    # Clay/Silt
    if {'CLAY','SILT'}.issubset(df.columns):
        clay = pd.to_numeric(df['CLAY'], errors='coerce')
        silt = pd.to_numeric(df['SILT'], errors='coerce')
        df['clay_to_silt_ratio'] = clay / (silt + 1.0)
    # Sand/Clay
    if {'SAND','CLAY'}.issubset(df.columns):
        sand = pd.to_numeric(df['SAND'], errors='coerce')
        clay = pd.to_numeric(df['CLAY'], errors='coerce')
        df['sand_to_clay_ratio'] = sand / (clay + 1.0)
    # Organic bulk density
    if {'ORG_CARBON','BULK'}.issubset(df.columns):
        oc = pd.to_numeric(df['ORG_CARBON'], errors='coerce')
        bulk = pd.to_numeric(df['BULK'], errors='coerce')
        df['organic_bulk_density'] = (oc/100.0) * bulk
    # Base saturation ratio
    if {'BSAT','CEC_SOIL'}.issubset(df.columns):
        bsat = pd.to_numeric(df['BSAT'], errors='coerce')
        cec_soil = pd.to_numeric(df['CEC_SOIL'], errors='coerce')
        df['base_saturation_ratio'] = bsat / (cec_soil + 1.0)
    # CEC efficiency ratio
    if {'CEC_EFF','CEC_SOIL'}.issubset(df.columns):
        cec_eff = pd.to_numeric(df['CEC_EFF'], errors='coerce')
        cec_soil = pd.to_numeric(df['CEC_SOIL'], errors='coerce')
        df['cec_eff_ratio'] = cec_eff / (cec_soil + 1.0)
    # Nutrient index
    if {'TOTAL_N','ORG_CARBON','BULK'}.issubset(df.columns):
        total_n = pd.to_numeric(df['TOTAL_N'], errors='coerce')
        oc = pd.to_numeric(df['ORG_CARBON'], errors='coerce')
        bulk = pd.to_numeric(df['BULK'], errors='coerce')
        df['nutrient_index'] = (total_n + oc/100.0) / (bulk + 1.0)
    return df

def one_hot_textures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base in ['TEXTURE_USDA','TEXTURE_SOTER']:
        # support lowercase columns
        if base in out.columns:
            series = out[base].astype(str)
        elif base.lower() in out.columns:
            series = out[base.lower()].astype(str)
            # normalize name by adding uppercase original for clarity
            out[base] = series
        else:
            continue
        dummies = pd.get_dummies(series, prefix=base.lower())
        out = pd.concat([out, dummies], axis=1)
    return out

def eda_summary(df: pd.DataFrame, out_path: Path) -> None:
    numeric = {c: df[c].describe().to_dict() for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    categorical = {c: df[c].value_counts().head(40).to_dict() for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
    summary = {"rows": len(df), "columns": list(df.columns), "numeric": numeric, "categorical": categorical}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Transform HWSD2 soil D1 layer into feature table")
    ap.add_argument('--out', type=Path, default=OUT_DIR_DEFAULT, help='Output directory')
    ap.add_argument('--eda', action='store_true', help='Write EDA summary JSON')
    ap.add_argument('--cap-iqr', action='store_true', help='Winsor IQR cap numeric columns')
    ap.add_argument('--derived', action='store_true', help='Add derived ratio / density features')
    ap.add_argument('--one-hot-texture', action='store_true', help='One-hot encode texture categorical columns')
    ap.add_argument('--stride', type=int, default=1, help='Spatial stride for downsampling raster grid')
    ap.add_argument('--sample-frac', type=float, default=None, help='Random sample fraction after stride')
    ap.add_argument('--max-rows', type=int, default=None, help='Cap number of rows')
    ap.add_argument('--sample-points', type=Path, default=None, help='CSV/Parquet with lat/lon for point sampling')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    ap.add_argument('--avoid-huge', action='store_true', help='If grid > 5e6 cells, auto stride to reduce size unless point sampling used')
    ap.add_argument('--parquet-compression', type=str, default='snappy', help='Parquet compression codec')
    ap.add_argument('--prefer-integrated', action='store_true', help='Use integrated soil_id_with_attrs NetCDF directly if available (skip CSV join)')
    args = ap.parse_args()

    df_raw = load_layer_table()
    df = subset_features(df_raw)
    # coerce numerics for numeric columns
    for c in df.columns:
        if c != 'HWSD2_SMU_ID' and c not in ['TEXTURE_USDA','TEXTURE_SOTER']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    mapping = build_mapping(df)

    ds = load_soil_id_grid()
    id_var = detect_id_var(ds)
    point_mode = False  # initialize early for JSON output later
    integrated_mode = args.prefer_integrated and ('soil_id_grid' in ds.data_vars) and ('soil_id' in ds.dims or 'soil_id' in ds.coords)
    # Extract coordinates
    lat_name = 'lat' if 'lat' in ds.coords else ('y' if 'y' in ds.coords else None)
    lon_name = 'lon' if 'lon' in ds.coords else ('x' if 'x' in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise ValueError('Could not find lat/lon or y/x coordinates in soil id grid.')
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    soil_df = None  # will assign below

    if integrated_mode:
        soil_grid = ds['soil_id_grid'].values
        soil_id_coord = ds['soil_id'].values if 'soil_id' in ds.coords else ds['soil_id'].values
        id_to_pos = {}
        for i, sid in enumerate(soil_id_coord):
            try:
                # Some coordinates may be tuples; pick first element if so
                val = sid[0] if isinstance(sid, (tuple, list)) else sid
                id_to_pos[int(val)] = i
            except Exception:
                continue
        wanted = {c.lower() for c in FEATURE_COLS if c != 'HWSD2_SMU_ID'}
        desired = [str(v) for v in ds.data_vars if str(v).lower() in wanted]
        H, W = soil_grid.shape
        flat_ids = soil_grid.reshape(-1)
        pos_array = np.array([id_to_pos.get(int(s), -1) for s in flat_ids], dtype=int)
        invalid_mask = pos_array == -1
        records = {
            'lat': np.repeat(lat_vals, W).tolist(),
            'lon': np.tile(lon_vals, H).tolist(),
            'soil_id': flat_ids.tolist(),
        }
        for var in desired:
            da = ds[var]
            if da.ndim == 1 and ('soil_id' in da.dims):
                vals = da.values
                mapped = np.where(invalid_mask, np.nan, vals[pos_array]).tolist()
                records[var.lower()] = mapped
        soil_df = pd.DataFrame(records)
    else:
        soil_ids = ds[id_var].values.squeeze()
        if soil_ids.ndim != 2:
            raise ValueError('Expected 2D soil id grid.')
        soil_grid = soil_ids
        H, W = soil_grid.shape

    # Adjust stride
    H, W = soil_grid.shape
    stride = max(1, args.stride)
    if not integrated_mode and args.avoid_huge and args.sample_points is None and stride == 1 and H*W > 5_000_000:
        stride = int(np.ceil(np.sqrt(H*W / 5_000_000)))
        print(f'[INFO] Auto stride set to {stride} to reduce size.')

    if integrated_mode and soil_df is not None:
        # Downsampling / sampling for integrated mode
        if not (args.sample_points is None and stride == 1 and args.sample_frac is None and args.max_rows is None):
            mask = np.zeros_like(soil_grid, dtype=bool)
            mask[::stride, ::stride] = True
            mask_flat = mask.reshape(-1)
            soil_df = soil_df.loc[mask_flat]
            if args.sample_frac is not None and 0 < args.sample_frac < 1:
                soil_df = soil_df.sample(frac=args.sample_frac, random_state=args.seed)
            if args.max_rows is not None:
                soil_df = soil_df.head(args.max_rows)
            if args.sample_points is not None and args.sample_points.exists():
                point_mode = True
                pts = pd.read_parquet(args.sample_points) if args.sample_points.suffix.lower()=='.parquet' else pd.read_csv(args.sample_points)
                cols_lower = {c.lower(): c for c in pts.columns}
                lat_col = cols_lower.get('lat') or cols_lower.get('latitude')
                lon_col = cols_lower.get('lon') or cols_lower.get('longitude')
                if not lat_col or not lon_col:
                    raise ValueError('Point file must contain lat/lon or latitude/longitude columns.')
                pts_lat = pd.to_numeric(pts[lat_col], errors='coerce').to_numpy()
                pts_lon = pd.to_numeric(pts[lon_col], errors='coerce').to_numpy()
                def nearest(arr, v):
                    return int(np.argmin(np.abs(arr - v)))
                y_idx = np.array([nearest(lat_vals, v) for v in pts_lat])
                x_idx = np.array([nearest(lon_vals, v) for v in pts_lon])
                pix_idx = y_idx * W + x_idx
                soil_df = soil_df.iloc[pix_idx]
                soil_df['lat'] = pts_lat
                soil_df['lon'] = pts_lon
    elif not integrated_mode:
        # Original mapping path
        soil_sub = soil_grid[::stride, ::stride]
        lat_sub = lat_vals[::stride]
        lon_sub = lon_vals[::stride]
        point_mode = args.sample_points is not None and args.sample_points.exists()
        if point_mode:
            pts = pd.read_parquet(args.sample_points) if args.sample_points.suffix.lower()=='.parquet' else pd.read_csv(args.sample_points)
            cols_lower = {c.lower(): c for c in pts.columns}
            lat_col = cols_lower.get('lat') or cols_lower.get('latitude')
            lon_col = cols_lower.get('lon') or cols_lower.get('longitude')
            if not lat_col or not lon_col:
                raise ValueError('Point file must contain lat/lon or latitude/longitude columns.')
            pts_lat = pd.to_numeric(pts[lat_col], errors='coerce').to_numpy()
            pts_lon = pd.to_numeric(pts[lon_col], errors='coerce').to_numpy()
            def nearest(arr, v):
                return int(np.argmin(np.abs(arr - v)))
            y_idx = np.array([nearest(lat_vals, v) for v in pts_lat])
            x_idx = np.array([nearest(lon_vals, v) for v in pts_lon])
            sampled_ids = soil_grid[y_idx, x_idx]
            lat_out = pts_lat
            lon_out = pts_lon
        else:
            flat = soil_sub.reshape(-1)
            lat_grid = np.repeat(lat_sub, len(lon_sub))
            lon_grid = np.tile(lon_sub, len(lat_sub))
            rng = np.random.default_rng(args.seed)
            if args.sample_frac is not None and 0 < args.sample_frac < 1:
                n = int(len(flat) * args.sample_frac)
                idx_sel = rng.choice(len(flat), size=n, replace=False)
            else:
                idx_sel = np.arange(len(flat))
            if args.max_rows is not None:
                idx_sel = idx_sel[:args.max_rows]
            sampled_ids = flat[idx_sel]
            lat_out = lat_grid[idx_sel]
            lon_out = lon_grid[idx_sel]
        records = {'lat': lat_out.tolist() if isinstance(lat_out, np.ndarray) else list(lat_out),
                   'lon': lon_out.tolist() if isinstance(lon_out, np.ndarray) else list(lon_out),
                   'soil_id': sampled_ids.tolist() if isinstance(sampled_ids, np.ndarray) else list(sampled_ids)}
        for col in FEATURE_COLS:
            if col == 'HWSD2_SMU_ID' or col not in df.columns:
                continue
            values = []
            for sid in records['soil_id']:
                rec = mapping.get(int(sid), {})
                values.append(rec.get(col, np.nan))
            records[col.lower()] = values
        soil_df = pd.DataFrame(records)

    # Clip percentage columns to valid bounds
    pct_cols = [c for c in soil_df.columns if c.lower() in ['coarse','sand','silt','clay','bsat','alum_sat','esp','gypsum']]
    for c in pct_cols:
        soil_df[c] = pd.to_numeric(soil_df[c], errors='coerce')
        soil_df[c] = soil_df[c].clip(lower=0, upper=100)
    if all(x in soil_df.columns for x in ['coarse','sand','silt','clay']):
        total_texture = soil_df['coarse'] + soil_df['sand'] + soil_df['silt'] + soil_df['clay']
        soil_df['texture_total_pct'] = total_texture
        # Flag rows where texture total deviates strongly from expected (~ >120 or <60)
        soil_df['texture_total_flag'] = ((total_texture > 120) | (total_texture < 60)).astype(int)

    # Derived features
    if args.derived:
        soil_df = add_derived_features(soil_df)

    # One-hot texture
    if args.one_hot_texture:
        # Use lower-case source columns if present; avoid duplicate uppercase copies
        tex_sources = []
        for tcol in ['texture_usda','texture_soter']:
            if tcol in soil_df.columns:
                tex_sources.append(tcol)
        if tex_sources:
            # Build one-hot from each source
            for src in tex_sources:
                dummies = pd.get_dummies(soil_df[src].astype(str), prefix=src)
                soil_df = pd.concat([soil_df, dummies], axis=1)

    # IQR capping
    cap_report = {}
    if args.cap_iqr:
        skip_cap = {'lat','lon','soil_id','texture_usda','texture_soter','TEXTURE_USDA','TEXTURE_SOTER'}
        for col in soil_df.columns:
            if col in skip_cap:
                continue
            if pd.api.types.is_numeric_dtype(soil_df[col]):
                arr = soil_df[col].to_numpy(dtype=float)
                capped, ncap, lo, hi = iqr_cap(arr)
                if ncap:
                    soil_df[col] = capped
                    cap_report[col] = {"capped": ncap, "lo": lo, "hi": hi}

    # Final numeric impute (median)
    for col in soil_df.columns:
        if pd.api.types.is_numeric_dtype(soil_df[col]) and soil_df[col].isna().any():
            med = soil_df[col].median()
            soil_df[col] = soil_df[col].fillna(med)

    # Output
    args.out.mkdir(parents=True, exist_ok=True)
    pq = args.out / 'soil_features.parquet'
    csv = args.out / 'soil_features.csv'
    try:
        soil_df.to_parquet(pq, index=False, compression=args.parquet_compression)
    except Exception as e:
        print(f"[WARN] Parquet write failed: {e}")
    soil_df.to_csv(csv, index=False)

    if args.eda:
        eda_summary(soil_df, args.out / 'soil_features_summary.json')
        if cap_report:
            (args.out / 'soil_features_capping.json').write_text(json.dumps(cap_report, indent=2), encoding='utf-8')

    print(json.dumps({
        'status':'ok',
        'rows': len(soil_df),
        'columns': len(soil_df.columns),
        'parquet': str(pq),
        'csv': str(csv),
        'point_mode': point_mode,
        'stride': stride,
        'sample_frac': args.sample_frac,
        'max_rows': args.max_rows,
        'derived': args.derived,
        'one_hot_texture': args.one_hot_texture,
        'capped_cols': list(cap_report.keys()),
        'integrated_mode': integrated_mode,
    }, indent=2))

if __name__ == '__main__':
    main()
