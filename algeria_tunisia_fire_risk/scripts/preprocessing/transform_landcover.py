#!/usr/bin/env python3
"""Transform land cover polygons into a cleaned feature table.

Steps:
1. Read processed `landcover_clipped.shp` (columns: ID, GRIDCODE, LCCCODE).
2. Load GlobCover legend (Value->Label) and LCCS Africa legend (MapCode/LCCCode/LCCLabel).
3. Choose most granular coding: prefer LCCCode (when multi-code list present, keep first token) else GRIDCODE value.
4. Produce cleaned table with columns:
   - id
   - latitude, longitude (centroid, WGS84)
   - gridcode
   - lcccode_raw
   - lcccode_primary
   - lcc_label (from legend maps when possible)
   - class_code (numeric) (fallback to gridcode if lcc absent)
   - area_m2 (geometry area in EPSG:3857) [optional retained for QA, flagged removable]
5. Derive one-hot columns for top K classes (default 25) and drop rarely used codes below min_count threshold.
6. Export parquet and csv under DATA_CLEANED/processed/landcover_features.(parquet|csv).
7. Write summary JSON with counts and mapping coverage.

Requires geopandas; if unavailable, exits gracefully.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import math
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / 'LANDCOVER' / 'processed'
LEGEND_DIR = BASE / 'LANDCOVER' / 'geonetwork_landcover_DZA_gc_adg'
OUT_DIR = BASE / 'DATA_CLEANED' / 'processed'

SHAPE = PROC / 'landcover_clipped.shp'
LEGEND1 = LEGEND_DIR / 'globcover_legend.xls'
LEGEND2 = LEGEND_DIR / 'globcover_LCCS_legend_africa.xls'

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
except Exception:
    gpd = None
    Polygon = MultiPolygon = None

try:
    import shapefile  # pyshp
except Exception:
    shapefile = None

import argparse

RE_SPLIT = re.compile(r"\s*//\s*")

def load_legends():
    legend_value_label = {}
    if LEGEND1.exists():
        try:
            df = pd.read_excel(LEGEND1)
            if {'Value','Label'}.issubset(df.columns):
                legend_value_label = {int(v): str(l) for v,l in zip(df['Value'], df['Label']) if pd.notna(v) and pd.notna(l)}
        except Exception:
            pass
    legend_lcc = {}
    mapcode_label = {}
    if LEGEND2.exists():
        try:
            df = pd.read_excel(LEGEND2)
            # LCCCode may contain list; MapCode numeric links to class
            if 'LCCCode' in df.columns:
                for _,row in df.iterrows():
                    code = row.get('LCCCode')
                    label = row.get('LCCLabel') or row.get('LCCOwnLabel') or row.get('LC')
                    mapcode = row.get('MapCode')
                    if pd.notna(code):
                        primary = str(code).split()[0]
                        legend_lcc[str(code)] = str(label) if pd.notna(label) else None
                        # also split composite codes
                        for token in RE_SPLIT.split(str(code)):
                            token = token.strip()
                            if token:
                                legend_lcc.setdefault(token, legend_lcc[str(code)])
                    if pd.notna(mapcode):
                        try:
                            mapcode_label[int(mapcode)] = str(label) if pd.notna(label) else None
                        except Exception:
                            pass
        except Exception:
            pass
    return legend_value_label, legend_lcc, mapcode_label

def choose_primary_lcc(raw: str) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    # Split composite code list
    parts = RE_SPLIT.split(raw)
    for p in parts:
        t = p.strip()
        if t:
            return t
    return None

def main():
    ap = argparse.ArgumentParser(description='Transform landcover shapefile to feature table')
    ap.add_argument('--top-k', type=int, default=25, help='Top K classes to one-hot encode')
    ap.add_argument('--min-count', type=int, default=50, help='Minimum occurrences to retain class in one-hot encoding')
    ap.add_argument('--keep-area', action='store_true', help='Keep area_m2 column (otherwise drop)')
    ap.add_argument('--sample-frac', type=float, default=None, help='Optional fraction sampling for performance')
    args = ap.parse_args()

    if not SHAPE.exists():
        print(json.dumps({'status':'missing_shapefile','path':SHAPE.as_posix()}))
        return
    if gpd is None:
        from simpledbf import Dbf5
        from math import nan
        dbf_path = SHAPE.with_suffix('.dbf')
        cent_lats = None; cent_lons = None
        if shapefile is not None and SHAPE.exists():
            try:
                sf = shapefile.Reader(SHAPE.as_posix())
                lats=[]; lons=[]
                for shp in sf.shapes():
                    pts = shp.points
                    if pts:
                        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                        lons.append(sum(xs)/len(xs)); lats.append(sum(ys)/len(ys))
                    else:
                        lons.append(nan); lats.append(nan)
                cent_lats=lats; cent_lons=lons
            except Exception as e:
                print(json.dumps({'status':'pyshp_centroid_failed','error':str(e)}))
        if not dbf_path.exists():
            print(json.dumps({'status':'missing_dbf','path':dbf_path.as_posix()})); return
        dbf = Dbf5(dbf_path.as_posix())
        raw = dbf.to_dataframe()
        legend_value_label, legend_lcc, mapcode_label = load_legends()
        df = pd.DataFrame({'id': raw.get('ID', range(len(raw))), 'gridcode': raw.get('GRIDCODE'), 'lcccode_raw': raw.get('LCCCODE')})
        df['lcccode_primary'] = df['lcccode_raw'].apply(choose_primary_lcc)
        if cent_lats is not None and len(cent_lats)==len(df):
            df['latitude']=cent_lats; df['longitude']=cent_lons
        else:
            df['latitude']=nan; df['longitude']=nan
        if args.keep_area: df['area_m2']=nan
        def map_label(row):
            primary=row['lcccode_primary']
            if primary and primary in legend_lcc: return legend_lcc.get(primary)
            gc=row['gridcode']
            if pd.notna(gc):
                try:
                    gc_int=int(gc)
                    if gc_int in legend_value_label: return legend_value_label[gc_int]
                except Exception: pass
            return None
        df['lcc_label']=df.apply(map_label,axis=1)
        def class_code(row):
            if pd.notna(row['gridcode']):
                try: return int(row['gridcode'])
                except Exception: pass
            lab=row['lcc_label']
            if lab: return abs(hash(lab)) % 100000
            prim=row['lcccode_primary']
            if prim:
                digits=re.findall(r'\d+', prim)
                if digits: return int(digits[0])
            return None
        df['class_code']=df.apply(class_code,axis=1)
        base_cols=['id','latitude','longitude','class_code','lcc_label','gridcode','lcccode_primary']
        if args.keep_area: base_cols.append('area_m2')
        df=df[base_cols]
        freq=df['class_code'].value_counts(); top_classes=freq.head(args.top_k)
        selected=top_classes[top_classes>=args.min_count].index.tolist(); one_hot_cols=[]
        for c in selected:
            coln=f'class_{c}'; df[coln]=(df['class_code']==c).astype('int8'); one_hot_cols.append(coln)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pq=OUT_DIR/'landcover_features.parquet'; csv=OUT_DIR/'landcover_features.csv'
        try: df.to_parquet(pq,index=False)
        except Exception as e: print('[WARN] parquet write failed', e)
        df.to_csv(csv,index=False)
        summary={'status':'ok','rows':int(len(df)),'columns':df.columns.tolist(),'top_classes':{str(k):int(v) for k,v in top_classes.to_dict().items()},'one_hot_cols':one_hot_cols,'legend_value_label_count':len(legend_value_label),'legend_lcc_count':len(legend_lcc),'mapcode_label_count':len(mapcode_label),'output_parquet':pq.as_posix(),'output_csv':csv.as_posix(),'centroids_method':'pyshp' if cent_lats is not None else 'none'}
        (OUT_DIR/'landcover_features_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return
    # Proceed with geopandas path
    gdf = gpd.read_file(SHAPE)
    orig_crs = gdf.crs
    if orig_crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        try:
            gdf = gdf.to_crs('EPSG:4326')
        except Exception:
            pass
    # Subsample for speed if requested
    if args.sample_frac and 0 < args.sample_frac < 1:
        gdf = gdf.sample(frac=args.sample_frac, random_state=0).reset_index(drop=True)

    legend_value_label, legend_lcc, mapcode_label = load_legends()

    # Build base dataframe
    df = pd.DataFrame()
    df['id'] = gdf.get('ID', range(len(gdf)))
    df['gridcode'] = gdf.get('GRIDCODE')
    df['lcccode_raw'] = gdf.get('LCCCODE')
    df['lcccode_primary'] = df['lcccode_raw'].apply(choose_primary_lcc)

    # Centroids for spatial join (use projected for accurate area then revert to WGS84)
    try:
        gdf_proj = gdf.to_crs('EPSG:3857')
    except Exception:
        gdf_proj = gdf
    cents = gdf_proj.geometry.centroid
    try:
        cents_wgs = gpd.GeoSeries(cents, crs=gdf_proj.crs).to_crs('EPSG:4326')
    except Exception:
        cents_wgs = cents
    df['latitude'] = cents_wgs.y
    df['longitude'] = cents_wgs.x

    # Area (m2) from projected geometry
    df['area_m2'] = gdf_proj.geometry.area

    # Map labels
    def map_label(row):
        # Prefer LCC primary label then grid legend
        primary = row['lcccode_primary']
        if primary and primary in legend_lcc:
            return legend_lcc.get(primary)
        gc = row['gridcode']
        if pd.notna(gc) and int(gc) in legend_value_label:
            return legend_value_label[int(gc)]
        return None
    df['lcc_label'] = df.apply(map_label, axis=1)

    # Unified class_code: try mapcode by reverse lookup of label; else gridcode
    # Simple approach: if lcc_label matches a legend value label exactly, keep gridcode; else fallback to hash of label
    def class_code(row):
        if pd.notna(row['gridcode']):
            try:
                return int(row['gridcode'])
            except Exception:
                pass
        lab = row['lcc_label']
        if lab:
            return abs(hash(lab)) % 100000
        prim = row['lcccode_primary']
        if prim:
            # derive numeric from digits
            digits = re.findall(r'\d+', prim)
            if digits:
                return int(digits[0])
        return None
    df['class_code'] = df.apply(class_code, axis=1)

    # Drop useless columns (keep id, coords, class_code, label) optionally area
    base_cols = ['id','latitude','longitude','class_code','lcc_label','gridcode','lcccode_primary']
    if args.keep_area:
        keep_cols = base_cols + ['area_m2']
    else:
        keep_cols = base_cols
        df = df.drop(columns=['area_m2'])
    df = df[keep_cols]

    # Class frequency
    freq = df['class_code'].value_counts()
    top_classes = freq.head(args.top_k)
    selected = top_classes[top_classes >= args.min_count].index.tolist()
    one_hot_cols = []
    for c in selected:
        coln = f'class_{c}'
        df[coln] = (df['class_code'] == c).astype('int8')
        one_hot_cols.append(coln)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq = OUT_DIR / 'landcover_features.parquet'
    csv = OUT_DIR / 'landcover_features.csv'
    try:
        df.to_parquet(pq, index=False)
    except Exception as e:
        print('[WARN] parquet write failed', e)
    df.to_csv(csv, index=False)

    summary = {
        'status':'ok',
        'rows': int(len(df)),
        'columns': df.columns.tolist(),
        'top_classes': {str(k): int(v) for k,v in top_classes.to_dict().items()},
        'one_hot_cols': one_hot_cols,
        'legend_value_label_count': len(legend_value_label),
        'legend_lcc_count': len(legend_lcc),
        'mapcode_label_count': len(mapcode_label),
        'output_parquet': pq.as_posix(),
        'output_csv': csv.as_posix()
    }
    (OUT_DIR / 'landcover_features_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

    # Post-fix: if latitude all NaN and geopandas available, recompute centroids
    if gpd is not None and df['latitude'].isna().all():
        try:
            gdf2 = gpd.read_file(SHAPE).to_crs('EPSG:4326')
            cents2 = gdf2.geometry.centroid
            df['latitude'] = cents2.y
            df['longitude'] = cents2.x
        except Exception as e:
            print(json.dumps({'status':'centroid_recompute_failed','error':str(e)}))
    if gpd is None and shapefile is not None:
        # pyshp centroid injection
        try:
            sf = shapefile.Reader(SHAPE.as_posix())
            lats = []
            lons = []
            for shp in sf.shapes():
                pts = shp.points
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    lons.append(sum(xs)/len(xs)); lats.append(sum(ys)/len(ys))
                else:
                    lons.append(math.nan); lats.append(math.nan)
            # Ensure length matches
            if len(lats) == len(df):
                df['latitude'] = lats
                df['longitude'] = lons
        except Exception as e:
            print(json.dumps({'status':'pyshp_centroid_failed','error':str(e)}))
    if gpd is not None and df['latitude'].isna().all() and shapefile is not None:
        try:
            sf = shapefile.Reader(SHAPE.as_posix())
            lats = []
            lons = []
            for shp in sf.shapes():
                pts = shp.points
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    lons.append(sum(xs)/len(xs)); lats.append(sum(ys)/len(ys))
                else:
                    lons.append(math.nan); lats.append(math.nan)
            if len(lats) == len(df):
                df['latitude'] = lats
                df['longitude'] = lons
                print(json.dumps({'status':'centroid_recompute_pyshp','rows':len(df)}))
        except Exception as e:
            print(json.dumps({'status':'centroid_pyshp_failed','error':str(e)}))
    if gpd is None:
        return  # Already handled fallback

if __name__ == '__main__':
    main()
