# scripts/soil_pipeline.py
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set

import geopandas as gpd
import rasterio
from rasterio.crs import CRS

from soil_cleaning import CleanerConfig, clean_soil_file
from scripts.config import get_base


BASE = get_base()
AOI_PATH = BASE / "shapefiles/algeria_tunisia.shp"
META_DIR = BASE / "processed" / "_metadata"
META_DIR.mkdir(parents=True, exist_ok=True)
DICT_CSV = META_DIR / "data_dictionary.csv"
LOG_MD = META_DIR / "log.md"

# Where your original per-depth rasters live
DEFAULT_IN_ROOT = BASE / "SOIL" / "processed" / "rasters_all_aoi"
# Where to write cleaned rasters
DEFAULT_OUT_ROOT = BASE / "SOIL" / "processed" / "rasters_clean"


def log_line(text: str):
    LOG_MD.parent.mkdir(parents=True, exist_ok=True)
    with LOG_MD.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")


def append_dictionary_row(out_path: Path, layer: str, layer_type: str, cfg: CleanerConfig):
    # probe raster
    try:
        with rasterio.open(out_path) as src:
            crs = str(src.crs) if isinstance(src.crs, CRS) else str(src.crs)
            res = f"{src.res[0]} x {src.res[1]}"
            nodata = src.nodata
            transform = str(src.transform)
    except Exception:
        crs = res = transform = ""
        nodata = ""

    actions = []
    if cfg.template_path:
        actions.append("reproject_match template")
    actions.append("clip AOI (drop)")
    if layer_type == "continuous":
        actions.extend([
            "sentinels→NaN",
            "valid-range clamp",
            f"winsorize {cfg.winsor_pcts}" if cfg.winsor_pcts else "no winsorize",
            f"median {cfg.median_kernel}px" if cfg.median_kernel else "no median",
            f"interp≤{cfg.interp_limit}px gaps",
            "resampling=bilinear",
        ])
    else:
        actions.append("resampling=nearest")

    DICT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not DICT_CSV.exists()
    with DICT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["layer", "path", "type", "CRS", "res", "nodata", "transform", "source", "date", "notes", "processing"])
        w.writerow([
            layer,
            out_path.as_posix(),
            layer_type,
            crs,
            res,
            nodata,
            transform,
            "",  # source unknown at this stage
            datetime.now().isoformat(timespec="seconds"),
            "",
            "; ".join(actions),
        ])


def discover_depth_dirs(in_root: Path, depths: Iterable[str]) -> Iterable[Path]:
    for d in depths:
        p = in_root / f"rasters_{d.lower()}"
        if p.is_dir():
            yield p


def parse_categorical_set(arg: str | None) -> Set[str]:
    if not arg:
        return set()
    # stems, comma separated
    return {s.strip().upper() for s in arg.split(",") if s.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="Aggressive cleaning for HWSD2 soil rasters (AOI crop, ranges, winsorize, de-speckle, COG)."
    )
    ap.add_argument("--aoi", default=AOI_PATH.as_posix(), help="AOI shapefile/GeoPackage")
    ap.add_argument("--in-root", default=DEFAULT_IN_ROOT.as_posix(), help="Root with rasters_* depth folders")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT.as_posix(), help="Output root for cleaned rasters")
    ap.add_argument("--depths", default="D1,D2,D3,D4,D5,D6,D7", help="Comma list of depths to process")
    ap.add_argument("--template", default=None, help="Optional template .tif to align all outputs")
    ap.add_argument("--categorical", default="", help="Comma list of stems that are categorical (nearest resample)")
    ap.add_argument("--winsor", default="1,99", help="Percentiles P1,P2 or 'none'")
    ap.add_argument("--median-kernel", type=int, default=3, help="Odd integer ≥3 (0 to disable)")
    ap.add_argument("--interp-limit", type=int, default=2, help="Fill NaN gaps up to N pixels")
    ap.add_argument("--sentinels", default="-9,-7,-9999", help="Comma list of sentinel values to treat as NaN")
    ap.add_argument("--ranges-json", default=None, help="Optional JSON with valid_ranges overrides")
    args = ap.parse_args()

    aoi = gpd.read_file(args.aoi)

    # Build CleanerConfig
    cfg = CleanerConfig()
    cfg.template_path = args.template
    cfg.interp_limit = int(args.interp_limit)
    cfg.median_kernel = int(args.median_kernel) if int(args.median_kernel) > 0 else None
    if args.winsor.lower() == "none":
        cfg.winsor_pcts = None
    else:
        p1, p2 = [float(x) for x in args.winsor.split(",")]
        cfg.winsor_pcts = (p1, p2)
    cfg.sentinels = tuple(float(x) for x in args.sentinels.split(","))
    if args.ranges_json:
        # merge overrides
        user = CleanerConfig.from_json(args.ranges_json)
        if user.valid_ranges:
            cfg.valid_ranges.update(user.valid_ranges)

    depths = [d.strip() for d in args.depths.split(",") if d.strip()]
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    categorical_stems = parse_categorical_set(args.categorical)

    count = 0
    for ddir in discover_depth_dirs(in_root, depths):
        rel = ddir.name  # e.g., rasters_d1
        out_dir = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for tif in sorted(ddir.glob("*.tif")):
            stem = tif.stem.upper()
            is_cat = stem in categorical_stems
            out_tif = out_dir / f"{tif.stem}_clean.tif"

            log_line(f"[SOIL] cleaning {tif.name} → {out_tif.name} "
                     f"({'categorical' if is_cat else 'continuous'})")
            try:
                clean_soil_file(tif.as_posix(), out_tif.as_posix(), aoi, cfg, is_categorical=is_cat)
                append_dictionary_row(out_tif, layer=tif.stem, layer_type=("categorical" if is_cat else "continuous"), cfg=cfg)
                log_line(f"[SOIL] cleaned OK {out_tif.name}")
                count += 1
            except Exception as e:
                log_line(f"[SOIL][ERROR] {tif.name}: {e}")

    print(f"Cleaned rasters: {count} → {out_root.as_posix()}")


if __name__ == "__main__":
    main()
