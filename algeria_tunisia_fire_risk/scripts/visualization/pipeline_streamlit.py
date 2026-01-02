#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧰 Algeria–Tunisia Geospatial Pipeline — Streamlit UI
Run processing modules (Climate, Elevation, Landcover, Soil CSV, Fire) and
inspect outputs, stats, and processing logs.

Requirements: GDAL in PATH (gdal_translate), rioxarray, geopandas, rasterio.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Repo base and module import ---
# Dynamically resolve BASE from the script's location
BASE = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "scripts"
META_ROOT = BASE / "processed" / "_metadata"
CLIMATE_OUT = BASE / "CLIMATE" / "processed"
ELEV_OUT = BASE / "ELEVATION" / "be15_grd" / "processed"
LANDCOVER_OUT = BASE / "LANDCOVER" / "processed"
FIRE_DIR = BASE / "FIRE"

# Add BASE to path so 'from scripts.config import ...' works inside geo_pipeline
if BASE.as_posix() not in sys.path:
    sys.path.insert(0, BASE.as_posix())

import importlib  # noqa: E402

try:
    gp = importlib.import_module("scripts.geo_pipeline")
except Exception as e:
    st.error(f"Failed to import geo_pipeline: {e}")
    st.stop()

st.set_page_config(page_title="Algeria–Tunisia Pipeline", layout="wide")
st.title("⚙️ Algeria–Tunisia Geospatial Pipeline")
st.caption("Run processing modules and review outputs/metadata.")

# --- Environment checks ---
col_env1, col_env2 = st.columns(2)
with col_env1:
    st.write("GDAL available:", bool(shutil.which("gdal_translate")))
with col_env2:
    st.write("Workspace:", BASE.as_posix())

st.divider()

# --- Controls ---
st.sidebar.header("Run Modules")
run_climate = st.sidebar.checkbox("Climate", value=False)
run_elev = st.sidebar.checkbox("Elevation", value=False)
run_land = st.sidebar.checkbox("Landcover", value=False)
run_soil = st.sidebar.checkbox("Soil CSV standardization", value=False)
run_fire = st.sidebar.checkbox("Fire (VIIRS 2024)", value=False)

run_selected = st.sidebar.button("▶️ Run Selected")
run_all = st.sidebar.button("🚀 Run ALL")

# --- Execution ---
if run_selected or run_all:
    sel = dict(
        climate=run_all or run_climate,
        elevation=run_all or run_elev,
        landcover=run_all or run_land,
        soil=run_all or run_soil,
        fire=run_all or run_fire,
    )

    with st.status("Processing…", expanded=True) as status:
        if sel["climate"]:
            st.write("🌦️ Running CLIMATE…")
            try:
                gp.process_climate()
                st.success("CLIMATE done")
            except Exception as e:
                st.error(f"CLIMATE failed: {e}")
        if sel["elevation"]:
            st.write("🏔️ Running ELEVATION…")
            try:
                gp.process_elevation()
                st.success("ELEVATION done")
            except Exception as e:
                st.error(f"ELEVATION failed: {e}")
        if sel["landcover"]:
            st.write("🪴 Running LANDCOVER…")
            try:
                gp.process_landcover()
                st.success("LANDCOVER done")
            except Exception as e:
                st.error(f"LANDCOVER failed: {e}")
        if sel["soil"]:
            st.write("🧹 Running SOIL CSV standardization…")
            try:
                gp.process_soil_clean_csvs()
                st.success("SOIL CSV done")
            except Exception as e:
                st.error(f"SOIL CSV failed: {e}")
        if sel["fire"]:
            st.write("🔥 Running FIRE monthly density…")
            try:
                gp.process_fire_2024()
                st.success("FIRE done")
            except Exception as e:
                st.error(f"FIRE failed: {e}")
        status.update(label="Processing complete", state="complete")

st.divider()

# --- Outputs / Metadata ---
st.header("📦 Outputs & Metadata")

colA, colB = st.columns([2, 1])
with colA:
    st.subheader("Data Dictionary")
    dd = META_ROOT / "data_dictionary.csv"
    if dd.exists():
        try:
            df = pd.read_csv(dd)
            st.dataframe(df.tail(200), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read data dictionary: {e}")
    else:
        st.info("No data_dictionary.csv yet.")

with colB:
    st.subheader("Processing Log (tail)")
    log = META_ROOT / "log.md"
    if log.exists():
        try:
            txt = log.read_text(encoding="utf-8").strip().splitlines()
            tail = "\n".join(txt[-30:])
            st.code(tail)
        except Exception as e:
            st.warning(f"Could not read log: {e}")
    else:
        st.info("No log.md yet.")

st.subheader("Browse Processed Files")

# --- Previews / AOI ---
try:
    AOI_GDF = gp.load_aoi()
except Exception:
    AOI_GDF = None

def _ensure_dataarray(raster):
    """Normalize rioxarray outputs to DataArray for safe previewing."""
    import xarray as xr
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

def _lookup_type(meta_df: pd.DataFrame | None, path: Path) -> str:
    if meta_df is None or meta_df.empty:
        # infer from filename
        n = path.name.lower()
        if "viirs_count" in n or "count" in n:
            return "count"
        if "landcover" in n or "class" in n:
            return "categorical"
        return "continuous"
    hits = meta_df[meta_df["path"] == path.as_posix()]
    if not hits.empty and "type" in hits.columns:
        return str(hits.iloc[0]["type"]).lower()
    return "continuous"

def _render_preview(ras: Path, lyr_type: str):
    try:
        da = _ensure_dataarray(rxr.open_rasterio(ras.as_posix(), masked=True)).squeeze()
        arr = np.asarray(da.values, dtype=float)
        nodata = da.rio.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            st.caption("No finite values to preview.")
            return
        if lyr_type in ("categorical", "count") or str(da.dtype).startswith("int"):
            cmap = "tab20"
            vmin = vmax = None
        else:
            cmap = "viridis"
            vmin, vmax = np.nanpercentile(finite, [2, 98])
        fig, ax = plt.subplots(figsize=(6, 4))
        if vmin is None or vmax is None:
            im = ax.imshow(arr, cmap=cmap)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=float(vmin), vmax=float(vmax))
        if AOI_GDF is not None:
            AOI_GDF.boundary.plot(ax=ax, color="red", linewidth=0.7)
        ax.set_title(ras.name)
        if lyr_type not in ("categorical", "count"):
            fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.caption(f"Preview failed: {e}")

def list_files(folder: Path, pattern: str = "*.tif", meta_df: pd.DataFrame | None = None, show_previews: bool = False):
    if not folder.exists():
        st.info(f"Folder not found: {folder}")
        return
    files = sorted(folder.glob(pattern))
    if not files:
        st.info("No matching files.")
        return
    for p in files:
        cols = st.columns([4, 2, 1])
        with cols[0]:
            st.write(p.name)
            stats_path = p.with_suffix(p.suffix + ".stats.json")
            if stats_path.exists():
                try:
                    st.json(json.loads(stats_path.read_text()))
                except Exception:
                    st.caption("stats.json present (could not parse)")
            if show_previews:
                lyr_type = _lookup_type(meta_df, p)
                _render_preview(p, lyr_type)
        with cols[1]:
            hist_png = p.with_suffix(".hist.png")
            if hist_png.exists():
                st.image(hist_png.as_posix(), caption="Histogram", use_column_width=True)
        with cols[2]:
            st.download_button("Download", data=p.read_bytes(), file_name=p.name)

dd_df = None
if (META_ROOT / "data_dictionary.csv").exists():
    try:
        dd_df = pd.read_csv(META_ROOT / "data_dictionary.csv")
    except Exception:
        dd_df = None

with st.expander("CLIMATE/processed"):
    show = st.checkbox("Show previews (CLIMATE)", value=False, key="clim_prev")
    list_files(CLIMATE_OUT, meta_df=dd_df, show_previews=show)
with st.expander("ELEVATION/processed"):
    show = st.checkbox("Show previews (ELEVATION)", value=False, key="elev_prev")
    list_files(ELEV_OUT, meta_df=dd_df, show_previews=show)
with st.expander("LANDCOVER/processed"):
    show = st.checkbox("Show previews (LANDCOVER)", value=False, key="land_prev")
    list_files(LANDCOVER_OUT, meta_df=dd_df, show_previews=show)
with st.expander("FIRE"):
    show = st.checkbox("Show previews (FIRE)", value=False, key="fire_prev")
    list_files(FIRE_DIR, pattern="viirs_count_*.tif", meta_df=dd_df, show_previews=show)

st.caption("Tip: Use the sidebar to select modules and rerun as needed. All outputs are saved as COGs with sidecar statistics and histograms.")
