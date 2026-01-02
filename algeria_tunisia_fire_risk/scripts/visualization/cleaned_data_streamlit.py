#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cleaned Data Comparison App

Compare original (source/processed) data with cleaned outputs under DATA_CLEANED.
Supports:
- Climate NetCDF: CLIMATE/processed/climate_2024.nc vs DATA_CLEANED/climate/climate_2024_clean.nc
- Soil NetCDF: SOIL/processed/soil_id_aoi.nc vs DATA_CLEANED/soil/soil_id_aoi_clean.nc
- Elevation raster: ELEVATION/.../processed/elevation_clipped.tif vs DATA_CLEANED/elevation/elevation_clipped_clean.tif
- Landcover vector: LANDCOVER/processed/landcover_clipped.shp vs DATA_CLEANED/landcover/landcover_clipped_clean.gpkg
- Fire CSVs: FIRE/viirs-jpss1_2024_*_type2.csv vs DATA_CLEANED/fire/*

Run:
  streamlit run scripts/cleaned_data_streamlit.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

BASE = Path("/home/swift/Desktop/DATA")
CLEAN = BASE / "DATA_CLEANED"
AOI_SHP = BASE / "shapefiles" / "algeria_tunisia.shp"

# ----------------------
# Helpers
# ----------------------

def _ensure_dataarray(raster) -> xr.DataArray:
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

@st.cache_data(show_spinner=False)
def load_aoi() -> Optional[gpd.GeoDataFrame]:
    if AOI_SHP.exists():
        aoi = gpd.read_file(AOI_SHP)
        if aoi.empty:
            return None
        aoi = aoi.to_crs("EPSG:4326")
        try:
            aoi["geometry"] = aoi.buffer(0)
        except Exception:
            pass
        return aoi
    return None

@st.cache_data(show_spinner=False)
def basic_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=float)
    valid = np.isfinite(a)
    if valid.sum() == 0:
        return {"count": 0, "min": np.nan, "mean": np.nan, "max": np.nan, "std": np.nan, "%nan": 100.0}
    v = a[valid]
    return {
        "count": int(v.size),
        "min": float(np.nanmin(v)),
        "mean": float(np.nanmean(v)),
        "max": float(np.nanmax(v)),
        "std": float(np.nanstd(v)),
        "%nan": float(100.0 * (a.size - v.size) / max(1, a.size)),
    }

@st.cache_data(show_spinner=False)
def open_nc(path: Path) -> Optional[xr.Dataset]:
    if not path.exists():
        return None
    try:
        return xr.open_dataset(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def open_tif(path: Path) -> Optional[xr.DataArray]:
    if not path.exists():
        return None
    try:
        raw = rxr.open_rasterio(path, masked=True)
        if isinstance(raw, list):
            raw = raw[0]
        if isinstance(raw, xr.Dataset):
            if len(raw.data_vars) == 1:
                raw = next(iter(raw.data_vars.values()))
            else:
                raw = raw.to_array().isel(variable=0)
        return raw.squeeze()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def open_vector(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
        return gdf
    except Exception:
        return None

# ----------------------
# Visualization helpers
# ----------------------

def show_raster_side_by_side(left: xr.DataArray, right: xr.DataArray, title_l: str, title_r: str, cmap: str = "viridis"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, da, title in zip(axes, [left, right], [title_l, title_r]):
        arr = np.asarray(da.values)
        nodata = getattr(da.rio, "nodata", None)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


def show_hist_overlay(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, bins: int = 60):
    va = a[np.isfinite(a)]
    vb = b[np.isfinite(b)]
    if va.size == 0 and vb.size == 0:
        st.info("No valid values to plot histograms.")
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    if va.size:
        ax.hist(va, bins=bins, alpha=0.5, label=label_a)
    if vb.size:
        ax.hist(vb, bins=bins, alpha=0.5, label=label_b)
    ax.legend()
    ax.set_title("Value distribution (overlay)")
    st.pyplot(fig)
    plt.close(fig)


# ----------------------
# UI
# ----------------------

st.set_page_config(page_title="Cleaned Data Comparison", layout="wide")

st.title("Cleaned Data Comparison")

section = st.sidebar.selectbox(
    "Dataset",
    [
        "Climate (NetCDF)",
        "Soil (NetCDF)",
        "Elevation (Raster)",
        "Landcover (Vector)",
        "Fire (CSV)",
    ],
)

st.sidebar.markdown("---")

# Paths
clim_before = BASE / "CLIMATE" / "processed" / "climate_2024.nc"
clim_after = CLEAN / "climate" / "climate_2024_clean.nc"
soil_before = BASE / "SOIL" / "processed" / "soil_id_aoi.nc"
soil_after = CLEAN / "soil" / "soil_id_aoi_clean.nc"
elev_before = BASE / "ELEVATION" / "be15_grd" / "be15_grd" / "processed" / "elevation_clipped.tif"
elev_after = CLEAN / "elevation" / "elevation_clipped_clean.tif"
lcov_before = BASE / "LANDCOVER" / "processed" / "landcover_clipped.shp"
lcov_after = CLEAN / "landcover" / "landcover_clipped_clean.gpkg"
fire_before = [
    BASE / "FIRE" / "viirs-jpss1_2024_Algeria_type2.csv",
    BASE / "FIRE" / "viirs-jpss1_2024_Tunisia_type2.csv",
]
fire_after = [
    CLEAN / "fire" / "viirs-jpss1_2024_Algeria_type2.csv",
    CLEAN / "fire" / "viirs-jpss1_2024_Tunisia_type2.csv",
]

aoi = load_aoi()

if section == "Climate (NetCDF)":
    ds_b = open_nc(clim_before)
    ds_a = open_nc(clim_after)
    if ds_b is None or ds_a is None:
        st.error("Missing climate NetCDF before/after. Ensure both files exist.")
    else:
        vars_b = list(ds_b.data_vars)
        var = st.selectbox("Variable", vars_b)
        has_time = "time" in ds_b[var].dims
        t_idx = 0
        if has_time:
            times = pd.to_datetime(ds_b["time"].values)
            t = st.selectbox("Time", list(times))
            t_idx = int(np.where(times == t)[0][0])
        da_b = ds_b[var].isel(time=t_idx) if has_time else ds_b[var]
        da_a = ds_a[var].isel(time=t_idx) if has_time else ds_a[var]
        st.subheader(f"{var} before vs after")
        show_raster_side_by_side(da_b, da_a, "Before", "After")
        st.subheader("Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.json(basic_stats(np.asarray(da_b.values)))
        with col2:
            st.json(basic_stats(np.asarray(da_a.values)))
        st.subheader("Distribution overlay")
        show_hist_overlay(np.asarray(da_b.values).ravel(), np.asarray(da_a.values).ravel(), "Before", "After")

elif section == "Soil (NetCDF)":
    ds_b = open_nc(soil_before)
    ds_a = open_nc(soil_after)
    if ds_b is None or ds_a is None:
        st.error("Missing soil NetCDF before/after.")
    else:
        # assume soil_id variable
        var = st.selectbox("Variable", list(ds_b.data_vars), index=0)
        da_b = ds_b[var]
        da_a = ds_a[var]
        st.subheader(f"{var} before vs after")
        show_raster_side_by_side(da_b, da_a, "Before", "After", cmap="tab20")
        st.subheader("Category counts")
        vb = pd.Series(np.asarray(da_b.values).ravel()).value_counts(dropna=True).head(20)
        va = pd.Series(np.asarray(da_a.values).ravel()).value_counts(dropna=True).head(20)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before top codes")
            st.dataframe(vb)
        with col2:
            st.write("After top codes")
            st.dataframe(va)

elif section == "Elevation (Raster)":
    da_b = open_tif(elev_before)
    da_a = open_tif(elev_after)
    if da_b is None or da_a is None:
        st.error("Missing elevation raster before/after.")
    else:
        st.subheader("Elevation before vs after")
        show_raster_side_by_side(da_b, da_a, "Before", "After", cmap="terrain")
        st.subheader("Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.json(basic_stats(np.asarray(da_b.values)))
        with col2:
            st.json(basic_stats(np.asarray(da_a.values)))
        st.subheader("Distribution overlay")
        show_hist_overlay(np.asarray(da_b.values).ravel(), np.asarray(da_a.values).ravel(), "Before", "After")

elif section == "Landcover (Vector)":
    gdf_b = open_vector(lcov_before)
    gdf_a = open_vector(lcov_after)
    if gdf_b is None or gdf_a is None:
        st.error("Missing landcover vector before/after.")
    else:
        st.subheader("Geometry health")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before")
            st.json({
                "rows": len(gdf_b),
                "missing_geom": int((gdf_b.geometry.isna() | gdf_b.geometry.is_empty).sum()),
                "crs": str(gdf_b.crs),
            })
        with col2:
            st.write("After")
            st.json({
                "rows": len(gdf_a),
                "missing_geom": int((gdf_a.geometry.isna() | gdf_a.geometry.is_empty).sum()),
                "crs": str(gdf_a.crs),
            })
        # attribute missingness comparison for top N columns
        st.subheader("Attribute missingness (top columns)")
        cols = [c for c in gdf_b.columns if c != gdf_b.geometry.name][:10]
        miss_b = {c: int(gdf_b[c].isna().sum()) for c in cols}
        miss_a = {c: int(gdf_a[c].isna().sum()) for c in cols if c in gdf_a.columns}
        dfm = pd.DataFrame({"before_missing": miss_b, "after_missing": miss_a}).fillna(0).astype(int)
        st.dataframe(dfm)

elif section == "Fire (CSV)":
    st.write("Comparing Algeria and Tunisia files (type 2)")
    pairs = list(zip(fire_before, fire_after))
    for before, after in pairs:
        st.markdown(f"### {before.name}")
        if not before.exists():
            st.error("Original file missing")
            continue
        dfb = pd.read_csv(before, low_memory=False)
        dfa = pd.read_csv(after, low_memory=False) if after.exists() else None
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before shape:", dfb.shape)
            st.dataframe(dfb.head(10))
        with col2:
            if dfa is not None:
                st.write("After shape:", dfa.shape)
                st.dataframe(dfa.head(10))
            else:
                st.warning("No cleaned counterpart found in DATA_CLEANED/fire/")
        # missingness
        st.write("Missing values per column (before)")
        st.dataframe(dfb.isna().sum())
        if dfa is not None:
            st.write("Missing values per column (after)")
            st.dataframe(dfa.isna().sum())

st.sidebar.info("Tip: use the sidebar to switch datasets and variables/time where applicable.")
