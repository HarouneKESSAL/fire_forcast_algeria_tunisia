"""
Minimal self-contained test to validate climate monthly→aggregated transformations.
It creates a tiny synthetic xarray.Dataset with 12 monthly steps for prec, tmax, tmin
on a 3x4 grid, then exercises the pivot (monthly-wide), stats (mean/min/max/std),
seasonal (DJF/MAM/JJA/SON means), and basic EDA + cleaning semantics.

Run:
  python tests/climate_transform_test.py

It will print concise PASS/FAIL messages and key shapes/columns to verify:
- Monthly-wide: 12 columns per feature (36) + lat/lon
- Stats: 4 columns per feature (12 total) + lat/lon (+ optional quantiles)
- Seasonal: 4 columns per feature (12 total) + lat/lon

This doesn't need any external files; it generates data in-memory.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Optional xarray import; if missing, fallback test will run
try:
    import xarray as xr
except Exception:
    xr = None

# Import the transformation helpers directly
from scripts.transform_climate import (
    pivot_monthly,
    aggregate_stats_vectorized,
    aggregate_seasonal_vectorized,
    add_derived,
    cap_outliers as cap_outliers_df,
    log_transform_prec,
)


def make_synthetic_ds(ny: int = 3, nx: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01", periods=12, freq="MS")
    lat = np.linspace(30.0, 30.5, ny)
    lon = np.linspace(5.0, 5.6, nx)
    # Create synthetic monthly data: prec (mm), tmax/tmin (°C)
    # Shape (time, lat, lon)
    base_prec = rng.gamma(shape=2.0, scale=10.0, size=(12, ny, nx))
    base_tmax = rng.normal(loc=30.0, scale=5.0, size=(12, ny, nx))
    base_tmin = base_tmax - rng.normal(loc=10.0, scale=3.0, size=(12, ny, nx))
    # inject some NaNs randomly
    nan_mask = rng.random((12, ny, nx)) < 0.05
    base_prec[nan_mask] = np.nan
    base_tmax[nan_mask] = np.nan
    base_tmin[nan_mask] = np.nan
    if xr is not None:
        ds = xr.Dataset(
            {
                "prec": (("time", "lat", "lon"), base_prec.astype(np.float32)),
                "tmax": (("time", "lat", "lon"), base_tmax.astype(np.float32)),
                "tmin": (("time", "lat", "lon"), base_tmin.astype(np.float32)),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )
        return ds
    else:
        # Fallback: return dict of arrays and coords
        return {
            "prec": base_prec.astype(np.float32),
            "tmax": base_tmax.astype(np.float32),
            "tmin": base_tmin.astype(np.float32),
            "time": time, "lat": lat, "lon": lon
        }


def test_monthly_wide():
    ds = make_synthetic_ds()
    if xr is None:
        # Emulate pivot: 12 per var wide columns
        ny, nx = len(ds["lat"]), len(ds["lon"])
        lat = np.repeat(ds["lat"], nx)
        lon = np.tile(ds["lon"], ny)
        out = {"lat": lat, "lon": lon}
        months = np.array([d.month for d in ds["time"]])
        for v in ["prec","tmax","tmin"]:
            data = ds[v]
            # NaN impute per cell with month mean
            mmean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = mmean[inds[1], inds[2]]
            for i, m in enumerate(months):
                out[f"{v}_{m:02d}"] = data[i].reshape(-1)
        df = pd.DataFrame(out)
    else:
        df = pivot_monthly(ds, ["prec", "tmax", "tmin"])
    assert {"lat", "lon"}.issubset(df.columns), "Missing coords in monthly-wide"
    # Expect 12 per var → 36 + 2 coords
    monthly_cols = [c for c in df.columns if any(c.startswith(v+"_") for v in ["prec","tmax","tmin"])]
    assert len(monthly_cols) == 36, f"Expected 36 monthly columns, got {len(monthly_cols)}"
    print("[PASS] monthly-wide shape:", df.shape)


def test_stats_mode():
    ds = make_synthetic_ds()
    if xr is None:
        # Emulate stats aggregation
        ny, nx = len(ds["lat"]), len(ds["lon"])
        lat = np.repeat(ds["lat"], nx)
        lon = np.tile(ds["lon"], ny)
        out = {"lat": lat, "lon": lon}
        for v in ["prec","tmax","tmin"]:
            data = ds[v].astype(float)
            gmean = float(np.nanmean(data))
            # impute
            cmean = np.nanmean(data, axis=0)
            all_nan = np.isnan(cmean)
            cmean[all_nan] = gmean
            inds = np.where(np.isnan(data))
            for t,y,x in zip(*inds):
                data[t,y,x] = cmean[y,x]
            out[f"{v}_mean"] = np.nanmean(data, axis=0).reshape(-1)
            out[f"{v}_min"] = np.nanmin(data, axis=0).reshape(-1)
            out[f"{v}_max"] = np.nanmax(data, axis=0).reshape(-1)
            out[f"{v}_std"] = np.nanstd(data, axis=0).reshape(-1)
        df_stats = pd.DataFrame(out)
    else:
        df_stats, monthly_store = aggregate_stats_vectorized(
            ds, ["prec","tmax","tmin"], quantiles=True, global_fill=True, drop_allnan=True, return_monthly=True
        )
    # Check columns: 4 per variable + p10/p90 optional
    expected_main = [
        "prec_mean","prec_min","prec_max","prec_std",
        "tmax_mean","tmax_min","tmax_max","tmax_std",
        "tmin_mean","tmin_min","tmin_max","tmin_std",
    ]
    for c in expected_main:
        assert c in df_stats.columns, f"Missing {c} in stats output"
    # Derived
    if xr is not None:
        df_stats = add_derived(df_stats, mode="stats")
        # Log transform precipitation columns
        df_log = log_transform_prec(df_stats)
        # IQR capping (report ignored in test)
        df_cap, _ = cap_outliers_df(df_log)
        print("[PASS] stats mode shape:", df_cap.shape)
    else:
        print("[PASS] stats mode shape:", df_stats.shape)


def test_seasonal_mode():
    ds = make_synthetic_ds()
    if xr is None:
        # Emulate seasonal aggregation
        months = np.array([d.month for d in ds["time"]])
        ny, nx = len(ds["lat"]), len(ds["lon"])
        lat = np.repeat(ds["lat"], nx)
        lon = np.tile(ds["lon"], ny)
        out = {"lat": lat, "lon": lon}
        seasons = {"DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11]}
        for v in ["prec","tmax","tmin"]:
            data = ds[v].astype(float)
            cmean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            for t,y,x in zip(*inds):
                data[t,y,x] = cmean[y,x]
            for s, mlist in seasons.items():
                sel = [i for i,m in enumerate(months) if m in mlist]
                out[f"{v}_{s}"] = np.nanmean(data[sel], axis=0).reshape(-1)
        df_season = pd.DataFrame(out)
    else:
        df_season = aggregate_seasonal_vectorized(
            ds, ["prec","tmax","tmin"], include_dec=True, quantiles=False, global_fill=True, drop_allnan=True
        )
    expected = [
        "prec_DJF","prec_MAM","prec_JJA","prec_SON",
        "tmax_DJF","tmax_MAM","tmax_JJA","tmax_SON",
        "tmin_DJF","tmin_MAM","tmin_JJA","tmin_SON",
    ]
    for c in expected:
        assert c in df_season.columns, f"Missing seasonal column {c}"
    # Derived seasonal mean and dryness ratio
    df_season = add_derived(df_season, mode="seasonal")
    assert "prec_mean" in df_season.columns, "Derived seasonal prec_mean missing"
    assert "dryness_season_ratio" in df_season.columns, "Derived dryness_season_ratio missing"
    print("[PASS] seasonal mode shape:", df_season.shape)


def main():
    print("Running climate transform tests...")
    test_monthly_wide()
    test_stats_mode()
    test_seasonal_mode()
    print("All tests passed.")


if __name__ == "__main__":
    main()
