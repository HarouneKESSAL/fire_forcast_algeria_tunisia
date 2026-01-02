# Algeria–Tunisia Geospatial Pipeline: End-to-End Guide

This document summarizes the data preparation, merging, feature engineering, and modeling workflow implemented under `scripts/` and the expected inputs/outputs. It reflects the recommendations provided (spatial/temporal filling, scaling after merge, climate aggregation, class imbalance handling, and model comparisons).

## 1) Datasets and Cleaning

- Fire (VIIRS):
  - Filter to `type==2` if present; retain core columns (`latitude`, `longitude`, `acq_date`, `frp`, etc.).
  - Create target label: fire = 1 for all real fire points.
  - Generate non-fire points (negatives): spatially even sampling within AOI, with a km buffer around fires to avoid label noise; enforce minimum spacing among negatives. Aim for 70–90% fire ratio overall.

- Land Cover:
  - Rasterize polygons to the template grid using nearest; keep the most precise coding system available.
  - Produce `landcover_codebook.json` mapping code → label.

- Climate:
  - Clip monthly rasters to AOI, align to template, convert units (K→°C if needed).
  - Aggregate per-point either as:
    - Stats: mean, min, max, std for each of `prec`, `tmax`, `tmin` (12 features total), or
    - Seasonal: DJF/MAM/JJA/SON means for each variable (12 features total)
  - Select via `--climate-mode stats|seasonal` when running `final-merge`.

- Elevation:
  - DEM → slope and aspect; clip and align.

- Soil (HWSD2):
  - Build `soil_id` grid; join D1 attributes (0–20 cm) along soil_id dimension, later merged by nearest `soil_id` at point locations.

Cleaning notes:
- Missing spatial values: fill via neighbor/spatial mean when sparse.
- Temporal gaps (if used): forward/backward fill or monthly climatology.
- Outliers: smart imputation uses IQR + data profiling (symmetric/asymmetric/categorical/spatial) with appropriate replacements.
- Feature scaling: performed after the final merge (not per-dataset) via pipelines.

## 2) Spatial Merge Strategy

Order: start from fire points (+ negatives) → join land cover (point-in-polygon), elevation/slope/aspect (raster sample), climate (nearest grid cell and aggregate), soil (sample `soil_id` then join attributes).

Resolution handling:
- Nearest for categorical (land cover).
- Bilinear/nearest sampling for continuous as appropriate.
- Optional quality flags: distance to source grid can be added.

## 3) Feature Engineering & Selection ✅ DONE

### Implementation (Friend's Improved A- Approach)

**Phase 1: Feature Engineering (19 Domain-Specific Features)**
- Climate Features (9):
  - `annual_temp_range`: tmax_max - tmin_min (captures seasonal extremes)
  - `avg_tmax`: Average maximum temperature
  - `avg_tmin`: Average minimum temperature
  - `diurnal_temp_range`: tmax_mean - tmin_mean (within-day variation)
  - `total_annual_precip`: Total precipitation
  - `precip_variability`: Precipitation std dev (drought indicator)
  - `dry_season_intensity`: prec_p10 (lowest month precipitation)
  - `heat_dryness_index`: tmax_mean × (1 - prec_p10/prec_total)
  - `drought_index`: tmax_mean / (prec_total + epsilon)

- Topography Features (4):
  - `terrain_complexity`: ruggedness × relative_elev
  - `elevation_terrain_interaction`: elevation × normalized ruggedness
  - `relative_elevation_effect`: relative_elev²
  - `terrain_roughness_index`: 1 / (1 + ruggedness)

- Soil Features (6):
  - `sand_clay_ratio`: sand / clay
  - `fine_fraction`: silt + clay
  - `organic_matter_content`: org_carbon
  - `moisture_retention`: clay × normalized org_carbon
  - `soil_acidity`: |7 - pH| (distance from neutral)
  - `nutrient_richness`: cec_soil (cation exchange capacity)

**Phase 2: One-Hot Encoding**
- Applied `pd.get_dummies()` to categorical features
- Result: 41 total features after encoding

**Phase 3: Multicollinearity Removal**
- Threshold: r > 0.90
- Strategy: Kept feature with higher target correlation when removing redundant pairs
- Result: 30 features after removal (11 redundant features removed)

**Phase 4: 4-Method Feature Selection (Consensus Approach)**
- Pre-filtered: Top 30 features by target correlation
- Applied 4 independent methods:
  1. ANOVA F-Test (statistical significance)
  2. Mutual Information (dependence measure)
  3. Random Forest Feature Importance (model-based)
  4. Target Correlation (direct relationship with fire)
- Composite Scoring: Mean of 4 normalized scores
- Final Selection: Top 20 features by consensus score

**Phase 5: StandardScaler Normalization**
- Method: Z-score normalization (mean = 0, std = 1)
- Applied to all 20 selected features
- Critical for machine learning models

**Output:**
- `engineered_features_scaled.csv`: Final dataset (71,050 × 23 = 20 features + 3 metadata)
- `feature_importance_scores.csv`: 4-method scoring details
- `selected_features_final.txt`: List of 20 selected features
- `feature_importance_heatmap.png`: Visualization of 4 methods
- `feature_importance_top20.png`: Bar chart of top features
- `feature_engineering_summary_friend_approach.json`: Complete metadata

**Improvements Over Initial Approach:**
✓ One-hot encoded categorical data (preserves information)
✓ Explicit multicollinearity removal (r > 0.90 threshold)
✓ 4 selection methods instead of 3 (more robust consensus)
✓ Systematic cascading: Top 30 → 4 methods → Top 20
✓ StandardScaler applied (critical for model performance)
✓ Reduced feature set (20 features, less overfitting risk)
✓ Production-ready preprocessing pipeline

**Grade: A-** (Friend's comprehensive best-practices approach successfully implemented)

---

## 4) In-Progress: Dataset Balancing & Model Training

### Next Steps:
1. **Dataset Balancing** (Phase 1):
   - Input: `engineered_features_scaled.csv` (71,050 × 23)
   - Current class ratio: ~4:1 (70% negatives, 25% fires)
   - Methods to explore:
     - SMOTE (Synthetic Minority Oversampling)
     - BalancedRandomForest (handles imbalance internally)
     - Stratified train/test split
   - Output: Balanced training/validation datasets

2. **Model Training** (Phase 2):
   - Candidates: Logistic Regression, Random Forest, XGBoost, SVM
   - Evaluation: Cross-validation with stratified splits
   - Metrics: AUC-ROC, Precision, Recall, F1-Score, Balanced Accuracy
   - Outputs: Model comparison, best model selection

3. **Model Interpretation** (Phase 3):
   - Feature importance analysis
   - SHAP values for explainability
   - Performance by region (Algeria vs Tunisia)

---

## 5) Final Features

- Coordinates: `latitude`, `longitude` (kept for spatial CV and distance metrics; can be dropped for certain models).
- Target: `fire` ∈ {0,1} with specified class ratio.
- Land cover: `landcover_code` (+ `landcover_label` if codebook exists).
- Elevation: `elevation`, `slope_deg`, `aspect_deg`.
- Climate (2024): `prec_mean|min|max|std`, `tmax_mean|min|max|std`, `tmin_mean|min|max|std`.
- Soil: `soil_id` + D1 attributes if available.

Output: `processed/merged_dataset.parquet` (CSV fallback).

## 6) Modeling (Original Plan)

Models compared:
- k-NN with `SMOTETomek` imbalance handling inside the pipeline (applied within folds).
- RandomForest (baseline).
- BalancedRandomForest (imblearn) for improved minority handling.

Evaluation protocol:
- Spatial 5-fold CV using KMeans on `(lat, lon)` to form `GroupKFold` blocks (reduces spatial leakage).
- Metrics: ROC-AUC, PR-AUC (minority-focused), F1 (minority), Balanced Accuracy.
- Outputs: `processed/model_results.json` with mean/std across folds.

Feature preprocessing:
- Numeric: `StandardScaler`.
- Categorical: `OneHotEncoder(handle_unknown=ignore)`.
- All inside `sklearn`/`imblearn` pipelines to avoid leakage.

## 7) Commands (Windows PowerShell)

Setup:
```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r visualization\requirements.txt
```

Prepare layers (as needed):
```powershell
python scripts\geo_pipeline.py fire-type2
python scripts\geo_pipeline.py climate
python scripts\geo_pipeline.py climate-nc-2024
python scripts\geo_pipeline.py elevation
python scripts\geo_pipeline.py landcover
python scripts\geo_pipeline.py soil-nc
python scripts\geo_pipeline.py soil-nc-full
```

Negatives + merge (target 80/20, 3 km exclusion):
```powershell
python scripts\geo_pipeline.py generate-negatives --fire-ratio 0.8 --min-dist-km 3.0
python scripts\geo_pipeline.py final-merge --fire-ratio 0.8 --min-dist-km 3.0 --climate-mode seasonal
```

Model evaluation:
```powershell
python scripts\geo_pipeline.py model-eval
type processed\model_results.json
```

## 8) Notes & Extensions

- Seasonal climate features: extend `final-merge` to compute DJF/MAM/JJA/SON means.
- Alternative imbalance: ADASYN or SMOTE-NC if categorical density is high.
- Threshold tuning: calibrate decision thresholds for recall/precision trade-offs.
- Interpretability: permutation importance for RF/BRF; SHAP on a subsample.
