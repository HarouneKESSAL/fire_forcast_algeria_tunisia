# Supervised Model Training Summary

This document summarizes our supervised pipeline: feature engineering, data cleaning, cross-validation strategy, threshold tuning, and model-specific approaches. It covers both from-scratch implementations and sklearn baselines, with best observed parameters where applicable. Unsupervised methods will be documented next.

## Pipeline Overview
- **Data Source:** f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv
- **Target:** `fire` (binary)
- **Train/Test Split:** Stratified (`test_size=0.2`, `random_state=42`), or Stratified K-Fold (`n_splits=3–5`).
- **Imbalance Handling:** Tomek Links + Edited Nearest Neighbours (applied to training folds only) when `--clean-data`.
- **Scaling:** Pre-computed engineered features are already scaled; no leakage introduced in scripts.
- **Thresholding:**
  - From-scratch fast mode (`--simple-cv`): fixed threshold (e.g., `0.77`) or default `0.5`.
  - Full tuning: select threshold via PR curve to maximize F1 or satisfy target precision/recall.
- **Progress + Interrupts:** All long runs show progress bars; KeyboardInterrupt saves partial results (summary JSON + CV predictions CSV).

## Feature Engineering (Supervised)
- **Source artifacts:** engineered feature tables built in Phase 1 and saved as `engineered_features_scaled.csv`.
- **Standardization:** features are pre-scaled to stabilize distance-based methods (KNN) and tree splits; prevents leakage by using already prepared tables.
- **Derivations:**
  - Aggregated climate summaries (e.g., temperature extremes, seasonal stats).
  - Elevation and terrain attributes (e.g., DEM values, cutouts for study area).
  - Landcover encodings aligned to regional classes; categorical features one-hot/ordinal encoded upstream.
  - Soil texture and attributes derived from HWSD2 layers; missing values imputed in preprocessing.
- **Filtering:** low-variance and redundant columns removed to reduce noise; domain-informed exclusions to avoid target leakage.
- **Consistency:** identical feature matrix used across models for fair comparison; train/test splits maintain stratification.

## Data Balancing (Role & Techniques)
- **Why balance?** Fire events are minority relative to non-fire, creating class imbalance that skews metrics (high accuracy, poor recall) and alters decision thresholds.
- **Techniques used:**
  - **Tomek Links** (under-sampling): removes borderline majority samples that overlap minority, cleaning decision boundaries.
  - **Edited Nearest Neighbours (ENN)** (under-sampling): removes misclassified samples based on neighbor consensus, further de-noising.
  - **SMOTE / SMOTETomek** (over/combined sampling; notebook exploration): synthetically oversamples minority and cleans overlaps.
  - **Class Weights** (sklearn models): adjusts loss to penalize misclassifying minority more (`balanced` / `balanced_subsample`).
- **Operational choice in scripts:** Apply Tomek+ENN to training folds only (`--clean-data`) to avoid leakage and stabilize CV; for RF we also validated `class_weight='balanced_subsample'` in sklearn.
- **Effects:**
  - Better minority recall at comparable precision when paired with tuned thresholds.
  - Smoother PR curves and more reliable threshold selection (e.g., RF tuned ~0.77).
  - Potential reduction in overfitting by simplifying class boundaries.

## Models

### 1) KNN (k-Nearest Neighbors)
**From-Scratch:** `scripts/knn_from_scratch.py`
- Distance metrics: Euclidean and Manhattan (vectorized NumPy), neighbors via `np.argpartition`.
- Majority voting with tie-breaker; batch prediction for memory efficiency.
- CLI: `--k`, `--metric`, `--k-grid`, `--test-size`, `--seed`, save options.
- Fast mode supported; progress bars; results saved to JSON/CSV.

**sklearn:** `KNeighborsClassifier`
- Typical params tested: `n_neighbors=15`, `weights='distance'`.
- Observations: competitive ROC-AUC on engineered features; prediction time scales with train size.

### 2) Decision Tree (CART)
**From-Scratch:** `scripts/decision_tree_from_scratch.py`
- CART with Gini impurity; optimized via threshold binning (`n_bins=256`) for speed.
- CV with optional Tomek+ENN cleaning; two modes:
  - `--simple-cv`: fixed threshold (default `0.77`), fast evaluation.
  - Full tuning: PR/ROC curves; F1-max threshold. Observed tuned threshold often at `≈ 1.0` due to coarse leaf probabilities.
- Example aggregated metrics (full tuning, 10% sample + cleaning):
  - Threshold: `1.0000`
  - ROC-AUC: `~0.81`, PR-AUC: `~0.65`, Precision: `~0.90`, Recall: `~0.64`, F1: `~0.75`.

**sklearn:** `DecisionTreeClassifier`
- Typical baseline: `max_depth=None`, `random_state=42`.
- Observations: single-tree performance below ensembles; sensitive to class imbalance unless balanced/class-weighted.

### 3) Random Forest
**From-Scratch:** `scripts/random_forest_from_scratch.py`
- Ensemble of from-scratch CART trees (bagging). Optimizations:
  - Threshold binning for splits (`n_bins=256`) → 5–10x faster tree building.
  - Vectorized predictions; progress bars; interrupt-safe saving.
- CV + thresholding:
  - `--simple-cv` fast mode: fixed threshold (e.g., `0.77`), no PR/ROC curves.
  - Full tuning: PR/ROC curves with optimization modes: `f1`, `precision_at_recall`, `recall_at_precision`.
- Small-sample demo (10%, 100 trees, cleaning, fixed thr 0.77): improved recall vs 0.5 threshold; F1 around `~0.62`.

**sklearn:** `RandomForestClassifier`
- Best observed params: `n_estimators=400`, `class_weight='balanced_subsample'`, `n_jobs=-1`, `random_state=42`.
- With Tomek+ENN cleaning and PR-based F1 threshold tuning (mean `~0.7715`):
  - ROC-AUC: `~0.9166`
  - PR-AUC: `~0.8886`
  - Precision: `~0.9290`
  - Recall: `~0.7701`
  - F1: `~0.8420`
- Notes: Tuned threshold `~0.77–0.78` balances precision/recall under class imbalance.

## Results Storage
- Supervised summaries: JSON in `f:/DATA/results/supervised/` (e.g., `*_summary.json`).
- Per-sample CV predictions: CSV in `f:/DATA/results/supervised/` (e.g., `*_predictions.csv`).
- JSON includes: model config, thresholds per fold, metrics per fold, optional curves, aggregated metrics, interrupt flags.

## Practical Guidance
- For fast educational runs: use sampling (`--sample-frac 0.05–0.10`), `--simple-cv`, and fixed thresholds (start at `0.77`).
- For best performance: use sklearn RF with Tomek+ENN cleaning and PR-based threshold tuning.
- Threshold knobs:
  - Fewer false alarms → raise threshold to `~0.80–0.81`.
  - Fewer misses → lower threshold to `~0.74–0.75`.

---

Next: Unsupervised methods (K-Means, DBSCAN), including from-scratch implementations, metrics (silhouette, Davies-Bouldin, Calinski-Harabasz), and supervised overlays (ARI/AMI) for evaluation.