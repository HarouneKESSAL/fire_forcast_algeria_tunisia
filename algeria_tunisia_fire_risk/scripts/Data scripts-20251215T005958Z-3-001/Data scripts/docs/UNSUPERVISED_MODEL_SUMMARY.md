# Unsupervised Methods Summary

This document summarizes the unsupervised analyses performed on the environmental dataset, mirroring the structure used for supervised models. It focuses on K-Means, DBSCAN, and CLARANS (medoid-based), their preprocessing, parameter selection, evaluation metrics, and practical recommendations. Compared to supervised work, the target label `fire` is removed and not used anywhere during clustering.

## Data & Features
- Source: Engineered environmental features prepared in the pipeline (see scripts in `scripts/` and outputs under `DATA_CLEANED/processed/`). The primary tabular file is typically `engineered_features_scaled.csv` or an equivalent with environmental predictors.
- Target removal: Because this is unsupervised, the `fire` label (and any derived supervised targets such as fold-specific labels) is removed prior to clustering. Downstream ARI/AMI comparisons (optional) use `fire` only for evaluation, never for fitting.
- Feature scope: Climate summaries (e.g., monthly/seasonal max/min/mean), elevation (raw and cut), soil attributes (texture codes, derived rasters), landcover categories (encoded). Categorical features are one-hot/ordinal encoded before clustering; continuous features are kept numeric.
- Geographic keys: IDs or coordinates used for joins (e.g., grid cell IDs) are retained only if needed to map back; they are excluded from the feature matrix used for clustering.
- Train/test splits: Not applicable; clustering is conducted on the available unlabeled feature matrix after cleaning. If a subset is used, it is for computational reasons, not for generalization.
- Sampling: For large matrices, random or spatially stratified downsampling may be applied to accelerate parameter sweeps. When sampling, preserve environmental diversity (e.g., stratify by elevation bands or landcover types) to avoid biased clusters.

## Preprocessing
- Cleaning: Apply the same robust cleaning used in supervised prep, referencing `DATA_CLEANED/processed/*summary.json` and `*capping.json` (e.g., winsorization/capping for extreme values). Impute missing values with simple strategies (median/most-frequent) to avoid disrupting distance calculations.
- Feature selection: Remove leakage-prone or post-hoc fields (e.g., any fold-specific metrics, labels, or features engineered with target-aware steps). Retain only environment-derived predictors.
- Scaling: Standardization (zero mean, unit variance) is recommended for K-Means. For DBSCAN, scaling helps make `eps` comparable across dimensions. If features vary in importance, consider scaling per domain (e.g., robust scaler for heavy-tailed climate variables).
- Encoding: Convert categorical landcover codes to one-hot or ordinal encodings. Keep the encoding consistent across the dataset.
- Dimensionality Reduction (optional): PCA can reduce noise and improve clustering stability. Aim for ≥90% variance retained; examine loadings to ensure environmental interpretability. UMAP/t-SNE are useful for visualization but are not used to fit clusters.
- De-duplication: If multiple rows represent the same spatial unit/time period, aggregate or deduplicate to avoid artificial micro-clusters.

## Methods

### K-Means
- Goal: Partition data into `k` compact, spherical clusters using Euclidean distance.
- Parameters: `k` (number of clusters), random initialization count, max iterations.
- Initialization: Use `k-means++` to improve initial centroids and convergence.
- Convergence & stability: Run multiple seeds (e.g., 10–20) and pick the solution with best internal metrics and stable centroids (low variance across seeds).
- Interpretability: Inspect cluster centroids (mean feature values) to understand environmental regimes; map clusters back to geography to assess spatial coherence.

### DBSCAN
- Goal: Density-based clustering that discovers arbitrarily shaped clusters and marks sparse regions as noise.
- Parameters: `eps` (radius), `min_samples` (minimum points to form a dense region). Consider feature scaling before tuning.
- Strengths: Robust to noise/outliers; no need to predefine `k`.
- Sensitivity: `eps` is scale-dependent; ensure features are standardized. Larger `min_samples` yields smoother clusters and more noise points.
- Labeling: Points not belonging to any dense region get label `-1` (noise). These may reflect rare or anomalous environmental conditions.

### CLARANS (Medoids)
- Goal: Partition data by selecting representative medoids (actual data points) and assigning each sample to its nearest medoid using Manhattan distance on standardized features.
- Implementation: From-scratch randomized medoid swap search with early stopping and interrupt-safe checkpoints.
- Sampling: Run on a large sampled subset to explore multiple `k` values efficiently; then map sampled medoids back to full data and assign all rows.
- Selection: Choose `k` by sample silhouette while avoiding tiny clusters.
- Saved outputs (under `f:/DATA/results/unsupervised/`):
  - `clarans_partial.json` — incremental improvements per `k`.
  - `clarans_summary.json` — parameters, per-`k` best medoids, sample metrics, and selected indices.
  - `clarans_labels.csv` — full-dataset assignments to nearest medoid.

## Evaluation Metrics
- Silhouette Score ($s \in [-1,1]$): Mean of per-sample silhouettes, $s = (b - a)/\max(a,b)$, where $a$ is intra-cluster distance and $b$ is nearest-cluster distance. Higher is better; values near 0 indicate overlapping clusters.
- Davies–Bouldin Index (DBI): Average ratio of within-cluster scatter to between-cluster separation. Lower is better; tends to penalize clusters that are close and diffuse.
- Calinski–Harabasz (CH) Index: $(\text{between dispersion})/(\text{within dispersion})$ scaled by cluster count; higher is better.
- Noise Fraction (DBSCAN): Proportion of points labeled `-1`. Extremely high noise suggests `eps` is too small or the data lacks dense structure.
- Stability: K-Means stability across seeds (centroid variance, adjusted labels via Hungarian matching). DBSCAN stability under small `eps` or `min_samples` changes (label consistency rate).

## Parameter Selection

### K-Means (`k` selection)
- Elbow Method (CH or inertia): Plot metric vs. `k` and select the elbow where gains diminish; corroborate with silhouette.
- Silhouette Analysis: Prefer `k` values that maximize silhouette and avoid tiny/degenerate clusters.
- Seed robustness: Confirm chosen `k` yields similar metrics across different initializations.
- Practical Bounds: Explore `k` in 3–15 (adjust to dataset size/complexity). Avoid very large `k` unless domain dictates many micro-regimes.

### DBSCAN (`eps`, `min_samples`)
- k-distance graph: Plot the distance to the `min_samples`-th nearest neighbor; choose `eps` around the elbow.
- Grid/Band Search: Sweep `eps` (e.g., 0.2–2.0 after scaling) and `min_samples` (e.g., 5–50). Prefer solutions with high silhouette, low DBI, and reasonable noise fraction (e.g., <30%).
- Post-check: Visualize cluster shapes (PCA/UMAP projection) and geographic distribution to ensure coherence.

## Results Summary (Template)
- K-Means:
  - Best `k`: <fill-in>
  - Silhouette: <fill-in>
  - DBI / CH: <fill-in>
  - Cluster sizes: <fill-in> (distribution across clusters)
  - Centroid interpretation: <fill-in> (key features defining each cluster)
  - Notes: Any clusters dominated by a single region/landcover? Any overlap visible in silhouette plots?
- DBSCAN:
  - Best `eps`, `min_samples`: <fill-in>
  - Silhouette: <fill-in>
  - DBI / CH: <fill-in>
  - Noise Fraction: <fill-in> (proportion labeled `-1`)
  - Cluster shapes: <fill-in> (arbitrary shapes vs compact)
  - Notes: Geographic/environmental coherence; sensitivity to parameter changes; anomalies flagged as noise.

- CLARANS:
  - Best `k`: <fill-in>
  - Silhouette (sample): <fill-in>
  - Cluster sizes (full): <fill-in>
  - Notes: Verify tiny clusters are avoided; check spatial coherence and feature regimes.

## Practical Interpretation
- High Silhouette with strong CH suggests compact, well-separated clusters.
- Low DBI indicates clusters that are distinct and internally tight; scrutinize clusters with high intra-scatter.
- DBSCAN noise helps flag atypical conditions (potential anomalies or rare regimes); inspect noise points for extreme feature values or sparse geographies.
- Environmental alignment: If clusters align with strata (elevation bands, landcover types, soil textures), treat them as regimes for downstream modeling, targeted sampling, or policy interventions.
- Operationalization: Use cluster IDs as categorical features in supervised models, or build regime-specific models where performance/behavior differs.

## Supervised Overlays (Optional)
- Comparison without leakage: After clustering (with `fire` removed), overlay the `fire` label to compute Adjusted Rand Index (ARI) or Adjusted Mutual Information (AMI). These do not affect clustering and are purely evaluative.
- Utility detection: Non-trivial ARI/AMI indicates clusters encode structure relevant to `fire`. Consider adding cluster ID as a feature or stratifying model thresholds by cluster.
- Caveat: High ARI/AMI does not guarantee causal relevance; validate via supervised performance improvements.

## Artifacts & Scripts
- Data: See `DATA_CLEANED/processed/` for cleaned feature sets and stats JSONs; unsupervised runs should explicitly drop `fire` before fitting.
- Scripts: `scripts/clip_landcover.py`, `scripts/merge_env.py`, `scripts/transform_*`, `scripts/clean_data.py` for preprocessing; `scripts/check_fire_distribution.py` for understanding class prevalence (used only post hoc).
- Notebooks: Use display-only notebooks to plot silhouette vs `k`, k-distance graphs, PCA overlays, and geographic cluster maps.

## Recommendations
- Start with K-Means (scaled features) to get a baseline partition; validate with silhouette and CH; confirm seed robustness.
- Run DBSCAN to capture non-spherical clusters and noise; tune `eps` via k-distance; target reasonable noise rates (<30%) unless anomalies are the focus.
- Consider PCA to reduce noise, then reevaluate clustering metrics and interpret loadings.
- Map clusters back to geography; inspect environmental summaries per cluster; document regime characteristics.
- Use clusters as categorical features in supervised models or to design regime-specific models and thresholds.

---

Fill in the template fields with your latest runs and, where useful, add small inline plots (silhouette vs. `k`, k-distance) in a notebook to keep everything display-only and reproducible. Always ensure `fire` (and any target-derived features) are removed before fitting clusters.