# Balancing Techniques

This note summarizes common imbalanced-learning strategies, how they work, pros/cons, and practical guidance for our fire dataset.

## Oversampling (create more minority samples)

- **SMOTE (Synthetic Minority Over-sampling Technique)**
  - What: Generates synthetic minority samples by interpolating between a sample and its nearest minority neighbors.
  - How: For each minority point, pick k-nearest minority neighbors; create synthetic points along the line segments.
  - Pros: Improves minority recall; simple and widely used; less overfitting than naive duplication.
  - Cons: Can create samples in overlapping/noisy regions; inflates dataset size → longer training; may worsen precision when label noise is high.
  - Use when: Minority truly under-represented but informative; noise is moderate; models struggle to learn minority decision boundary.

- **ADASYN (Adaptive Synthetic Sampling)**
  - What: A variant of SMOTE that generates more synthetic samples in hard-to-learn regions (near class overlap).
  - How: Density-weighted oversampling: points with many majority neighbors get more synthetic samples.
  - Pros: Targets difficult areas; can boost recall for complex boundaries.
  - Cons: May amplify boundary noise and overlap; similar runtime inflation as SMOTE; precision can drop.
  - Use when: Minority is complex and under-represented; willing to trade precision for recall.

- **SMOTE + Tomek Links / SMOTE + ENN**
  - What: Oversample first (SMOTE), then clean borderline/overlap using Tomek Links (pairwise) or ENN (rule-based removal).
  - Pros: Controls some of SMOTE’s overlap artifacts; often better precision than SMOTE alone.
  - Cons: Still inflates data; adds complexity; cleaning may remove informative points if noise assumptions are wrong.

## Undersampling (remove majority samples)

- **RandomUnderSampler**
  - What: Randomly drop majority samples to reach target ratio.
  - Pros: Fast; reduces runtime; simple baseline.
  - Cons: Can discard informative majority structure; risk of underfitting.

- **Tomek Links**
  - What: Identify majority–minority nearest-neighbor pairs that are each other’s closest neighbor (Tomek link). Removing the majority part of the link cleans overlap.
  - Pros: Simple boundary cleaning; improves class separation; reduces label noise around decision boundary.
  - Cons: Local/greedy; may remove many points when classes heavily overlap.

- **ENN (Edited Nearest Neighbors)**
  - What: Remove samples whose label disagrees with the majority of their k-nearest neighbors.
  - Pros: Cleans mislabeled or noisy samples; sharpens boundary; reduces confusion.
  - Cons: Choice of k matters; can over-prune; removes both classes if noisy.

- **NearMiss (variants 1/2/3)**
  - What: Select majority samples closest to minority samples by specific heuristics.
  - Pros: Keeps informative majority near boundary; reduces dataset size.
  - Cons: Heuristic-dependent; may skew spatial/feature distribution.

## Reweighting (no change in dataset size)

- **Class Weights (e.g., `class_weight='balanced'`)**
  - What: Increase loss penalty for minority misclassification based on inverse class frequency.
  - Pros: No data inflation; robust for linear/tree models; fast.
  - Cons: If extreme noise exists, weighting alone won’t remove it; KNN/SVM with certain kernels may benefit less than trees.

- **Balanced Random Forest**
  - What: Random Forest variant that undersamples majority class in each tree and/or uses class weights.
  - Pros: Good minority focus without global oversampling; strong baseline.
  - Cons: Slightly more complex; tuning needed.

### Deeper: Class Weights
- Mechanism: Many estimators (e.g., `RandomForestClassifier`, `LogisticRegression`, `SVC`) support `class_weight`. When set to `'balanced'`, weights are computed as `n_samples / (n_classes * n_samples_in_class)`. `'balanced_subsample'` (RF) recomputes per bootstrap subsample, stabilizing trees under imbalance.
- Practical tips:
  - Use `'balanced_subsample'` for Random Forest to adapt each tree’s sample.
  - Still clean obvious noise (e.g., with Tomek/ENN) — weights don’t remove mislabeled points.
  - Track minority recall and precision; class weights often improve recall with minimal runtime cost.

## Cross-Validation Under Imbalance

- **StratifiedKFold**
  - What: A k-fold CV variant that preserves class proportions in each fold.
  - Why: Prevents folds with too few minority samples, yielding stable metrics for imbalanced datasets.
  - How: Split `X, y` into `k` folds with similar class ratios; for each fold, train on `k-1` folds and validate on the remaining.
  - Metrics: Report ROC-AUC (threshold-independent), PR-AUC (sensitive to class imbalance), plus precision/recall/F1.
  - Tips:
    - Use consistent `random_state` for reproducibility.
    - Aggregate metrics across folds (mean ± std) to detect variance.
    - Combine with class weights and/or boundary cleaning (Tomek/ENN) applied within the training folds only to avoid leakage.

## Choosing for Our Fire Dataset

- Observation: We likely have many “fake” fire points (label noise/false positives). In such cases, oversampling (SMOTE/ADASYN) can replicate or synthesize around noise and overlapping regions, hurting precision and increasing runtime.
- Recommendation:
  - Prefer boundary-cleaning undersampling: **Tomek Links + ENN** to remove ambiguous/overlap and mislabeled points.
  - Combine with **class weights** (e.g., Random Forest with `class_weight='balanced_subsample'`) to avoid dataset inflation and keep speed.
  - Validate with **StratifiedKFold** and report ROC-AUC, PR-AUC, F1, precision/recall to ensure both classes are represented fairly.

## Quick Patterns

- If minority is rare but clean → try class weights, then SMOTE-Tomek if recall still low.
- If minority has label noise (fake positives) or strong overlap → Tomek Links / ENN first; avoid oversampling until boundary is clean.
- If runtime is critical → class weights or undersampling; avoid oversampling.

## Minimal API References (imbalanced-learn)

- `imblearn.over_sampling.SMOTE`
- `imblearn.over_sampling.ADASYN`
- `imblearn.combine.SMOTETomek`, `SMOTEENN`
- `imblearn.under_sampling.RandomUnderSampler`, `NearMiss`
- `imblearn.under_sampling.TomekLinks`, `EditedNearestNeighbours`
- `imblearn.ensemble.BalancedRandomForestClassifier`

These enable building pipelines (e.g., scikit-learn `Pipeline`) combining cleaning + modeling.