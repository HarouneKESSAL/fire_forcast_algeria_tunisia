# From-Scratch vs scikit-learn Implementations

Comparison of custom implementations with scikit-learn equivalents.

---

## KNN (k-Nearest Neighbors)

### Core Hyperparameters

| Parameter | From-Scratch | scikit-learn | Notes |
|-----------|-------------|-------------|-------|
| **Number of neighbors** | `--k` (default: 11) | `n_neighbors` (default: 5) | Control neighborhood size |
| **Distance metric** | `--metric` (default: "manhattan") | `metric` (default: "minkowski") | euclidean, manhattan, etc. |
| **Voting strategy** | uniform only | `weights` (uniform, distance) | From-scratch: uniform (equal weight); sklearn: can weight by distance |
| **Search algorithm** | brute force | `algorithm` (auto, brute, kd_tree, ball_tree) | From-scratch: O(n) per query; sklearn: can optimize |
| **Test-train split** | `--test-size` (default: 0.2) | N/A (user-defined) | From-scratch: built-in via argparse |
| **Random seed** | `--seed` (default: 42) | `random_state` | For reproducibility |

### Implementation Details

#### From-Scratch (`scripts/knn_from_scratch.py`)
- **Distance calculation:**
  - Euclidean: `sqrt(sum((train - x)^2))`
  - Manhattan: `sum(abs(train - x))`
  - Vectorized with NumPy (`einsum`, `abs`)
- **Neighbor selection:** `np.argpartition` for partial sort (O(n) best-case)
- **Voting:** `np.unique` to count class occurrences; break ties by smallest class value
- **Prediction:** batch processing (default batch_size=256) to manage memory
- **Scaling:** StandardScaler on train split only (no leakage)
- **Progress:** tqdm with fallback to no-op if not available
- **Output:** predictions CSV + results summary JSON

#### scikit-learn KNeighborsClassifier
- **Distance calculation:** pluggable metrics (Minkowski, Euclidean, Manhattan, custom)
- **Neighbor selection:** optimized tree-based searches (KD-Tree, Ball-Tree) or brute force
- **Voting:** uniform or distance-weighted
- **Prediction:** vectorized batch prediction built-in
- **Scaling:** user responsibility (fit on train, apply to test)
- **Output:** fitted model; predictions via `.predict()`

### Feature Selection

| Aspect | From-Scratch | scikit-learn |
|--------|-------------|-------------|
| **Method** | Top-N variance (excluding target) | User provides features |
| **Default N** | 20 | N/A |
| **Target exclusion** | Automatic | User responsibility |
| **Variance ranking** | `df[cols].var().sort_values()` | N/A |

### Evaluation Metrics

Both compute:
- **Accuracy:** fraction of correct predictions
- **Precision (macro):** average precision across classes
- **Recall (macro):** average recall across classes
- **F1 (macro):** harmonic mean of precision/recall
- **Confusion Matrix:** predicted vs. true labels

### CLI Usage

**From-Scratch:**
```bash
# Single k
python scripts/knn_from_scratch.py --k 11 --metric manhattan --test-size 0.2 --seed 42

# k-grid sweep
python scripts/knn_from_scratch.py --k-grid 3,5,7,9,11,15 --metric manhattan

# Skip saving
python scripts/knn_from_scratch.py --no-save
```

**scikit-learn (typical):**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=11, metric='manhattan')
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)
```

### Results Storage

**From-Scratch:**
- Predictions: `f:/DATA/results/supervised/knn_predictions.csv`
- Summary: `f:/DATA/results/supervised/knn_from_scratch_summary.json` (includes k, metric, features, metrics)

**scikit-learn:**
- User-defined storage (typically pickle for model or CSV for predictions)

### Known Differences / Trade-offs

| Aspect | From-Scratch | scikit-learn |
|--------|-------------|-------------|
| **Memory** | O(d×n) for brute force | O(d×n) (brute) or O(d log n) (tree) |
| **Speed** | Slower (pure NumPy, no C extensions) | Faster (C-optimized, tree structures) |
| **Flexibility** | Custom distance functions easy | Pluggable metrics, weights |
| **Production** | Educational, debugging | Battle-tested, optimized |
| **Code size** | ~300 lines | ~10,000+ lines (C backend) |

---

## DBSCAN (Density-Based Spatial Clustering)

### Core Hyperparameters

| Parameter | From-Scratch | scikit-learn | Notes |
|-----------|-------------|-------------|-------|
| **Neighborhood radius** | `--eps` | `eps` | Maximum distance to form neighborhood |
| **Min core points** | `--min-samples` | `min_samples` | Minimum neighbors to be a core point |
| **Distance metric** | `--metric` (default: "manhattan") | `metric` (default: "euclidean") | euclidean, manhattan, etc. |
| **Grid search** | `--eps-grid`, `--min-samples-grid` | N/A | From-scratch: built-in sweep |
| **Random seed** | `--seed` (default: 42) | `random_state` | For reproducibility (if applicable) |

### Implementation Details

#### From-Scratch (`scripts/dbscan_from_scratch.py` — TBD)
- **Neighborhood discovery:** distance matrix or pairwise calculations
- **Core point identification:** count neighbors within eps
- **Cluster formation:** union-find or iterative expansion
- **Noise labeling:** points not in any cluster
- **Output:** cluster labels, diagnostics

#### scikit-learn DBSCAN
- **Neighborhood discovery:** BallTree or KD-Tree for efficiency
- **Core point identification:** k-d tree radius search
- **Cluster formation:** union-find with efficiency optimizations
- **Noise labeling:** points with label -1
- **Output:** fitted model with `.labels_` attribute

### Evaluation Metrics

Both compute (where applicable):
- **Silhouette Score:** cluster cohesion and separation
- **Davies-Bouldin Index:** average similarity ratio
- **Calinski-Harabasz Index:** cluster separation
- **Noise Rate:** fraction of points labeled -1

### CLI Usage

**From-Scratch (TBD):**
```bash
# Single parameter set
python scripts/dbscan_from_scratch.py --eps 1.0 --min-samples 10 --metric manhattan

# Parameter grid sweep
python scripts/dbscan_from_scratch.py --eps-grid 0.5,1.0,1.5 --min-samples-grid 5,10,15
```

**scikit-learn (typical):**
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

dbscan = DBSCAN(eps=1.0, min_samples=10, metric='manhattan')
labels = dbscan.fit_predict(X_s)
```

### Known Differences / Trade-offs

| Aspect | From-Scratch | scikit-learn |
|--------|-------------|-------------|
| **Memory** | O(n²) for full distance matrix (high!) | O(n log n) with tree structures |
| **Speed** | Slower (pure Python/NumPy) | Faster (C backend, tree optimization) |
| **Scalability** | Limited (avoid large n) | Excellent (handles 100k+ points) |
| **Flexibility** | Custom clustering logic | Standard DBSCAN only |
| **Production** | Educational, small datasets | Battle-tested, scalable |

---

## Summary Table: Feature & Performance Comparison

| Feature | KNN From-Scratch | KNN scikit-learn | DBSCAN From-Scratch | DBSCAN scikit-learn |
|---------|-----------------|-----------------|-------------------|-------------------|
| **Hyperparameter tuning** | ✅ CLI grid sweep | ❌ User code | ✅ CLI grid sweep | ❌ User code |
| **Built-in evaluation** | ✅ Accuracy, precision, recall, F1, CM | ❌ User responsibility | ✅ Silhouette, DB, CH | ❌ User responsibility |
| **Memory efficient** | ⚠️ O(d×n) | ✅ Optimized | ❌ O(n²) | ✅ O(n log n) |
| **Speed** | ❌ Slow | ✅ Fast | ❌ Slow | ✅ Fast |
| **Educational** | ✅ Clear code | ❌ Complex C backend | ✅ Clear code | ❌ Complex C backend |
| **Production ready** | ❌ Not recommended | ✅ Recommended | ❌ Not recommended | ✅ Recommended |

---

## Next Steps

- [ ] Create `scripts/dbscan_from_scratch.py` with parameter grid sweep
- [ ] Update DBSCAN section above with actual implementation details
- [ ] Run both implementations on the same dataset
- [ ] Compare:
  - Execution time (single run + grid sweep)
  - Memory usage
  - Result quality (silhouette, cluster counts, noise rate)
  - Output format and interpretability
- [ ] Add performance benchmarks and visualizations

---

## References

- scikit-learn KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- scikit-learn DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- Project scripts:
  - `f:/DATA/scripts/knn_from_scratch.py`
  - `f:/DATA/scripts/dbscan_from_scratch.py` (TBD)
  - `f:/DATA/scripts/unsupervised_pipeline.py` (uses scikit-learn)
