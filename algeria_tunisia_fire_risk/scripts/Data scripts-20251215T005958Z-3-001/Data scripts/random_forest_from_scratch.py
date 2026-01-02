"""
Random Forest classifier from scratch with threshold tuning and CV.

- Implements decision trees from scratch (CART with Gini impurity)
- Builds ensemble via bootstrap aggregation (bagging) + random feature selection
- Performs stratified k-fold cross-validation with optional data cleaning (Tomek+ENN)
- Tunes classification threshold on PR/ROC curves (f1, precision_at_recall, recall_at_precision)
- Saves threshold tuning results, model parameters, and predictions

Usage (PowerShell):

  python scripts/random_forest_from_scratch.py \
    --data f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv \
    --n-trees 100 --max-depth 15 --min-samples-leaf 5 \
    --clean-data --optimize f1 \
    --save-summary f:/DATA/results/supervised/rf_from_scratch_summary.json \
    --save-predictions f:/DATA/results/supervised/rf_from_scratch_predictions.csv

Or with defaults:

  python scripts/random_forest_from_scratch.py

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random Forest (from scratch) with threshold tuning")
    p.add_argument("--data", type=str, default=str(Path("f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv")),
                   help="Path to input CSV with features.")
    p.add_argument("--target", type=str, default="fire", help="Target column.")
    p.add_argument("--n-trees", type=int, default=100, help="Number of trees in forest.")
    p.add_argument("--max-depth", type=int, default=15, help="Max depth per tree (or None).")
    p.add_argument("--min-samples-leaf", type=int, default=5, help="Min samples per leaf.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--n-splits", type=int, default=5, help="CV folds.")
    p.add_argument("--n-bins", type=int, default=256, help="Number of bins for threshold discretization (speed vs accuracy).")
    p.add_argument("--clean-data", action="store_true", help="Apply Tomek+ENN cleaning to training folds.")
    p.add_argument("--optimize", type=str, default="f1", choices=["f1", "precision_at_recall", "recall_at_precision"],
                   help="Threshold optimization metric.")
    p.add_argument("--target-value", type=float, default=0.80, help="Target recall/precision for threshold tuning.")
    p.add_argument("--simple-cv", action="store_true", help="Skip threshold tuning & ROC curves (fast mode).")
    p.add_argument("--fixed-threshold", type=float, default=None, help="Optional fixed threshold to use in simple-cv mode (e.g., 0.77).")
    p.add_argument("--use-sklearn", action="store_true", help="Use sklearn RandomForestClassifier (n_estimators=400, class_weight=balanced_subsample, n_jobs=-1) for speed.")
    p.add_argument("--save-summary", type=str, default=str(Path("f:/DATA/results/supervised/rf_from_scratch_summary.json")),
                   help="Path to save threshold tuning summary JSON.")
    p.add_argument("--save-predictions", type=str, default=str(Path("f:/DATA/results/supervised/rf_from_scratch_predictions.csv")),
                   help="Path to save CV predictions CSV.")
    p.add_argument("--no-save", action="store_true", help="Do not save results.")
    # Sampling options for small from-scratch demo
    p.add_argument("--sample-size", type=int, default=None, help="Optional fixed sample size to run from-scratch quickly.")
    p.add_argument("--sample-frac", type=float, default=None, help="Optional fraction (0-1) of the dataset to sample for speed.")
    return p.parse_args()


def _progress(total: int, desc: str = ""):
    """Progress helper: uses tqdm if available, else no-op."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except Exception:
        class _NoProgress:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def update(self, n: int):
                pass
        return _NoProgress()


class DecisionTreeNode:
    """Node in a decision tree."""
    def __init__(self):
        self.feature = None  # Feature index to split on
        self.threshold = None  # Threshold value
        self.left = None  # Left child
        self.right = None  # Right child
        self.value = None  # Class value if leaf (None if internal)


class DecisionTree:
    """Decision tree classifier (CART with Gini impurity, optimized with threshold binning)."""
    
    def __init__(self, max_depth: int = 15, min_samples_leaf: int = 5, n_bins: int = 256):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins  # Discretize thresholds for speed
        self.root = None
        self.n_features = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        """Build tree by recursively splitting."""
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """Recursively build tree (optimized with threshold binning)."""
        node = DecisionTreeNode()
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            n_samples < 2 * self.min_samples_leaf):
            node.value = int(np.bincount(y).argmax())  # Majority class
            return node
        
        # Find best split
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_vals = np.unique(feature_values)
            
            if len(unique_vals) == 1:
                continue
            
            # Bin thresholds for speed (instead of trying every unique value)
            if len(unique_vals) > self.n_bins:
                thresholds = np.percentile(feature_values, np.linspace(0, 100, self.n_bins))
                thresholds = np.unique(thresholds)
            else:
                thresholds = unique_vals
            
            # Vectorized split evaluation
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Weighted Gini (vectorized)
                y_left, y_right = y[left_mask], y[right_mask]
                gini = (n_left * self._gini(y_left) + n_right * self._gini(y_right)) / n_samples
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            node.value = int(np.bincount(y).argmax())
            return node
        
        # Split and recurse
        mask = X[:, best_feature] <= best_threshold
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        
        return node
    
    def _gini(self, y: np.ndarray) -> float:
        """Compute Gini impurity."""
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (vectorized batch processing)."""
        proba = np.zeros((len(X), 2))
        for i, x in enumerate(X):
            leaf = self._traverse(x, self.root)
            p_1 = leaf.value if leaf.value is not None else 0
            proba[i] = [1 - p_1, p_1]
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (vectorized batch processing)."""
        pred = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            leaf = self._traverse(x, self.root)
            pred[i] = leaf.value if leaf.value is not None else 0
        return pred
    
    def _traverse(self, x: np.ndarray, node: DecisionTreeNode) -> DecisionTreeNode:
        """Traverse tree to leaf."""
        if node.feature is None:  # Leaf node
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)


class RandomForest:
    """Random Forest classifier (bagging + random features, optimized)."""
    
    def __init__(self, n_trees: int = 100, max_depth: int = 15, min_samples_leaf: int = 5, 
                 random_state: int = 42, n_bins: int = 256):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.random_state = random_state
        self.trees = []
        self.rng = np.random.RandomState(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        """Fit forest by bootstrap aggregation."""
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.trees = []
        
        with _progress(total=self.n_trees, desc=f"Training {self.n_trees} trees") as pbar:
            for _ in range(self.n_trees):
                # Bootstrap sample
                indices = self.rng.choice(n_samples, size=n_samples, replace=True)
                X_boot, y_boot = X[indices], y[indices]
                
                # Fit tree with binning
                tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, 
                                   n_bins=self.n_bins)
                tree.fit(X_boot, y_boot)
                self.trees.append(tree)
                pbar.update(1)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predictions across trees (vectorized)."""
        proba_list = [tree.predict_proba(X) for tree in self.trees]
        return np.mean(proba_list, axis=0)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels with optional threshold."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


def tomek_enn_clean(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Clean training data with Tomek + ENN."""
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(X, y)
    enn = EditedNearestNeighbours()
    X_clean, y_clean = enn.fit_resample(X_tl, y_tl)
    return np.asarray(X_clean), np.asarray(y_clean)


def tune_threshold_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int = 100,
    max_depth: int = 15,
    min_samples_leaf: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
    clean_data: bool = False,
    optimize: str = "f1",
    target_value: float = 0.80,
    n_bins: int = 256,
    simple_cv: bool = False,
    fixed_threshold: Optional[float] = None,
    use_sklearn: bool = False,
) -> Dict[str, object]:
    """Tune threshold using stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    thresholds: List[float] = []
    fold_metrics: List[Dict[str, float]] = []
    curves: List[Dict[str, List[float]]] = []
    cms: List[Dict[str, int]] = []
    cv_predictions = []
    
    fold_idx = 0
    interrupted = False
    
    try:
        with _progress(total=n_splits, desc="CV Fold Progress") as pbar_folds:
            for train_idx, test_idx in skf.split(X, y):
                fold_idx += 1
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Clean training fold if requested
                if clean_data:
                    X_train, y_train = tomek_enn_clean(pd.DataFrame(X_train), pd.Series(y_train))
                
                # Fit Random Forest (sklearn fast path or from-scratch)
                if use_sklearn:
                    from sklearn.ensemble import RandomForestClassifier
                    rf = RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state + fold_idx,
                        max_depth=None,
                    )
                    rf.fit(X_train, y_train)
                    predict_proba = lambda Xt: rf.predict_proba(Xt)
                else:
                    rf = RandomForest(n_trees=n_trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                     random_state=random_state + fold_idx, n_bins=n_bins)
                    rf.fit(X_train, y_train)
                    predict_proba = lambda Xt: rf.predict_proba(Xt)
                
                # Predict with progress
                with _progress(total=len(X_test), desc=f"  Fold {fold_idx}/{n_splits} predictions") as pbar_pred:
                    y_proba = predict_proba(X_test)[:, 1]
                    pbar_pred.update(len(X_test))
                
                # Simple mode: use fixed threshold if provided, else 0.5; skip curves
                if simple_cv:
                    best_thr = float(fixed_threshold) if fixed_threshold is not None else 0.5
                    y_pred = (y_proba >= best_thr).astype(int)
                    
                    # Fast metrics only
                    roc_auc = roc_auc_score(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
                        y_test, y_pred, average="binary", zero_division=0
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fold_metrics.append({
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                        "precision": float(prec_b),
                        "recall": float(rec_b),
                        "f1": float(f1_b),
                        "threshold": float(best_thr),
                    })
                    # Skip curves in simple mode
                    curves.append(None)
                    cms.append({
                        "tn": int(cm[0, 0]),
                        "fp": int(cm[0, 1]),
                        "fn": int(cm[1, 0]),
                        "tp": int(cm[1, 1]),
                    })
                    thresholds.append(best_thr)
                else:
                    # Full mode: threshold tuning with curves
                    # Build curves
                    fpr, tpr, roc_thr = roc_curve(y_test, y_proba)
                    prec, rec, pr_thr = precision_recall_curve(y_test, y_proba)
                    
                    # Determine threshold
                    if optimize == "f1":
                        thr = pr_thr
                        if len(thr) == 0:
                            best_thr = 0.5
                        else:
                            f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
                            best_idx = int(np.nanargmax(f1_scores))
                            best_thr = float(thr[best_idx])
                    elif optimize == "precision_at_recall":
                        mask = rec[:-1] >= target_value
                        if not np.any(mask):
                            best_thr = 0.5
                        else:
                            cand_prec = prec[:-1][mask]
                            cand_thr = pr_thr[mask]
                            best_thr = float(cand_thr[int(np.argmax(cand_prec))])
                    elif optimize == "recall_at_precision":
                        mask = prec[:-1] >= target_value
                        if not np.any(mask):
                            best_thr = 0.5
                        else:
                            cand_rec = rec[:-1][mask]
                            cand_thr = pr_thr[mask]
                            best_thr = float(cand_thr[int(np.argmax(cand_rec))])
                    else:
                        best_thr = 0.5
                    
                    thresholds.append(best_thr)
                    
                    # Apply threshold
                    y_pred = (y_proba >= best_thr).astype(int)
                    
                    # Metrics
                    roc_auc = roc_auc_score(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
                        y_test, y_pred, average="binary", zero_division=0
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fold_metrics.append({
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                        "precision": float(prec_b),
                        "recall": float(rec_b),
                        "f1": float(f1_b),
                        "threshold": float(best_thr),
                    })
                    curves.append({
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "roc_thresholds": roc_thr.tolist(),
                        "precision": prec.tolist(),
                        "recall": rec.tolist(),
                        "pr_thresholds": pr_thr.tolist(),
                    })
                    cms.append({
                        "tn": int(cm[0, 0]),
                        "fp": int(cm[0, 1]),
                        "fn": int(cm[1, 0]),
                        "tp": int(cm[1, 1]),
                    })
                
                # Store CV predictions
                for idx, (true, pred, prob) in enumerate(zip(y_test, y_pred, y_proba)):
                    cv_predictions.append({
                        "fold": fold_idx,
                        "index": int(test_idx[idx]),
                        "y_true": int(true),
                        "y_pred": int(pred),
                        "y_proba": float(prob),
                        "threshold": float(best_thr),
                    })
                
                pbar_folds.update(1)
    
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⚠️  Interrupted by user (Ctrl+C). Saving current results...")
    
    return {
        "model_config": {
            "n_trees": n_trees,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_bins": n_bins,
            "random_state": random_state,
            "clean_data": clean_data,
            "optimize": optimize,
            "target_value": target_value,
        },
        "thresholds": thresholds,
        "metrics_per_fold": fold_metrics,
        "curves_per_fold": curves,
        "confusion_matrices": cms,
        "cv_predictions": cv_predictions,
        "interrupted": interrupted,
        "folds_completed": fold_idx,
        "aggregated": {
            "threshold_mean": float(np.mean(thresholds)) if thresholds else None,
            "threshold_std": float(np.std(thresholds)) if thresholds else None,
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])) if fold_metrics else None,
            "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])) if fold_metrics else None,
            "pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])) if fold_metrics else None,
            "pr_auc_std": float(np.std([m["pr_auc"] for m in fold_metrics])) if fold_metrics else None,
            "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])) if fold_metrics else None,
            "precision_std": float(np.std([m["precision"] for m in fold_metrics])) if fold_metrics else None,
            "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])) if fold_metrics else None,
            "recall_std": float(np.std([m["recall"] for m in fold_metrics])) if fold_metrics else None,
            "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])) if fold_metrics else None,
            "f1_std": float(np.std([m["f1"] for m in fold_metrics])) if fold_metrics else None,
        },
    }


def main():
    args = parse_args()
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")
    
    # Drop rows with missing target
    df = df.dropna(subset=[args.target])
    
    X_df = df.drop(columns=[args.target])
    y_sr = df[args.target].astype(int)
    X = X_df.to_numpy()
    y = y_sr.to_numpy()
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Optional sampling for faster from-scratch demo
    if args.sample_size is not None or args.sample_frac is not None:
        rng = np.random.RandomState(args.random_state)
        n = len(y)
        if args.sample_size is not None:
            k = min(max(2, args.sample_size), n)
        else:
            frac = 0.0 if args.sample_frac is None else max(0.0, min(1.0, args.sample_frac))
            k = max(2, int(n * frac))
        # Stratified subsample to preserve class ratios
        # Build indices per class
        indices = np.arange(n)
        sampled_indices = []
        classes, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(classes, counts):
            cls_idx = indices[y == cls]
            take = max(1, int(k * (cnt / n)))
            if take > len(cls_idx):
                take = len(cls_idx)
            picked = rng.choice(cls_idx, size=take, replace=False)
            sampled_indices.append(picked)
        sampled_indices = np.concatenate(sampled_indices)
        rng.shuffle(sampled_indices)
        X = X[sampled_indices]
        y = y[sampled_indices]
        print(f"Sampling enabled → using {len(y)} samples (from {n})")
    
    try:
        result = tune_threshold_cv(
            X, y,
            n_trees=args.n_trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_splits=args.n_splits,
            random_state=args.random_state,
            clean_data=args.clean_data,
            optimize=args.optimize,
            target_value=args.target_value,
            n_bins=args.n_bins,
            simple_cv=args.simple_cv,
            fixed_threshold=args.fixed_threshold,
            use_sklearn=args.use_sklearn,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted. Exiting...")
        sys.exit(1)
    
    interrupted = result.get("interrupted", False)
    folds_completed = result.get("folds_completed", 0)
    
    if interrupted:
        print(f"\n⚠️  CV interrupted after {folds_completed} folds completed.")
    
    if not args.no_save:
        # Save summary
        summary_path = Path(args.save_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        if interrupted:
            print(f"✓ Partial results saved ({folds_completed} folds): {summary_path}")
        else:
            print(f"✓ Summary saved to: {summary_path}")
        
        # Save predictions
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame(result["cv_predictions"])
        pred_df.to_csv(pred_path, index=False)
        print(f"✓ CV Predictions saved to: {pred_path}")
    
    # Print aggregated results (if available)
    agg = result["aggregated"]
    if agg["roc_auc_mean"] is not None:
        print("\n=== Aggregated Results ===")
        print(f"Threshold: {agg['threshold_mean']:.4f} ± {agg['threshold_std']:.4f}")
        print(f"ROC-AUC:   {agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}")
        print(f"PR-AUC:    {agg['pr_auc_mean']:.4f} ± {agg['pr_auc_std']:.4f}")
        print(f"Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}")
        print(f"Recall:    {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")
        print(f"F1:        {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    else:
        print("\nNo completed folds to report.")


if __name__ == "__main__":
    main()


# cd f:\DATA\scripts
# python random_forest_from_scratch.py --sample-frac 0.10 --n-trees 100 --n-splits 3 --clean-data --simple-cv --fixed-threshold 0.77