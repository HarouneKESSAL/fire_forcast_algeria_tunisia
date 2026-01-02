"""
Decision Tree classifier from scratch with stratified CV, optional Tomek+ENN cleaning,
sampling for quick demos, and simple-cv fixed-threshold evaluation.

Outputs:
- Summary JSON: f:/DATA/results/supervised/dt_from_scratch_summary.json
- Predictions CSV: f:/DATA/results/supervised/dt_from_scratch_predictions.csv
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
    p = argparse.ArgumentParser(description="Decision Tree (from scratch) with CV and simple evaluation")
    p.add_argument("--data", type=str, default=str(Path("f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv")),
                   help="Path to input CSV with features.")
    p.add_argument("--target", type=str, default="fire", help="Target column.")
    p.add_argument("--max-depth", type=int, default=15, help="Max depth per tree.")
    p.add_argument("--min-samples-leaf", type=int, default=5, help="Min samples per leaf.")
    p.add_argument("--n-splits", type=int, default=5, help="CV folds.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--clean-data", action="store_true", help="Apply Tomek+ENN cleaning to training folds.")
    p.add_argument("--simple-cv", action="store_true", help="Skip curve-based threshold tuning (fast).")
    p.add_argument("--fixed-threshold", type=float, default=0.77, help="Threshold for simple-cv mode.")
    p.add_argument("--sample-size", type=int, default=None, help="Fixed sample size for quick runs.")
    p.add_argument("--sample-frac", type=float, default=None, help="Fraction of dataset to sample for speed.")
    p.add_argument("--save-summary", type=str, default=str(Path("f:/DATA/results/supervised/dt_from_scratch_summary.json")),
                   help="Path to save summary JSON.")
    p.add_argument("--save-predictions", type=str, default=str(Path("f:/DATA/results/supervised/dt_from_scratch_predictions.csv")),
                   help="Path to save CV predictions CSV.")
    p.add_argument("--no-save", action="store_true", help="Do not save results.")
    return p.parse_args()


def _progress(total: int, desc: str = ""):
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
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class DecisionTree:
    def __init__(self, max_depth: int = 15, min_samples_leaf: int = 5, n_bins: int = 256):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.root = None
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        node = DecisionTreeNode()
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < 2 * self.min_samples_leaf):
            node.value = int(np.bincount(y).argmax())
            return node
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_vals = np.unique(feature_values)
            if len(unique_vals) == 1:
                continue
            if len(unique_vals) > self.n_bins:
                thresholds = np.percentile(feature_values, np.linspace(0, 100, self.n_bins))
                thresholds = np.unique(thresholds)
            else:
                thresholds = unique_vals
            for threshold in thresholds:
                left = feature_values <= threshold
                right = ~left
                n_left, n_right = left.sum(), right.sum()
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                y_left, y_right = y[left], y[right]
                gini = (n_left * self._gini(y_left) + n_right * self._gini(y_right)) / n_samples
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            node.value = int(np.bincount(y).argmax())
            return node
        
        mask = X[:, best_feature] <= best_threshold
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def _gini(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _traverse(self, x: np.ndarray, node: DecisionTreeNode) -> DecisionTreeNode:
        if node.feature is None:
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), 2))
        for i, x in enumerate(X):
            leaf = self._traverse(x, self.root)
            p1 = leaf.value if leaf.value is not None else 0
            proba[i] = [1 - p1, p1]
        return proba


def tomek_enn_clean(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(X, y)
    enn = EditedNearestNeighbours()
    X_clean, y_clean = enn.fit_resample(X_tl, y_tl)
    return np.asarray(X_clean), np.asarray(y_clean)


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    max_depth: int,
    min_samples_leaf: int,
    simple_cv: bool,
    fixed_threshold: float,
    clean_data: bool,
) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    thresholds: List[float] = []
    fold_metrics: List[Dict[str, float]] = []
    curves: List[Optional[Dict[str, List[float]]]] = []
    cms: List[Dict[str, int]] = []
    cv_predictions: List[Dict[str, object]] = []

    fold_idx = 0
    interrupted = False
    try:
        with _progress(total=n_splits, desc="CV Fold Progress") as pbar_folds:
            for train_idx, test_idx in skf.split(X, y):
                fold_idx += 1
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if clean_data:
                    X_train, y_train = tomek_enn_clean(pd.DataFrame(X_train), pd.Series(y_train))

                dt = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_bins=256)
                dt.fit(X_train, y_train)

                with _progress(total=len(X_test), desc=f"  Fold {fold_idx}/{n_splits} predictions") as pbar_pred:
                    y_proba = dt.predict_proba(X_test)[:, 1]
                    pbar_pred.update(len(X_test))

                if simple_cv:
                    thr = float(fixed_threshold)
                    y_pred = (y_proba >= thr).astype(int)
                    roc_auc = roc_auc_score(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
                        y_test, y_pred, average="binary", zero_division=0
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    thresholds.append(thr)
                    fold_metrics.append({
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                        "precision": float(prec_b),
                        "recall": float(rec_b),
                        "f1": float(f1_b),
                        "threshold": float(thr),
                    })
                    curves.append(None)
                    cms.append({
                        "tn": int(cm[0, 0]),
                        "fp": int(cm[0, 1]),
                        "fn": int(cm[1, 0]),
                        "tp": int(cm[1, 1]),
                    })
                else:
                    # Full mode: build curves and choose threshold by F1
                    fpr, tpr, roc_thr = roc_curve(y_test, y_proba)
                    prec, rec, pr_thr = precision_recall_curve(y_test, y_proba)
                    if len(pr_thr) == 0:
                        thr = 0.5
                    else:
                        f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
                        best_idx = int(np.nanargmax(f1_scores))
                        thr = float(pr_thr[best_idx])
                    y_pred = (y_proba >= thr).astype(int)
                    roc_auc = roc_auc_score(y_test, y_proba)
                    pr_auc = average_precision_score(y_test, y_proba)
                    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
                        y_test, y_pred, average="binary", zero_division=0
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    thresholds.append(thr)
                    fold_metrics.append({
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                        "precision": float(prec_b),
                        "recall": float(rec_b),
                        "f1": float(f1_b),
                        "threshold": float(thr),
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

                for idx, (true, pred, prob) in enumerate(zip(y_test, y_pred, y_proba)):
                    cv_predictions.append({
                        "fold": fold_idx,
                        "index": int(test_idx[idx]),
                        "y_true": int(true),
                        "y_pred": int(pred),
                        "y_proba": float(prob),
                        "threshold": float(thr),
                    })

                pbar_folds.update(1)
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⚠️  Interrupted by user (Ctrl+C). Saving current results...")

    return {
        "model_config": {
            "model": "DecisionTree",
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "clean_data": clean_data,
            "simple_cv": simple_cv,
            "fixed_threshold": float(fixed_threshold),
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

    df = df.dropna(subset=[args.target])
    X_df = df.drop(columns=[args.target])
    y_sr = df[args.target].astype(int)
    X = X_df.to_numpy()
    y = y_sr.to_numpy()

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Optional sampling
    if args.sample_size is not None or args.sample_frac is not None:
        rng = np.random.RandomState(args.random_state)
        n = len(y)
        if args.sample_size is not None:
            k = min(max(2, args.sample_size), n)
        else:
            frac = 0.0 if args.sample_frac is None else max(0.0, min(1.0, args.sample_frac))
            k = max(2, int(n * frac))
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
        result = run_cv(
            X=X,
            y=y,
            n_splits=args.n_splits,
            random_state=args.random_state,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            simple_cv=args.simple_cv,
            fixed_threshold=args.fixed_threshold,
            clean_data=args.clean_data,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted. Exiting...")
        sys.exit(1)

    interrupted = result.get("interrupted", False)
    folds_completed = result.get("folds_completed", 0)
    if interrupted:
        print(f"\n⚠️  CV interrupted after {folds_completed} folds completed.")

    if not args.no_save:
        summary_path = Path(args.save_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✓ Summary saved to: {summary_path}")

        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame(result["cv_predictions"])
        pred_df.to_csv(pred_path, index=False)
        print(f"✓ CV Predictions saved to: {pred_path}")

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



