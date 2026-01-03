"""
K-Means clustering from scratch (Lloyd's algorithm).

- Loads engineered features dataset (default: f:/DATA/DATA_CLEANED/processed/engineered_features_tomek_enn_balanced.csv)
- Selects top-N variance features excluding the target (default target: 'fire')
- Standardizes features using full dataset statistics
- Implements K-Means (random init, assign, update centroid loop) without scikit-learn
- Evaluates on k-grid sweep with silhouette, Davies-Bouldin, Calinski-Harabasz metrics
- Optionally saves cluster labels and results summary

Usage (PowerShell):

  python scripts/model_training/kmeans_from_scratch.py \
    --data f:/DATA/DATA_CLEANED/processed/engineered_features_tomek_enn_balanced.csv \
    --target fire --features-top 20 --k-grid 2,3,4,5,6 --seed 42

Or single k:

  python scripts/model_training/kmeans_from_scratch.py --k 2 --seed 42

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-Means (from scratch) clustering")
    p.add_argument("--data", type=str, default=str(Path("f:/DATA/DATA_CLEANED/processed/engineered_features_tomek_enn_balanced.csv")),
                   help="Path to input CSV with features.")
    p.add_argument("--target", type=str, default="fire", help="Target column to exclude from features.")
    p.add_argument("--features-top", type=int, default=20, help="Use top-N variance features (excluding target).")
    p.add_argument("--k", type=int, default=None, help="Number of clusters (overrides --k-grid if set).")
    p.add_argument("--k-grid", type=str, default="2,4,6",
                   help="Comma-separated list of k values to sweep (default: '2,4,6').")
    p.add_argument("--max-iter", type=int, default=300, help="Maximum iterations for K-Means convergence.")
    p.add_argument("--tol", type=float, default=1e-4, help="Tolerance for centroid shift (convergence criterion).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    p.add_argument("--save-labels", type=str, default=str(Path("f:/DATA/results/unsupervised/kmeans_from_scratch_labels.csv")),
                   help="Path to save cluster labels CSV.")
    p.add_argument("--save-summary", type=str, default=str(Path("f:/DATA/results/unsupervised/kmeans_from_scratch_summary.json")),
                   help="Path to save results summary JSON.")
    p.add_argument("--no-save", action="store_true", help="Do not save labels or summary.")
    return p.parse_args()


def select_features_by_variance(df: pd.DataFrame, target: str, top_n: int) -> List[str]:
    cols = [c for c in df.columns if c != target]
    variances = df[cols].var().sort_values(ascending=False)
    return list(variances.head(top_n).index)


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


def kmeans_init_centroids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Initialize k centroids by random selection from data points."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def kmeans_assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid (Euclidean distance)."""
    # X: (n, d), centroids: (k, d)
    # distances: (n, k)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
    distances = np.sqrt(np.einsum("nkd,nkd->nk", diff, diff))  # (n, k)
    return np.argmin(distances, axis=1)  # (n,)


def kmeans_update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Update centroids as mean of assigned points. Return new centroids and max shift."""
    centroids_old = np.zeros((k, X.shape[1]))
    centroids_new = np.zeros((k, X.shape[1]))
    
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            centroids_new[i] = X[mask].mean(axis=0)
        else:
            # If a cluster is empty, keep old centroid (or reinit)
            centroids_new[i] = centroids_old[i]
    
    # Compute max shift (for convergence check)
    shift = np.linalg.norm(centroids_new - centroids_old, axis=1).max()
    return centroids_new, shift


def kmeans_fit(X: np.ndarray, k: int, max_iter: int = 300, tol: float = 1e-4, seed: int = 42) -> dict:
    """Run K-Means and return centroids and labels."""
    centroids = kmeans_init_centroids(X, k, seed)
    labels = kmeans_assign(X, centroids)
    
    with _progress(total=max_iter, desc=f"K-Means (k={k})") as pbar:
        for iteration in range(max_iter):
            centroids, shift = kmeans_update_centroids(X, labels, k)
            if shift < tol:
                pbar.update(max_iter - iteration)  # Update remaining
                break
            labels = kmeans_assign(X, centroids)
            pbar.update(1)
    
    return {
        "centroids": centroids,
        "labels": labels,
        "iterations": iteration + 1,
    }


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute silhouette, Davies-Bouldin, and Calinski-Harabasz scores."""
    m = {}
    try:
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            m["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        m["silhouette"] = None
    
    try:
        if len(set(labels)) > 1:
            m["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            m["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        m["davies_bouldin"] = None
        m["calinski_harabasz"] = None
    
    return m


def compute_inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """Compute sum of squared distances to nearest centroid."""
    diff = X - centroids[labels]
    return float(np.sum(diff ** 2))


def run_single(df: pd.DataFrame, target: str, top_n: int, k: int, max_iter: int, tol: float, seed: int,
               save_labels_path: Optional[Path] = None, save_summary_path: Optional[Path] = None) -> dict:
    """Run single K-Means and optionally save outputs."""
    feature_cols = select_features_by_variance(df, target, top_n)
    X = df[feature_cols].values
    
    # Scale
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # Fit
    fit_result = kmeans_fit(X_s, k=k, max_iter=max_iter, tol=tol, seed=seed)
    centroids = fit_result["centroids"]
    labels = fit_result["labels"]
    
    # Metrics
    metrics = compute_metrics(X_s, labels)
    inertia = compute_inertia(X_s, centroids, labels)
    
    # Save labels
    if save_labels_path is not None:
        save_labels_path = Path(save_labels_path)
        save_labels_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cluster_label": labels}).to_csv(save_labels_path, index=False)
    
    result = {
        "k": int(k),
        "features": feature_cols,
        "centroids": centroids.tolist(),
        "labels": list(map(int, labels)),
        "metrics": metrics,
        "inertia": inertia,
        "iterations": fit_result["iterations"],
        "max_iter": max_iter,
        "tol": tol,
        "seed": seed,
    }
    
    # Save summary
    if save_summary_path is not None:
        save_summary_path = Path(save_summary_path)
        save_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Summary saved to: {save_summary_path}")
    
    return result


def run_grid(df: pd.DataFrame, target: str, top_n: int, k_list: List[int], max_iter: int, tol: float, seed: int,
             save_summary_path: Optional[Path] = None) -> dict:
    """Run K-Means sweep over k values."""
    results = []
    best = None
    best_sil = -1.0
    
    with _progress(total=len(k_list), desc="K-Means k sweep") as pbar:
        for k in k_list:
            res = run_single(df, target, top_n, k, max_iter, tol, seed,
                           save_labels_path=None, save_summary_path=None)
            results.append(res)
            sil = res["metrics"].get("silhouette")
            if sil is not None and sil > best_sil:
                best_sil = sil
                best = res
            pbar.update(1)
    
    grid_result = {"best": best, "results": results}
    
    # Save summary
    if save_summary_path is not None:
        save_summary_path = Path(save_summary_path)
        save_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_summary_path, "w") as f:
            json.dump(grid_result, f, indent=2)
        print(f"Grid summary saved to: {save_summary_path}")
    
    return grid_result


def main():
    args = parse_args()
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data.")
    
    # Drop rows with missing target
    df = df.dropna(subset=[args.target])
    
    summary_path = None if args.no_save else Path(args.save_summary)
    labels_path = None if args.no_save else Path(args.save_labels)
    
    # If --k is explicitly set, use single k; otherwise use --k-grid (default: 2,4,6)
    if args.k is not None:
        res = run_single(df, args.target, args.features_top, args.k, args.max_iter, args.tol, args.seed,
                        save_labels_path=labels_path, save_summary_path=summary_path)
        print(json.dumps(res, indent=2))
    else:
        k_list = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]
        grid_res = run_grid(df, args.target, args.features_top, k_list, args.max_iter, args.tol, args.seed,
                           save_summary_path=summary_path)
        best = grid_res["best"]
        print("\nBest by silhouette:")
        print(json.dumps(best, indent=2))
        
        # Save best labels
        if labels_path is not None:
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"cluster_label": best["labels"]}).to_csv(labels_path, index=False)
            print(f"Best labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
