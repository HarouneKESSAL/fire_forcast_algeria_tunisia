"""
DBSCAN clustering from scratch (density-based spatial clustering, optimized).

- Loads engineered features dataset (default: f:/DATA/DATA_CLEANED/processed/engineered_features_tomek_enn_balanced.csv)
- Selects top-N variance features excluding the target (default target: 'fire')
- Standardizes features using full dataset statistics
- Implements DBSCAN (vectorized neighborhoods, core points, BFS clustering) without scikit-learn
- Evaluates on eps/min_samples grid sweep with silhouette, Davies-Bouldin, Calinski-Harabasz metrics
- Optionally saves cluster labels CSV and results summary JSON
- Gracefully handles Ctrl+C interrupts with partial result saving

Usage (PowerShell):

  python scripts/model_training/dbscan_from_scratch.py \
    --eps-grid 0.5,0.8,1.0,1.5,2.0 --min-samples-grid 5,10,15,20 \
    --save-labels f:/DATA/results/unsupervised/dbscan_labels.csv \
    --save-summary f:/DATA/results/unsupervised/dbscan_summary.json

Or single eps/min_samples:

  python scripts/model_training/dbscan_from_scratch.py --eps 2.0 --min-samples 20 \
    --save-labels f:/DATA/results/unsupervised/dbscan_labels.csv \
    --save-summary f:/DATA/results/unsupervised/dbscan_summary.json

Skip saving:

  python scripts/model_training/dbscan_from_scratch.py --no-save

"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, adjusted_mutual_info_score,
                             homogeneity_score, completeness_score)
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DBSCAN (from scratch) clustering")
    p.add_argument("--data", type=str, default=str(Path("f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv")),
                   help="Path to input CSV with features.")
    p.add_argument("--target", type=str, default="fire", help="Target column to exclude from features (and optionally use for supervised metrics).")
    p.add_argument("--features-top", type=int, default=20, help="Use top-N variance features (excluding target).")
    p.add_argument("--eps", type=float, default=None, help="Eps radius (used if --eps-grid not provided).")
    p.add_argument("--min-samples", type=int, default=20, help="Min samples for core point (default: 20).")
    p.add_argument("--metric", type=str, default="manhattan", choices=["euclidean", "manhattan"],
                   help="Distance metric.")
    p.add_argument("--eps-grid", type=str, default="0.5,0.8,1.0,1.5,2.0",
                   help="Comma-separated eps values to sweep (default: '0.5,0.8,1.0,1.5,2.0').")
    p.add_argument("--min-samples-grid", type=str, default="5,10,15,20",
                   help="Comma-separated min_samples values to sweep (default: '5,10,15,20').")
    p.add_argument("--use-supervised", action="store_true",
                   help="Compute supervised metrics (ARI, AMI, purity) if target column available.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save-labels", type=str, default=str(Path("f:/DATA/results/unsupervised/dbscan_from_scratch_labels.csv")),
                   help="Path to save cluster labels CSV.")
    p.add_argument("--save-summary", type=str, default=str(Path("f:/DATA/results/unsupervised/dbscan_from_scratch_summary.json")),
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


def _distance(x1: np.ndarray, x2: np.ndarray, metric: str = "manhattan") -> float:
    """Compute distance between two points."""
    if metric == "euclidean":
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))
    elif metric == "manhattan":
        return float(np.sum(np.abs(x1 - x2)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _distances_vectorized(X: np.ndarray, point: np.ndarray, metric: str = "manhattan") -> np.ndarray:
    """Compute distances from a point to all rows in X (vectorized for speed)."""
    if metric == "euclidean":
        diff = X - point  # Broadcasting: (n, d)
        return np.sqrt(np.sum(diff ** 2, axis=1))  # (n,)
    elif metric == "manhattan":
        diff = X - point
        return np.sum(np.abs(diff), axis=1)  # (n,)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _neighborhood(X: np.ndarray, point_idx: int, eps: float, metric: str) -> Set[int]:
    """Find all neighbors of a point within eps distance (vectorized)."""
    x_point = X[point_idx]
    distances = _distances_vectorized(X, x_point, metric)
    neighbors = set(np.where(distances <= eps)[0])
    return neighbors


def dbscan_fit(X: np.ndarray, eps: float, min_samples: int, metric: str = "manhattan") -> np.ndarray:
    """
    Run DBSCAN and return cluster labels.
    
    Algorithm:
    1. Find neighbors for each point (distance <= eps)
    2. Mark core points (>= min_samples neighbors including itself)
    3. Form clusters by connecting core points
    4. Assign border points to nearby clusters
    5. Mark remaining as noise (-1)
    """
    n = len(X)
    labels = np.full(n, -1, dtype=int)  # -1 = noise
    cluster_id = 0
    
    # Precompute neighborhoods (caching)
    neighborhoods = {}
    with _progress(total=n, desc=f"DBSCAN neighborhoods (eps={eps}, ms={min_samples})") as pbar:
        for i in range(n):
            neighborhoods[i] = _neighborhood(X, i, eps, metric)
            pbar.update(1)
    
    # Find core points
    core_points = {i for i in range(n) if len(neighborhoods[i]) >= min_samples}
    
    # BFS to form clusters (using deque for efficiency)
    visited = set()
    for point_idx in range(n):
        if point_idx in visited or point_idx not in core_points:
            continue
        
        # Start new cluster from this core point
        queue = deque([point_idx])
        visited.add(point_idx)
        labels[point_idx] = cluster_id
        
        # BFS expand cluster
        while queue:
            current = queue.popleft()  # Much faster than list.pop(0)
            for neighbor_idx in neighborhoods[current]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    labels[neighbor_idx] = cluster_id
                    # Only expand from core points
                    if neighbor_idx in core_points:
                        queue.append(neighbor_idx)
        
        cluster_id += 1
    
    return labels


def compute_metrics(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None) -> dict:
    """
    Compute clustering metrics (unsupervised + optionally supervised).
    
    Unsupervised: silhouette, davies_bouldin, calinski_harabasz
    Supervised (if y_true provided): ARI, AMI, homogeneity, completeness, purity
    """
    m = {}
    
    # Unsupervised metrics
    try:
        n_clusters = len(set(labels) - {-1})
        if n_clusters > 1 and n_clusters < len(labels):
            m["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        m["silhouette"] = None
    
    try:
        n_clusters = len(set(labels) - {-1})
        if n_clusters > 1 and n_clusters < len(labels):
            m["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            m["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        m["davies_bouldin"] = None
        m["calinski_harabasz"] = None
    
    # Supervised metrics (if true labels provided)
    if y_true is not None:
        try:
            m["ARI"] = float(adjusted_rand_score(y_true, labels))
            m["AMI"] = float(adjusted_mutual_info_score(y_true, labels, average_method='arithmetic'))
            m["homogeneity"] = float(homogeneity_score(y_true, labels))
            m["completeness"] = float(completeness_score(y_true, labels))
        except Exception:
            m["ARI"] = None
            m["AMI"] = None
            m["homogeneity"] = None
            m["completeness"] = None
        
        # Purity: sum of max class counts per cluster / total
        try:
            n_clusters = len(set(labels) - {-1})
            if n_clusters > 0:
                cluster_purities = []
                for c in set(labels):
                    if c == -1:  # Skip noise
                        continue
                    members = y_true[labels == c]
                    if len(members) == 0:
                        continue
                    most_common = np.bincount(members.astype(int)).max()
                    cluster_purities.append(most_common)
                m["purity"] = float(np.sum(cluster_purities) / len(labels)) if cluster_purities else None
            else:
                m["purity"] = None
        except Exception:
            m["purity"] = None
    
    return m


def run_single(df: pd.DataFrame, target: str, top_n: int, eps: float, min_samples: int, metric: str,
               save_labels_path: Optional[Path] = None, save_summary_path: Optional[Path] = None,
               y_true: Optional[np.ndarray] = None) -> dict:
    """Run single DBSCAN and optionally save outputs."""
    feature_cols = select_features_by_variance(df, target, top_n)
    X = df[feature_cols].values
    
    # Scale
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # Fit
    labels = dbscan_fit(X_s, eps=eps, min_samples=min_samples, metric=metric)
    
    # Metrics
    metrics = compute_metrics(X_s, labels, y_true=y_true)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    
    # Save labels
    if save_labels_path is not None:
        save_labels_path = Path(save_labels_path)
        save_labels_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cluster_label": labels}).to_csv(save_labels_path, index=False)
    
    result = {
        "eps": float(eps),
        "min_samples": int(min_samples),
        "metric": metric,
        "features": feature_cols,
        "labels": list(map(int, labels)),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_rate": float(n_noise / len(labels)),
        "metrics": metrics,
    }
    
    # Save summary
    if save_summary_path is not None:
        save_summary_path = Path(save_summary_path)
        save_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Summary saved to: {save_summary_path}")
    
    return result


def run_grid(df: pd.DataFrame, target: str, top_n: int, eps_list: List[float], min_samples_list: List[int],
             metric: str, save_summary_path: Optional[Path] = None, y_true: Optional[np.ndarray] = None) -> dict:
    """Run DBSCAN sweep over eps and min_samples grids. Gracefully handles Ctrl+C interrupts."""
    results = []
    best = None
    best_score = -np.inf
    
    total_runs = len(eps_list) * len(min_samples_list)
    interrupted = False
    
    try:
        with _progress(total=total_runs, desc="DBSCAN grid sweep") as pbar:
            for eps in eps_list:
                for min_samples in min_samples_list:
                    res = run_single(df, target, top_n, eps, min_samples, metric,
                                   save_labels_path=None, save_summary_path=None, y_true=y_true)
                    results.append(res)
                    
                    # Improved scoring: prefer reasonable clusters (2-15), low noise, good metrics
                    n_clusters = res["n_clusters"]
                    noise_rate = res["noise_rate"]
                    sil = res["metrics"].get("silhouette")
                    ari = res["metrics"].get("ARI")
                    
                    # Primary score: use ARI if available (supervised), else silhouette
                    if ari is not None:
                        primary_score = ari
                    else:
                        primary_score = sil if sil is not None else 0
                    
                    # Penalize too many clusters or excessive noise
                    cluster_penalty = max(0, (n_clusters - 10) * 0.01) if n_clusters > 10 else 0
                    noise_penalty = max(0, (noise_rate - 0.3) * 0.5) if noise_rate > 0.3 else 0
                    score = primary_score - cluster_penalty - noise_penalty
                    
                    if score > best_score:
                        best_score = score
                        best = res
                    
                    pbar.update(1)
    
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⚠️  Interrupted by user (Ctrl+C). Saving current results...")
    
    # Always save, even if interrupted
    grid_result = {"best": best, "results": results, "interrupted": interrupted, "total_completed": len(results)}
    
    # Save summary
    if save_summary_path is not None:
        save_summary_path = Path(save_summary_path)
        save_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_summary_path, "w") as f:
            json.dump(grid_result, f, indent=2)
        if interrupted:
            print(f"✓ Partial results saved ({len(results)} runs completed): {save_summary_path}")
        else:
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
    
    # Optionally load true labels for supervised metrics
    y_true = None
    if args.use_supervised and args.target in df.columns:
        y_true = df[args.target].values
    
    summary_path = None if args.no_save else Path(args.save_summary)
    labels_path = None if args.no_save else Path(args.save_labels)
    
    # If --eps is explicitly set, use single eps/min_samples; otherwise use grid defaults
    if args.eps is not None:
        res = run_single(df, args.target, args.features_top, args.eps, args.min_samples, args.metric,
                        save_labels_path=labels_path, save_summary_path=summary_path, y_true=y_true)
        print(json.dumps(res, indent=2))
    else:
        eps_list = [float(x.strip()) for x in args.eps_grid.split(",") if x.strip()]
        min_samples_list = [int(x.strip()) for x in args.min_samples_grid.split(",") if x.strip()]
        
        try:
            grid_res = run_grid(df, args.target, args.features_top, eps_list, min_samples_list, args.metric,
                               save_summary_path=summary_path, y_true=y_true)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted. Exiting...")
            sys.exit(1)
        
        best = grid_res["best"]
        interrupted = grid_res.get("interrupted", False)
        completed = grid_res.get("total_completed", 0)
        
        if interrupted:
            print(f"\n⚠️  Grid sweep interrupted after {completed} runs.")
        
        if best is not None:
            print("\nBest result so far:")
            print(json.dumps(best, indent=2))
            
            # Save best labels
            if labels_path is not None:
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"cluster_label": best["labels"]}).to_csv(labels_path, index=False)
                print(f"\n✓ Best labels saved to: {labels_path}")
        else:
            print("No valid clusters found in any run.")


if __name__ == "__main__":
    main()
