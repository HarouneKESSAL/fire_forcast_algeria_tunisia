"""
KNN (k-Nearest Neighbors) from scratch for classification.

- Loads engineered features dataset (default: f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv)
- Selects top-N variance features excluding the target (default target: 'fire')
- Standardizes features using train split statistics (no leakage)
- Implements KNN prediction without scikit-learn (pure NumPy for distances and voting)
- Evaluates on a stratified train/test split and optionally sweeps a k-grid
- Optionally saves predictions to results/supervised/knn_predictions.csv

Usage (PowerShell):

  python scripts/knn_from_scratch.py \
    --data f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv \
    --target fire --features-top 20 --metric manhattan --k 11 --test-size 0.2 --seed 42

Or sweep k values:

  python scripts/knn_from_scratch.py --k-grid 3,5,7,9,11,15

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Optional progress helper (uses tqdm if available, else no-op)
def _progress(total: int, desc: str = ""):
    try:
        from tqdm import tqdm  # type: ignore

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KNN (from scratch) classifier")
    p.add_argument("--data", type=str, default=str(Path("f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv")),
                   help="Path to input CSV with features and target.")
    p.add_argument("--target", type=str, default="fire", help="Target column name for classification.")
    p.add_argument("--features-top", type=int, default=20, help="Use top-N variance features (excluding target).")
    p.add_argument("--metric", type=str, default="manhattan", choices=["euclidean", "manhattan"],
                   help="Distance metric for KNN.")
    p.add_argument("--k", type=int, default=11, help="Number of neighbors (used if --k-grid not provided).")
    p.add_argument("--k-grid", type=str, default=None,
                   help="Comma-separated list of k values to sweep (e.g., '3,5,7,9,11').")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction for train/test split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save-preds", type=str, default=str(Path("f:/DATA/results/supervised/knn_predictions.csv")),
                   help="Path to save predictions CSV.")
    p.add_argument("--save-summary", type=str, default=str(Path("f:/DATA/results/supervised/knn_from_scratch_summary.json")),
                   help="Path to save results summary JSON.")
    p.add_argument("--no-save", action="store_true", help="Do not save predictions CSV or summary JSON.")
    return p.parse_args()


def select_features_by_variance(df: pd.DataFrame, target: str, top_n: int) -> List[str]:
    cols = [c for c in df.columns if c != target]
    variances = df[cols].var().sort_values(ascending=False)
    return list(variances.head(top_n).index)


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Return mean, std for columns
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _distances(train: np.ndarray, x: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        # sqrt(sum((train - x)**2, axis=1))
        d = train - x
        return np.sqrt(np.einsum("ij,ij->i", d, d))
    else:
        # manhattan
        return np.sum(np.abs(train - x), axis=1)


def _vote_majority(classes: np.ndarray) -> int:
    # classes is a 1D array of neighbor labels
    vals, counts = np.unique(classes, return_counts=True)
    # break ties by choosing the class with smallest value (stable)
    return int(vals[np.argmax(counts)])


def knn_predict(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, k: int, metric: str,
                batch_size: int = 256) -> np.ndarray:
    n_test = test_X.shape[0]
    preds = np.empty(n_test, dtype=train_y.dtype)
    with _progress(total=n_test, desc="KNN predict") as pbar:
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            for i in range(start, end):
                d = _distances(train_X, test_X[i], metric)
                # partial sort to get k smallest indices
                idx = np.argpartition(d, kth=min(k, len(d)-1))[:k]
                neighbor_classes = train_y[idx]
                preds[i] = _vote_majority(neighbor_classes)
            pbar.update(end - start)
    return preds


def evaluate_knn(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def save_results_summary(result: dict, summary_path: Optional[Path]) -> None:
    """Save KNN results summary (metrics + metadata) to JSON for comparison."""
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {summary_path}")


def run_single(df: pd.DataFrame, target: str, top_n: int, metric: str, k: int, test_size: float, seed: int,
               save_path: Optional[Path], save_summary_path: Optional[Path] = None) -> dict:
    feature_cols = select_features_by_variance(df, target, top_n)
    X = df[feature_cols].values
    y = df[target].values

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) <= 20 else None
    )

    # scale using train stats only
    mean, std = standardize_fit(X_train)
    X_train_s = standardize_transform(X_train, mean, std)
    X_test_s = standardize_transform(X_test, mean, std)

    # predict
    y_pred = knn_predict(X_train_s, y_train, X_test_s, k=k, metric=metric)

    # eval
    metrics = evaluate_knn(y_test, y_pred)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        })
        out_df.to_csv(save_path, index=False)

    result = {
        "k": int(k),
        "metric": metric,
        "features": feature_cols,
        "metrics": metrics,
        "test_size": test_size,
        "seed": seed,
    }
    
    save_results_summary(result, save_summary_path)

    return result


def run_grid(df: pd.DataFrame, target: str, top_n: int, metric: str, k_list: List[int], test_size: float, seed: int,
             save_summary_path: Optional[Path] = None) -> dict:
    results = []
    best = None
    best_f1 = -1.0
    with _progress(total=len(k_list), desc="K sweep") as pbar:
        for k in k_list:
            print(f"Evaluating k={k} ...")
            res = run_single(df, target, top_n, metric, k, test_size, seed, save_path=None, save_summary_path=None)
            results.append(res)
            f1 = res["metrics"]["f1_macro"]
            if f1 > best_f1:
                best_f1 = f1
                best = res
            pbar.update(1)
    grid_result = {"best": best, "results": results}
    save_results_summary(grid_result, save_summary_path)
    return grid_result


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data.")

    # Drop rows with missing target or NaNs in candidate features
    df = df.dropna(subset=[args.target])

    summary_path = None if args.no_save else Path(args.save_summary)
    pred_path = None if args.no_save else Path(args.save_preds)

    if args.k_grid:
        k_list = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]
        grid_res = run_grid(df, args.target, args.features_top, args.metric, k_list, args.test_size, args.seed,
                            save_summary_path=summary_path)
        best = grid_res["best"]
        print("\nBest by f1_macro:")
        print(json.dumps(best, indent=2))
        # Save predictions for best k
        final = run_single(df, args.target, args.features_top, args.metric, best["k"], args.test_size, args.seed,
                          save_path=pred_path, save_summary_path=None)
        print("\nFinal (saved) run:")
        print(json.dumps(final, indent=2))
    else:
        res = run_single(df, args.target, args.features_top, args.metric, args.k, args.test_size, args.seed,
                        save_path=pred_path, save_summary_path=summary_path)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
