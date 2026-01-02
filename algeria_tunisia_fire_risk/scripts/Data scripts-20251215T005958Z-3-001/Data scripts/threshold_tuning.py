import json
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours


def load_data(csv_path: Path, target: str = "fire") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y


def tomek_enn_clean(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(X, y)
    enn = EditedNearestNeighbours()
    X_clean, y_clean = enn.fit_resample(X_tl, y_tl)
    return X_clean, y_clean


def tune_threshold_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    optimize: str = "f1",  # or "precision_at_recall" or "recall_at_precision"
    target_value: float = 0.80,
) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )

    thresholds: List[float] = []
    fold_metrics: List[Dict[str, float]] = []
    curves: List[Dict[str, List[float]]] = []
    cms: List[Dict[str, int]] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clean only training fold
        X_train_clean, y_train_clean = tomek_enn_clean(pd.DataFrame(X_train), pd.Series(y_train))
        X_train_clean = np.asarray(X_train_clean)
        y_train_clean = np.asarray(y_train_clean)

        rf.fit(X_train_clean, y_train_clean)
        y_proba = rf.predict_proba(X_test)[:, 1]

        # Build curves
        fpr, tpr, roc_thr = roc_curve(y_test, y_proba)
        prec, rec, pr_thr = precision_recall_curve(y_test, y_proba)

        # Determine threshold
        if optimize == "f1":
            # Use PR curve thresholds (exclude last point which has no threshold)
            thr = pr_thr
            if len(thr) == 0:
                # fallback: default 0.5
                best_thr = 0.5
            else:
                f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
                best_idx = int(np.nanargmax(f1_scores))
                best_thr = float(thr[best_idx])
        elif optimize == "precision_at_recall":
            # pick highest precision meeting recall >= target_value
            mask = rec[:-1] >= target_value
            if not np.any(mask):
                best_thr = 0.5
            else:
                cand_prec = prec[:-1][mask]
                cand_thr = pr_thr[mask]
                best_thr = float(cand_thr[int(np.argmax(cand_prec))])
        elif optimize == "recall_at_precision":
            # pick highest recall meeting precision >= target_value
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

    return {
        "thresholds": thresholds,
        "metrics_per_fold": fold_metrics,
        "curves_per_fold": curves,
        "confusion_matrices": cms,
        "aggregated": {
            "threshold_mean": float(np.mean(thresholds)),
            "threshold_std": float(np.std(thresholds)),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
            "pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])),
            "pr_auc_std": float(np.std([m["pr_auc"] for m in fold_metrics])),
            "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
            "precision_std": float(np.std([m["precision"] for m in fold_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
            "recall_std": float(np.std([m["recall"] for m in fold_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
            "f1_std": float(np.std([m["f1"] for m in fold_metrics])),
        },
    }


def main():
    data_path = Path("f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv")
    out_dir = Path("f:/DATA/results/supervised")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_df, y_sr = load_data(data_path, target="fire")
    X = X_df.to_numpy()
    y = y_sr.to_numpy()

    result = tune_threshold_cv(X, y, n_splits=5, random_state=42, optimize="f1")

    json_path = out_dir / "rf_tomek_enn_threshold_tuning.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved threshold tuning outputs to: {json_path}")


if __name__ == "__main__":
    main()
