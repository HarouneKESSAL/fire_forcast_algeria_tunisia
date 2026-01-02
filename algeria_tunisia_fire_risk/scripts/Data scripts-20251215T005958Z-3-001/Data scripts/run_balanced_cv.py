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
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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


def get_models() -> Dict[str, Pipeline]:
    scaler = StandardScaler()
    models = {
        "knn": Pipeline([
            ("scaler", scaler),
            ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        ]),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }
    return models


def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, object],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: Dict[str, Dict[str, List[float]]] = {}
    for name in models:
        results[name] = {
            "roc_auc": [],
            "pr_auc": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply Tomek+ENN cleaning on training fold only (avoid leakage)
        X_train_clean, y_train_clean = tomek_enn_clean(pd.DataFrame(X_train), pd.Series(y_train))
        X_train_clean = np.asarray(X_train_clean)
        y_train_clean = np.asarray(y_train_clean)

        for name, model in models.items():
            clf = model
            clf.fit(X_train_clean, y_train_clean)
            # Probabilities for AUCs; fall back to decision_function if needed
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                # Map decision scores to [0,1] via rank-based scaling
                scores = clf.decision_function(X_test)
                ranks = (scores - scores.min()) / (scores.ptp() if scores.ptp() > 0 else 1.0)
                y_proba = ranks
            else:
                # Use predictions as pseudo-proba
                y_proba = clf.predict(X_test)

            y_pred = clf.predict(X_test)

            results[name]["roc_auc"].append(roc_auc_score(y_test, y_proba))
            results[name]["pr_auc"].append(average_precision_score(y_test, y_proba))
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="binary", zero_division=0
            )
            results[name]["precision"].append(prec)
            results[name]["recall"].append(rec)
            results[name]["f1"].append(f1)

    # Aggregate mean/std
    aggregated: Dict[str, Dict[str, float]] = {}
    for name, metrics in results.items():
        aggregated[name] = {}
        for m, vals in metrics.items():
            aggregated[name][f"{m}_mean"] = float(np.mean(vals))
            aggregated[name][f"{m}_std"] = float(np.std(vals))
    return aggregated


def main():
    # Default data path; adjust if needed
    data_path = Path("f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv")
    out_dir = Path("f:/DATA/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_df, y_sr = load_data(data_path, target="fire")
    X = X_df.to_numpy()
    y = y_sr.to_numpy()

    models = get_models()
    aggregated = evaluate_cv(X, y, models, n_splits=5, random_state=42)

    # Save JSON
    json_path = out_dir / "tomek_enn_classweights_cv_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # Also save CSV-friendly table
    rows = []
    for name, metrics in aggregated.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "tomek_enn_classweights_cv_metrics.csv"
    df_out.to_csv(csv_path, index=False)

    print(f"Saved metrics to: {json_path}")
    print(f"Saved metrics to: {csv_path}")


if __name__ == "__main__":
    main()
