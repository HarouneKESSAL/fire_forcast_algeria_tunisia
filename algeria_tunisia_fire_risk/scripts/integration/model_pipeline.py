#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
)
from sklearn.cluster import KMeans

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

from scripts.config import get_base


def _load_merged(path: Path | None = None) -> pd.DataFrame:
    base = get_base()
    # Primary expected location (legacy): base/processed/merged_dataset.parquet
    # Actual current pipeline output (from merge_env): base/DATA_CLEANED/processed/merged_dataset.parquet
    # Allow explicit path override; else probe common candidates.
    if path is not None:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        alt_csv = path.with_suffix('.csv')
        if alt_csv.exists():
            return pd.read_csv(alt_csv)
        raise FileNotFoundError("merged_dataset not found at explicit path " + str(path))

    candidates = [
        base / "processed" / "merged_dataset.parquet",
        base / "processed" / "merged_dataset.csv",
        base / "DATA_CLEANED" / "processed" / "merged_dataset.parquet",
        base / "DATA_CLEANED" / "processed" / "merged_dataset.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return pd.read_parquet(cand) if cand.suffix == '.parquet' else pd.read_csv(cand)
    raise FileNotFoundError("merged_dataset not found. Checked: " + ", ".join(str(c) for c in candidates))


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    y = df["fire"].astype(int)

    # Basic coordinate features retained for distance-based models; remove if leakage suspected
    base_drop = {"fire"}
    # Remove any columns that are non-informative identifiers
    leak_like = {"dt", "month", "country"}

    # Identify categorical vs numeric
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for c in df.columns:
        if c in base_drop or c in leak_like:
            continue
        if c in ("latitude", "longitude"):
            num_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    X = df[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[(
        "onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)
    )])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return pre


def _spatial_groups(lat: np.ndarray, lon: np.ndarray, n_groups: int = 5, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
    coords = np.column_stack([lat, lon])
    return km.fit_predict(coords)


def evaluate_models(n_splits: int = 5, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    df = _load_merged()
    X, y, num_cols, cat_cols = _select_features(df)
    pre = _make_preprocessor(num_cols, cat_cols)

    # Spatial GroupKFold by clustering lat/lon
    if {"latitude", "longitude"}.issubset(df.columns):
        groups = _spatial_groups(df["latitude"].values, df["longitude"].values, n_groups=n_splits)
        gkf = GroupKFold(n_splits=n_splits)
        splits = gkf.split(X, y, groups)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = skf.split(X, y)

    results: Dict[str, List[Dict[str, float]]] = {"knn": [], "rf": [], "brf": []}

    # Pipelines
    knn_pipe = ImbPipeline(steps=[
        ("pre", pre),
        ("balance", SMOTETomek(random_state=random_state)),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", metric="minkowski", p=2)),
    ])

    rf_pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state, class_weight=None)),
    ])

    brf_pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", BalancedRandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state)),
    ])

    def _metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "f1_minority": float(f1_score(y_true, y_pred, pos_label=1)),
            "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        }

    for tr_idx, te_idx in splits:
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # k-NN
        knn_pipe.fit(X_tr, y_tr)
        prob_knn = knn_pipe.predict_proba(X_te)[:, 1]
        pred_knn = (prob_knn >= 0.5).astype(int)
        results["knn"].append(_metrics(y_te, prob_knn, pred_knn))

        # RF
        rf_pipe.fit(X_tr, y_tr)
        prob_rf = rf_pipe.predict_proba(X_te)[:, 1]
        pred_rf = (prob_rf >= 0.5).astype(int)
        results["rf"].append(_metrics(y_te, prob_rf, pred_rf))

        # Balanced RF
        brf_pipe.fit(X_tr, y_tr)
        prob_brf = brf_pipe.predict_proba(X_te)[:, 1]
        pred_brf = (prob_brf >= 0.5).astype(int)
        results["brf"].append(_metrics(y_te, prob_brf, pred_brf))

    # Aggregate
    summary: Dict[str, Dict[str, float]] = {}
    for model, rows in results.items():
        if not rows:
            continue
        dfm = pd.DataFrame(rows)
        summary[model] = {
            **{f"mean_{c}": float(dfm[c].mean()) for c in dfm.columns},
            **{f"std_{c}": float(dfm[c].std()) for c in dfm.columns},
        }
    return summary


def save_results(summary: Dict[str, Dict[str, float]]) -> Path:
    base = get_base()
    out = base / "processed" / "model_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


def main():
    summary = evaluate_models()
    out = save_results(summary)
    print("Model evaluation saved to", out)


if __name__ == "__main__":
    main()
