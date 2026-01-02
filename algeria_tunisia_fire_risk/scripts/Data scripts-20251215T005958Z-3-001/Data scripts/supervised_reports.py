"""
Generate meaningful plots and quick threshold sweeps from saved supervised results.

Usage (PowerShell):

  # 1) Plot per-fold metrics and (if present) curves for RF
  python scripts/supervised_reports.py --summary f:/DATA/results/rf_tomek_enn_threshold_tuning.json --out f:/DATA/results/plots_rf

  # 2) Plot DT from-scratch metrics (no curves in simple-cv)
  python scripts/supervised_reports.py --summary f:/DATA/results/supervised/dt_from_scratch_summary.json --out f:/DATA/results/plots_dt

  # 3) Threshold sweep on predictions CSV
  python scripts/supervised_reports.py --pred-csv f:/DATA/results/supervised/dt_from_scratch_predictions.csv --sweep "0.70,0.75,0.80,0.90,0.95,1.00" --out f:/DATA/results/plots_dt
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_out(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def plot_fold_metrics(summary: Dict[str, object], outdir: Path, title: str):
    metrics = summary.get("metrics_per_fold", [])
    if not metrics:
        return
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    data = {k: [m.get(k, None) for m in metrics] for k in keys}
    fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys), 4))
    for i, k in enumerate(keys):
        axes[i].bar(range(1, len(data[k])+1), data[k])
        axes[i].set_title(k)
        axes[i].set_xlabel("Fold")
        axes[i].set_ylabel(k)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outdir / "fold_metrics.png", dpi=150)
    plt.close(fig)


def plot_curves(summary: Dict[str, object], outdir: Path):
    curves = summary.get("curves_per_fold", [])
    if not curves:
        return
    # ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    has_any = False
    for i, c in enumerate(curves, start=1):
        if c and "fpr" in c and "tpr" in c:
            ax_roc.plot(c["fpr"], c["tpr"], label=f"Fold {i}")
            has_any = True
    if has_any:
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC Curves (per fold)")
        ax_roc.legend()
        fig_roc.tight_layout()
        fig_roc.savefig(outdir / "roc_curves.png", dpi=150)
    plt.close(fig_roc)

    # PR curves
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    has_any = False
    for i, c in enumerate(curves, start=1):
        if c and "precision" in c and "recall" in c:
            ax_pr.plot(c["recall"], c["precision"], label=f"Fold {i}")
            has_any = True
    if has_any:
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("PR Curves (per fold)")
        ax_pr.legend()
        fig_pr.tight_layout()
        fig_pr.savefig(outdir / "pr_curves.png", dpi=150)
    plt.close(fig_pr)


def threshold_sweep(pred_csv: Path, thresholds: List[float]) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    y_true = df["y_true"].values
    y_proba = df["y_proba"].values
    rows = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, default=None, help="Path to summary JSON (optional).")
    p.add_argument("--pred-csv", type=str, default=None, help="Path to predictions CSV (optional).")
    p.add_argument("--sweep", type=str, default=None, help="Comma-separated thresholds for sweep (e.g., 0.70,0.75,0.80,0.90,0.95,1.00).")
    p.add_argument("--out", type=str, required=True, help="Output directory for plots and sweep results.")
    args = p.parse_args()

    outdir = Path(args.out)
    ensure_out(outdir)

    if args.summary:
        summary = json.loads(Path(args.summary).read_text())
        title = f"Metrics: {Path(args.summary).name}"
        plot_fold_metrics(summary, outdir, title)
        plot_curves(summary, outdir)

        # Save aggregated metrics table
        agg = summary.get("aggregated", {})
        pd.DataFrame([agg]).to_csv(outdir / "aggregated_metrics.csv", index=False)

    if args.pred_csv and args.sweep:
        thresholds = [float(x) for x in args.sweep.split(",")]
        df = threshold_sweep(Path(args.pred_csv), thresholds)
        df.to_csv(outdir / "threshold_sweep.csv", index=False)
        # Plot F1 vs threshold
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(df["threshold"], df["f1"], marker="o")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_title("F1 vs Threshold")
        fig.tight_layout()
        fig.savefig(outdir / "f1_vs_threshold.png", dpi=150)
        plt.close(fig)

    print(f"✓ Reports saved in: {outdir}")


if __name__ == "__main__":
    main()
