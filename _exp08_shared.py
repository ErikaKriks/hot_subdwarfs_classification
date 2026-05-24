from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from _common import flatten_feature_blocks_asym, json_safe, l2_normalize


def load_step02(root: Path):
    """Load 02_generate_basis_features.py as an isolated module."""
    sys.modules.pop("bp_basis_step02", None)
    spec = spec_from_file_location("bp_basis_step02", root / "02_generate_basis_features.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_features(step02, bp, rp, basis: str, k_bp: int, k_rp: int):
    """Fit basis separately for BP/RP, concatenate, L2-normalize."""
    bp_fit = step02.build_block_fit(bp, basis, "none", k_bp)
    rp_fit = step02.build_block_fit(rp, basis, "none", k_rp)
    feat_df = flatten_feature_blocks_asym(
        bp.source_ids, bp.labels, bp_fit.coeffs, rp_fit.coeffs
    )
    coeff_cols = [c for c in feat_df.columns if c.startswith("c")]
    feat_df = l2_normalize(feat_df, coeff_cols=coeff_cols)
    x = feat_df[coeff_cols].to_numpy(dtype=np.float64)
    y = feat_df["y"].astype(int).to_numpy()
    return x, y


def pick_youden_threshold(y_true, y_prob, grid_size: int = 200) -> float:
    thresholds = np.linspace(0, 1, grid_size)
    best_j, best_thr = -1.0, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def pick_f1_threshold(y_true, y_prob, grid_size: int = 200) -> float:
    thresholds = np.linspace(0, 1, grid_size)
    best_f1, best_thr = -1.0, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
        "f1": f1,
        "youden_j": sens + spec - 1.0,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def load_completed(
    raw_path: Path,
    *,
    allow_legacy_without_threshold: bool = False,
) -> set[tuple]:
    """Return completed rows keyed by K_BP/K_RP/split/hpo_n_iter/threshold_method."""
    if not raw_path.exists():
        return set()
    try:
        df = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError:
        print(
            f"  WARNING: {raw_path.name} exists but is empty (0 bytes). "
            "Treating as fresh start."
        )
        return set()
    if df.empty:
        return set()
    if "threshold_method" not in df.columns:
        if allow_legacy_without_threshold:
            return set(
                df[["K_BP", "K_RP", "split", "hpo_n_iter"]].itertuples(
                    index=False, name=None
                )
            )
        raise KeyError("threshold_method")
    return set(
        df[["K_BP", "K_RP", "split", "hpo_n_iter", "threshold_method"]].itertuples(
            index=False, name=None
        )
    )


def all_threshold_methods_done(
    completed: set[tuple], k_bp: int, k_rp: int, split: str, hpo_n_iter: int
) -> bool:
    return (
        (k_bp, k_rp, split, hpo_n_iter, "youden") in completed
        and (k_bp, k_rp, split, hpo_n_iter, "f1") in completed
    )


def print_pr_auc_summary(
    raw_csv: Path,
    *,
    allow_legacy_without_threshold: bool = False,
    top_n: int = 10,
) -> None:
    """Print top-N (K_BP, K_RP) cells by mean PR-AUC."""
    if not raw_csv.exists():
        return
    df = pd.read_csv(raw_csv)
    if "pr_auc" not in df.columns:
        return

    if "threshold_method" in df.columns:
        for method in df["threshold_method"].unique():
            sub = df[df["threshold_method"] == method]
            summary = (
                sub.groupby(["K_BP", "K_RP"], sort=False)["pr_auc"]
                .agg(["mean", "std"])
                .sort_values("mean", ascending=False)
                .reset_index()
            )
            top = summary.head(top_n)
            print(
                f"\n  Top {top_n} (K_BP, K_RP) by mean PR-AUC  [threshold={method}]"
                f"  ({len(summary)} total):"
            )
            print("  " + "-" * 60)
            for _, row in top.iterrows():
                print(
                    f"  K_BP={int(row['K_BP']):3d}  K_RP={int(row['K_RP']):3d}  "
                    f"PR-AUC = {row['mean']:.4f} +/- {row['std']:.4f}"
                )
        print()
        return

    if not allow_legacy_without_threshold:
        raise KeyError("threshold_method")

    summary = (
        df.groupby(["K_BP", "K_RP"], sort=False)["pr_auc"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    top = summary.head(top_n)
    print(f"\n  Top {top_n} (K_BP, K_RP) by mean PR-AUC  ({len(summary)} total):")
    print("  " + "-" * 52)
    for _, row in top.iterrows():
        print(
            f"  K_BP={int(row['K_BP']):3d}  K_RP={int(row['K_RP']):3d}  "
            f"PR-AUC = {row['mean']:.4f} +/- {row['std']:.4f}"
        )
    print()
