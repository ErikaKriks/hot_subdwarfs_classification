#!/usr/bin/env python3
"""Smoke test: independent K_BP x K_RP basis-expansion grid.

Verifies the classification pipeline end-to-end on a tiny grid before
committing to the full multi-day run.  BP and RP polynomial orders are
varied INDEPENDENTLY (2-D grid), no smoothing is applied.

Grid (defaults):
    K_BP       : [5, 10, 20]
    K_RP       : [5, 10, 20]
    basis      : one of chebyshev, legendre, bspline  (CLI arg)
    classifier : one of LR, XGB                       (CLI arg)
    splits     : rep0 only (5-fold RSKF)

Outputs:
    results/smoke_kbp_krp_{clf}_{basis}.csv   -- one row per cell

Usage:
    python smoke_07_kbp_krp_grid.py --clf LR  --basis chebyshev
    python smoke_07_kbp_krp_grid.py --clf XGB --basis legendre
    python smoke_07_kbp_krp_grid.py --clf LR  --basis bspline \\
           --k-bp-values 5 10 --k-rp-values 5 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*max_iter was reached.*")

ROOT = Path(__file__).resolve().parent
sys.modules.pop("bp_basis_step02", None)
_spec = spec_from_file_location("bp_basis_step02", ROOT / "02_generate_basis_features.py")
step02 = module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = step02
_spec.loader.exec_module(step02)

from _common import (  # noqa: E402
    BP_SAMPLED_CSV,
    DATA_DIR,
    RESULTS_DIR,
    RP_SAMPLED_CSV,
    l2_normalize,
)

RANDOM_STATE = 42
N_JOBS = 8


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clf", required=True, choices=["LR", "XGB"],
                   help="Classifier to run (one only)")
    p.add_argument("--basis", required=True,
                   choices=["chebyshev", "legendre", "bspline"],
                   help="Basis type (one only)")
    p.add_argument("--k-bp-values", nargs="+", type=int, default=[5, 10, 20],
                   help="K_BP values to sweep (default: 5 10 20)")
    p.add_argument("--k-rp-values", nargs="+", type=int, default=[5, 10, 20],
                   help="K_RP values to sweep (default: 5 10 20)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Feature generation
# ═══════════════════════════════════════════════════════════════════════

def flatten_feature_blocks_asym(
    source_ids: np.ndarray,
    labels: np.ndarray,
    bp_coeffs: np.ndarray,
    rp_coeffs: np.ndarray,
) -> pd.DataFrame:
    """Like _common.flatten_feature_blocks but allows K_BP != K_RP."""
    bp_coeffs = np.asarray(bp_coeffs, dtype=float)
    rp_coeffs = np.asarray(rp_coeffs, dtype=float)
    if bp_coeffs.shape[0] != rp_coeffs.shape[0]:
        raise ValueError(
            f"Row count mismatch: BP {bp_coeffs.shape[0]} vs RP {rp_coeffs.shape[0]}"
        )
    total_cols = bp_coeffs.shape[1] + rp_coeffs.shape[1]
    columns = [f"c{i:03d}" for i in range(total_cols)]
    stacked = np.hstack([bp_coeffs, rp_coeffs])
    out = pd.DataFrame(stacked, columns=columns)
    out.insert(0, "y", np.asarray(labels, dtype=int))
    out.insert(0, "source_id", np.asarray(source_ids))
    return out


def generate_features(bp, rp, basis: str, K_BP: int, K_RP: int):
    """Fit basis separately for BP/RP, concatenate, L2-normalise."""
    bp_fit = step02.build_block_fit(bp, basis, "none", K_BP)
    rp_fit = step02.build_block_fit(rp, basis, "none", K_RP)
    feat_df = flatten_feature_blocks_asym(
        bp.source_ids, bp.labels, bp_fit.coeffs, rp_fit.coeffs,
    )
    coeff_cols = [c for c in feat_df.columns if c.startswith("c")]
    feat_df = l2_normalize(feat_df, coeff_cols=coeff_cols)
    X = feat_df[coeff_cols].to_numpy(dtype=np.float64)
    y = feat_df["y"].astype(int).to_numpy()
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════

def pick_youden_threshold(y_true, y_prob, grid_size=200):
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


def evaluate(y_true, y_prob, threshold):
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


# ═══════════════════════════════════════════════════════════════════════
# Classifier runners  (n_iter=10 for smoke test)
# ═══════════════════════════════════════════════════════════════════════

def run_lr(X_tr, y_tr, X_te, y_te):
    """Logistic regression with RandomizedSearchCV (n_iter=10, smoke)."""
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    param_dist = {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga"],
        "clf__class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=10, cv=inner_cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=N_JOBS,
    )[:, 1]
    thr = pick_youden_threshold(y_tr, y_prob_oof)

    y_prob_te = best_pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)
    metrics["best_cv_roc_auc"] = search.best_score_
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    metrics["best_params"] = json.dumps({k: _json_safe(v) for k, v in best_params.items()})
    return metrics


def run_xgb(X_tr, y_tr, X_te, y_te):
    """XGBoost with RandomizedSearchCV (n_iter=10, smoke)."""
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("clf", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=1, verbosity=0,
        )),
    ])
    param_dist = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [3, 5, 7, 10],
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__scale_pos_weight": [1, 3, 4],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=10, cv=inner_cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=N_JOBS,
    )[:, 1]
    thr = pick_youden_threshold(y_tr, y_prob_oof)

    y_prob_te = best_pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)
    metrics["best_cv_roc_auc"] = search.best_score_
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    metrics["best_params"] = json.dumps({k: _json_safe(v) for k, v in best_params.items()})
    return metrics


def _json_safe(v):
    if isinstance(v, (np.integer, np.int64)):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return round(float(v), 6)
    return v


CLF_RUNNERS = {
    "LR": run_lr,
    "XGB": run_xgb,
}


# ═══════════════════════════════════════════════════════════════════════
# Resume support
# ═══════════════════════════════════════════════════════════════════════

def load_completed(raw_path: Path) -> set[tuple]:
    """Return set of (K_BP, K_RP, split) already done."""
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    return set(
        df[["K_BP", "K_RP", "split"]].itertuples(index=False, name=None)
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    clf_name = args.clf.upper()
    basis = args.basis
    k_bp_values = args.k_bp_values
    k_rp_values = args.k_rp_values

    print("=" * 70)
    print("  Smoke test: K_BP x K_RP grid")
    print(f"  classifier={clf_name}  basis={basis}")
    print("=" * 70)

    # Load spectra
    bp = step02.load_block(BP_SAMPLED_CSV)
    rp = step02.load_block(RP_SAMPLED_CSV)
    step02.check_alignment(bp, rp)
    print(f"BP shape: {bp.flux.shape}")
    print(f"RP shape: {rp.flux.shape}")

    # Load splits — rep0 only
    splits_path = DATA_DIR / "splits_rskf.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {splits_path}. "
            "Copy from transformation_experiment/data/splits_rskf.json."
        )
    with splits_path.open() as fh:
        all_splits = json.load(fh)

    splits_dict = {k: v for k, v in all_splits.items() if k.startswith("rep0_")}
    split_names = sorted(splits_dict.keys())
    print(f"Splits: {len(split_names)} (rep0 only)")

    # Output path
    raw_csv = RESULTS_DIR / f"smoke_kbp_krp_{clf_name}_{basis}.csv"

    # Resume
    completed = load_completed(raw_csv)
    print(f"Already completed: {len(completed)} cells")

    # Build work list: (K_BP, K_RP, split)
    work = []
    for k_bp in k_bp_values:
        for k_rp in k_rp_values:
            for sname in split_names:
                if (k_bp, k_rp, sname) not in completed:
                    work.append((k_bp, k_rp, sname))

    total = len(work)
    print(f"Remaining work: {total} cells")
    print(f"Grid: K_BP={k_bp_values}, K_RP={k_rp_values}")
    print()

    if total == 0:
        print("Nothing to do.")
        _print_summary(raw_csv)
        return

    csv_header_written = raw_csv.exists() and raw_csv.stat().st_size > 0
    runner = CLF_RUNNERS[clf_name]

    done = 0
    cell_times: list[float] = []
    t_start = time.time()

    # Group by (K_BP, K_RP) to generate features once per cell
    work.sort(key=lambda x: (x[0], x[1]))
    for (k_bp, k_rp), group_iter in groupby(work, key=lambda x: (x[0], x[1])):
        group = list(group_iter)
        t_feat = time.time()
        X, y = generate_features(bp, rp, basis, k_bp, k_rp)
        feat_seconds = time.time() - t_feat
        print(f"  >> features: K_BP={k_bp} K_RP={k_rp} {basis} -> "
              f"{X.shape[1]}D ({X.shape[0]} samples) in {feat_seconds:.1f}s",
              flush=True)

        for (_, _, sname) in group:
            split = splits_dict[sname]
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            t_cell = time.time()
            metrics = runner(X_tr, y_tr, X_te, y_te)
            cell_seconds = time.time() - t_cell
            cell_times.append(cell_seconds)

            row = {
                "K_BP": k_bp,
                "K_RP": k_rp,
                "basis": basis,
                "classifier": clf_name,
                "split": sname,
                **metrics,
            }

            row_df = pd.DataFrame([row])
            row_df.to_csv(
                raw_csv, mode="a", header=not csv_header_written, index=False,
            )
            csv_header_written = True

            done += 1
            elapsed = time.time() - t_start
            avg_cell = np.mean(cell_times)
            eta = avg_cell * (total - done)

            print(
                f"  [{done}/{total}] K_BP={k_bp} K_RP={k_rp} split={sname}  "
                f"PR-AUC={metrics['pr_auc']:.4f}  "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                flush=True,
            )

    elapsed_total = time.time() - t_start
    print(f"\nFinished {done} cells in {elapsed_total / 60:.1f} minutes "
          f"(avg {np.mean(cell_times):.1f}s/cell).")
    _print_summary(raw_csv)


def _print_summary(raw_csv: Path) -> None:
    """Print compact groupby summary: mean/std PR-AUC per (K_BP, K_RP)."""
    if not raw_csv.exists():
        return
    df = pd.read_csv(raw_csv)
    if "pr_auc" not in df.columns:
        return
    summary = (
        df.groupby(["K_BP", "K_RP"], sort=False)["pr_auc"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    print("\n  PR-AUC summary (mean +/- std) per (K_BP, K_RP):")
    print("  " + "-" * 48)
    for _, r in summary.iterrows():
        print(f"  K_BP={int(r['K_BP']):3d}  K_RP={int(r['K_RP']):3d}  "
              f"PR-AUC = {r['mean']:.4f} +/- {r['std']:.4f}")
    print()


if __name__ == "__main__":
    main()
