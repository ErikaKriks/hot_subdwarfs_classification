#!/usr/bin/env python3
"""Production (K_BP, K_RP) grid sweep — Logistic Regression.

Sweeps all combinations of independent BP and RP polynomial orders on a
20x20 grid (K_BP, K_RP in 1..20) with no smoothing (sigma=0).  Each run
processes exactly ONE basis type, making it suitable for parallel dispatch
across HPC accounts.

By default, uses FIXED hyperparameters (per-basis) determined via
preliminary HPO (08_hpo_preliminary.py).  Pass --run-hpo to revert to
per-cell RandomizedSearchCV, or --fixed-params '{"key":val,...}' to
override the built-in set.

Threshold selection: each cell is evaluated twice — once with the
Youden-J-optimal threshold and once with the F1-optimal threshold, both
derived from out-of-fold training predictions.

Run conventions:
    - Each (classifier, basis) pair is one standalone run, intended to be
      assigned to exactly one worker/account.
    - Workers never share output files.  If two workers must run the same
      (classifier, basis) pair (e.g. to split K_BP ranges), use --worker-id
      to disambiguate and split the K_BP list between them.  Merge CSVs
      manually afterwards.
    - All workers MUST use identical splits_rskf.json.  Verify by comparing
      file hashes before distributing.

Grid (defaults):
    K_BP       : 1, 2, ..., 20
    K_RP       : 1, 2, ..., 20
    basis      : one of chebyshev, legendre, bspline  (CLI arg, required)
    classifier : LR  (hard-coded)
    splits     : all 50 from splits_rskf.json (--smoke for rep0 only)
    sigma      : 0 (no smoothing)

Outputs:
    results/kbp_krp_grid_LR_{basis}[_{worker-id}].csv

Usage:
    python 08_kbp_krp_grid_lr.py --basis chebyshev
    python 08_kbp_krp_grid_lr.py --basis legendre --smoke
    python 08_kbp_krp_grid_lr.py --basis chebyshev --run-hpo
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
from scipy.stats import loguniform
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")
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
CLF_NAME = "LR"
N_ITER = 30
BSPLINE_MIN_K = 4

# Fixed hyperparameters from preliminary HPO (08_hpo_preliminary.py).
# Per-basis because optimal C and class_weight differ across bases.
# Placeholder values — will be filled after running preliminary HPO.
FIXED_PARAMS_LR = {
    "chebyshev": {"C": 0.321, "penalty": "l1", "class_weight": "balanced"},
    "legendre":  {"C": 0.241, "penalty": "l1", "class_weight": "balanced"},
    "bspline":   {"C": 0.509, "penalty": "l2", "class_weight": "balanced"},
}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--basis", required=True,
                   choices=["chebyshev", "legendre", "bspline"],
                   help="Basis type (exactly one per run)")
    p.add_argument("--k-bp-values", nargs="+", type=int,
                   default=list(range(1, 21)),
                   help="K_BP values to sweep (default: 1..20)")
    p.add_argument("--k-rp-values", nargs="+", type=int,
                   default=list(range(1, 21)),
                   help="K_RP values to sweep (default: 1..20)")
    p.add_argument("--smoke", action="store_true",
                   help="Rep0 folds only (5 splits) — local sanity check "
                        "before HPC submission")
    p.add_argument("--worker-id", default="",
                   help="Appended to output filename to disambiguate "
                        "parallel workers on the same (clf, basis) pair")
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Parallelism for HPO and CV (default: 8)")
    p.add_argument("--fixed-params", type=str, default=None,
                   help="JSON string of fixed hyperparameters (overrides "
                        "the built-in FIXED_PARAMS_LR for this basis)")
    p.add_argument("--run-hpo", action="store_true",
                   help="Run per-cell RandomizedSearchCV instead of using "
                        "fixed params (the old behaviour)")
    p.add_argument("--force-mixed-budget", action="store_true",
                   help="Allow appending rows with a different N_ITER "
                        "to an existing CSV (otherwise the script refuses)")
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


def pick_f1_threshold(y_true, y_prob, grid_size=200):
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
# Classifier runners
# ═══════════════════════════════════════════════════════════════════════

def _lr_pipeline(**kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, solver="saga",
            random_state=RANDOM_STATE, **kwargs,
        )),
    ])


def run_lr(X_tr, y_tr, X_te, y_te, n_jobs: int):
    """Logistic regression with RandomizedSearchCV (n_iter=30), dual thresholds."""
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
        pipeline, param_dist, n_iter=N_ITER, cv=inner_cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=n_jobs,
    )[:, 1]

    thr_youden = pick_youden_threshold(y_tr, y_prob_oof)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oof)

    y_prob_te = best_pipe.predict_proba(X_te)[:, 1]

    metrics_youden = evaluate(y_te, y_prob_te, thr_youden)
    metrics_f1 = evaluate(y_te, y_prob_te, thr_f1)

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    params_json = json.dumps({k: _json_safe(v) for k, v in best_params.items()})

    for m in (metrics_youden, metrics_f1):
        m["best_cv_roc_auc"] = search.best_score_
        m["best_params"] = params_json

    metrics_youden["threshold_method"] = "youden"
    metrics_f1["threshold_method"] = "f1"

    return metrics_youden, metrics_f1


def run_lr_fixed(X_tr, y_tr, X_te, y_te, fixed_params, n_jobs):
    """LR with pre-determined hyperparameters — no HPO, OOF for thresholds."""
    pipe = _lr_pipeline(**fixed_params)
    pipe.fit(X_tr, y_tr)

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=n_jobs,
    )[:, 1]

    thr_youden = pick_youden_threshold(y_tr, y_prob_oof)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oof)

    y_prob_te = pipe.predict_proba(X_te)[:, 1]

    metrics_youden = evaluate(y_te, y_prob_te, thr_youden)
    metrics_f1 = evaluate(y_te, y_prob_te, thr_f1)

    params_json = json.dumps({k: _json_safe(v) for k, v in fixed_params.items()})

    for m in (metrics_youden, metrics_f1):
        m["best_cv_roc_auc"] = float("nan")
        m["best_params"] = params_json

    metrics_youden["threshold_method"] = "youden"
    metrics_f1["threshold_method"] = "f1"

    return metrics_youden, metrics_f1


def _json_safe(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 6)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _json_safe(val) for k, val in v.items()}
    return v


# ═══════════════════════════════════════════════════════════════════════
# Resume support
# ═══════════════════════════════════════════════════════════════════════

def load_completed(raw_path: Path) -> set[tuple]:
    """Return set of (K_BP, K_RP, split, hpo_n_iter, threshold_method) already done."""
    if not raw_path.exists():
        return set()
    try:
        df = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError:
        print(f"  WARNING: {raw_path.name} exists but is empty (0 bytes). "
              "Treating as fresh start.")
        return set()
    if df.empty:
        return set()
    if "threshold_method" not in df.columns:
        return set(
            df[["K_BP", "K_RP", "split", "hpo_n_iter"]]
            .itertuples(index=False, name=None)
        )
    return set(
        df[["K_BP", "K_RP", "split", "hpo_n_iter", "threshold_method"]]
        .itertuples(index=False, name=None)
    )


def _all_threshold_methods_done(completed, k_bp, k_rp, sname, n_iter):
    """Check whether both threshold rows exist for a cell."""
    return (
        (k_bp, k_rp, sname, n_iter, "youden") in completed
        and (k_bp, k_rp, sname, n_iter, "f1") in completed
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    basis = args.basis
    k_bp_values = args.k_bp_values
    k_rp_values = args.k_rp_values
    n_jobs = args.n_jobs
    n_iter = N_ITER

    if args.run_hpo:
        fixed_params = None
    elif args.fixed_params is not None:
        fixed_params = json.loads(args.fixed_params)
    else:
        fixed_params = FIXED_PARAMS_LR[basis]

    print("=" * 70)
    print(f"  08 - K_BP x K_RP grid sweep  [{CLF_NAME}]")
    if fixed_params:
        print(f"  basis={basis}  FIXED PARAMS (no HPO)  n_jobs={n_jobs}")
        print(f"  params: {json.dumps(fixed_params)}")
    else:
        print(f"  basis={basis}  n_iter={n_iter}  n_jobs={n_jobs}")
    if args.smoke:
        print("  MODE: smoke (rep0 only)")
    print("=" * 70)

    bp = step02.load_block(BP_SAMPLED_CSV)
    rp = step02.load_block(RP_SAMPLED_CSV)
    step02.check_alignment(bp, rp)
    print(f"BP shape: {bp.flux.shape}")
    print(f"RP shape: {rp.flux.shape}")

    splits_path = DATA_DIR / "splits_rskf.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {splits_path}. "
            "Copy from transformation_experiment/data/splits_rskf.json."
        )
    with splits_path.open() as fh:
        all_splits = json.load(fh)

    if args.smoke:
        splits_dict = {k: v for k, v in all_splits.items()
                       if k.startswith("rep0_")}
        print(f"Splits: {len(splits_dict)} (rep0 only)")
    else:
        splits_dict = all_splits
        print(f"Splits: {len(splits_dict)} (all)")

    split_names = sorted(splits_dict.keys())

    suffix = f"_{args.worker_id}" if args.worker_id else ""
    raw_csv = RESULTS_DIR / f"kbp_krp_grid_{CLF_NAME}_{basis}{suffix}.csv"

    completed = load_completed(raw_csv)
    print(f"Already completed: {len(completed)} rows")

    if basis == "bspline":
        bp_before, rp_before = len(k_bp_values), len(k_rp_values)
        k_bp_values = [k for k in k_bp_values if k >= BSPLINE_MIN_K]
        k_rp_values = [k for k in k_rp_values if k >= BSPLINE_MIN_K]
        skipped_bp = bp_before - len(k_bp_values)
        skipped_rp = rp_before - len(k_rp_values)
        if skipped_bp or skipped_rp:
            print(f"  NOTE: bspline (degree 3) requires K >= {BSPLINE_MIN_K}. "
                  f"Dropped {skipped_bp} K_BP and {skipped_rp} K_RP values.")

    work = []
    for k_bp in k_bp_values:
        for k_rp in k_rp_values:
            for sname in split_names:
                if not _all_threshold_methods_done(completed, k_bp, k_rp, sname, n_iter):
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

    done = 0
    cell_times: list[float] = []
    t_start = time.time()

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
            if fixed_params:
                metrics_youden, metrics_f1 = run_lr_fixed(
                    X_tr, y_tr, X_te, y_te, fixed_params, n_jobs,
                )
            else:
                metrics_youden, metrics_f1 = run_lr(
                    X_tr, y_tr, X_te, y_te, n_jobs,
                )
            cell_seconds = time.time() - t_cell
            cell_times.append(cell_seconds)

            for metrics in (metrics_youden, metrics_f1):
                row = {
                    "K_BP": k_bp,
                    "K_RP": k_rp,
                    "basis": basis,
                    "classifier": CLF_NAME,
                    "hpo_n_iter": n_iter,
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
                f"PR-AUC={metrics_youden['pr_auc']:.4f}  "
                f"F1(youden)={metrics_youden['f1']:.4f}  "
                f"F1(f1thr)={metrics_f1['f1']:.4f}  "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                flush=True,
            )

            if done % 50 == 0:
                print(
                    f"  ** CHECKPOINT: {done}/{total} cells done, "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining **",
                    flush=True,
                )

    elapsed_total = time.time() - t_start
    print(f"\nFinished {done} cells in {elapsed_total / 60:.1f} minutes "
          f"(avg {np.mean(cell_times):.1f}s/cell).")
    _print_summary(raw_csv)


def _print_summary(raw_csv: Path) -> None:
    """Print top-10 (K_BP, K_RP) cells by mean PR-AUC, per threshold method."""
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
            top = summary.head(10)
            print(f"\n  Top 10 (K_BP, K_RP) by mean PR-AUC  [threshold={method}]"
                  f"  ({len(summary)} total):")
            print("  " + "-" * 60)
            for _, r in top.iterrows():
                print(f"  K_BP={int(r['K_BP']):3d}  K_RP={int(r['K_RP']):3d}  "
                      f"PR-AUC = {r['mean']:.4f} +/- {r['std']:.4f}")
    else:
        summary = (
            df.groupby(["K_BP", "K_RP"], sort=False)["pr_auc"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        top = summary.head(10)
        print(f"\n  Top 10 (K_BP, K_RP) by mean PR-AUC  ({len(summary)} total):")
        print("  " + "-" * 52)
        for _, r in top.iterrows():
            print(f"  K_BP={int(r['K_BP']):3d}  K_RP={int(r['K_RP']):3d}  "
                  f"PR-AUC = {r['mean']:.4f} +/- {r['std']:.4f}")
    print()


if __name__ == "__main__":
    main()
