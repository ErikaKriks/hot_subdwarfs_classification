#!/usr/bin/env python3
"""XGBoost sweep on independent (K_BP, K_RP) cells

Runs one basis per launch and writes
`results/kbp_krp_grid_XGB_{basis}[_{worker-id}].csv`

Uses fixed params by default
`--run-hpo` enables per-cell RandomizedSearchCV
Each cell emits two rows: youden and f1 threshold policies

Examples:
    python 08_kbp_krp_grid_xgb.py --basis chebyshev
    python 08_kbp_krp_grid_xgb.py --basis chebyshev --smoke
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

ROOT = Path(__file__).resolve().parent
from _exp08_shared import (  # noqa: E402
    all_threshold_methods_done,
    evaluate,
    generate_features,
    load_completed,
    load_step02,
    pick_f1_threshold,
    pick_youden_threshold,
    print_pr_auc_summary,
)

from _common import (  # noqa: E402
    BP_SAMPLED_CSV,
    DATA_DIR,
    RESULTS_DIR,
    RP_SAMPLED_CSV,
    json_safe,
)

step02 = load_step02(ROOT)

RANDOM_STATE = 42
CLF_NAME = "XGB"
N_ITER = 20
BSPLINE_MIN_K = 4

# Fixed hyperparameters from preliminary HPO (08_hpo_preliminary.py)
# Per-basis because optimal params may differ across bases
# Placeholder values — will be filled after running preliminary HPO
FIXED_PARAMS_XGB = {
    "chebyshev": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.024, "subsample": 0.775, "colsample_bytree": 0.894, "scale_pos_weight": 2},
    "legendre":  {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.028, "subsample": 0.712, "colsample_bytree": 0.876, "scale_pos_weight": 2},
    "bspline":   {"n_estimators": 300, "max_depth": 9, "learning_rate": 0.021, "subsample": 0.648, "colsample_bytree": 0.634, "scale_pos_weight": 2},
}


# CLI

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
                        "the built-in FIXED_PARAMS_XGB for this basis)")
    p.add_argument("--run-hpo", action="store_true",
                   help="Run per-cell RandomizedSearchCV instead of using "
                        "fixed params (the old behaviour)")
    p.add_argument("--force-mixed-budget", action="store_true",
                   help="Allow appending rows with a different N_ITER "
                        "to an existing CSV (otherwise the script refuses)")
    return p.parse_args()


# XGB runner

def run_xgb(X_tr, y_tr, X_te, y_te, n_jobs: int):
    """XGBoost with RandomizedSearchCV (n_iter=20), dual thresholds."""
    # Inner CV tunes params; XGB estimator itself runs single-threaded
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
        pipeline, param_dist, n_iter=N_ITER, cv=inner_cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=n_jobs,
        error_score="raise",
    )
    # Fit search and keep only the best pipeline
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_

    # Use train OOF probs for thresholds; no peeking at test data
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
    params_json = json.dumps({k: json_safe(v) for k, v in best_params.items()})

    for m in (metrics_youden, metrics_f1):
        m["best_cv_roc_auc"] = search.best_score_
        m["best_params"] = params_json

    metrics_youden["threshold_method"] = "youden"
    metrics_f1["threshold_method"] = "f1"

    return metrics_youden, metrics_f1


def run_xgb_fixed(X_tr, y_tr, X_te, y_te, fixed_params, n_jobs):
    """XGB with pre-determined hyperparameters — no HPO, OOF for thresholds."""
    # Fixed mode keeps thresholding comparable with HPO mode
    clf = XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=1, verbosity=0, **fixed_params,
    )
    clf.fit(X_tr, y_tr)

    # Same threshold routine as HPO mode
    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        clf, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=n_jobs,
    )[:, 1]

    thr_youden = pick_youden_threshold(y_tr, y_prob_oof)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oof)

    y_prob_te = clf.predict_proba(X_te)[:, 1]

    metrics_youden = evaluate(y_te, y_prob_te, thr_youden)
    metrics_f1 = evaluate(y_te, y_prob_te, thr_f1)

    params_json = json.dumps({k: json_safe(v) for k, v in fixed_params.items()})

    for m in (metrics_youden, metrics_f1):
        m["best_cv_roc_auc"] = float("nan")
        m["best_params"] = params_json

    metrics_youden["threshold_method"] = "youden"
    metrics_f1["threshold_method"] = "f1"

    return metrics_youden, metrics_f1


# Resume utilities

def _all_threshold_methods_done(completed, k_bp, k_rp, sname, n_iter):
    return all_threshold_methods_done(completed, k_bp, k_rp, sname, n_iter)


# Run loop

def main():
    args = parse_args()
    basis = args.basis
    k_bp_values = args.k_bp_values
    k_rp_values = args.k_rp_values
    n_jobs = args.n_jobs
    n_iter = N_ITER

    if args.run_hpo:
        # Enable per-cell RandomizedSearchCV
        fixed_params = None
    elif args.fixed_params is not None:
        # Optional ad-hoc override from CLI
        fixed_params = json.loads(args.fixed_params)
    else:
        # Default production params by basis
        fixed_params = FIXED_PARAMS_XGB[basis]

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
    # Quick guard: ensure both spectral blocks still align row-wise
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
        # Smoke mode runs only rep0 folds
        splits_dict = {k: v for k, v in all_splits.items()
                       if k.startswith("rep0_")}
        print(f"Splits: {len(splits_dict)} (rep0 only)")
    else:
        splits_dict = all_splits
        print(f"Splits: {len(splits_dict)} (all)")

    split_names = sorted(splits_dict.keys())

    suffix = f"_{args.worker_id}" if args.worker_id else ""
    raw_csv = RESULTS_DIR / f"kbp_krp_grid_{CLF_NAME}_{basis}{suffix}.csv"

    # Accept old XGB runs saved before threshold_method was introduced
    completed = load_completed(raw_csv, allow_legacy_without_threshold=True)
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
    # One work item = (K_BP, K_RP, split)
    for k_bp in k_bp_values:
        for k_rp in k_rp_values:
            for sname in split_names:
                if not _all_threshold_methods_done(completed, k_bp, k_rp, sname, n_iter):
                    work.append((k_bp, k_rp, sname))

    total = len(work)
    print(f"Remaining work: {total} cells")
    print(f"Grid: K_BP={k_bp_values}, K_RP={k_rp_values}")
    # checking
    print()

    if total == 0:
        # Edge case: resume can leave zero pending cells
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
        # Avoid recomputing basis features for each split in the same K pair
        t_feat = time.time()
        X, y = generate_features(step02, bp, rp, basis, k_bp, k_rp)
        feat_seconds = time.time() - t_feat
        print(f"  >> features: K_BP={k_bp} K_RP={k_rp} {basis} -> "
              f"{X.shape[1]}D ({X.shape[0]} samples) in {feat_seconds:.1f}s",
              flush=True)

        for (_, _, sname) in group:
            split = splits_dict[sname]
            # Train/test subsets come from precomputed split indices
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            t_cell = time.time()
            if fixed_params:
                metrics_youden, metrics_f1 = run_xgb_fixed(
                    X_tr, y_tr, X_te, y_te, fixed_params, n_jobs,
                )
            else:
                metrics_youden, metrics_f1 = run_xgb(
                    X_tr, y_tr, X_te, y_te, n_jobs,
                )
            cell_seconds = time.time() - t_cell
            cell_times.append(cell_seconds)

            for metrics in (metrics_youden, metrics_f1):
                # One CSV row per threshold method
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
            # ETA based on current average cell runtime
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
    print_pr_auc_summary(raw_csv, allow_legacy_without_threshold=True)


if __name__ == "__main__":
    main()
