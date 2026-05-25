#!/usr/bin/env python3
"""RF grid run for one basis at a time

Writes `results/kbp_krp_grid_RF_{basis}[_{worker-id}].csv`
Default path uses fixed params from preliminary HPO
Optional path (`--run-hpo`) uses per-cell Optuna
Thresholds are selected from OOB probabilities and saved as youden + f1 rows

Examples:
    python 08_kbp_krp_grid_rf.py --basis chebyshev
    python 08_kbp_krp_grid_rf.py --basis chebyshev --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
CLF_NAME = "RF"
N_TRIALS_DEFAULT = 30
OPTUNA_TIMEOUT_DEFAULT = 300
BSPLINE_MIN_K = 4

# Fixed hyperparameters from preliminary HPO (08_hpo_preliminary.py, 100 trials)
# Stable across K values and all 3 bases (55 cells pooled)
FIXED_PARAMS_RF = {
    "n_estimators": 300,
    "max_depth": 20,
    "max_features": 0.3,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "class_weight": None,
    "criterion": "gini",
}


# CLI setup

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
                   help="Rep0 folds only (5 splits), reduced HPO budget "
                        "— local sanity check before HPC submission")
    p.add_argument("--worker-id", default="",
                   help="Appended to output filename to disambiguate "
                        "parallel workers on the same (clf, basis) pair")
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Parallelism for CV inside Optuna (default: 8)")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                   help=f"Optuna trials per cell (default: {N_TRIALS_DEFAULT})")
    p.add_argument("--timeout", type=int, default=OPTUNA_TIMEOUT_DEFAULT,
                   help=f"Optuna timeout in seconds per cell (default: {OPTUNA_TIMEOUT_DEFAULT})")
    p.add_argument("--fixed-params", type=str, default=None,
                   help="JSON string of fixed hyperparameters (overrides "
                        "the built-in FIXED_PARAMS_RF)")
    p.add_argument("--run-hpo", action="store_true",
                   help="Run per-cell Optuna HPO instead of using fixed "
                        "params (the old behaviour before preliminary HPO)")
    p.add_argument("--force-mixed-budget", action="store_true",
                   help="Allow appending rows with a different N_TRIALS "
                        "to an existing CSV (otherwise the script refuses)")
    return p.parse_args()


# RF model routine

def _rf_model(**kwargs):
    return RandomForestClassifier(
        oob_score=True, bootstrap=True,
        random_state=RANDOM_STATE, n_jobs=-1, **kwargs,
    )


def _rf_param_fn(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500, 700]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def run_rf(X_tr, y_tr, X_te, y_te, n_trials, timeout, n_jobs):
    """Random Forest with Optuna TPE, using OOB for validation & thresholds."""

    def objective(trial):
        # Each trial trains one RF and scores OOB ROC-AUC
        params = _rf_param_fn(trial)
        clf = _rf_model(**params)
        clf.fit(X_tr, y_tr)
        # OOB score is cheap and works well here
        y_oob = clf.oob_decision_function_[:, 1]
        return roc_auc_score(y_tr, y_oob)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Budget is controlled by (n_trials, timeout)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_trial.params
    best_oob_score = study.best_value

    best_clf = _rf_model(**best_params)
    best_clf.fit(X_tr, y_tr)

    # Reuse OOB probabilities for threshold selection (no extra CV)
    y_prob_oob = best_clf.oob_decision_function_[:, 1]
    thr_youden = pick_youden_threshold(y_tr, y_prob_oob)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oob)

    y_prob_te = best_clf.predict_proba(X_te)[:, 1]

    metrics_youden = evaluate(y_te, y_prob_te, thr_youden)
    metrics_f1 = evaluate(y_te, y_prob_te, thr_f1)

    params_json = json.dumps({k: json_safe(v) for k, v in best_params.items()})

    for m in (metrics_youden, metrics_f1):
        m["best_cv_roc_auc"] = best_oob_score
        m["best_params"] = params_json

    metrics_youden["threshold_method"] = "youden"
    metrics_f1["threshold_method"] = "f1"

    return metrics_youden, metrics_f1


def run_rf_fixed(X_tr, y_tr, X_te, y_te, fixed_params):
    """RF with pre-determined hyperparameters — no HPO, OOB for thresholds."""
    # Fixed mode still uses OOB for threshold calibration
    clf = _rf_model(**fixed_params)
    clf.fit(X_tr, y_tr)

    y_prob_oob = clf.oob_decision_function_[:, 1]
    thr_youden = pick_youden_threshold(y_tr, y_prob_oob)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oob)

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

def _all_threshold_methods_done(completed, k_bp, k_rp, sname, n_trials):
    return all_threshold_methods_done(completed, k_bp, k_rp, sname, n_trials)


# Execution loop

def main():
    args = parse_args()
    basis = args.basis
    k_bp_values = args.k_bp_values
    k_rp_values = args.k_rp_values
    n_jobs = args.n_jobs

    if args.run_hpo:
        # Per-cell Optuna search
        fixed_params = None
    elif args.fixed_params is not None:
        # Manual JSON override (patogu greitam testui)
        fixed_params = json.loads(args.fixed_params)
    else:
        # Default from preliminary HPO
        fixed_params = FIXED_PARAMS_RF

    if args.smoke:
        # Keep smoke runs short regardless of CLI budget
        n_trials = 5
        timeout = 60
    else:
        n_trials = args.n_trials
        timeout = args.timeout

    print("=" * 70)
    print(f"  08 - K_BP x K_RP grid sweep  [{CLF_NAME}]")
    if fixed_params:
        print(f"  basis={basis}  FIXED PARAMS (no HPO)  n_jobs={n_jobs}")
        print(f"  params: {json.dumps(fixed_params)}")
    else:
        print(f"  basis={basis}  n_trials={n_trials}  timeout={timeout}s  n_jobs={n_jobs}")
    if args.smoke:
        print("  MODE: smoke (rep0 only, 5 trials, 60s timeout)")
    print("=" * 70)

    bp = step02.load_block(BP_SAMPLED_CSV)
    rp = step02.load_block(RP_SAMPLED_CSV)
    # Consistency guard: detect BP/RP row drift early
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
        # Only rep0 folds for quick local run-through
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

    if completed:
        # Guard against mixing different HPO budgets in one CSV
        existing_budgets = {t[3] for t in completed}
        foreign = existing_budgets - {n_trials}
        if foreign:
            foreign_counts = {b: sum(1 for t in completed if t[3] == b)
                              for b in foreign}
            current_count = sum(1 for t in completed if t[3] == n_trials)
            parts = [f"{cnt} rows with hpo_n_iter={int(b)}"
                     for b, cnt in sorted(foreign_counts.items())]
            parts.append(f"{current_count} rows with current n_trials={n_trials}")
            msg = (f"WARNING: CSV contains mixed HPO budgets: "
                   f"{', '.join(parts)}.\n"
                   f"         New rows will be appended with n_trials={n_trials}. "
                   f"Summary will mix budgets.")
            if args.force_mixed_budget:
                print(f"  {msg}")
                print("  --force-mixed-budget is set, proceeding anyway.")
            else:
                print(f"\n  {msg}")
                print("  Refusing to proceed. Pass --force-mixed-budget to override.\n")
                sys.exit(1)

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
                if not _all_threshold_methods_done(completed, k_bp, k_rp, sname, n_trials):
                    work.append((k_bp, k_rp, sname))

    total = len(work)
    print(f"Remaining work: {total} cells")
    print(f"Grid: K_BP={k_bp_values}, K_RP={k_rp_values}")
    # Mano praktika: prieš paleidžiant ilgą run dar kartą pasižiūriu šitą eilutę
    print()

    if total == 0:
        # Edge case: no remaining cells means resume is complete
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
        # Feature extraction is expensive; reuse across splits
        t_feat = time.time()
        X, y = generate_features(step02, bp, rp, basis, k_bp, k_rp)
        feat_seconds = time.time() - t_feat
        print(f"  >> features: K_BP={k_bp} K_RP={k_rp} {basis} -> "
              f"{X.shape[1]}D ({X.shape[0]} samples) in {feat_seconds:.1f}s",
              flush=True)

        for (_, _, sname) in group:
            split = splits_dict[sname]
            # Apply split indices to cached feature matrix
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            t_cell = time.time()
            if fixed_params:
                metrics_youden, metrics_f1 = run_rf_fixed(
                    X_tr, y_tr, X_te, y_te, fixed_params,
                )
            else:
                metrics_youden, metrics_f1 = run_rf(
                    X_tr, y_tr, X_te, y_te, n_trials, timeout, n_jobs,
                )
            cell_seconds = time.time() - t_cell
            cell_times.append(cell_seconds)

            for metrics in (metrics_youden, metrics_f1):
                # Store threshold variants as separate rows
                row = {
                    "K_BP": k_bp,
                    "K_RP": k_rp,
                    "basis": basis,
                    "classifier": CLF_NAME,
                    "hpo_n_iter": n_trials,
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
            # ETA from observed cell durations
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
    print_pr_auc_summary(raw_csv)


if __name__ == "__main__":
    main()
