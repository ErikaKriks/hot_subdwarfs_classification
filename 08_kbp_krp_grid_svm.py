#!/usr/bin/env python3
"""SVM RBF grid experiment for one basis

Outputs go to `results/kbp_krp_grid_SVM_{basis}[_{worker-id}].csv`
Default mode uses fixed params from preliminary HPO
`--run-hpo` switches to per-cell Optuna
Probabilities are calibrated (Platt scaling) and both thresholds are saved

Examples:
    python 08_kbp_krp_grid_svm.py --basis chebyshev
    python 08_kbp_krp_grid_svm.py --basis chebyshev --smoke
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
CLF_NAME = "SVM"
N_TRIALS_DEFAULT = 50
OPTUNA_TIMEOUT_DEFAULT = 300
BSPLINE_MIN_K = 4

# Fixed hyperparameters from preliminary HPO (08_hpo_preliminary.py, 100 trials)
# Per-basis because bspline has distinctly different optimal C/gamma
FIXED_PARAMS_SVM = {
    "chebyshev": {"C": 10, "gamma": 0.001, "class_weight": "balanced"},
    "legendre":  {"C": 10, "gamma": 0.001, "class_weight": "balanced"},
    "bspline":   {"C": 1,  "gamma": 0.03,  "class_weight": "balanced"},
}


# CLI options

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
                   help="Parallelism for CV inside Optuna and "
                        "CalibratedClassifierCV (default: 8)")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                   help=f"Optuna trials per cell (default: {N_TRIALS_DEFAULT})")
    p.add_argument("--timeout", type=int, default=OPTUNA_TIMEOUT_DEFAULT,
                   help=f"Optuna timeout in seconds per cell (default: {OPTUNA_TIMEOUT_DEFAULT})")
    p.add_argument("--fixed-params", type=str, default=None,
                   help="JSON string of fixed hyperparameters (overrides "
                        "the built-in FIXED_PARAMS_SVM for this basis)")
    p.add_argument("--run-hpo", action="store_true",
                   help="Run per-cell Optuna HPO instead of using fixed "
                        "params (the old behaviour before preliminary HPO)")
    p.add_argument("--force-mixed-budget", action="store_true",
                   help="Allow appending rows with a different N_TRIALS "
                        "to an existing CSV (otherwise the script refuses)")
    return p.parse_args()


# SVM routine

def _svm_pipeline(**kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", probability=False,
            random_state=RANDOM_STATE, **kwargs,
        )),
    ])


def _svm_param_fn(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def run_svm(X_tr, y_tr, X_te, y_te, n_trials, timeout, n_jobs):
    """SVM RBF with Optuna TPE + CalibratedClassifierCV for probabilities."""
    # CV optimizes ranking quality before probability calibration
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = _svm_param_fn(trial)
        pipe = _svm_pipeline(**params)
        scores = cross_val_score(
            pipe, X_tr, y_tr, cv=inner_cv,
            scoring="roc_auc", n_jobs=n_jobs,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Stop at n_trials or timeout, whichever comes first
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_trial.params
    best_cv_score = study.best_value
    best_pipe = _svm_pipeline(**best_params)
    best_pipe.fit(X_tr, y_tr)

    # Calibrate probabilities after selecting best raw SVM
    cal_pipe = CalibratedClassifierCV(
        best_pipe, cv=3, method="sigmoid", n_jobs=n_jobs,
    )
    cal_pipe.fit(X_tr, y_tr)

    # Calibrated OOF probs are used only for threshold tuning
    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        cal_pipe, X_tr, y_tr, cv=oof_cv,
        method="predict_proba", n_jobs=n_jobs,
    )[:, 1]

    thr_youden = pick_youden_threshold(y_tr, y_prob_oof)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oof)

    y_prob_te = cal_pipe.predict_proba(X_te)[:, 1]

    metrics_youden = evaluate(y_te, y_prob_te, thr_youden)
    metrics_f1 = evaluate(y_te, y_prob_te, thr_f1)

    params_json = json.dumps({k: json_safe(v) for k, v in best_params.items()})

    for m in (metrics_youden, metrics_f1):
        m["best_cv_roc_auc"] = best_cv_score
        m["best_params"] = params_json

    metrics_youden["threshold_method"] = "youden"
    metrics_f1["threshold_method"] = "f1"

    return metrics_youden, metrics_f1


def run_svm_fixed(X_tr, y_tr, X_te, y_te, fixed_params, n_jobs):
    """SVM with pre-determined hyperparameters — no HPO, calibrated probabilities."""
    # No Optuna in fixed mode, but still calibrate probabilities
    best_pipe = _svm_pipeline(**fixed_params)
    best_pipe.fit(X_tr, y_tr)

    # Keep behavior aligned with HPO mode
    cal_pipe = CalibratedClassifierCV(
        best_pipe, cv=3, method="sigmoid", n_jobs=n_jobs,
    )
    cal_pipe.fit(X_tr, y_tr)

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        cal_pipe, X_tr, y_tr, cv=oof_cv,
        method="predict_proba", n_jobs=n_jobs,
    )[:, 1]

    thr_youden = pick_youden_threshold(y_tr, y_prob_oof)
    thr_f1 = pick_f1_threshold(y_tr, y_prob_oof)

    y_prob_te = cal_pipe.predict_proba(X_te)[:, 1]

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


# Main execution

def main():
    args = parse_args()
    basis = args.basis
    k_bp_values = args.k_bp_values
    k_rp_values = args.k_rp_values
    n_jobs = args.n_jobs

    if args.run_hpo:
        # Per-cell Optuna search path
        fixed_params = None
    elif args.fixed_params is not None:
        # One-off override from CLI JSON
        fixed_params = json.loads(args.fixed_params)
    else:
        # Basis-specific defaults from preliminary search
        fixed_params = FIXED_PARAMS_SVM[basis]

    if args.smoke:
        # Smoke mode trims Optuna budget to finish quickly
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
    # Consistency guard: BP/RP alignment must hold before feature generation
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
        # Keep smoke mode to one repetition (čia tik greitas prasukimas)
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
        # Make mixed-budget runs explicit; easy to miss otherwise
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
    # If this line looks odd, usually CLI args or bspline filtering is wrong
    print()

    if total == 0:
        # Edge case: if resume says done, exit cleanly
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
        # Cache features for this K pair across all folds
        t_feat = time.time()
        X, y = generate_features(step02, bp, rp, basis, k_bp, k_rp)
        feat_seconds = time.time() - t_feat
        print(f"  >> features: K_BP={k_bp} K_RP={k_rp} {basis} -> "
              f"{X.shape[1]}D ({X.shape[0]} samples) in {feat_seconds:.1f}s",
              flush=True)

        for (_, _, sname) in group:
            split = splits_dict[sname]
            # Split arrays index into the cached feature matrix
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            t_cell = time.time()
            if fixed_params:
                metrics_youden, metrics_f1 = run_svm_fixed(
                    X_tr, y_tr, X_te, y_te, fixed_params, n_jobs,
                )
            else:
                metrics_youden, metrics_f1 = run_svm(
                    X_tr, y_tr, X_te, y_te, n_trials, timeout, n_jobs,
                )
            cell_seconds = time.time() - t_cell
            cell_times.append(cell_seconds)

            for metrics in (metrics_youden, metrics_f1):
                # Save both threshold policies for later comparison
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
            # Running ETA from observed per-cell durations
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
