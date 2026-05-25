#!/usr/bin/env python3
"""Preliminary HPO search for RF, SVM, LR, and XGB at representative K values.

Runs Optuna TPE at 4 symmetric K pairs (K_BP=K_RP in {3,8,15,20}) using
rep0 splits (5 folds) to identify stable hyperparameters.  If the best
params are consistent across K values, they can be fixed for the full
20x20 grid sweep, reducing compute by 30-40x.

Outputs:
    results/hpo_preliminary_{CLF}_{basis}.csv
    + console summary of parameter stability

Usage:
    python 08_hpo_preliminary.py --clf rf  --basis chebyshev
    python 08_hpo_preliminary.py --clf svm --basis chebyshev
    python 08_hpo_preliminary.py --clf lr  --basis chebyshev
    python 08_hpo_preliminary.py --clf xgb --basis chebyshev
    python 08_hpo_preliminary.py --clf lr  --basis chebyshev --n-trials 80
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
from _exp08_shared import generate_features, load_step02  # noqa: E402

from _common import (  # noqa: E402
    BP_SAMPLED_CSV,
    DATA_DIR,
    RESULTS_DIR,
    RP_SAMPLED_CSV,
    json_safe,
)

step02 = load_step02(ROOT)

RANDOM_STATE = 42
REPRESENTATIVE_K = [3, 8, 15, 20]
BSPLINE_MIN_K = 4


# CLI args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clf", required=True, choices=["rf", "svm", "lr", "xgb"],
                   help="Classifier to tune")
    p.add_argument("--basis", required=True,
                   choices=["chebyshev", "legendre", "bspline"],
                   help="Basis type")
    p.add_argument("--n-trials", type=int, default=100,
                   help="Optuna trials per cell (default: 100)")
    p.add_argument("--timeout", type=int, default=600,
                   help="Optuna timeout in seconds per cell (default: 600)")
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Parallelism (default: 8)")
    p.add_argument("--lr-penalty", choices=["l1", "l2", "both"], default="both",
                   help="Restrict LR HPO to a single penalty (default: both). "
                        "Use 'l2' to study ridge behaviour for correlated bases.")
    return p.parse_args()


# RF HPO (OOB-based)

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


def run_hpo_rf(X_tr, y_tr, n_trials, timeout):
    def objective(trial):
        params = _rf_param_fn(trial)
        clf = RandomForestClassifier(
            oob_score=True, bootstrap=True,
            random_state=RANDOM_STATE, n_jobs=-1, **params,
        )
        clf.fit(X_tr, y_tr)
        # RF gets OOB predictions "for free", so we skip extra CV
        y_oob = clf.oob_decision_function_[:, 1]
        return roc_auc_score(y_tr, y_oob)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Shared Optuna budget for this (K, split) cell
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_trial.params, study.best_value


# SVM HPO (CV-based)

def _svm_param_fn(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def run_hpo_svm(X_tr, y_tr, n_trials, timeout, n_jobs):
    # Standard 3-fold CV objective for SVM/LR/XGB
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = _svm_param_fn(trial)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=False,
                        random_state=RANDOM_STATE, **params)),
        ])
        scores = cross_val_score(
            pipe, X_tr, y_tr, cv=inner_cv,
            scoring="roc_auc", n_jobs=n_jobs,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_trial.params, study.best_value


# LR HPO (CV-based)

def _lr_param_fn(trial, penalty_choices=("l1", "l2")):
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "penalty": trial.suggest_categorical("penalty", list(penalty_choices)),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def run_hpo_lr(X_tr, y_tr, n_trials, timeout, n_jobs, penalty_choices=("l1", "l2")):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = _lr_param_fn(trial, penalty_choices=penalty_choices)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="saga", max_iter=5000,
                random_state=RANDOM_STATE, **params,
            )),
        ])
        scores = cross_val_score(
            pipe, X_tr, y_tr, cv=inner_cv,
            scoring="roc_auc", n_jobs=n_jobs,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_trial.params, study.best_value


# XGB HPO (CV-based)

def _xgb_param_fn(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500, 700]),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2, 3, 4]),
    }


def run_hpo_xgb(X_tr, y_tr, n_trials, timeout, n_jobs):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = _xgb_param_fn(trial)
        clf = XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=1, verbosity=0, **params,
        )
        scores = cross_val_score(
            clf, X_tr, y_tr, cv=inner_cv,
            scoring="roc_auc", n_jobs=n_jobs,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_trial.params, study.best_value


# Helpers

def print_param_stability(all_params, clf_name):
    """Print frequency of each parameter value across all HPO results."""
    if not all_params:
        return

    param_names = sorted(all_params[0].keys())
    n = len(all_params)

    print(f"\n{'=' * 60}")
    print(f"  Parameter stability across {n} cells ({clf_name})")
    print(f"{'=' * 60}")

    suggested = {}
    for pname in param_names:
        # Frequency table helps decide if one fixed value is robust
        values = [p[pname] for p in all_params]
        counts = Counter(values)
        most_common_val, most_common_count = counts.most_common(1)[0]

        parts = []
        for val, cnt in counts.most_common():
            val_str = str(val) if not isinstance(val, float) else f"{val:.4g}"
            parts.append(f"{val_str} ({cnt}/{n})")
        stability = "STABLE" if most_common_count >= n * 0.6 else "VARIABLE"

        print(f"  {pname:25s}: {', '.join(parts)}  [{stability}]")
        suggested[pname] = most_common_val

    print(f"\n  Suggested fixed params:")
    print(f"  {json.dumps({k: json_safe(v) for k, v in suggested.items()}, indent=2)}")
    print()

    return suggested


# Main loop

def main():
    args = parse_args()
    clf = args.clf.upper()
    basis = args.basis
    n_trials = args.n_trials
    timeout = args.timeout
    n_jobs = args.n_jobs
    lr_penalty = args.lr_penalty

    if clf == "LR" and lr_penalty != "both":
        lr_penalty_choices = (lr_penalty,)
    else:
        lr_penalty_choices = ("l1", "l2")

    print("=" * 60)
    print(f"  HPO Preliminary: {clf} {basis}")
    print(f"  n_trials={n_trials}  timeout={timeout}s  n_jobs={n_jobs}")
    if clf == "LR":
        print(f"  LR penalty search space: {list(lr_penalty_choices)}")
    print(f"  K values: {REPRESENTATIVE_K}")
    print("=" * 60)

    bp = step02.load_block(BP_SAMPLED_CSV)
    rp = step02.load_block(RP_SAMPLED_CSV)
    # Quick guard: fail fast if BP and RP sources drift apart
    step02.check_alignment(bp, rp)

    splits_path = DATA_DIR / "splits_rskf.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}.")
    with splits_path.open() as fh:
        all_splits = json.load(fh)

    splits_dict = {k: v for k, v in all_splits.items()
                   if k.startswith("rep0_")}
    split_names = sorted(splits_dict.keys())
    print(f"Splits: {len(split_names)} (rep0 only)")

    k_values = REPRESENTATIVE_K
    if basis == "bspline":
        # Cubic B-spline basis is not valid for very small K
        k_values = [k for k in k_values if k >= BSPLINE_MIN_K]
        print(f"  bspline: K values adjusted to {k_values}")

    if clf == "LR" and lr_penalty != "both":
        raw_csv = RESULTS_DIR / f"hpo_preliminary_{clf}_{basis}_{lr_penalty}.csv"
    else:
        raw_csv = RESULTS_DIR / f"hpo_preliminary_{clf}_{basis}.csv"
    # Append rows incrementally so interrupted runs keep partial output
    csv_header_written = False
    all_best_params = []
    rows = []

    t_start = time.time()
    total = len(k_values) * len(split_names)
    done = 0

    for k in k_values:
        t_feat = time.time()
        # Generate one symmetric (K_BP=K_RP=K) feature block per K
        X, y = generate_features(step02, bp, rp, basis, k, k)
        feat_s = time.time() - t_feat
        print(f"\n  >> K_BP={k} K_RP={k} {basis} -> {X.shape[1]}D "
              f"({X.shape[0]} samples) in {feat_s:.1f}s")

        for sname in split_names:
            split = splits_dict[sname]
            # Preliminary HPO uses only the training side of each split
            train_idx = np.asarray(split["train"], dtype=int)
            X_tr, y_tr = X[train_idx], y[train_idx]

            t_cell = time.time()
            if clf == "RF":
                best_params, best_score = run_hpo_rf(
                    X_tr, y_tr, n_trials, timeout,
                )
            elif clf == "SVM":
                best_params, best_score = run_hpo_svm(
                    X_tr, y_tr, n_trials, timeout, n_jobs,
                )
            elif clf == "LR":
                best_params, best_score = run_hpo_lr(
                    X_tr, y_tr, n_trials, timeout, n_jobs,
                    penalty_choices=lr_penalty_choices,
                )
            else:
                best_params, best_score = run_hpo_xgb(
                    X_tr, y_tr, n_trials, timeout, n_jobs,
                )
            cell_s = time.time() - t_cell

            done += 1
            all_best_params.append(best_params)
            params_json = json.dumps({k2: json_safe(v)
                                      for k2, v in best_params.items()})

            row = {
                "K_BP": k, "K_RP": k, "basis": basis,
                "classifier": clf, "split": sname,
                "n_trials": n_trials,
                "best_cv_roc_auc": round(best_score, 6),
                "best_params": params_json,
            }
            rows.append(row)

            row_df = pd.DataFrame([row])
            # Stream rows to disk so long runs stay resumable
            row_df.to_csv(raw_csv, mode="a",
                          header=not csv_header_written, index=False)
            csv_header_written = True

            elapsed = time.time() - t_start
            eta = (elapsed / done) * (total - done)
            print(f"  [{done}/{total}] K={k} {sname}  "
                  f"best_cv={best_score:.4f}  {cell_s:.1f}s  "
                  f"ETA={eta:.0f}s", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nFinished {done} cells in {elapsed_total / 60:.1f} minutes.")

    # Per-K summary (čia patogu greitai pamatyti trendą)
    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print(f"  Per-K summary (mean best_cv_roc_auc across splits)")
    print(f"{'=' * 60}")
    for k in k_values:
        sub = df[df["K_BP"] == k]
        mean_score = sub["best_cv_roc_auc"].mean()
        std_score = sub["best_cv_roc_auc"].std()
        print(f"  K_BP={k:3d}  K_RP={k:3d}  "
              f"ROC-AUC = {mean_score:.4f} +/- {std_score:.4f}")

    suggested = print_param_stability(all_best_params, clf)

    print(f"Results saved to: {raw_csv}")
    print(f"\nTo use fixed params in the grid sweep:")
    if suggested:
        safe_params = {k2: json_safe(v) for k2, v in suggested.items()}
        params_str = json.dumps(safe_params)
        script_map = {
            "RF": "08_kbp_krp_grid_rf.py",
            "SVM": "08_kbp_krp_grid_svm.py",
            "LR": "08_kbp_krp_grid_lr.py",
            "XGB": "08_kbp_krp_grid_xgb.py",
        }
        script = script_map[clf]
        print(f"  python {script} --basis {basis} "
              f"--fixed-params '{params_str}'")


if __name__ == "__main__":
    main()
