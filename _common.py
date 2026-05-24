from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
THESIS_ROOT = WORKSPACE_ROOT.parent
LEGACY_DIR = WORKSPACE_ROOT / "transformation_experiment"

from numpy.polynomial import chebyshev, hermite_e, laguerre, legendre  # noqa: E402

_POLY_CLASSES = {
    "Chebyshev": chebyshev.Chebyshev,
    "Hermite": hermite_e.HermiteE,
    "Laguerre": laguerre.Laguerre,
    "Legendre": legendre.Legendre,
}


def fit_polynomial(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    basis: str = "Chebyshev",
    n_coeffs: int = 20,
) -> dict:
    """Fit an orthogonal polynomial to a spectrum."""
    if basis not in _POLY_CLASSES:
        raise ValueError(f"Unknown basis {basis!r}. Valid: {list(_POLY_CLASSES)}")
    fitted = _POLY_CLASSES[basis].fit(wavelengths, flux, deg=n_coeffs - 1)
    y_hat = fitted(wavelengths)
    residuals = flux - y_hat
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((flux - np.mean(flux)) ** 2)
    return {
        "fitted_values": y_hat,
        "coefficients": fitted.coef,
        "metrics": {
            "RMSE": float(np.sqrt(np.mean(residuals ** 2))),
            "MAE": float(np.mean(np.abs(residuals))),
            "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        },
    }


def l2_normalize(
    df: pd.DataFrame,
    coeff_cols: list[str] | None = None,
) -> pd.DataFrame:
    """L2-normalize coefficient rows (returns a copy)."""
    if coeff_cols is None:
        coeff_cols = [c for c in df.columns if c.startswith("c")]
    cols = list(coeff_cols)
    out = df.copy()
    X = out[cols].values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    out[cols] = X / safe_norms
    return out


DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
for directory in (DATA_DIR, FEATURES_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RAW_CLASSIFICATION_CSV = LEGACY_DIR / "classification_with_c110_d110_errors_snr.csv"
LEGACY_OG_XP = LEGACY_DIR / "data" / "og_xp.csv"
LEGACY_SPLITS = LEGACY_DIR / "data" / "splits.json"

LOCAL_OG_XP = DATA_DIR / "og_xp_bp_rp.csv"
LOCAL_SPLITS = DATA_DIR / "splits.json"
LOCAL_MANIFEST = ROOT / "manifest.json"

BP_SAMPLED_CSV = DATA_DIR / "bp_sampled_spectra.csv"
RP_SAMPLED_CSV = DATA_DIR / "rp_sampled_spectra.csv"


def save_manifest(extra: dict | None = None) -> None:
    manifest = {
        "root": str(ROOT),
        "legacy_dir": str(LEGACY_DIR),
        "raw_classification_csv": str(RAW_CLASSIFICATION_CSV),
        "legacy_og_xp": str(LEGACY_OG_XP),
        "legacy_splits": str(LEGACY_SPLITS),
        "local_og_xp": str(LOCAL_OG_XP),
        "bp_sampled_csv": str(BP_SAMPLED_CSV),
        "rp_sampled_csv": str(RP_SAMPLED_CSV),
    }
    if extra:
        manifest.update(extra)
    LOCAL_MANIFEST.write_text(json.dumps(manifest, indent=2))


def get_wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("wl_") or c.startswith("u_")]


def get_wavelengths(df: pd.DataFrame) -> np.ndarray:
    cols = get_wavelength_columns(df)
    if not cols:
        raise ValueError("No sampling columns found. Expected columns like 'wl_336' or 'u_0.000'.")
    return np.asarray([float(c.split("_", 1)[1]) for c in cols], dtype=float)


def flatten_feature_blocks(
    source_ids: Iterable[int],
    labels: Iterable[int],
    bp_coeffs: np.ndarray,
    rp_coeffs: np.ndarray,
) -> pd.DataFrame:
    bp_coeffs = np.asarray(bp_coeffs, dtype=float)
    rp_coeffs = np.asarray(rp_coeffs, dtype=float)
    if bp_coeffs.shape != rp_coeffs.shape:
        raise ValueError(
            f"BP/RP coefficient matrices must match shape, got {bp_coeffs.shape} vs {rp_coeffs.shape}."
        )
    n_rows, n_coeffs = bp_coeffs.shape
    columns = [f"c{i:03d}" for i in range(2 * n_coeffs)]
    stacked = np.hstack([bp_coeffs, rp_coeffs])
    out = pd.DataFrame(stacked, columns=columns)
    out.insert(0, "y", np.asarray(list(labels), dtype=int))
    out.insert(0, "source_id", np.asarray(list(source_ids)))
    return out


def load_split_records() -> list[dict]:
    if not LOCAL_SPLITS.exists():
        raise FileNotFoundError(
            f"Split file not found: {LOCAL_SPLITS}. Run 01_prepare_inputs.py first."
        )
    with LOCAL_SPLITS.open() as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        return [
            {"train_idx": v["train"], "test_idx": v["test"]}
            for _, v in sorted(raw.items())
        ]
    return raw


def align_bp_rp_frames(bp_df: pd.DataFrame, rp_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["source_id", "y"]
    bp_wl = get_wavelength_columns(bp_df)
    rp_wl = get_wavelength_columns(rp_df)
    merged = bp_df[merge_cols + bp_wl].merge(
        rp_df[merge_cols + rp_wl],
        on=merge_cols,
        how="inner",
        suffixes=("_bp", "_rp"),
    )
    if merged.empty:
        raise ValueError("BP/RP merge produced no rows. Check source_id and y alignment.")
    return merged
