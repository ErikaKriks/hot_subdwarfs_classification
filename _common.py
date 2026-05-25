from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Core path anchors used across scripts
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
    # Keep this strict so typos fail fast (čia tikrai sutaupo nervų)
    if basis not in _POLY_CLASSES:
        raise ValueError(f"Unknown basis {basis!r}. Valid: {list(_POLY_CLASSES)}")
    # deg = n_coeffs - 1 because polynomial degree is zero-indexed
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
        # Convention in this repo: learned coefficients start with "c"
        coeff_cols = [c for c in df.columns if c.startswith("c")]
    cols = list(coeff_cols)
    # Copy on purpose: callers often reuse the original frame later
    out = df.copy()
    X = out[cols].values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Protect against zero vectors to avoid division warnings/NaNs
    safe_norms = np.where(norms > 0, norms, 1.0)
    out[cols] = X / safe_norms
    return out


# Standard local directories
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
for directory in (DATA_DIR, FEATURES_DIR, RESULTS_DIR):
    # Safe to call repeatedly; keeps first-run UX smooth
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
    # Lightweight provenance snapshot for local runs
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
    # Nice for quick diffs when paths or data sources change
    LOCAL_MANIFEST.write_text(json.dumps(manifest, indent=2))


def get_wavelength_columns(df: pd.DataFrame) -> list[str]:
    # Support both original sampled names and normalized-grid names
    return [c for c in df.columns if c.startswith("wl_") or c.startswith("u_")]


def get_wavelengths(df: pd.DataFrame) -> np.ndarray:
    cols = get_wavelength_columns(df)
    if not cols:
        raise ValueError("No sampling columns found. Expected columns like 'wl_336' or 'u_0.000'.")
    # Parse numeric suffix after first underscore, preserving column order
    return np.asarray([float(c.split("_", 1)[1]) for c in cols], dtype=float)


def flatten_feature_blocks(
    source_ids: Iterable[int],
    labels: Iterable[int],
    bp_coeffs: np.ndarray,
    rp_coeffs: np.ndarray,
) -> pd.DataFrame:
    bp_coeffs = np.asarray(bp_coeffs, dtype=float)
    rp_coeffs = np.asarray(rp_coeffs, dtype=float)
    # Symmetric version: BP and RP must have identical coefficient shapes
    if bp_coeffs.shape != rp_coeffs.shape:
        raise ValueError(
            f"BP/RP coefficient matrices must match shape, got {bp_coeffs.shape} vs {rp_coeffs.shape}."
        )
    n_rows, n_coeffs = bp_coeffs.shape
    # 2 * n_coeffs because we concatenate BP and RP side by side
    columns = [f"c{i:03d}" for i in range(2 * n_coeffs)]
    stacked = np.hstack([bp_coeffs, rp_coeffs])
    out = pd.DataFrame(stacked, columns=columns)
    out.insert(0, "y", np.asarray(list(labels), dtype=int))
    out.insert(0, "source_id", np.asarray(list(source_ids)))
    return out


def flatten_feature_blocks_asym(
    source_ids: Iterable[int],
    labels: Iterable[int],
    bp_coeffs: np.ndarray,
    rp_coeffs: np.ndarray,
) -> pd.DataFrame:
    """Flatten BP/RP coefficient blocks allowing K_BP != K_RP."""
    bp_coeffs = np.asarray(bp_coeffs, dtype=float)
    rp_coeffs = np.asarray(rp_coeffs, dtype=float)
    # Asymmetric version: only row count must match
    if bp_coeffs.shape[0] != rp_coeffs.shape[0]:
        raise ValueError(
            f"Row count mismatch: BP {bp_coeffs.shape[0]} vs RP {rp_coeffs.shape[0]}."
        )
    total_cols = bp_coeffs.shape[1] + rp_coeffs.shape[1]
    columns = [f"c{i:03d}" for i in range(total_cols)]
    stacked = np.hstack([bp_coeffs, rp_coeffs])
    out = pd.DataFrame(stacked, columns=columns)
    out.insert(0, "y", np.asarray(list(labels), dtype=int))
    out.insert(0, "source_id", np.asarray(list(source_ids)))
    return out


def json_safe(value):
    """Convert NumPy-heavy values into JSON-serializable Python objects."""
    # Order matters here: np.bool_ is also np.integer on some NumPy versions
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 6)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        # Recurse nested structures used in params/metrics payloads
        return {k: json_safe(v) for k, v in value.items()}
    return value


def load_split_records() -> list[dict]:
    # Main split loader used by non-RSKF scripts
    if not LOCAL_SPLITS.exists():
        raise FileNotFoundError(
            f"Split file not found: {LOCAL_SPLITS}. Run 01_prepare_inputs.py first."
        )
    with LOCAL_SPLITS.open() as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        # Legacy shape: {"split_0": {"train": [...], "test": [...]}, ...}
        return [
            {"train_idx": v["train"], "test_idx": v["test"]}
            for _, v in sorted(raw.items())
        ]
    # Newer shape is already a list of split dicts
    return raw


def align_bp_rp_frames(bp_df: pd.DataFrame, rp_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows present in both blocks with matching label/source_id
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
    # patikrinimas 
    return merged
