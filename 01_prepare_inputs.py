#!/usr/bin/env python3
from __future__ import annotations

import json

import pandas as pd

from _common import (
    LEGACY_OG_XP,
    LEGACY_SPLITS,
    LOCAL_OG_XP,
    LOCAL_SPLITS,
    RAW_CLASSIFICATION_CSV,
    save_manifest,
)


def build_bp_rp_coeff_table() -> pd.DataFrame:
    if LEGACY_OG_XP.exists():
        df = pd.read_csv(LEGACY_OG_XP)
        expected_bp = [f"bp_{i:02d}" for i in range(55)]
        expected_rp = [f"rp_{i:02d}" for i in range(55)]
        missing = [c for c in expected_bp + expected_rp if c not in df.columns]
        if missing:
            raise KeyError(f"{LEGACY_OG_XP} is missing expected columns: {missing[:8]}")
        return df[["source_id", "y"] + expected_bp + expected_rp].copy()

    df = pd.read_csv(RAW_CLASSIFICATION_CSV)
    coeff_cols = [f"c{i:03d}" for i in range(110)]
    missing = [c for c in coeff_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{RAW_CLASSIFICATION_CSV} is missing coefficient columns: {missing[:8]}")

    out = pd.DataFrame(
        {
            "source_id": df["source_id"].values,
            "y": df["y"].astype(int).values,
        }
    )
    for i in range(55):
        out[f"bp_{i:02d}"] = df[f"c{i:03d}"].astype(float).values
    for i in range(55):
        out[f"rp_{i:02d}"] = df[f"c{i + 55:03d}"].astype(float).values
    return out


def copy_splits_if_available() -> bool:
    if not LEGACY_SPLITS.exists():
        return False
    with LEGACY_SPLITS.open() as fh:
        splits = json.load(fh)
    LOCAL_SPLITS.write_text(json.dumps(splits, indent=2))
    return True


def main() -> None:
    df = build_bp_rp_coeff_table()
    LOCAL_OG_XP.write_text(df.to_csv(index=False))
    have_splits = copy_splits_if_available()

    save_manifest(
        {
            "rows": int(len(df)),
            "bp_coeff_columns": 55,
            "rp_coeff_columns": 55,
            "copied_splits": have_splits,
            "next_expected_inputs": [
                "data/bp_sampled_spectra.csv",
                "data/rp_sampled_spectra.csv",
            ],
        }
    )

    print(f"Saved clean BP/RP coefficient table -> {LOCAL_OG_XP}")
    if have_splits:
        print(f"Copied legacy splits -> {LOCAL_SPLITS}")
    else:
        print("Legacy splits.json not found; create splits before running classification.")
    print("Expected next-step inputs:")
    print("  - data/bp_sampled_spectra.csv")
    print("  - data/rp_sampled_spectra.csv")


if __name__ == "__main__":
    main()
