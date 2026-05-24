#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from _common import (
    BP_SAMPLED_CSV,
    FEATURES_DIR,
    RESULTS_DIR,
    RP_SAMPLED_CSV,
    flatten_feature_blocks,
    fit_polynomial,
    get_wavelength_columns,
    get_wavelengths,
    l2_normalize,
    save_manifest,
)

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POLY_BASES = {
    "chebyshev": "Chebyshev",
    "legendre": "Legendre",
}
ALL_BASES = ("chebyshev", "legendre", "bspline", "fourier")
ALL_SMOOTHING = ("none", "savgol", "gaussian", "spline")
PLOT_COLORS = {
    "chebyshev": "#1f77b4",
    "legendre": "#ff7f0e",
    "bspline": "#2ca02c",
    "fourier": "#d62728",
}
PLOT_LINESTYLES = {
    "none": "-",
    "savgol": "--",
    "gaussian": "-.",
    "spline": ":",
}

PLOT_SMOOTHER_COLORS = {
    "none": "#555555",
    "gaussian": "#e41a1c",
    "savgol": "#377eb8",
    "spline": "#4daf4a",
}
PLOT_BASIS_MARKERS = {
    "bspline": "o",
    "chebyshev": "s",
    "legendre": "^",
    "fourier": "D",
}
PLOT_BASIS_LINESTYLES = {
    "bspline": "-",
    "chebyshev": "--",
    "legendre": "-.",
    "fourier": ":",
}


@dataclass
class SpectrumBlock:
    source_ids: np.ndarray
    labels: np.ndarray
    wavelengths: np.ndarray
    flux: np.ndarray


@dataclass
class BlockFitResult:
    coeffs: np.ndarray
    smoothed_flux: np.ndarray
    reconstructed_flux: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BP/RP-separated basis features.")
    parser.add_argument(
        "--bases",
        nargs="+",
        default=list(ALL_BASES),
        choices=list(ALL_BASES),
    )
    parser.add_argument(
        "--smoothing",
        nargs="+",
        default=["none"],
        choices=list(ALL_SMOOTHING),
    )
    parser.add_argument(
        "--k-list",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 30, 35, 40],
    )
    parser.add_argument(
        "--preview-stars",
        type=int,
        default=4,
        help="How many example stars to keep for reconstruction overlay plots.",
    )
    parser.add_argument(
        "--top-configs",
        type=int,
        default=3,
        help="How many top reconstruction configs to show per arm in the summary plot.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_block(csv_path: Path) -> SpectrumBlock:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing sampled spectrum file: {csv_path}\n"
            "This script expects BP and RP spectra to be calibrated separately first."
        )
    df = pd.read_csv(csv_path)
    wl_cols = get_wavelength_columns(df)
    return SpectrumBlock(
        source_ids=df["source_id"].to_numpy(),
        labels=df["y"].astype(int).to_numpy(),
        wavelengths=get_wavelengths(df),
        flux=df[wl_cols].to_numpy(dtype=float),
    )


def check_alignment(bp: SpectrumBlock, rp: SpectrumBlock) -> None:
    if not np.array_equal(bp.source_ids, rp.source_ids):
        raise ValueError("BP and RP source_id order does not match.")
    if not np.array_equal(bp.labels, rp.labels):
        raise ValueError("BP and RP labels do not match.")


def smooth_flux(flux_row: np.ndarray, method: str, **kwargs) -> np.ndarray:
    flux_row = np.asarray(flux_row, dtype=float)
    if method == "none":
        return flux_row
    if method == "savgol":
        window = kwargs.get("window_length", 11)
        window = min(window, len(flux_row) if len(flux_row) % 2 else len(flux_row) - 1)
        window = max(window, 5)
        if window % 2 == 0:
            window -= 1
        polyorder = kwargs.get("polyorder", 3)
        polyorder = min(polyorder, window - 1)
        return savgol_filter(flux_row, window_length=window, polyorder=polyorder, mode="interp")
    if method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        return gaussian_filter1d(flux_row, sigma=sigma, mode="nearest")
    if method == "spline":
        x = np.arange(len(flux_row), dtype=float)
        s_factor = kwargs.get("s_factor", 0.05)
        spline = UnivariateSpline(x, flux_row, s=len(flux_row) * s_factor)
        return spline(x)
    raise ValueError(f"Unknown smoothing method: {method}")


def normalized_grid(wavelengths: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    wl_min, wl_max = float(wl.min()), float(wl.max())
    return 2.0 * (wl - wl_min) / (wl_max - wl_min) - 1.0


def bspline_design_matrix(x: np.ndarray, n_coeffs: int, degree: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    if n_coeffs <= degree:
        raise ValueError(f"n_coeffs must be > degree for B-splines, got {n_coeffs} <= {degree}")

    n_internal = n_coeffs - degree - 1
    if n_internal > 0:
        internal = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)
    knots = np.concatenate(
        [
            np.repeat(x_min, degree + 1),
            internal,
            np.repeat(x_max, degree + 1),
        ]
    )

    basis_rows = []
    for i in range(n_coeffs):
        coeffs = np.zeros(n_coeffs, dtype=float)
        coeffs[i] = 1.0
        basis_rows.append(BSpline(knots, coeffs, degree)(x))
    return np.column_stack(basis_rows)


def fourier_design_matrix(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x)]
    harmonic = 1
    while len(cols) < n_coeffs:
        cols.append(np.cos(np.pi * harmonic * x))
        if len(cols) >= n_coeffs:
            break
        cols.append(np.sin(np.pi * harmonic * x))
        harmonic += 1
    return np.column_stack(cols[:n_coeffs])


def basis_design_matrix(wavelengths: np.ndarray, basis: str, n_coeffs: int) -> np.ndarray:
    x = normalized_grid(wavelengths)
    if basis == "chebyshev":
        return np.polynomial.chebyshev.chebvander(x, n_coeffs - 1)
    if basis == "legendre":
        return np.polynomial.legendre.legvander(x, n_coeffs - 1)
    if basis == "bspline":
        return bspline_design_matrix(x, n_coeffs=n_coeffs, degree=3)
    if basis == "fourier":
        return fourier_design_matrix(x, n_coeffs=n_coeffs)
    raise ValueError(f"Unsupported basis: {basis}")


def fit_basis(wavelengths: np.ndarray, flux_row: np.ndarray, basis: str, n_coeffs: int) -> np.ndarray:
    if basis in POLY_BASES:
        result = fit_polynomial(
            wavelengths=wavelengths,
            flux=flux_row,
            basis=POLY_BASES[basis],
            n_coeffs=n_coeffs,
        )
        return np.asarray(result["coefficients"][:n_coeffs], dtype=float)
    if basis == "bspline":
        design = basis_design_matrix(wavelengths, basis=basis, n_coeffs=n_coeffs)
        coeffs, *_ = np.linalg.lstsq(design, flux_row, rcond=None)
        return coeffs.astype(float)
    if basis == "fourier":
        design = basis_design_matrix(wavelengths, basis=basis, n_coeffs=n_coeffs)
        coeffs, *_ = np.linalg.lstsq(design, flux_row, rcond=None)
        return coeffs.astype(float)
    raise ValueError(f"Unsupported basis: {basis}")


def reconstruct_flux(wavelengths: np.ndarray, coeffs: np.ndarray, basis: str) -> np.ndarray:
    design = basis_design_matrix(wavelengths, basis=basis, n_coeffs=len(coeffs))
    return design @ np.asarray(coeffs, dtype=float)


def build_block_fit(
    block: SpectrumBlock, basis: str, smoothing: str, n_coeffs: int, **smooth_kwargs
) -> BlockFitResult:
    coeffs = np.zeros((block.flux.shape[0], n_coeffs), dtype=float)
    smoothed_flux = np.zeros_like(block.flux, dtype=float)
    reconstructed_flux = np.zeros_like(block.flux, dtype=float)
    for i in range(block.flux.shape[0]):
        smoothed = smooth_flux(block.flux[i], smoothing, **smooth_kwargs)
        row_coeffs = fit_basis(block.wavelengths, smoothed, basis, n_coeffs)
        coeffs[i] = row_coeffs
        smoothed_flux[i] = smoothed
        reconstructed_flux[i] = reconstruct_flux(block.wavelengths, row_coeffs, basis)
    return BlockFitResult(
        coeffs=coeffs,
        smoothed_flux=smoothed_flux,
        reconstructed_flux=reconstructed_flux,
    )


def compute_metric_arrays(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, np.ndarray]:
    residual = reference - reconstructed
    mse = np.mean(residual**2, axis=1)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual), axis=1)
    ref_norm = np.linalg.norm(reference, axis=1)
    rel_l2 = np.linalg.norm(residual, axis=1) / np.maximum(ref_norm, 1e-12)
    centered = reference - reference.mean(axis=1, keepdims=True)
    sst = np.sum(centered**2, axis=1)
    sse = np.sum(residual**2, axis=1)
    r2 = np.where(sst > 1e-12, 1.0 - (sse / sst), np.nan)
    return {
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
        "r2": r2,
    }


def metric_frame(
    block: SpectrumBlock,
    arm: str,
    basis: str,
    smoothing: str,
    n_coeffs: int,
    smoothed_flux: np.ndarray,
    reconstructed_flux: np.ndarray,
) -> pd.DataFrame:
    metrics_raw = compute_metric_arrays(block.flux, reconstructed_flux)
    metrics_smoothed = compute_metric_arrays(smoothed_flux, reconstructed_flux)
    frame = pd.DataFrame(
        {
            "source_id": block.source_ids,
            "y": block.labels,
            "arm": arm,
            "basis": basis,
            "smoothing": smoothing,
            "n_coeffs": n_coeffs,
            "rmse_to_raw": metrics_raw["rmse"],
            "mae_to_raw": metrics_raw["mae"],
            "rel_l2_to_raw": metrics_raw["rel_l2"],
            "r2_to_raw": metrics_raw["r2"],
            "rmse_to_smoothed": metrics_smoothed["rmse"],
            "mae_to_smoothed": metrics_smoothed["mae"],
            "rel_l2_to_smoothed": metrics_smoothed["rel_l2"],
            "r2_to_smoothed": metrics_smoothed["r2"],
        }
    )
    return frame


def combined_metric_frame(
    bp_block: SpectrumBlock,
    rp_block: SpectrumBlock,
    basis: str,
    smoothing: str,
    n_coeffs: int,
    bp_smoothed_flux: np.ndarray,
    rp_smoothed_flux: np.ndarray,
    bp_reconstructed_flux: np.ndarray,
    rp_reconstructed_flux: np.ndarray,
) -> pd.DataFrame:
    combined_raw = np.hstack([bp_block.flux, rp_block.flux])
    combined_smoothed = np.hstack([bp_smoothed_flux, rp_smoothed_flux])
    combined_reconstructed = np.hstack([bp_reconstructed_flux, rp_reconstructed_flux])

    metrics_raw = compute_metric_arrays(combined_raw, combined_reconstructed)
    metrics_smoothed = compute_metric_arrays(combined_smoothed, combined_reconstructed)
    return pd.DataFrame(
        {
            "source_id": bp_block.source_ids,
            "y": bp_block.labels,
            "arm": "bp_rp_combined",
            "basis": basis,
            "smoothing": smoothing,
            "n_coeffs": n_coeffs,
            "rmse_to_raw": metrics_raw["rmse"],
            "mae_to_raw": metrics_raw["mae"],
            "rel_l2_to_raw": metrics_raw["rel_l2"],
            "r2_to_raw": metrics_raw["r2"],
            "rmse_to_smoothed": metrics_smoothed["rmse"],
            "mae_to_smoothed": metrics_smoothed["mae"],
            "rel_l2_to_smoothed": metrics_smoothed["rel_l2"],
            "r2_to_smoothed": metrics_smoothed["r2"],
        }
    )


def summarize_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.endswith("_to_raw") or c.endswith("_to_smoothed")]
    summary = (
        df.groupby(["arm", "basis", "smoothing", "n_coeffs"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col if stat == "" else f"{col}_{stat}"
        for col, stat in summary.columns.to_flat_index()
    ]
    return summary


def feature_path(basis: str, smoothing: str, n_coeffs: int) -> Path:
    return FEATURES_DIR / f"{basis}_{smoothing}_{n_coeffs:02d}_L2.csv"


def choose_example_indices(labels: np.ndarray, n_examples: int) -> np.ndarray:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if n_examples <= 0:
        return np.array([], dtype=int)

    chosen: list[int] = []
    per_class = max(1, n_examples // max(len(unique), 1))
    for label in unique:
        idx = np.flatnonzero(labels == label)
        chosen.extend(idx[:per_class].tolist())

    if len(chosen) < n_examples:
        remaining = [i for i in range(len(labels)) if i not in set(chosen)]
        chosen.extend(remaining[: n_examples - len(chosen)])

    return np.asarray(chosen[:n_examples], dtype=int)


def plot_metric_curves(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    metric_names = ("rmse_to_raw_mean", "rel_l2_to_raw_mean", "r2_to_raw_mean")
    metric_titles = {
        "rmse_to_raw_mean": "RMSE to Raw Flux",
        "rel_l2_to_raw_mean": "Relative L2 Error to Raw Flux",
        "r2_to_raw_mean": "R^2 to Raw Flux",
    }
    saved_paths: list[Path] = []

    for arm in ("bp", "rp", "bp_rp_combined"):
        arm_df = summary_df[summary_df["arm"] == arm].copy()
        if arm_df.empty:
            continue
        for metric in metric_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            for (basis, smoothing), group in arm_df.groupby(["basis", "smoothing"]):
                group = group.sort_values("n_coeffs")
                ax.plot(
                    group["n_coeffs"],
                    group[metric],
                    marker=PLOT_BASIS_MARKERS.get(basis, "o"),
                    markersize=7,
                    color=PLOT_SMOOTHER_COLORS.get(smoothing, "black"),
                    linestyle=PLOT_BASIS_LINESTYLES.get(basis, "-"),
                    label=f"{basis} + {smoothing}",
                )
            arm_label = "BP+RP combined" if arm == "bp_rp_combined" else arm.upper()
            ax.set_title(f"{arm_label} reconstruction comparison: {metric_titles[metric]}")
            ax.set_xlabel("Number of coefficients (K)")
            ax.set_ylabel(metric_titles[metric])
            ax.grid(alpha=0.25)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            fig.tight_layout()
            out_path = output_dir / f"{arm}_{metric}.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)
    return saved_paths


def plot_best_reconstructions(
    arm: str,
    block: SpectrumBlock,
    summary_df: pd.DataFrame,
    preview_cache: dict[tuple[str, str, int, str], np.ndarray],
    preview_idx: np.ndarray,
    output_dir: Path,
    top_n: int,
) -> Path | None:
    arm_summary = summary_df[summary_df["arm"] == arm].sort_values("rel_l2_to_raw_mean")
    if arm_summary.empty or len(preview_idx) == 0:
        return None

    best = arm_summary.head(top_n)
    n_rows = len(preview_idx)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, max(3, 2.8 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, row_idx in zip(axes, preview_idx):
        ax.plot(
            block.wavelengths,
            block.flux[row_idx],
            color="black",
            linewidth=2.0,
            label="raw",
        )
        for _, cfg in best.iterrows():
            key = (
                str(cfg["basis"]),
                str(cfg["smoothing"]),
                int(cfg["n_coeffs"]),
                arm,
            )
            recon = preview_cache[key]
            ax.plot(
                block.wavelengths,
                recon[np.where(preview_idx == row_idx)[0][0]],
                linewidth=1.6,
                color=PLOT_COLORS.get(str(cfg["basis"]), "gray"),
                linestyle=PLOT_LINESTYLES.get(str(cfg["smoothing"]), "-"),
                label=f"{cfg['basis']} + {cfg['smoothing']} (K={int(cfg['n_coeffs'])})",
            )
        ax.set_ylabel("Flux")
        ax.set_title(f"{arm.upper()} source_id={int(block.source_ids[row_idx])} y={int(block.labels[row_idx])}")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    axes[0].legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    out_path = output_dir / f"{arm}_best_reconstruction_examples.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def rank_combined_configs(summary_df: pd.DataFrame) -> pd.DataFrame:
    combined = summary_df[summary_df["arm"] == "bp_rp_combined"].copy()
    if combined.empty:
        return combined
    return combined.sort_values(
        ["rel_l2_to_raw_mean", "rmse_to_raw_mean", "n_coeffs"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def rank_basis_only_configs(
    summary_df: pd.DataFrame,
    arm: str = "bp_rp_combined",
    smoothing: str = "none",
) -> pd.DataFrame:
    filtered = summary_df[
        (summary_df["arm"] == arm) & (summary_df["smoothing"] == smoothing)
    ].copy()
    if filtered.empty:
        return filtered
    return filtered.sort_values(
        ["rel_l2_to_raw_mean", "rmse_to_raw_mean", "n_coeffs"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def best_k_per_basis(
    ranking_df: pd.DataFrame,
    metric: str = "rel_l2_to_raw_mean",
) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df
    best = (
        ranking_df.sort_values([metric, "rmse_to_raw_mean", "n_coeffs"])
        .groupby("basis", as_index=False)
        .first()
    )
    return best.sort_values([metric, "rmse_to_raw_mean", "n_coeffs"]).reset_index(drop=True)


def get_row_index_by_source_id(block: SpectrumBlock, source_id: int) -> int:
    matches = np.flatnonzero(block.source_ids == source_id)
    if len(matches) == 0:
        raise ValueError(f"source_id={source_id} not found in sampled spectra.")
    return int(matches[0])


def plot_selected_star_reconstructions(
    bp_block: SpectrumBlock,
    rp_block: SpectrumBlock,
    source_id: int,
    bases: list[str] | tuple[str, ...],
    k_list: list[int] | tuple[int, ...],
    smoothing: str = "none",
    **smooth_kwargs,
) -> plt.Figure:
    row_idx = get_row_index_by_source_id(bp_block, int(source_id))
    k_list = list(k_list)
    bases = list(bases)

    fig, axes = plt.subplots(
        2,
        len(k_list),
        figsize=(4.2 * len(k_list), 8),
        sharex=False,
        sharey="row",
    )
    if len(k_list) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col_idx, n_coeffs in enumerate(k_list):
        for row_axes, block, arm_label in (
            (axes[0, col_idx], bp_block, "BP"),
            (axes[1, col_idx], rp_block, "RP"),
        ):
            raw_flux = block.flux[row_idx]
            row_axes.plot(
                block.wavelengths,
                raw_flux,
                color="black",
                linewidth=2.0,
                label="raw",
            )
            smoothed_flux = smooth_flux(raw_flux, smoothing, **smooth_kwargs)
            if smoothing != "none":
                row_axes.plot(
                    block.wavelengths,
                    smoothed_flux,
                    color="#666666",
                    linewidth=1.4,
                    linestyle="--",
                    label=f"smoothed ({smoothing})",
                )
            for basis in bases:
                coeffs = fit_basis(block.wavelengths, smoothed_flux, basis, n_coeffs)
                recon = reconstruct_flux(block.wavelengths, coeffs, basis)
                row_axes.plot(
                    block.wavelengths,
                    recon,
                    linewidth=1.5,
                    color=PLOT_COLORS.get(basis, None),
                    label=basis,
                )
            row_axes.set_title(f"{arm_label}, K={n_coeffs}")
            row_axes.grid(alpha=0.2)
            row_axes.set_xlabel("Wavelength (nm)")
        axes[0, col_idx].set_ylabel("Flux")
        axes[1, col_idx].set_ylabel("Flux")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    fig.suptitle(f"source_id={int(source_id)} reconstruction comparison", y=1.02)
    fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper center", ncol=min(6, len(unique_labels)))
    fig.tight_layout()
    return fig


def plot_selected_star_reconstructions_interactive(
    bp_block: SpectrumBlock,
    rp_block: SpectrumBlock,
    source_id: int,
    bases: list[str] | tuple[str, ...],
    k_list: list[int] | tuple[int, ...],
    smoothing: str = "none",
    **smooth_kwargs,
):
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    row_idx = get_row_index_by_source_id(bp_block, int(source_id))
    k_list = list(k_list)
    bases = list(bases)

    n_cols = 2
    k_rows = math.ceil(len(k_list) / n_cols)
    n_rows = k_rows * 2

    subplot_titles: list[str] = []
    for arm in ("BP", "RP"):
        for k in k_list:
            subplot_titles.append(f"{arm}, K={k}")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    shown_legends: set[str] = set()

    for arm_idx, (block, arm_label) in enumerate(
        [(bp_block, "BP"), (rp_block, "RP")]
    ):
        raw_flux = block.flux[row_idx]
        for k_idx, n_coeffs in enumerate(k_list):
            row = arm_idx * k_rows + k_idx // n_cols + 1
            col = k_idx % n_cols + 1

            fig.add_trace(
                go.Scattergl(
                    x=block.wavelengths,
                    y=raw_flux,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="raw",
                    legendgroup="raw",
                    showlegend="raw" not in shown_legends,
                ),
                row=row,
                col=col,
            )
            shown_legends.add("raw")

            smoothed_flux = smooth_flux(raw_flux, smoothing, **smooth_kwargs)
            if smoothing != "none":
                legend_name = f"smoothed ({smoothing})"
                fig.add_trace(
                    go.Scattergl(
                        x=block.wavelengths,
                        y=smoothed_flux,
                        mode="lines",
                        line=dict(color="#666666", width=1.4, dash="dash"),
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=legend_name not in shown_legends,
                    ),
                    row=row,
                    col=col,
                )
                shown_legends.add(legend_name)

            for basis in bases:
                coeffs = fit_basis(block.wavelengths, smoothed_flux, basis, n_coeffs)
                recon = reconstruct_flux(block.wavelengths, coeffs, basis)
                fig.add_trace(
                    go.Scattergl(
                        x=block.wavelengths,
                        y=recon,
                        mode="lines",
                        line=dict(
                            color=PLOT_COLORS.get(basis, None),
                            width=1.5,
                        ),
                        name=basis,
                        legendgroup=basis,
                        showlegend=basis not in shown_legends,
                    ),
                    row=row,
                    col=col,
                )
                shown_legends.add(basis)

    fig.update_layout(
        title_text=f"source_id={int(source_id)} reconstruction comparison",
        height=300 * n_rows,
        width=950,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="Flux")
    return fig


def main() -> None:
    args = parse_args()
    bp = load_block(BP_SAMPLED_CSV)
    rp = load_block(RP_SAMPLED_CSV)
    check_alignment(bp, rp)

    reconstruction_dir = RESULTS_DIR / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    preview_idx = choose_example_indices(bp.labels, n_examples=args.preview_stars)
    preview_cache: dict[tuple[str, str, int, str], np.ndarray] = {}

    generated = 0
    metric_frames: list[pd.DataFrame] = []

    for basis in args.bases:
        for smoothing in args.smoothing:
            for n_coeffs in args.k_list:
                out_path = feature_path(basis, smoothing, n_coeffs)
                if out_path.exists() and not args.force:
                    print(
                        f"Refreshing analysis for existing file: {out_path.name} "
                        f"(use --force only if you want to explicitly recompute everything)"
                    )
                else:
                    print(f"Fitting basis={basis:10s} smoothing={smoothing:8s} K={n_coeffs:02d}")
                bp_fit = build_block_fit(bp, basis, smoothing, n_coeffs)
                rp_fit = build_block_fit(rp, basis, smoothing, n_coeffs)

                feat_df = flatten_feature_blocks(
                    bp.source_ids,
                    bp.labels,
                    bp_fit.coeffs,
                    rp_fit.coeffs,
                )
                coeff_cols = [c for c in feat_df.columns if c.startswith("c")]
                feat_df = l2_normalize(feat_df, coeff_cols=coeff_cols)
                feat_df.to_csv(out_path, index=False)

                metric_frames.append(
                    metric_frame(
                        bp,
                        arm="bp",
                        basis=basis,
                        smoothing=smoothing,
                        n_coeffs=n_coeffs,
                        smoothed_flux=bp_fit.smoothed_flux,
                        reconstructed_flux=bp_fit.reconstructed_flux,
                    )
                )
                metric_frames.append(
                    metric_frame(
                        rp,
                        arm="rp",
                        basis=basis,
                        smoothing=smoothing,
                        n_coeffs=n_coeffs,
                        smoothed_flux=rp_fit.smoothed_flux,
                        reconstructed_flux=rp_fit.reconstructed_flux,
                    )
                )
                metric_frames.append(
                    combined_metric_frame(
                        bp,
                        rp,
                        basis=basis,
                        smoothing=smoothing,
                        n_coeffs=n_coeffs,
                        bp_smoothed_flux=bp_fit.smoothed_flux,
                        rp_smoothed_flux=rp_fit.smoothed_flux,
                        bp_reconstructed_flux=bp_fit.reconstructed_flux,
                        rp_reconstructed_flux=rp_fit.reconstructed_flux,
                    )
                )

                preview_cache[(basis, smoothing, n_coeffs, "bp")] = bp_fit.reconstructed_flux[preview_idx].copy()
                preview_cache[(basis, smoothing, n_coeffs, "rp")] = rp_fit.reconstructed_flux[preview_idx].copy()
                generated += 1

    if not metric_frames:
        raise RuntimeError(
            "No basis features were generated. Use --force to recompute existing outputs and refresh summaries."
        )

    raw_metrics = pd.concat(metric_frames, ignore_index=True)
    raw_metrics_path = reconstruction_dir / "reconstruction_metrics_by_star.csv"
    raw_metrics.to_csv(raw_metrics_path, index=False)

    summary = summarize_metric_frame(raw_metrics)
    summary_path = reconstruction_dir / "reconstruction_summary.csv"
    summary.to_csv(summary_path, index=False)

    combined_ranking = rank_combined_configs(summary)
    combined_ranking_path = reconstruction_dir / "reconstruction_summary_combined_ranked.csv"
    combined_ranking.to_csv(combined_ranking_path, index=False)

    metric_plot_paths = plot_metric_curves(summary, reconstruction_dir)
    best_bp_plot = plot_best_reconstructions(
        arm="bp",
        block=bp,
        summary_df=summary,
        preview_cache=preview_cache,
        preview_idx=preview_idx,
        output_dir=reconstruction_dir,
        top_n=args.top_configs,
    )
    best_rp_plot = plot_best_reconstructions(
        arm="rp",
        block=rp,
        summary_df=summary,
        preview_cache=preview_cache,
        preview_idx=preview_idx,
        output_dir=reconstruction_dir,
        top_n=args.top_configs,
    )

    save_manifest(
        {
            "generated_feature_files": generated,
            "feature_dir": str(FEATURES_DIR),
            "reconstruction_dir": str(reconstruction_dir),
            "reconstruction_metrics_by_star": str(raw_metrics_path),
            "reconstruction_summary": str(summary_path),
            "reconstruction_summary_combined_ranked": str(combined_ranking_path),
            "bases": list(args.bases),
            "smoothing": list(args.smoothing),
            "k_list": list(args.k_list),
            "preview_source_ids": bp.source_ids[preview_idx].tolist(),
            "plot_files": [str(p) for p in metric_plot_paths],
            "best_reconstruction_plots": [
                str(path) for path in (best_bp_plot, best_rp_plot) if path is not None
            ],
        }
    )

    print(f"Saved feature files          -> {FEATURES_DIR}")
    print(f"Saved per-star metrics       -> {raw_metrics_path}")
    print(f"Saved reconstruction summary -> {summary_path}")
    print(f"Saved combined ranking       -> {combined_ranking_path}")
    if metric_plot_paths:
        print(f"Saved metric plots           -> {reconstruction_dir}")
    if best_bp_plot or best_rp_plot:
        print(f"Saved best-example plots     -> {reconstruction_dir}")


if __name__ == "__main__":
    main()
