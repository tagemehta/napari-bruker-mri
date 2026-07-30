"""
Visually inspect curve-fit quality for saved ROIs — run with:
    uv run python inspect_roi_fit.py

For each ROI saved on a disc's map (``analyze_disc.py``), plots the measured
signal decay (points, one series per pixel in the ROI) against each pixel's
own fitted curve (thin lines) — one subplot per ROI. Lets you see, region by
region, whether the mono-exponential model actually tracks the raw data or
is being pulled around by noisy/partial-volume pixels.

Requires a map saved with ``save_map`` *after* ``analysis.maps.MapResult``
gained its ``diagnostics`` field (s0/bias/r2 per pixel) — older saved maps
without a ``..._diagnostics.npz`` file will raise. Re-run ``analyze_disc.py``
to regenerate if needed.

Uses the same offset mono-exponential model as the fit itself:
    T2:  S(TE) = bias + (s0 - bias) * exp(-TE / T2)
    ADC: S(b)  = bias + (s0 - bias) * exp(-b * ADC)
(see analysis/fitting.py module docstring for why this form, not a
zero-intercept one).
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.maps import PARAMETER_MAPS_DIR, _parse_te_labels, _read_dw_eff_bval, load_map
from analysis.rois import load_rois, rasterize_roi
from bruker_browser.loader import load_2dseq

# ── change these to explore different studies / discs ──────────────────────
STUDY_PATH = "patients/real_data/RA_2022_07_20.f22"
STUDY_NAME = "Rat Spine 690731"
EXP_ID = "7"
KIND = "adc"  # "t2" or "adc"
TARGET_ROIS: list[str] | None = None  # None = every saved ROI for this map
# ───────────────────────────────────────────────────────────────────────────


def _predict_t2(te: np.ndarray, bias: float, s0: float, t2: float) -> np.ndarray:
    return bias + (s0 - bias) * np.exp(-te / t2)


def _predict_adc(b: np.ndarray, bias: float, s0: float, adc: float) -> np.ndarray:
    return bias + (s0 - bias) * np.exp(-b * adc)


def _load_raw_stack(study_path: Path, exp_id: str, study_name: str, kind: str):
    """
    Reload proc 1's raw signal stack and its x-axis (echo times or
    b-values), moved to axis 0: ``(n_points, *spatial)`` — matching the
    spatial shape the saved map/diagnostics already use.
    """
    result = load_2dseq(study_path, exp_id=exp_id, proc_id="1", study_name=study_name, zoom_to=0)

    if kind == "t2":
        fg = next((fg for fg in result.frame_groups if "ECHO" in fg.name), None)
        if fg is None:
            raise RuntimeError(f"No FG_ECHO axis found in {study_name} exp {exp_id}.")
        x = np.array(_parse_te_labels(fg.labels))
    else:
        fg = next(
            (fg for fg in result.frame_groups if "MOVIE" in fg.name or "DIFFUSION" in fg.name),
            None,
        )
        if fg is None:
            raise RuntimeError(f"No diffusion frame group found in {study_name} exp {exp_id}.")
        x = np.array(_read_dw_eff_bval(study_path / exp_id / "method"))

    stack = np.moveaxis(result.data, fg.axis, 0)  # (n_points, *spatial)
    return stack, x


def plot_roi_fit(ax, roi, stack: np.ndarray, x: np.ndarray, map_data: np.ndarray, diagnostics: dict, kind: str) -> None:
    """Plot one ROI's measured decay points + each pixel's fitted curve onto ``ax``."""
    mask = rasterize_roi(roi, map_data.shape)
    pixel_idx = np.argwhere(mask)

    predict = _predict_t2 if kind == "t2" else _predict_adc
    x_smooth = np.linspace(x.min(), x.max() * 1.05, 200)

    param = map_data  # t2 or adc, per pixel
    s0, bias, r2 = diagnostics["s0"], diagnostics["bias"], diagnostics["r2"]

    r2_vals = []
    for idx in map(tuple, pixel_idx):
        p = param[idx]
        if np.isnan(p):
            continue  # excluded/failed/out-of-range pixel — nothing to plot
        signal = stack[(slice(None), *idx)]
        ax.scatter(x, signal, s=10, alpha=0.35, color="tab:blue")
        ax.plot(x_smooth, predict(x_smooth, bias[idx], s0[idx], p), alpha=0.35, color="tab:orange", linewidth=1)
        r2_vals.append(r2[idx])

    n = len(r2_vals)
    mean_param = np.nanmean(param[mask])
    unit = "ms" if kind == "t2" else "mm^2/s"
    title = f"{roi.name} ({roi.tissue or '?'})  n={n}"
    subtitle = f"mean {kind.upper()}={mean_param:.4g}{unit}  mean R2={np.nanmean(r2_vals):.3f}" if n else "no fitted pixels"
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.set_xlabel("TE (ms)" if kind == "t2" else "b-value (s/mm^2)")
    ax.set_ylabel("signal (a.u.)")


def main() -> None:
    study_path = Path(STUDY_PATH)

    print(f"Loading saved {KIND.upper()} map for {STUDY_NAME} exp {EXP_ID}...")
    map_result = load_map(STUDY_NAME, EXP_ID, KIND, PARAMETER_MAPS_DIR)
    if map_result.diagnostics is None:
        raise RuntimeError(
            f"No saved fit diagnostics for {STUDY_NAME} exp {EXP_ID} ({KIND}) — "
            "re-run analyze_disc.py to regenerate this map with diagnostics."
        )

    rois = load_rois(STUDY_NAME, EXP_ID, KIND, PARAMETER_MAPS_DIR)
    if TARGET_ROIS:
        rois = [r for r in rois if r.name in TARGET_ROIS]
    if not rois:
        raise RuntimeError(f"No matching saved ROIs for {STUDY_NAME} exp {EXP_ID} ({KIND}).")

    print(f"Reloading raw proc 1 stack for {STUDY_NAME} exp {EXP_ID}...")
    stack, x = _load_raw_stack(study_path, EXP_ID, STUDY_NAME, KIND)

    n = len(rois)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, roi in zip(axes.flat, rois):
        plot_roi_fit(ax, roi, stack, x, map_result.data, map_result.diagnostics, KIND)
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{STUDY_NAME} exp {EXP_ID} ({KIND.upper()}) — measured signal (blue) vs fitted curve (orange), per pixel")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
