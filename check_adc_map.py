"""
Quick visual check for ADC maps — run with:
    uv run python check_adc_map.py

Opens napari with layers for one disc, all pixel-aligned to the same slice(s):
  1. High-b anatomy image (raw signal, proc 1)  — sanity check: does the anatomy look right?
  2. PV ADC map (proc 2, ISA "diffusion constant")
  3. Freshly fitted ADC map (from proc 1 DWI stack, all directions pooled/trace)
  4. Difference (fitted - PV)                — should be near-zero everywhere plausible
  5. R² map from the fit                    — low R² reveals non-mono-exponential voxels
  6. Bias (fitted noise-floor term A)       — only nonzero if bias_free=True

This fit reproduces PV's own "dtraceb" macro (bias fixed at 0, trace-corrected
b-values pooled across all 3 directions) — see analysis/fitting.py module
docstring and AGENTS.md. PV's proc-2 map is not always the same shape as
proc 1's DWI stack (see check_t2_map.py / AGENTS.md "Two T2-map layouts" for
the analogous slice-matching issue) — this script uses
``analysis.maps.match_slices_to_proc1`` to keep every layer voxel-aligned.

What to look for
-----------------
- PV and fitted ADC maps should closely agree — differences beyond
  ~5e-5 mm^2/s in plausible-range tissue are worth investigating.
- R² < 0.9 in disc tissue warrants investigation (partial volume, motion).
- Typical IVD ADC values at 9.4T: roughly 0.0005-0.0020 mm^2/s.
- For per-direction inspection (anisotropy vs true biology), use
  check_diffusion_directions.py instead.

The body below is guarded by ``if __name__ == "__main__":`` because
fit_adc_monoexp() may parallelize the fit across worker processes. On macOS
/ Windows those workers re-import this file ("spawn" start method) — without
the guard they would re-run the whole script (re-open napari, etc).
"""

import logging

import napari
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from pathlib import Path

from analysis.fitting import ADC_MAX_MM2S, ADC_MIN_MM2S, fit_adc_monoexp
from analysis.maps import _read_dw_eff_bval, get_adc_map, match_slices_to_proc1
from bruker_browser.loader import load_2dseq

# ── change these to explore different studies / discs ──────────────────────
STUDY_PATH = "patients/real_data/RA_2022_07_20.f22"
EXP_ID = "11"
STUDY_NAME = "RA_2022_07_20"
# ───────────────────────────────────────────────────────────────────────────

_ADC_CONTRAST = [0.0002, 0.0025]  # mm^2/s, typical IVD ADC range at 9.4T


def main() -> None:
    study_path = Path(STUDY_PATH)

    print(f"Loading DWI stack (proc 1) for {STUDY_NAME} exp {EXP_ID}...")
    raw = load_2dseq(study_path, EXP_ID, proc_id="1", study_name=STUDY_NAME, zoom_to=0)
    dw_fg = next(fg for fg in raw.frame_groups if "MOVIE" in fg.name or "DIFFUSION" in fg.name)
    b_values = _read_dw_eff_bval(study_path / EXP_ID / "method")
    frames = np.moveaxis(raw.data, dw_fg.axis, 0)  # (n_frames, [Z,] H, W)
    highb_anatomy = frames[int(np.argmax(b_values))]

    print("Loading PV ADC map (proc 2)...")
    pv_map = get_adc_map(study_path, EXP_ID, study_name=STUDY_NAME)
    print(f"  source={pv_map.source}  shape={pv_map.data.shape}")
    if pv_map.source != "pv":
        raise SystemExit(
            f"No PV ADC map available for {STUDY_NAME} exp {EXP_ID} — nothing to compare against."
        )

    print("Matching PV map slice(s) to proc-1 slice indices (by VisuCorePosition)...")
    slice_idx = match_slices_to_proc1(study_path, EXP_ID, proc_id="2")
    print(f"  PV map slice(s) -> proc-1 slice index(es): {slice_idx}")

    if frames.ndim == 4:  # (n_frames, Z, H, W)
        dwi = frames[:, slice_idx, :, :]
        if len(slice_idx) == 1:
            dwi = dwi[:, 0]  # drop trivial Z axis to match a single-slice PV map
            highb_anatomy = highb_anatomy[slice_idx[0]]
        else:
            highb_anatomy = highb_anatomy[slice_idx]
    else:  # (n_frames, H, W) — already single-slice
        dwi = frames

    print(f"Fitting ADC on the matched slice(s) (shape {dwi.shape[1:]})...")
    fit = fit_adc_monoexp(dwi, b_values)
    print(
        f"  Fitted: n_fitted={fit.n_fitted}  n_out_of_range={fit.n_out_of_range}  n_failed={fit.n_failed}"
    )

    # ── Quantitative comparison over pixels PV considers plausible ─────────
    valid = (
        np.isfinite(pv_map.data)
        & (pv_map.data >= ADC_MIN_MM2S)
        & (pv_map.data <= ADC_MAX_MM2S)
    )
    diff = fit.adc[valid] - pv_map.data[valid]
    diff = diff[np.isfinite(diff)]
    if diff.size:
        print()
        print(f"PV vs fitted over {diff.size} plausible-range pixels:")
        print(f"  median|diff| = {np.median(np.abs(diff)):.2e} mm^2/s")
        print(f"  within 5e-5   = {np.mean(np.abs(diff) < 5e-5) * 100:.1f}%")
        print(f"  within 1e-4   = {np.mean(np.abs(diff) < 1e-4) * 100:.1f}%")

    diff_map = fit.adc - pv_map.data

    # ── Open napari ─────────────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"ADC check — {STUDY_NAME} exp {EXP_ID}")

    viewer.add_image(
        highb_anatomy,
        name="Highest-b image (anatomy)",
        scale=pv_map.scale,
        colormap="gray",
        opacity=0.6,
    )

    viewer.add_image(
        pv_map.data,
        name="ADC map — PV proc 2",
        scale=pv_map.scale,
        colormap="magma",
        contrast_limits=_ADC_CONTRAST,
    )

    viewer.add_image(
        fit.adc,
        name="ADC map — fitted",
        scale=pv_map.scale,
        colormap="magma",
        contrast_limits=_ADC_CONTRAST,
    )

    viewer.add_image(
        diff_map,
        name="Difference (fitted - PV)",
        scale=pv_map.scale,
        colormap="turbo",
        contrast_limits=[-0.0002, 0.0002],
        visible=False,
    )

    viewer.add_image(
        fit.r2,
        name="R² (fit quality)",
        scale=pv_map.scale,
        colormap="viridis",
        contrast_limits=[0.8, 1.0],
        visible=False,
    )

    viewer.add_image(
        fit.bias,
        name="Bias (noise floor A)",
        scale=pv_map.scale,
        colormap="viridis",
        visible=False,
    )

    print()
    print("Napari open.")
    print("  - Toggle layers in the layer list to compare PV vs fitted.")
    print("  - Both ADC maps are set to the same colour scale (0.0002-0.0025 mm^2/s).")
    print("  - Difference / R² / Bias layers are hidden by default — enable to inspect.")
    print("  - Typical IVD ADC at 9.4T: ~0.0005-0.0020 mm^2/s")

    napari.run()


if __name__ == "__main__":
    main()
