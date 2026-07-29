"""
Quick visual check for T2 maps — run with:
    uv run python check_t2_map.py

Opens napari with layers for one disc, all pixel-aligned to the same slice(s):
  1. Echo 1 image (raw signal, proc 1)      — sanity check: does the anatomy look right?
  2. PV T2 map (proc 2, ISA "T2 relaxation time")
  3. Freshly fitted T2 map (from proc 1 echo stack)
  4. Difference (fitted - PV)                — should be near-zero in disc tissue
  5. R² map from the fit                    — low R² reveals non-mono-exponential voxels
  6. Bias (fitted noise-floor term A)       — large values flag low-SNR pixels

PV's proc-2 map is not always the same shape as proc 1's echo stack (it may
cover only one slice, or the same slices in a different order) — see
AGENTS.md "Two T2-map layouts".  This script uses
``analysis.maps.match_slices_to_proc1`` to find the correct proc-1 slice(s)
for each PV map slice (matched on VisuCorePosition) so every layer here
lines up voxel-for-voxel and only the necessary slice(s) are fitted.

What to look for
----------------
- PV and fitted T2 maps should closely agree in disc tissue (NP/AF) — this
  fit reproduces PV's own macro (see analysis/fitting.py docstring), so
  differences beyond ~1-2 ms in plausible-range tissue are worth investigating.
- Background pixels: PV map may have spurious high T2 values; fitted map
  should have those clipped to NaN (shown as transparent in napari).
- R² < 0.9 in disc tissue warrants investigation (partial volume, B1 issues).
- Typical IVD T2 values at 9.4T: NP ~60–100 ms, AF ~30–60 ms.

The body below is guarded by ``if __name__ == "__main__":`` because
fit_t2_monoexp() may parallelize the fit across worker processes.  On macOS
/ Windows those workers re-import this file ("spawn" start method) — without
the guard they would re-run the whole script (re-open napari, etc).
"""

import logging

import napari
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from pathlib import Path

from analysis.fitting import T2_MAX_MS, T2_MIN_MS, fit_t2_monoexp
from analysis.maps import _parse_te_labels, get_pv_t2_map, match_slices_to_proc1
from bruker_browser.loader import load_2dseq

# ── change these to explore different studies / discs ──────────────────────
STUDY_PATH = "patients/real_data/RA_2022_07_20.f21"
EXP_ID = "9"
STUDY_NAME = "RA_2022_07_20"
# ───────────────────────────────────────────────────────────────────────────


def main() -> None:
    study_path = Path(STUDY_PATH)

    print(f"Loading echo stack (proc 1) for {STUDY_NAME} exp {EXP_ID}...")
    raw = load_2dseq(study_path, EXP_ID, proc_id="1", study_name=STUDY_NAME, zoom_to=0)
    echo_fg = next(fg for fg in raw.frame_groups if "ECHO" in fg.name)
    te_ms = _parse_te_labels(echo_fg.labels)
    echoes_full = np.moveaxis(raw.data, echo_fg.axis, 0)  # (n_echo, [Z,] H, W)

    print("Loading PV T2 map (proc 2)...")
    pv_map = get_pv_t2_map(study_path, EXP_ID, study_name=STUDY_NAME)
    if pv_map is None:
        raise SystemExit(
            f"No PV T2 map available for {STUDY_NAME} exp {EXP_ID} — nothing to compare against."
        )
    print(f"  shape={pv_map.data.shape}")

    print("Matching PV map slice(s) to proc-1 slice indices (by VisuCorePosition)...")
    slice_idx = match_slices_to_proc1(study_path, EXP_ID, proc_id="2")
    print(f"  PV map slice(s) -> proc-1 slice index(es): {slice_idx}")

    # Select only the matched slice(s) from the echo stack, in PV's slice order,
    # so every layer below shares PV's exact shape.
    if echoes_full.ndim == 4:  # (n_echo, Z, H, W)
        echoes = echoes_full[:, slice_idx, :, :]
        if len(slice_idx) == 1:
            echoes = echoes[:, 0]  # drop trivial Z axis to match a single-slice PV map
        first_echo = echoes[0]
    else:  # (n_echo, H, W) — already single-slice
        echoes = echoes_full
        first_echo = echoes[0]

    print(f"Fitting T2 on the matched slice(s) (shape {echoes.shape[1:]})...")
    fit = fit_t2_monoexp(echoes, te_ms)
    print(
        f"  Fitted: n_fitted={fit.n_fitted}  n_out_of_range={fit.n_out_of_range}  n_failed={fit.n_failed}"
    )

    # ── Quantitative comparison over pixels PV considers plausible ─────────
    valid = (
        np.isfinite(pv_map.data)
        & (pv_map.data >= T2_MIN_MS)
        & (pv_map.data <= T2_MAX_MS)
    )
    diff = fit.t2[valid] - pv_map.data[valid]
    diff = diff[np.isfinite(diff)]
    if diff.size:
        print()
        print(f"PV vs fitted over {diff.size} plausible-range pixels:")
        print(f"  median|diff| = {np.median(np.abs(diff)):.2f} ms")
        print(f"  within 1 ms  = {np.mean(np.abs(diff) < 1) * 100:.1f}%")
        print(f"  within 5 ms  = {np.mean(np.abs(diff) < 5) * 100:.1f}%")

    diff_map = fit.t2 - pv_map.data

    # ── Open napari ─────────────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"T2 check — {STUDY_NAME} exp {EXP_ID}")

    viewer.add_image(
        first_echo,
        name="Echo 1 (anatomy)",
        scale=pv_map.scale,
        colormap="gray",
        opacity=0.6,
    )

    viewer.add_image(
        pv_map.data,
        name="T2 map — PV proc 2",
        scale=pv_map.scale,
        colormap="magma",
        contrast_limits=[5, 200],
    )

    viewer.add_image(
        fit.t2,
        name="T2 map — fitted",
        scale=pv_map.scale,
        colormap="magma",
        contrast_limits=[5, 200],
    )

    viewer.add_image(
        diff_map,
        name="Difference (fitted - PV)",
        scale=pv_map.scale,
        colormap="turbo",
        contrast_limits=[-20, 20],
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
    print("  - Both T2 maps are set to the same colour scale (5-200 ms).")
    print("  - Difference / R² / Bias layers are hidden by default — enable to inspect.")
    print("  - Expected disc T2: NP ~60-100 ms, AF ~30-60 ms (9.4T)")

    napari.run()


if __name__ == "__main__":
    main()
