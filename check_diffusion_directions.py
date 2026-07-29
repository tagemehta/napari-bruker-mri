"""
Visual check for per-direction diffusion signal / ADC — run with:
    uv run python check_diffusion_directions.py

This dataset's DWI acquisition uses 3 diffusion directions x 5 b-values
(0/200/400/800/1200 s/mm^2). PV's ADC map (proc 2, "dtraceb" macro) combines
all 3 directions into a single orientation-averaged ("trace") ADC value per
pixel. That's the right model for this acquisition (see AGENTS.md), but it
also means an "interesting" ADC finding could be a real isotropic diffusivity
change OR an artifact of tissue anisotropy (the AF's collagen lamellae are
highly orientation-dependent) that happens to show up differently depending
on how the disc was angled relative to the 3 gradient directions.

This script fits ADC *per direction* instead of combining them, so you can
see directly whether a region's apparent ADC depends on which direction you
look from:
  - Direction 1/2/3 ADC maps  — fit independently from each direction's own
    5-point (b=0..1200) signal decay.
  - Direction spread (max-min) — large spread = direction-dependent
    (anisotropy/orientation artifact candidate). Near-zero spread = isotropic,
    supports a true biological ADC change.
  - PV's combined ADC (proc 2), for reference, if available.
  - High-b anatomy image per direction, for a visual sanity check.

Per-direction ADC is fit with the same validated NLLS model used for the
combined ("trace") map (`analysis/fitting.py::fit_adc_monoexp`, bias fixed
at 0), just applied to each direction's own 5-point (b=0..1200) sub-stack
instead of all 3 directions pooled — so these numbers are directly
comparable to PV's combined ADC.

Guarded by ``if __name__ == "__main__":`` since fit_adc_monoexp() may
parallelize the fit across worker processes (spawn re-imports this file on
macOS/Windows).
"""

import logging
import re
from pathlib import Path

import napari
import numpy as np

from analysis.fitting import fit_adc_monoexp
from analysis.maps import _read_dw_eff_bval, get_pv_adc_map, match_slices_to_proc1
from bruker_browser.loader import load_2dseq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── change these to explore different studies / discs ──────────────────────
STUDY_PATH = "patients/real_data/RA_2022_07_20.f22"
EXP_ID = "11"
STUDY_NAME = "RA_2022_07_20"
# ───────────────────────────────────────────────────────────────────────────

_ADC_CONTRAST = [0.0002, 0.0025]  # mm^2/s, typical IVD ADC range at 9.4T


def _estimate_noise_sigma(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Estimate noise sigma from two independently-acquired images of the same
    (near-zero-b) condition, via the classic difference-image method:
    sigma = std(A - B) / sqrt(2).

    This dataset's diffusion series has no reliable background/air region
    to sample (small cropped EPI FOV, tissue fills the corners — verified
    against this acquisition), so a corner-patch or intensity-percentile
    background estimate is not usable here. Instead we exploit that each
    direction's own lowest b-value (~3 s/mm^2) is an independent repeat
    measurement of essentially the same signal (diffusion attenuation at
    b~3 is <0.5% for typical IVD ADC, negligible next to measurement
    noise) — so the two images differ only by noise, and their difference
    directly reveals sigma without any spatial assumption.
    """
    diff = frame_a.astype(float) - frame_b.astype(float)
    return float(np.std(diff) / np.sqrt(2))


def main() -> None:
    study_path = Path(STUDY_PATH)

    print(f"Loading DWI stack (proc 1) for {STUDY_NAME} exp {EXP_ID}...")
    raw = load_2dseq(study_path, EXP_ID, proc_id="1", study_name=STUDY_NAME, zoom_to=0)
    movie_fg = next(fg for fg in raw.frame_groups if "MOVIE" in fg.name or "DIFFUSION" in fg.name)
    b_values = _read_dw_eff_bval(study_path / EXP_ID / "method")
    if len(b_values) != movie_fg.size:
        raise RuntimeError(
            f"PVM_DwEffBval has {len(b_values)} values but frame group has "
            f"{movie_fg.size} frames — acquisition layout differs from expected."
        )

    frames = np.moveaxis(raw.data, movie_fg.axis, 0)  # (n_frames, [Z,] H, W)

    # Group frame indices by direction number parsed from the FG label
    # ("Dir 1 B 3", "Dir 2 B 217", ... ; "A0 1 B 3" is the shared b~0 image).
    direction_indices: dict[int, list[int]] = {}
    for i, label in enumerate(movie_fg.labels):
        m = re.search(r"Dir (\d+)", label)
        dir_num = int(m.group(1)) if m else 0
        direction_indices.setdefault(dir_num, []).append(i)

    dirs = sorted(k for k in direction_indices if k != 0)
    print(f"Found directions: {dirs} ({len(dirs)} x {len(direction_indices[dirs[0]])} b-values each)")

    print("Fitting per-direction ADC (fit_adc_monoexp, bias fixed at 0)...")
    direction_adc = {}
    direction_r2 = {}
    direction_hi_b = {}
    for d in dirs:
        idx = direction_indices[d]
        b_this_dir = [b_values[i] for i in idx]
        stack = frames[idx]
        fit = fit_adc_monoexp(stack, b_this_dir)
        direction_adc[d] = fit.adc
        direction_r2[d] = fit.r2
        direction_hi_b[d] = stack[np.argmax(b_this_dir)]  # highest-b image, for anatomy

    spread = np.nanmax(np.stack(list(direction_adc.values())), axis=0) - np.nanmin(
        np.stack(list(direction_adc.values())), axis=0
    )

    print("Estimating noise level from two near-b0 repeat acquisitions...")
    # The global "A0" frame (dir 0) and each direction's own lowest-b frame
    # are independent measurements of essentially the same (b~3) condition —
    # use them as a noise-estimation pair. Fall back to two directions' own
    # low-b frames if there's no separate A0 label.
    if 0 in direction_indices:
        ref_a = frames[direction_indices[0][0]]
        ref_b = frames[direction_indices[dirs[0]][0]]
    else:
        ref_a = frames[direction_indices[dirs[0]][0]]
        ref_b = frames[direction_indices[dirs[1]][0]]
    sigma = _estimate_noise_sigma(ref_a, ref_b)
    print(f"  sigma (noise) ~ {sigma:.1f} (raw intensity units)")

    # SNR at the *highest* b-value per direction — this is where a bias=0
    # mono-exponential model is most exposed to the Rician noise floor,
    # since that's the point closest to (or below) the noise plateau.
    snr_highb = {d: direction_hi_b[d] / sigma for d in dirs}

    print()
    print("Noise-floor artifact check (ADC vs SNR at highest b):")
    for d in dirs:
        adc = direction_adc[d]
        snr = snr_highb[d]
        valid = np.isfinite(adc) & np.isfinite(snr) & (snr > 0)
        if valid.sum() < 20:
            print(f"  Dir {d}: not enough valid pixels to check")
            continue
        a, s = adc[valid], snr[valid]
        corr = float(np.corrcoef(a, s)[0, 1])
        hi_thresh = np.percentile(a, 90)
        hi_mask = a >= hi_thresh
        print(
            f"  Dir {d}: corr(ADC, SNR)={corr:+.2f}   "
            f"median SNR in top-10% ADC={np.median(s[hi_mask]):.1f}   "
            f"median SNR elsewhere={np.median(s[~hi_mask]):.1f}"
        )
    print()
    print("  Interpretation:")
    print("  - Strong negative correlation (roughly < -0.3) AND top-10%-ADC")
    print("    pixels having noticeably lower SNR than the rest => the high")
    print("    ADC values are likely a noise-floor fitting artifact (the")
    print("    bias=0 model can't represent a noise plateau, so it distorts")
    print("    the fitted decay rate), not a real diffusivity difference.")
    print("  - Weak/near-zero correlation, or top-10%-ADC pixels with SNR")
    print("    similar to (or higher than) the rest => the elevated ADC is")
    print("    NOT explained by low SNR — more likely genuine biology.")
    print("  - Rule of thumb: SNR < ~3 at the highest b-value is where Rician")
    print("    noise-floor bias becomes significant for this kind of fit.")

    print("Looking for PV's combined ADC map (proc 2)...")
    pv_adc = get_pv_adc_map(study_path, EXP_ID, STUDY_NAME)

    scale = raw.scale[: frames.ndim - 1] if frames.ndim > 3 else raw.scale
    # drop the direction/movie axis from scale (already excluded by moveaxis+indexing)
    scale = tuple(s for ax, s in enumerate(raw.scale) if ax != movie_fg.axis)

    viewer = napari.Viewer(title=f"Diffusion directions — {STUDY_NAME} exp {EXP_ID}")

    for d in dirs:
        viewer.add_image(
            direction_hi_b[d],
            name=f"Dir {d} — high-b anatomy",
            scale=scale,
            colormap="gray",
            visible=(d == dirs[0]),
        )

    for d in dirs:
        viewer.add_image(
            direction_adc[d],
            name=f"ADC — direction {d}",
            scale=scale,
            colormap="magma",
            contrast_limits=_ADC_CONTRAST,
        )

    viewer.add_image(
        spread,
        name="Direction spread (max-min ADC)",
        scale=scale,
        colormap="turbo",
        contrast_limits=[0.0, 0.001],
        visible=False,
    )

    for d in dirs:
        viewer.add_image(
            direction_r2[d],
            name=f"R² — direction {d}",
            scale=scale,
            colormap="viridis",
            contrast_limits=[0.8, 1.0],
            visible=False,
        )

    for d in dirs:
        viewer.add_image(
            snr_highb[d],
            name=f"SNR at highest b — direction {d}",
            scale=scale,
            colormap="viridis",
            contrast_limits=[0.0, 8.0],
            visible=False,
        )

    low_snr_mask = np.any(
        np.stack([snr_highb[d] < 3.0 for d in dirs]), axis=0
    ).astype(np.uint8)
    viewer.add_labels(
        low_snr_mask,
        name="Low-SNR risk (<3 at highest b, any direction)",
        scale=scale,
        visible=False,
    )

    if pv_adc is not None:
        data = pv_adc.data
        label = pv_adc.meta.get("pv_label", "?")
        print(f"  PV ADC label: {label!r}  shape={data.shape}")
        if data.shape != spread.shape:
            print("  Shape differs from proc 1 — matching slices via VisuCorePosition...")
            slice_idx = match_slices_to_proc1(study_path, EXP_ID, proc_id="2")
            print(f"  proc-2 slice(s) -> proc-1 slice index(es): {slice_idx}")
        viewer.add_image(
            data,
            name="ADC — PV combined (trace)",
            scale=scale,
            colormap="magma",
            contrast_limits=_ADC_CONTRAST,
        )
    else:
        print("  No PV ADC map available for this experiment.")

    print()
    print("Napari open.")
    print("  - Compare 'ADC — direction N' layers: consistent values across")
    print("    directions support true isotropic biology; large differences")
    print("    suggest anisotropy/orientation is driving the result.")
    print("  - Enable 'Direction spread' to see where directions disagree most.")
    print("  - 'ADC — PV combined (trace)' is what PV reports as THE adc map.")

    napari.run()


if __name__ == "__main__":
    main()
