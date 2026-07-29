"""
T2 and ADC map fitting — pure numerical functions.

Design constraints
------------------
- No I/O, no Bruker dependencies, no napari.  Takes plain numpy arrays and
  returns plain numpy arrays.  This makes the functions independently testable
  and reusable outside the viewer.
- Caller supplies echo times in milliseconds and signal arrays in napari axis
  order (echoes first, spatial axes last).
- NaN is used throughout to signal "not fitted" — masked pixels, failed
  convergence, and out-of-range T2 values all produce NaN rather than 0 or a
  sentinel number, so downstream callers can use np.nanmean / np.nanstd safely.

T2 model: offset mono-exponential
----------------------------------
    S(TE) = A + C * exp(-TE / T2)

This matches ParaVision's own T2 fitting macro for this dataset (ISA function
``t2blmodes_RA``, decoded from ``pdata/2/isa``/``visu_pars`` in the raw study
folders).  Two details were reverse-engineered from PV's output and are
essential to reproducing it:

1. ``A`` is a free noise-floor / bias term, not fixed at zero.  Magnitude MR
   images never decay to true zero — late echoes sit on a positive noise
   floor — so a pure ``C * exp(-TE/T2)`` model systematically biases T2
   estimates in low-SNR disc tissue where several echoes are near the noise
   floor.
2. The **first echo's nominal TE is treated as ~0 ms** during fitting
   (``_FIRST_ECHO_ANCHOR_MS``), rather than its true value.  MSME's first
   echo is known to suffer stimulated-echo contamination; anchoring it at
   t≈0 uses its measured signal as an amplitude reference without letting it
   distort the fitted decay rate.

Validated against PV proc-2 "T2 relaxation time" maps across three studies
(RA_2022_07_20, RA_2022_07_21, RA_2022_07_27): median absolute error 0.02 ms,
>99% of pixels within 1 ms.  A handful of early studies (e.g. RA_2022_07_18)
used a simpler PV macro (``t2vtr``) with the bias term fixed at zero and the
true first-echo TE — that is the ``A → 0`` degenerate case of this same
model, so this fit generalizes across both.

Fitted per pixel using scipy.optimize.least_squares (Trust Region Reflective,
with an analytic Jacobian).  Initial guesses are derived from the data itself
(linear regression on log-signal, skipping the anchored first echo) so the
fit converges reliably across a wide range of T2 values.

Noise masking
-------------
Pixels whose first-echo signal is below `noise_threshold` (fraction of the
per-slice maximum) are excluded from fitting and returned as NaN.  Default
0.05 (5 %).  This prevents noise-floor pixels from producing nonsense T2
values that would bias ROI statistics.

T2 plausibility bounds
----------------------
Physically plausible range for IVD tissue at 9.4 T: 5–200 ms.
Fitted T2 values outside this range are replaced with NaN and flagged in
T2FitResult.n_out_of_range so the caller can assess data quality.

ADC model: trace-weighted mono-exponential
-------------------------------------------
    S(b) = C * exp(-b * D)          [bias fixed at 0 by default]

This matches ParaVision's own ADC macro for this dataset (ISA function
``dtraceb``, "Diffusion using trace of B-Matrix", decoded from
``pdata/2/isa``). The acquisition here (3 orthogonal diffusion directions x
5 b-values: 0/200/400/800/1200 s/mm^2, 16 images total) only supports an
orientation-averaged ("trace") ADC — not a full diffusion tensor, IVIM, or
kurtosis fit, all of which need denser angular/b-value sampling than this
protocol provides. PV fits all 16 images together against each image's
trace-corrected effective b-value (``PVM_DwEffBval`` — already accounts for
B-matrix cross-terms, not just the nominal gradient b), with the noise-floor
bias term fixed at zero (unlike the T2 macro used in this dataset).

``fit_adc_monoexp`` reproduces this by default (``bias_free=False``). Pass
``bias_free=True`` to free the offset term as a diagnostic: if an ADC value
changes substantially between the two modes, the fixed-bias fit was likely
distorted by the Rician noise floor in that pixel (magnitude images never
truly decay to zero at high b in low-SNR tissue) rather than reflecting a
clean diffusion-driven decay.

Caveat: with only 5 b-values per direction (or 16 pooled across all 3
directions for the combined trace fit), the 3-parameter bias-free model is
only weakly constrained — on real data its ADC estimates can differ from
the fixed-bias fit by an amount comparable to the ADC value itself, and R²
improves only modestly (e.g. median 0.86 -> 0.91 in one tested dataset).
Use ``bias_free=True`` as a *sensitivity check* (does the elevated-ADC
region move when bias is freed, yes/no?) rather than as a trustworthy
absolute ADC number in its own right.

Because this is a 2-parameter model (``bias_free=False``) or a 3-parameter
one (``bias_free=True``), no first-echo-style anchoring trick is needed —
every b-value's own signal is used directly, unlike the T2 fit.

ADC plausibility bounds
------------------------
Generous physical range covering restricted-to-free diffusion at ~room/body
temperature: 0.00001–0.004 mm^2/s. Fitted ADC values outside this range are
replaced with NaN and flagged in ADCFitResult.n_out_of_range.

Per-direction fitting
----------------------
``fit_adc_monoexp`` fits whatever stack of (b, signal) pairs it's given — it
has no built-in notion of "direction". To fit ADC per diffusion direction
instead of PV's combined trace fit (e.g. to check whether an elevated ADC
region is consistent across directions or is an anisotropy/orientation
artifact — AF's collagen lamellae are highly anisotropic), call this
function separately for each direction's own 5-image sub-stack and its own
b-values.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

_log = logging.getLogger(__name__)

# Physical plausibility bounds for IVD T2 at 9.4 T
T2_MIN_MS: float = 5.0
T2_MAX_MS: float = 200.0

# Physical plausibility bounds for ADC (mm^2/s) — restricted to free diffusion
ADC_MIN_MM2S: float = 0.00001
ADC_MAX_MM2S: float = 0.004

# Default noise-floor threshold: pixels below this fraction of the per-slice
# maximum first-echo signal are excluded from fitting.
DEFAULT_NOISE_THRESHOLD: float = 0.05

# The first echo's nominal TE is replaced by this near-zero value during
# fitting, matching PV's t2blmodes_RA convention (see module docstring).
_FIRST_ECHO_ANCHOR_MS: float = 1e-6

# Minimum number of pixels to fit before spinning up a process pool.  Below
# this, ProcessPoolExecutor's startup/pickling overhead outweighs the gain
# from parallel fitting.
_PARALLEL_THRESHOLD: int = 2000


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class T2FitResult:
    """
    Output of :func:`fit_t2_monoexp`.

    All spatial arrays share the same shape as the input spatial dimensions.
    NaN marks pixels that were masked, failed to converge, or produced a T2
    outside the plausibility bounds.

    Attributes
    ----------
    t2:
        T2 map in milliseconds.
    s0:
        Extrapolated signal at TE=0 (``A + C``, arbitrary units matching
        input).
    bias:
        Fitted noise-floor / offset term ``A`` in ``S = A + C * exp(-TE/T2)``.
        Should be small relative to ``s0`` for good-SNR pixels; large values
        indicate the pixel is dominated by noise floor.
    r2:
        Coefficient of determination (0–1) for each pixel fit.  Low values
        indicate poor mono-exponential behaviour (e.g. B1 inhomogeneity,
        partial volume).
    noise_mask:
        Boolean array — True where pixel was below the noise threshold and
        therefore not fitted.
    n_fitted:
        Number of pixels that were fitted (not masked, converged).
    n_failed:
        Number of pixels that were attempted but failed to converge.
    n_out_of_range:
        Number of fitted pixels whose T2 fell outside [T2_MIN_MS, T2_MAX_MS]
        and were replaced with NaN.
    te_ms:
        Echo times used for fitting (copy of input, true values — the
        first-echo anchoring described in the module docstring is applied
        internally and not reflected here).
    """

    t2: np.ndarray
    s0: np.ndarray
    bias: np.ndarray
    r2: np.ndarray
    noise_mask: np.ndarray
    n_fitted: int
    n_failed: int
    n_out_of_range: int
    te_ms: list[float]


@dataclass
class ADCFitResult:
    """
    Output of :func:`fit_adc_monoexp`.

    All spatial arrays share the same shape as the input spatial dimensions.
    NaN marks pixels that were masked, failed to converge, or produced an
    ADC outside the plausibility bounds.

    Attributes
    ----------
    adc:
        ADC map in mm^2/s.
    s0:
        Extrapolated signal at b=0 (``A + C``, arbitrary units matching
        input).
    bias:
        Fitted noise-floor / offset term ``A``. Fixed at 0 unless
        ``bias_free=True`` was passed to :func:`fit_adc_monoexp`.
    r2:
        Coefficient of determination (0-1) for each pixel fit. Low values
        indicate non-mono-exponential diffusion behaviour (restricted
        compartments, partial volume) that this simple model can't capture.
    noise_mask:
        Boolean array — True where pixel was below the noise threshold and
        therefore not fitted.
    n_fitted:
        Number of pixels that were fitted (not masked, converged).
    n_failed:
        Number of pixels that were attempted but failed to converge.
    n_out_of_range:
        Number of fitted pixels whose ADC fell outside
        [ADC_MIN_MM2S, ADC_MAX_MM2S] and were replaced with NaN.
    b_values:
        b-values used for fitting (copy of input, s/mm^2).
    """

    adc: np.ndarray
    s0: np.ndarray
    bias: np.ndarray
    r2: np.ndarray
    noise_mask: np.ndarray
    n_fitted: int
    n_failed: int
    n_out_of_range: int
    b_values: list[float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_t2_monoexp(
    echoes: np.ndarray,
    te_ms: list[float],
    noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
    n_jobs: int | None = None,
) -> T2FitResult:
    """
    Fit an offset mono-exponential T2 decay pixel-by-pixel.

    Parameters
    ----------
    echoes:
        Signal array with echoes on axis 0 and spatial axes after.
        Shape: ``(n_echoes, *spatial)`` — e.g. ``(16, 8, 48, 48)`` for a
        16-echo, 8-slice, 48×48 dataset.  All spatial layouts are supported
        (single slice, multi-slice, 2-D, 3-D).
        **Must be at native (non-upsampled) resolution.**
    te_ms:
        Echo times in milliseconds, one per echo.  Length must equal
        ``echoes.shape[0]``.  The smallest value is treated as the first
        echo and anchored to ~0 ms during fitting (see module docstring).
    noise_threshold:
        Pixels whose first-echo signal is below
        ``noise_threshold * slice_max`` are excluded (NaN in output).
        Default 0.05 (5 % of per-slice maximum).
    n_jobs:
        Number of worker processes for the per-pixel fit.  ``None``
        (default) uses all available CPU cores, but only when there are
        enough pixels to fit (>= ``_PARALLEL_THRESHOLD``) to outweigh
        process-pool startup cost; otherwise it runs serially in the
        current process.  Pass ``1`` to force serial execution.

        Multiprocessing here uses the "spawn" start method on macOS and
        Windows, which re-imports the calling script in each worker.  Any
        script that calls this with more than one process (directly or via
        :func:`~analysis.maps.fit_t2_map`) **must** guard its top-level
        execution with ``if __name__ == "__main__":`` — see ``main.py`` /
        ``check_maps.py`` — or the workers will re-run the whole script.

    Returns
    -------
    T2FitResult
        Maps and quality statistics.  Spatial shape matches ``echoes[0]``.
    """
    if len(te_ms) != echoes.shape[0]:
        raise ValueError(
            f"len(te_ms)={len(te_ms)} does not match echoes.shape[0]={echoes.shape[0]}"
        )
    if len(te_ms) < 3:
        raise ValueError("At least 3 echoes are required for a reliable T2 fit.")

    te = np.array(te_ms, dtype=float)
    first_idx = int(np.argmin(te))
    te_fit = te.copy()
    te_fit[first_idx] = _FIRST_ECHO_ANCHOR_MS

    spatial_shape = echoes.shape[1:]
    n_pixels = int(np.prod(spatial_shape))

    # Flatten to (n_echoes, n_pixels) for vectorised masking
    flat = echoes.reshape(echoes.shape[0], n_pixels).astype(float)

    # --- Noise mask: per-slice if multi-slice, global otherwise ---
    noise_mask_flat = _build_noise_mask(flat, spatial_shape, noise_threshold, first_idx)

    # Output arrays (NaN = not fitted)
    t2_flat = np.full(n_pixels, np.nan)
    s0_flat = np.full(n_pixels, np.nan)
    bias_flat = np.full(n_pixels, np.nan)
    r2_flat = np.full(n_pixels, np.nan)

    fit_indices = np.where(~noise_mask_flat)[0]
    n_to_fit = len(fit_indices)

    workers = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, n_to_fit))

    if n_to_fit == 0:
        n_failed = 0
    elif workers > 1 and n_to_fit >= _PARALLEL_THRESHOLD:
        chunks = np.array_split(fit_indices, workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunk_results = list(
                executor.map(_fit_columns, [flat[:, c] for c in chunks], [te_fit] * len(chunks))
            )
        n_failed = 0
        for chunk, (t2_c, s0_c, bias_c, r2_c, nf) in zip(chunks, chunk_results):
            t2_flat[chunk] = t2_c
            s0_flat[chunk] = s0_c
            bias_flat[chunk] = bias_c
            r2_flat[chunk] = r2_c
            n_failed += nf
    else:
        t2_c, s0_c, bias_c, r2_c, n_failed = _fit_columns(flat[:, fit_indices], te_fit)
        t2_flat[fit_indices] = t2_c
        s0_flat[fit_indices] = s0_c
        bias_flat[fit_indices] = bias_c
        r2_flat[fit_indices] = r2_c

    # --- Plausibility clipping ---
    out_of_range_mask = ~np.isnan(t2_flat) & (
        (t2_flat < T2_MIN_MS) | (t2_flat > T2_MAX_MS)
    )
    n_out_of_range = int(np.sum(out_of_range_mask))
    t2_flat[out_of_range_mask] = np.nan

    n_fitted = int(np.sum(~np.isnan(t2_flat)))

    _log.debug(
        "T2 fit: %d fitted, %d failed, %d out-of-range [%.0f–%.0f ms], %d masked",
        n_fitted,
        n_failed,
        n_out_of_range,
        T2_MIN_MS,
        T2_MAX_MS,
        int(np.sum(noise_mask_flat)),
    )

    return T2FitResult(
        t2=t2_flat.reshape(spatial_shape),
        s0=s0_flat.reshape(spatial_shape),
        bias=bias_flat.reshape(spatial_shape),
        r2=r2_flat.reshape(spatial_shape),
        noise_mask=noise_mask_flat.reshape(spatial_shape),
        n_fitted=n_fitted,
        n_failed=n_failed,
        n_out_of_range=n_out_of_range,
        te_ms=list(te_ms),
    )


def fit_adc_monoexp(
    dwi: np.ndarray,
    b_values: list[float],
    noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
    bias_free: bool = False,
    n_jobs: int | None = None,
) -> ADCFitResult:
    """
    Fit a mono-exponential ADC decay pixel-by-pixel: ``S(b) = C * exp(-b*D)``.

    Parameters
    ----------
    dwi:
        Signal array with b-values on axis 0 and spatial axes after.
        Shape: ``(n_b, *spatial)``. Pass one direction's own sub-stack to
        fit per-direction ADC, or all directions pooled (with their
        trace-corrected effective b-values) to reproduce PV's combined
        "trace" ADC — see module docstring.
        **Must be at native (non-upsampled) resolution.**
    b_values:
        b-values in s/mm^2, one per image. Length must equal
        ``dwi.shape[0]``. Use ``PVM_DwEffBval`` (trace-corrected), not the
        nominal 0/200/400/800/1200 values, when reproducing PV's fit.
    noise_threshold:
        Pixels whose lowest-b signal is below
        ``noise_threshold * slice_max`` are excluded (NaN in output).
        Default 0.05 (5% of per-slice maximum).
    bias_free:
        If False (default), the noise-floor term is fixed at 0, matching
        PV's ``dtraceb`` macro. If True, it's a free third parameter — a
        diagnostic to check whether a fitted ADC value is being distorted
        by the Rician noise floor (see module docstring).
    n_jobs:
        Same semantics as :func:`fit_t2_monoexp` — ``None`` auto-parallelizes
        across CPU cores above ``_PARALLEL_THRESHOLD`` pixels; ``1`` forces
        serial. Same spawn/``__main__``-guard caveat applies.

    Returns
    -------
    ADCFitResult
        Maps and quality statistics. Spatial shape matches ``dwi[0]``.
    """
    if len(b_values) != dwi.shape[0]:
        raise ValueError(
            f"len(b_values)={len(b_values)} does not match dwi.shape[0]={dwi.shape[0]}"
        )
    if len(b_values) < 3:
        raise ValueError("At least 3 b-values are required for a reliable ADC fit.")

    b = np.array(b_values, dtype=float)
    first_idx = int(np.argmin(b))

    spatial_shape = dwi.shape[1:]
    n_pixels = int(np.prod(spatial_shape))

    flat = dwi.reshape(dwi.shape[0], n_pixels).astype(float)

    noise_mask_flat = _build_noise_mask(flat, spatial_shape, noise_threshold, first_idx)

    adc_flat = np.full(n_pixels, np.nan)
    s0_flat = np.full(n_pixels, np.nan)
    bias_flat = np.full(n_pixels, np.nan)
    r2_flat = np.full(n_pixels, np.nan)

    fit_indices = np.where(~noise_mask_flat)[0]
    n_to_fit = len(fit_indices)

    workers = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, n_to_fit))

    if n_to_fit == 0:
        n_failed = 0
    elif workers > 1 and n_to_fit >= _PARALLEL_THRESHOLD:
        chunks = np.array_split(fit_indices, workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunk_results = list(
                executor.map(
                    _fit_columns_adc,
                    [flat[:, c] for c in chunks],
                    [b] * len(chunks),
                    [bias_free] * len(chunks),
                )
            )
        n_failed = 0
        for chunk, (adc_c, s0_c, bias_c, r2_c, nf) in zip(chunks, chunk_results):
            adc_flat[chunk] = adc_c
            s0_flat[chunk] = s0_c
            bias_flat[chunk] = bias_c
            r2_flat[chunk] = r2_c
            n_failed += nf
    else:
        adc_c, s0_c, bias_c, r2_c, n_failed = _fit_columns_adc(
            flat[:, fit_indices], b, bias_free
        )
        adc_flat[fit_indices] = adc_c
        s0_flat[fit_indices] = s0_c
        bias_flat[fit_indices] = bias_c
        r2_flat[fit_indices] = r2_c

    out_of_range_mask = ~np.isnan(adc_flat) & (
        (adc_flat < ADC_MIN_MM2S) | (adc_flat > ADC_MAX_MM2S)
    )
    n_out_of_range = int(np.sum(out_of_range_mask))
    adc_flat[out_of_range_mask] = np.nan

    n_fitted = int(np.sum(~np.isnan(adc_flat)))

    _log.debug(
        "ADC fit: %d fitted, %d failed, %d out-of-range [%.5f-%.5f mm^2/s], %d masked, bias_free=%s",
        n_fitted,
        n_failed,
        n_out_of_range,
        ADC_MIN_MM2S,
        ADC_MAX_MM2S,
        int(np.sum(noise_mask_flat)),
        bias_free,
    )

    return ADCFitResult(
        adc=adc_flat.reshape(spatial_shape),
        s0=s0_flat.reshape(spatial_shape),
        bias=bias_flat.reshape(spatial_shape),
        r2=r2_flat.reshape(spatial_shape),
        noise_mask=noise_mask_flat.reshape(spatial_shape),
        n_fitted=n_fitted,
        n_failed=n_failed,
        n_out_of_range=n_out_of_range,
        b_values=list(b_values),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_noise_mask(
    flat: np.ndarray,
    spatial_shape: tuple[int, ...],
    threshold: float,
    first_idx: int,
) -> np.ndarray:
    """
    Return a boolean flat mask — True where signal is below the noise floor.

    For multi-slice data the threshold is computed per-slice so that slices
    with intrinsically lower SNR (e.g. edge slices) are not over-masked.
    For single-slice (2-D) data the global maximum is used.

    ``flat`` has shape ``(n_echoes, n_pixels)``.
    ``spatial_shape`` is the un-flattened spatial shape (e.g. ``(8, 48, 48)``).
    ``first_idx`` is the index of the first (shortest-TE) echo, used as the
    reference signal for thresholding.
    """
    first_echo = flat[first_idx]  # (n_pixels,)

    # Detect multi-slice: first spatial dim assumed to be Z if ndim >= 3
    if len(spatial_shape) >= 3:
        n_slices = spatial_shape[0]
        pixels_per_slice = int(np.prod(spatial_shape[1:]))
        mask = np.zeros(first_echo.shape[0], dtype=bool)
        for s in range(n_slices):
            start = s * pixels_per_slice
            end = start + pixels_per_slice
            sl = first_echo[start:end]
            slice_max = float(np.max(sl)) if sl.size > 0 else 0.0
            # A slice_max of zero means the entire slice is empty — mask it all.
            if slice_max <= 0.0:
                mask[start:end] = True
            else:
                mask[start:end] = sl < threshold * slice_max
        return mask

    # 2-D or single-slice
    global_max = float(np.max(first_echo)) if first_echo.size > 0 else 0.0
    if global_max <= 0.0:
        return np.ones(first_echo.shape[0], dtype=bool)
    return first_echo < threshold * global_max


def _initial_offset_guess(
    signal: np.ndarray, te_fit: np.ndarray, first_idx: int
) -> tuple[float, float, float]:
    """
    Estimate A, C, and T2 for the initial optimizer parameters.

    C and T2 come from a log-linear regression on the echoes *excluding* the
    anchored first echo (its true decay behaviour is contaminated by
    stimulated echoes, so it shouldn't drive the slope estimate).  A starts
    at the dimmest observed echo, a reasonable noise-floor proxy.

    Falls back to (0.0, signal[first_idx], 50.0) if the regression fails.
    """
    rest_signal = np.delete(signal, first_idx)
    rest_te = np.delete(te_fit, first_idx)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_sig = np.log(rest_signal)

    valid = np.isfinite(log_sig) & (rest_signal > 0)
    a_guess = max(float(np.min(signal)), 0.0)

    if np.sum(valid) < 2:
        c_guess = float(signal[first_idx]) if signal[first_idx] > 0 else 1.0
        return a_guess, c_guess, 50.0

    # Linear fit: log(S) = log(C) - TE/T2
    coeffs = np.polyfit(rest_te[valid], log_sig[valid], 1)
    slope, intercept = coeffs
    c_guess = float(np.exp(intercept))
    # slope = -1/T2; guard against zero or positive slope
    t2_guess = float(-1.0 / slope) if slope < 0 else 50.0
    # Clamp to a reasonable search range so the optimizer starts sensibly
    t2_guess = float(np.clip(t2_guess, 1.0, 500.0))
    return a_guess, c_guess, t2_guess


def _residuals(p: np.ndarray, signal: np.ndarray, te_fit: np.ndarray) -> np.ndarray:
    a, c, t2 = p
    return signal - (a + c * np.exp(-te_fit / t2))


def _jacobian(p: np.ndarray, signal: np.ndarray, te_fit: np.ndarray) -> np.ndarray:
    """
    Analytic Jacobian of :func:`_residuals` w.r.t. (a, c, t2).

    residual = signal - (a + c * exp(-te/t2)), so:
        d(residual)/da  = -1
        d(residual)/dc  = -exp(-te/t2)
        d(residual)/dt2 = -c * exp(-te/t2) * te / t2**2

    Supplying this avoids finite-difference Jacobian estimation (3 extra
    residual evaluations per step for a 3-parameter model), which is the
    dominant cost when fitting hundreds of thousands of pixels.
    """
    _, c, t2 = p
    exp_term = np.exp(-te_fit / t2)
    d_da = -np.ones_like(te_fit)
    d_dc = -exp_term
    d_dt2 = -c * exp_term * te_fit / t2**2
    return np.stack([d_da, d_dc, d_dt2], axis=1)


def _fit_columns(
    flat_chunk: np.ndarray, te_fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit every column (pixel) in ``flat_chunk`` — shape ``(n_echoes, n_pixels)``.

    Module-level and picklable so it can run as a :class:`ProcessPoolExecutor`
    task; also used directly for the serial (single-process) path so the
    fitting logic isn't duplicated between the two.

    Returns ``(t2, s0, bias, r2, n_failed)`` — the first four are one entry
    per column (NaN where the fit failed to converge); ``n_failed`` is the
    count of non-converged columns in this chunk.
    """
    n = flat_chunk.shape[1]
    t2 = np.full(n, np.nan)
    s0 = np.full(n, np.nan)
    bias = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    n_failed = 0
    for i in range(n):
        signal = flat_chunk[:, i]
        t2_val, s0_val, bias_val, r2_val, converged = _fit_pixel(signal, te_fit)
        if not converged:
            n_failed += 1
            continue
        t2[i] = t2_val
        s0[i] = s0_val
        bias[i] = bias_val
        r2[i] = r2_val
    return t2, s0, bias, r2, n_failed


def _fit_pixel(
    signal: np.ndarray,
    te_fit: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """
    Fit one pixel's echo train to S = A + C * exp(-TE_fit / T2).

    ``te_fit`` already has the first echo's time anchored near zero (see
    :func:`fit_t2_monoexp`).

    Returns (t2_ms, s0, bias, r2, converged).
    """
    first_idx = int(np.argmin(te_fit))
    a_guess, c_guess, t2_guess = _initial_offset_guess(signal, te_fit, first_idx)

    # Bounds: A and C must be non-negative (magnitude MR signal); T2 allowed a
    # wider range than the plausibility window so that out-of-range values can
    # be detected and reported rather than silently clamped by the optimiser.
    try:
        result = least_squares(
            _residuals,
            x0=[a_guess, c_guess, t2_guess],
            jac=_jacobian,
            bounds=([0.0, 0.0, 0.5], [np.inf, np.inf, 2000.0]),
            args=(signal, te_fit),
            method="trf",
        )
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, 0.0, False

    if not result.success:
        return 0.0, 0.0, 0.0, 0.0, False

    a_val, c_val, t2_val = result.x
    s0_val = a_val + c_val

    # R² = 1 - SS_res / SS_tot
    fitted = a_val + c_val * np.exp(-te_fit / t2_val)
    ss_res = float(np.sum((signal - fitted) ** 2))
    ss_tot = float(np.sum((signal - np.mean(signal)) ** 2))
    r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return t2_val, s0_val, a_val, r2_val, True


# ---------------------------------------------------------------------------
# ADC internal helpers
# ---------------------------------------------------------------------------


def _initial_guess_adc(
    signal: np.ndarray, b: np.ndarray, bias_free: bool
) -> tuple[float, float, float]:
    """
    Estimate A, C, and D for the initial optimizer parameters, via a
    log-linear regression on all points (no anchoring needed for ADC — see
    module docstring). Falls back to a flat guess if the regression fails.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sig = np.log(np.clip(signal, 1.0, None))

    valid = np.isfinite(log_sig)
    a_guess = max(float(np.min(signal)), 0.0) if bias_free else 0.0

    if np.sum(valid) < 2:
        c_guess = float(signal[int(np.argmin(b))]) if signal.size else 1.0
        return a_guess, c_guess, 1e-3

    # log(S) = log(C) - b*D
    coeffs = np.polyfit(b[valid], log_sig[valid], 1)
    slope, intercept = coeffs
    c_guess = float(np.exp(intercept))
    # slope = -D; guard against zero or positive slope
    d_guess = float(-slope) if slope < 0 else 1e-3
    d_guess = float(np.clip(d_guess, 1e-6, ADC_MAX_MM2S * 2))
    return a_guess, c_guess, d_guess


def _residuals_adc_fixed(p: np.ndarray, signal: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, d = p
    return signal - c * np.exp(-b * d)


def _jacobian_adc_fixed(p: np.ndarray, signal: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, d = p
    exp_term = np.exp(-b * d)
    d_dc = -exp_term
    d_dd = c * b * exp_term
    return np.stack([d_dc, d_dd], axis=1)


def _residuals_adc_free(p: np.ndarray, signal: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, c, d = p
    return signal - (a + c * np.exp(-b * d))


def _jacobian_adc_free(p: np.ndarray, signal: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, c, d = p
    exp_term = np.exp(-b * d)
    d_da = -np.ones_like(b)
    d_dc = -exp_term
    d_dd = c * b * exp_term
    return np.stack([d_da, d_dc, d_dd], axis=1)


def _fit_columns_adc(
    flat_chunk: np.ndarray, b: np.ndarray, bias_free: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit every column (pixel) in ``flat_chunk`` — shape ``(n_b, n_pixels)``.

    Module-level and picklable so it can run as a :class:`ProcessPoolExecutor`
    task; also used directly for the serial (single-process) path.

    Returns ``(adc, s0, bias, r2, n_failed)``.
    """
    n = flat_chunk.shape[1]
    adc = np.full(n, np.nan)
    s0 = np.full(n, np.nan)
    bias = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    n_failed = 0
    for i in range(n):
        signal = flat_chunk[:, i]
        adc_val, s0_val, bias_val, r2_val, converged = _fit_pixel_adc(signal, b, bias_free)
        if not converged:
            n_failed += 1
            continue
        adc[i] = adc_val
        s0[i] = s0_val
        bias[i] = bias_val
        r2[i] = r2_val
    return adc, s0, bias, r2, n_failed


def _fit_pixel_adc(
    signal: np.ndarray,
    b: np.ndarray,
    bias_free: bool,
) -> tuple[float, float, float, float, bool]:
    """
    Fit one pixel's b-value series to ``S = A + C * exp(-b*D)`` (A fixed at
    0 unless ``bias_free``).

    Returns (adc_mm2s, s0, bias, r2, converged).
    """
    a_guess, c_guess, d_guess = _initial_guess_adc(signal, b, bias_free)

    try:
        if bias_free:
            result = least_squares(
                _residuals_adc_free,
                x0=[a_guess, c_guess, d_guess],
                jac=_jacobian_adc_free,
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, ADC_MAX_MM2S * 4]),
                args=(signal, b),
                method="trf",
            )
        else:
            result = least_squares(
                _residuals_adc_fixed,
                x0=[c_guess, d_guess],
                jac=_jacobian_adc_fixed,
                bounds=([0.0, 0.0], [np.inf, ADC_MAX_MM2S * 4]),
                args=(signal, b),
                method="trf",
            )
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, 0.0, False

    if not result.success:
        return 0.0, 0.0, 0.0, 0.0, False

    if bias_free:
        a_val, c_val, d_val = result.x
    else:
        a_val = 0.0
        c_val, d_val = result.x

    s0_val = a_val + c_val

    fitted = a_val + c_val * np.exp(-b * d_val)
    ss_res = float(np.sum((signal - fitted) ** 2))
    ss_tot = float(np.sum((signal - np.mean(signal)) ** 2))
    r2_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return d_val, s0_val, a_val, r2_val, True
