"""
Unified T2 / ADC map access.

Two families of entry point, deliberately kept separate — neither ever
falls back to the other, so a caller always knows exactly where its map's
numbers came from:

- ``get_pv_t2_map`` / ``get_pv_adc_map`` — PV-only: extract ParaVision's own
  proc 2 map. Returns ``None`` if proc 2 doesn't exist or has no matching
  ISA frame; never fits anything itself. Used by ``check_t2_map.py`` /
  ``check_adc_map.py`` (whose whole job is comparing PV against a fresh
  fit) and by ``analyze_disc.py``'s smoke check.
- ``fit_t2_map`` / ``fit_adc_map`` — fit-only: always fit fresh from proc 1,
  never touch proc 2. This is what ``analyze_disc.py`` (the per-disc ROI
  analysis pipeline) uses to generate the maps it saves and computes ROI
  stats against — results should never depend on whether ParaVision
  happened to have already computed a map for a given experiment.

Both families return the map at **native (non-upsampled) resolution** so
values are physically correct, and record where a map came from in
``MapResult.source`` (``"pv"`` or ``"fitted"``) so callers can report it.

PV extraction (``get_pv_t2_map`` / ``get_pv_adc_map``)
-------------------------------------------------------
1. Check whether ``pdata/2/2dseq`` exists.
2. If yes: load proc 2, look for an FG_ISA frame labelled "T2 relaxation
   time" (T2) or "diffusion constant" (ADC, PV's ``dtraceb`` macro output).
3. If found: extract that frame → ``source = "pv"``.
4. Otherwise: return ``None``.

Fitting (``fit_t2_map`` / ``fit_adc_map``)
---------------------------------------------
Loads proc 1's raw stack (echo stack for T2, DWI stack for ADC) and fits
pixel-by-pixel — mono-exponential T2, or combined ("trace") ADC pooling all
3 diffusion directions' own trace-corrected effective b-values
(``PVM_DwEffBval``). See ``analysis/fitting.py`` module docstring and
AGENTS.md for the model details (why "trace" ADC, not DTI/IVIM/DKI, given
this dataset's acquisition).

Saving / loading
----------------
Maps are saved as ``parameter_maps/{study_name}_exp{exp_id}_t2.npy`` with a
companion ``.json`` carrying scale, source, and fitting metadata.
``save_map`` / ``load_map`` handle both files atomically. ``parameter_maps/``
is a dedicated top-level folder (sibling to ``patients/``) for all generated
maps, ROIs (``analysis/rois.py``), and the aggregated stats CSV, kept
separate from raw acquisition data.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)

MapSource = Literal["pv", "fitted"]

PARAMETER_MAPS_DIR = "parameter_maps"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MapResult:
    """
    A quantitative map for one disc (one experiment).

    Attributes
    ----------
    data:
        Map array at native resolution in napari axis order ``[Z, Y, X]`` or
        ``[Y, X]`` for single-slice data.  Units: ms for T2, mm²/s for ADC.
    scale:
        Voxel size in mm per axis, matching ``data``'s axis order.
    kind:
        ``"t2"`` or ``"adc"``.
    source:
        ``"pv"`` if extracted from a ParaVision proc 2 map,
        ``"fitted"`` if computed here from the raw echo/DWI stack.
    study_name:
        Human-readable study identifier (e.g. ``"RA_2022_07_20.f21"``).
    exp_id:
        Experiment folder name (e.g. ``"9"``).
    meta:
        Extra provenance: echo times used, PV ISA label, fit statistics, etc.
    diagnostics:
        Per-pixel fit-quality arrays, same shape as ``data`` — ``s0``
        (extrapolated signal at TE=0/b=0), ``bias`` (fitted noise-floor
        offset ``A``), ``r2`` (goodness of fit). ``None`` for PV-sourced
        maps (``source="pv"``), which were never fit here. Lets a saved map
        be re-inspected later (see ``inspect_roi_fit.py``) without redoing
        the fit — e.g. plotting an ROI's measured signal decay against its
        fitted curve.
    """

    data: np.ndarray
    scale: tuple[float, ...]
    kind: str
    source: MapSource
    study_name: str
    exp_id: str
    meta: dict
    diagnostics: dict[str, np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_t2_map(
    study_path: Path | str,
    exp_id: str,
    study_name: str = "",
) -> MapResult:
    """
    Always fit a fresh T2 map from proc 1's echo stack — never uses PV's
    proc 2 map.

    Used by ``analyze_disc.py``: per-disc analysis always generates its own
    map, so results don't depend on whether ParaVision happened to compute
    one for this experiment. Compare against PV's own map only as a
    separate smoke check (``get_pv_t2_map``), never as the analysis source.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name
    return _fit_t2_from_proc1(study_path, exp_id, study_name)


def fit_adc_map(
    study_path: Path | str,
    exp_id: str,
    study_name: str = "",
) -> MapResult:
    """
    Always fit a fresh combined ("trace") ADC map from proc 1's DWI stack —
    never uses PV's proc 2 map.

    See ``fit_t2_map`` docstring — same rationale.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name
    return _fit_adc_from_proc1(study_path, exp_id, study_name)


def get_pv_t2_map(
    study_path: Path | str, exp_id: str, study_name: str = ""
) -> MapResult | None:
    """
    Extract a T2 map from ParaVision's own proc 2 "T2 relaxation time" ISA
    frame. Never fits anything itself — returns ``None`` if proc 2 doesn't
    exist or has no matching frame, rather than falling back to
    ``fit_t2_map``.

    Used by ``check_t2_map.py`` and ``analyze_disc.py``'s PV smoke check.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name

    from analysis.fitting import T2_MAX_MS, T2_MIN_MS

    def matches(label: str) -> bool:
        low = label.lower()
        return "t2" in low or "relaxation" in low

    return _try_load_pv_map(
        study_path, exp_id, study_name, "t2", matches, T2_MIN_MS, T2_MAX_MS, "ms"
    )


def get_pv_adc_map(
    study_path: Path | str, exp_id: str, study_name: str = ""
) -> MapResult | None:
    """
    Extract an ADC map from ParaVision's own proc 2 "diffusion constant" ISA
    frame (excludes "std dev of diffusion constant", which also contains
    that substring). Never fits anything itself — returns ``None`` if proc 2
    doesn't exist or has no matching frame, rather than falling back to
    ``fit_adc_map``.

    Used by ``check_adc_map.py``/``check_diffusion_directions.py`` and
    ``analyze_disc.py``'s PV smoke check.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name

    from analysis.fitting import ADC_MAX_MM2S, ADC_MIN_MM2S

    def matches(label: str) -> bool:
        low = label.lower()
        return "diffusion constant" in low and "std dev" not in low

    return _try_load_pv_map(
        study_path,
        exp_id,
        study_name,
        "adc",
        matches,
        ADC_MIN_MM2S,
        ADC_MAX_MM2S,
        "mm^2/s",
    )


def fit_anatomy_image(
    study_path: Path | str,
    exp_id: str,
    study_name: str = "",
    kind: str = "t2",
    zoom_to: int = 0,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """
    Anatomy image on the same *physical grid* as ``fit_t2_map``/``fit_adc_map``:
    always proc 1's own raw stack (first echo for T2, lowest-b ~b0 DWI frame
    for ADC — highest, most reliable SNR everywhere, rather than the
    highest-b frame where high-diffusivity tissue can sit near the Rician
    noise floor) — never PV's proc 2 signal-intensity frame, which can cover
    a different or reordered subset of slices than proc 1 (see AGENTS.md
    "Two T2-map layouts").

    ``zoom_to`` (default 0, i.e. native resolution — pixel-identical to the
    map) optionally upsamples the returned image for a nicer image to trace
    ROI boundaries on, the same cubic-spline display upsampling
    ``bruker_browser.loader.load_2dseq`` uses elsewhere. The physical scale
    is adjusted proportionally (see ``zoom_spatial``), so the returned
    image still covers the exact same real-world footprint as the map even
    though its pixel count differs — draw ROIs on it with a napari Shapes
    layer whose own ``scale`` is set to the *map's* scale (not this image's)
    and the saved vertices land in native map pixel coordinates regardless
    of this zoom. See ``analyze_disc.py`` for the intended usage.

    Results are cached to ``{results_dir}/`` (same folder as the maps) so
    reopening ``analyze_disc.py`` for the same disc skips the full
    brukerapi load + scipy zoom on subsequent runs. The zoom level is baked
    into the cache filename, so a change in ``ANATOMY_ZOOM_TO`` produces a
    fresh file rather than serving a stale one.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name

    try:
        image, scale = load_anatomy(study_name, exp_id, kind, zoom_to, results_dir)
        _log.debug(
            "Loaded cached anatomy for %s exp %s (%s, z%d)",
            study_name,
            exp_id,
            kind,
            zoom_to,
        )
        return image, scale
    except FileNotFoundError:
        pass

    image, scale = _anatomy_from_proc1(study_path, exp_id, study_name, kind, zoom_to)
    save_anatomy(image, scale, study_name, exp_id, kind, zoom_to, results_dir)
    return image, scale


def _anatomy_from_proc1(
    study_path: Path, exp_id: str, study_name: str, kind: str, zoom_to: int = 0
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Anatomy image from proc 1's raw stack: first echo (T2) or lowest-b DWI frame (ADC)."""
    from bruker_browser.loader import load_2dseq, zoom_spatial

    try:
        # zoom_to=0: load at native resolution and pick the one frame we
        # need first — zooming happens below, on just that frame. Loading
        # with zoom_to=zoom_to here would run cubic-spline interpolation
        # (scipy.ndimage.zoom, order=3) over the *entire* echo/DWI stack
        # before any frame is selected, costing ~N× the necessary work for
        # an N-frame stack (see zoom_spatial's docstring / AGENTS.md
        # "Display upsampling").
        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="1",
            study_name=study_name,
            zoom_to=0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load proc 1 for {study_name} exp {exp_id}"
        ) from exc

    if kind == "t2":
        fg = next((fg for fg in result.frame_groups if "ECHO" in fg.name), None)
    else:
        fg = next(
            (
                fg
                for fg in result.frame_groups
                if "MOVIE" in fg.name or "DIFFUSION" in fg.name
            ),
            None,
        )
    if fg is None:
        raise RuntimeError(
            f"No anatomy frame group found in proc 1 for {study_name} exp {exp_id} ({kind})"
        )

    frames = np.moveaxis(result.data, fg.axis, 0)
    if kind == "t2":
        image = frames[0]  # first echo
    else:
        # Lowest-b (~b0) frame: highest, most reliable SNR everywhere,
        # including NP where high diffusivity can push the highest-b
        # signal down near the Rician noise floor (see
        # check_diffusion_directions.py's SNR-red-flag rationale) — a
        # noisy "anatomy" image would be worse to trace ROI boundaries on
        # than one with weaker NP/AF contrast but a real signal everywhere.
        b_values = _read_dw_eff_bval(study_path / exp_id / "method")
        image = frames[int(np.argmin(b_values))]

    image = image.astype(float)
    scale = _drop_axis(result.scale, fg.axis)

    if zoom_to and zoom_to > 1:
        # Only the trailing two axes are true in-plane spatial axes (napari
        # convention: spatial dims are always ordered last — see
        # _reorder_for_napari). Anything left in front after frame selection
        # (e.g. a real Z/slice axis) is through-plane and must not be
        # upsampled — zoom_to controls in-plane display resolution only.
        # Using range(image.ndim) here previously zoomed Z too, turning 8
        # real slices into 256 interpolated ones.
        n_spatial = min(2, image.ndim)
        spatial_indices = list(range(image.ndim - n_spatial, image.ndim))
        image, scale = zoom_spatial(image, scale, spatial_indices, zoom_to)

    return image, scale


def sanitize_stem(s: str) -> str:
    """
    Make a human-entered string (e.g. ``SUBJECT_study_name``, which can
    contain spaces or other free-text characters) safe to use as a
    filename component. Replaces anything that isn't a word character,
    dash, dot, or space with an underscore.
    """
    return re.sub(r"[^\w\-. ]", "_", s)


def map_path(
    study_name: str,
    exp_id: str,
    kind: str,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> Path:
    """Path to a saved map's ``.npy`` file, without touching disk."""
    stem = f"{sanitize_stem(study_name)}_exp{exp_id}_{kind}"
    return Path(results_dir) / f"{stem}.npy"


def anatomy_path(
    study_name: str,
    exp_id: str,
    kind: str,
    zoom_to: int,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> Path:
    """
    Path to a cached anatomy image ``.npy`` file, without touching disk.

    The zoom level is baked into the filename so a change in
    ``ANATOMY_ZOOM_TO`` produces a fresh cache entry rather than silently
    serving a stale one.
    """
    stem = f"{sanitize_stem(study_name)}_exp{exp_id}_{kind}_anatomy_z{zoom_to}"
    return Path(results_dir) / f"{stem}.npy"


def save_anatomy(
    image: np.ndarray,
    scale: tuple[float, ...],
    study_name: str,
    exp_id: str,
    kind: str,
    zoom_to: int,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> Path:
    """Cache an anatomy image to disk as ``{stem}.npy`` + ``{stem}_scale.json``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    npy = anatomy_path(study_name, exp_id, kind, zoom_to, results_dir)
    json_path = npy.with_suffix(".json")
    np.save(npy, image)
    json_path.write_text(json.dumps({"scale": list(scale)}))
    _log.debug("Cached anatomy to %s", npy)
    return npy


def load_anatomy(
    study_name: str,
    exp_id: str,
    kind: str,
    zoom_to: int,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """
    Load a cached anatomy image.

    Raises
    ------
    FileNotFoundError
        If no cached anatomy exists for this (study, exp, kind, zoom_to).
    """
    npy = anatomy_path(study_name, exp_id, kind, zoom_to, results_dir)
    if not npy.exists():
        raise FileNotFoundError(
            f"No cached anatomy for {study_name} exp {exp_id} ({kind}, z{zoom_to}): {npy}"
        )
    scale = tuple(json.loads(npy.with_suffix(".json").read_text())["scale"])
    return np.load(npy), scale


def save_map(result: MapResult, results_dir: Path | str = PARAMETER_MAPS_DIR) -> Path:
    """
    Save a MapResult to ``{results_dir}/``.

    Writes:
    - ``{stem}.npy``            — the map array
    - ``{stem}.json``           — scale, source, kind, and meta dict
    - ``{stem}_diagnostics.npz`` — per-pixel s0/bias/r2, only if
      ``result.diagnostics`` is set (fitted maps only; not written at all
      for PV-sourced maps, rather than writing an empty file)

    Returns the path to the ``.npy`` file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{sanitize_stem(result.study_name)}_exp{result.exp_id}_{result.kind}"
    npy_path = results_dir / f"{stem}.npy"
    json_path = results_dir / f"{stem}.json"

    np.save(npy_path, result.data)

    if result.diagnostics is not None:
        np.savez(results_dir / f"{stem}_diagnostics.npz", **result.diagnostics)

    meta_out = {
        "scale": list(result.scale),
        "kind": result.kind,
        "source": result.source,
        "study_name": result.study_name,
        "exp_id": result.exp_id,
        "meta": result.meta,
    }
    json_path.write_text(json.dumps(meta_out, indent=2))

    _log.debug("Saved %s map to %s", result.kind.upper(), npy_path)
    return npy_path


def load_map(
    study_name: str,
    exp_id: str,
    kind: str,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> MapResult:
    """
    Load a previously saved MapResult from ``{results_dir}/``.

    Raises
    ------
    FileNotFoundError
        If the ``.npy`` or ``.json`` file does not exist.
    """
    results_dir = Path(results_dir)
    stem = f"{sanitize_stem(study_name)}_exp{exp_id}_{kind}"
    npy_path = results_dir / f"{stem}.npy"
    json_path = results_dir / f"{stem}.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"No saved {kind} map for {study_name} exp {exp_id}: {npy_path}"
        )

    data = np.load(npy_path)
    meta_in = json.loads(json_path.read_text())

    diagnostics_path = results_dir / f"{stem}_diagnostics.npz"
    diagnostics = None
    if diagnostics_path.exists():
        with np.load(diagnostics_path) as npz:
            diagnostics = {k: npz[k] for k in npz.files}

    return MapResult(
        data=data,
        scale=tuple(meta_in["scale"]),
        kind=meta_in["kind"],
        source=meta_in["source"],
        study_name=meta_in["study_name"],
        exp_id=meta_in["exp_id"],
        meta=meta_in["meta"],
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Internal: PV proc-2 extraction
# ---------------------------------------------------------------------------


def _load_proc2(study_path: Path, exp_id: str, study_name: str):
    """
    Load an experiment's proc 2 (derived-map) dataset at native resolution,
    or return None if it doesn't exist or fails to load.
    """
    if not (study_path / exp_id / "pdata" / "2" / "2dseq").exists():
        return None
    try:
        from bruker_browser.loader import load_2dseq

        return load_2dseq(
            study_path, exp_id=exp_id, proc_id="2", study_name=study_name, zoom_to=0
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not load proc 2 for %s exp %s: %s", study_name, exp_id, exc)
        return None


def _extract_isa_frame(
    result, matches
) -> tuple[np.ndarray, tuple[float, ...], str] | None:
    """
    Return ``(data, scale, label)`` for the first FG_ISA frame whose label
    satisfies ``matches(label)`` — a single derived-map plane with the
    FG_ISA axis dropped — or None if no frame matches.

    PV packs several derived planes (the map, its std-dev, the base signal
    image, …) into one FG_ISA axis; every proc-2 extraction here is "pick
    the plane whose label matches, drop that axis".
    """
    for fg in result.frame_groups:
        if fg.name != "FG_ISA":
            continue
        for i, label in enumerate(fg.labels):
            if matches(label):
                data = np.take(result.data, i, axis=fg.axis)
                return data, _drop_axis(result.scale, fg.axis), label
    return None


def _warn_if_implausible(
    data: np.ndarray,
    lo: float,
    hi: float,
    unit: str,
    study_name: str,
    exp_id: str,
    kind: str,
) -> None:
    """Log a warning if fewer than 10% of a PV map's pixels fall in ``[lo, hi]`` (a units red flag)."""
    n_total = int(np.sum(~np.isnan(data)))
    if n_total == 0:
        return
    n_plausible = int(np.sum((data >= lo) & (data <= hi)))
    if n_plausible / n_total < 0.1:
        _log.warning(
            "PV %s map for %s exp %s: only %d/%d pixels in plausible range [%g-%g %s]. Check units.",
            kind.upper(),
            study_name,
            exp_id,
            n_plausible,
            n_total,
            lo,
            hi,
            unit,
        )


def _try_load_pv_map(
    study_path: Path,
    exp_id: str,
    study_name: str,
    kind: str,
    matches,
    lo: float,
    hi: float,
    unit: str,
) -> MapResult | None:
    """
    Extract a derived map (``kind``) from PV proc 2 — shared by T2 and ADC.

    Returns None if proc 2 is absent or has no FG_ISA frame matching
    ``matches``. PV uses 0 as a "not fitted" sentinel, replaced with NaN so
    downstream stats ignore it; ``[lo, hi]``/``unit`` drive a units-sanity
    warning.
    """
    result = _load_proc2(study_path, exp_id, study_name)
    if result is None:
        return None

    frame = _extract_isa_frame(result, matches)
    if frame is None:
        _log.debug(
            "proc 2 for %s exp %s has no %s ISA frame (labels: %s)",
            study_name,
            exp_id,
            kind.upper(),
            [fg.labels for fg in result.frame_groups],
        )
        return None

    data, scale, label = frame
    data = data.astype(float)
    data[data == 0.0] = np.nan
    _warn_if_implausible(data, lo, hi, unit, study_name, exp_id, kind)

    return MapResult(
        data=data,
        scale=scale,
        kind=kind,
        source="pv",
        study_name=study_name,
        exp_id=exp_id,
        meta={"pv_label": label},
    )


# ---------------------------------------------------------------------------
# Internal: fitting from proc 1
# ---------------------------------------------------------------------------


def _fit_adc_from_proc1(
    study_path: Path,
    exp_id: str,
    study_name: str,
) -> MapResult:
    """
    Load the raw proc 1 DWI stack and fit a combined ("trace") ADC pixel-by-pixel,
    pooling all diffusion directions with their trace-corrected effective b-values.

    Raises
    ------
    RuntimeError
        If proc 1 cannot be loaded, has no diffusion frame group, or
        ``PVM_DwEffBval`` cannot be parsed from ``method``.
    """
    from analysis.fitting import fit_adc_monoexp
    from bruker_browser.loader import load_2dseq

    try:
        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="1",
            study_name=study_name,
            zoom_to=0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load proc 1 for {study_name} exp {exp_id}"
        ) from exc

    dw_fg = next(
        (
            fg
            for fg in result.frame_groups
            if "MOVIE" in fg.name or "DIFFUSION" in fg.name
        ),
        None,
    )
    if dw_fg is None:
        raise RuntimeError(
            f"No diffusion frame group found in {study_name} exp {exp_id}. "
            f"Frame groups: {[fg.name for fg in result.frame_groups]}"
        )

    b_values = _read_dw_eff_bval(study_path / exp_id / "method")
    if len(b_values) != dw_fg.size:
        raise RuntimeError(
            f"PVM_DwEffBval has {len(b_values)} values but frame group has "
            f"{dw_fg.size} frames for {study_name} exp {exp_id}."
        )

    dwi = np.moveaxis(result.data, dw_fg.axis, 0)  # (n_b, *spatial)
    scale_no_dw = _drop_axis(result.scale, dw_fg.axis)

    fit = fit_adc_monoexp(dwi, b_values)

    _log.info(
        "Fitted ADC for %s exp %s: %d fitted, %d failed, %d out-of-range",
        study_name,
        exp_id,
        fit.n_fitted,
        fit.n_failed,
        fit.n_out_of_range,
    )

    return MapResult(
        data=fit.adc,
        scale=scale_no_dw,
        kind="adc",
        source="fitted",
        study_name=study_name,
        exp_id=exp_id,
        meta={
            "b_values": b_values,
            "n_fitted": fit.n_fitted,
            "n_failed": fit.n_failed,
            "n_out_of_range": fit.n_out_of_range,
        },
        diagnostics={"s0": fit.s0, "bias": fit.bias, "r2": fit.r2},
    )


def _fit_t2_from_proc1(
    study_path: Path,
    exp_id: str,
    study_name: str,
) -> MapResult:
    """
    Load the raw proc 1 MSME echo stack and fit T2 pixel-by-pixel.

    Raises
    ------
    RuntimeError
        If proc 1 cannot be loaded or has no FG_ECHO axis.
    """
    from analysis.fitting import fit_t2_monoexp
    from bruker_browser.loader import load_2dseq

    try:
        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="1",
            study_name=study_name,
            zoom_to=0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load proc 1 for {study_name} exp {exp_id}"
        ) from exc

    # Find FG_ECHO axis and extract echo times
    echo_fg = next((fg for fg in result.frame_groups if "ECHO" in fg.name), None)
    if echo_fg is None:
        raise RuntimeError(
            f"No FG_ECHO axis found in {study_name} exp {exp_id}. "
            f"Frame groups: {[fg.name for fg in result.frame_groups]}"
        )

    te_ms = _parse_te_labels(echo_fg.labels)

    # Move echo axis to position 0: (n_echoes, *spatial)
    echoes = np.moveaxis(result.data, echo_fg.axis, 0)

    # Remove the echo axis from the scale tuple
    scale_no_echo = _drop_axis(result.scale, echo_fg.axis)

    fit = fit_t2_monoexp(echoes, te_ms)

    _log.info(
        "Fitted T2 for %s exp %s: %d fitted, %d failed, %d out-of-range",
        study_name,
        exp_id,
        fit.n_fitted,
        fit.n_failed,
        fit.n_out_of_range,
    )

    return MapResult(
        data=fit.t2,
        scale=scale_no_echo,
        kind="t2",
        source="fitted",
        study_name=study_name,
        exp_id=exp_id,
        meta={
            "te_ms": te_ms,
            "n_fitted": fit.n_fitted,
            "n_failed": fit.n_failed,
            "n_out_of_range": fit.n_out_of_range,
        },
        diagnostics={"s0": fit.s0, "bias": fit.bias, "r2": fit.r2},
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_te_labels(labels: list[str]) -> list[float]:
    """
    Extract echo times in ms from FrameGroupInfo label strings.

    Labels are expected in the form ``"TE 10.4 ms"`` (produced by
    ``_build_fg_labels`` in loader.py).  Falls back to extracting the first
    float found in each string if the format differs.

    Raises
    ------
    ValueError
        If no numeric value can be extracted from any label.
    """
    import re

    times = []
    for label in labels:
        m = re.search(r"[-+]?\d*\.?\d+", label)
        if m:
            times.append(float(m.group()))
        else:
            raise ValueError(f"Cannot parse echo time from label: {label!r}")
    return times


def _drop_axis(scale: tuple[float, ...], axis: int) -> tuple[float, ...]:
    """Return a scale tuple with the element at ``axis`` removed."""
    lst = list(scale)
    lst.pop(axis)
    return tuple(lst)


def _read_dw_eff_bval(method_path: Path) -> list[float]:
    """
    Parse ``PVM_DwEffBval`` (one float per DWI frame, s/mm^2) from a raw
    ``method`` file.

    These are the trace-corrected effective b-values PV itself fits
    against (they differ slightly from the nominal 0/200/400/800/1200
    values because they account for B-matrix cross-terms) — using them,
    not the nominal values, is required to reproduce PV's ADC fit.

    Raises
    ------
    RuntimeError
        If ``PVM_DwEffBval`` cannot be found in the file.
    """
    import re

    text = method_path.read_text(encoding="latin-1", errors="replace")
    m = re.search(
        r"##\$PVM_DwEffBval=\([^\n]*\)\s*\n(.*?)(?=\n##|\n\$\$)", text, re.DOTALL
    )
    if not m:
        raise RuntimeError(f"PVM_DwEffBval not found in {method_path}")
    return [float(x) for x in m.group(1).split()]


def _read_visu_core_positions(visu_path: Path) -> list[tuple[float, float, float]]:
    """
    Parse ``VisuCorePosition`` (one xyz triple per slice, in mm) from a raw
    ``visu_pars`` file.  Returns ``[]`` if the file or parameter is missing.
    """
    import re

    try:
        text = visu_path.read_text(encoding="latin-1", errors="replace")
    except OSError:
        return []

    m = re.search(r"##\$VisuCorePosition=\([^\n]*\)\s*\n(.*?)(?=\n##)", text, re.DOTALL)
    if not m:
        return []

    nums = [float(x) for x in m.group(1).split()]
    return [tuple(nums[i : i + 3]) for i in range(0, len(nums), 3)]


def match_slices_to_proc1(
    study_path: Path | str,
    exp_id: str,
    proc_id: str = "2",
) -> list[int]:
    """
    Map each slice of ``proc_id`` (e.g. a T2/ADC parametric map) to its
    index in proc 1's raw echo/DWI stack, by nearest ``VisuCorePosition``.

    A PV parametric map is sometimes computed for only one (or a reordered
    subset) of proc 1's slices -- see AGENTS.md "Two T2-map layouts".
    Comparing or overlaying a map against proc 1 requires knowing which raw
    slice each map slice actually corresponds to; array index alone is not
    reliable across layouts.

    Returns one proc-1 slice index per proc-``proc_id`` slice, in
    proc-``proc_id`` order.  Raises ``RuntimeError`` if slice positions
    cannot be read from either proc's ``visu_pars``.
    """
    import math

    study_path = Path(study_path)
    pos1 = _read_visu_core_positions(study_path / exp_id / "pdata" / "1" / "visu_pars")
    pos2 = _read_visu_core_positions(
        study_path / exp_id / "pdata" / proc_id / "visu_pars"
    )
    if not pos1 or not pos2:
        raise RuntimeError(
            f"Could not read VisuCorePosition for {study_path} exp {exp_id} "
            f"(proc 1 or proc {proc_id})"
        )

    return [min(range(len(pos1)), key=lambda i: math.dist(p2, pos1[i])) for p2 in pos2]
