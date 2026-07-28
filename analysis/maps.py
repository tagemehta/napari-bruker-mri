"""
Unified T2 / ADC map access.

Decision logic for ADC
-----------------------
Mirrors T2: check ``pdata/2/2dseq`` for an FG_ISA frame labelled
"diffusion constant" (PV's ``dtraceb`` macro output) first; fall back to
fitting a fresh combined ("trace") ADC from proc 1's DWI stack using all 3
diffusion directions' own trace-corrected effective b-values
(``PVM_DwEffBval``) if proc 2 is absent or has no ADC frame. See
``analysis/fitting.py`` module docstring and AGENTS.md for the model
details (why "trace" ADC, not DTI/IVIM/DKI, given this dataset's
acquisition).

This module is the single entry point for obtaining a quantitative map for
one disc (one experiment within a study).  It decides automatically whether
to use a ParaVision-generated proc 2 map or to fit one fresh from the raw
echo stack, and always returns the map at **native (non-upsampled) resolution**
so values are physically correct.

Decision logic for T2
---------------------
1. Check whether ``pdata/2/2dseq`` exists.
2. If yes: load proc 2, look for an FG_ISA frame labelled "T2 relaxation time".
3. If found: extract that frame → ``source = "pv"``.
4. If proc 2 is absent, or has no T2 ISA frame:
   load proc 1 echo stack + echo times → fit mono-exponential → ``source = "fitted"``.

The source is recorded in ``MapResult.source`` so callers can report it and
notebook 01 can compare PV vs fitted side-by-side.

Saving / loading
----------------
Maps are saved as ``results/maps/{study_name}_exp{exp_id}_t2.npy`` with a
companion ``.json`` carrying scale, source, and fitting metadata.
``save_map`` / ``load_map`` handle both files atomically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)

MapSource = Literal["pv", "fitted"]


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
    """

    data: np.ndarray
    scale: tuple[float, ...]
    kind: str
    source: MapSource
    study_name: str
    exp_id: str
    meta: dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_t2_map(
    study_path: Path | str,
    exp_id: str,
    study_name: str = "",
) -> MapResult:
    """
    Return a T2 map for one disc experiment.

    Tries to use the PV-generated proc 2 "T2 relaxation time" ISA frame first.
    Falls back to fitting the raw proc 1 MSME echo stack if proc 2 is absent
    or does not contain a T2 frame.

    Parameters
    ----------
    study_path:
        Path to the Bruker study folder (contains the ``subject`` file).
    exp_id:
        Experiment number as a string (e.g. ``"9"``).
    study_name:
        Human-readable label used in saved filenames and ``MapResult``.
        Defaults to the study folder name.

    Returns
    -------
    MapResult
        T2 map at native resolution with provenance metadata.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name

    # Try PV proc 2 first
    pv_result = _try_load_pv_t2(study_path, exp_id, study_name)
    if pv_result is not None:
        _log.info(
            "T2 map for %s exp %s: using PV proc 2 (%s)",
            study_name,
            exp_id,
            pv_result.meta.get("pv_label", "?"),
        )
        return pv_result

    # Fall back to fitting
    _log.info(
        "T2 map for %s exp %s: no valid PV proc 2 — fitting from echo stack",
        study_name,
        exp_id,
    )
    return _fit_t2_from_proc1(study_path, exp_id, study_name)


def get_adc_map(
    study_path: Path | str,
    exp_id: str,
    study_name: str = "",
) -> MapResult:
    """
    Return a combined ("trace") ADC map for one disc experiment.

    Tries to use PV's own proc 2 "diffusion constant" ISA frame first
    (``dtraceb`` macro). Falls back to fitting the raw proc 1 DWI stack
    (all 3 diffusion directions pooled, trace-corrected b-values) if proc 2
    is absent or has no ADC frame.

    Parameters
    ----------
    study_path:
        Path to the Bruker study folder (contains the ``subject`` file).
    exp_id:
        Experiment number as a string (e.g. ``"11"``).
    study_name:
        Human-readable label used in saved filenames and ``MapResult``.
        Defaults to the study folder name.

    Returns
    -------
    MapResult
        ADC map (mm^2/s) at native resolution with provenance metadata.
    """
    study_path = Path(study_path)
    if not study_name:
        study_name = study_path.name

    pv_result = _try_load_pv_adc(study_path, exp_id, study_name)
    if pv_result is not None:
        _log.info(
            "ADC map for %s exp %s: using PV proc 2 (%s)",
            study_name,
            exp_id,
            pv_result.meta.get("pv_label", "?"),
        )
        return pv_result

    _log.info(
        "ADC map for %s exp %s: no valid PV proc 2 — fitting from DWI stack",
        study_name,
        exp_id,
    )
    return _fit_adc_from_proc1(study_path, exp_id, study_name)


def save_map(result: MapResult, results_dir: Path | str = "results") -> Path:
    """
    Save a MapResult to ``{results_dir}/maps/``.

    Writes two files:
    - ``{stem}.npy``  — the map array
    - ``{stem}.json`` — scale, source, kind, and meta dict

    Returns the path to the ``.npy`` file.
    """
    results_dir = Path(results_dir)
    maps_dir = results_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{result.study_name}_exp{result.exp_id}_{result.kind}"
    npy_path = maps_dir / f"{stem}.npy"
    json_path = maps_dir / f"{stem}.json"

    np.save(npy_path, result.data)

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
    results_dir: Path | str = "results",
) -> MapResult:
    """
    Load a previously saved MapResult from ``{results_dir}/maps/``.

    Raises
    ------
    FileNotFoundError
        If the ``.npy`` or ``.json`` file does not exist.
    """
    results_dir = Path(results_dir)
    stem = f"{study_name}_exp{exp_id}_{kind}"
    npy_path = results_dir / "maps" / f"{stem}.npy"
    json_path = results_dir / "maps" / f"{stem}.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"No saved {kind} map for {study_name} exp {exp_id}: {npy_path}"
        )

    data = np.load(npy_path)
    meta_in = json.loads(json_path.read_text())

    return MapResult(
        data=data,
        scale=tuple(meta_in["scale"]),
        kind=meta_in["kind"],
        source=meta_in["source"],
        study_name=meta_in["study_name"],
        exp_id=meta_in["exp_id"],
        meta=meta_in["meta"],
    )


# ---------------------------------------------------------------------------
# Internal: PV map extraction
# ---------------------------------------------------------------------------


def _try_load_pv_t2(
    study_path: Path,
    exp_id: str,
    study_name: str,
) -> MapResult | None:
    """
    Attempt to extract a T2 map from PV proc 2.

    Returns None if proc 2 does not exist or does not contain a frame
    labelled "T2 relaxation time" in its FG_ISA axis.
    """
    proc2_2dseq = study_path / exp_id / "pdata" / "2" / "2dseq"
    if not proc2_2dseq.exists():
        return None

    try:
        from bruker_browser.loader import load_2dseq

        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="2",
            study_name=study_name,
            zoom_to=0,  # always native resolution for analysis
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not load proc 2 for %s exp %s: %s", study_name, exp_id, exc)
        return None

    # Find an FG_ISA axis whose labels contain a T2 entry
    t2_fg = None
    t2_idx = None
    t2_label = None
    for fg in result.frame_groups:
        if fg.name != "FG_ISA":
            continue
        for i, label in enumerate(fg.labels):
            if "t2" in label.lower() or "relaxation" in label.lower():
                t2_fg = fg
                t2_idx = i
                t2_label = label
                break
        if t2_fg is not None:
            break

    if t2_fg is None:
        _log.debug(
            "proc 2 for %s exp %s has no T2 ISA frame (labels: %s)",
            study_name,
            exp_id,
            [fg.labels for fg in result.frame_groups],
        )
        return None

    # Extract the T2 frame by indexing the FG_ISA axis
    data = np.take(result.data, t2_idx, axis=t2_fg.axis)
    scale = _drop_axis(result.scale, t2_fg.axis)

    # PV uses 0 as a sentinel for "not fitted" pixels (background, failed fits).
    # Replace with NaN so downstream stats (np.nanmean etc.) ignore them correctly.
    data = data.astype(float)
    data[data == 0.0] = np.nan

    # PV T2 maps are stored in ms but sometimes scaled by a factor — values
    # outside the plausibility range are a signal to check units.
    from analysis.fitting import T2_MAX_MS, T2_MIN_MS

    n_total = int(np.sum(~np.isnan(data)))
    n_plausible = int(np.sum((data >= T2_MIN_MS) & (data <= T2_MAX_MS)))
    if n_total > 0 and n_plausible / n_total < 0.1:
        _log.warning(
            "PV T2 map for %s exp %s: only %d/%d pixels in plausible range "
            "[%.0f–%.0f ms]. Check units.",
            study_name,
            exp_id,
            n_plausible,
            n_total,
            T2_MIN_MS,
            T2_MAX_MS,
        )

    return MapResult(
        data=data,
        scale=scale,
        kind="t2",
        source="pv",
        study_name=study_name,
        exp_id=exp_id,
        meta={"pv_label": t2_label, "pv_isa_axis": t2_fg.axis, "pv_isa_index": t2_idx},
    )


def _try_load_pv_adc(
    study_path: Path,
    exp_id: str,
    study_name: str,
) -> MapResult | None:
    """
    Attempt to extract an ADC map from PV proc 2.

    Returns None if proc 2 does not exist or does not contain a frame
    labelled "diffusion constant" in its FG_ISA axis. Excludes the "std dev
    of diffusion constant" frame, which also contains that substring.
    """
    proc2_2dseq = study_path / exp_id / "pdata" / "2" / "2dseq"
    if not proc2_2dseq.exists():
        return None

    try:
        from bruker_browser.loader import load_2dseq

        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="2",
            study_name=study_name,
            zoom_to=0,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not load proc 2 for %s exp %s: %s", study_name, exp_id, exc)
        return None

    adc_fg = None
    adc_idx = None
    adc_label = None
    for fg in result.frame_groups:
        if fg.name != "FG_ISA":
            continue
        for i, label in enumerate(fg.labels):
            low = label.lower()
            if "diffusion constant" in low and "std dev" not in low:
                adc_fg = fg
                adc_idx = i
                adc_label = label
                break
        if adc_fg is not None:
            break

    if adc_fg is None:
        _log.debug(
            "proc 2 for %s exp %s has no ADC ISA frame (labels: %s)",
            study_name,
            exp_id,
            [fg.labels for fg in result.frame_groups],
        )
        return None

    data = np.take(result.data, adc_idx, axis=adc_fg.axis)
    scale = _drop_axis(result.scale, adc_fg.axis)

    # PV uses 0 as a sentinel for "not fitted" pixels.
    data = data.astype(float)
    data[data == 0.0] = np.nan

    from analysis.fitting import ADC_MAX_MM2S, ADC_MIN_MM2S

    n_total = int(np.sum(~np.isnan(data)))
    n_plausible = int(np.sum((data >= ADC_MIN_MM2S) & (data <= ADC_MAX_MM2S)))
    if n_total > 0 and n_plausible / n_total < 0.1:
        _log.warning(
            "PV ADC map for %s exp %s: only %d/%d pixels in plausible range "
            "[%.5f-%.5f mm^2/s]. Check units.",
            study_name,
            exp_id,
            n_plausible,
            n_total,
            ADC_MIN_MM2S,
            ADC_MAX_MM2S,
        )

    return MapResult(
        data=data,
        scale=scale,
        kind="adc",
        source="pv",
        study_name=study_name,
        exp_id=exp_id,
        meta={"pv_label": adc_label, "pv_isa_axis": adc_fg.axis, "pv_isa_index": adc_idx},
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
    from bruker_browser.loader import load_2dseq
    from analysis.fitting import fit_adc_monoexp

    try:
        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="1",
            study_name=study_name,
            zoom_to=0,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load proc 1 for {study_name} exp {exp_id}"
        ) from exc

    dw_fg = next(
        (fg for fg in result.frame_groups if "MOVIE" in fg.name or "DIFFUSION" in fg.name),
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
    from bruker_browser.loader import load_2dseq
    from analysis.fitting import fit_t2_monoexp

    try:
        result = load_2dseq(
            study_path,
            exp_id=exp_id,
            proc_id="1",
            study_name=study_name,
            zoom_to=0,
        )
    except Exception as exc:  # noqa: BLE001
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
    m = re.search(r"##\$PVM_DwEffBval=\([^\n]*\)\s*\n(.*?)(?=\n##|\n\$\$)", text, re.DOTALL)
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

    return [
        min(range(len(pos1)), key=lambda i: math.dist(p2, pos1[i])) for p2 in pos2
    ]
