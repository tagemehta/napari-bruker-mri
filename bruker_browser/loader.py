"""
Bruker data loading helpers.

All heavy I/O (brukerapi) lives here so the UI layer stays thin.

Axis convention
---------------
brukerapi returns data with spatial axes **first**:
    (X, Y)              for 2-D single-slice
    (X, Y, N_slices)    for 2-D multi-slice
    (X, Y, Z)           for 3-D
    (X, Y, N_slices, N_fg1, N_fg2, ...)  for multi-echo, diffusion, etc.

napari expects spatial axes **last**:
    (Y, X)
    (N_slices, Y, X)
    (Z, Y, X)
    (N_fg2, N_fg1, N_slices, Y, X)

`_reorder_for_napari` performs this reversal and builds a matching scale
tuple so voxel sizes stay aligned with their axes.

Frame-group axes
----------------
Non-spatial dimensions (echoes, diffusion directions, parametric maps, …) are
described by `FrameGroupInfo` objects returned in `LoadResult.frame_groups`.
The UI uses these to show per-axis selectors so the user can quickly flip
between e.g. T2 / R2 / amplitude without reloading from disk.

FG_SLICE is treated as a spatial (Z) axis and is **not** included in
`frame_groups` — slices always remain stacked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

_log = logging.getLogger(__name__)

# FG types that should remain stacked as the Z axis, not selectable.
# Also includes "frame" which brukerapi uses for a trivial single-frame axis.
_SLICE_FG_TYPES = {"FG_SLICE", "FG_SLICE_ORIENT", "SLICE", "FRAME"}


def _normalise_dim_type(raw_dim_type: list) -> list[str]:
    """
    Normalise a brukerapi dim_type list to plain uppercase strings.

    brukerapi wraps FG names in angle brackets: '<FG_SLICE>' → 'FG_SLICE'.
    The 'spatial' and 'frame' labels are already plain strings.
    """
    result = []
    for t in raw_dim_type:
        s = str(t).strip("<>").strip().upper()
        # brukerapi uses 'spatial' (lowercase) for spatial axes
        result.append(s)
    return result


@dataclass
class FrameGroupInfo:
    """
    Describes one selectable (non-spatial, non-slice) frame-group axis.

    Attributes
    ----------
    name:
        brukerapi dim_type label, e.g. ``"FG_ECHO"``, ``"FG_ISA"``,
        ``"FG_DIFFUSION"``.
    size:
        Number of elements along this axis in the napari-ordered array.
    labels:
        Human-readable label per element (``len == size``).  Examples:
        ``["TE 10.4 ms", "TE 20.8 ms", …]`` for echoes,
        ``["T2 relaxation time", "signal intensity", …]`` for ISA maps.
    axis:
        Index of this FG in the napari array (0-based, outermost first).
    """

    name: str
    size: int
    labels: list[str]
    axis: int


class LoadResult(NamedTuple):
    """
    Return value of :func:`load_2dseq`.

    Attributes
    ----------
    data:
        Array in napari axis order ``[FG..., Z, Y, X]``.  Spatial axes are
        upsampled if ``zoom_to`` was set.
    layer_name:
        Suggested napari layer name, e.g. ``"Rat Spine 690728 - 9:2"``.
    scale:
        Voxel size in mm per axis, matching ``data``'s axis order.
        FG axes use scale 1.0; FG_SLICE axes carry the slice thickness.
    metadata:
        Dict of provenance fields (study_path, exp_id, proc_id, dim_type, …)
        plus ``_full_data``, ``_full_scale``, and ``_frame_groups`` written
        by :func:`~bruker_browser.tile_viewer.load_into_viewer` for the FG
        selector widget.
    frame_groups:
        One :class:`FrameGroupInfo` per selectable FG axis.  Empty for
        purely spatial datasets (FLASH, RARE single-echo, etc.).
    """

    data: np.ndarray
    layer_name: str
    scale: tuple[float, ...]
    metadata: dict
    frame_groups: list[FrameGroupInfo]


def load_2dseq(
    study_path: Path | str,
    exp_id: str,
    proc_id: str = "1",
    study_name: str = "",
    zoom_to: int = 0,
) -> LoadResult:
    """
    Load a 2dseq dataset using brukerapi and return a LoadResult.

    Parameters
    ----------
    study_path:
        Path to the Bruker Study folder (the one containing the 'subject' file).
    exp_id:
        Experiment/scan number as a string (e.g. "1", "3").
    proc_id:
        Processing folder number (default "1").
    study_name:
        Human-readable study label included in the napari layer name.
    zoom_to:
        Minimum size (pixels) for each spatial axis.  Axes already at or above
        this size are left unchanged.  Set to 0 or 1 to disable interpolation.
        Default 256 matches ParaVision's display interpolation for small matrices.

    Returns
    -------
    LoadResult
        numpy array in napari axis order (spatial last), matching scale tuple,
        suggested layer name, raw metadata dict, and frame-group descriptors.
    """
    from brukerapi.folders import Study  # import here to keep startup fast

    study_path = Path(study_path)
    study = Study(str(study_path))

    dataset = study.get_dataset(exp_id=exp_id, proc_id=proc_id)
    dataset.load()

    raw: np.ndarray = dataset.data

    # --- dim_type tells us which axes are spatial vs frame-group ---
    dim_type: list[str] = []
    # brukerapi raises undocumented exception types for missing/unsupported
    # parameters depending on PV version — broad catches are intentional here.
    try:
        dim_type = _normalise_dim_type(dataset.dim_type)
    except Exception:  # noqa: BLE001
        _log.debug("dim_type not available for exp %s; assuming all-spatial", exp_id)

    # --- reorder axes so spatial dims are last (napari convention) ---
    # dim_type is also returned cleaned (trivial size-1 FG axes removed)
    data, spatial_scale_indices, dim_type = _reorder_for_napari(raw, dim_type)

    # --- layer name: "{study_name} - {exp_id}:{proc_id}" e.g. "Rat Spine 690728 - 9:2" ---
    proc_suffix = f":{proc_id}" if proc_id != "1" else ""
    scan_part = f"{exp_id}{proc_suffix}"
    layer_name = f"{study_name} - {scan_part}" if study_name else scan_part

    # --- voxel scale (in napari axis order) ---
    scale = _extract_scale(
        dataset,
        n_dims=data.ndim,
        spatial_scale_indices=spatial_scale_indices,
        dim_type=dim_type,
    )

    # --- frame-group descriptors (for the UI selector) ---
    frame_groups = _extract_frame_groups(dataset, dim_type, data.shape)

    # --- metadata ---
    meta: dict = {
        "study_path": str(study_path),
        "exp_id": exp_id,
        "proc_id": proc_id,
        "study_name": study_name,
        "raw_shape": raw.shape,
        "dim_type": dim_type,
        "napari_shape": data.shape,
    }
    try:
        meta["resolution_mm"] = dataset.resolution.tolist()
    except Exception:  # noqa: BLE001
        pass

    # --- optional spatial upsampling to match PV display interpolation ---
    if zoom_to and zoom_to > 1:
        data, scale = zoom_spatial(data, scale, spatial_scale_indices, zoom_to)

    _log.debug(
        "Loaded exp %s: raw shape %s dim_type %s -> napari shape %s scale %s fg=%s",
        exp_id,
        raw.shape,
        dim_type,
        data.shape,
        scale,
        [(fg.name, fg.size) for fg in frame_groups],
    )

    return LoadResult(
        data=data,
        layer_name=layer_name,
        scale=scale,
        metadata=meta,
        frame_groups=frame_groups,
    )


# ---------------------------------------------------------------------------
# Frame-group extraction
# ---------------------------------------------------------------------------


def _extract_frame_groups(
    dataset, dim_type: list[str], napari_shape: tuple[int, ...]
) -> list[FrameGroupInfo]:
    """
    Build FrameGroupInfo objects for each selectable (non-spatial, non-slice) FG axis.

    `dim_type` must be the **cleaned** list returned by `_reorder_for_napari`
    (trivial size-1 FG axes already removed, all entries normalised to uppercase).

    After reordering the napari array layout is:
        axis 0 .. k-1  : FG axes (reversed from cleaned dim_type order)
        axis k .. end  : spatial axes

    We only describe axes that are neither spatial nor slice-stack.
    """
    ndim = len(napari_shape)
    if not dim_type or len(dim_type) != ndim:
        return []

    # Partition cleaned dim_type indices
    fg_idx = [i for i, t in enumerate(dim_type) if t != "SPATIAL"]
    # After reorder: FG block occupies napari axes 0..len(fg_idx)-1 in reversed order

    # Per-element comments from visu_pars
    elem_comments = _read_fg_elem_comments(dataset)
    echo_times = _read_echo_times(dataset)

    groups: list[FrameGroupInfo] = []
    for napari_axis, orig_axis in enumerate(fg_idx[::-1]):
        fg_type = dim_type[orig_axis]
        # Skip slice axes — those stay stacked as Z
        if fg_type in _SLICE_FG_TYPES:
            continue

        # Read size from the napari array shape (post-squeeze, post-transpose)
        size = napari_shape[napari_axis]

        # Build per-element labels
        labels = _build_fg_labels(
            fg_type=fg_type,
            size=size,
            orig_axis=orig_axis,
            elem_comments=elem_comments,
            echo_times=echo_times,
            n_fg_total=len(fg_idx),
        )

        groups.append(
            FrameGroupInfo(
                name=fg_type,
                size=size,
                labels=labels,
                axis=napari_axis,
            )
        )

    return groups


def _read_fg_elem_comments(dataset) -> list[str]:
    """
    Return the flat ``VisuFGElemComment`` list from ``visu_pars``, or ``[]``.

    ``VisuFGElemComment`` is a JCAMP-DX array with one entry per unique
    combination of frame-group elements, covering all FG axes combined.

    ``dataset.path`` is the ``2dseq`` file path; ``visu_pars`` lives in the
    same directory.  We read the file directly because brukerapi does not
    expose a reliable ``get_value`` API across all supported PV versions.
    """
    try:
        for candidate in (
            Path(dataset.path).parent / "visu_pars",
            Path(dataset.path) / "visu_pars",
        ):
            if candidate.exists():
                return _parse_fg_elem_comment_from_file(candidate)
    except Exception:  # noqa: BLE001
        _log.debug("Could not locate visu_pars for VisuFGElemComment fallback")

    return []


def _parse_fg_elem_comment_from_file(visu_path: Path) -> list[str]:
    """
    Parse VisuFGElemComment array from a raw visu_pars file.

    The JCAMP-DX format stores each string element as <value> on its own line,
    but values can contain embedded newlines when they exceed PV's 65-char line
    limit.  We extract the full block between the parameter header and the next
    ## marker, then use regex to pull all <...> values including multi-line ones.
    """
    import re

    try:
        text = visu_path.read_text(encoding="latin-1", errors="replace")
    except OSError:
        return []

    # Isolate the block for this parameter
    m = re.search(r"##\$VisuFGElemComment=[^\n]*\n(.*?)(?=\n##)", text, re.DOTALL)
    if not m:
        return []

    block = m.group(1)
    # Extract all <...> values; allow newlines inside the angle brackets
    values = re.findall(r"<([^>]*)>", block)
    # Collapse embedded newlines and normalise whitespace
    return [" ".join(v.split()) for v in values if v.strip()]


def _read_echo_times(dataset) -> list[float]:
    """Return per-echo TE values in ms, or []."""
    try:
        arr = np.asarray(dataset.TE, dtype=float).ravel()
        if len(arr) >= 1:
            return arr.tolist()
    except Exception:  # noqa: BLE001
        pass
    return []


def _build_fg_labels(
    fg_type: str,
    size: int,
    orig_axis: int,
    elem_comments: list[str],
    echo_times: list[float],
    n_fg_total: int,
) -> list[str]:
    """
    Build a human-readable label list for one FG axis.

    Priority:
    1. VisuFGElemComment entries (if count matches size)
    2. Echo times for FG_ECHO axes
    3. Generic "{FG_TYPE} {i}" fallback
    """
    # 1. Per-element comments — only useful if the total count equals this axis size
    #    (they may cover all FG axes combined if there's only one non-slice FG)
    if elem_comments and len(elem_comments) == size:
        return [c[:40] for c in elem_comments]  # cap length for UI

    # 2. Echo times for echo axes
    if "ECHO" in fg_type.upper() and echo_times and len(echo_times) == size:
        return [f"TE {te:.1f} ms" for te in echo_times]

    # 3. Generic fallback
    short_name = fg_type.replace("FG_", "")
    return [f"{short_name} {i + 1}" for i in range(size)]


# ---------------------------------------------------------------------------
# Axis reordering
# ---------------------------------------------------------------------------


def _reorder_for_napari(
    data: np.ndarray,
    dim_type: list[str],
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Reverse axis order so spatial axes are last, matching napari's convention.

    brukerapi places spatial axes first: (X, Y[, Z/slices][, FG...])
    napari wants spatial axes last:      ([FG...,] Z/slices, Y, X)

    Returns
    -------
    reordered : np.ndarray
    spatial_scale_indices : list[int]
        Indices (in the *reordered* array) that correspond to spatial axes,
        used later to attach the correct voxel size only to spatial dims.
    """
    ndim = data.ndim

    if ndim <= 1:
        return data, list(range(ndim)), dim_type

    if not dim_type or len(dim_type) != ndim:
        _log.debug(
            "dim_type length %d does not match ndim %d; reversing all axes",
            len(dim_type),
            ndim,
        )
        reordered = np.transpose(data, axes=list(range(ndim - 1, -1, -1)))
        return reordered, list(range(ndim)), dim_type

    # Drop size-1 non-spatial axes (e.g. brukerapi's 'frame' placeholder) from
    # both the array and dim_type so all downstream code stays consistent.
    trivial = tuple(
        i for i, t in enumerate(dim_type) if t != "SPATIAL" and data.shape[i] == 1
    )
    if trivial:
        data = np.squeeze(data, axis=trivial)
        dim_type = [t for i, t in enumerate(dim_type) if i not in trivial]
        ndim = data.ndim

    spatial_idx = [i for i, t in enumerate(dim_type) if t == "SPATIAL"]
    fg_idx = [i for i, t in enumerate(dim_type) if t != "SPATIAL"]

    # FG axes reversed (outermost first), then spatial axes reversed (X last)
    new_order = fg_idx[::-1] + spatial_idx[::-1]
    reordered = np.transpose(data, axes=new_order)

    spatial_scale_indices = list(range(len(fg_idx), ndim))
    # Return cleaned dim_type alongside so callers can pass it to _extract_frame_groups
    return reordered, spatial_scale_indices, dim_type


# ---------------------------------------------------------------------------
# Spatial upsampling
# ---------------------------------------------------------------------------


def zoom_spatial(
    data: np.ndarray,
    scale: tuple[float, ...],
    spatial_scale_indices: list[int],
    zoom_to: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """
    Upsample spatial axes to at least `zoom_to` pixels using cubic interpolation.

    This matches ParaVision's display behaviour for small acquisition matrices
    (e.g. 48×48 MSME).  The physical scale is adjusted proportionally so that
    napari still shows correct mm coordinates.

    Non-spatial (FG) axes are never resampled. Cubic-spline prefiltering costs
    scale with the *total* array size, not just the spatial axes, so callers
    that only need one frame out of a multi-frame (echo/b-value) stack should
    select that frame first and call this on just the 2D/3D result — zooming
    the whole stack and discarding all but one frame wastes ~N× the spline
    work for an N-frame stack (see ``analysis.maps._anatomy_from_proc1``).

    Axes already >= zoom_to are left unchanged.
    """
    from scipy.ndimage import zoom  # import here — only needed when zoom_to > 1

    zoom_factors = [1.0] * data.ndim
    new_scale = list(scale)

    for ax in spatial_scale_indices:
        current = data.shape[ax]
        if current < zoom_to:
            factor = zoom_to / current
            zoom_factors[ax] = factor
            # Scale the voxel size down proportionally so physical size is preserved
            new_scale[ax] = scale[ax] / factor

    if all(f == 1.0 for f in zoom_factors):
        return data, scale

    # order=3 cubic spline — same visual quality as PV's default interpolation
    data = zoom(data, zoom_factors, order=3)
    return data, tuple(new_scale)


# ---------------------------------------------------------------------------
# Scale extraction
# ---------------------------------------------------------------------------


def _extract_scale(
    dataset,
    n_dims: int,
    spatial_scale_indices: list[int],
    dim_type: list[str],
) -> tuple[float, ...]:
    """
    Build a per-axis scale tuple (mm) for a napari Image layer.

    In napari axis order the layout is:
        [FG axes ...,  Z/slice axis (optional),  Y axis,  X axis]

    Uses brukerapi's `dataset.resolution` property which returns voxel size
    in mm as a 1-D array: (in-plane-1, in-plane-2[, slice]).

    For 2-D multi-slice data the slice axis is FG_SLICE in dim_type — it
    lives in the FG block, not in spatial_scale_indices — so we assign the
    slice thickness to it separately.

    Frame-group axes that are not FG_SLICE get scale = 1.0.
    """
    resolution = _get_resolution(dataset)  # (2,) or (3,) in PV order
    scale = [1.0] * n_dims

    if resolution is not None and len(resolution) >= 2:
        # In-plane (Y, X): last two spatial napari axes
        in_plane = resolution[:2]
        for out_pos, vox in zip(reversed(spatial_scale_indices[-2:]), in_plane):
            scale[out_pos] = float(vox)

        # Slice spacing: assign to any FG_SLICE axis in the FG block
        slice_mm = float(resolution[2]) if len(resolution) >= 3 else None
        if slice_mm is not None:
            fg_idx = [i for i, t in enumerate(dim_type) if t != "SPATIAL"]
            for napari_axis, orig_axis in enumerate(fg_idx[::-1]):
                if dim_type[orig_axis] in _SLICE_FG_TYPES:
                    scale[napari_axis] = slice_mm
                    break

    return tuple(scale)


def _get_resolution(dataset) -> np.ndarray | None:
    """
    Return voxel resolution in mm as a 1-D numpy array, or None on failure.

    brukerapi exposes this as `dataset.resolution` — in-plane sizes first,
    slice thickness last for multi-slice datasets.
    """
    try:
        res = np.asarray(dataset.resolution, dtype=float).ravel()
        if len(res) >= 2:
            return res
    except Exception:  # noqa: BLE001
        pass
    _log.debug("Could not read dataset.resolution; using 1 mm isotropic")
    return None
