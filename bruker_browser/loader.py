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
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

_log = logging.getLogger(__name__)


class LoadResult(NamedTuple):
    data: np.ndarray
    layer_name: str
    scale: tuple[float, ...]  # voxel size per axis, napari order (matches data)
    metadata: dict


def load_2dseq(study_path: Path | str, exp_id: str, proc_id: str = "1") -> LoadResult:
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

    Returns
    -------
    LoadResult
        numpy array in napari axis order (spatial last), matching scale tuple,
        suggested layer name, and raw metadata dict.
    """
    from brukerapi.folders import Study  # import here to keep startup fast

    study_path = Path(study_path)
    study = Study(str(study_path))

    dataset = study.get_dataset(exp_id=exp_id, proc_id=proc_id)
    dataset.load()

    raw: np.ndarray = dataset.data

    # --- dim_type tells us which axes are spatial vs frame-group ---
    dim_type: list[str] = []
    try:
        dim_type = list(dataset.dim_type)
    except Exception:
        _log.debug("dim_type not available for exp %s; assuming all-spatial", exp_id)

    # --- reorder axes so spatial dims are last (napari convention) ---
    data, spatial_scale_indices = _reorder_for_napari(raw, dim_type)

    # --- layer name ---
    seq = ""
    try:
        seq = dataset.get_value("PULPROG").strip("<>")
    except Exception:
        _log.debug("PULPROG not available for exp %s", exp_id)
    # Include proc_id when it is not the default "1" so loaded layers
    # are clearly distinguishable (e.g. "32 - DtiEpi / proc 2").
    proc_suffix = f" / proc {proc_id}" if proc_id != "1" else ""
    layer_name = f"{exp_id} - {seq}{proc_suffix}" if seq else f"{exp_id}{proc_suffix}"

    # --- voxel scale (in napari axis order) ---
    scale = _extract_scale(
        dataset, n_dims=data.ndim, spatial_scale_indices=spatial_scale_indices
    )

    # --- metadata ---
    meta: dict = {
        "study_path": str(study_path),
        "exp_id": exp_id,
        "proc_id": proc_id,
        "sequence": seq,
        "raw_shape": raw.shape,
        "dim_type": dim_type,
        "napari_shape": data.shape,
    }
    for key in ("VisuCoreSize", "VisuCoreFOV", "VisuCoreOrientation"):
        try:
            meta[key] = dataset.get_value(key)
        except Exception:
            _log.debug("Parameter %s not available for exp %s", key, exp_id)

    _log.debug(
        "Loaded exp %s: raw shape %s dim_type %s -> napari shape %s scale %s",
        exp_id,
        raw.shape,
        dim_type,
        data.shape,
        scale,
    )

    return LoadResult(data=data, layer_name=layer_name, scale=scale, metadata=meta)


# ---------------------------------------------------------------------------
# Axis reordering
# ---------------------------------------------------------------------------


def _reorder_for_napari(
    data: np.ndarray,
    dim_type: list[str],
) -> tuple[np.ndarray, list[int]]:
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
        # Nothing useful to reorder
        return data, list(range(ndim))

    if not dim_type or len(dim_type) != ndim:
        # No reliable dim_type — assume first N axes are spatial, rest are FG.
        # Reverse the whole thing as the safest default.
        _log.debug(
            "dim_type length %d does not match ndim %d; reversing all axes",
            len(dim_type),
            ndim,
        )
        reordered = np.transpose(data, axes=list(range(ndim - 1, -1, -1)))
        spatial_indices = list(range(ndim))  # treat all as spatial for scale purposes
        return reordered, spatial_indices

    # Partition indices into spatial vs frame-group
    spatial_idx = [i for i, t in enumerate(dim_type) if "spatial" in t.lower()]
    fg_idx = [i for i, t in enumerate(dim_type) if "spatial" not in t.lower()]

    # New order: frame-group axes first (reversed for natural FG ordering),
    # then spatial axes reversed (Z→Y→X becomes X→Y→Z after reversal: last=X, napari col)
    # napari dim order for display: (..., Z, row=Y, col=X)
    # brukerapi dim order:          (col=X, row=Y, Z, ...)
    # So: reverse spatial block, keep FG block in original relative order
    new_order = fg_idx[::-1] + spatial_idx[::-1]
    reordered = np.transpose(data, axes=new_order)

    # Spatial axes end up at positions len(fg_idx) .. ndim-1 in the new array
    spatial_scale_indices = list(range(len(fg_idx), ndim))

    return reordered, spatial_scale_indices


# ---------------------------------------------------------------------------
# Scale extraction
# ---------------------------------------------------------------------------


def _extract_scale(
    dataset,
    n_dims: int,
    spatial_scale_indices: list[int],
) -> tuple[float, ...]:
    """
    Build a per-axis scale tuple (mm) for a napari Image layer.

    Spatial axes get the voxel size derived from VisuCoreFOV / VisuCoreSize.
    Frame-group axes (echoes, diffusion directions, etc.) get scale = 1.0.
    """
    voxel_mm = _voxel_size_mm(dataset)

    scale = [1.0] * n_dims
    n_spatial = len(spatial_scale_indices)

    if voxel_mm and len(voxel_mm) >= n_spatial:
        # voxel_mm is in (X, Y[, Z]) order from brukerapi;
        # spatial_scale_indices run from last-Z to last-X in the reordered array,
        # so we assign in reverse to keep Z↔Z, Y↔Y, X↔X.
        for out_pos, vox in zip(reversed(spatial_scale_indices), voxel_mm):
            scale[out_pos] = vox
    else:
        for i in spatial_scale_indices:
            scale[i] = 1.0

    return tuple(scale)


def _voxel_size_mm(dataset) -> list[float]:
    """Return per-spatial-axis voxel size in mm, or [] on failure."""
    try:
        fov = np.asarray(dataset.get_value("VisuCoreFOV"), dtype=float).ravel()
        size = np.asarray(dataset.get_value("VisuCoreSize"), dtype=float).ravel()
        if fov.shape == size.shape and len(fov) >= 2:
            return (fov / size).tolist()
    except Exception:
        _log.debug(
            "Could not derive voxel scale from VisuCoreFOV/VisuCoreSize; using 1mm isotropic"
        )
    return []
