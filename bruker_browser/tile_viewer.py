"""
load_into_viewer — load a Bruker proc directly into a napari viewer.

The tile-slot grid has been removed.  This module now provides a single
function used by the tree widget's double-click / drag handler.  Multiple
procs are just stacked as separate napari layers; the built-in layer list
is sufficient for switching between them.

Axis handling
-------------
After load_2dseq the data array has the layout:

    [FG axes ...,  Z/slice axis (optional, scale=slice_thickness),  Y,  X]

frame_groups only describes *selectable* FG axes (not FG_SLICE).  The slice
axis remains in the array and napari treats it as the Z dimension.

To show the initial frame we index each selectable FG axis at position 0,
using each FrameGroupInfo.axis to find the right axis in the current array
(accounting for axes already removed by earlier indexing operations).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .loader import FrameGroupInfo, load_2dseq

if TYPE_CHECKING:
    import napari

_log = logging.getLogger(__name__)


def load_into_viewer(
    viewer: napari.Viewer,
    study_path: str | Path,
    exp_id: str,
    proc_id: str = "1",
    study_name: str = "",
) -> None:
    """
    Load a 2dseq proc into *viewer* as a napari Image layer.

    If a layer with the same name already exists it is replaced in-place so
    double-clicking the same proc twice does not accumulate duplicate layers.
    """
    try:
        result = load_2dseq(Path(study_path), exp_id, proc_id, study_name=study_name)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to load exp %s proc %s: %s", exp_id, proc_id, exc)
        return

    display_data, display_scale = _initial_display(
        result.data, result.scale, result.frame_groups
    )

    # Augment metadata with everything needed for the FG selector widget:
    # the full array (pre-display), its scale, and the frame_group descriptors.
    meta = dict(result.metadata)
    meta["_full_data"] = result.data
    meta["_full_scale"] = result.scale
    meta["_frame_groups"] = result.frame_groups

    name = result.layer_name
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = display_data
        layer.scale = display_scale
        layer.metadata.update(meta)
    else:
        layer = viewer.add_image(
            display_data,
            name=name,
            scale=display_scale,
            metadata=meta,
        )

    # Enable continuous auto-contrast so limits update as the user scrolls
    # through slices or switches frame groups, matching PV's default behaviour.
    layer._keep_auto_contrast = True

    _log.debug(
        "Loaded %s into viewer (shape=%s scale=%s)",
        name,
        display_data.shape,
        display_scale,
    )


def _initial_display(
    data: np.ndarray,
    scale: tuple[float, ...],
    frame_groups: list[FrameGroupInfo],
) -> tuple[np.ndarray, tuple[float, ...]]:
    """
    Extract the first element along each selectable FG axis, leaving the
    slice/Z axis and spatial axes intact.

    FrameGroupInfo.axis gives the position of each FG in the *full* array.
    As we index each one out, subsequent axes shift left by 1 — we track
    this with an offset.

    Returns the sub-array and matching scale tuple for napari.
    """
    if not frame_groups:
        return data, scale

    # Sort by axis position so we process outermost axes first.
    # Each time we index one out, axes with higher numbers shift down by 1.
    sorted_fgs = sorted(frame_groups, key=lambda fg: fg.axis)

    scale_list = list(scale)
    axis_offset = 0
    for fg in sorted_fgs:
        ax = fg.axis - axis_offset
        data = np.take(data, 0, axis=ax)
        scale_list.pop(ax)
        axis_offset += 1

    return data, tuple(scale_list)
