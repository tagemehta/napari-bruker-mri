"""
ROI definitions and stats for per-disc T2/ADC analysis.

ROIs are drawn on a map's own signal-intensity ("anatomy") image — e.g. an
MSME echo for T2, a high-b DWI image for ADC — rather than on the
parametric map itself. That's standard practice (better contrast for
delineating tissue boundaries) and it also sidesteps a real correctness
problem: T2 and ADC acquisitions for the same disc do not share a pixel
grid. Verified against real data (``RA_2022_07_20.f22`` exp 6/7, same
disc): T2 (MSME) uses FOV 10x12mm / matrix 48x60, ADC (DtiEpi) uses FOV
20.8x12mm / matrix 128x72, with different in-plane center offsets too.
Projecting one ROI across both grids would require real affine
registration; instead, each anatomy image shares its *own* map's exact
native grid (both are derived from the same acquisition), so ROIs are
always drawn and stored separately per map kind and rasterized straight
onto that map's array — no coordinate transform needed anywhere.

Saved layout
------------
- ``parameter_maps/rois/{study_name}_exp{exp_id}_{kind}.json`` — one file
  per (study_name, exp_id, kind), holding every ROI drawn for that map.
- ``parameter_maps/roi_stats.csv`` — aggregated stats across every disc,
  one row per (study_name, exp_id, roi_name, map_kind); recomputing a
  disc's stats updates its rows in place instead of duplicating them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import re

import numpy as np
import pandas as pd

from analysis.maps import PARAMETER_MAPS_DIR, sanitize_stem

_log = logging.getLogger(__name__)

_ROI_NAME_RE = re.compile(r"^ROI\s+(\d+)$", re.IGNORECASE)

ROI_STATS_COLUMNS = [
    "study_name",
    "exp_id",
    "roi_name",
    "tissue",
    "map_kind",
    "map_source",
    "mean",
    "min",
    "max",
    "std",
    "n_pixels",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RoiRecord:
    """
    One drawn ROI, as stored on disk and passed between the annotation
    script and the plotting/stats functions below.

    Attributes
    ----------
    name:
        Free-text label, e.g. ``"NP"``, ``"AF sample 2"``.
    tissue:
        ``"NP"`` or ``"AF"`` — used for grouping/filtering in the
        aggregated CSV. Not validated against a fixed set; whatever the
        user enters is stored as-is.
    shape_type:
        napari ``Shapes`` shape type, e.g. ``"polygon"``.
    vertices:
        Shape vertices in the *same data-pixel coordinates* as the map
        array this ROI belongs to (i.e. ``layer.data[i].tolist()`` from
        the Shapes layer drawn on that map's own anatomy image).
    study_name, exp_id, kind:
        Identify which map this ROI was drawn against (``kind`` is
        ``"t2"`` or ``"adc"``).
    """

    name: str
    tissue: str
    shape_type: str
    vertices: list[list[float]]
    study_name: str
    exp_id: str
    kind: str


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def rois_path(
    study_name: str,
    exp_id: str,
    kind: str,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> Path:
    """Path to a disc's saved ROI file, without touching disk."""
    stem = f"{sanitize_stem(study_name)}_exp{exp_id}_{kind}"
    return Path(results_dir) / "rois" / f"{stem}.json"


def save_rois(
    rois: list[RoiRecord],
    study_name: str,
    exp_id: str,
    kind: str,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> Path:
    """Save the full ROI set for one (study, exp, kind), overwriting any previous file."""
    path = rois_path(study_name, exp_id, kind, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rois], indent=2))
    _log.debug("Saved %d ROIs to %s", len(rois), path)
    return path


def load_rois(
    study_name: str,
    exp_id: str,
    kind: str,
    results_dir: Path | str = PARAMETER_MAPS_DIR,
) -> list[RoiRecord]:
    """
    Load a previously saved ROI set.

    Raises
    ------
    FileNotFoundError
        If no ROI file exists for this (study, exp, kind).
    """
    path = rois_path(study_name, exp_id, kind, results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved ROIs for {study_name} exp {exp_id} ({kind}): {path}"
        )
    records = json.loads(path.read_text())
    return [RoiRecord(**r) for r in records]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def compute_roi_stats(mask: np.ndarray, map_data: np.ndarray) -> dict:
    """
    Mean/min/max/SD of ``map_data`` inside ``mask``, ignoring NaN pixels
    (maps use NaN for background/out-of-plausible-range pixels).

    Returns all-NaN stats with ``n_pixels=0`` for an empty or fully-NaN
    mask, rather than raising — one bad/empty ROI shouldn't abort stats
    for the rest of a disc.
    """
    values = map_data[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "n_pixels": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "n_pixels": int(values.size),
    }


def rasterize_roi(roi: RoiRecord, mask_shape: tuple[int, ...]) -> np.ndarray:
    """
    Rasterize a saved ROI's vertices into a boolean mask of ``mask_shape``.

    Lets stats be recomputed from saved ROI vertices without a live napari
    Shapes layer. ``napari.layers.Shapes`` is used purely as a rasterizer
    here (a plain data-layer object, no Qt/viewer involved), so this still
    works headlessly.
    """
    from napari.layers import Shapes

    layer = Shapes(data=[np.array(roi.vertices)], shape_type=[roi.shape_type])
    return layer.to_masks(mask_shape)[0]


_ROI_EDGE_WIDTH = 0.1  # thin outline — native map grids are only tens of pixels across

# Rectangle ("square") ROIs are kept to a small, fixed set of sizes for
# annotation consistency — native map grids are small enough (tens of
# px/axis) that free-hand rectangles vary wildly in pixel count otherwise.
# A freshly drawn rectangle always spawns at this fixed side length,
# regardless of how big the draw gesture was...
_SQUARE_SPAWN_SIDE_PX = 2.0  # -> 4px area
# ...and whatever size it's resized to (via napari's own resize handles)
# gets snapped to whichever of these areas is closest the moment that resize
# gesture finishes (mouse release — see _snap_selected_rectangles_on_release
# in add_roi_shapes_layer), so the snap is visible live on-canvas. This is
# the *only* place snapping happens — deliberately not repeated at save
# time, so what's saved is always exactly what's on-canvas, no surprise
# silent resize. Polygons are untouched throughout — this only ever applies
# to shape_type == "rectangle".
_ALLOWED_SQUARE_AREAS = (4.0, 9.0, 16.0)


def _rect_center(verts: np.ndarray) -> np.ndarray:
    """Center (Y, X) of a rectangle's bounding box."""
    spatial = np.asarray(verts, dtype=float)[:, -2:]
    return (spatial.min(axis=0) + spatial.max(axis=0)) / 2.0


def _square_vertices(center: np.ndarray, side: float, leading: np.ndarray) -> np.ndarray:
    """
    Build a rectangle's 4 corner vertices for an axis-aligned square of
    ``side`` pixels centered at ``center`` (Y, X). ``leading`` is the
    non-spatial (e.g. Z) coordinate prefix, copied to all 4 corners.
    """
    half = side / 2.0
    lo = center - half
    hi = center + half
    corners_spatial = np.array(
        [[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]], [hi[0], lo[1]]]
    )
    leading_rows = np.tile(leading, (4, 1))
    return np.hstack([leading_rows, corners_spatial])


def _snap_to_nearest_square_vertices(verts: np.ndarray) -> np.ndarray:
    """
    Snap a rectangle's current bounding box to the nearest allowed area
    (see ``_ALLOWED_SQUARE_AREAS``), producing a true square (equal Y/X
    side) centered on its current center.
    """
    verts = np.asarray(verts, dtype=float)
    spatial = verts[:, -2:]
    lo = spatial.min(axis=0)
    hi = spatial.max(axis=0)
    size = hi - lo
    area = float(size[0] * size[1])
    nearest_area = min(_ALLOWED_SQUARE_AREAS, key=lambda a: abs(a - area))
    side = nearest_area**0.5
    center = (lo + hi) / 2.0
    return _square_vertices(center, side, verts[0, :-2])


def add_roi_shapes_layer(
    viewer,
    name: str,
    ndim: int,
    rois: list[RoiRecord],
    scale: tuple,
    edge_width: float = _ROI_EDGE_WIDTH,
):
    """
    Add one napari Shapes layer for a map's ROIs, pre-populated with ``rois``.

    Unlike ParaVision's own annotation tool, this doesn't restrict a disc to
    a fixed set of slots — draw as many ROIs as you want (NP, AF, extra
    samples, ...) as separate shapes in this one layer, matching how
    ``roi_annotator_widget.RoiAnnotatorWidget`` creates them one at a time.

    Each shape carries its ``name``/``tissue`` in the layer's per-shape
    ``properties``, which napari keeps aligned with the shape data across
    interactive add/delete. A newly drawn shape starts with a blank name,
    but is immediately auto-labelled "ROI 1", "ROI 2", ... (draw order) and
    shown on-canvas via the layer's ``text``, so freshly drawn shapes are
    distinguishable at a glance before being renamed for real (e.g. via the
    widget's list, or ``analyze_disc.py``'s post-close terminal prompt).

    Requires a live napari viewer.
    """
    layer = viewer.add_shapes(
        name=name,
        ndim=ndim,
        scale=scale,
        edge_width=edge_width,
        face_color="transparent",
        edge_color="yellow",
        properties={"name": [], "tissue": []},
        property_choices={"tissue": ["NP", "AF"]},
        text={"string": "{name}", "size": 9, "color": "white", "anchor": "upper_left"},
    )
    if rois:
        layer.add(
            [np.array(r.vertices) for r in rois],
            shape_type=[r.shape_type for r in rois],
        )
        layer.properties = {
            "name": np.array([r.name for r in rois], dtype=object),
            "tissue": np.array([r.tissue for r in rois], dtype=object),
        }

    # New interactively-drawn shapes start blank; the callback below fills
    # in "ROI N" (N = one more than the highest existing "ROI <n>" name) the
    # moment each one is finished.
    _blank_properties = {
        "name": np.array([""], dtype=object),
        "tissue": np.array([""], dtype=object),
    }
    layer.current_properties = _blank_properties

    def _next_roi_number(names: list) -> int:
        """
        One more than the highest ``"ROI <n>"`` name currently present, not
        a running count — so deleting the highest-numbered ROI frees its
        number again (1..6, delete "ROI 6" -> next is "ROI 6" again), while
        deleting a lower one doesn't (1..6, delete "ROI 2" -> next is still
        "ROI 7"). Renamed ROIs (anything not matching the pattern) are
        simply ignored when computing the max.
        """
        nums = [
            int(match.group(1))
            for name in names
            if (match := _ROI_NAME_RE.match(str(name).strip()))
        ]
        return max(nums, default=0) + 1

    def _autofill_blank_names(event=None) -> None:
        names = list(layer.properties.get("name", []))
        tissues = list(layer.properties.get("tissue", []))
        changed = False
        for i in range(len(names)):
            if not str(names[i]).strip():
                names[i] = f"ROI {_next_roi_number(names)}"
                changed = True
        if changed:
            layer.properties = {
                "name": np.array(names, dtype=object),
                "tissue": np.array(tissues, dtype=object),
            }
        # napari resyncs current_properties (the template for the *next*
        # new shape) to an existing shape's values on selection/deletion —
        # force it back to blank on every data change, not just when we
        # filled something in, or the next new shape silently inherits a
        # stale non-blank name and never gets auto-numbered.
        layer.current_properties = _blank_properties

    layer.events.data.connect(_autofill_blank_names)

    # Tracks how many shapes existed as of the last event, so a growth in
    # shape count (vs. a resize/delete of an existing one) can be detected —
    # only freshly-added rectangles get forced to the spawn size below.
    # Starts at len(rois) so preloaded ROIs (added above, before this
    # callback is connected) are never touched by it.
    _known_shape_count = [len(rois)]

    def _force_spawn_square_size(event=None) -> None:
        n = len(layer.data)
        if n <= _known_shape_count[0]:
            # Shrunk (delete) or unchanged (a resize of an existing shape,
            # which is intentionally left alone until save) — just resync.
            _known_shape_count[0] = n
            return
        new_data = list(layer.data)
        changed = False
        for i in range(_known_shape_count[0], n):
            if layer.shape_type[i] != "rectangle":
                continue
            center = _rect_center(new_data[i])
            leading = np.asarray(new_data[i], dtype=float)[0, :-2]
            new_data[i] = _square_vertices(center, _SQUARE_SPAWN_SIDE_PX, leading)
            changed = True
        # Updated before the mutating assignment below so the nested
        # events.data fire it triggers (napari's Shapes.data setter fires
        # synchronously mid-assignment) sees n <= _known_shape_count[0] and
        # returns immediately instead of recursing.
        _known_shape_count[0] = n
        if changed:
            layer.data = new_data

    layer.events.data.connect(_force_spawn_square_size)

    def _snap_selected_rectangles_on_release(layer, event):
        """
        Mouse-drag callback (napari's generator convention: runs up to the
        first ``yield`` on press, is resumed once per move, and resumes one
        final time — past the ``while`` loop — on release) that snaps
        whichever rectangle(s) were just dragged/resized to the nearest
        allowed square area, live, so the snap is visible immediately
        instead of only appearing once the ROI set is saved. A no-op for a
        plain move (size unchanged -> already at an allowed area -> nothing
        to snap) and for anything that isn't a rectangle.
        """
        yield
        while event.type == "mouse_move":
            yield
        new_data = None
        for idx in list(layer.selected_data):
            if idx >= len(layer.data) or layer.shape_type[idx] != "rectangle":
                continue
            verts = np.asarray(layer.data[idx], dtype=float)
            snapped = _snap_to_nearest_square_vertices(verts)
            if not np.allclose(verts, snapped):
                if new_data is None:
                    new_data = list(layer.data)
                new_data[idx] = snapped
        if new_data is not None:
            layer.data = new_data

    layer.mouse_drag_callbacks.append(_snap_selected_rectangles_on_release)
    return layer


def collect_rois_from_shapes_layer(
    shapes_layer,
    study_name: str,
    exp_id: str,
    kind: str,
) -> list[RoiRecord]:
    """
    Build one :class:`RoiRecord` per shape currently in ``shapes_layer``
    (as created by :func:`add_roi_shapes_layer`), reading ``name``/``tissue``
    back from the layer's ``properties``.

    Rectangle vertices are already snapped to an allowed square area by the
    time they get here — live, the moment a resize gesture finishes (see
    ``_snap_selected_rectangles_on_release`` in ``add_roi_shapes_layer`` —
    what you see on-canvas is exactly what gets saved, no silent resize
    happens here).
    """
    names = list(shapes_layer.properties.get("name", []))
    tissues = list(shapes_layer.properties.get("tissue", []))
    rois = []
    for i in range(len(shapes_layer.data)):
        shape_type = shapes_layer.shape_type[i]
        vertices = np.asarray(shapes_layer.data[i], dtype=float)
        rois.append(
            RoiRecord(
                name=str(names[i]) if i < len(names) else f"ROI {i + 1}",
                tissue=str(tissues[i]) if i < len(tissues) else "",
                shape_type=shape_type,
                vertices=vertices.tolist(),
                study_name=study_name,
                exp_id=exp_id,
                kind=kind,
            )
        )
    return rois


def upsert_roi_stats(
    rows: list[dict], results_dir: Path | str = PARAMETER_MAPS_DIR
) -> Path:
    """
    Append ``rows`` to ``{results_dir}/roi_stats.csv``, overwriting any
    existing row with the same (study_name, exp_id, roi_name, map_kind)
    key rather than duplicating it — so recomputing a disc's stats
    updates its rows in place.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "roi_stats.csv"

    key_cols = ["study_name", "exp_id", "roi_name", "map_kind"]

    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={c: str for c in key_cols})
    else:
        old = pd.DataFrame(columns=ROI_STATS_COLUMNS)

    new = pd.DataFrame(rows, columns=ROI_STATS_COLUMNS)
    for c in key_cols:
        new[c] = new[c].astype(str)

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)

    _log.debug("Wrote %d rows to %s", len(combined), csv_path)
    return csv_path
