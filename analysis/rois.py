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

import numpy as np
import pandas as pd

from analysis.maps import PARAMETER_MAPS_DIR, sanitize_stem

_log = logging.getLogger(__name__)

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
    # in "ROI N" (N = draw order) the moment each one is finished.
    _blank_properties = {
        "name": np.array([""], dtype=object),
        "tissue": np.array([""], dtype=object),
    }
    layer.current_properties = _blank_properties

    # Boxed (not a plain local) so the closure below can persist a running
    # count across multiple calls — each call only sees however many blanks
    # remain *at that moment*, which isn't the same as overall draw order
    # once earlier blanks have already been filled in.
    _next_roi_number = [1]

    def _autofill_blank_names(event=None) -> None:
        names = list(layer.properties.get("name", []))
        tissues = list(layer.properties.get("tissue", []))
        changed = False
        for i in range(len(names)):
            if not str(names[i]).strip():
                names[i] = f"ROI {_next_roi_number[0]}"
                _next_roi_number[0] += 1
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
    """
    names = list(shapes_layer.properties.get("name", []))
    tissues = list(shapes_layer.properties.get("tissue", []))
    rois = []
    for i in range(len(shapes_layer.data)):
        rois.append(
            RoiRecord(
                name=str(names[i]) if i < len(names) else f"ROI {i + 1}",
                tissue=str(tissues[i]) if i < len(tissues) else "",
                shape_type=shapes_layer.shape_type[i],
                vertices=np.array(shapes_layer.data[i]).tolist(),
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
