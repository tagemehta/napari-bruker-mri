"""
RoiAnnotatorWidget — dock widget for interactive ROI drawing.

Rebuilds ParaVision's own "New -> pick shape -> draw -> named object"
annotation workflow inside napari: click New, choose Polygon or Rectangle,
trace it on the canvas, and it's immediately auto-labelled "ROI 1",
"ROI 2", ... and listed here. Draw as many as you want for a disc in one
uninterrupted pass, then double-click a list entry (or select it and click
"Rename Selected") to rename/tag it.

Rectangle ("square") sizing is fixed for annotation consistency: every
freshly drawn rectangle spawns as a 2x2px (4px area) box regardless of the
drag gesture (see analysis/rois.py::_SQUARE_SPAWN_SIDE_PX) — resize it with
napari's own handles if you want it bigger. The instant you finish a resize
(mouse release), it snaps live to the nearest of 4/9/16px area (2x2/3x3/4x4)
(analysis/rois.py::_snap_selected_rectangles_on_release), so you see the
snap happen on-canvas rather than only discovering it once saved — every
persisted rectangle is one of exactly three sizes.

Whichever layer is selected in napari's own layer list when "New" is
clicked decides which map ("target") the new ROI belongs to. Each target
owns exactly one napari Shapes layer (e.g. "T2 ROIs", "ADC ROIs") that all
of its ROIs live in as separate shapes — T2 and ADC never share a Shapes
layer, since they don't share a pixel grid (see analysis/rois.py module
docstring).
"""

from __future__ import annotations

import logging

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_log = logging.getLogger(__name__)

# Maps the combo box's display text to the napari Shapes drawing mode it arms.
_SHAPE_MODES = {"Polygon": "add_polygon", "Rectangle": "add_rectangle"}
_ROLE = Qt.UserRole


class _RoiTarget:
    """One annotatable map: its Shapes layer plus the layer names that select it."""

    def __init__(self, target_name: str, shapes_layer, member_layer_names: list[str]):
        self.target_name = target_name
        self.shapes_layer = shapes_layer
        self.member_layer_names = set(member_layer_names) | {shapes_layer.name}


class RoiAnnotatorWidget(QWidget):
    """
    Dock widget: New/Finish/Delete buttons + shape-type selector + a live
    list of every ROI drawn so far, across all registered targets.

    Parameters
    ----------
    napari_viewer:
        The napari.Viewer instance.

    Usage
    -----
    Call :meth:`register_target` once per map (T2, ADC, ...) right after
    adding its image and Shapes layers to the viewer, e.g.::

        widget = RoiAnnotatorWidget(viewer)
        widget.register_target("T2", t2_shapes_layer, ["T2 anatomy", "T2 map"])
        widget.register_target("ADC", adc_shapes_layer, ["ADC anatomy", "ADC map"])
        viewer.window.add_dock_widget(widget, name="ROI Annotator")
    """

    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._targets: dict[str, _RoiTarget] = {}
        self._drawing_target: _RoiTarget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._target_label = QLabel("Target: (select a layer)")
        layout.addWidget(self._target_label)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_rename)
        layout.addWidget(self._list, stretch=1)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Shape:"))
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(list(_SHAPE_MODES))
        shape_row.addWidget(self._shape_combo, stretch=1)
        layout.addLayout(shape_row)

        button_row = QHBoxLayout()
        self._new_button = QPushButton("New")
        self._new_button.clicked.connect(self._on_new)
        button_row.addWidget(self._new_button)
        self._finish_button = QPushButton("Finish")
        self._finish_button.clicked.connect(self._on_finish)
        button_row.addWidget(self._finish_button)
        layout.addLayout(button_row)

        rename_button = QPushButton("Rename Selected")
        rename_button.clicked.connect(self._on_rename_selected)
        layout.addWidget(rename_button)

        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self._on_delete)
        layout.addWidget(delete_button)

        self.viewer.layers.selection.events.changed.connect(self._on_layer_changed)
        self._on_layer_changed()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_target(
        self, target_name: str, shapes_layer, member_layer_names: list[str]
    ) -> None:
        """
        Register one annotatable map. ``member_layer_names`` should list
        every other layer (anatomy image, parametric map image, ...) that
        belongs to this map, so selecting any of them in napari's layer
        list also arms this target for the next "New" click.
        """
        target = _RoiTarget(target_name, shapes_layer, member_layer_names)
        self._targets[target_name] = target
        shapes_layer.events.data.connect(self._refresh_list)
        self._refresh_list()

    # ------------------------------------------------------------------
    # Layer-selection tracking
    # ------------------------------------------------------------------

    def _current_target(self) -> _RoiTarget | None:
        """Return the target owning the single selected napari layer, or None."""
        selected = list(self.viewer.layers.selection)
        if len(selected) != 1:
            return None
        name = selected[0].name
        for target in self._targets.values():
            if name in target.member_layer_names:
                return target
        return None

    def _on_layer_changed(self, event=None) -> None:
        target = self._current_target()
        self._target_label.setText(
            f"Target: {target.target_name}" if target else "Target: (select a layer)"
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        """Arm the currently targeted map's Shapes layer for drawing one new ROI."""
        target = self._current_target()
        if target is None:
            _log.warning(
                "RoiAnnotatorWidget: no target selected — click one of a "
                "map's layers (anatomy, parametric map, or its ROI layer) first."
            )
            return
        self._drawing_target = target
        mode = _SHAPE_MODES[self._shape_combo.currentText()]
        self.viewer.layers.selection.active = target.shapes_layer
        target.shapes_layer.mode = mode

    def _on_finish(self) -> None:
        """Manually drop out of drawing mode (napari itself finishes a shape on double-click/Enter)."""
        if self._drawing_target is not None:
            self._drawing_target.shapes_layer.mode = "pan_zoom"
            self._drawing_target = None

    # ------------------------------------------------------------------
    # List / rename / delete
    # ------------------------------------------------------------------

    def _refresh_list(self, event=None) -> None:
        self._list.clear()
        for target in self._targets.values():
            names = target.shapes_layer.properties.get("name", [])
            tissues = target.shapes_layer.properties.get("tissue", [])
            for i, name in enumerate(names):
                tissue = tissues[i] if i < len(tissues) else ""
                label = f"{target.target_name} — {name}"
                if tissue:
                    label += f" ({tissue})"
                item = QListWidgetItem(label)
                item.setData(_ROLE, (target.target_name, i))
                self._list.addItem(item)

    def _on_rename_selected(self) -> None:
        """Rename button handler — same as double-clicking the selected list entry."""
        item = self._list.currentItem()
        if item is not None:
            self._on_rename(item)

    def _on_rename(self, item: QListWidgetItem) -> None:
        """Rename a shape via a single text dialog."""
        target_name, idx = item.data(_ROLE)
        target = self._targets[target_name]
        names = list(target.shapes_layer.properties.get("name", []))
        tissues = list(target.shapes_layer.properties.get("tissue", []))
        if idx >= len(names):
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename ROI", "Name:", text=str(names[idx])
        )
        if not ok or not new_name.strip():
            return
        names[idx] = new_name.strip()

        target.shapes_layer.properties = {
            "name": np.array(names, dtype=object),
            "tissue": np.array(tissues, dtype=object),
        }
        self._refresh_list()

    def _on_delete(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        target_name, idx = item.data(_ROLE)
        target = self._targets[target_name]
        if idx < len(target.shapes_layer.data):
            target.shapes_layer.selected_data = {idx}
            target.shapes_layer.remove_selected()
        self._refresh_list()
