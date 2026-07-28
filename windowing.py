"""
FrameGroupWidget — right-panel dock for switching between frame-group axes.

When a Bruker layer is selected in napari, this widget reads
``layer.metadata["_frame_groups"]`` and builds one labelled QComboBox per
selectable FG axis (echoes, parametric-map type, diffusion direction, …).
Changing a dropdown re-indexes ``layer.metadata["_full_data"]`` and updates
``layer.data`` in-place — no disk re-read required.  Contrast limits are
preserved across the update.

Layers with no frame-group metadata (plain FLASH/RARE scans, non-Bruker
layers) show an empty widget.
"""

from __future__ import annotations

import logging

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_log = logging.getLogger(__name__)


class FrameGroupWidget(QWidget):
    """
    Dock widget showing one labelled dropdown per selectable FG axis.

    Parameters
    ----------
    napari_viewer:
        The napari.Viewer instance to track layer selections from.
    """

    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.layer = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Container rebuilt each time a new layer is selected
        self._fg_container = QWidget()
        self._fg_layout = QVBoxLayout(self._fg_container)
        self._fg_layout.setContentsMargins(0, 0, 0, 0)
        self._fg_layout.setSpacing(4)
        layout.addWidget(self._fg_container)
        layout.addStretch()

        # Parallel list of (QComboBox, FrameGroupInfo) for the active layer
        self._fg_combos: list[tuple[QComboBox, object]] = []

        self.viewer.layers.selection.events.changed.connect(self._on_layer_changed)
        self._on_layer_changed()

    # ------------------------------------------------------------------

    def _current_image(self):
        """Return the single selected Image layer, or None."""
        selected = list(self.viewer.layers.selection)
        if len(selected) != 1:
            return None
        layer = selected[0]
        return layer if layer.__class__.__name__ == "Image" else None

    def _on_layer_changed(self, event=None) -> None:
        """Rebuild dropdowns whenever the napari layer selection changes."""
        self.layer = self._current_image()
        self._rebuild()

    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """
        Tear down existing dropdowns and rebuild them for the active layer.

        Shows nothing when the layer has no frame-group metadata (e.g.
        non-Bruker layers or purely spatial scans with no FG axes).
        """
        for combo, _ in self._fg_combos:
            combo.deleteLater()
        self._fg_combos = []
        while self._fg_layout.count():
            item = self._fg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.layer is None:
            return
        frame_groups = self.layer.metadata.get("_frame_groups", [])
        if not frame_groups:
            return

        for fg in frame_groups:
            row = QHBoxLayout()
            label = QLabel(fg.name.replace("FG_", "").title() + ":")
            label.setMinimumWidth(60)
            combo = QComboBox()
            combo.addItems(fg.labels)
            row.addWidget(label)
            row.addWidget(combo, stretch=1)
            container = QWidget()
            container.setLayout(row)
            self._fg_layout.addWidget(container)
            self._fg_combos.append((combo, fg))
            combo.currentIndexChanged.connect(self._apply_selection)

    def _apply_selection(self) -> None:
        """
        Re-index the full array using the current dropdown positions and push
        the result into the active layer.

        Processes FG axes outermost-first, tracking the axis offset as each
        axis is removed by indexing.  Preserves contrast limits across the
        update so the view does not reset.
        """
        if self.layer is None:
            return

        meta = self.layer.metadata
        full_data: np.ndarray | None = meta.get("_full_data")
        full_scale: tuple | None = meta.get("_full_scale")
        frame_groups: list | None = meta.get("_frame_groups")
        if full_data is None or full_scale is None or frame_groups is None:
            return

        data = full_data
        scale_list = list(full_scale)
        sorted_pairs = sorted(
            zip(self._fg_combos, frame_groups), key=lambda t: t[1].axis
        )
        axis_offset = 0
        for (combo, _), fg in sorted_pairs:
            ax = fg.axis - axis_offset
            data = np.take(data, combo.currentIndex(), axis=ax)
            scale_list.pop(ax)
            axis_offset += 1

        prev_limits = self.layer.contrast_limits
        self.layer.data = data
        self.layer.scale = tuple(scale_list)
        try:
            self.layer.contrast_limits = prev_limits
        except Exception:  # noqa: BLE001
            self.layer.reset_contrast_limits()
