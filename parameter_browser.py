"""
ParameterBrowserWidget — right-panel dock listing saved parameter maps/ROIs.

Purely read-only: it scans ``parameter_maps/`` (see ``analysis/maps.py`` /
``analysis/rois.py``) for anything already generated for the currently
loaded study (via ``analyze_disc.py``) and lets you load it into the
viewer with a button click. It never generates or draws anything itself —
that stays in ``analyze_disc.py``, run as a separate, deliberate step.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from analysis.maps import PARAMETER_MAPS_DIR, load_map, sanitize_stem
from analysis.rois import add_roi_shapes_layer, load_rois, rois_path

_log = logging.getLogger(__name__)

_ENTRY_ROLE = Qt.UserRole


class ParameterBrowserWidget(QWidget):
    """
    Dock widget showing every saved T2/ADC map (and ROI set, if any) found
    in ``parameter_maps/`` for the currently loaded study.

    Parameters
    ----------
    napari_viewer:
        The napari.Viewer instance to add loaded layers to.
    results_dir:
        Where ``analyze_disc.py`` saves its output. Defaults to
        ``analysis.maps.PARAMETER_MAPS_DIR``.
    """

    def __init__(
        self, napari_viewer, results_dir: Path | str = PARAMETER_MAPS_DIR
    ) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.results_dir = Path(results_dir)
        self._study_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._study_label = QLabel("(no study loaded)")
        layout.addWidget(self._study_label)

        self._list = QListWidget()
        layout.addWidget(self._list, stretch=1)

        load_button = QPushButton("Load Selected")
        load_button.clicked.connect(self._on_load_selected)
        layout.addWidget(load_button)

    # ------------------------------------------------------------------

    def on_experiment_activated(
        self, study_path: str, exp_id: str, proc_id: str, study_name: str
    ) -> None:
        """Connect this to ``BrukerTreeWidget.experiment_activated`` in main.py."""
        if study_name == self._study_name:
            return  # already browsing this study; no need to rescan
        self._study_name = study_name
        self._study_label.setText(f"Study: {study_name}")
        self._rescan()

    # ------------------------------------------------------------------

    def _rescan(self) -> None:
        self._list.clear()
        if not self._study_name or not self.results_dir.exists():
            return

        stem = sanitize_stem(self._study_name)
        pattern = re.compile(rf"^{re.escape(stem)}_exp(?P<exp_id>.+)_(?P<kind>t2|adc)$")

        found = []
        for npy_path in sorted(self.results_dir.glob(f"{stem}_exp*_*.npy")):
            m = pattern.match(npy_path.stem)
            if m:
                found.append((m.group("exp_id"), m.group("kind")))

        for exp_id, kind in found:
            has_rois = rois_path(
                self._study_name, exp_id, kind, self.results_dir
            ).exists()
            label = f"exp {exp_id} — {kind.upper()}" + (
                "  (+ ROIs)" if has_rois else ""
            )
            item = QListWidgetItem(label)
            item.setData(_ENTRY_ROLE, (exp_id, kind))
            self._list.addItem(item)

        if not found:
            self._list.addItem("(none found)")

    # ------------------------------------------------------------------

    def _on_load_selected(self) -> None:
        item = self._list.currentItem()
        if item is None or item.data(_ENTRY_ROLE) is None or not self._study_name:
            return
        exp_id, kind = item.data(_ENTRY_ROLE)

        try:
            result = load_map(self._study_name, exp_id, kind, self.results_dir)
        except FileNotFoundError as exc:
            _log.warning("Could not load %s: %s", item.text(), exc)
            return

        layer_name = f"{kind.upper()} exp{exp_id}"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = result.data
        else:
            self.viewer.add_image(
                result.data,
                name=layer_name,
                scale=result.scale,
                colormap="magma",
            )

        try:
            rois = load_rois(self._study_name, exp_id, kind, self.results_dir)
        except FileNotFoundError:
            return

        # Remove any previously loaded ROI layer for this exp/kind before
        # re-adding, so re-loading the same map doesn't stack duplicates.
        roi_layer_name = f"{kind.upper()} exp{exp_id} ROIs"
        if roi_layer_name in self.viewer.layers:
            self.viewer.layers.remove(roi_layer_name)
        add_roi_shapes_layer(
            self.viewer, roi_layer_name, result.data.ndim, rois, result.scale
        )
        self._jump_to_roi_slice(rois)

    def _jump_to_roi_slice(self, rois) -> None:
        """
        Move the viewer to a slice that actually contains an ROI.

        ROIs are per-slice in napari (a shape only exists on the Z index it
        was drawn on — see AGENTS.md), but napari opens a freshly loaded
        multi-slice layer on the *middle* slice regardless. Without this,
        loading a map+ROIs here silently shows nothing whenever the ROIs
        happen to live on a different slice, which looks like a bug.
        """
        if not rois:
            return
        # vertices are (Z, Y, X, ...); every vertex of a given shape shares
        # the same leading (non-spatial) axes, so the first vertex suffices.
        z = int(rois[0].vertices[0][0])
        if self.viewer.dims.ndim >= 1:
            self.viewer.dims.set_current_step(0, z)
