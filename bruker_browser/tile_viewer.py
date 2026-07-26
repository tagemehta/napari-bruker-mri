"""
TiledViewerWidget — a 2×2 grid of "slots".

Each slot can hold one napari Image layer.  The user can:
  - Drop an experiment from the BrukerTreeWidget onto a slot.
  - Click "Load" inside a slot (when an experiment is staged in the tree).
  - Click "✕" inside a slot to clear its layer from the napari viewer.

The widget communicates back to the main napari viewer via a reference
passed at construction time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .loader import load_2dseq
from .tree_widget import BRUKER_EXP_MIME

if TYPE_CHECKING:
    import napari


_SLOT_STYLE_EMPTY = (
    "QFrame { border: 2px dashed #555; border-radius: 6px; background: #1a1a1a; }"
)
_SLOT_STYLE_FILLED = (
    "QFrame { border: 2px solid #4caf50; border-radius: 6px; background: #1e2a1e; }"
)
_SLOT_STYLE_HOVER = (
    "QFrame { border: 2px dashed #4caf50; border-radius: 6px; background: #1e2a1e; }"
)


class _SlotWidget(QFrame):
    """A single drop-target slot in the 2×2 grid."""

    def __init__(
        self, slot_index: int, viewer: napari.Viewer, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._index = slot_index
        self._viewer = viewer
        self._layer_name: str | None = None

        self.setAcceptDrops(True)
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(_SLOT_STYLE_EMPTY)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Title row
        title_row = QHBoxLayout()
        self._title_label = QLabel(f"Slot {self._index + 1}")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self._clear_btn = QPushButton("✕")
        self._clear_btn.setFixedSize(22, 22)
        self._clear_btn.setToolTip("Remove layer from viewer")
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self._on_clear)
        title_row.addWidget(self._title_label)
        title_row.addStretch()
        title_row.addWidget(self._clear_btn)
        layout.addLayout(title_row)

        # Info label
        self._info_label = QLabel("Drop experiment here\nor double-click in tree")
        self._info_label.setAlignment(Qt.AlignCenter)
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self._info_label, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_experiment(self, study_path: str, exp_id: str, proc_id: str = "1") -> None:
        """Load a 2dseq dataset and add/replace it in the napari viewer."""
        self._clear_layer()
        try:
            result = load_2dseq(Path(study_path), exp_id, proc_id)
        except Exception as exc:
            _log.debug("Failed to load exp %s proc %s: %s", exp_id, proc_id, exc)
            self._info_label.setText(f"Error loading:\n{exc}")
            return

        # Add layer to napari
        layer = self._viewer.add_image(
            result.data,
            name=result.layer_name,
            scale=result.scale,
            metadata=result.metadata,
        )
        self._layer_name = layer.name
        self._mark_filled(layer.name)

    def _mark_filled(self, name: str) -> None:
        self._title_label.setText(f"Slot {self._index + 1}: {name}")
        self._info_label.setText("")
        self._clear_btn.setVisible(True)
        self.setStyleSheet(_SLOT_STYLE_FILLED)

    def _mark_empty(self) -> None:
        self._title_label.setText(f"Slot {self._index + 1}")
        self._info_label.setText("Drop experiment here\nor double-click in tree")
        self._clear_btn.setVisible(False)
        self.setStyleSheet(_SLOT_STYLE_EMPTY)
        self._layer_name = None

    def _clear_layer(self) -> None:
        if self._layer_name and self._layer_name in self._viewer.layers:
            self._viewer.layers.remove(self._layer_name)
        self._layer_name = None

    def _on_clear(self) -> None:
        self._clear_layer()
        self._mark_empty()

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(BRUKER_EXP_MIME):
            event.acceptProposedAction()
            self.setStyleSheet(_SLOT_STYLE_HOVER)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        if self._layer_name:
            self.setStyleSheet(_SLOT_STYLE_FILLED)
        else:
            self.setStyleSheet(_SLOT_STYLE_EMPTY)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        raw = event.mimeData().data(BRUKER_EXP_MIME)
        try:
            payload = json.loads(bytes(raw).decode())
            self.load_experiment(
                payload["study_path"], payload["exp_id"], payload.get("proc_id", "1")
            )
            event.acceptProposedAction()
        except Exception as exc:
            _log.debug("Drop failed: %s", exc)
            self._info_label.setText(f"Drop error:\n{exc}")
            self.setStyleSheet(_SLOT_STYLE_EMPTY)


class TiledViewerWidget(QWidget):
    """
    2×2 grid of _SlotWidget instances docked into the napari window.

    External code can call `load_into_next_slot(study_path, exp_id, proc_id)`
    to fill the first empty slot (used when the user double-clicks a tree item).
    """

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._slots: list[_SlotWidget] = []
        self._build_ui()

    def _build_ui(self) -> None:
        self.setMinimumWidth(440)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        header = QLabel("Image Slots  (drop experiments from the tree)")
        header.setStyleSheet("font-weight: bold; font-size: 11px; color: #aaa;")
        outer.addWidget(header)

        grid = QGridLayout()
        grid.setSpacing(8)

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, (row, col) in enumerate(positions):
            slot = _SlotWidget(slot_index=idx, viewer=self._viewer, parent=self)
            self._slots.append(slot)
            grid.addWidget(slot, row, col)

        outer.addLayout(grid, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_into_next_slot(
        self, study_path: str, exp_id: str, proc_id: str = "1"
    ) -> None:
        """Load an experiment into the first empty slot (or slot 0 if all full)."""
        for slot in self._slots:
            if slot._layer_name is None:
                slot.load_experiment(study_path, exp_id, proc_id)
                return
        # All full: overwrite slot 0
        self._slots[0].load_experiment(study_path, exp_id, proc_id)

    def load_into_slot(
        self, index: int, study_path: str, exp_id: str, proc_id: str = "1"
    ) -> None:
        """Load an experiment into a specific slot by index (0–3)."""
        if 0 <= index < len(self._slots):
            self._slots[index].load_experiment(study_path, exp_id, proc_id)
