"""
BrukerTreeWidget — four-level collapsible tree:
    Patient  →  Study  →  Experiment  →  Processing (proc)

Users can:
  - Click "Open Patient Folder" to choose the root patients directory.
  - Double-click a proc row to load it into the next free tile slot.
  - Drag a proc row onto a specific tile slot.
"""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import QMimeData, Qt, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .models import ExpNode, ProcNode, StudyNode, scan_patient_list

# MIME type used when dragging a proc row
BRUKER_EXP_MIME = "application/x-bruker-experiment"

# QTreeWidgetItem column indices
COL_ID = 0  # exp # / proc #
COL_SEQ = 1  # sequence (exp level) or protocol label (proc level)
COL_DESC = 2  # scan description (exp) or series comment (proc)

# Role storing the ProcNode on leaf items
_PROC_ROLE = Qt.UserRole


class BrukerTreeWidget(QWidget):
    """
    Left-panel dock widget: Patient → Study → Experiment → Proc tree.

    Signals
    -------
    experiment_activated(study_path: str, exp_id: str, proc_id: str)
        Emitted when the user double-clicks a proc item.
    """

    experiment_activated = Signal(str, str, str)  # study_path, exp_id, proc_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._open_btn = QPushButton("Open Patient Folder…")
        self._open_btn.clicked.connect(self._on_open_folder)
        toolbar.addWidget(self._open_btn)
        layout.addLayout(toolbar)

        self._root_label = QLabel("")
        self._root_label.setWordWrap(True)
        self._root_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self._root_label)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["#  /  Name", "Sequence / Protocol", "Description"])
        self._tree.header().setDefaultSectionSize(130)
        self._tree.setDragEnabled(True)
        self._tree.setDragDropMode(QTreeWidget.DragOnly)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Custom MIME data for dragging
        self._tree.mimeData = self._mime_data_for_items  # type: ignore[method-assign]

        layout.addWidget(self._tree)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Patient List Folder", str(Path.home())
        )
        if folder:
            self._load_root(folder)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        proc_node = item.data(COL_ID, _PROC_ROLE)
        if not isinstance(proc_node, ProcNode):
            return
        self.experiment_activated.emit(
            str(proc_node.exp.study_path),
            proc_node.exp.exp_id,
            proc_node.proc_id,
        )

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _load_root(self, root: str) -> None:
        self._root_label.setText(root)
        self._tree.clear()

        patients = scan_patient_list(root)
        if not patients:
            self._tree.addTopLevelItem(QTreeWidgetItem(["(No Bruker studies found)"]))
            return

        bold_font = QFont()
        bold_font.setBold(True)
        italic_font = QFont()
        italic_font.setItalic(True)

        for patient in patients:
            p_item = QTreeWidgetItem([patient.display_name])
            p_item.setFont(COL_ID, bold_font)
            p_item.setFlags(p_item.flags() & ~Qt.ItemIsDragEnabled)

            for study in patient.studies:
                s_item = QTreeWidgetItem([study.display_name])
                s_item.setFont(COL_ID, bold_font)
                s_item.setFlags(s_item.flags() & ~Qt.ItemIsDragEnabled)

                for exp in study.experiments:
                    e_item = _make_exp_item(exp, study, italic_font)

                    for proc in exp.proc_nodes():
                        p_row = _make_proc_item(proc)
                        e_item.addChild(p_row)

                    s_item.addChild(e_item)
                    # Auto-expand experiments that have more than one proc
                    # so the user immediately sees the proc options.
                    if exp.proc_nodes().__len__() > 1:
                        e_item.setExpanded(True)

                p_item.addChild(s_item)

            self._tree.addTopLevelItem(p_item)
            p_item.setExpanded(True)
            for i in range(p_item.childCount()):
                p_item.child(i).setExpanded(True)

        self._tree.resizeColumnToContents(COL_ID)
        self._tree.resizeColumnToContents(COL_SEQ)

    # ------------------------------------------------------------------
    # Drag support
    # ------------------------------------------------------------------

    def _mime_data_for_items(self, items: list[QTreeWidgetItem]) -> QMimeData:
        mime = QMimeData()
        for item in items:
            proc_node = item.data(COL_ID, _PROC_ROLE)
            if isinstance(proc_node, ProcNode):
                payload = {
                    "study_path": str(proc_node.exp.study_path),
                    "exp_id": proc_node.exp.exp_id,
                    "proc_id": proc_node.proc_id,
                }
                mime.setData(BRUKER_EXP_MIME, json.dumps(payload).encode())
                break
        return mime


# ------------------------------------------------------------------
# Item factory helpers
# ------------------------------------------------------------------


def _make_exp_item(
    exp: ExpNode, study: StudyNode, italic_font: QFont
) -> QTreeWidgetItem:
    """Create the experiment-level tree row (not directly loadable)."""
    item = QTreeWidgetItem([exp.exp_id, exp.sequence, exp.description])
    item.setFont(COL_ID, italic_font)
    item.setFlags(item.flags() & ~Qt.ItemIsDragEnabled)
    item.setToolTip(
        COL_ID,
        f"Study: {study.path}\nExp: {exp.exp_id}\n"
        f"Double-click a proc below to load it.",
    )
    return item


def _make_proc_item(proc: ProcNode) -> QTreeWidgetItem:
    """Create a proc-level leaf row (draggable, double-clickable)."""
    item = QTreeWidgetItem([
        f"  proc {proc.proc_id}",
        proc.protocol,
        proc.comment,
    ])
    item.setData(COL_ID, _PROC_ROLE, proc)
    item.setToolTip(
        COL_ID,
        f"Exp {proc.exp.exp_id} / proc {proc.proc_id}\n"
        f"Protocol: {proc.protocol}\n"
        f"Comment: {proc.comment}",
    )
    return item
