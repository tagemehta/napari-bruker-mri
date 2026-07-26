"""
MRI Viewer — napari-based Bruker study browser.

Run with:
    uv run python main.py

Layout:
    Left dock  : BrukerTreeWidget  (Patient → Study → Experiment tree)
    Right dock : TiledViewerWidget (2×2 image slots)

Double-click an experiment in the tree to load it into the next empty slot.
Drag an experiment from the tree onto a specific slot for precise placement.
"""

import napari

from bruker_browser.tile_viewer import TiledViewerWidget
from bruker_browser.tree_widget import BrukerTreeWidget


def main() -> None:
    viewer = napari.Viewer(title="Bruker MRI Viewer")

    # --- Left panel: study browser tree ---
    tree_widget = BrukerTreeWidget()
    viewer.window.add_dock_widget(
        tree_widget,
        name="Study Browser",
        area="left",
    )

    # --- Right panel: 2×2 image slots ---
    tile_widget = TiledViewerWidget(viewer=viewer)
    viewer.window.add_dock_widget(
        tile_widget,
        name="Image Slots",
        area="right",
    )

    # Wire double-click in the tree to loading into the next free slot
    tree_widget.experiment_activated.connect(tile_widget.load_into_next_slot)

    napari.run()


if __name__ == "__main__":
    main()
