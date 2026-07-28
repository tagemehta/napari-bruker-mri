"""
MRI Viewer — napari-based Bruker study browser.

Run with:
    uv run python main.py

Layout:
    Left dock  : BrukerTreeWidget  (Patient → Study → Experiment → Proc tree)
    Centre     : napari viewer     (images loaded here as layers)
    Right dock : FrameGroupWidget  (dropdowns to switch echoes / T2 params / etc.)

Double-click a proc row in the tree to load it as a napari layer.
Loading the same proc again replaces the existing layer in-place.

Contrast is handled by napari's built-in controls.  Auto-contrast is enabled
per layer so limits update as you scroll through slices.
"""

import napari

from bruker_browser.tile_viewer import load_into_viewer
from bruker_browser.tree_widget import BrukerTreeWidget
from windowing import FrameGroupWidget


def main() -> None:
    viewer = napari.Viewer(title="Bruker MRI Viewer")

    tree_widget = BrukerTreeWidget()
    viewer.window.add_dock_widget(tree_widget, name="Study Browser", area="left")
    viewer.window.add_dock_widget(
        FrameGroupWidget(viewer), name="Frame Groups", area="right"
    )

    # Wire double-click directly to napari viewer
    tree_widget.experiment_activated.connect(
        lambda study_path, exp_id, proc_id, study_name: load_into_viewer(
            viewer, study_path, exp_id, proc_id, study_name=study_name
        )
    )

    napari.run()


if __name__ == "__main__":
    main()
