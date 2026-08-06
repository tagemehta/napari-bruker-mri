"""
Reproduce ParaVision's own "load a scan, see where it sits on the sagittal
localizer" workflow: project each disc-imaging experiment's slice-stack
center into the sagittal localizer's voxel space (via each dataset's affine)
and mark it on the sagittal image — edit the constants below and run with:
    uv run python plot_disc_positions_on_sagittal.py

This does NOT guess a Top/Bottom sign convention — it draws the marker on
the actual sagittal anatomy so you can read off Top vs Bottom by eye, same
as you would in ParaVision.

Geometry: each dataset's affine (via brukerapi) maps (spatial_i, spatial_j,
slice_k, 1) voxel indices to (x, y, z, 1) world mm. For each requested
exp_id, the voxel-space center of its spatial+slice dims is projected to
world space, then mapped into the sagittal dataset's voxel space by
inverting the sagittal affine — giving a pixel position (row, col) on a
specific sagittal slice index.

Shows the annotated sagittal image in a matplotlib window.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Edit these constants ---
STUDY_PATH = "patients/real_data/RA_2022_08_16.fu2"  # study folder
SAGITTAL_EXP_ID = "5"  # sagittal localizer experiment
DISC_EXP_IDS = ["11", "18"]  # disc-imaging experiments to mark


def _spatial_and_slice_axes(dataset) -> list[int]:
    """Return axis indices for (spatial, spatial, FG_SLICE) in that order — the
    3 axes an affine's voxel->world mapping actually covers. FG_ECHO / other
    frame-group axes (e.g. b-value, repetition) don't move physical position
    and are excluded."""
    dim_type = [str(t) for t in dataset.dim_type]
    spatial = [i for i, t in enumerate(dim_type) if t == "spatial"]
    slice_axis = [i for i, t in enumerate(dim_type) if "SLICE" in t]
    return spatial + slice_axis


def _world_center(dataset) -> np.ndarray:
    axes = _spatial_and_slice_axes(dataset)
    shape = dataset.data.shape
    center_vox = [(shape[i] - 1) / 2 for i in axes]
    world = dataset.affine @ np.array(center_vox + [1.0])
    return world[:3]


def main() -> None:
    from brukerapi.folders import Study

    study_path, sag_exp_id, disc_exp_ids = STUDY_PATH, SAGITTAL_EXP_ID, DISC_EXP_IDS
    study = Study(study_path)

    sag = study.get_dataset(exp_id=sag_exp_id, proc_id="1")
    sag.load()
    sag_axes = _spatial_and_slice_axes(sag)
    if len(sag_axes) != 3:
        raise RuntimeError(
            f"Expected 3 spatial+slice axes for sagittal exp {sag_exp_id}, got {sag_axes}"
        )
    sag_affine_inv = np.linalg.inv(sag.affine)

    markers = []  # (exp_id, scan_name, row, col, slice_idx)
    for exp_id in disc_exp_ids:
        d = study.get_dataset(exp_id=exp_id, proc_id="1")
        d.load()
        world = _world_center(d)
        vox = sag_affine_inv @ np.array([*world, 1.0])
        row, col, slice_idx = vox[:3]
        markers.append((exp_id, round(row), round(col), round(slice_idx)))

    # Pick the sagittal slice closest to the median of the requested markers'
    # slice index, so all markers land on (or very near) the displayed slice.
    slice_idx = int(round(np.median([m[3] for m in markers])))
    slice_idx = max(0, min(sag.data.shape[sag_axes[2]] - 1, slice_idx))

    sag_img = np.take(sag.data, slice_idx, axis=sag_axes[2])
    # sag_img's axes are still in original (spatial0, spatial1) order matching
    # sag_axes[0], sag_axes[1] — no further reordering needed since we indexed
    # out the slice axis directly.

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sag_img.T, cmap="gray", origin="upper")
    colors = plt.cm.tab10.colors
    for i, (exp_id, row, col, m_slice) in enumerate(markers):
        marker = "o" if m_slice == slice_idx else "x"
        ax.plot(
            row,
            col,
            marker,
            color=colors[i % len(colors)],
            markersize=14,
            markeredgewidth=3,
        )
        label = f"exp{exp_id}" + (
            ""
            if m_slice == slice_idx
            else f" (slice {m_slice}, off by {m_slice - slice_idx})"
        )
        ax.annotate(
            label,
            (row, col),
            textcoords="offset points",
            xytext=(10, 10),
            color=colors[i % len(colors)],
            fontsize=12,
            fontweight="bold",
        )
    ax.set_title(f"{study_path}  —  sagittal exp{sag_exp_id}, slice {slice_idx}")
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    for exp_id, row, col, m_slice in markers:
        print(f"  exp{exp_id}: sagittal voxel (row={row}, col={col}, slice={m_slice})")


if __name__ == "__main__":
    main()
