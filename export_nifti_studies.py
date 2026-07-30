"""
Bulk-export raw Bruker acquisitions to NIfTI for viewing in 3D Slicer — run with:
    uv run python export_nifti_studies.py

Scans every study folder in ``patients/real_data`` whose date stamp matches
``STUDY_MONTH_FILTER`` (default July + August 2022) and exports proc 1 (the
scanner's own reconstruction — see ``bruker_browser/models.py::ProcNode``)
of every experiment that has one. Output goes to ``nifti_export/{study
folder name}/exp{N}.nii.gz`` plus a same-named ``.json`` sidecar.

Why proc 1 only
---------------
Higher-numbered procs are ParaVision's own derived outputs (T2/ADC maps,
etc.) — already handled by the dedicated ``analysis/`` pipeline
(``analyze_disc.py``, ``inspect_roi_fit.py``). This script is for general
"look at the raw images in Slicer" viewing, so it sticks to the one proc
every imaging experiment has.

Multi-axis limitation
----------------------
NIfTI/Slicer only really support 3 spatial axes + a single 4th "volume"
axis. When an experiment has more than one Bruker frame-group axis (e.g.
diffusion direction *and* b-value), they're flattened into one combined
4th axis in row-major order — Slicer's volume slider then steps through
every combination rather than letting you pick axes independently. The
sidecar JSON's ``volume_labels`` lists what each 4th-axis index actually
is (e.g. ``"Dir1 | B415"``) so you're not scrolling blind.

Orientation caveat
-------------------
The affine written here is a simple diagonal voxel-size matrix (no patient
position/orientation) — accurate spacing, but not meant for cross-study
spatial registration. Fine for the "convert for easier viewing" use case
this script exists for.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

import nibabel as nib
import numpy as np

from bruker_browser.loader import load_2dseq
from bruker_browser.models import StudyNode

# ── change these to export a different slice of the data ───────────────────
REAL_DATA_DIR = Path("patients/real_data")
STUDY_MONTH_FILTER = ("_07_", "_08_")  # substrings matched against folder names
OUTPUT_DIR = Path("nifti_export")
PROC_ID = "1"
# ─────────────────────────────────────────────────────────────────────────────


def _iter_target_studies():
    for entry in sorted(REAL_DATA_DIR.iterdir()):
        if not entry.is_dir() or not (entry / "subject").exists():
            continue
        if any(tag in entry.name for tag in STUDY_MONTH_FILTER):
            yield entry


def _to_nifti_array(data: np.ndarray, scale: tuple[float, ...], frame_groups) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert a ``load_2dseq`` array into a NIfTI-ready one, ``(spatial...,
    [volume])``, plus a matching diagonal affine and per-volume labels.

    ``frame_groups[i].axis`` is the only reliable positional index into
    ``data``/``scale`` here — ``metadata["dim_type"]`` is *not* reordered to
    match the final transposed axes (it reflects the pre-transpose brukerapi
    order), and FG axes are not guaranteed to be contiguous or leading (e.g.
    a single-slice-package T2 scan comes back as
    ``(slice, echo, Y, X)`` — the FG_ECHO axis sits *between* the spatial
    ones). ``scale`` is positionally aligned with ``data`` though (verified:
    each axis's scale entry matches its own size/spacing), so spatial axes
    are found by elimination and pulled out by index rather than assumed to
    be a fixed contiguous block.

    Any frame-group axes present are flattened into a single trailing
    "volume" axis (see module docstring) — ``volume_labels[i]`` describes
    what combination of FG values that index holds; empty if there were no
    FG axes (plain spatial-only volume).
    """
    fgs = sorted(frame_groups, key=lambda fg: fg.axis)
    fg_axes = [fg.axis for fg in fgs]
    n_spatial = data.ndim - len(fg_axes)
    spatial_axes = [a for a in range(data.ndim) if a not in fg_axes]

    # Moves fg_axes to the trailing positions; the remaining (spatial) axes
    # shift left preserving their original relative order — so arr's first
    # n_spatial axes are exactly `spatial_axes`, in order.
    arr = np.moveaxis(data, fg_axes, range(n_spatial, data.ndim))
    volume_labels: list[str] = []
    if fg_axes:
        n_vol = int(np.prod([data.shape[a] for a in fg_axes]))
        arr = arr.reshape(arr.shape[:n_spatial] + (n_vol,))
        import itertools

        volume_labels = [" | ".join(combo) for combo in itertools.product(*(fg.labels for fg in fgs))]

    spatial_scale = [scale[a] for a in spatial_axes]
    if n_spatial == 2:
        arr = np.expand_dims(arr, axis=2)
        spatial_scale.append(1.0)

    affine = np.diag(spatial_scale + [1.0])
    return arr, affine, volume_labels


class NotAnImageError(Exception):
    """Raised for datasets with no spatial axes at all (e.g. 1-D spectroscopy)."""


def export_experiment(study_path: Path, exp_id: str, study_name: str, out_dir: Path) -> Path | None:
    """Load proc 1 of one experiment and write it out as ``{out_dir}/exp{exp_id}.nii.gz`` + sidecar."""
    result = load_2dseq(study_path, exp_id=exp_id, proc_id=PROC_ID, study_name=study_name, zoom_to=0)

    if result.data.ndim - len(result.frame_groups) == 0:
        raise NotAnImageError(
            f"exp {exp_id} has no spatial axes (dim_type={result.metadata['dim_type']}) — "
            "not an image (e.g. 1-D spectroscopy), skipping."
        )

    arr, affine, volume_labels = _to_nifti_array(result.data, result.scale, result.frame_groups)
    img = nib.Nifti1Image(arr, affine)

    out_dir.mkdir(parents=True, exist_ok=True)
    nii_path = out_dir / f"exp{exp_id}.nii.gz"
    nib.save(img, nii_path)

    sidecar = {
        "study_path": str(study_path),
        "study_name": study_name,
        "exp_id": exp_id,
        "proc_id": PROC_ID,
        "raw_shape": list(result.metadata["raw_shape"]),
        "dim_type": result.metadata["dim_type"],
        "scale_mm": list(result.scale),
        "frame_groups": [
            {"name": fg.name, "size": fg.size, "labels": fg.labels} for fg in result.frame_groups
        ],
        "volume_labels": volume_labels,
    }
    nii_path.with_suffix("").with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return nii_path


def main() -> None:
    n_ok, n_skipped, n_failed = 0, 0, 0
    for study_path in _iter_target_studies():
        study = StudyNode(path=study_path)
        out_dir = OUTPUT_DIR / study_path.name
        _log.info("Study %s (%s, %s)", study_path.name, study.study_name, study.date)

        for exp in study.experiments:
            proc_ids = [p.proc_id for p in exp.proc_nodes()]
            if PROC_ID not in proc_ids:
                n_skipped += 1
                continue
            try:
                nii_path = export_experiment(study_path, exp.exp_id, study.study_name, out_dir)
                _log.info("  exp %-3s %-10s -> %s", exp.exp_id, exp.sequence, nii_path)
                n_ok += 1
            except NotAnImageError as exc:
                _log.info("  exp %s skipped: %s", exp.exp_id, exc)
                n_skipped += 1
            except Exception as exc:  # noqa: BLE001 — one bad experiment shouldn't abort the sweep
                _log.warning("  exp %s failed: %s", exp.exp_id, exc)
                n_failed += 1

    _log.info("Done: %d exported, %d skipped (no proc %s), %d failed", n_ok, n_skipped, PROC_ID, n_failed)


if __name__ == "__main__":
    main()
