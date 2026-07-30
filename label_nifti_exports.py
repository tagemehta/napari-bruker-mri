"""
Print (and save) a human-readable index of what each exported NIfTI file
actually is — run with:
    uv run python label_nifti_exports.py

``export_nifti_studies.py`` names files ``exp{N}.nii.gz``, matching the
Bruker experiment folder number — it doesn't carry the sequence/protocol
label (e.g. "MSME.ppg — 5-T2 Map Fat Sat") the way the ``BrukerTreeWidget``
file browser shows it. This walks ``nifti_export/`` and, for each exported
file, looks that label back up from the source study (same
``bruker_browser/models.py`` classes the tree widget itself uses — see
``ExpNode.sequence``/``.description``, ``ProcNode.protocol``/``.comment``)
so you can tell T2 from ADC from a plain scout scan without opening each
file in Slicer.

Output: ``nifti_export/INDEX.md``, grouped by study, one line per exported
file.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

from bruker_browser.models import StudyNode
from export_nifti_studies import PROC_ID, REAL_DATA_DIR, OUTPUT_DIR

INDEX_PATH = OUTPUT_DIR / "INDEX.md"


def build_index() -> str:
    lines = ["# NIfTI export index", ""]

    for study_dir in sorted(OUTPUT_DIR.iterdir()):
        if not study_dir.is_dir():
            continue
        study_path = REAL_DATA_DIR / study_dir.name
        if not study_path.exists():
            _log.warning("Source study for %s not found at %s — skipping", study_dir.name, study_path)
            continue

        study = StudyNode(path=study_path)
        lines.append(f"## {study_dir.name} — {study.study_name}")
        lines.append(f"Subject: {study.subject_name or '(unknown)'}  |  Date: {study.date or '(unknown)'}")
        lines.append("")
        lines.append("| file | exp | sequence | description | proc protocol/comment |")
        lines.append("|---|---|---|---|---|")

        exp_by_id = {e.exp_id: e for e in study.experiments}
        nii_files = sorted(
            study_dir.glob("exp*.nii.gz"),
            key=lambda p: int(p.stem.removesuffix(".nii").removeprefix("exp")),
        )
        for nii_path in nii_files:
            exp_id = nii_path.stem.removesuffix(".nii").removeprefix("exp")
            exp = exp_by_id.get(exp_id)
            if exp is None:
                lines.append(f"| {nii_path.name} | {exp_id} | (exp folder not found) | | |")
                continue
            proc = next((p for p in exp.proc_nodes() if p.proc_id == PROC_ID), None)
            proc_label = proc.description if proc else "(no proc)"
            lines.append(
                f"| {nii_path.name} | {exp_id} | {exp.sequence or '—'} | {exp.description or '—'} | {proc_label or '—'} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    text = build_index()
    print(text)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(text)
    _log.info("Wrote %s", INDEX_PATH)


if __name__ == "__main__":
    main()
