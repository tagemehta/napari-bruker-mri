"""
Print (and save) a human-readable index of the raw Bruker studies — run with:
    uv run python index_bruker_studies.py

Same idea as ``label_nifti_exports.py``, but reads straight from
``patients/real_data`` — every experiment/proc the file browser would show,
regardless of whether it's been exported to NIfTI (or even has a proc 1 at
all, e.g. AdjRefScan/localizer-only experiments). Use this to browse what's
actually in the raw data; use ``label_nifti_exports.py`` to label files you
already exported.

Output: ``patients/real_data/INDEX.md``, grouped by study, one line per
experiment/proc.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

from bruker_browser.models import StudyNode
from export_nifti_studies import REAL_DATA_DIR, STUDY_MONTH_FILTER, _iter_target_studies

INDEX_PATH = REAL_DATA_DIR / "INDEX.md"


def build_index() -> str:
    lines = ["# Bruker study index", f"(studies matching {STUDY_MONTH_FILTER})", ""]

    for study_path in _iter_target_studies():
        study = StudyNode(path=study_path)
        lines.append(f"## {study_path.name} — {study.study_name}")
        lines.append(f"Subject: {study.subject_name or '(unknown)'}  |  Date: {study.date or '(unknown)'}")
        lines.append("")
        lines.append("| exp | sequence | description | procs |")
        lines.append("|---|---|---|---|")

        for exp in study.experiments:
            procs = exp.proc_nodes()
            if procs:
                proc_col = "; ".join(f"{p.proc_id}={p.description or '—'}" for p in procs)
            else:
                proc_col = "(none)"
            lines.append(f"| {exp.exp_id} | {exp.sequence or '—'} | {exp.description or '—'} | {proc_col} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    text = build_index()
    print(text)
    INDEX_PATH.write_text(text)
    _log.info("Wrote %s", INDEX_PATH)


if __name__ == "__main__":
    main()
