"""
Inspect every experiment in a Bruker study folder and group them by physical
slice-package geometry, so you can tell which experiments are repeats of the
same disc location vs. scans of a genuinely different location (e.g. a
"Top" vs "Bottom" disc pair) — run with:
    uv run python inspect_scan_positions.py patients/real_data/RA_2022_08_16.fu2

Why this works: PVM_SPackArrSliceOffset / ReadOffset / Phase1Offset (in the
`method` file) define where the imaging slice package sits in physical
scanner space for that experiment. Two experiments with (near-)identical
offsets are the same disc location — one is a repeat/redo of the other.
Experiments with different offsets are genuinely different locations (e.g.
two different discs, imaged "Top" and "Bottom" as noted in
patients/rat-grouping.csv's "Position on Sagittal" column).

We read the raw JCAMP-DX `acqp`/`method` files directly with regex rather
than going through brukerapi's parameter API, matching the approach already
used in bruker_browser/loader.py's `_parse_fg_elem_comment_from_file` — the
brukerapi parameter API is unreliable across PV versions for parameters like
these, but the raw text files are simple and stable to parse.

This script does NOT know which offset sign corresponds to "Top" vs "Bottom"
— that's a physical convention (how the rat was laid in the coil) that only
you can confirm, e.g. by cross-referencing a session where rat-grouping.csv
already has "Position on Sagittal" filled in against that session's own scan
geometry. Once you know the sign convention for a given coil/orientation
setup, hardcode it in `_label_position` below and it'll auto-label future
studies with the same setup.

Writes nothing — prints a report to stdout.
"""

import re
import sys
from pathlib import Path

import numpy as np


def _read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1", errors="replace")


def _extract_scalar(text: str, key: str) -> float | None:
    """Extract a single-value JCAMP-DX parameter, e.g. ``##$PVM_SPackArrReadOffset=( 1 )\\n0.7``."""
    m = re.search(rf"##\$_?{key}=\([^\n]*\)\s*\n([^\n#]+)", text)
    if m:
        try:
            return float(m.group(1).split()[0])
        except ValueError:
            return None
    # Also handle scalar params with no array header, e.g. ##$Key=1.23
    m = re.search(rf"##\$_?{key}=([^\n]+)", text)
    if m:
        try:
            return float(m.group(1).strip())
        except ValueError:
            return None
    return None


def _extract_string(text: str, key: str) -> str | None:
    # Array header followed by a <...>-quoted value on the next line.
    m = re.search(rf"##\$_?{key}=\([^\n]*\)\s*\n<([^>]*)>", text)
    if m:
        return m.group(1).strip()
    # Quoted value on the same line.
    m = re.search(rf"##\$_?{key}=<([^>]*)>", text)
    if m:
        return m.group(1).strip()
    # Array header followed by a bare (unquoted) value on the next line,
    # e.g. ##$PVM_SPackArrSliceOrient=( 1 )\naxial
    m = re.search(rf"##\$_?{key}=\([^\n]*\)\s*\n([^\n#<]+)", text)
    if m:
        return m.group(1).strip()
    # Bare (unbracketed) value on the same line, e.g. ##$Method=FLASH
    m = re.search(rf"##\$_?{key}=([^\n(<][^\n]*)", text)
    if m:
        return m.group(1).strip()
    return None


def _extract_timestamp(text: str) -> str | None:
    m = re.search(r"\$\$ (\w+ \w+ +\d+ [\d:]+ \d{4})", text)
    return m.group(1) if m else None


def inspect_study(study_path: Path) -> list[dict]:
    records = []
    for exp_dir in sorted(
        (p for p in study_path.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        acqp_path = exp_dir / "acqp"
        method_path = exp_dir / "method"
        if not acqp_path.exists() or not method_path.exists():
            continue

        acqp_text = _read_text(acqp_path)
        method_text = _read_text(method_path)

        scan_name = _extract_string(acqp_text, "ACQ_scan_name")
        method = _extract_string(method_text, "Method")
        timestamp = _extract_timestamp(method_text)

        slice_off = _extract_scalar(method_text, "PVM_SPackArrSliceOffset")
        read_off = _extract_scalar(method_text, "PVM_SPackArrReadOffset")
        phase1_off = _extract_scalar(method_text, "PVM_SPackArrPhase1Offset")
        slice_orient = _extract_string(method_text, "PVM_SPackArrSliceOrient")

        records.append(
            {
                "exp_id": exp_dir.name,
                "scan_name": scan_name,
                "method": method,
                "timestamp": timestamp,
                "slice_offset": slice_off,
                "read_offset": read_off,
                "phase1_offset": phase1_off,
                "slice_orient": slice_orient,
            }
        )
    return records


SLICE_OFFSET_TOL_MM = 1.5
"""
Two experiments within this tolerance on PVM_SPackArrSliceOffset are treated
as the same physical disc location. This is the discriminator to use across
different acquisition types (Axial localizer / T2 Map / ADC) for the same
disc, because PVM_SPackArrReadOffset and PVM_SPackArrPhase1Offset depend on
each sequence's own read/phase axis convention and are NOT comparable across
sequence types — only the slice-select offset (perpendicular to the imaging
plane) stays consistent for the same physical slice regardless of sequence.
1.5mm was chosen empirically: exact repeats share slice_offset to ~1e-10mm,
but a "redo after re-shimming" of the same disc can drift ~0.5-0.6mm; two
genuinely different discs in this dataset are ~7-8mm apart.
"""


def _cluster_by_slice_offset(records: list[dict]) -> list[list[dict]]:
    """
    Cluster by slice_offset, but ONLY within the same slice orientation
    (axial/sagittal/coronal). Different orientations image different planes
    through the animal — their slice_offset values are along physically
    different axes and are not comparable to each other at all, so two scans
    in different orientations must never be merged into one "location" no
    matter how close their raw offset numbers happen to be.
    """
    clusters: list[list[dict]] = []
    for rec in sorted(records, key=lambda r: int(r["exp_id"])):
        placed = False
        for cluster in clusters:
            if (
                cluster[0]["slice_orient"] == rec["slice_orient"]
                and abs(cluster[0]["slice_offset"] - rec["slice_offset"]) <= SLICE_OFFSET_TOL_MM
            ):
                cluster.append(rec)
                placed = True
                break
        if not placed:
            clusters.append([rec])
    return clusters


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: uv run python {sys.argv[0]} <study_path>")
        sys.exit(1)

    study_path = Path(sys.argv[1])
    records = inspect_study(study_path)

    print(f"=== {study_path} — {len(records)} experiment(s) ===\n")
    header = f"{'exp':>4} {'method':<12} {'scan_name':<24} {'slice_off':>10} {'read_off':>9} {'phase1_off':>10}  {'timestamp':<24}"
    print(header)
    print("-" * len(header))
    for rec in records:
        print(
            f"{rec['exp_id']:>4} {str(rec['method']):<12} {str(rec['scan_name']):<24} "
            f"{rec['slice_offset'] if rec['slice_offset'] is not None else '':>10} "
            f"{rec['read_offset'] if rec['read_offset'] is not None else '':>9} "
            f"{rec['phase1_offset'] if rec['phase1_offset'] is not None else '':>10}  "
            f"{str(rec['timestamp']):<24}"
        )

    # Only cluster experiments that actually carry slice geometry (imaging scans,
    # not pilots/adjustments where these params are absent or meaningless), and
    # restrict to the orientation actually used for the quantitative maps (T2
    # Map / ADC) — whole-spine Sagittal/Coronal overview scans use a different
    # orientation and aren't per-disc acquisitions, so they're excluded here
    # rather than reported as spurious extra "disc locations".
    map_orientations = {
        r["slice_orient"]
        for r in records
        if r["scan_name"] and ("T2 Map" in r["scan_name"] or "ADC" in r["scan_name"] or "Diffusion" in r["scan_name"])
    }
    geo_records = [
        r
        for r in records
        if r["slice_offset"] is not None
        and r["scan_name"]
        and r["slice_orient"] in map_orientations
        and not r["scan_name"].startswith(("ADJ_", "1-TriPilot"))
    ]
    clusters = _cluster_by_slice_offset(geo_records)
    # Order clusters by when they first appear chronologically (lowest exp_id) —
    # in every session we've cross-checked against rat-grouping.csv's Position
    # column, the disc scanned FIRST is the one recorded as "Top"/"Disc 1", and
    # the one scanned second is "Bottom"/"Disc 2" (a workflow habit, not a scan
    # geometry fact, so treat the label as a strong hint, not certainty).
    clusters.sort(key=lambda c: min(int(r["exp_id"]) for r in c))

    print(f"\n=== Physical-location clusters ({len(clusters)} distinct disc location(s)) ===")
    print(
        "Experiments in the same cluster sit within "
        f"{SLICE_OFFSET_TOL_MM}mm slice-offset — i.e. they're repeats/redos of\n"
        "the same disc, not different discs. Clusters are ordered by which was\n"
        "scanned first; per every session we could cross-check against\n"
        "rat-grouping.csv's Position column, first-scanned = Top/Disc 1 and\n"
        "second-scanned = Bottom/Disc 2 — confirm this holds for your setup\n"
        "before trusting the label blindly.\n"
    )
    guess_labels = ["Disc 1 (likely 'Top')", "Disc 2 (likely 'Bottom')"]
    for i, recs in enumerate(clusters):
        label = guess_labels[i] if i < len(guess_labels) else f"Disc {i + 1} (extra location)"
        slice_off = recs[0]["slice_offset"]
        exp_ids = ", ".join(f"exp{r['exp_id']} ({r['scan_name']})" for r in recs)
        print(f"{label}  [slice_offset~{slice_off:.2f}]")
        print(f"    {exp_ids}")


if __name__ == "__main__":
    main()
