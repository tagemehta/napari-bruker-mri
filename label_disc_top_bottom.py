"""
Generate Top/Bottom disc labels for every rat with saved ROI stats — run with:
    uv run python label_disc_top_bottom.py

Method, in two parts:

1. **Clustering exp_ids into physical disc locations**: uses
   PVM_SPackArrSliceOffset + PVM_SPackArrSliceOrient parsed straight out of
   each experiment's raw `method` file (the same fields and tolerance as
   inspect_scan_positions.py — see that file's docstring for why this, and
   not the brukerapi affine's full 3-D world center, is the right
   discriminator: a disc's T2 map and ADC map are prescribed with the same
   through-plane slice offset but often a different in-plane FOV/offset, so
   comparing full 3-D affine centers across sequence types is noisy —
   verified against this dataset: T2 exp6 vs. ADC exp7 on the same disc
   share slice_offset to machine precision, 6.8606mm, despite the affine's
   full 3-D centers differing by several mm in-plane).

2. **Labeling which cluster is Top vs. Bottom**: like
   plot_disc_positions_on_sagittal.py, reads each cluster's brukerapi affine
   to get its physical world Z (Bruker's world Z axis is the magnet bore's
   long axis, i.e. the rat's head-to-tail axis, and stays consistent across
   sequences/orientations/sessions given a fixed coil/positioning protocol).
   This was calibrated against the one disc pair whose Top/Bottom was
   already confirmed by eye against the sagittal localizer
   (RA_2022_08_16.fu2 / rat 693427, via plot_disc_positions_on_sagittal.py):
       exp8  (Top)    world Z = +5.38 mm
       exp11 (Bottom) world Z = -2.28 mm
   So here: whichever cluster has the LARGER average world Z is labeled
   Top, the smaller is Bottom.

Caveat: step 2 assumes every rat was scanned head-first/tail-first the same
way as 693427. If a session ever used different positioning, that rat's
pair would be mislabeled here — spot-check against
plot_disc_positions_on_sagittal.py if a label looks suspicious.

For each rat (study_name) with rows in roi_stats.csv:
  1. Resolve which raw study folder it came from, via
     patients/real_data/INDEX.md's folder -> rat-ID labels, disambiguated by
     checking which candidate folder actually contains every exp_id
     referenced for that study_name (handles rats with more than one
     session folder, e.g. a pilot-only stub plus the real session, or a
     rescan).
  2. Cluster exp_ids by slice_offset + slice_orient (same physical disc).
  3. Label: exactly 2 clusters -> Top/Bottom by average world Z; 1 cluster
     -> "only one disc analyzed"; >2 clusters -> flagged ambiguous for
     manual review; unresolved raw folder -> flagged unresolved.

Writes parameter_maps/disc_top_bottom_labels.csv (one row per analyzed
exp_id) and prints a summary report.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from inspect_scan_positions import SLICE_OFFSET_TOL_MM, _extract_scalar, _extract_string, _read_text
from plot_disc_positions_on_sagittal import _world_center

ROI_STATS_PATH = Path("parameter_maps/roi_stats.csv")
INDEX_PATH = Path("patients/real_data/INDEX.md")
RAW_DATA_DIR = Path("patients/real_data")
OUT_PATH = Path("parameter_maps/disc_top_bottom_labels.csv")


def _extract_rat_number(study_name: str) -> str:
    m = re.search(r"\d{5,}", str(study_name))
    if not m:
        raise ValueError(f"No rat number found in study_name: {study_name!r}")
    return m.group()


# Known rat-ID typos in INDEX.md, traced back to the raw Bruker `subject`
# file's manually-typed SUBJECT_study_name field (an operator data-entry
# mistake at scan time, e.g. RA_2022_08_17.fv2/subject literally contains
# "<639428>") — not something INDEX.md's generator introduced. Looked up as
# a fallback when the exact rat number has no INDEX.md match.
_KNOWN_INDEX_TYPOS = {"693428": "639428", "693429": "639429"}


def _parse_index(index_path: Path) -> dict[str, list[tuple[str, str]]]:
    """rat_number -> [(folder, full_label), ...], in INDEX.md order."""
    mapping: dict[str, list[tuple[str, str]]] = {}
    for line in index_path.read_text().splitlines():
        m = re.match(r"^## (\S+) — (.+)$", line)
        if not m:
            continue
        folder, label = m.group(1), m.group(2)
        rat_m = re.search(r"\d{5,}", label)
        if not rat_m:
            continue
        mapping.setdefault(rat_m.group(), []).append((folder, label))
    return mapping


def _resolve_folder(study_name: str, exp_ids: set[str], candidates: list[tuple[str, str]]) -> str | None:
    """Pick whichever candidate folder actually has all the given exp_ids on
    disk. If more than one does (e.g. a rat with a pilot-only stub folder
    plus the real session, or an original + rescan session), break the tie
    using any extra descriptive word in study_name (e.g. "rescan",
    "second_try") against each candidate's INDEX.md label."""
    matches = [
        (folder, label)
        for folder, label in candidates
        if all((RAW_DATA_DIR / folder / e / "acqp").exists() for e in exp_ids)
    ]
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        descriptor = re.sub(r"\d+", "", study_name).strip().lower()
        if descriptor:
            desc_matches = [
                folder for folder, label in matches
                if descriptor in re.sub(r"\d+", "", label).strip().lower()
            ]
            if len(desc_matches) == 1:
                return desc_matches[0]
    return None


def _slice_geometry(folder: str, exp_id: str) -> tuple[float | None, str | None]:
    method_path = RAW_DATA_DIR / folder / exp_id / "method"
    if not method_path.exists():
        return None, None
    text = _read_text(method_path)
    return (
        _extract_scalar(text, "PVM_SPackArrSliceOffset"),
        _extract_string(text, "PVM_SPackArrSliceOrient"),
    )


def _cluster_by_geometry(exp_ids: list[str], folder: str, tol: float) -> tuple[list[list[str]], list[str]]:
    """Group exp_ids into physical disc locations via slice_offset + slice_orient
    (same fields/tolerance as inspect_scan_positions.py — see that module's
    docstring for why this beats comparing full affine world centers across
    sequence types). Returns (clusters, notes) where notes lists any exp_id
    dropped for missing geometry."""
    geo = []
    notes = []
    for exp_id in exp_ids:
        offset, orient = _slice_geometry(folder, exp_id)
        if offset is None or orient is None:
            notes.append(f"exp{exp_id}: missing slice geometry — dropped")
            continue
        geo.append({"exp_id": exp_id, "slice_offset": offset, "slice_orient": orient})

    clusters: list[list[str]] = []
    for rec in sorted(geo, key=lambda r: int(r["exp_id"])):
        for cluster_ids in clusters:
            first = next(g for g in geo if g["exp_id"] == cluster_ids[0])
            if (
                first["slice_orient"] == rec["slice_orient"]
                and abs(first["slice_offset"] - rec["slice_offset"]) <= tol
            ):
                cluster_ids.append(rec["exp_id"])
                break
        else:
            clusters.append([rec["exp_id"]])
    return clusters, notes


def main() -> None:
    from brukerapi.folders import Study

    stats = pd.read_csv(ROI_STATS_PATH).dropna(subset=["study_name"])
    stats["study_name"] = stats["study_name"].astype(str)
    stats["exp_id"] = stats["exp_id"].astype(int).astype(str)
    stats["rat_number"] = stats["study_name"].map(_extract_rat_number)

    index_map = _parse_index(INDEX_PATH)

    rows: list[dict] = []
    report: list[str] = []

    for study_name, grp in stats.groupby("study_name"):
        rat_number = grp["rat_number"].iloc[0]
        exp_ids = sorted(grp["exp_id"].unique(), key=int)
        candidates = index_map.get(rat_number) or index_map.get(_KNOWN_INDEX_TYPOS.get(rat_number, ""), [])
        folder = _resolve_folder(study_name, set(exp_ids), candidates)

        if folder is None:
            report.append(
                f"{study_name}: could not resolve a unique raw study folder "
                f"(candidates: {candidates or 'none'}) — skipped"
            )
            rows += [
                {"study_name": study_name, "exp_id": e, "label": "unresolved", "note": "no unique raw folder match"}
                for e in exp_ids
            ]
            continue

        clusters, geo_notes = _cluster_by_geometry(exp_ids, folder, SLICE_OFFSET_TOL_MM)
        for note in geo_notes:
            report.append(f"{study_name} {note}")

        # world Z (for Top/Bottom ordering only — see module docstring) per exp_id
        world_z: dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = Study(str(RAW_DATA_DIR / folder))
            for cluster_ids in clusters:
                for exp_id in cluster_ids:
                    try:
                        d = study.get_dataset(exp_id=exp_id, proc_id="1")
                        d.load()
                        world_z[exp_id] = float(_world_center(d)[2])
                    except Exception as e:  # noqa: BLE001
                        report.append(f"{study_name} exp{exp_id} ({folder}): failed to load ({e}) — skipped")

        clusters = [[e for e in c if e in world_z] for c in clusters]
        clusters = [c for c in clusters if c]

        if len(clusters) == 1:
            exp_list = clusters[0]
            rows += [
                {"study_name": study_name, "exp_id": e, "label": "only", "note": "only one disc location analyzed"}
                for e in exp_list
            ]
            report.append(f"{study_name} ({folder}): only one disc analyzed — exp {exp_list}")

        elif len(clusters) == 2:
            clusters.sort(key=lambda c: np.mean([world_z[e] for e in c]), reverse=True)
            for label, cluster_ids in zip(["Top", "Bottom"], clusters):
                for exp_id in cluster_ids:
                    rows.append({
                        "study_name": study_name, "exp_id": exp_id, "label": label,
                        "note": f"world_z={world_z[exp_id]:.2f}mm",
                    })
            report.append(f"{study_name} ({folder}): Top=exp{clusters[0]}  Bottom=exp{clusters[1]}")

        else:
            for cluster_ids in clusters:
                for exp_id in cluster_ids:
                    rows.append({
                        "study_name": study_name, "exp_id": exp_id, "label": "ambiguous",
                        "note": f"{len(clusters)} distinct locations found, world_z={world_z[exp_id]:.2f}mm",
                    })
            report.append(
                f"{study_name} ({folder}): {len(clusters)} distinct disc locations found "
                "— ambiguous, needs manual review"
            )

    out = pd.DataFrame.from_records(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} row(s) to {OUT_PATH}\n")
    print("\n".join(report))

    n_pairs = sum(1 for l in report if "Top=exp" in l)
    n_only = sum(1 for l in report if "only one disc" in l)
    n_ambiguous = sum(1 for l in report if "ambiguous" in l)
    n_unresolved = sum(1 for l in report if "skipped" in l)
    print(
        f"\n=== Summary: {n_pairs} rat(s) with a Top/Bottom pair, "
        f"{n_only} with only one disc analyzed, {n_ambiguous} ambiguous, "
        f"{n_unresolved} unresolved ==="
    )


if __name__ == "__main__":
    main()
