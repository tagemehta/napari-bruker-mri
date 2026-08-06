"""
Apply per-pixel signal-quality filtering to every ROI, verify it looks
reasonable (before/after tables), then compare irradiated vs control trends
across every available timepoint — run with:
    uv run python snr_filter_and_compare.py

Filtering criterion differs by map kind, because the two fits use different
noise models (see analysis/fitting.py):
- T2: fit includes a free noise-floor term ``bias`` (``A`` in
  ``S = A + C*exp(-TE/T2)``), so ``s0 / |bias|`` is a real per-pixel SNR
  proxy — pixels near the noise floor get a large bias relative to s0.
  Keep pixels with SNR >= T2_SNR_MIN.
- ADC: fit fixes bias at 0 (matches PV's own macro — see fitting.py "ADC
  model" section), so ``s0/bias`` is undefined (bias is uniformly 0,
  confirmed against every saved ADC diagnostics file in parameter_maps/).
  Use R² instead — a poor mono-exponential fit is the available proxy for
  an untrustworthy ADC pixel. Keep pixels with R² >= ADC_R2_MIN.

Cohort: every ROI in parameter_maps/roi_stats_with_grouping.csv. Per
patients/rat-grouping.csv, groups map to timepoints as follows:
  0d      Control                          (baseline, pre-irradiation)
  7d      Group 5, Group 6   (irradiated)
  14d     Group 3, Group 4   (irradiated)
  28d     Group 1, Group 2, extra (irradiated)
  28d-C   Control                          (time-matched sham for 28d)
"extra" (690735-690738) is a 28d irradiated cohort not folded into Group 1/2
proper, but is still irradiated tissue at the same timepoint, so it's pooled
in with the other 28d irradiated rats here. Every irradiated timepoint is
compared against BOTH controls available (0d baseline and 28d-C time-matched)
so trends over time, and the effect of the specific control chosen, are both
visible.

Writes parameter_maps/roi_stats_snr_filtered.csv (per-ROI before/after) and
prints a full irradiated-vs-control-by-timepoint trend summary.
"""

import numpy as np
import pandas as pd
from scipy import stats

from analysis.maps import load_map
from analysis.rois import compute_roi_stats, load_rois, rasterize_roi

GROUPED_STATS_PATH = "parameter_maps/roi_stats_with_grouping.csv"
OUT_PATH = "parameter_maps/roi_stats_snr_filtered.csv"

T2_SNR_MIN = 2.0
ADC_R2_MIN = 0.7

_IRRADIATED_GROUPS = {"Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6", "extra"}
_CONTROL_LABEL = "Control"

# Per-user convention (ROI names as drawn in roi_annotator_widget.py):
# ROI 1 = NP whole-region polygon, ROI 2 = AF whole-region polygon,
# ROI 3/4 = small NP sampling squares, ROI 5/6/7 = small AF sampling squares.
_ROI_TISSUE = {
    "ROI 1": ("NP", "whole"),
    "ROI 2": ("AF", "whole"),
    "ROI 3": ("NP", "sample"),
    "ROI 4": ("NP", "sample"),
    "ROI 5": ("AF", "sample"),
    "ROI 6": ("AF", "sample"),
    "ROI 7": ("AF", "sample"),
}


def _cohort(group: str) -> str:
    if group in _IRRADIATED_GROUPS:
        return "irradiated"
    if group == _CONTROL_LABEL:
        return "control"
    return "other"


def _quality_mask(kind: str, diagnostics: dict | None, shape: tuple) -> np.ndarray:
    if diagnostics is None:
        return np.zeros(shape, dtype=bool)
    if kind == "t2":
        snr = diagnostics["s0"] / np.abs(diagnostics["bias"])
        return np.isfinite(snr) & (snr >= T2_SNR_MIN)
    r2 = diagnostics["r2"]
    return np.isfinite(r2) & (r2 >= ADC_R2_MIN)


def main() -> None:
    grouped = pd.read_csv(GROUPED_STATS_PATH).dropna(subset=["study_name"])
    cohort_rows = grouped.copy()
    cohort_rows["cohort"] = cohort_rows["group"].map(_cohort)
    cohort_rows = cohort_rows[cohort_rows["cohort"] != "other"]

    map_cache: dict[tuple, object] = {}
    roi_cache: dict[tuple, list] = {}

    records = []
    for _, row in cohort_rows.iterrows():
        study_name = str(row["study_name"])
        exp_id = str(int(row["exp_id"]))
        kind = row["map_kind"]
        roi_name = row["roi_name"]

        map_key = (study_name, exp_id, kind)
        if map_key not in map_cache:
            map_cache[map_key] = load_map(study_name, exp_id, kind)
        map_result = map_cache[map_key]

        roi_key = (study_name, exp_id, kind)
        if roi_key not in roi_cache:
            roi_cache[roi_key] = load_rois(study_name, exp_id, kind)
        roi = next((r for r in roi_cache[roi_key] if r.name == roi_name), None)
        if roi is None:
            continue

        roi_mask = rasterize_roi(roi, map_result.data.shape)
        quality_mask = _quality_mask(kind, map_result.diagnostics, map_result.data.shape)
        filtered_mask = roi_mask & quality_mask

        before = compute_roi_stats(roi_mask, map_result.data)
        after = compute_roi_stats(filtered_mask, map_result.data)
        pct_kept = (after["n_pixels"] / before["n_pixels"]) if before["n_pixels"] else np.nan

        tissue, region_type = _ROI_TISSUE.get(roi_name, ("unknown", "unknown"))
        records.append({
            "study_name": study_name,
            "exp_id": exp_id,
            "roi_name": roi_name,
            "tissue": tissue,
            "region_type": region_type,
            "map_kind": kind,
            "cohort": row["cohort"],
            "group": row["group"],
            "test_end_point": row["test_end_point"],
            "n_before": before["n_pixels"],
            "n_after": after["n_pixels"],
            "pct_kept": pct_kept,
            "mean_before": before["mean"],
            "mean_after": after["mean"],
            "std_before": before["std"],
            "std_after": after["std"],
        })

    out = pd.DataFrame.from_records(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} row(s) to {OUT_PATH}\n")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)

    print("=== Per-ROI before/after (sanity check) ===")
    print(
        out[
            [
                "study_name", "exp_id", "roi_name", "tissue", "region_type",
                "map_kind", "cohort", "n_before", "n_after", "pct_kept",
                "mean_before", "mean_after",
            ]
        ].round(4).to_string(index=False)
    )

    zeroed = out[(out["n_before"] > 0) & (out["n_after"] == 0)]
    if len(zeroed):
        print(
            f"\nNOTE: {len(zeroed)} ROI(s) had every pixel filtered out "
            "(likely the 4-px sampling squares landing entirely on noise-floor "
            "or poorly-fit pixels) — excluded from the trend comparison below."
        )

    valid = out[out["n_after"] > 0]

    print("\n=== Irradiated vs Control by timepoint — whole-region ROI (ROI 1=NP, ROI 2=AF) ===")
    whole = (
        valid[valid["region_type"] == "whole"]
        .groupby(["tissue", "map_kind", "test_end_point", "cohort"])
        .agg(
            n_rats=("study_name", "nunique"),
            mean_of_roi_means=("mean_after", "mean"),
            std_of_roi_means=("mean_after", "std"),
            median_pct_kept=("pct_kept", "median"),
        )
        .round(5)
    )
    print(whole.to_string())

    print("\n=== Irradiated vs Control by timepoint — sampling squares (ROI 3/4=NP, ROI 5/6/7=AF) ===")
    samples = (
        valid[valid["region_type"] == "sample"]
        .groupby(["tissue", "map_kind", "test_end_point", "cohort"])
        .agg(
            n_rats=("study_name", "nunique"),
            n_roi=("roi_name", "count"),
            mean_of_roi_means=("mean_after", "mean"),
            std_of_roi_means=("mean_after", "std"),
            median_pct_kept=("pct_kept", "median"),
        )
        .round(5)
    )
    print(samples.to_string())

    print(
        "\n=== Irradiated (all timepoints pooled) vs Control (0d + 28d-C pooled) — "
        "Welch's t-test, one value per rat ==="
    )
    print(
        "(per-rat value = mean of that rat's ROI mean_after values, so each rat "
        "contributes once regardless of how many ROIs/timepoints it has)\n"
    )
    per_rat = (
        valid.groupby(["tissue", "map_kind", "region_type", "cohort", "study_name"])
        .agg(rat_mean=("mean_after", "mean"))
        .reset_index()
    )
    ttest_rows = []
    for (tissue, map_kind, region_type), grp in per_rat.groupby(
        ["tissue", "map_kind", "region_type"]
    ):
        irr = grp[grp["cohort"] == "irradiated"]["rat_mean"]
        ctl = grp[grp["cohort"] == "control"]["rat_mean"]
        if len(irr) < 2 or len(ctl) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(irr, ctl, equal_var=False)
        ttest_rows.append(
            {
                "tissue": tissue,
                "map_kind": map_kind,
                "region_type": region_type,
                "n_irradiated": len(irr),
                "n_control": len(ctl),
                "mean_irradiated": irr.mean(),
                "mean_control": ctl.mean(),
                "pct_diff": (irr.mean() - ctl.mean()) / ctl.mean() * 100,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )
    ttest_df = pd.DataFrame.from_records(ttest_rows).sort_values("p_value")
    print(ttest_df.round(5).to_string(index=False))

    print(
        "\n=== Per-timepoint irradiated vs control — Welch's t-test, one value per rat ==="
    )
    print(
        "(7d/14d irradiated compared against the 0d baseline control, since neither "
        "has a time-matched control cohort; 28d irradiated compared against the "
        "time-matched 28d-C control)\n"
    )
    per_rat_tp = (
        valid.groupby(
            ["tissue", "map_kind", "region_type", "test_end_point", "cohort", "study_name"]
        )
        .agg(rat_mean=("mean_after", "mean"))
        .reset_index()
    )
    _CONTROL_FOR = {"7d": "0d", "14d": "0d", "28d": "28d-C"}
    tp_rows = []
    for irr_tp, ctrl_tp in _CONTROL_FOR.items():
        irr_all = per_rat_tp[
            (per_rat_tp["test_end_point"] == irr_tp) & (per_rat_tp["cohort"] == "irradiated")
        ]
        ctl_all = per_rat_tp[
            (per_rat_tp["test_end_point"] == ctrl_tp) & (per_rat_tp["cohort"] == "control")
        ]
        for (tissue, map_kind, region_type), irr_grp in irr_all.groupby(
            ["tissue", "map_kind", "region_type"]
        ):
            ctl_grp = ctl_all[
                (ctl_all["tissue"] == tissue)
                & (ctl_all["map_kind"] == map_kind)
                & (ctl_all["region_type"] == region_type)
            ]
            irr = irr_grp["rat_mean"]
            ctl = ctl_grp["rat_mean"]
            if len(irr) < 2 or len(ctl) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(irr, ctl, equal_var=False)
            tp_rows.append(
                {
                    "timepoint": irr_tp,
                    "vs_control": ctrl_tp,
                    "tissue": tissue,
                    "map_kind": map_kind,
                    "region_type": region_type,
                    "n_irradiated": len(irr),
                    "n_control": len(ctl),
                    "mean_irradiated": irr.mean(),
                    "mean_control": ctl.mean(),
                    "pct_diff": (irr.mean() - ctl.mean()) / ctl.mean() * 100,
                    "t_stat": t_stat,
                    "p_value": p_val,
                }
            )
    tp_df = pd.DataFrame.from_records(tp_rows).sort_values(["timepoint", "p_value"])
    print(tp_df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
