"""
Compare irradiated vs. control ROI trends across every available timepoint,
with pooled and per-timepoint Welch's t-tests — run with:
    uv run python compare_trends.py

Reads parameter_maps/roi_stats_with_grouping.csv directly (no raw-map/ROI
reloading needed) — run join_roi_stats_grouping.py first if that file
doesn't exist yet or is stale. Per-pixel SNR/R² quality filtering was
dropped from this analysis: it wasn't adding anything over just using
analyze_disc.py's own fitted ROI means, and made every re-run reload and
re-rasterize every map/ROI from disk (slow) for little benefit. If a
per-pixel quality proxy is ever needed again, ``git log`` for
``snr_filter_and_compare.py``.

Cohort: every ROI in roi_stats_with_grouping.csv. Per patients/rat-grouping.csv,
groups map to timepoints as follows:
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
"""

import pandas as pd
from scipy import stats

STATS_PATH = "parameter_maps/roi_stats_with_grouping.csv"

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


def main() -> None:
    stats_df = pd.read_csv(STATS_PATH).dropna(subset=["study_name"])
    stats_df["cohort"] = stats_df["group"].map(_cohort)
    stats_df = stats_df[stats_df["cohort"] != "other"]
    tissue_region = stats_df["roi_name"].apply(lambda r: _ROI_TISSUE.get(r, ("unknown", "unknown")))
    stats_df["tissue"] = tissue_region.apply(lambda t: t[0])
    stats_df["region_type"] = tissue_region.apply(lambda t: t[1])

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)

    print("=== Irradiated vs Control by timepoint — whole-region ROI (ROI 1=NP, ROI 2=AF) ===")
    whole = (
        stats_df[stats_df["region_type"] == "whole"]
        .groupby(["tissue", "map_kind", "test_end_point", "cohort"])
        .agg(
            n_rats=("study_name", "nunique"),
            mean_of_roi_means=("mean", "mean"),
            std_of_roi_means=("mean", "std"),
        )
        .round(5)
    )
    print(whole.to_string())

    print("\n=== Irradiated vs Control by timepoint — sampling squares (ROI 3/4=NP, ROI 5/6/7=AF) ===")
    samples = (
        stats_df[stats_df["region_type"] == "sample"]
        .groupby(["tissue", "map_kind", "test_end_point", "cohort"])
        .agg(
            n_rats=("study_name", "nunique"),
            n_roi=("roi_name", "count"),
            mean_of_roi_means=("mean", "mean"),
            std_of_roi_means=("mean", "std"),
        )
        .round(5)
    )
    print(samples.to_string())

    print(
        "\n=== Irradiated (all timepoints pooled) vs Control (0d + 28d-C pooled) — "
        "Welch's t-test, one value per rat ==="
    )
    print(
        "(per-rat value = mean of that rat's ROI mean values, so each rat "
        "contributes once regardless of how many ROIs/timepoints it has)\n"
    )
    per_rat = (
        stats_df.groupby(["tissue", "map_kind", "region_type", "cohort", "study_name"])
        .agg(rat_mean=("mean", "mean"))
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
        stats_df.groupby(
            ["tissue", "map_kind", "region_type", "test_end_point", "cohort", "study_name"]
        )
        .agg(rat_mean=("mean", "mean"))
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
