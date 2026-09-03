"""
Compare irradiated vs. control ROI trends across every available timepoint —
run with:
    uv run python compare_trends.py

Statistics, in two parts:
  1. Omnibus one-way ANOVA across 5 conditions (0d-control, 7d-irr, 14d-irr,
     28d-irr, 28d-C-control) as an overall "is anything different anywhere"
     check. A full Group x Timepoint two-way ANOVA isn't usable here:
     irradiated rats are never scanned at 0d and control rats are never
     scanned at 7d/14d, so several (group, timepoint) cells are
     structurally empty, not just small — a two-way model can't estimate
     an interaction from that. (There's deliberately no single pooled
     "irradiated vs control" test: pooling 0d — a pre-treatment baseline —
     with 28d-C — a time-matched sham — into one "control" bucket, and all
     three post-irradiation timepoints into one "irradiated" bucket, blurs
     exactly the timepoint structure this analysis cares about.)
  2. Planned contrasts via Welch's t-test, Benjamini-Hochberg FDR-corrected
     across the whole family (5 contrasts x 8 tissue/map_kind/region_type
     combinations = 40 tests). Two kinds:
       - vs-control: 7d-irr vs 0d-control, 14d-irr vs 0d-control, 28d-irr
         vs 28d-C-control (7d/14d have no time-matched control, so they're
         compared against the 0d baseline instead).
       - timepoint-to-timepoint, within the irradiated cohort only: 7d-irr
         vs 14d-irr, 14d-irr vs 28d-irr — does the irradiated group itself
         change over time, independent of any control comparison.

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

    # --- Per-timepoint comparison: one-way ANOVA + planned contrasts -------
    #
    # A full Group x Timepoint two-way ANOVA isn't estimable here: irradiated
    # rats are never scanned at 0d and control rats are never scanned at
    # 7d/14d, so several (group, timepoint) cells are structurally empty, not
    # just small. Instead, each rat is assigned one of 5 mutually-exclusive
    # conditions, and:
    #   1. An omnibus one-way ANOVA across all 5 conditions checks whether
    #      there's any difference anywhere, as an overall sanity check.
    #   2. Only the biologically meaningful contrasts are actually tested via
    #      Welch's t-test — comparing every condition to every other (e.g.
    #      0d-control vs 28d-C-control) would answer questions nobody's
    #      asking here. Two kinds of contrast:
    #        - vs. control: 7d-irr vs 0d-control, 14d-irr vs 0d-control,
    #          28d-irr vs 28d-C-control (7d/14d have no time-matched control,
    #          so they're compared against the 0d baseline instead).
    #        - timepoint-to-timepoint, within the irradiated cohort only:
    #          7d-irr vs 14d-irr, 14d-irr vs 28d-irr — does the irradiated
    #          group itself change as time passes, independent of any
    #          control comparison.
    #   3. Because that's still many tests (5 contrasts x 8 tissue/map_kind/
    #      region_type combinations = 40), a Benjamini-Hochberg FDR
    #      correction is applied across the whole family, so the reported
    #      significance isn't just each test's own uncorrected p-value.
    print("\n=== Per-timepoint irradiated vs control — one-way ANOVA (5 conditions) ===")
    print(
        "(one value per rat; conditions = 0d-control, 7d-irr, 14d-irr, 28d-irr, "
        "28d-C-control — this is an omnibus check across all 5, not the pairwise "
        "comparisons of interest, which follow below)\n"
    )

    def _condition(row) -> str | None:
        if row["cohort"] == "control" and row["test_end_point"] in ("0d", "28d-C"):
            return f"{row['test_end_point']}-control"
        if row["cohort"] == "irradiated" and row["test_end_point"] in ("7d", "14d", "28d"):
            return f"{row['test_end_point']}-irr"
        return None

    per_rat_tp = (
        stats_df.groupby(
            ["tissue", "map_kind", "region_type", "test_end_point", "cohort", "study_name"]
        )
        .agg(rat_mean=("mean", "mean"))
        .reset_index()
    )
    per_rat_tp["condition"] = per_rat_tp.apply(_condition, axis=1)
    per_rat_tp = per_rat_tp.dropna(subset=["condition"])

    anova_rows = []
    for (tissue, map_kind, region_type), grp in per_rat_tp.groupby(
        ["tissue", "map_kind", "region_type"]
    ):
        groups = [g["rat_mean"].to_numpy() for _, g in grp.groupby("condition")]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        anova_rows.append({
            "tissue": tissue, "map_kind": map_kind, "region_type": region_type,
            "n_conditions": len(groups), "f_stat": f_stat, "p_value": p_val,
        })
    anova_df = pd.DataFrame.from_records(anova_rows).sort_values("p_value")
    print(anova_df.round(5).to_string(index=False))

    print(
        "\n=== Planned contrasts — Welch's t-test, Benjamini-Hochberg FDR-corrected ==="
    )
    print(
        "(vs-control: 7d/14d-irr compared against the 0d baseline control, since "
        "neither has a time-matched control cohort; 28d-irr compared against the "
        "time-matched 28d-C control. timepoint-to-timepoint: irradiated cohort "
        "compared against itself at the previous timepoint, independent of control. "
        "p_adj controls the false discovery rate across all contrasts x tissue x "
        "map_kind x region_type combinations together)\n"
    )
    _CONTRASTS = [
        ("7d-irr", "0d-control", "vs-control"),
        ("14d-irr", "0d-control", "vs-control"),
        ("28d-irr", "28d-C-control", "vs-control"),
        ("14d-irr", "7d-irr", "timepoint-to-timepoint"),
        ("28d-irr", "14d-irr", "timepoint-to-timepoint"),
    ]
    contrast_rows = []
    for cond_a, cond_b, kind in _CONTRASTS:
        for (tissue, map_kind, region_type), grp in per_rat_tp.groupby(
            ["tissue", "map_kind", "region_type"]
        ):
            a = grp[grp["condition"] == cond_a]["rat_mean"]
            b = grp[grp["condition"] == cond_b]["rat_mean"]
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            contrast_rows.append({
                "contrast": f"{cond_a} vs {cond_b}",
                "kind": kind,
                "tissue": tissue,
                "map_kind": map_kind,
                "region_type": region_type,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "pct_diff": (a.mean() - b.mean()) / b.mean() * 100,
                "t_stat": t_stat,
                "p_value": p_val,
            })
    contrast_df = pd.DataFrame.from_records(contrast_rows)
    contrast_df["p_adj"] = _benjamini_hochberg(contrast_df["p_value"].to_numpy())
    contrast_df = contrast_df.sort_values("p_adj")
    print(contrast_df.round(5).to_string(index=False))


def _benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted p-values (same order as the input)."""
    n = len(p_values)
    order = p_values.argsort()
    ranked = p_values[order] * n / (pd.Series(range(1, n + 1)))
    # Enforce monotonicity: adjusted p-values must not decrease going up in rank.
    adjusted = ranked.to_numpy()[::-1]
    adjusted = pd.Series(adjusted).cummin().to_numpy()[::-1]
    adjusted = adjusted.clip(max=1.0)
    out = pd.Series(index=order, data=adjusted).sort_index()
    return out.to_numpy()


if __name__ == "__main__":
    main()
