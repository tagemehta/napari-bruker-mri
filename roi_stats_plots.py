"""Group-level T2 and ADC summary plots from parameter_maps/roi_stats_with_grouping.csv.

For each rat/disc, `roi_stats_with_grouping.csv` has one row per ROI per map
(T2/ADC), where "ROI 1" is always the nucleus pulposus (NP) and "ROI 2" is
always the annulus fibrosus (AF) (see AGENTS.md, "Per-disc ROI analysis").
This script produces three kinds of figures per map kind (T2, ADC), split into
NP and AF panels:

- A swarm plot of each disc's ROI-level mean and SD, grouped by irradiation
  group and timepoint (`plot_roi_value_swarm` / `make_roi_swarm_figure`).
- A group mean +/- SEM trend plot across timepoints, with each group's mean
  re-baselined against the 0d-Control mean (see "Baseline-subtracted
  progression" below) so the radiation-driven change over time reads
  directly off the y-axis (`plot_group_mean_trend` /
  `make_baseline_delta_trend_figure`). The non-baselined version of this
  plot showed the same shape shifted by a constant and was dropped as
  redundant — `git log` for `make_group_trend_figure` if the raw-value
  version is ever needed again.
- A bar chart of compare_trends.py's planned-contrast Welch's t-tests (%
  difference + FDR-adjusted p-value per contrast), for a presentation-ready
  summary of "how much do irradiated and control differ"
  (`make_contrast_test_figure`).

Group naming:
- Group 1/2 - 28d irradiated
- Group 3/4 - 14d irradiated
- Group 5/6 - 7d irradiated
- Extra - 28d irradiated (irradiated at a different dose/protocol)
- Control - split into "0d-Control" (baseline) and "28d-Control"
  (time-matched sham control) based on test_end_point

## Baseline-subtracted progression

Irradiated rats are never scanned at 0d (see compare_trends.py's module
docstring), so there's no per-rat baseline to subtract. Instead, each ROI's
0d-Control group mean (per tissue x map_kind) is used as a single shared
baseline, subtracted from every group's value at every timepoint before
averaging. This shifts every group's trend line down by the same constant
per panel — it does not change the shape of any single group's trend, and
group-vs-group comparisons at a given timepoint are unaffected — it just
gives progression-over-time a "change from baseline" y-axis instead of an
absolute one.

## T-test summary figure

`make_contrast_test_figure` visualizes compare_trends.py's planned
contrasts (`compute_planned_contrasts`) restricted to the whole-region ROIs
(ROI 1=NP, ROI 2=AF) — the same Welch's t-test / Benjamini-Hochberg
FDR-corrected results printed to the console by `compare_trends.py`, as a
bar chart of % difference per contrast with each bar labeled with its
FDR-adjusted p-value, faceted by tissue (NP/AF) x map kind (T2/ADC).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from compare_trends import (
    CONTRASTS,
    compute_planned_contrasts,
    load_and_annotate,
    per_rat_timepoint_means,
)

data = pd.read_csv(
    "/Users/smehta/Desktop/Dev/kraken/mri-viewer/parameter_maps/roi_stats_with_grouping.csv"
)
data["group"] = data.apply(
    lambda row: (
        "0d-Control"
        if row["group"] == "Control" and row["test_end_point"] != "28d-C"
        else "28d-Control"
        if row["group"] == "Control" and row["test_end_point"] == "28d-C"
        else row["group"]
    ),
    axis=1,
)

np_adc = data[(data["roi_name"] == "ROI 1") & (data["map_kind"] == "adc")]
af_adc = data[(data["roi_name"] == "ROI 2") & (data["map_kind"] == "adc")]
np_t2 = data[(data["roi_name"] == "ROI 1") & (data["map_kind"] == "t2")]
af_t2 = data[(data["roi_name"] == "ROI 2") & (data["map_kind"] == "t2")]

# Chronological x-axis order; 28d and 28d-C are kept as separate categories
# (irradiated vs. time-matched control) even though both represent day 28.
TIMEPOINT_ORDER = ["0d", "7d", "14d", "28d", "28d-C"]


def plot_roi_value_swarm(ax, df, y, title, ylabel, show_legend):
    """Swarm plot of one value column (e.g. per-ROI "mean" or "std") across
    timepoints, one point per disc, colored by group."""
    sns.swarmplot(
        data=df,
        x="test_end_point",
        y=y,
        hue="group",
        order=TIMEPOINT_ORDER,
        ax=ax,
        legend=show_legend,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def plot_group_mean_trend(
    ax, df, title, ylabel, show_legend, y="mean", zero_line=False
):
    """Group mean +/- SEM of per-ROI values at each timepoint, dodged so
    groups sharing a timepoint (e.g. Group 5/6 at 7d) don't overlap."""
    sns.pointplot(
        data=df,
        x="test_end_point",
        y=y,
        hue="group",
        order=TIMEPOINT_ORDER,
        ax=ax,
        errorbar="se",
        dodge=True,
        linestyle="none",
        legend=show_legend,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    else:
        ax.set_ylim(bottom=0)


def add_baseline_delta(df):
    """Return a copy of df with a "delta" column: each row's "mean" minus the
    0d-Control group mean for that same roi_name/map_kind (see module
    docstring, "Baseline-subtracted progression")."""
    baseline_rows = df[df["group"] == "0d-Control"]
    baseline = baseline_rows.groupby(["roi_name", "map_kind"])["mean"].mean()
    out = df.copy()
    out["delta"] = out.apply(
        lambda row: row["mean"] - baseline[(row["roi_name"], row["map_kind"])], axis=1
    )
    return out


def make_baseline_delta_trend_figure(np_df, af_df, tissue_label, value_label):
    """NP/AF side-by-side group mean +/- SEM trend plots, re-baselined
    against the 0d-Control mean so radiation-driven change over time reads
    directly off the y-axis instead of the raw value (see module docstring,
    "Baseline-subtracted progression")."""
    np_delta = add_baseline_delta(np_df)
    af_delta = add_baseline_delta(af_df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    plot_group_mean_trend(
        axes[0],
        np_delta,
        f"NP - {value_label}",
        f"\u0394 {value_label} from 0d baseline",
        show_legend=True,
        y="delta",
        zero_line=True,
    )
    plot_group_mean_trend(
        axes[1],
        af_delta,
        f"AF - {value_label}",
        f"\u0394 {value_label} from 0d baseline",
        show_legend=False,
        y="delta",
        zero_line=True,
    )
    axes[0].set_xlabel("Time point")
    axes[1].set_xlabel("Time point")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    fig.suptitle(f"{tissue_label} - change from 0d baseline (mean \u00b1 SEM)", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    return fig


def make_roi_swarm_figure(np_df, af_df, tissue_label, value_label):
    """2x2 grid (NP/AF rows x mean/SD columns) of per-disc ROI value swarm
    plots for one map kind (T2 or ADC)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    plot_roi_value_swarm(
        axes[0, 0],
        np_df,
        "mean",
        f"NP - {value_label} (mean)",
        value_label,
        show_legend=True,
    )
    plot_roi_value_swarm(
        axes[0, 1],
        np_df,
        "std",
        f"NP - {value_label} (SD)",
        value_label,
        show_legend=False,
    )
    plot_roi_value_swarm(
        axes[1, 0],
        af_df,
        "mean",
        f"AF - {value_label} (mean)",
        value_label,
        show_legend=False,
    )
    plot_roi_value_swarm(
        axes[1, 1],
        af_df,
        "std",
        f"AF - {value_label} (SD)",
        value_label,
        show_legend=False,
    )
    axes[1, 0].set_xlabel("Time point")
    axes[1, 1].set_xlabel("Time point")

    # Shared figure legend taken from the top-left axes.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.suptitle(tissue_label, y=0.97)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    return fig


def make_contrast_test_figure(contrast_df):
    """Bar chart of compare_trends.py's planned-contrast Welch's t-tests:
    % difference per contrast, each bar labeled with its FDR-adjusted
    p-value, faceted by tissue (NP/AF) x map kind (T2/ADC).

    Every facet uses the same fixed contrast order (CONTRASTS, not
    contrast_df's own p_adj-sorted order) so bars line up 1:1 across
    facets/rows even though each facet's own significance ranking differs
    (compute_planned_contrasts sorts by p_adj globally, which is only
    meaningful within a single facet)."""
    tissues = ["NP", "AF"]
    map_kinds = ["t2", "adc"]
    contrast_order = [f"{a} vs {b}" for a, b, _ in CONTRASTS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for row, tissue in enumerate(tissues):
        for col, map_kind in enumerate(map_kinds):
            ax = axes[row, col]
            sub = (
                contrast_df[
                    (contrast_df["tissue"] == tissue)
                    & (contrast_df["map_kind"] == map_kind)
                ]
                .set_index("contrast")
                .reindex(contrast_order)
            )

            bars = ax.bar(contrast_order, sub["pct_diff"], color="steelblue")
            for bar, p_adj in zip(bars, sub["p_adj"]):
                if pd.isna(p_adj):
                    continue
                height = bar.get_height()
                ax.annotate(
                    f"p={p_adj:.3g}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8,
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{tissue} - {map_kind.upper()}")
            ax.set_ylabel("% difference")
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.suptitle(
        "Planned contrasts (Welch's t-test, FDR-corrected) \u2014 whole-region ROIs",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_ratio_swarm_figure(wide):
    """Create swarm plot for AF/NP ratio."""
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_roi_value_swarm(
        ax[0],
        wide[wide["map_kind"] == "adc"],
        "np_af_ratio",
        "ADC AF/NP Ratio",
        ylabel="AF/NP Ratio",
        show_legend=True,
    )
    plot_roi_value_swarm(
        ax[1],
        wide[wide["map_kind"] == "t2"],
        "np_af_ratio",
        "T2AF/NP Ratio",
        ylabel="AF/NP Ratio",
        show_legend=False,
    )

    ax[0].set_xlabel("Timepoint")
    ax[1].set_xlabel("Timepoint")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].get_legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5)
    return fig


make_roi_swarm_figure(np_adc, af_adc, "ADC", "ADC")
make_roi_swarm_figure(np_t2, af_t2, "T2", "T2 (ms)")

make_baseline_delta_trend_figure(np_adc, af_adc, "ADC", "ADC")
make_baseline_delta_trend_figure(np_t2, af_t2, "T2", "T2 (ms)")

stats_df = load_and_annotate()
per_rat_tp = per_rat_timepoint_means(stats_df)
contrast_df = compute_planned_contrasts(per_rat_tp)
make_contrast_test_figure(contrast_df[contrast_df["region_type"] == "whole"])

wide = data.pivot_table(
    index=["rat_number", "exp_id", "map_kind", "group", "test_end_point"],
    columns="roi_name",
    values="mean",
).reset_index()

wide["np_af_ratio"] = wide["ROI 2"] / wide["ROI 1"]  # AF/NP

make_ratio_swarm_figure(wide)
plt.show()
