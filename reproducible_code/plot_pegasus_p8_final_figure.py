#!/usr/bin/env python3
"""Create the final non-overlapping P8 figure from locked result data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "results/pegasus_pscale/p8_daoc_our_coord_cache/analysis"
SUMMARY_PATH = ANALYSIS_DIR / "daoc_coord_cache_extension_summary.json"
OUTPUT_STEM = ANALYSIS_DIR / "p8_controlled_comparison_final"

METHODS = (
    "daoc_paper",
    "daoc_our_coord_cache",
    "discrete_sac_std_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
DISPLAY = {
    "daoc_paper": "DAOC",
    "daoc_our_coord_cache": "DAOC + DCC",
    "discrete_sac_std_cache": "SAC + DAOC-Cache",
    "coord_cache_discrete_sac": "SAC + DCC",
    "lean_our": "HERO-DAG",
}
COLORS = {
    "daoc_paper": "#6B7280",
    "daoc_our_coord_cache": "#E69F00",
    "discrete_sac_std_cache": "#CC79A7",
    "coord_cache_discrete_sac": "#8B6BB1",
    "lean_our": "#007C91",
}
MARKERS = {
    "daoc_paper": "o",
    "daoc_our_coord_cache": "P",
    "discrete_sac_std_cache": "s",
    "coord_cache_discrete_sac": "D",
    "lean_our": "^",
}


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, half


def main():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if not all(summary["integrity"].values()):
        raise RuntimeError("Refusing to plot a result that failed integrity checks")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.35),
        sharey=True,
        layout="constrained",
    )
    metrics = (
        ("mean_finish_time", "Mean DAG completion time (s)", "(a) Mean"),
        ("mean_p95_finish_time", "P95 DAG completion time (s)", "(b) P95"),
    )
    positions = np.arange(len(METHODS))[::-1]
    rng = np.random.default_rng(20260811)
    for ax, (metric, xlabel, title) in zip(axes, metrics):
        for position, method in zip(positions, METHODS):
            values = np.asarray(
                [row["methods"][method][metric] for row in summary["per_seed"]],
                dtype=float,
            )
            mean, half = mean_ci95(values)
            ax.barh(
                position,
                mean,
                height=0.58,
                color=COLORS[method],
                edgecolor="#333333",
                linewidth=0.6,
                zorder=2,
            )
            ax.errorbar(
                mean,
                position,
                xerr=half,
                color="#202020",
                capsize=2.5,
                linewidth=0.9,
                zorder=4,
            )
            jitter = rng.uniform(-0.15, 0.15, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), position) + jitter,
                marker=MARKERS[method],
                s=16,
                color="white",
                edgecolor="#222222",
                linewidth=0.55,
                zorder=5,
            )
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([DISPLAY[method] for method in METHODS])
    fig.suptitle(
        "Controlled cache and scheduler comparisons",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
