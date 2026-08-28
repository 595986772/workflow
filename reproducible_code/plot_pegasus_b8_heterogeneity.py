#!/usr/bin/env python3
"""Plot the three-seed fixed-budget cache-heterogeneity experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_line_style as line_style

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results/pegasus_pscale/p15_b8_heterogeneity"
DATA_PATH = RESULT_ROOT / "analysis/seed_level_results.csv"
SUMMARY_PATH = RESULT_ROOT / "analysis/heterogeneity_summary.json"
FIGURE_DIR = ROOT / "paper_drafts/figures_topconf"
OUTPUT_STEM = FIGURE_DIR / "fig_cache_capacity_heterogeneity"
STYLE_PATH = (
    ROOT.parent
    / ".codex/skills/scientific-visualization/assets/publication.mplstyle"
)

PROFILES = ("H0", "H1", "H2", "H3")
METHODS = (
    "daoc_paper",
    "daoc_our_coord_cache",
    "our_flat_ddqn",
    "lean_our",
)
DISPLAY = {
    "daoc_paper": "DAOC",
    "daoc_our_coord_cache": "DAOC + DCC",
    "our_flat_ddqn": "DDQN + DCC",
    "lean_our": "OUR",
}
VARIANCES = {"H0": 0.16, "H1": 0.36, "H2": 0.56, "H3": 0.96}


def load_rows() -> list[dict]:
    with DATA_PATH.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def method_values(rows: list[dict], profile: str, method: str) -> np.ndarray:
    selected = sorted(
        (
            (int(row["seed"]), float(row["mean_average_finish_time"]))
            for row in rows
            if row["profile"] == profile and row["method"] == method
        ),
        key=lambda item: item[0],
    )
    if [seed for seed, _ in selected] != [51, 52, 53]:
        raise RuntimeError(f"Missing seed-level data for {profile}/{method}")
    return np.asarray([value for _, value in selected], dtype=float)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError("Refusing to plot an incomplete analysis")
    rows = load_rows()

    line_style.configure_style(STYLE_PATH)
    figure, axis = plt.subplots(
        figsize=(line_style.DOUBLE_COLUMN, line_style.LINE_FIGURE_HEIGHT),
        layout="constrained",
    )
    x = np.arange(len(PROFILES), dtype=float)
    raw_offsets = np.asarray([-0.035, 0.0, 0.035])

    all_upper = []
    all_lower = []
    for method in METHODS:
        samples = np.vstack(
            [method_values(rows, profile, method) for profile in PROFILES]
        )
        means = samples.mean(axis=1)
        standard_deviations = samples.std(axis=1, ddof=1)
        all_upper.extend(means + standard_deviations)
        all_lower.extend(means - standard_deviations)
        axis.errorbar(
            x,
            means,
            yerr=standard_deviations,
            **line_style.errorbar_kwargs(method, label=DISPLAY[method]),
        )
        for profile_index in range(len(PROFILES)):
            axis.scatter(
                x[profile_index] + raw_offsets,
                samples[profile_index],
                marker="o",
                s=10,
                facecolor="white",
                edgecolor=line_style.COLORS[method],
                linewidth=0.55,
                alpha=0.82,
                zorder=6,
            )

    lower = max(0.0, np.floor((min(all_lower) - 0.04) / 0.1) * 0.1)
    upper = np.ceil((max(all_upper) + 0.06) / 0.1) * 0.1
    axis.set_xlim(-0.24, len(PROFILES) - 0.76)
    axis.set_ylim(lower, upper)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{profile}\nVar={VARIANCES[profile]:.2f}" for profile in PROFILES]
    )
    axis.set_xlabel("Cache-capacity profile (fixed total budget B=8)")
    axis.set_ylabel("Mean DAG completion time (s)")
    line_style.style_axis(axis, grid_axis="y")
    line_style.legend_above(axis, ncol=4)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(
            OUTPUT_STEM.with_suffix(f".{suffix}"),
            dpi=400 if suffix == "png" else None,
            bbox_inches=None,
            facecolor="white",
        )
    plt.close(figure)

    manifest = {
        "figure": str(OUTPUT_STEM),
        "source_data": str(DATA_PATH.resolve()),
        "analysis_summary": str(SUMMARY_PATH.resolve()),
        "replication_unit": "training seed",
        "seeds": [51, 52, 53],
        "estimator": "mean",
        "uncertainty": "sample standard deviation across three seeds",
        "raw_observations": "three open circles per method and profile",
        "capacity_budget": 8,
        "capacity_variances": VARIANCES,
        "axis_break": False,
        "transformations": ["none"],
    }
    OUTPUT_STEM.with_suffix(".export.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
