#!/usr/bin/env python3
"""Render a normalized multi-metric latency profile preview."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_topconf_paper_figures as figure_style
from regenerate_critical_path_latency_figure import DEFAULT_ANALYSIS_DIR


ROOT = Path(__file__).resolve().parent
SOURCE = DEFAULT_ANALYSIS_DIR / "critical_path_method_summary.csv"
OUTPUT_DIR = ROOT / "paper_drafts" / "figures_topconf"
STEM = "fig03b_latency_profile_preview"

METHODS = (
    "greedy",
    "daoc_paper",
    "discrete_sac_std_cache",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
METRICS = (
    ("computation_s", "Computation"),
    ("total_s", "Total"),
    ("waiting_s", "Waiting"),
    ("service_loading_s", "Service loading"),
)
LINESTYLES = {
    "greedy": (0, (3.0, 2.0)),
    "daoc_paper": "-",
    "discrete_sac_std_cache": (0, (5.0, 1.8)),
    "daoc_our_coord_cache": (0, (1.5, 1.5)),
    "coord_cache_discrete_sac": "-.",
    "lean_our": "-",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_method = {
        method: {
            row["latency_component_key"]: row
            for row in rows
            if row["method_key"] == method
        }
        for method in METHODS
    }
    raw: dict[str, dict[str, float]] = {}
    for method in METHODS:
        raw[method] = {
            "computation_s": float(
                by_method[method]["computation_s"]["mean_component_time_s"]
            ),
            "waiting_s": float(
                by_method[method]["waiting_s"]["mean_component_time_s"]
            ),
            "service_loading_s": float(
                by_method[method]["service_loading_s"]["mean_component_time_s"]
            ),
            "total_s": float(
                by_method[method]["computation_s"]["mean_completion_time_s"]
            ),
        }
    maxima = {
        key: max(raw[method][key] for method in METHODS)
        for key, _ in METRICS
    }
    return {
        method: {key: raw[method][key] / maxima[key] for key, _ in METRICS}
        for method in METHODS
    }


def write_data(values: dict[str, dict[str, float]]) -> Path:
    path = OUTPUT_DIR / f"{STEM}_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method_key", "method", *[key for key, _ in METRICS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {
                    "method_key": method,
                    "method": figure_style.DISPLAY[method],
                    **{key: f"{values[method][key]:.9f}" for key, _ in METRICS},
                }
            )
    return path


def render() -> dict[str, Path]:
    values = normalize(read_rows(SOURCE))
    data_path = write_data(values)
    figure_style.configure_style()

    figure, axis = plt.subplots(
        figsize=(figure_style.DOUBLE_COLUMN, 73 * figure_style.MM)
    )
    figure.subplots_adjust(left=0.095, right=0.985, top=0.79, bottom=0.22)

    x = np.arange(len(METRICS), dtype=float)
    axis.set_xlim(-0.10, len(METRICS) - 0.90)
    axis.set_ylim(0.0, 1.06)
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in METRICS])
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_ylabel("Normalized latency")
    axis.grid(axis="y", color="#D8DDE0", linewidth=0.65, alpha=0.95)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7E898F")
    axis.spines["bottom"].set_color("#7E898F")

    for method in METHODS:
        y = np.asarray([values[method][key] for key, _ in METRICS])
        is_our = method == "lean_our"
        is_reference = method == "greedy"
        axis.plot(
            x,
            y,
            color=figure_style.COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0 if is_our else (1.05 if is_reference else 1.35),
            marker=figure_style.MARKERS[method],
            markersize=5.0 if is_our else 3.8,
            markerfacecolor=figure_style.COLORS[method] if is_our else "white",
            markeredgecolor=figure_style.COLORS[method],
            markeredgewidth=0.8,
            zorder=5 if is_our else (2 if is_reference else 3),
            label=figure_style.DISPLAY[method],
        )

    axis.text(
        0.0,
        1.035,
        "Lower is better",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#59636A",
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.975),
        ncol=3,
        frameon=False,
        columnspacing=1.55,
        handlelength=2.6,
        labelspacing=0.55,
    )

    output_stem = OUTPUT_DIR / STEM
    figure_style.export_figure(
        figure,
        output_stem,
        formats=("pdf", "png", "svg"),
        dpi=400,
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        font_mode="truetype",
        overwrite=True,
        mkdir=True,
        metadata={
            "Creator": "plot_latency_profile_preview.py",
            "Title": "Normalized multi-metric latency profile",
        },
        provenance={
            "raw_data": [str(SOURCE.relative_to(ROOT))],
            "derived_data": [str(data_path.relative_to(ROOT))],
            "transformations": [
                "normalize each latency metric by its maximum across methods",
                "connect each method's normalized values across four latency metrics",
            ],
            "uncertainty": "not shown; absolute total-latency confidence intervals remain in the companion result figure",
            "validation": "all normalized values are constrained to [0, 1]",
            "skill": "scientific-figure-making",
            "skill_commit": figure_style.SCI_VIS_COMMIT,
        },
        write_manifest=True,
    )
    plt.close(figure)
    return {
        suffix: output_stem.with_suffix(f".{suffix}")
        for suffix in ("png", "pdf", "svg", "export.json")
    }


if __name__ == "__main__":
    print(render())
