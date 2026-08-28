#!/usr/bin/env python3
"""Render a DAOC Fig. 6-style normalized latency radar preview."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_topconf_paper_figures as figure_style
from regenerate_critical_path_latency_figure import (
    DEFAULT_ANALYSIS_DIR,
)


ROOT = Path(__file__).resolve().parent
SOURCE = DEFAULT_ANALYSIS_DIR / "critical_path_method_summary.csv"
OUTPUT_DIR = ROOT / "paper_drafts" / "figures_topconf"
STEM = "fig03b_latency_radar_preview"

METHODS = (
    "greedy",
    "daoc_paper",
    "discrete_sac_std_cache",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
AXES = (
    ("computation_s", "Computation\nlatency"),
    ("total_s", "Total\nlatency"),
    ("waiting_s", "Waiting\nlatency"),
    ("service_loading_s", "Service loading\nlatency"),
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


def normalized_values(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_method = {
        method: {
            row["latency_component_key"]: row
            for row in rows
            if row["method_key"] == method
        }
        for method in METHODS
    }
    raw = {}
    for method in METHODS:
        raw[method] = {
            "computation_s": float(
                by_method[method]["computation_s"]["mean_component_time_s"]
            ),
            "waiting_s": float(
                by_method[method]["waiting_s"]["mean_component_time_s"]
            ),
            "service_loading_s": float(
                by_method[method]["service_loading_s"][
                    "mean_component_time_s"
                ]
            ),
            "total_s": float(
                by_method[method]["computation_s"]["mean_completion_time_s"]
            ),
        }
    maxima = {
        key: max(raw[method][key] for method in METHODS)
        for key, _ in AXES
    }
    return {
        method: {
            key: raw[method][key] / maxima[key]
            for key, _ in AXES
        }
        for method in METHODS
    }


def write_data(values: dict[str, dict[str, float]]) -> Path:
    path = OUTPUT_DIR / f"{STEM}_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method_key", "method", *[key for key, _ in AXES]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {
                    "method_key": method,
                    "method": figure_style.DISPLAY[method],
                    **{
                        key: f"{values[method][key]:.9f}"
                        for key, _ in AXES
                    },
                }
            )
    return path


def render() -> dict[str, Path]:
    values = normalized_values(read_rows(SOURCE))
    data_path = write_data(values)
    figure_style.configure_style()

    figure = plt.figure(
        figsize=(figure_style.DOUBLE_COLUMN, 92 * figure_style.MM)
    )
    axis = figure.add_subplot(111, polar=True)
    figure.subplots_adjust(left=0.055, right=0.70, top=0.84, bottom=0.16)

    count = len(AXES)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles)
    axis.set_xticklabels([label for _, label in AXES])
    axis.tick_params(axis="x", pad=10)
    axis.set_ylim(0, 1.04)
    axis.set_yticks((0.25, 0.50, 0.75, 1.00))
    axis.set_yticklabels(("0.25", "0.50", "0.75", "1.00"))
    axis.set_rlabel_position(315)
    axis.tick_params(axis="y", labelsize=5.6, colors="#667078", pad=1)
    axis.grid(color="#C9CED2", linewidth=0.62, alpha=0.85)
    axis.spines["polar"].set_color("#89949A")
    axis.spines["polar"].set_linewidth(0.75)

    for method in METHODS:
        series = np.asarray(
            [values[method][key] for key, _ in AXES], dtype=float
        )
        closed_series = np.r_[series, series[0]]
        is_our = method == "lean_our"
        is_reference = method == "greedy"
        axis.plot(
            closed_angles,
            closed_series,
            color=figure_style.COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0 if is_our else (1.0 if is_reference else 1.25),
            marker=figure_style.MARKERS[method],
            markersize=5.0 if is_our else 3.6,
            markerfacecolor=(
                figure_style.COLORS[method] if is_our else "white"
            ),
            markeredgecolor=figure_style.COLORS[method],
            markeredgewidth=0.75,
            zorder=5 if is_our else (2 if is_reference else 3),
            label=figure_style.DISPLAY[method],
        )
        if is_our:
            axis.fill(
                closed_angles,
                closed_series,
                color=figure_style.COLORS[method],
                alpha=0.10,
                zorder=1,
            )

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.71, 0.50),
        frameon=False,
        handlelength=2.6,
        labelspacing=0.78,
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
            "Creator": "plot_latency_radar_preview.py",
            "Title": "Normalized critical-path latency profile",
        },
        provenance={
            "raw_data": [str(SOURCE.relative_to(ROOT))],
            "derived_data": [str(data_path.relative_to(ROOT))],
            "transformations": [
                "normalize each latency dimension by its maximum across compared methods",
                "render four latency dimensions on a common zero-to-one radar scale",
            ],
            "uncertainty": "not shown on radar; absolute total-latency confidence intervals remain in the companion result figure",
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
