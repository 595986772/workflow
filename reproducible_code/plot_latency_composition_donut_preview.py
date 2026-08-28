#!/usr/bin/env python3
"""Render a publication-style donut-matrix latency composition preview."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_topconf_paper_figures as figure_style
from regenerate_critical_path_latency_figure import (
    COMPONENT_COLORS,
    COMPONENT_FIELDS,
    DEFAULT_ANALYSIS_DIR,
    METHODS,
)


ROOT = Path(__file__).resolve().parent
SOURCE = DEFAULT_ANALYSIS_DIR / "critical_path_method_summary.csv"
OUTPUT_DIR = ROOT / "paper_drafts" / "figures_topconf"
STEM = "fig03b_latency_composition_donut_preview"

VISIBLE_FIELDS = (
    "service_loading_s",
    "waiting_s",
    "computation_s",
)
SHORT_LABELS = {
    "service_loading_s": "Service",
    "waiting_s": "Waiting",
    "computation_s": "Processing",
}
LABEL_POSITIONS = {
    "service_loading_s": (0.58, -1.12, "left"),
    "waiting_s": (-1.02, 0.56, "right"),
    "computation_s": (-0.42, 1.10, "center"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_ms(value_ms: float) -> str:
    if value_ms < 1:
        return f"{value_ms:.2f}"
    if value_ms < 10:
        return f"{value_ms:.1f}"
    return f"{value_ms:.0f}"


def render() -> dict[str, Path]:
    rows = read_rows(SOURCE)
    by_method = {
        method: [row for row in rows if row["method_key"] == method]
        for method in METHODS
    }
    figure_style.configure_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(figure_style.DOUBLE_COLUMN, 116 * figure_style.MM),
        subplot_kw={"aspect": "equal"},
        layout="constrained",
    )

    minimum_dependency_ms = float("inf")
    maximum_dependency_ms = 0.0
    for axis, method in zip(axes.flat, METHODS):
        indexed = {
            row["latency_component_key"]: row for row in by_method[method]
        }
        values = np.asarray(
            [float(indexed[field]["mean_component_time_s"]) for field in COMPONENT_FIELDS]
        )
        shares = np.asarray(
            [float(indexed[field]["mean_share_percent"]) for field in COMPONENT_FIELDS]
        )
        dependency_ms = 1000.0 * float(
            indexed["dependency_transfer_s"]["mean_component_time_s"]
        )
        minimum_dependency_ms = min(minimum_dependency_ms, dependency_ms)
        maximum_dependency_ms = max(maximum_dependency_ms, dependency_ms)
        colors = [COMPONENT_COLORS[field] for field in COMPONENT_FIELDS]
        wedges, _ = axis.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            radius=1.0,
            wedgeprops={
                "width": 0.34,
                "edgecolor": "white",
                "linewidth": 1.05,
            },
        )

        total_s = float(indexed[COMPONENT_FIELDS[0]]["mean_completion_time_s"])
        ci_s = float(
            indexed[COMPONENT_FIELDS[0]]["completion_ci95_half_width_s"]
        )
        axis.text(
            0,
            0.085,
            f"{total_s:.3f} s",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#253238",
        )
        axis.text(
            0,
            -0.095,
            f"95% CI +/- {ci_s:.3f}",
            ha="center",
            va="center",
            fontsize=5.2,
            color="#59636A",
        )

        for field in VISIBLE_FIELDS:
            field_index = COMPONENT_FIELDS.index(field)
            wedge = wedges[field_index]
            theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
            anchor_x = 0.91 * np.cos(theta)
            anchor_y = 0.91 * np.sin(theta)
            label_x, label_y, alignment = LABEL_POSITIONS[field]
            value_ms = 1000.0 * values[field_index]
            share = shares[field_index]
            axis.annotate(
                (
                    f"{SHORT_LABELS[field]}\n"
                    f"{format_ms(value_ms)} ms ({share:.1f}%)"
                ),
                xy=(anchor_x, anchor_y),
                xytext=(label_x, label_y),
                ha=alignment,
                va="center",
                fontsize=5.15,
                linespacing=0.92,
                color="#30383D",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#7A8388",
                    "linewidth": 0.55,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

        axis.set_title(
            figure_style.DISPLAY[method],
            fontsize=8.2,
            fontweight="bold" if method == "lean_our" else "normal",
            color=figure_style.COLORS[method],
            pad=10,
        )
        axis.set_xlim(-1.48, 1.48)
        axis.set_ylim(-1.31, 1.31)
        axis.set_axis_off()

    figure.text(
        0.5,
        0.01,
        (
            "User-input transfer = 0 ms. Dependency transfer is included "
            f"in each ring ({minimum_dependency_ms:.2f}--"
            f"{maximum_dependency_ms:.2f} ms) but is visually negligible."
        ),
        ha="center",
        va="bottom",
        fontsize=5.7,
        color="#566168",
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
            "Creator": "plot_latency_composition_donut_preview.py",
            "Title": "Critical-path latency composition donut matrix",
        },
        provenance={
            "raw_data": [str(SOURCE.relative_to(ROOT))],
            "transformations": [
                "render non-overlapping realized-critical-path latency components as rings",
                "place total completion time and confidence interval at each ring center",
            ],
            "uncertainty": "95% Student-t CI over ten independent seed-level means",
            "validation": "absolute component values and shares are read without modification",
            "skill": "scientific-visualization",
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
