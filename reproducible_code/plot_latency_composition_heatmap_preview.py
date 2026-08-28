#!/usr/bin/env python3
"""Render a table-heatmap alternative for the latency-composition figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import plot_topconf_paper_figures as figure_style
from regenerate_critical_path_latency_figure import (
    COMPONENT_FIELDS,
    COMPONENT_LABELS,
    DEFAULT_ANALYSIS_DIR,
    METHODS,
)


ROOT = Path(__file__).resolve().parent
SOURCE = DEFAULT_ANALYSIS_DIR / "critical_path_method_summary.csv"
OUTPUT_DIR = ROOT / "paper_drafts" / "figures_topconf"
STEM = "fig03b_latency_composition_heatmap_preview"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_ms(value_ms: float) -> str:
    if value_ms == 0:
        return "0"
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
    completion_means = np.asarray(
        [float(by_method[method][0]["mean_completion_time_s"]) for method in METHODS]
    )
    completion_errors = np.asarray(
        [
            float(by_method[method][0]["completion_ci95_half_width_s"])
            for method in METHODS
        ]
    )
    component_ms = np.zeros((len(METHODS), len(COMPONENT_FIELDS)))
    component_shares = np.zeros_like(component_ms)
    for method_index, method in enumerate(METHODS):
        indexed = {
            row["latency_component_key"]: row for row in by_method[method]
        }
        for field_index, field in enumerate(COMPONENT_FIELDS):
            component_ms[method_index, field_index] = (
                1000.0 * float(indexed[field]["mean_component_time_s"])
            )
            component_shares[method_index, field_index] = float(
                indexed[field]["mean_share_percent"]
            )

    figure_style.configure_style()
    figure, (performance_axis, heatmap_axis) = plt.subplots(
        1,
        2,
        figsize=(figure_style.DOUBLE_COLUMN, 82 * figure_style.MM),
        gridspec_kw={"width_ratios": (0.98, 1.42)},
        layout="constrained",
    )
    positions = np.arange(len(METHODS), dtype=float)
    bars = performance_axis.barh(
        positions,
        completion_means,
        xerr=completion_errors,
        height=0.58,
        color=[figure_style.COLORS[method] for method in METHODS],
        edgecolor="#30363A",
        linewidth=0.7,
        error_kw={
            "ecolor": "#30363A",
            "elinewidth": 0.8,
            "capsize": 2.5,
            "capthick": 0.8,
        },
        zorder=2,
    )
    completion_limit = float(np.max(completion_means + completion_errors))
    for bar, mean, error in zip(bars, completion_means, completion_errors):
        performance_axis.text(
            mean + error + 0.03 * completion_limit,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f}",
            ha="left",
            va="center",
            fontsize=6.4,
            color="#30363A",
        )
    performance_axis.set_yticks(positions)
    performance_axis.set_yticklabels(
        [figure_style.DISPLAY[method] for method in METHODS]
    )
    performance_axis.invert_yaxis()
    performance_axis.set_xlim(0, completion_limit * 1.24)
    performance_axis.set_xlabel("Mean DAG completion time (s)")
    figure_style.panel_label(performance_axis, "(a)")
    figure_style.style_axis(performance_axis, grid="x")

    share_cmap = LinearSegmentedColormap.from_list(
        "component_share",
        ("#F7F9FA", "#D8E5EB", "#8EADB9", "#416F82"),
    )
    image = heatmap_axis.imshow(
        component_shares,
        cmap=share_cmap,
        vmin=0,
        vmax=100,
        interpolation="nearest",
        aspect="auto",
    )
    short_labels = {
        "user_input_transfer_s": "Input\ntransfer",
        "dependency_transfer_s": "Dependency\ntransfer",
        "service_loading_s": "Service\nloading",
        "waiting_s": "Waiting",
        "computation_s": "Task\nprocessing",
    }
    heatmap_axis.set_xticks(np.arange(len(COMPONENT_FIELDS)))
    heatmap_axis.set_xticklabels(
        [short_labels[field] for field in COMPONENT_FIELDS]
    )
    heatmap_axis.xaxis.tick_top()
    heatmap_axis.tick_params(
        axis="x",
        top=False,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        pad=4,
    )
    heatmap_axis.set_yticks(positions)
    heatmap_axis.set_yticklabels([])
    heatmap_axis.tick_params(axis="y", left=False)
    for row_index in range(len(METHODS)):
        for column_index in range(len(COMPONENT_FIELDS)):
            share = component_shares[row_index, column_index]
            value_ms = component_ms[row_index, column_index]
            color = "white" if share >= 55 else "#253238"
            heatmap_axis.text(
                column_index,
                row_index,
                f"{format_ms(value_ms)}\n({share:.1f}%)",
                ha="center",
                va="center",
                fontsize=5.25,
                linespacing=0.9,
                color=color,
            )
    heatmap_axis.set_xticks(
        np.arange(-0.5, len(COMPONENT_FIELDS), 1), minor=True
    )
    heatmap_axis.set_yticks(
        np.arange(-0.5, len(METHODS), 1), minor=True
    )
    heatmap_axis.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.4,
    )
    heatmap_axis.tick_params(which="minor", bottom=False, left=False)
    heatmap_axis.set_xlabel(
        "Mean component time (ms); share in parentheses"
    )
    heatmap_axis.xaxis.set_label_position("bottom")
    figure_style.panel_label(heatmap_axis, "(b)")
    for side in ("top", "right", "bottom", "left"):
        heatmap_axis.spines[side].set_visible(True)
        heatmap_axis.spines[side].set_color("#30363A")
        heatmap_axis.spines[side].set_linewidth(0.65)

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
            "Creator": "plot_latency_composition_heatmap_preview.py",
            "Title": "DAG completion time and critical-path component table",
        },
        provenance={
            "raw_data": [str(SOURCE.relative_to(ROOT))],
            "transformations": [
                "convert critical-path component means from seconds to milliseconds",
                "encode within-method component share by cell intensity",
            ],
            "uncertainty": "95% Student-t CI over ten seed means for completion time",
            "validation": "uses the repaired realized-critical-path summary unchanged",
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
