#!/usr/bin/env python3
"""Render an OSDI-style absolute/relative latency-breakdown preview."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter

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
STEM = "fig03b_latency_composition_paired_stacks_preview"

DOMINANT_FIELDS = (
    "service_loading_s",
    "waiting_s",
    "computation_s",
)
SHORT_LABELS = {
    "service_loading_s": "Service loading",
    "waiting_s": "Waiting",
    "computation_s": "Task processing",
}

# A cool editorial palette keeps the dominant service-loading segment calm,
# while the warm waiting accent and slate processing segment remain distinct.
COMPONENT_COLORS = {
    "user_input_transfer_s": "#A9C1CF",
    "dependency_transfer_s": "#78A69B",
    "service_loading_s": "#6F91B5",
    "waiting_s": "#C67868",
    "computation_s": "#817694",
}
OUR_COLOR = "#164E70"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render() -> dict[str, Path]:
    rows = read_rows(SOURCE)
    by_method = {
        method: {
            row["latency_component_key"]: row
            for row in rows
            if row["method_key"] == method
        }
        for method in METHODS
    }

    figure_style.configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(figure_style.DOUBLE_COLUMN, 94 * figure_style.MM),
        gridspec_kw={"width_ratios": (1.24, 1.0), "wspace": 0.10},
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.175,
        right=0.988,
        top=0.825,
        bottom=0.205,
        wspace=0.20,
    )
    absolute_axis, share_axis = axes
    positions = np.arange(len(METHODS), dtype=float)
    bar_height = 0.58

    values_ms = {
        field: np.asarray(
            [
                1000.0
                * float(by_method[method][field]["mean_component_time_s"])
                for method in METHODS
            ]
        )
        for field in COMPONENT_FIELDS
    }
    shares = {
        field: np.asarray(
            [
                float(by_method[method][field]["mean_share_percent"])
                for method in METHODS
            ]
        )
        for field in COMPONENT_FIELDS
    }
    totals_ms = np.asarray(
        [
            1000.0
            * float(
                by_method[method][COMPONENT_FIELDS[0]][
                    "mean_completion_time_s"
                ]
            )
            for method in METHODS
        ]
    )
    ci_ms = np.asarray(
        [
            1000.0
            * float(
                by_method[method][COMPONENT_FIELDS[0]][
                    "completion_ci95_half_width_s"
                ]
            )
            for method in METHODS
        ]
    )

    absolute_left = np.zeros(len(METHODS), dtype=float)
    for field in COMPONENT_FIELDS:
        absolute_axis.barh(
            positions,
            values_ms[field],
            left=absolute_left,
            height=bar_height,
            color=COMPONENT_COLORS[field],
            edgecolor="white",
            linewidth=0.72,
            label=COMPONENT_LABELS[field],
            zorder=2,
        )
        absolute_left += values_ms[field]

    if not np.allclose(absolute_left, totals_ms, atol=1e-5):
        raise RuntimeError("Absolute latency components do not match totals")

    absolute_axis.errorbar(
        totals_ms,
        positions,
        xerr=ci_ms,
        fmt="none",
        ecolor="#354047",
        elinewidth=0.78,
        capsize=2.4,
        capthick=0.78,
        zorder=4,
    )

    service_ms = values_ms["service_loading_s"]
    waiting_ms = values_ms["waiting_s"]
    processing_ms = values_ms["computation_s"]
    for index, method in enumerate(METHODS):
        service_start = sum(
            values_ms[field][index]
            for field in COMPONENT_FIELDS
            if COMPONENT_FIELDS.index(field)
            < COMPONENT_FIELDS.index("service_loading_s")
        )
        absolute_axis.text(
            service_start + service_ms[index] / 2.0,
            positions[index],
            f"{service_ms[index]:.0f}",
            ha="center",
            va="center",
            fontsize=5.9,
            color="#252B2F",
            fontweight="bold" if method == "lean_our" else "normal",
            zorder=5,
        )
        label_x = totals_ms[index] + ci_ms[index] + 28.0
        absolute_axis.text(
            label_x,
            positions[index] - 0.075,
            f"{totals_ms[index]:.0f} +/- {ci_ms[index]:.0f}",
            ha="left",
            va="center",
            fontsize=5.75,
            color="#263238",
            fontweight="bold" if method == "lean_our" else "normal",
        )
        absolute_axis.text(
            label_x,
            positions[index] + 0.135,
            f"W {waiting_ms[index]:.0f}  |  P {processing_ms[index]:.0f}",
            ha="left",
            va="center",
            fontsize=4.85,
            color="#667078",
        )

    share_left = np.zeros(len(METHODS), dtype=float)
    for field in COMPONENT_FIELDS:
        share_axis.barh(
            positions,
            shares[field],
            left=share_left,
            height=bar_height,
            color=COMPONENT_COLORS[field],
            edgecolor="white",
            linewidth=0.72,
            zorder=2,
        )
        share_left += shares[field]

    if not np.allclose(share_left, 100.0, atol=1e-5):
        raise RuntimeError("Relative latency components do not sum to 100%")

    for index, method in enumerate(METHODS):
        service_share = shares["service_loading_s"][index]
        waiting_share = shares["waiting_s"][index]
        processing_share = shares["computation_s"][index]
        transfer_share = sum(
            shares[field][index]
            for field in ("user_input_transfer_s", "dependency_transfer_s")
        )
        share_axis.text(
            transfer_share + service_share / 2.0,
            positions[index],
            f"{service_share:.1f}%",
            ha="center",
            va="center",
            fontsize=5.75,
            color="#252B2F",
            fontweight="bold" if method == "lean_our" else "normal",
            zorder=5,
        )
        # Only sufficiently wide segments receive in-bar labels. The exact
        # values remain available in the absolute panel without crowding.
        if waiting_share >= 7.0:
            share_axis.text(
                transfer_share + service_share + waiting_share / 2.0,
                positions[index],
                f"{waiting_share:.1f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color="white",
                zorder=5,
            )
        if processing_share >= 7.0:
            share_axis.text(
                100.0 - processing_share / 2.0,
                positions[index],
                f"{processing_share:.1f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color="white",
                zorder=5,
            )

    our_index = METHODS.index("lean_our")
    absolute_axis.add_patch(
        Rectangle(
            (0, our_index - bar_height / 2),
            totals_ms[our_index],
            bar_height,
            fill=False,
            edgecolor=OUR_COLOR,
            linewidth=1.05,
            zorder=6,
        )
    )
    share_axis.add_patch(
        Rectangle(
            (0, our_index - bar_height / 2),
            100,
            bar_height,
            fill=False,
            edgecolor=OUR_COLOR,
            linewidth=1.05,
            zorder=6,
        )
    )

    absolute_axis.set_yticks(positions)
    absolute_axis.set_yticklabels(
        [figure_style.DISPLAY[method] for method in METHODS]
    )
    absolute_axis.invert_yaxis()
    for tick, method in zip(absolute_axis.get_yticklabels(), METHODS):
        if method == "lean_our":
            tick.set_fontweight("bold")
            tick.set_color(OUR_COLOR)

    absolute_limit = float(np.max(totals_ms + ci_ms)) * 1.19
    absolute_axis.set_xlim(0, absolute_limit)
    absolute_axis.xaxis.set_major_locator(MultipleLocator(500))
    absolute_axis.set_xlabel("Critical-path latency (ms)")
    figure_style.style_axis(absolute_axis, grid="x")
    absolute_axis.text(
        0,
        1.025,
        "(a) Absolute latency",
        transform=absolute_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
    )

    share_axis.set_xlim(0, 100)
    share_axis.xaxis.set_major_locator(MultipleLocator(20))
    share_axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    share_axis.set_xlabel("Critical-path component share")
    share_axis.tick_params(axis="y", left=False, labelleft=False)
    figure_style.style_axis(share_axis, grid="x")
    share_axis.text(
        0,
        1.025,
        "(b) Relative composition",
        transform=share_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
    )

    handles, labels = absolute_axis.get_legend_handles_labels()
    selected = [COMPONENT_FIELDS.index(field) for field in DOMINANT_FIELDS]
    figure.legend(
        [handles[index] for index in selected],
        [SHORT_LABELS[field] for field in DOMINANT_FIELDS],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.985),
        frameon=False,
        columnspacing=1.15,
        handlelength=1.35,
    )
    figure.text(
        0.5,
        0.028,
        (
            "Labels in (a): service loading | total +/- 95% CI; "
            "W/P = waiting/processing. Transfers are at most 0.26 ms."
        ),
        ha="center",
        va="bottom",
        fontsize=5.15,
        color="#59636A",
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
            "Creator": "plot_latency_composition_paired_stacks_preview.py",
            "Title": "Absolute and relative critical-path latency breakdown",
        },
        provenance={
            "raw_data": [str(SOURCE.relative_to(ROOT))],
            "transformations": [
                "render absolute critical-path components as horizontal stacked bars",
                "render the same components as aligned normalized stacked bars",
            ],
            "uncertainty": "95% Student-t CI over ten independent seed-level means",
            "validation": "component sums and normalized shares are checked before export",
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
