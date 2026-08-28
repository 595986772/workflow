#!/usr/bin/env python3
"""Render a normalized latency small-multiple dot plot preview."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_topconf_paper_figures as figure_style
from plot_latency_profile_preview import METRICS, METHODS, SOURCE, normalize, read_rows


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "paper_drafts" / "figures_topconf"
STEM = "fig03b_latency_dotmatrix_preview"

PALETTE = {
    "greedy": "#6F7A80",
    "daoc_paper": "#83788F",
    "discrete_sac_std_cache": "#AD7867",
    "daoc_our_coord_cache": "#668C87",
    "coord_cache_discrete_sac": "#7196AC",
    "lean_our": "#174F73",
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

    figure, axes = plt.subplots(
        1,
        len(METRICS),
        sharey=True,
        figsize=(figure_style.DOUBLE_COLUMN, 69 * figure_style.MM),
        gridspec_kw={"wspace": 0.11},
    )
    figure.subplots_adjust(left=0.205, right=0.985, top=0.79, bottom=0.21)

    y = np.arange(len(METHODS), dtype=float)
    row_labels = [figure_style.DISPLAY[method] for method in METHODS]

    for metric_index, ((key, label), axis) in enumerate(zip(METRICS, axes)):
        axis.set_xlim(0.0, 1.16)
        axis.set_ylim(len(METHODS) - 0.45, -0.55)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.set_title(label, pad=7, fontsize=6.6, fontweight="bold")
        axis.grid(axis="x", color="#D8DDE0", linewidth=0.60, alpha=0.95)
        axis.grid(axis="y", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#8A9499")
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelsize=5.6, pad=3)

        for row, method in enumerate(METHODS):
            value = values[method][key]
            is_our = method == "lean_our"
            axis.hlines(
                row,
                0.0,
                value,
                color=PALETTE[method],
                linewidth=1.75 if is_our else 1.0,
                alpha=0.95 if is_our else 0.62,
                zorder=2,
            )
            axis.scatter(
                value,
                row,
                s=35 if is_our else 22,
                marker=figure_style.MARKERS[method],
                facecolor=PALETTE[method] if is_our else "white",
                edgecolor=PALETTE[method],
                linewidth=0.9,
                zorder=4,
            )
            axis.text(
                min(value + 0.045, 1.075),
                row,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=5.2,
                color=PALETTE[method],
                fontweight="bold" if is_our else "normal",
            )

        axis.set_yticks(y)
        if metric_index == 0:
            axis.set_yticklabels(row_labels)
            for tick, method in zip(axis.get_yticklabels(), METHODS):
                tick.set_color(PALETTE[method])
                tick.set_fontweight("bold" if method == "lean_our" else "normal")
                tick.set_fontsize(5.8)
        else:
            axis.tick_params(axis="y", labelleft=False)

    figure.text(
        0.205,
        0.895,
        "Normalized latency (lower is better)",
        ha="left",
        va="center",
        fontsize=6.0,
        color="#58636A",
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
            "Creator": "plot_latency_dotmatrix_preview.py",
            "Title": "Normalized latency comparison by metric",
        },
        provenance={
            "raw_data": [str(SOURCE.relative_to(ROOT))],
            "derived_data": [str(data_path.relative_to(ROOT))],
            "transformations": [
                "normalize each latency metric by its maximum across methods",
                "show methods as aligned horizontal dot plots on a shared zero-to-one scale",
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
