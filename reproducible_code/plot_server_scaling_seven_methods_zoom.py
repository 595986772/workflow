#!/usr/bin/env python3
"""Render the seven-method server sweep on a complete continuous y axis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pegasus_server_scaling_heuristics_protocol import METHODS as HEURISTICS
from pegasus_server_scaling_protocol import CAPACITY_PROFILES, SERVER_COUNTS
from run_pegasus_server_scaling_heuristics import (
    ALL_METHODS,
    DISPLAY as INTERNAL_DISPLAY,
)
import paper_line_style as line_style


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = (
    ROOT
    / "results/pegasus_pscale/p13_server_scaling_heuristics/analysis"
    / "server_scaling_seven_methods_summary.json"
)
FIGURE_DIR = ROOT / "paper_drafts/figures_topconf"
STYLE_PATH = (
    ROOT.parent
    / ".codex/skills/scientific-visualization/assets/publication.mplstyle"
)
DISPLAY = {
    **INTERNAL_DISPLAY,
    "greedy": "SA-Nearest",
    "our_flat_ddqn": "DDQN + DCC",
    "coord_cache_discrete_sac": "SAC + DCC",
    "lean_our": "OUR",
}


def metric(summary: dict, method: str, field: str) -> np.ndarray:
    return np.asarray(
        [
            summary["results"][str(servers)]["method_summary"][method][
                "mean_completion_time_s"
            ][field]
            for servers in SERVER_COUNTS
        ],
        dtype=float,
    )


def draw_method(
    axis,
    summary: dict,
    method: str,
    *,
    show_error: bool = True,
):
    means = metric(summary, method, "mean")
    errors = metric(summary, method, "ci95_half_width")
    return axis.errorbar(
        np.asarray(SERVER_COUNTS, dtype=float),
        means,
        yerr=errors if show_error else None,
        **line_style.errorbar_kwargs(method, label=DISPLAY[method]),
    )


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    line_style.configure_style(STYLE_PATH)
    figure, axis = plt.subplots(
        figsize=(line_style.DOUBLE_COLUMN, line_style.LINE_FIGURE_HEIGHT),
        layout="constrained",
    )

    for method in ALL_METHODS:
        draw_method(axis, summary, method)

    full_upper = max(
        float(np.max(metric(summary, method, "mean") + metric(summary, method, "ci95_half_width")))
        for method in ALL_METHODS
    )
    y_step = 0.25
    y_upper = np.ceil((full_upper * 1.05) / y_step) * y_step
    axis.set_xlim(min(SERVER_COUNTS) - 0.7, max(SERVER_COUNTS) + 0.7)
    axis.set_ylim(0.0, y_upper)
    axis.set_yticks(np.arange(0.0, y_upper + y_step / 2.0, y_step))
    axis.set_xticks(SERVER_COUNTS)
    axis.set_xticklabels(
        [
            f"{servers}\n(B={sum(CAPACITY_PROFILES[servers])})"
            for servers in SERVER_COUNTS
        ]
    )
    axis.set_xlabel("Number of edge servers")
    axis.set_ylabel("Mean DAG completion time (s)")
    line_style.style_axis(axis, grid_axis="y")
    line_style.legend_above(axis, ncol=4)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "fig10_server_count_sensitivity_7methods",
        "fig10_server_count_sensitivity",
    ):
        for suffix in ("pdf", "svg", "png"):
            figure.savefig(
                FIGURE_DIR / f"{name}.{suffix}",
                dpi=400 if suffix == "png" else None,
                bbox_inches=None,
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
