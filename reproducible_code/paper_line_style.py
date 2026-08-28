#!/usr/bin/env python3
"""Shared publication style for every formal line chart in the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


MM = 1.0 / 25.4
DOUBLE_COLUMN = 180 * MM
LINE_FIGURE_HEIGHT = 78 * MM

# Keep method identity stable across every trend figure. Color is reinforced by
# marker and dash pattern so the plots remain interpretable in grayscale.
COLORS = {
    "random": "#C9CED2",
    "nearest": "#9BA6AC",
    "greedy": "#5F6A70",
    "dqn_wdsa_std_cache": "#B45F57",
    "discrete_sac_std_cache": "#9A7B62",
    "daoc_paper": "#7B708E",
    "daoc_our_coord_cache": "#5E897E",
    "our_flat_ddqn": "#4E8A74",
    "coord_cache_discrete_sac": "#5F8FA8",
    "lean_our": "#1F77A8",
}

MARKERS = {
    "random": "v",
    "nearest": "s",
    "greedy": "+",
    "dqn_wdsa_std_cache": "^",
    "discrete_sac_std_cache": "h",
    "daoc_paper": "o",
    "daoc_our_coord_cache": "D",
    "our_flat_ddqn": "<",
    "coord_cache_discrete_sac": "X",
    "lean_our": "*",
}

LINESTYLES = {
    "random": (0, (1.2, 1.6)),
    "nearest": (0, (4.0, 2.0)),
    "greedy": (0, (6.0, 1.8, 1.2, 1.8)),
    "dqn_wdsa_std_cache": (0, (2.0, 1.3)),
    "discrete_sac_std_cache": (0, (4.5, 1.5, 1.0, 1.5)),
    "daoc_paper": "-",
    "daoc_our_coord_cache": (0, (5.0, 1.7)),
    "our_flat_ddqn": (0, (6.0, 1.8, 1.2, 1.8)),
    "coord_cache_discrete_sac": (0, (3.0, 1.3)),
    "lean_our": "-",
}


def configure_style(style_path: Path | None = None) -> None:
    if style_path is not None and style_path.is_file():
        plt.style.use(str(style_path))
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.unicode_minus": False,
        }
    )


def line_kwargs(method: str, *, label: str | None = None) -> dict:
    is_ours = method == "lean_our"
    kwargs = {
        "color": COLORS[method],
        "linestyle": LINESTYLES[method],
        "linewidth": 1.55 if is_ours else 1.10,
        "marker": MARKERS[method],
        "markerfacecolor": COLORS[method] if is_ours else "white",
        "markeredgecolor": COLORS[method],
        "markeredgewidth": 0.80,
        "markersize": 6.0 if is_ours else 4.4,
        "zorder": 5 if is_ours else 3,
    }
    if label is not None:
        kwargs["label"] = label
    return kwargs


def errorbar_kwargs(method: str, *, label: str | None = None) -> dict:
    return {
        **line_kwargs(method, label=label),
        "elinewidth": 0.72,
        "capsize": 2.0,
        "capthick": 0.72,
    }


def style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#596168")
    axis.spines["bottom"].set_color("#596168")
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(direction="out", length=3.0, width=0.6, colors="#4B5359")
    axis.grid(
        axis=grid_axis,
        color="#D7DADD",
        linewidth=0.55,
        linestyle=(0, (3.0, 3.0)),
        alpha=0.90,
        zorder=0,
    )


def legend_above(axis, *, ncol: int, y: float = 1.015):
    return axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.35,
        columnspacing=1.0,
        handletextpad=0.55,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
