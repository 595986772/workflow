#!/usr/bin/env python3
"""Plot the preregistered online-training curves for the 26k comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

import paper_line_style as line_style

from pegasus_common_horizon_protocol import (
    ANALYSIS_DIR,
    FINAL_DIR,
    FINAL_SEEDS,
    FIXED_TRAIN_EPISODES,
    P3_FINAL_DIR,
    P6_LEARNING_DIR,
    P8_FINAL_DIR,
    TAIL_START_EPISODE,
)


HEURISTIC_ROOT = (
    Path(__file__).resolve().parent
    / "results/pegasus_pscale/p6_baselines_ablation/heuristics"
)
HEURISTICS = ("random", "nearest", "greedy")
METHODS = (
    "dqn_wdsa_std_cache",
    "discrete_sac_std_cache",
    "daoc_paper",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
PANELS = (
    (
        "Standard-cache schedulers",
        (
            "dqn_wdsa_std_cache",
            "discrete_sac_std_cache",
            "daoc_paper",
        ),
    ),
    (
        "Coordinated-cache schedulers",
        (
            "daoc_paper",
            "daoc_our_coord_cache",
            "coord_cache_discrete_sac",
            "lean_our",
        ),
    ),
)
RUN_ROOTS = {
    "daoc_paper": P3_FINAL_DIR,
    "dqn_wdsa_std_cache": P6_LEARNING_DIR,
    "daoc_our_coord_cache": P8_FINAL_DIR,
    "discrete_sac_std_cache": FINAL_DIR,
    "coord_cache_discrete_sac": FINAL_DIR,
    "lean_our": FINAL_DIR,
}
DISPLAY_NAMES = {
    "random": "Random",
    "nearest": "Nearest",
    "greedy": "SA-Nearest",
    "daoc_paper": "DAOC",
    "dqn_wdsa_std_cache": "DQN-NoDSA",
    "daoc_our_coord_cache": "DAOC + DCC",
    "discrete_sac_std_cache": "SAC + DAOC-Cache",
    "coord_cache_discrete_sac": "SAC + DCC",
    "lean_our": "OUR",
}
# Color and line style are both varied so the figure remains readable in grayscale.
STYLES = {
    "random": ("#A5A5A5", (0, (1, 1))),
    "nearest": ("#7A9E9F", (0, (4, 2))),
    "greedy": ("#8C7A5B", (0, (6, 2, 1, 2))),
    "daoc_paper": ("#4C78A8", "--"),
    "dqn_wdsa_std_cache": ("#79706E", ":"),
    "daoc_our_coord_cache": ("#ECA82C", "-."),
    "discrete_sac_std_cache": ("#B279A2", (0, (5, 2))),
    "coord_cache_discrete_sac": ("#D65F45", (0, (3, 1, 1, 1))),
    "lean_our": ("#238B45", "-"),
}
COMBINED_ORDER = (
    "dqn_wdsa_std_cache",
    "discrete_sac_std_cache",
    "daoc_paper",
    "coord_cache_discrete_sac",
    "daoc_our_coord_cache",
    "lean_our",
    "random",
    "nearest",
    "greedy",
)
MEASURED_EXTENSION_METHODS = (
    "discrete_sac_std_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
MEASURED_EXTENSION_START = 10_500
REFERENCE_DISPLAY = {
    **DISPLAY_NAMES,
}
REFERENCE_COLORS = {
    "random": "#C2C7CF",
    "nearest": "#AEB5BF",
    "greedy": "#747F8D",
    "dqn_wdsa_std_cache": "#8AB6D6",
    "discrete_sac_std_cache": "#4C88B7",
    "daoc_paper": "#24557A",
    "daoc_our_coord_cache": "#D99A2B",
    "coord_cache_discrete_sac": "#2A8F7B",
    "lean_our": "#D34A4A",
}
REFERENCE_MARKERS = {
    "random": "v",
    "nearest": "s",
    "greedy": "P",
    "dqn_wdsa_std_cache": "^",
    "discrete_sac_std_cache": "h",
    "daoc_paper": "o",
    "daoc_our_coord_cache": "P",
    "coord_cache_discrete_sac": "X",
    "lean_our": "*",
}
REFERENCE_LINESTYLES = {
    "random": (0, (1.2, 1.5)),
    "nearest": (0, (4.0, 1.8)),
    "greedy": (0, (6.0, 1.8, 1.2, 1.8)),
    "dqn_wdsa_std_cache": (0, (2.0, 1.2)),
    "discrete_sac_std_cache": (0, (4.0, 1.4, 1.0, 1.4)),
    "daoc_paper": "-",
    "daoc_our_coord_cache": (0, (5.0, 1.5)),
    "coord_cache_discrete_sac": (0, (3.0, 1.2)),
    "lean_our": "-",
}

# DAOC Fig. 4 uses a single boxed axis, light Cartesian grid, and redundant
# line/marker encodings. Keep that structure while using a colorblind-safer
# palette and placing the larger comparison legend outside the data region.
DAOC_STYLE_ORDER = (
    "random",
    "nearest",
    "greedy",
    "dqn_wdsa_std_cache",
    "discrete_sac_std_cache",
    "daoc_paper",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
DAOC_STYLE_COLORS = {
    "random": "#A0A0A0",
    "nearest": "#697681",
    "greedy": "#242A2E",
    "dqn_wdsa_std_cache": "#7656A3",
    "discrete_sac_std_cache": "#16856F",
    "daoc_paper": "#2F6FA3",
    "daoc_our_coord_cache": "#C45A3C",
    "coord_cache_discrete_sac": "#A64D79",
    "lean_our": "#A97800",
}
DAOC_STYLE_MARKERS = {
    "random": "^",
    "nearest": "x",
    "greedy": "+",
    "dqn_wdsa_std_cache": "P",
    "discrete_sac_std_cache": "h",
    "daoc_paper": "s",
    "daoc_our_coord_cache": "v",
    "coord_cache_discrete_sac": "D",
    "lean_our": "o",
}
DAOC_STYLE_LINESTYLES = {
    "random": (0, (1.2, 1.8)),
    "nearest": (0, (5.0, 2.0)),
    "greedy": (0, (8.0, 2.0, 1.5, 2.0)),
    "dqn_wdsa_std_cache": (0, (3.0, 1.5)),
    "discrete_sac_std_cache": (0, (6.0, 1.7, 1.2, 1.7)),
    "daoc_paper": "-",
    "daoc_our_coord_cache": (0, (5.0, 1.6)),
    "coord_cache_discrete_sac": (0, (6.0, 1.6, 1.2, 1.6)),
    "lean_our": "-",
}

# A saturated line/marker mapping matching the supplied DAOC-style reference.
REFERENCE_FIGURE_COLORS = {
    "random": "#87058C",
    "nearest": "#00D6DC",
    "greedy": "#111111",
    "dqn_wdsa_std_cache": "#078C12",
    "discrete_sac_std_cache": "#F000D7",
    "daoc_paper": "#F01818",
    "daoc_our_coord_cache": "#F5A000",
    "coord_cache_discrete_sac": "#A92E2E",
    "lean_our": "#1515E8",
}
REFERENCE_FIGURE_MARKERS = {
    "random": "^",
    "nearest": "x",
    "greedy": "s",
    "dqn_wdsa_std_cache": "s",
    "discrete_sac_std_cache": "*",
    "daoc_paper": "D",
    "daoc_our_coord_cache": "v",
    "coord_cache_discrete_sac": "P",
    "lean_our": "o",
}

# figures4papers semantic palette: proposed method in anchor blue, heuristic
# references in neutrals, and learning alternatives in distinct muted hues.
FIGURES4PAPERS_COLORS = {
    "random": "#CFCECE",
    "nearest": "#8F9498",
    "greedy": "#4D4D4D",
    "dqn_wdsa_std_cache": "#B64342",
    "discrete_sac_std_cache": "#42949E",
    "daoc_paper": "#9A4D8E",
    "daoc_our_coord_cache": "#5F9561",
    "coord_cache_discrete_sac": "#B47A32",
    "lean_our": "#0F4D92",
}
FIGURES4PAPERS_MARKERS = {
    "random": "^",
    "nearest": "x",
    "greedy": "+",
    "dqn_wdsa_std_cache": "s",
    "discrete_sac_std_cache": "D",
    "daoc_paper": "P",
    "daoc_our_coord_cache": "v",
    "coord_cache_discrete_sac": "h",
    "lean_our": "o",
}
FIGURES4PAPERS_LINESTYLES = {
    "random": (0, (1.5, 1.8)),
    "nearest": (0, (5.0, 2.0)),
    "greedy": (0, (7.0, 2.0, 1.2, 2.0)),
    "dqn_wdsa_std_cache": (0, (3.5, 1.4)),
    "discrete_sac_std_cache": (0, (6.0, 1.6, 1.2, 1.6)),
    "daoc_paper": "-",
    "daoc_our_coord_cache": (0, (5.0, 1.6)),
    "coord_cache_discrete_sac": (0, (7.0, 1.6, 1.2, 1.6)),
    "lean_our": "-",
}

# Nature-inspired editorial palette. Heuristics use neutral blue-grays,
# standard-cache learners use muted warm/violet hues, coordinated controls use
# subdued blue-greens, and OUR is the sole saturated editorial blue.
TOPCONF_COLORS = {
    "random": "#C9CED2",
    "nearest": "#9BA6AC",
    "greedy": "#5F6A70",
    "dqn_wdsa_std_cache": "#B45F57",
    "discrete_sac_std_cache": "#9A7B62",
    "daoc_paper": "#7B708E",
    "daoc_our_coord_cache": "#5E897E",
    "coord_cache_discrete_sac": "#5F8FA8",
    "lean_our": "#1F77A8",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth-window", type=int, default=500)
    parser.add_argument("--plot-stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dir(method, seed):
    return RUN_ROOTS[method] / "runs" / method / f"seed_{seed}"


def collect_heuristic_levels():
    levels = {}
    sources = []
    for method in HEURISTICS:
        seed_values = []
        for seed in FINAL_SEEDS:
            path = HEURISTIC_ROOT / "runs" / method / f"seed_{seed}" / "episodes.csv"
            rows = []
            with path.open(newline="", encoding="utf-8") as input_file:
                for row in csv.DictReader(input_file):
                    episode = int(row["episode"])
                    if row["phase"] == "train" and 4001 <= episode <= 5000:
                        rows.append(row)
            if len(rows) != 1000:
                raise RuntimeError(f"Invalid heuristic tail in {path}: {len(rows)}")
            seed_values.append(
                float(np.mean([float(row["average_finish_time"]) for row in rows]))
            )
            sources.append(
                {
                    "method": method,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "online_tail": [4001, 5000],
                }
            )
        values = np.asarray(seed_values, dtype=float)
        half_width = (
            stats.t.ppf(0.975, len(FINAL_SEEDS) - 1)
            * values.std(ddof=1)
            / np.sqrt(len(FINAL_SEEDS))
        )
        levels[method] = {
            "mean": float(values.mean()),
            "lower": float(values.mean() - half_width),
            "upper": float(values.mean() + half_width),
            "seed_values": values,
        }
    return levels, sources


def read_training_series(path):
    episodes = []
    latency = []
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row["phase"] != "train":
                continue
            episodes.append(int(row["episode"]))
            latency.append(float(row["average_finish_time"]))
    expected = list(range(1, FIXED_TRAIN_EPISODES + 1))
    if episodes != expected:
        raise RuntimeError(
            f"Training episodes are incomplete or unordered in {path}: "
            f"found {len(episodes)}, expected {FIXED_TRAIN_EPISODES}"
        )
    return np.asarray(latency, dtype=float)


def trailing_mean(values, window):
    if window < 1 or window > len(values):
        raise ValueError(f"Invalid smoothing window: {window}")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return (cumulative[window:] - cumulative[:-window]) / window


def collect_curves(window, stride):
    if stride < 1:
        raise ValueError("plot-stride must be positive")
    first_episode = window
    all_episodes = np.arange(
        first_episode,
        FIXED_TRAIN_EPISODES + 1,
        dtype=int,
    )
    indices = np.arange(0, len(all_episodes), stride, dtype=int)
    if indices[-1] != len(all_episodes) - 1:
        indices = np.append(indices, len(all_episodes) - 1)
    episodes = all_episodes[indices]
    curves = {}
    sources = []
    for method in METHODS:
        seed_curves = []
        for seed in FINAL_SEEDS:
            path = run_dir(method, seed) / "episodes.csv"
            series = read_training_series(path)
            seed_curves.append(trailing_mean(series, window)[indices])
            sources.append(
                {
                    "method": method,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
        values = np.vstack(seed_curves)
        mean = values.mean(axis=0)
        half_width = (
            stats.t.ppf(0.975, len(FINAL_SEEDS) - 1)
            * values.std(axis=0, ddof=1)
            / np.sqrt(len(FINAL_SEEDS))
        )
        curves[method] = {
            "mean": mean,
            "lower": mean - half_width,
            "upper": mean + half_width,
            "seed_values": values,
        }
    return episodes, curves, sources


def write_source_data(path, episodes, curves):
    fields = (
        "episode",
        "method",
        "seed_count",
        "smoothed_mean",
        "ci95_lower",
        "ci95_upper",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            curve = curves[method]
            for index, episode in enumerate(episodes):
                writer.writerow(
                    {
                        "episode": int(episode),
                        "method": method,
                        "seed_count": len(FINAL_SEEDS),
                        "smoothed_mean": float(curve["mean"][index]),
                        "ci95_lower": float(curve["lower"][index]),
                        "ci95_upper": float(curve["upper"][index]),
                    }
                )


def draw_figure(output_dir, episodes, curves, heuristic_levels, window):
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.1,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.35),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        for panel_index, (axis, panel) in enumerate(zip(axes, PANELS)):
            title, methods = panel
            axis.axvspan(
                TAIL_START_EPISODE,
                FIXED_TRAIN_EPISODES,
                color="#E6E6E6",
                alpha=0.65,
                linewidth=0,
                zorder=0,
            )
            for method in methods:
                color, line_style = STYLES[method]
                curve = curves[method]
                line_width = 2.0 if method == "lean_our" else 1.35
                zorder = 4 if method == "lean_our" else 2
                axis.fill_between(
                    episodes,
                    curve["lower"],
                    curve["upper"],
                    color=color,
                    alpha=0.10 if method != "lean_our" else 0.16,
                    linewidth=0,
                    zorder=zorder - 1,
                )
                axis.plot(
                    episodes,
                    curve["mean"],
                    color=color,
                    linestyle=line_style,
                    linewidth=line_width,
                    label=DISPLAY_NAMES[method],
                    zorder=zorder,
                )
            if panel_index == 0:
                for method in HEURISTICS:
                    color, line_style = STYLES[method]
                    level = heuristic_levels[method]
                    axis.axhline(
                        level["mean"],
                        color=color,
                        linestyle=line_style,
                        linewidth=1.15,
                        label=DISPLAY_NAMES[method],
                        zorder=1,
                    )
            axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left")
            axis.set_xlabel("Training episode")
            axis.set_xlim(0, FIXED_TRAIN_EPISODES)
            axis.set_xticks(np.arange(0, 25001, 5000))
            axis.set_xticklabels(["0", "5k", "10k", "15k", "20k", "25k"])
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(
                frameon=False,
                loc="upper right",
                handlelength=2.7,
                ncol=2 if panel_index == 0 else 1,
                columnspacing=0.9,
            )
        axes[0].set_ylabel("Mean DAG completion time (s)")
        axes[1].text(
            25500,
            axes[1].get_ylim()[1],
            "reported tail",
            ha="right",
            va="top",
            color="#555555",
            fontsize=7.0,
        )
        figure.suptitle(
            f"Online scheduling performance ({window}-episode trailing mean; "
            "shading: 95% CI across 10 seeds)",
            fontsize=9.2,
        )
        for suffix, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"online_training_curves.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def draw_tail_zoom(output_dir, episodes, curves, window):
    methods = (
        "daoc_paper",
        "coord_cache_discrete_sac",
        "daoc_our_coord_cache",
        "lean_our",
    )
    start_episode = 20000
    mask = episodes >= start_episode
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(
            figsize=(3.55, 2.75),
            layout="constrained",
        )
        axis.axvspan(
            TAIL_START_EPISODE,
            FIXED_TRAIN_EPISODES,
            color="#E6E6E6",
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        for method in methods:
            color, line_style = STYLES[method]
            curve = curves[method]
            line_width = 2.0 if method == "lean_our" else 1.35
            zorder = 4 if method == "lean_our" else 2
            axis.fill_between(
                episodes[mask],
                curve["lower"][mask],
                curve["upper"][mask],
                color=color,
                alpha=0.10 if method != "lean_our" else 0.16,
                linewidth=0,
                zorder=zorder - 1,
            )
            axis.plot(
                episodes[mask],
                curve["mean"][mask],
                color=color,
                linestyle=line_style,
                linewidth=line_width,
                label=DISPLAY_NAMES[method],
                zorder=zorder,
            )
        axis.set(
            xlim=(start_episode, FIXED_TRAIN_EPISODES),
            xlabel="Training episode",
            ylabel="Mean DAG completion time (s)",
        )
        axis.set_xticks((20000, 22000, 24000, 26000))
        axis.set_xticklabels(("20k", "22k", "24k", "26k"))
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper right", handlelength=2.7)
        axis.text(
            25900,
            axis.get_ylim()[1],
            "reported tail",
            ha="right",
            va="top",
            color="#555555",
            fontsize=7.0,
        )
        for suffix, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"online_training_tail_zoom.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def draw_combined_figure(output_dir, episodes, curves, heuristic_levels, window):
    """Render one publication-width axis containing every comparison method."""
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.15, 3.65))
        figure.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.855)

        axis.axvspan(
            TAIL_START_EPISODE,
            FIXED_TRAIN_EPISODES,
            color="#E8E8E8",
            alpha=0.72,
            linewidth=0,
            zorder=0,
        )

        # Heuristics have no trainable policy, so their online tails are references.
        for method in HEURISTICS:
            color, line_style = STYLES[method]
            axis.axhline(
                heuristic_levels[method]["mean"],
                color=color,
                linestyle=line_style,
                linewidth=1.05,
                alpha=0.92,
                zorder=1,
            )

        for method in METHODS:
            color, line_style = STYLES[method]
            curve = curves[method]
            is_ours = method == "lean_our"
            line_width = 2.25 if is_ours else 1.35
            zorder = 5 if is_ours else 3
            axis.fill_between(
                episodes,
                curve["lower"],
                curve["upper"],
                color=color,
                alpha=0.14 if is_ours else 0.065,
                linewidth=0,
                zorder=zorder - 1,
            )
            axis.plot(
                episodes,
                curve["mean"],
                color=color,
                linestyle=line_style,
                linewidth=line_width,
                solid_capstyle="round",
                zorder=zorder,
            )

        axis.set_xlim(0, FIXED_TRAIN_EPISODES)
        axis.set_ylim(0, 2.05)
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Mean DAG completion time (s)")
        axis.set_xticks((0, 5000, 10000, 15000, 20000, 26000))
        axis.set_xticklabels(("0", "5k", "10k", "15k", "20k", "26k"))
        axis.set_yticks(np.arange(0, 2.01, 0.25))
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.82)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3.2, width=0.7)
        axis.text(
            25800,
            2.015,
            "Reporting window",
            ha="right",
            va="top",
            color="#555555",
            fontsize=7.2,
        )

        handles = []
        for method in COMBINED_ORDER:
            color, line_style = STYLES[method]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=line_style,
                    linewidth=(2.25 if method == "lean_our" else 1.35),
                    label=DISPLAY_NAMES[method],
                )
            )
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            frameon=False,
            handlelength=3.0,
            handletextpad=0.6,
            columnspacing=1.35,
            borderaxespad=0,
        )

        for suffix, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"online_training_curves_combined.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def measured_extension_summary(episodes, curves):
    episode_to_index = {
        int(episode): index for index, episode in enumerate(episodes)
    }
    start_index = episode_to_index[MEASURED_EXTENSION_START]
    end_index = episode_to_index[FIXED_TRAIN_EPISODES]
    summary = {}
    for method in MEASURED_EXTENSION_METHODS:
        seed_values = curves[method]["seed_values"]
        starts = seed_values[:, start_index]
        ends = seed_values[:, end_index]
        changes = ends - starts
        change_mean = float(changes.mean())
        change_half = float(
            stats.t.ppf(0.975, len(FINAL_SEEDS) - 1)
            * changes.std(ddof=1)
            / np.sqrt(len(FINAL_SEEDS))
        )
        summary[method] = {
            "start_episode": MEASURED_EXTENSION_START,
            "end_episode": FIXED_TRAIN_EPISODES,
            "start_mean_seconds": float(starts.mean()),
            "end_mean_seconds": float(ends.mean()),
            "relative_change_percent": float(
                100.0 * (ends.mean() - starts.mean()) / starts.mean()
            ),
            "paired_change_seconds": change_mean,
            "paired_change_ci95_seconds": [
                change_mean - change_half,
                change_mean + change_half,
            ],
        }
    return summary


def write_measured_extension_source(path, episodes, curves, heuristic_levels):
    block_mask = episodes % 500 == 0
    block_episodes = episodes[block_mask]
    fields = (
        "episode",
        "method",
        "segment",
        "seed_count",
        "mean_finish_time",
        "ci95_lower",
        "ci95_upper",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            curve = curves[method]
            means = curve["mean"][block_mask]
            lowers = curve["lower"][block_mask]
            uppers = curve["upper"][block_mask]
            for episode, mean, lower, upper in zip(
                block_episodes,
                means,
                lowers,
                uppers,
            ):
                writer.writerow(
                    {
                        "episode": int(episode),
                        "method": method,
                        "segment": "measured_full_horizon",
                        "seed_count": len(FINAL_SEEDS),
                        "mean_finish_time": float(mean),
                        "ci95_lower": float(lower),
                        "ci95_upper": float(upper),
                    }
                )
        for method in HEURISTICS:
            level = heuristic_levels[method]
            for episode in block_episodes:
                writer.writerow(
                    {
                        "episode": int(episode),
                        "method": method,
                        "segment": "non_learning_reference",
                        "seed_count": len(FINAL_SEEDS),
                        "mean_finish_time": level["mean"],
                        "ci95_lower": level["lower"],
                        "ci95_upper": level["upper"],
                    }
                )


def draw_measured_extension_figure(
    output_dir,
    episodes,
    curves,
    heuristic_levels,
):
    """Match the earlier reported-extension figure using measured 26k tails."""
    block_mask = episodes % 500 == 0
    block_episodes = episodes[block_mask]
    block_x = block_episodes / 1000.0
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.6,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.15, 3.55))
        figure.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.96)

        reference_x = np.linspace(0.0, 26.0, 261)
        for method in HEURISTICS:
            axis.plot(
                reference_x,
                np.full_like(reference_x, heuristic_levels[method]["mean"]),
                color=REFERENCE_COLORS[method],
                linestyle=REFERENCE_LINESTYLES[method],
                linewidth=1.05,
                marker=REFERENCE_MARKERS[method],
                markevery=26,
                markerfacecolor=REFERENCE_COLORS[method],
                markeredgecolor="#303030",
                markeredgewidth=0.35,
                markersize=3.1,
                label=REFERENCE_DISPLAY[method],
                zorder=1,
            )

        for method in METHODS:
            curve = curves[method]
            means = curve["mean"][block_mask]
            lows = curve["lower"][block_mask]
            highs = curve["upper"][block_mask]
            color = REFERENCE_COLORS[method]
            linewidth = 1.7 if method == "lean_our" else 1.15
            if method in MEASURED_EXTENSION_METHODS:
                axis.fill_between(
                    block_x,
                    lows,
                    highs,
                    color=color,
                    alpha=0.07 if method != "lean_our" else 0.11,
                    linewidth=0,
                    zorder=2,
                )
                axis.plot(
                    block_x,
                    means,
                    color=color,
                    linestyle="-",
                    linewidth=linewidth,
                    marker=REFERENCE_MARKERS[method],
                    markevery=4,
                    markerfacecolor=(
                        color if method == "lean_our" else "white"
                    ),
                    markeredgecolor="#303030",
                    markeredgewidth=0.4,
                    markersize=3.5,
                    label=REFERENCE_DISPLAY[method],
                    zorder=4,
                )
            else:
                axis.plot(
                    block_x,
                    means,
                    color=color,
                    linestyle=REFERENCE_LINESTYLES[method],
                    linewidth=linewidth,
                    marker=REFERENCE_MARKERS[method],
                    markevery=4,
                    markerfacecolor="white",
                    markeredgecolor="#303030",
                    markeredgewidth=0.4,
                    markersize=3.4,
                    label=REFERENCE_DISPLAY[method],
                    zorder=3,
                )

        axis.set_xlim(0, 26)
        axis.set_ylim(0, 2.05)
        axis.set_xlabel(r"Training episode ($\times 10^3$)")
        axis.set_ylabel("Online DAG completion time (s)")
        axis.set_xticks((0, 5, 10, 15, 20, 25))
        axis.set_yticks(np.arange(0, 2.01, 0.25))
        axis.grid(
            axis="both",
            color="#D8D8D8",
            linewidth=0.55,
            linestyle="-",
        )
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#606060")
            spine.set_linewidth(0.6)
        axis.legend(
            ncol=2,
            loc="upper right",
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor="#D6D6D6",
            columnspacing=1.0,
            handlelength=2.5,
        )
        for suffix, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"online_training_curves_measured_extension.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def draw_daoc_style_comparison(
    output_dir,
    episodes,
    curves,
    heuristic_levels,
):
    """Render the measured comparison with the visual structure of DAOC Fig. 4."""
    dense_x = episodes / 1000.0
    episode_step = float(np.median(np.diff(episodes)))
    # Markers remain readable while the lines retain every measured smoothed
    # observation. This targets one marker per 1,000 training episodes.
    marker_stride = max(1, int(round(2000.0 / episode_step)))
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.15, 3.45))
        # Reserve a dedicated strip at the upper right. The legend remains
        # visually attached to the chart but never overlaps the data region.
        figure.subplots_adjust(left=0.105, right=0.665, bottom=0.175, top=0.96)

        for method in HEURISTICS:
            color = DAOC_STYLE_COLORS[method]
            marker = DAOC_STYLE_MARKERS[method]
            level = heuristic_levels[method]["mean"]
            axis.plot(
                dense_x,
                np.full_like(dense_x, level),
                color=color,
                linestyle=DAOC_STYLE_LINESTYLES[method],
                linewidth=1.25,
                marker=marker,
                markevery=marker_stride,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.75,
                markersize=2.8,
                label=REFERENCE_DISPLAY[method],
                zorder=2,
                solid_capstyle="round",
                dash_capstyle="round",
                antialiased=True,
            )

        for method in METHODS:
            color = DAOC_STYLE_COLORS[method]
            marker = DAOC_STYLE_MARKERS[method]
            curve = curves[method]
            means = curve["mean"]
            is_ours = method == "lean_our"
            axis.plot(
                dense_x,
                means,
                color=color,
                linestyle=DAOC_STYLE_LINESTYLES[method],
                linewidth=2.3 if is_ours else 1.4,
                marker=marker,
                markevery=marker_stride,
                markerfacecolor=color,
                markeredgecolor=("#242424" if is_ours else "white"),
                markeredgewidth=0.65,
                markersize=4.2 if is_ours else 3.3,
                label=REFERENCE_DISPLAY[method],
                zorder=5 if is_ours else 3,
                solid_capstyle="round",
                dash_capstyle="round",
                antialiased=True,
            )

        axis.set_xlim(0, 26)
        axis.set_ylim(0, 2.05)
        axis.set_xlabel(r"Training episode ($\times 10^3$)")
        axis.set_ylabel("Application finishing time (s)")
        axis.set_xticks((0, 5, 10, 15, 20, 25))
        axis.set_yticks(np.arange(0, 2.01, 0.25))
        axis.grid(axis="y", color="#E0E2E4", linewidth=0.5, alpha=0.9)
        axis.grid(axis="x", color="#EFF0F1", linewidth=0.4, alpha=0.75)
        axis.set_axisbelow(True)
        axis.tick_params(
            direction="out",
            length=3.0,
            width=0.65,
            colors="#30343A",
        )
        for name, spine in axis.spines.items():
            spine.set_visible(True)
            spine.set_color("#363B41" if name in ("left", "bottom") else "#AEB2B6")
            spine.set_linewidth(0.72 if name in ("left", "bottom") else 0.5)

        handles, labels = axis.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        ordered_labels = [REFERENCE_DISPLAY[method] for method in DAOC_STYLE_ORDER]
        figure.legend(
            [handle_by_label[label] for label in ordered_labels],
            ordered_labels,
            loc="upper left",
            bbox_to_anchor=(0.69, 0.94),
            ncol=1,
            title="METHODS",
            title_fontproperties={"size": 7.0, "weight": "semibold"},
            frameon=False,
            handlelength=2.4,
            handletextpad=0.6,
            borderpad=0.0,
            labelspacing=0.72,
            alignment="left",
        )

        for suffix, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"online_training_curves_daoc_style.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def draw_reference_style_comparison(
    output_dir,
    episodes,
    curves,
    heuristic_levels,
):
    """Render a separate version matching the supplied saturated reference."""
    dense_x = episodes.astype(float)
    episode_step = float(np.median(np.diff(episodes)))
    marker_stride = max(1, int(round(2000.0 / episode_step)))
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 11.2,
        "legend.fontsize": 8.7,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(8.25, 4.65))
        # The reference uses a large upper-right legend. Keep the same visual
        # weight in a reserved right strip so it cannot hide high-latency lines.
        figure.subplots_adjust(left=0.095, right=0.665, bottom=0.16, top=0.97)

        for method in HEURISTICS:
            color = REFERENCE_FIGURE_COLORS[method]
            marker = REFERENCE_FIGURE_MARKERS[method]
            axis.plot(
                dense_x,
                np.full_like(dense_x, heuristic_levels[method]["mean"]),
                color=color,
                linestyle="-",
                linewidth=1.25,
                marker=marker,
                markevery=marker_stride,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.65,
                markersize=6.0,
                label=REFERENCE_DISPLAY[method],
                zorder=2,
            )

        for method in METHODS:
            color = REFERENCE_FIGURE_COLORS[method]
            marker = REFERENCE_FIGURE_MARKERS[method]
            is_ours = method == "lean_our"
            axis.plot(
                dense_x,
                curves[method]["mean"],
                color=color,
                linestyle="-",
                linewidth=1.65 if is_ours else 1.25,
                marker=marker,
                markevery=marker_stride,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.65,
                markersize=7.0 if is_ours else 6.0,
                label=REFERENCE_DISPLAY[method],
                zorder=5 if is_ours else 3,
            )

        axis.set_xlim(0, FIXED_TRAIN_EPISODES)
        axis.set_ylim(0, 2.05)
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Application finishing time (s)")
        axis.set_xticks((0, 5000, 10000, 15000, 20000, 25000))
        axis.set_yticks(np.arange(0, 2.01, 0.25))
        axis.tick_params(direction="out", length=3.5, width=0.8, colors="#111111")
        axis.grid(
            axis="both",
            color="#B9B9B9",
            linewidth=0.75,
            linestyle="-",
            alpha=0.9,
        )
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(0.8)

        handles, labels = axis.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        ordered_labels = [REFERENCE_DISPLAY[method] for method in DAOC_STYLE_ORDER]
        legend = figure.legend(
            [handle_by_label[label] for label in ordered_labels],
            ordered_labels,
            loc="upper left",
            bbox_to_anchor=(0.68, 0.95),
            ncol=1,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#C8C8C8",
            fancybox=True,
            handlelength=2.8,
            handletextpad=0.75,
            borderpad=0.55,
            labelspacing=0.7,
        )
        legend.get_frame().set_linewidth(0.8)

        for suffix, options in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            figure.savefig(
                output_dir / f"online_training_curves_reference_style.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def draw_figures4papers_comparison(
    output_dir,
    episodes,
    curves,
    heuristic_levels,
):
    """Render a figures4papers house-style comparison with a legend panel."""
    dense_x = episodes / 1000.0
    episode_step = float(np.median(np.diff(episodes)))
    marker_stride = max(1, int(round(2500.0 / episode_step)))
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.2,
        "axes.labelsize": 10.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "axes.linewidth": 1.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, (axis, legend_axis) = plt.subplots(
            1,
            2,
            figsize=(8.2, 3.55),
            gridspec_kw={"width_ratios": (4.5, 1.65), "wspace": 0.04},
        )
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.97)

        for method in HEURISTICS:
            color = FIGURES4PAPERS_COLORS[method]
            marker = FIGURES4PAPERS_MARKERS[method]
            axis.plot(
                dense_x,
                np.full_like(dense_x, heuristic_levels[method]["mean"]),
                color=color,
                linestyle=FIGURES4PAPERS_LINESTYLES[method],
                linewidth=1.75,
                marker=marker,
                markevery=marker_stride,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=3.6,
                label=REFERENCE_DISPLAY[method],
                zorder=2,
                solid_capstyle="round",
                dash_capstyle="round",
            )

        for method in METHODS:
            color = FIGURES4PAPERS_COLORS[method]
            marker = FIGURES4PAPERS_MARKERS[method]
            is_ours = method == "lean_our"
            axis.plot(
                dense_x,
                curves[method]["mean"],
                color=color,
                linestyle=FIGURES4PAPERS_LINESTYLES[method],
                linewidth=3.0 if is_ours else 2.0,
                marker=marker,
                markevery=marker_stride,
                markerfacecolor=color if is_ours else "white",
                markeredgecolor="#272727" if is_ours else color,
                markeredgewidth=0.85,
                markersize=4.8 if is_ours else 3.7,
                label=REFERENCE_DISPLAY[method],
                zorder=5 if is_ours else 3,
                solid_capstyle="round",
                dash_capstyle="round",
            )

        axis.set_xlim(0, 26)
        axis.set_ylim(0, 2.05)
        axis.set_xlabel(r"Training episode ($\times 10^3$)")
        axis.set_ylabel("Application finishing time (s)")
        axis.set_xticks((0, 5, 10, 15, 20, 25))
        axis.set_yticks(np.arange(0, 2.01, 0.25))
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.65, alpha=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=3.5, width=1.0, color="#272727")
        axis.spines["left"].set_color("#272727")
        axis.spines["bottom"].set_color("#272727")
        axis.spines["left"].set_linewidth(1.35)
        axis.spines["bottom"].set_linewidth(1.35)

        handles, labels = axis.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        ordered_labels = [REFERENCE_DISPLAY[method] for method in DAOC_STYLE_ORDER]
        legend_axis.set_axis_off()
        legend_axis.legend(
            [handle_by_label[label] for label in ordered_labels],
            ordered_labels,
            loc="center left",
            title="METHODS",
            title_fontproperties={"size": 8.0, "weight": "bold"},
            handlelength=2.8,
            handletextpad=0.7,
            labelspacing=0.8,
            borderaxespad=0.0,
            alignment="left",
        )

        for suffix, options in (
            ("svg", {}),
            ("pdf", {}),
            ("png", {"dpi": 400}),
        ):
            figure.savefig(
                output_dir / f"online_training_curves_figures4papers.{suffix}",
                facecolor="white",
                **options,
            )
        plt.close(figure)


def draw_topconf_comparison(
    output_dir,
    episodes,
    curves,
    heuristic_levels,
    output_stem="online_training_curves_topconf",
):
    """Render the formal convergence chart with the shared line-figure style."""
    dense_x = episodes / 1000.0
    episode_step = float(np.median(np.diff(episodes)))
    marker_stride = max(1, int(round(2500.0 / episode_step)))
    line_style.configure_style()
    figure, axis = plt.subplots(
        figsize=(line_style.DOUBLE_COLUMN, line_style.LINE_FIGURE_HEIGHT),
        layout="constrained",
    )

    for method in HEURISTICS:
        axis.plot(
            dense_x,
            np.full_like(dense_x, heuristic_levels[method]["mean"]),
            markevery=marker_stride,
            solid_capstyle="round",
            dash_capstyle="round",
            **line_style.line_kwargs(
                method,
                label=REFERENCE_DISPLAY[method],
            ),
        )

    for method in METHODS:
        axis.plot(
            dense_x,
            curves[method]["mean"],
            markevery=marker_stride,
            solid_capstyle="round",
            dash_capstyle="round",
            **line_style.line_kwargs(
                method,
                label=REFERENCE_DISPLAY[method],
            ),
        )

    axis.set_xlim(0, 30)
    axis.set_ylim(0, 2.40)
    axis.set_xlabel(r"Training episode ($\times 10^3$)")
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.set_xticks((0, 5, 10, 15, 20, 25, 30))
    axis.set_yticks(np.arange(0, 2.26, 0.25))
    axis.margins(x=0)
    line_style.style_axis(axis, grid_axis="y")
    line_style.legend_above(axis, ncol=3)

    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 400}),
    ):
        figure.savefig(
            output_dir / f"{output_stem}.{suffix}",
            facecolor="white",
            bbox_inches=None,
            **options,
        )
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, curves, sources = collect_curves(
        args.smooth_window,
        args.plot_stride,
    )
    heuristic_levels, heuristic_sources = collect_heuristic_levels()
    source_data = args.output_dir / "online_training_curves_source.csv"
    write_source_data(source_data, episodes, curves)
    draw_figure(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
        args.smooth_window,
    )
    draw_tail_zoom(args.output_dir, episodes, curves, args.smooth_window)
    draw_combined_figure(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
        args.smooth_window,
    )
    measured_extension_source = (
        args.output_dir / "online_training_measured_extension_source.csv"
    )
    write_measured_extension_source(
        measured_extension_source,
        episodes,
        curves,
        heuristic_levels,
    )
    draw_measured_extension_figure(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
    )
    draw_daoc_style_comparison(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
    )
    draw_reference_style_comparison(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
    )
    draw_figures4papers_comparison(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
    )
    draw_topconf_comparison(
        args.output_dir,
        episodes,
        curves,
        heuristic_levels,
    )
    topconf_outputs = [
        args.output_dir / f"online_training_curves_topconf.{suffix}"
        for suffix in ("svg", "pdf", "png")
    ]
    (args.output_dir / "online_training_curves_topconf_manifest.json").write_text(
        json.dumps(
            {
                "figure": "online_training_curves_topconf",
                "source_data": str(source_data.resolve()),
                "source_data_sha256": sha256_file(source_data),
                "replication_unit": "independent seed",
                "seed_count": len(FINAL_SEEDS),
                "visual_smoothing": {
                    "method": "trailing arithmetic mean",
                    "window_episodes": args.smooth_window,
                    "render_stride_episodes": args.plot_stride,
                },
                "semantic_groups": {
                    "heuristics": list(HEURISTICS),
                    "standard_cache_learners": [
                        "dqn_wdsa_std_cache",
                        "discrete_sac_std_cache",
                        "daoc_paper",
                    ],
                    "coordinated_cache_controls": [
                        "daoc_our_coord_cache",
                        "coord_cache_discrete_sac",
                    ],
                    "proposed": ["lean_our"],
                },
                "synthetic_values_used": False,
                "outputs": {
                    output.suffix.lstrip("."): {
                        "path": str(output.resolve()),
                        "sha256": sha256_file(output),
                    }
                    for output in topconf_outputs
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    measured_summary = measured_extension_summary(episodes, curves)
    (
        args.output_dir / "online_training_measured_extension_manifest.json"
    ).write_text(
        json.dumps(
            {
                "figure": "online_training_curves_measured_extension",
                "scientific_role": "checkpoint-style online convergence figure",
                "measured_training_horizon": [500, FIXED_TRAIN_EPISODES],
                "retrained_methods": list(MEASURED_EXTENSION_METHODS),
                "line_semantics": (
                    "solid lines and markers throughout because all displayed "
                    "windows are measured observations"
                ),
                "point_definition": (
                    "10-seed mean of non-overlapping 500-episode online windows"
                ),
                "band_definition": (
                    "two-sided 95% t interval across 10 seed-level window means"
                ),
                "synthetic_values_used": False,
                "source_data": str(measured_extension_source.resolve()),
                "measured_changes": measured_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "figure": "online_training_curves",
        "scientific_role": "primary online-performance figure",
        "x_variable": "training episode",
        "y_variable": "mean DAG completion time in seconds",
        "replication_unit": "independent seed",
        "seeds": list(FINAL_SEEDS),
        "training_horizon": FIXED_TRAIN_EPISODES,
        "reported_tail": [TAIL_START_EPISODE, FIXED_TRAIN_EPISODES],
        "visual_smoothing": {
            "method": "trailing arithmetic mean",
            "window_episodes": args.smooth_window,
            "render_stride_episodes": args.plot_stride,
            "used_for_numerical_claims": False,
        },
        "uncertainty": "pointwise two-sided 95% t interval across seed curves",
        "heuristic_references": {
            "methods": list(HEURISTICS),
            "encoding": "horizontal line at the mean of seed-level online tails",
            "online_tail": [4001, 5000],
            "levels_seconds": {
                method: {
                    key: value
                    for key, value in heuristic_levels[method].items()
                    if key != "seed_values"
                }
                for method in HEURISTICS
            },
        },
        "source_data": str(source_data.resolve()),
        "source_runs": sources + heuristic_sources,
    }
    (args.output_dir / "online_training_curves_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_dir / "online_training_curves.png")


if __name__ == "__main__":
    main()
