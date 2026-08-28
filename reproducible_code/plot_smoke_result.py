#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the average application finish time from a smoke test."
    )
    parser.add_argument("values", type=Path, help="Path to values.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("smoke_test_convergence"),
        help="Output path without an extension",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    values = np.loadtxt(args.values, dtype=float, ndmin=1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("The input must contain at least one finite value.")

    episodes = np.arange(1, values.size + 1)
    post_initial_mean = values[1:].mean() if values.size > 1 else values.mean()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    ax.plot(
        episodes,
        values,
        color="#087E8B",
        linewidth=2.4,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Average finish time",
        zorder=3,
    )
    ax.axhline(
        post_initial_mean,
        color="#F97316",
        linewidth=1.7,
        linestyle="--",
        label=f"Mean after episode 1: {post_initial_mean:.3f} s",
        zorder=2,
    )

    ax.scatter(
        episodes[0],
        values[0],
        color="#DC2626",
        edgecolor="white",
        linewidth=1.5,
        s=70,
        zorder=4,
    )
    ax.annotate(
        f"Initialization\n{values[0]:.3f} s",
        xy=(episodes[0], values[0]),
        xytext=(episodes[0] + 0.55, values[0] * 0.88),
        arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 1},
        fontsize=9.5,
        color="#374151",
    )

    ax.set_title("DQN Smoke Test: Application Finish Time", loc="left", pad=15)
    ax.text(
        0,
        1.015,
        "3 users | 3 servers | 3 services | 6-task DAG | seed 2",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#6B7280",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average application finish time (s)")
    ax.set_xticks(episodes)
    ax.set_xlim(0.7, episodes[-1] + 0.3)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    ax.text(
        0,
        -0.18,
        "Smoke test only: one run and eight episodes; this figure verifies execution, not convergence.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6B7280",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)

    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
