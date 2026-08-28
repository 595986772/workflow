#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from information_protocol import INFORMATION_PROTOCOL_VERSION


COLORS = [
    "#4B5563",
    "#2563EB",
    "#E76F51",
    "#7C3AED",
    "#0891B2",
    "#65A30D",
    "#D97706",
    "#DC2626",
]

PAIRED_FIELDS = [
    "label",
    "display_name",
    "reference",
    "reference_display_name",
    "pairs",
    "wins",
    "mean_ratio",
    "median_ratio",
    "ratio_ci95",
    "mean_improvement_percent",
    "median_improvement_percent",
    "improvement_ci95",
]

PLOT_NAMES = {
    "guided_full": "DAOC",
    "cpr_reward": "CPR",
    "cpr_cache": "CPR + local cache",
    "cpr_coord_cache": "CPR + coordinated cache",
    "cpr_joint_cache": "CPR + joint cache",
    "hybrid_reward": "Hybrid reward",
    "our": "OUR",
    "lean_our": "Lean OUR",
    "hcpr_telemetry_pd3qn": "OUR",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate independent reproduction runs."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    standard_error = stats.sem(values)
    return float(stats.t.ppf(0.975, values.size - 1) * standard_error)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(suite_dir):
    manifest_path = suite_dir / "suite_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    config_by_label = {
        config["label"]: config
        for config in manifest.get("algorithms", [])
    }
    runs = []
    for summary_path in sorted((suite_dir / "runs").glob("*/seed_*/summary.json")):
        summary = read_json(summary_path)
        if summary.get("status") != "complete":
            continue
        if (
            summary.get("information_protocol_version")
            != INFORMATION_PROTOCOL_VERSION
        ):
            raise RuntimeError(
                "Refusing to aggregate a pre-fairness run: "
                f"{summary_path}. Rerun it under "
                f"{INFORMATION_PROTOCOL_VERSION}."
            )
        run_dir = summary_path.parent
        config = read_json(run_dir / "config.json")
        runs.append(
            {
                "summary": summary,
                "config": config,
                "run_dir": run_dir,
            }
        )
    if not runs:
        raise RuntimeError(f"No completed runs found under {suite_dir / 'runs'}")
    return manifest, config_by_label, runs


def ordered_labels(manifest, runs):
    available = {run["summary"]["label"] for run in runs}
    labels = [
        config["label"]
        for config in manifest.get("algorithms", [])
        if config["label"] in available
    ]
    labels.extend(sorted(available - set(labels)))
    return labels


def seed_metric(runs_by_label, label, section, metric):
    return [
        run["summary"][section][metric]
        for run in runs_by_label[label]
        if run["summary"].get(section) is not None
    ]


def aggregate_seed_summaries(
    suite_dir,
    labels,
    config_by_label,
    runs_by_label,
):
    rows = []
    for label in labels:
        runs = runs_by_label[label]
        config = config_by_label.get(label, {})
        row = {
            "label": label,
            "display_name": config.get("display_name", label.replace("_", " ").title()),
            "algorithm": runs[0]["summary"]["algorithm"],
            "beta": runs[0]["config"]["arguments"]["beta"],
            "beta_min": runs[0]["config"]["arguments"]["beta_min"],
            "beta_decay": runs[0]["config"]["arguments"]["beta_decay"],
            "seeds": len(runs),
        }
        metrics = {
            "eval_finish_time": ("eval", "mean_average_finish_time"),
            "eval_p95_finish_time": ("eval", "mean_p95_finish_time"),
            "eval_cache_hit_rate": ("eval", "mean_cache_hit_rate"),
            "paper_training_finish_time": (
                "paper_training_metric",
                "mean_average_finish_time",
            ),
            "train_tail_finish_time": ("train_tail", "mean_average_finish_time"),
            "train_tail_cache_hit_rate": ("train_tail", "mean_cache_hit_rate"),
            "train_tail_cache_replacements": (
                "train_tail",
                "mean_cache_replacements",
            ),
            "wall_time_sec": (None, "total_wall_time_sec"),
        }
        for output_name, (section, metric) in metrics.items():
            if section is None:
                values = [run["summary"][metric] for run in runs]
            else:
                values = seed_metric(runs_by_label, label, section, metric)
            row[f"{output_name}_mean"] = float(np.mean(values))
            row[f"{output_name}_ci95"] = confidence_interval(values)
        rows.append(row)

    output_path = suite_dir / "aggregate_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def paired_comparisons(
    suite_dir,
    labels,
    config_by_label,
    runs_by_label,
):
    values_by_label = {
        label: {
            run["summary"]["seed"]: run["summary"]["eval"][
                "mean_average_finish_time"
            ]
            for run in runs_by_label[label]
        }
        for label in labels
    }
    comparisons = comparison_pairs(labels, values_by_label)

    rows, plot_data = calculate_paired_rows(
        comparisons,
        values_by_label,
        config_by_label,
        plot_reference="nearest",
    )

    output_path = suite_dir / "paired_comparisons.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PAIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return plot_data


def comparison_pairs(labels, values_by_label):
    comparisons = [(label, "nearest") for label in labels]
    for guided_label in (
        "guided_full",
        "guided_decay",
        "gate_only",
        "history_only",
        "proposed_full",
        "cpr_reward",
        "cpr_cache",
        "cpr_coord_cache",
        "cpr_joint_cache",
        "hybrid_reward",
        "our",
        "lean_our",
        "hcpr_telemetry_pd3qn",
    ):
        if guided_label in values_by_label and "unguided_full" in values_by_label:
            comparisons.append((guided_label, "unguided_full"))
        if guided_label in values_by_label and "greedy" in values_by_label:
            comparisons.append((guided_label, "greedy"))
        if (
            guided_label in values_by_label
            and guided_label != "guided_full"
            and "guided_full" in values_by_label
        ):
            comparisons.append((guided_label, "guided_full"))
    if {"guided_decay", "guided_full"}.issubset(values_by_label):
        comparisons.append(("guided_decay", "guided_full"))
    if {"proposed_full", "gate_only"}.issubset(values_by_label):
        comparisons.append(("proposed_full", "gate_only"))
    if {"proposed_full", "history_only"}.issubset(values_by_label):
        comparisons.append(("proposed_full", "history_only"))
    if {"cpr_reward", "guided_full"}.issubset(values_by_label):
        comparisons.append(("cpr_reward", "guided_full"))
    if {"cpr_cache", "guided_full"}.issubset(values_by_label):
        comparisons.append(("cpr_cache", "guided_full"))
    if {"cpr_cache", "cpr_reward"}.issubset(values_by_label):
        comparisons.append(("cpr_cache", "cpr_reward"))
    if {"cpr_coord_cache", "guided_full"}.issubset(values_by_label):
        comparisons.append(("cpr_coord_cache", "guided_full"))
    if {"cpr_coord_cache", "cpr_reward"}.issubset(values_by_label):
        comparisons.append(("cpr_coord_cache", "cpr_reward"))
    if {"cpr_coord_cache", "cpr_cache"}.issubset(values_by_label):
        comparisons.append(("cpr_coord_cache", "cpr_cache"))
    if {"cpr_joint_cache", "guided_full"}.issubset(values_by_label):
        comparisons.append(("cpr_joint_cache", "guided_full"))
    if {"cpr_joint_cache", "cpr_reward"}.issubset(values_by_label):
        comparisons.append(("cpr_joint_cache", "cpr_reward"))
    if {"cpr_joint_cache", "cpr_coord_cache"}.issubset(values_by_label):
        comparisons.append(("cpr_joint_cache", "cpr_coord_cache"))
    if {"hybrid_reward", "guided_full"}.issubset(values_by_label):
        comparisons.append(("hybrid_reward", "guided_full"))
    if {"hybrid_reward", "cpr_reward"}.issubset(values_by_label):
        comparisons.append(("hybrid_reward", "cpr_reward"))
    if {"our", "guided_full"}.issubset(values_by_label):
        comparisons.append(("our", "guided_full"))
    if {"our", "cpr_reward"}.issubset(values_by_label):
        comparisons.append(("our", "cpr_reward"))
    if {"our", "cpr_coord_cache"}.issubset(values_by_label):
        comparisons.append(("our", "cpr_coord_cache"))
    if {"our", "hybrid_reward"}.issubset(values_by_label):
        comparisons.append(("our", "hybrid_reward"))
    if {"our", "cpr_joint_cache"}.issubset(values_by_label):
        comparisons.append(("our", "cpr_joint_cache"))
    if {"cpqac", "correct_ddqn"}.issubset(values_by_label):
        comparisons.append(("cpqac", "correct_ddqn"))
    if {"cpqac", "guided_full"}.issubset(values_by_label):
        comparisons.append(("cpqac", "guided_full"))
    if {"cpqac", "our"}.issubset(values_by_label):
        comparisons.append(("cpqac", "our"))
    if {"telemetry_cpqac", "telemetry_ddqn"}.issubset(
        values_by_label
    ):
        comparisons.append(
            ("telemetry_cpqac", "telemetry_ddqn")
        )
    if {"telemetry_cpqac", "cpqac"}.issubset(values_by_label):
        comparisons.append(("telemetry_cpqac", "cpqac"))
    if {"telemetry_cpqac", "guided_full"}.issubset(
        values_by_label
    ):
        comparisons.append(
            ("telemetry_cpqac", "guided_full")
        )
    if {"telemetry_cpqac", "our"}.issubset(values_by_label):
        comparisons.append(("telemetry_cpqac", "our"))
    for reference in (
        "correct_ddqn",
        "cpqac",
        "guided_full",
        "our",
    ):
        if {"capq", reference}.issubset(values_by_label):
            comparisons.append(("capq", reference))
    for variant in ("capq_tail50", "capq_mean"):
        for reference in (
            "correct_ddqn",
            "capq",
            "guided_full",
            "our",
        ):
            if {variant, reference}.issubset(values_by_label):
                comparisons.append((variant, reference))
    if {"guided_cpqac", "guided_correct_ddqn"}.issubset(
        values_by_label
    ):
        comparisons.append(
            ("guided_cpqac", "guided_correct_ddqn")
        )
    if {"guided_cpqac", "guided_full"}.issubset(
        values_by_label
    ):
        comparisons.append(("guided_cpqac", "guided_full"))
    if {"guided_cpqac", "our"}.issubset(values_by_label):
        comparisons.append(("guided_cpqac", "our"))
    for reference in (
        "guided_correct_ddqn",
        "correct_ddqn",
        "guided_full",
        "our",
    ):
        if {"pd3qn", reference}.issubset(values_by_label):
            comparisons.append(("pd3qn", reference))
    return list(dict.fromkeys(comparisons))


def calculate_paired_rows(
    comparisons,
    values_by_label,
    config_by_label,
    plot_reference=None,
):
    rows = []
    plot_data = []
    for label, reference in comparisons:
        if reference not in values_by_label:
            continue
        common_seeds = sorted(
            set(values_by_label[label]) & set(values_by_label[reference])
        )
        ratios = np.asarray(
            [
                values_by_label[label][seed]
                / values_by_label[reference][seed]
                for seed in common_seeds
            ],
            dtype=float,
        )
        improvements = 100.0 * (1.0 - ratios)
        row = {
            "label": label,
            "display_name": config_by_label.get(label, {}).get(
                "display_name",
                label,
            ),
            "reference": reference,
            "reference_display_name": config_by_label.get(reference, {}).get(
                "display_name",
                reference,
            ),
            "pairs": len(common_seeds),
            "wins": int(np.sum(ratios < 1.0)),
            "mean_ratio": float(ratios.mean()),
            "median_ratio": float(np.median(ratios)),
            "ratio_ci95": confidence_interval(ratios),
            "mean_improvement_percent": float(improvements.mean()),
            "median_improvement_percent": float(np.median(improvements)),
            "improvement_ci95": confidence_interval(improvements),
        }
        rows.append(row)
        if reference == plot_reference:
            plot_data.append(
                {
                    "label": label,
                    "display_name": row["display_name"],
                    "ratios": ratios,
                }
            )

    return rows, plot_data


def paired_paper_training_comparisons(
    suite_dir,
    labels,
    config_by_label,
    runs_by_label,
):
    values_by_label = {
        label: {
            run["summary"]["seed"]: run["summary"]["paper_training_metric"][
                "mean_average_finish_time"
            ]
            for run in runs_by_label[label]
        }
        for label in labels
    }
    rows, _ = calculate_paired_rows(
        comparison_pairs(labels, values_by_label),
        values_by_label,
        config_by_label,
    )
    output_path = suite_dir / "paired_paper_training_comparisons.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PAIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_episode_rows(run):
    rows = []
    with (run["run_dir"] / "episodes.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        for row in csv.DictReader(input_file):
            for key, value in list(row.items()):
                if key in {
                    "label",
                    "algorithm",
                    "reward_mode",
                    "cache_policy",
                    "cache_information_regime",
                    "scenario_fingerprint",
                    "base_scenario_fingerprint",
                    "phase",
                } or key.endswith("_json"):
                    continue
                row[key] = float(value) if value != "" else math.nan
            rows.append(row)
    return rows


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        result[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)
    return result


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "legend.frameon": False,
        }
    )


def clean_axis(axis, grid=True):
    axis.set_facecolor("#F8FAFC")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)


def save_figure(figure, output_base):
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_training_curves(
    suite_dir,
    labels,
    config_by_label,
    episode_rows_by_label,
):
    heuristic = [
        label
        for label in labels
        if config_by_label.get(label, {}).get("family") == "heuristic"
    ]
    learning = [label for label in labels if label not in heuristic]
    groups = [heuristic, learning]
    titles = ["Heuristic baselines", "DQN ablations"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for axis, group, title in zip(axes, groups, titles):
        for label in group:
            seed_series = episode_rows_by_label[label]
            min_length = min(len(values) for values in seed_series)
            window = max(1, min_length // 50)
            smoothed = np.asarray(
                [smooth(values[:min_length], window) for values in seed_series]
            )
            mean = smoothed.mean(axis=0)
            ci = np.asarray(
                [
                    confidence_interval(smoothed[:, index])
                    for index in range(min_length)
                ]
            )
            config = config_by_label.get(label, {})
            color = COLORS[labels.index(label) % len(COLORS)]
            x = np.arange(1, min_length + 1)
            axis.plot(
                x,
                mean,
                color=color,
                linewidth=1.8,
                label=config.get("display_name", label),
            )
            if smoothed.shape[0] > 1:
                axis.fill_between(
                    x,
                    mean - ci,
                    mean + ci,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        clean_axis(axis)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Average application finish time (s)")
        axis.set_ylim(bottom=0)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=8)

    figure.suptitle(
        "Training Performance Across Independent Seeds",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, suite_dir / "training_convergence")


def plot_evaluation(suite_dir, aggregate_rows, runs_by_label):
    names = [
        PLOT_NAMES.get(row["label"], row["display_name"])
        for row in aggregate_rows
    ]
    means = [row["eval_finish_time_mean"] for row in aggregate_rows]
    errors = [row["eval_finish_time_ci95"] for row in aggregate_rows]
    colors = [COLORS[index % len(COLORS)] for index in range(len(names))]

    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    x = np.arange(len(names))
    axis.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for index, row in enumerate(aggregate_rows):
        seed_values = [
            run["summary"]["eval"]["mean_average_finish_time"]
            for run in runs_by_label[row["label"]]
        ]
        offsets = np.linspace(-0.08, 0.08, len(seed_values))
        axis.scatter(
            index + offsets,
            seed_values,
            color="#111827",
            edgecolor="white",
            linewidth=0.7,
            s=30,
            zorder=4,
        )
    clean_axis(axis)
    axis.set_title("Frozen-Policy Evaluation", loc="left", fontsize=15)
    axis.set_ylabel("Average application finish time (s)")
    axis.set_xticks(x, names, rotation=15, ha="right")
    axis.set_ylim(bottom=0)
    save_figure(figure, suite_dir / "evaluation_finish_time")


def plot_main_evaluation(suite_dir, aggregate_rows, runs_by_label):
    key_labels = [
        "guided_full",
        "cpr_reward",
        "cpr_coord_cache",
        "cpr_joint_cache",
        "hybrid_reward",
        "our",
    ]
    row_by_label = {
        row["label"]: row for row in aggregate_rows
    }
    rows = [
        row_by_label[label]
        for label in key_labels
        if label in row_by_label
    ]
    if len(rows) < 2:
        return

    names = [
        PLOT_NAMES.get(row["label"], row["display_name"])
        for row in rows
    ]
    means = [row["eval_finish_time_mean"] for row in rows]
    errors = [row["eval_finish_time_ci95"] for row in rows]
    y = np.arange(len(rows))
    colors = [
        "#4B5563",
        "#2563EB",
        "#7C3AED",
        "#0891B2",
        "#65A30D",
        "#E76F51",
    ][:len(rows)]

    figure, axis = plt.subplots(
        figsize=(9.2, 5.4),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    axis.barh(
        y,
        means,
        xerr=errors,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for index, row in enumerate(rows):
        seed_values = [
            run["summary"]["eval"]["mean_average_finish_time"]
            for run in runs_by_label[row["label"]]
        ]
        offsets = np.linspace(-0.10, 0.10, len(seed_values))
        axis.scatter(
            seed_values,
            index + offsets,
            color="#111827",
            edgecolor="white",
            linewidth=0.7,
            s=34,
            zorder=4,
        )
    clean_axis(axis, grid=False)
    axis.grid(
        axis="x",
        color="#CBD5E1",
        linewidth=0.7,
        alpha=0.8,
    )
    axis.set_axisbelow(True)
    axis.set_title(
        "Independent Frozen-Policy Evaluation",
        loc="left",
        fontsize=15,
    )
    axis.set_xlabel("Average application finish time (s)")
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    save_figure(figure, suite_dir / "main_evaluation")


def plot_paper_training_metric(suite_dir, aggregate_rows, runs_by_label):
    names = [
        PLOT_NAMES.get(row["label"], row["display_name"])
        for row in aggregate_rows
    ]
    means = [
        row["paper_training_finish_time_mean"]
        for row in aggregate_rows
    ]
    errors = [
        row["paper_training_finish_time_ci95"]
        for row in aggregate_rows
    ]
    colors = [COLORS[index % len(COLORS)] for index in range(len(names))]

    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    x = np.arange(len(names))
    axis.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for index, row in enumerate(aggregate_rows):
        seed_values = [
            run["summary"]["paper_training_metric"][
                "mean_average_finish_time"
            ]
            for run in runs_by_label[row["label"]]
        ]
        offsets = np.linspace(-0.08, 0.08, len(seed_values))
        axis.scatter(
            index + offsets,
            seed_values,
            color="#111827",
            edgecolor="white",
            linewidth=0.7,
            s=30,
            zorder=4,
        )
    clean_axis(axis)
    axis.set_title("Paper-Style Late-Training Metric", loc="left", fontsize=15)
    axis.set_ylabel("Average application finish time (s)")
    axis.set_xticks(x, names, rotation=15, ha="right")
    axis.set_ylim(bottom=0)
    save_figure(figure, suite_dir / "paper_training_finish_time")


def plot_paired_performance(suite_dir, plot_data):
    names = [item["display_name"] for item in plot_data]
    x = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for index, item in enumerate(plot_data):
        ratios = item["ratios"]
        offsets = np.linspace(-0.08, 0.08, len(ratios))
        axis.scatter(
            index + offsets,
            ratios,
            color=COLORS[index % len(COLORS)],
            edgecolor="white",
            linewidth=0.8,
            s=48,
            zorder=4,
        )
        axis.hlines(
            np.median(ratios),
            index - 0.18,
            index + 0.18,
            color="#111827",
            linewidth=2,
            zorder=3,
        )

    clean_axis(axis)
    axis.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1.3)
    axis.set_yscale("log")
    axis.set_title(
        "Paired Performance Relative to Nearest",
        loc="left",
        fontsize=15,
    )
    axis.set_ylabel("Finish-time ratio (lower is better)")
    axis.set_xticks(x, names, rotation=24, ha="right")
    axis.text(
        0,
        -0.28,
        "Each point is one matched topology seed; horizontal marks show medians.",
        transform=axis.transAxes,
        fontsize=9,
        color="#6B7280",
    )
    save_figure(figure, suite_dir / "paired_relative_performance")


def plot_guidance_ablation(suite_dir, runs_by_label):
    labels = [
        label
        for label in ("unguided_full", "guided_full", "guided_decay")
        if label in runs_by_label
    ]
    if "unguided_full" not in labels or len(labels) < 2:
        return
    values_by_label = {
        label: {
            run["summary"]["seed"]: run["summary"]["eval"][
                "mean_average_finish_time"
            ]
            for run in runs_by_label[label]
        }
        for label in labels
    }
    common_seeds = sorted(
        set.intersection(
            *(set(values_by_label[label]) for label in labels)
        )
    )
    if not common_seeds:
        return

    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for index, seed in enumerate(common_seeds):
        values = [values_by_label[label][seed] for label in labels]
        axis.plot(
            range(len(labels)),
            values,
            color=COLORS[index % len(COLORS)],
            linewidth=1.8,
            marker="o",
            markersize=7,
            label=f"Seed {seed}",
        )
    clean_axis(axis)
    axis.set_title("Effect of Guided Action Shaping", loc="left", fontsize=15)
    axis.set_ylabel("Average application finish time (s)")
    display_names = {
        "unguided_full": "Unguided",
        "guided_full": "Fixed beta",
        "guided_decay": "Paper decay",
    }
    axis.set_xticks(
        range(len(labels)),
        [display_names[label] for label in labels],
    )
    axis.set_xlim(-0.25, len(labels) - 0.75)
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.text(
        0,
        -0.18,
        "Matched topology seeds; every downward line favors guided training.",
        transform=axis.transAxes,
        fontsize=9,
        color="#6B7280",
    )
    save_figure(figure, suite_dir / "guidance_ablation")


def plot_innovation_ablation(suite_dir, runs_by_label):
    labels = [
        label
        for label in (
            "guided_full",
            "gate_only",
            "history_only",
            "proposed_full",
        )
        if label in runs_by_label
    ]
    if len(labels) < 2 or "guided_full" not in labels:
        return

    values_by_label = {
        label: {
            run["summary"]["seed"]: run["summary"]["eval"][
                "mean_average_finish_time"
            ]
            for run in runs_by_label[label]
        }
        for label in labels
    }
    common_seeds = sorted(
        set.intersection(
            *(set(values_by_label[label]) for label in labels)
        )
    )
    if not common_seeds:
        return

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for index, seed in enumerate(common_seeds):
        axis.plot(
            range(len(labels)),
            [values_by_label[label][seed] for label in labels],
            color=COLORS[index % len(COLORS)],
            linewidth=1.8,
            marker="o",
            markersize=7,
            label=f"Seed {seed}",
        )
    clean_axis(axis)
    axis.set_title("Sample-Efficient Method Ablation", loc="left", fontsize=15)
    axis.set_ylabel("Frozen-policy finish time (s)")
    display_names = {
        "guided_full": "Original guided",
        "gate_only": "Adaptive gate only",
        "history_only": "History only",
        "proposed_full": "Full method",
    }
    axis.set_xticks(
        range(len(labels)),
        [display_names[label] for label in labels],
    )
    axis.set_xlim(-0.25, len(labels) - 0.75)
    axis.set_ylim(bottom=0)
    axis.legend()
    save_figure(figure, suite_dir / "innovation_ablation")


def plot_guidance_handoff(
    suite_dir,
    config_by_label,
    runs_by_label,
):
    labels = [
        label
        for label in ("gate_only", "history_only", "proposed_full")
        if label in runs_by_label
    ]
    if not labels:
        return

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    for label_index, label in enumerate(labels):
        action_rate_series = []
        probability_series = []
        for run in runs_by_label[label]:
            train_rows = [
                row for row in read_episode_rows(run)
                if row["phase"] == "train"
            ]
            action_rate_series.append(
                np.asarray(
                    [
                        row["feedback_guidance_rate"]
                        + row["adaptive_cache_rate"]
                        for row in train_rows
                    ],
                    dtype=float,
                )
            )
            probability_series.append(
                np.asarray(
                    [
                        row["mean_guidance_probability"]
                        for row in train_rows
                    ],
                    dtype=float,
                )
            )

        minimum_length = min(
            len(series) for series in action_rate_series
        )
        window = max(1, minimum_length // 50)
        x = np.arange(1, minimum_length + 1)
        color = COLORS[label_index % len(COLORS)]
        display_name = config_by_label.get(label, {}).get(
            "display_name",
            label,
        )
        for axis, series_group in zip(
            axes,
            (action_rate_series, probability_series),
        ):
            smoothed = np.asarray(
                [
                    smooth(series[:minimum_length], window)
                    for series in series_group
                ]
            )
            axis.plot(
                x,
                smoothed.mean(axis=0),
                color=color,
                linewidth=1.8,
                label=display_name,
            )

    titles = ["Observed guidance action rate", "Guidance probability"]
    for axis, title in zip(axes, titles):
        clean_axis(axis)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Training episode")
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Expert-to-DQN Handoff",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(figure, suite_dir / "guidance_handoff")


def write_sample_efficiency_table(
    suite_dir,
    labels,
    config_by_label,
    runs_by_label,
):
    if "guided_full" not in runs_by_label:
        return

    reference_thresholds = {}
    for run in runs_by_label["guided_full"]:
        train_rows = [
            row for row in read_episode_rows(run)
            if row["phase"] == "train"
        ]
        values = np.asarray(
            [row["average_finish_time"] for row in train_rows],
            dtype=float,
        )
        window = max(5, values.size // 50)
        smoothed = smooth(values, window)
        tail = max(10, values.size // 10)
        reference_thresholds[run["summary"]["seed"]] = float(
            smoothed[-tail:].mean()
        )

    rows = []
    for label in labels:
        for run in runs_by_label[label]:
            seed = run["summary"]["seed"]
            if seed not in reference_thresholds:
                continue
            train_rows = [
                row for row in read_episode_rows(run)
                if row["phase"] == "train"
            ]
            values = np.asarray(
                [row["average_finish_time"] for row in train_rows],
                dtype=float,
            )
            window = max(5, values.size // 50)
            smoothed = smooth(values, window)
            threshold = reference_thresholds[seed]
            reached_indices = np.flatnonzero(smoothed <= threshold)
            row = {
                "label": label,
                "display_name": config_by_label.get(label, {}).get(
                    "display_name",
                    label,
                ),
                "seed": seed,
                "training_episodes": int(values.size),
                "moving_window": window,
                "mean_training_finish_time": float(values.mean()),
                "guided_final_threshold": threshold,
                "reached_guided_final_threshold": bool(
                    reached_indices.size
                ),
                "episodes_to_guided_final_threshold": (
                    int(reached_indices[0] + 1)
                    if reached_indices.size
                    else ""
                ),
            }
            for fraction in (0.1, 0.25, 0.5, 1.0):
                stop = max(1, int(round(values.size * fraction)))
                key = f"finish_time_at_{int(fraction * 100)}pct_budget"
                row[key] = float(smoothed[stop - 1])
            rows.append(row)

    if not rows:
        return
    with (suite_dir / "sample_efficiency.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_latency_breakdown(
    suite_dir,
    labels,
    config_by_label,
    eval_rows_by_label,
):
    share_fields = [
        ("computing_share", "Computing", "#2563EB"),
        ("data_transfer_share", "Input transfer", "#0891B2"),
        ("predecessor_share", "DAG transfer", "#7C3AED"),
        ("service_share", "Service loading", "#E76F51"),
        ("waiting_share", "Waiting", "#65A30D"),
    ]
    names = [
        PLOT_NAMES.get(
            label,
            config_by_label.get(label, {}).get(
                "display_name",
                label,
            ),
        )
        for label in labels
    ]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for field, component_name, color in share_fields:
        values = [
            np.mean([row[field] for row in eval_rows_by_label[label]])
            for label in labels
        ]
        axis.bar(
            x,
            values,
            bottom=bottom,
            label=component_name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += np.asarray(values)
    clean_axis(axis)
    axis.set_title("Evaluation Latency Composition", loc="left", fontsize=15)
    axis.set_ylabel("Fraction of cumulative latency")
    axis.set_xticks(x, names, rotation=15, ha="right")
    axis.set_ylim(0, 1.02)
    axis.legend(ncol=3, fontsize=8, loc="upper center")
    save_figure(figure, suite_dir / "latency_breakdown")


def plot_cache_metrics(suite_dir, aggregate_rows):
    names = [
        PLOT_NAMES.get(row["label"], row["display_name"])
        for row in aggregate_rows
    ]
    hit_means = [row["eval_cache_hit_rate_mean"] for row in aggregate_rows]
    hit_errors = [row["eval_cache_hit_rate_ci95"] for row in aggregate_rows]
    replacement_means = [
        row["train_tail_cache_replacements_mean"]
        for row in aggregate_rows
    ]
    replacement_errors = [
        row["train_tail_cache_replacements_ci95"]
        for row in aggregate_rows
    ]
    colors = [COLORS[index % len(COLORS)] for index in range(len(names))]
    x = np.arange(len(names))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axes[0].bar(
        x,
        hit_means,
        yerr=hit_errors,
        capsize=3,
        color=colors,
        edgecolor="white",
    )
    axes[0].set_title("Evaluation cache hit rate", loc="left")
    axes[0].set_ylabel("Hit rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(
        x,
        replacement_means,
        yerr=replacement_errors,
        capsize=3,
        color=colors,
        edgecolor="white",
    )
    axes[1].set_title("Late-training cache replacements", loc="left")
    axes[1].set_ylabel("Replacements per episode")
    axes[1].set_ylim(bottom=0)

    for axis in axes:
        clean_axis(axis)
        axis.set_xticks(x, names, rotation=15, ha="right")
    figure.suptitle("Caching Behavior", fontsize=15, fontweight="bold")
    save_figure(figure, suite_dir / "cache_metrics")


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    configure_plot_style()
    manifest, config_by_label, runs = load_runs(suite_dir)
    labels = ordered_labels(manifest, runs)

    runs_by_label = defaultdict(list)
    train_series_by_label = defaultdict(list)
    eval_rows_by_label = defaultdict(list)
    for run in runs:
        label = run["summary"]["label"]
        runs_by_label[label].append(run)
        episode_rows = read_episode_rows(run)
        train_series_by_label[label].append(
            [
                row["average_finish_time"]
                for row in episode_rows
                if row["phase"] == "train"
            ]
        )
        eval_rows_by_label[label].extend(
            [row for row in episode_rows if row["phase"] == "eval"]
        )

    aggregate_rows = aggregate_seed_summaries(
        suite_dir,
        labels,
        config_by_label,
        runs_by_label,
    )
    paired_plot_data = paired_comparisons(
        suite_dir,
        labels,
        config_by_label,
        runs_by_label,
    )
    paired_paper_training_comparisons(
        suite_dir,
        labels,
        config_by_label,
        runs_by_label,
    )
    plot_training_curves(
        suite_dir,
        labels,
        config_by_label,
        train_series_by_label,
    )
    plot_evaluation(suite_dir, aggregate_rows, runs_by_label)
    plot_main_evaluation(suite_dir, aggregate_rows, runs_by_label)
    plot_paper_training_metric(suite_dir, aggregate_rows, runs_by_label)
    plot_paired_performance(suite_dir, paired_plot_data)
    plot_guidance_ablation(suite_dir, runs_by_label)
    plot_innovation_ablation(suite_dir, runs_by_label)
    plot_guidance_handoff(
        suite_dir,
        config_by_label,
        runs_by_label,
    )
    write_sample_efficiency_table(
        suite_dir,
        labels,
        config_by_label,
        runs_by_label,
    )
    plot_latency_breakdown(
        suite_dir,
        labels,
        config_by_label,
        eval_rows_by_label,
    )
    plot_cache_metrics(suite_dir, aggregate_rows)
    print(suite_dir / "aggregate_summary.csv")


if __name__ == "__main__":
    main()
