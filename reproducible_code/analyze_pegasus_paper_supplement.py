#!/usr/bin/env python3
"""Analyze all frozen post-lock experiments required for the paper."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze_a0_coordination import (
    aggregate_run,
    evaluation_rows,
    paired_superiority,
)
from pegasus_paper_supplement_protocol import (
    ABLATION_METHODS,
    ABLATION_REFERENCE_METHOD,
    ABLATION_SEEDS,
    CAPACITY_PROFILES,
    FAMILIES,
    HETEROGENEITY_METHODS,
    HETEROGENEITY_SEEDS,
    P3_FINAL_DIR,
    P3_FINAL_LOCK_PATH,
    PROTOCOL_VERSION,
    SCALING_USER_COUNTS,
    specification,
    supplement_source_hash,
    validate_parent_freeze,
)


DISPLAY = {
    "daoc_paper": "DAOC-paper",
    "centralized_greedy_daoc": "Centralized-Greedy-DQN",
    "lean_our": "OUR",
    "our_dqn": "OUR-DQN",
    "our_no_coord_cache": "OUR-noCoordCache",
}
COLORS = {
    "daoc_paper": "#42475F",
    "centralized_greedy_daoc": "#E2A33A",
    "lean_our": "#2E86A6",
    "our_dqn": "#6E8B74",
    "our_no_coord_cache": "#B45F5F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * stats.sem(values)
    )
    return mean, half


def run_dir(suite, label, seed):
    return Path(suite) / "runs" / label / f"seed_{seed}"


def aggregate_methods(suite, methods, seeds):
    per_seed = []
    for seed in seeds:
        row = {"seed": seed, "methods": {}}
        for label in methods:
            row["methods"][label] = aggregate_run(
                evaluation_rows(run_dir(suite, label, seed) / "episodes.csv")
            )
        per_seed.append(row)
    means = {
        label: {
            metric: float(
                np.mean(
                    [entry["methods"][label][metric] for entry in per_seed]
                )
            )
            for metric in per_seed[0]["methods"][label]
        }
        for label in methods
    }
    return per_seed, means


def paired_delta_summary(values):
    values = np.asarray(values, dtype=float)
    mean, half = mean_ci95(values)
    try:
        p_value = float(
            stats.wilcoxon(values, alternative="greater").pvalue
        )
    except ValueError:
        p_value = 1.0
    return {
        "pairs": len(values),
        "mean_delta_sec": mean,
        "ci95_lower_sec": mean - half,
        "ci95_upper_sec": mean + half,
        "wilcoxon_one_sided_p": p_value,
        "positive_seeds": int(np.sum(values > 0)),
    }


def collect_heterogeneity(result_root):
    integrity = read_json(result_root / "heterogeneity/integrity.json")
    profiles = {}
    for profile, capacities in CAPACITY_PROFILES.items():
        suite = result_root / "heterogeneity" / profile
        per_seed, means = aggregate_methods(
            suite,
            HETEROGENEITY_METHODS,
            HETEROGENEITY_SEEDS,
        )
        comparisons = {
            f"our_vs_{reference}": paired_superiority(
                [
                    row["methods"][reference]["mean_finish_time"]
                    for row in per_seed
                ],
                [
                    row["methods"]["lean_our"]["mean_finish_time"]
                    for row in per_seed
                ],
                formal=False,
            )
            for reference in ("daoc_paper", "centralized_greedy_daoc")
        }
        profiles[profile] = {
            "capacity_multiset": list(capacities),
            "total_budget": int(sum(capacities)),
            "capacity_mean": float(np.mean(capacities)),
            "capacity_variance": float(np.var(capacities)),
            "per_seed": per_seed,
            "method_aggregates": means,
            "comparisons": comparisons,
        }
    hetero = profiles["heterogeneous_b10"]["per_seed"]
    uniform = profiles["uniform_b10"]["per_seed"]
    advantage_change = {}
    for reference in ("daoc_paper", "centralized_greedy_daoc"):
        deltas = []
        for uniform_row, hetero_row in zip(uniform, hetero):
            uniform_advantage = (
                uniform_row["methods"][reference]["mean_finish_time"]
                - uniform_row["methods"]["lean_our"]["mean_finish_time"]
            )
            hetero_advantage = (
                hetero_row["methods"][reference]["mean_finish_time"]
                - hetero_row["methods"]["lean_our"]["mean_finish_time"]
            )
            deltas.append(hetero_advantage - uniform_advantage)
        advantage_change[f"our_vs_{reference}"] = paired_delta_summary(deltas)
    gate = {
        profile: all(
            comparison["wins"] >= 2
            and comparison["mean_improvement_sec"] > 0
            for comparison in profiles[profile]["comparisons"].values()
        )
        for profile in profiles
    }
    gate["passed"] = bool(all(gate.values()))
    return {
        "integrity": integrity,
        "profiles": profiles,
        "heterogeneity_effect_on_our_advantage": advantage_change,
        "gate": gate,
    }


def collect_ablation(result_root):
    integrity = read_json(result_root / "ablation/integrity.json")
    methods = (ABLATION_REFERENCE_METHOD,) + tuple(ABLATION_METHODS)
    per_seed = []
    for seed in ABLATION_SEEDS:
        row = {"seed": seed, "methods": {}}
        row["methods"][ABLATION_REFERENCE_METHOD] = aggregate_run(
            evaluation_rows(
                run_dir(
                    P3_FINAL_DIR,
                    ABLATION_REFERENCE_METHOD,
                    seed,
                )
                / "episodes.csv"
            )
        )
        for label in ABLATION_METHODS:
            row["methods"][label] = aggregate_run(
                evaluation_rows(
                    run_dir(result_root / "ablation", label, seed)
                    / "episodes.csv"
                )
            )
        per_seed.append(row)
    means = {
        label: {
            metric: float(
                np.mean(
                    [entry["methods"][label][metric] for entry in per_seed]
                )
            )
            for metric in per_seed[0]["methods"][label]
        }
        for label in methods
    }
    comparisons = {
        f"our_vs_{label}": paired_superiority(
            [row["methods"][label]["mean_finish_time"] for row in per_seed],
            [
                row["methods"][ABLATION_REFERENCE_METHOD]["mean_finish_time"]
                for row in per_seed
            ],
            formal=True,
        )
        for label in ABLATION_METHODS
    }
    p95_comparisons = {
        f"our_vs_{label}": paired_superiority(
            [
                row["methods"][label]["mean_p95_finish_time"]
                for row in per_seed
            ],
            [
                row["methods"][ABLATION_REFERENCE_METHOD][
                    "mean_p95_finish_time"
                ]
                for row in per_seed
            ],
            formal=True,
        )
        for label in ABLATION_METHODS
    }
    gate = {
        "pairwise_pd3qn": comparisons["our_vs_our_dqn"]["passed"],
        "coordinated_cache": comparisons[
            "our_vs_our_no_coord_cache"
        ]["passed"],
    }
    gate["passed"] = bool(all(gate.values()))
    return {
        "integrity": integrity,
        "per_seed": per_seed,
        "method_aggregates": means,
        "paired_comparisons": comparisons,
        "p95_paired_comparisons": p95_comparisons,
        "gate": gate,
    }


def collect_final_context():
    summary = read_json(
        P3_FINAL_DIR / "analysis/pegasus_paper_closure_summary.json"
    )
    return {
        "method_aggregates": summary["method_aggregates"],
        "family_aggregates": summary["family_aggregates"],
        "paired_comparisons": summary["paired_comparisons"],
        "p95_paired_comparisons": summary["p95_paired_comparisons"],
        "per_seed": summary["per_seed"],
        "gate": summary["gate"],
    }


def final_seed_values(final_context, label, metric):
    return [
        row["methods"][label][metric]
        for row in final_context["per_seed"]
    ]


def plot_main_statistics(output_dir, final_context):
    methods = tuple(HETEROGENEITY_METHODS)
    metrics = (
        ("mean_finish_time", "Mean DAG completion time (s)"),
        ("mean_p95_finish_time", "P95 DAG completion time (s)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    rng = np.random.default_rng(20260805)
    for axis, (metric, ylabel) in zip(axes, metrics):
        means = []
        errors = []
        for index, label in enumerate(methods):
            values = final_seed_values(final_context, label, metric)
            mean, half = mean_ci95(values)
            means.append(mean)
            errors.append(half)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                color="white",
                edgecolor=COLORS[label],
                linewidth=1.1,
                s=26,
                zorder=3,
            )
        axis.bar(
            np.arange(len(methods)),
            means,
            yerr=errors,
            capsize=4,
            color=[COLORS[label] for label in methods],
            alpha=0.92,
        )
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels([DISPLAY[label] for label in methods], rotation=18)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"paper_main_statistics.{suffix}", dpi=240)
    plt.close(figure)


def common_validation_curve(label):
    curves = []
    for seed in range(51, 61):
        summary = read_json(run_dir(P3_FINAL_DIR, label, seed) / "summary.json")
        curves.append(
            {
                int(row["episode"]): float(row["mean_average_finish_time"])
                for row in summary["checkpoint_validation"]
            }
        )
    episodes = sorted(set.intersection(*(set(curve) for curve in curves)))
    means = []
    lows = []
    highs = []
    for episode in episodes:
        values = [curve[episode] for curve in curves]
        mean, half = mean_ci95(values)
        means.append(mean)
        lows.append(mean - half)
        highs.append(mean + half)
    return np.asarray(episodes), np.asarray(means), np.asarray(lows), np.asarray(highs)


def plot_convergence(output_dir):
    figure, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for label in HETEROGENEITY_METHODS:
        episodes, means, lows, highs = common_validation_curve(label)
        axis.plot(
            episodes,
            means,
            color=COLORS[label],
            label=DISPLAY[label],
            linewidth=2,
        )
        axis.fill_between(episodes, lows, highs, color=COLORS[label], alpha=0.16)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Frozen-validation completion time (s)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"validation_convergence.{suffix}", dpi=240)
    plt.close(figure)


def plot_workflow_mechanism(output_dir, final_context):
    methods = tuple(HETEROGENEITY_METHODS)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), constrained_layout=True)
    x = np.arange(len(FAMILIES))
    width = 0.25
    for index, label in enumerate(methods):
        axes[0].bar(
            x + (index - 1) * width,
            [
                final_context["family_aggregates"][label][family][
                    "mean_finish_time"
                ]
                for family in FAMILIES
            ],
            width=width,
            color=COLORS[label],
            label=DISPLAY[label],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FAMILIES, rotation=18)
    axes[0].set_ylabel("Mean DAG completion time (s)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    metric_names = ("mean_cache_hit_rate", "mean_remote_loading_rate")
    metric_labels = ("Cache hit", "Remote loading")
    x2 = np.arange(len(metric_names))
    for index, label in enumerate(methods):
        axes[1].bar(
            x2 + (index - 1) * width,
            [
                final_context["method_aggregates"][label][metric]
                for metric in metric_names
            ],
            width=width,
            color=COLORS[label],
            label=DISPLAY[label],
        )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"workflow_cache_mechanism.{suffix}", dpi=240)
    plt.close(figure)


def first_eval_row(directory):
    with (Path(directory) / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        for row in csv.DictReader(input_file):
            if row["phase"] == "eval":
                return row
    raise RuntimeError(f"No evaluation row in {directory}")


def average_cache_heatmap(label):
    matrices = []
    capacity_order = None
    for seed in range(51, 61):
        directory = run_dir(P3_FINAL_DIR, label, seed)
        summary = read_json(directory / "summary.json")
        capacities = {
            int(key): int(value)
            for key, value in summary["server_capacities"].items()
        }
        order = sorted(capacities, key=lambda server: (capacities[server], server))
        capacity_order = [capacities[server] for server in order]
        cache = json.loads(first_eval_row(directory)["cache_matrix_json"])
        matrix = np.zeros((10, 10), dtype=float)
        for row_index, server_id in enumerate(order):
            for service_id in cache[str(server_id)]:
                if int(service_id) > 0:
                    matrix[row_index, int(service_id) - 1] = 1.0
        matrices.append(matrix)
    return np.mean(matrices, axis=0), capacity_order


def plot_cache_heatmaps(output_dir):
    methods = tuple(HETEROGENEITY_METHODS)
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), constrained_layout=True)
    image = None
    for axis, label in zip(axes, methods):
        matrix, capacities = average_cache_heatmap(label)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(DISPLAY[label])
        axis.set_xlabel("Service ID")
        axis.set_xticks(np.arange(10))
        axis.set_xticklabels(np.arange(1, 11))
        axis.set_yticks(np.arange(10))
        axis.set_yticklabels([f"rank {i + 1} (K={k})" for i, k in enumerate(capacities)])
    axes[0].set_ylabel("Servers sorted by cache capacity")
    figure.colorbar(image, ax=axes, label="Caching frequency across 10 seeds")
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cache_heatmaps.{suffix}", dpi=240)
    plt.close(figure)


def plot_supplement_overview(output_dir, heterogeneity, ablation, scaling):
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), constrained_layout=True)
    methods = tuple(HETEROGENEITY_METHODS)
    profiles = ("uniform_b10", "heterogeneous_b10")
    x = np.arange(len(profiles))
    width = 0.25
    for index, label in enumerate(methods):
        means = []
        errors = []
        for profile in profiles:
            values = [
                row["methods"][label]["mean_finish_time"]
                for row in heterogeneity["profiles"][profile]["per_seed"]
            ]
            mean, half = mean_ci95(values)
            means.append(mean)
            errors.append(half)
        axes[0, 0].bar(
            x + (index - 1) * width,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=COLORS[label],
            label=DISPLAY[label],
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(("Uniform B10", "Heterogeneous B10"))
    axes[0, 0].set_ylabel("Mean completion time (s)")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    ablation_methods = ("lean_our", "our_dqn", "our_no_coord_cache")
    means = []
    errors = []
    for label in ablation_methods:
        values = [
            row["methods"][label]["mean_finish_time"]
            for row in ablation["per_seed"]
        ]
        mean, half = mean_ci95(values)
        means.append(mean)
        errors.append(half)
    axes[0, 1].bar(
        np.arange(len(ablation_methods)),
        means,
        yerr=errors,
        capsize=4,
        color=[COLORS[label] for label in ablation_methods],
    )
    axes[0, 1].set_xticks(np.arange(len(ablation_methods)))
    axes[0, 1].set_xticklabels(
        [DISPLAY[label] for label in ablation_methods], rotation=16
    )
    axes[0, 1].set_ylabel("Mean completion time (s)")
    axes[0, 1].grid(axis="y", alpha=0.25)

    users = list(SCALING_USER_COUNTS)
    for label in methods:
        axes[1, 0].plot(
            users,
            [
                scaling["results"][str(count)]["method_means"][label][
                    "mean_finish_time"
                ]
                for count in users
            ],
            marker="o",
            linewidth=2,
            color=COLORS[label],
            label=DISPLAY[label],
        )
    axes[1, 0].set_xlabel("Concurrent users")
    axes[1, 0].set_ylabel("Mean completion time (s)")
    axes[1, 0].set_xticks(users)
    axes[1, 0].grid(alpha=0.25)

    twenty = scaling["results"]["20"]["method_means"]
    x2 = np.arange(len(methods))
    inference = [twenty[label]["inference_ms_per_task"] for label in methods]
    cache = [twenty[label]["cache_decision_mean_ms"] for label in methods]
    axes[1, 1].bar(x2 - 0.18, inference, width=0.36, label="Policy inference")
    axes[1, 1].bar(x2 + 0.18, cache, width=0.36, label="Cache decision")
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels([DISPLAY[label] for label in methods], rotation=16)
    axes[1, 1].set_ylabel("Milliseconds per call")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"paper_supplement_overview.{suffix}", dpi=240)
    plt.close(figure)


def write_report(path, summary):
    hetero = summary["heterogeneity"]
    ablation = summary["ablation"]
    scaling = summary["scaling"]
    lines = [
        "# Pegasus论文必须补充实验总报告",
        "",
        f"- 协议：`{PROTOCOL_VERSION}`，算法保持P3冻结状态。",
        f"- 总门槛：`{summary['gate']['passed']}`。",
        "",
        "## 1. 固定预算容量异构对照",
        "",
        "| 环境 | 方法 | Mean | P95 | 命中率 | 远程加载率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in ("uniform_b10", "heterogeneous_b10"):
        for label in HETEROGENEITY_METHODS:
            values = hetero["profiles"][profile]["method_aggregates"][label]
            lines.append(
                f"| {profile} | {DISPLAY[label]} | "
                f"{values['mean_finish_time']:.6f} | "
                f"{values['mean_p95_finish_time']:.6f} | "
                f"{values['mean_cache_hit_rate']:.4f} | "
                f"{values['mean_remote_loading_rate']:.4f} |"
            )
    lines.extend(["", "## 2. 十Seed主要模块消融", ""])
    for key, comparison in ablation["paired_comparisons"].items():
        lines.append(
            f"- `{key}`：改善 `{comparison['mean_improvement_percent']:.3f}%`，"
            f"胜出 `{comparison['wins']}/10`，95% CI下界 "
            f"`{comparison['ci95_lower_sec']:.6f} s`，"
            f"p=`{comparison['wilcoxon_one_sided_p']:.6g}`，"
            f"pass=`{comparison['passed']}`。"
        )
    lines.extend(
        [
            "",
            "## 3. 冻结模型规模与开销",
            "",
            "| Users | 方法 | Mean | P95 | 推理ms | tasks/s | 缓存决策ms | 协调字节 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for users in scaling["user_counts"]:
        values = scaling["results"][str(users)]["method_means"]
        for label in HETEROGENEITY_METHODS:
            row = values[label]
            lines.append(
                f"| {users} | {DISPLAY[label]} | "
                f"{row['mean_finish_time']:.6f} | "
                f"{row['mean_p95_finish_time']:.6f} | "
                f"{row['inference_ms_per_task']:.4f} | "
                f"{row['tasks_per_wall_second']:.2f} | "
                f"{row['cache_decision_mean_ms']:.4f} | "
                f"{row['coordination_bytes']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## 写作结论",
            "",
            "- 主要创新只保留协调缓存与Pairwise PD3QN。",
            "- 因果遥测仍作为辅助机制，不提升为主要创新。",
            "- 突发负载与动态恢复不属于本文实验边界。",
            "- 缓存决策耗时和协调通信量来自单独微基准，不能再写为0。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent = validate_parent_freeze()
    heterogeneity = collect_heterogeneity(args.result_root)
    ablation = collect_ablation(args.result_root)
    scaling = read_json(args.result_root / "scaling/user_scaling_summary.json")
    final_context = collect_final_context()
    integrity = {
        "parent_frozen_hash_exact": bool(parent),
        "supplement_source_hash_recorded": bool(supplement_source_hash()),
        "heterogeneity_integrity": all(heterogeneity["integrity"].values()),
        "ablation_integrity": all(ablation["integrity"].values()),
        "scaling_methods_paired": scaling["all_methods_scenario_paired"],
        "scaling_20_user_exact": scaling[
            "reference_reproduction_audit"
        ]["all_exact"],
        "p3_final_gate_still_passed": final_context["gate"]["passed"],
    }
    gate = {
        "integrity": all(integrity.values()),
        "heterogeneity_control": heterogeneity["gate"]["passed"],
        "primary_ablation": ablation["gate"]["passed"],
        "scaling_complete": bool(
            scaling.get("status") == "complete"
            and scaling.get("all_methods_scenario_paired")
        ),
    }
    gate["passed"] = bool(all(gate.values()))
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "specification": specification(),
        "integrity": integrity,
        "heterogeneity": heterogeneity,
        "ablation": ablation,
        "scaling": scaling,
        "frozen_main_context": final_context,
        "gate": gate,
    }
    write_json(args.output_dir / "paper_supplement_summary.json", summary)
    write_report(args.output_dir / "PAPER_SUPPLEMENT_REPORT_ZH.md", summary)
    plot_main_statistics(args.output_dir, final_context)
    plot_convergence(args.output_dir)
    plot_workflow_mechanism(args.output_dir, final_context)
    plot_cache_heatmaps(args.output_dir)
    plot_supplement_overview(args.output_dir, heterogeneity, ablation, scaling)


if __name__ == "__main__":
    main()
