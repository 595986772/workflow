#!/usr/bin/env python3
"""Analyze paired static A0 coordination experiments."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from a0_coordination_protocol import (
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    EXPECTED_DATASET_SHA256,
    METHOD_LABELS,
)
from capacity_protocol import deterministic_capacity_assignment


DISPLAY_NAMES = {
    "guided_full": "DAOC",
    "centralized_greedy_daoc": "Centralized-Greedy",
    "lean_our": "OUR",
}
COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
    "oracle": "#2A9D8F",
}


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", choices=CAPACITY_PROFILES, required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_dir(suite_dir, label, seed):
    return suite_dir / "runs" / label / f"seed_{seed}"


def evaluation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 frozen scenarios in {path}")
    return rows


def aggregate_run(rows):
    metrics = {
        "average_finish_time": "mean_finish_time",
        "p95_finish_time": "mean_p95_finish_time",
        "waiting_latency": "mean_waiting_latency",
        "cache_hit_rate": "mean_cache_hit_rate",
        "cache_service_coverage": "mean_cache_service_coverage",
        "cache_remote_loading_rate": "mean_remote_loading_rate",
        "service_latency": "mean_service_loading_latency",
        "policy_inference_time_per_decision_ms": (
            "mean_inference_time_per_decision_ms"
        ),
        "cache_decision_wall_time_sec": "cache_decision_wall_time_sec",
        "cache_migration_time_sec": "cache_migration_time_sec",
    }
    return {
        output: float(np.mean([float(row[source]) for row in rows]))
        for source, output in metrics.items()
    }


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return mean - half, mean + half


def paired_superiority(reference, ours, formal):
    reference = np.asarray(reference, dtype=float)
    ours = np.asarray(ours, dtype=float)
    improvements = reference - ours
    lower, upper = confidence_interval(improvements)
    nonzero = improvements[np.abs(improvements) > 1e-12]
    p_value = (
        float(stats.wilcoxon(nonzero, alternative="greater").pvalue)
        if len(nonzero)
        else 1.0
    )
    wins = int(np.sum(improvements > 0))
    passed = (
        lower > 0 and p_value < 0.05 and wins >= 7
        if formal
        else improvements.mean() > 0 and wins >= 2
    )
    return {
        "pairs": len(improvements),
        "reference_mean": float(reference.mean()),
        "our_mean": float(ours.mean()),
        "mean_improvement_sec": float(improvements.mean()),
        "mean_improvement_percent": float(
            100.0 * improvements.mean() / reference.mean()
        ),
        "ci95_lower_sec": lower,
        "ci95_upper_sec": upper,
        "wilcoxon_one_sided_p": p_value,
        "wins": wins,
        "passed": bool(passed),
    }


def oracle_per_seed(oracle_dir):
    path = oracle_dir / "oracle_floor_per_seed.csv"
    with path.open(newline="", encoding="utf-8") as input_file:
        return {
            int(row["seed"]): float(row["oracle_floor"])
            for row in csv.DictReader(input_file)
        }


def collect(args):
    expected_multiset = CAPACITY_PROFILES[args.budget]
    per_seed = []
    integrity = {
        "dataset_hash": True,
        "all_methods_converged": True,
        "scenario_banks_paired": True,
        "capacity_assignments_exact": True,
        "total_budget_exact": True,
        "networks_frozen_in_evaluation": True,
    }
    heatmaps = {
        label: np.zeros((10, 10), dtype=float)
        for label in METHOD_LABELS
    }
    for seed in args.seeds:
        method_values = {}
        scenario_banks = []
        expected_capacity = deterministic_capacity_assignment(
            expected_multiset,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
        )
        for label in METHOD_LABELS:
            directory = run_dir(args.suite_dir, label, seed)
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            rows = evaluation_rows(directory / "episodes.csv")
            method_values[label] = aggregate_run(rows)
            integrity["dataset_hash"] &= (
                summary.get("dag_dataset", {}).get("sha256")
                == EXPECTED_DATASET_SHA256
            )
            integrity["all_methods_converged"] &= bool(
                summary.get("eligible_for_comparison")
                and summary.get("convergence", {}).get("reached")
            )
            capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            integrity["capacity_assignments_exact"] &= (
                capacities == expected_capacity
            )
            integrity["total_budget_exact"] &= (
                sum(capacities.values()) == sum(expected_multiset)
            )
            integrity["networks_frozen_in_evaluation"] &= bool(
                summary.get("evaluation_state_frozen")
            )
            scenario_banks.append(
                read_json(directory / "evaluation_scenarios.json")
            )
            matrix = json.loads(rows[-1]["cache_matrix_json"])
            for server_id, services in matrix.items():
                for service in services:
                    heatmaps[label][int(server_id), int(service) - 1] += 1
            if config["arguments"].get("bandwidth") != 15000:
                raise RuntimeError(f"Wrong bandwidth in {directory}")
        integrity["scenario_banks_paired"] &= all(
            bank == scenario_banks[0] for bank in scenario_banks[1:]
        )
        per_seed.append(
            {
                "seed": seed,
                "methods": method_values,
                "selected_checkpoint_episode": {
                    label: read_json(
                        run_dir(args.suite_dir, label, seed)
                        / "summary.json"
                    )["selected_checkpoint_episode"]
                    for label in METHOD_LABELS
                },
            }
        )
    return per_seed, integrity, heatmaps


def analyze(args):
    args.suite_dir = args.suite_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, integrity, heatmaps = collect(args)
    formal = args.mode == "final"
    comparisons = {}
    for reference_label in (
        "guided_full",
        "centralized_greedy_daoc",
    ):
        comparisons[f"our_vs_{reference_label}"] = paired_superiority(
            [
                row["methods"][reference_label]["mean_finish_time"]
                for row in per_seed
            ],
            [
                row["methods"]["lean_our"]["mean_finish_time"]
                for row in per_seed
            ],
            formal=formal,
        )
    method_aggregates = {
        label: {
            metric: float(
                np.mean(
                    [row["methods"][label][metric] for row in per_seed]
                )
            )
            for metric in per_seed[0]["methods"][label]
        }
        for label in METHOD_LABELS
    }
    oracle_path = args.suite_dir / "oracle"
    oracle = oracle_per_seed(oracle_path)
    if set(oracle) != set(args.seeds):
        raise RuntimeError("Oracle seeds do not match the static suite")
    oracle_mean = float(np.mean([oracle[seed] for seed in args.seeds]))
    our_mean = method_aggregates["lean_our"]["mean_finish_time"]
    gate = {
        "integrity": all(integrity.values()),
        "our_beats_daoc": comparisons["our_vs_guided_full"]["passed"],
        "our_beats_centralized_greedy": comparisons[
            "our_vs_centralized_greedy_daoc"
        ]["passed"],
    }
    gate["passed"] = bool(all(gate.values()))
    result = {
        "status": "complete",
        "mode": args.mode,
        "budget_profile": args.budget,
        "capacity_multiset": CAPACITY_PROFILES[args.budget],
        "seeds": args.seeds,
        "integrity": integrity,
        "method_aggregates": method_aggregates,
        "oracle": {
            "mean": oracle_mean,
            "our_gap_sec": our_mean - oracle_mean,
            "our_gap_percent_of_oracle": (
                100.0 * (our_mean - oracle_mean) / oracle_mean
            ),
            "clairvoyant_diagnostic_only": True,
        },
        "paired_superiority": comparisons,
        "per_seed": per_seed,
        "gate": gate,
        "claim_scope": "A0_controlled_mechanism_only",
    }
    write_json(args.output_dir / "static_summary.json", result)
    plot_main(args.output_dir, result)
    plot_heatmaps(args.output_dir, heatmaps, len(args.seeds))
    write_report(args.output_dir / "STATIC_REPORT_ZH.md", result)
    return result


def plot_main(output_dir, result):
    labels = METHOD_LABELS + ["oracle"]
    values = [
        result["method_aggregates"][label]["mean_finish_time"]
        for label in METHOD_LABELS
    ] + [result["oracle"]["mean"]]
    names = [DISPLAY_NAMES[label] for label in METHOD_LABELS] + ["Oracle"]
    figure, axis = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    axis.bar(
        names,
        values,
        color=[COLORS[label] for label in labels],
    )
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "static_main.png", dpi=220)
    figure.savefig(output_dir / "static_main.pdf")
    plt.close(figure)


def plot_heatmaps(output_dir, heatmaps, seeds):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.1),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, label in zip(axes, METHOD_LABELS):
        image = axis.imshow(
            heatmaps[label] / seeds,
            vmin=0,
            vmax=1,
            cmap="viridis",
            aspect="auto",
        )
        axis.set_title(DISPLAY_NAMES[label])
        axis.set_xlabel("Service")
        axis.set_xticks(range(10), range(1, 11))
    axes[0].set_ylabel("Server")
    figure.colorbar(image, ax=axes, label="Placement frequency")
    figure.savefig(output_dir / "cache_heatmaps.png", dpi=220)
    figure.savefig(output_dir / "cache_heatmaps.pdf")
    plt.close(figure)


def write_report(path, result):
    lines = [
        f"# A0静态实验：{result['budget_profile']}",
        "",
        "A0是受控服务对齐消融数据集，不得称为无偏Alibaba holdout。",
        "",
        "| 方法 | 平均完成时间 | P95 | 等待时延 | 命中率 | 覆盖率 | 远程加载率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in METHOD_LABELS:
        row = result["method_aggregates"][label]
        lines.append(
            f"| {DISPLAY_NAMES[label]} | {row['mean_finish_time']:.6f} | "
            f"{row['mean_p95_finish_time']:.6f} | "
            f"{row['mean_waiting_latency']:.6f} | "
            f"{row['mean_cache_hit_rate']:.4f} | "
            f"{row['mean_cache_service_coverage']:.4f} | "
            f"{row['mean_remote_loading_rate']:.4f} |"
        )
    lines.extend(
        [
            f"| Oracle | {result['oracle']['mean']:.6f} | - | - | - | - | - |",
            "",
            f"OUR距Oracle容量感知诊断参考还有 "
            f"{result['oracle']['our_gap_sec']:.6f} s "
            f"({result['oracle']['our_gap_percent_of_oracle']:.3f}%)。",
            "",
            "## 门槛",
            "",
        ]
    )
    for key, comparison in result["paired_superiority"].items():
        lines.append(
            f"- `{key}`: 改善 {comparison['mean_improvement_percent']:.3f}%，"
            f"CI下界 {comparison['ci95_lower_sec']:.6f} s，"
            f"p={comparison['wilcoxon_one_sided_p']:.6g}，"
            f"胜出 {comparison['wins']}/{comparison['pairs']}，"
            f"通过={comparison['passed']}。"
        )
    lines.append(f"\n总门槛通过：`{result['gate']['passed']}`。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    result = analyze(args)
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
