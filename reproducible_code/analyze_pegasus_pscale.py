#!/usr/bin/env python3
"""Analyze the governed three-seed Pegasus P-Scale experiment."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_a0_coordination import (
    DISPLAY_NAMES,
    aggregate_run,
    evaluation_rows,
    paired_superiority,
)
from capacity_protocol import deterministic_capacity_assignment
from pegasus_pscale_protocol import (
    CACHE_CALIBRATION_EPISODES,
    CAPACITY_PROFILES,
    DEVELOPMENT_SEEDS,
    EVALUATION_BANK_SCOPE,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    PROTOCOL_VERSION,
    VALIDATION_SCENARIOS,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


METHODS = ("guided_full", "centralized_greedy_daoc", "lean_our")
CAPACITY_MULTISET = CAPACITY_PROFILES["B8"]
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
    "oracle": "#2A9D8F",
    "perfect_cache": "#7B2CBF",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def oracle_references(oracle_dir):
    capacity_aware = {}
    perfect_cache = {}
    path = Path(oracle_dir) / "oracle_floor_per_seed.csv"
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            seed = int(row["seed"])
            capacity_aware[seed] = float(row["oracle_floor"])
            perfect_cache[seed] = float(row["perfect_cache_floor"])
    return capacity_aware, perfect_cache


def run_dir(suite_dir, label, seed):
    return suite_dir / "runs" / label / f"seed_{seed}"


def workload_view(bank):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def completion_audit(rows):
    return all(
        int(row["real_task_count"]) == int(row["completed_task_count"])
        and int(row["all_tasks_executed_once"]) == 1
        for row in rows
    )


def family_rows(rows, bank, family):
    episodes = {
        int(record["episode"])
        for record in bank
        if record.get("workflow_family") == family
    }
    return [row for row in rows if int(row["episode"]) in episodes]


def collect(args):
    per_seed = []
    per_family_seed = {
        label: {family: [] for family in FAMILIES}
        for label in METHODS
    }
    integrity = {
        "dataset_hash_exact": True,
        "completion_protocol_exact": True,
        "all_real_tasks_executed_once": True,
        "all_methods_converged": True,
        "scenario_banks_paired": True,
        "effective_scenarios_unique": True,
        "family_bank_exact_20_each": True,
        "capacity_assignments_exact": True,
        "total_budget_eight": True,
        "networks_and_caches_frozen_in_evaluation": True,
        "infrastructure_bank_scope_exact": True,
        "cache_calibration_horizon_exact": True,
        "validation_bank_size_exact": True,
        "task_limit_31": True,
        "bandwidth_15khz": True,
        "h8v1_modules_exact": True,
        "baseline_did_not_receive_coverage_constraint": True,
    }

    for seed in DEVELOPMENT_SEEDS:
        expected_capacity = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        methods = {}
        banks = []
        checkpoints = {}
        for label in METHODS:
            directory = run_dir(args.suite_dir, label, seed)
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            rows = evaluation_rows(directory / "episodes.csv")
            bank = read_json(directory / "evaluation_scenarios.json")
            methods[label] = aggregate_run(rows)
            checkpoints[label] = summary["selected_checkpoint_episode"]

            integrity["dataset_hash_exact"] &= (
                summary.get("dag_dataset", {}).get("sha256")
                == EXPECTED_DATASET_SHA256
            )
            integrity["completion_protocol_exact"] &= (
                summary.get("dag_completion_protocol_version")
                == DAG_COMPLETION_PROTOCOL_VERSION
            )
            integrity["all_real_tasks_executed_once"] &= completion_audit(rows)
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
            integrity["total_budget_eight"] &= sum(capacities.values()) == 8
            integrity["networks_and_caches_frozen_in_evaluation"] &= bool(
                summary.get("evaluation_state_frozen")
                and summary.get("evaluation_scenario_count") == 100
            )
            integrity["effective_scenarios_unique"] &= bool(
                summary.get("evaluation_unique_base_scenarios") == 100
                and len({row["base_fingerprint"] for row in bank}) == 100
            )
            arguments = config.get("arguments", {})
            integrity["task_limit_31"] &= arguments.get("num_tasks") == 31
            integrity["bandwidth_15khz"] &= arguments.get("bandwidth") == 15000
            integrity["infrastructure_bank_scope_exact"] &= (
                arguments.get("eval_bank_scope") == EVALUATION_BANK_SCOPE
            )
            integrity["cache_calibration_horizon_exact"] &= (
                arguments.get("cache_freeze_episode")
                == CACHE_CALIBRATION_EPISODES
                and summary["selected_checkpoint_episode"]
                > CACHE_CALIBRATION_EPISODES
            )
            integrity["validation_bank_size_exact"] &= (
                arguments.get("validation_scenarios")
                == VALIDATION_SCENARIOS
            )
            family_counts = Counter(
                record.get("workflow_family") for record in bank
            )
            integrity["family_bank_exact_20_each"] &= (
                family_counts == Counter({family: 20 for family in FAMILIES})
            )
            for family in FAMILIES:
                expected_key = f"pegasus_full_{family.lower()}"
                selected = [
                    record for record in bank
                    if record.get("workflow_family") == family
                ]
                integrity["family_bank_exact_20_each"] &= all(
                    set(record["user_graph_keys"].values()) == {expected_key}
                    for record in selected
                )
                integrity["family_bank_exact_20_each"] &= (
                    len({record["base_fingerprint"] for record in selected})
                    == 20
                )
                selected_rows = family_rows(rows, bank, family)
                if len(selected_rows) != 20:
                    raise RuntimeError(
                        f"Expected 20 {family} rows in {directory}"
                    )
                per_family_seed[label][family].append(
                    aggregate_run(selected_rows)
                )

            if label == "lean_our":
                modules = set(
                    summary.get("method_modules", {}).get("active", [])
                )
                expected_modules = {
                    "pairwise_dueling_double_dqn",
                    "causal_history_telemetry",
                    "causal_dependency_aware_joint_cache",
                    "scarcity_aware_service_coverage_constraint",
                }
                integrity["h8v1_modules_exact"] &= (
                    expected_modules.issubset(modules)
                    and arguments.get("cache_coverage_constraint") is True
                )
            else:
                integrity[
                    "baseline_did_not_receive_coverage_constraint"
                ] &= not bool(
                    arguments.get("cache_coverage_constraint", False)
                )
            banks.append(workload_view(bank))
        integrity["scenario_banks_paired"] &= all(
            bank == banks[0] for bank in banks[1:]
        )
        per_seed.append(
            {
                "seed": seed,
                "methods": methods,
                "selected_checkpoint_episode": checkpoints,
            }
        )

    aggregates = {
        label: {
            metric: float(
                np.mean([row["methods"][label][metric] for row in per_seed])
            )
            for metric in per_seed[0]["methods"][label]
        }
        for label in METHODS
    }
    family_aggregates = {
        label: {
            family: {
                metric: float(np.mean([row[metric] for row in records]))
                for metric in records[0]
            }
            for family, records in families.items()
        }
        for label, families in per_family_seed.items()
    }
    comparisons = {
        f"our_vs_{reference}": paired_superiority(
            [row["methods"][reference]["mean_finish_time"] for row in per_seed],
            [row["methods"]["lean_our"]["mean_finish_time"] for row in per_seed],
            formal=False,
        )
        for reference in ("guided_full", "centralized_greedy_daoc")
    }
    p95 = paired_superiority(
        [
            row["methods"]["centralized_greedy_daoc"][
                "mean_p95_finish_time"
            ]
            for row in per_seed
        ],
        [row["methods"]["lean_our"]["mean_p95_finish_time"] for row in per_seed],
        formal=False,
    )
    capacity_oracle, perfect_cache_floor = oracle_references(
        args.suite_dir / "oracle"
    )
    if (
        set(capacity_oracle) != set(DEVELOPMENT_SEEDS)
        or set(perfect_cache_floor) != set(DEVELOPMENT_SEEDS)
    ):
        raise RuntimeError("Oracle seed mismatch")
    capacity_oracle_mean = float(np.mean(list(capacity_oracle.values())))
    perfect_cache_mean = float(np.mean(list(perfect_cache_floor.values())))
    our_mean = aggregates["lean_our"]["mean_finish_time"]
    gate = {
        "integrity": all(integrity.values()),
        "our_beats_daoc_mean_and_two_of_three": comparisons[
            "our_vs_guided_full"
        ]["passed"],
        "our_beats_central_mean_and_two_of_three": comparisons[
            "our_vs_centralized_greedy_daoc"
        ]["passed"],
        "p95_beats_central_mean_and_two_of_three": p95["passed"],
    }
    gate["passed"] = all(gate.values())
    return {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": "controlled_pegasus_pscale_three_seed_development",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "seeds": list(DEVELOPMENT_SEEDS),
        "families": list(FAMILIES),
        "capacity_multiset": list(CAPACITY_MULTISET),
        "integrity": integrity,
        "method_aggregates": aggregates,
        "family_aggregates": family_aggregates,
        "paired_comparisons": comparisons,
        "p95_our_vs_central": p95,
        "oracle": {
            "capacity_aware_diagnostic": {
                "mean": capacity_oracle_mean,
                "our_gap_sec": our_mean - capacity_oracle_mean,
                "our_gap_percent": (
                    100
                    * (our_mean - capacity_oracle_mean)
                    / capacity_oracle_mean
                ),
                "certified_global_floor": False,
            },
            "certified_perfect_cache_floor": {
                "mean": perfect_cache_mean,
                "our_gap_sec": our_mean - perfect_cache_mean,
                "our_gap_percent": (
                    100 * (our_mean - perfect_cache_mean) / perfect_cache_mean
                ),
                "certified_global_floor": True,
            },
        },
        "gate": gate,
        "per_seed": per_seed,
    }


def plot_results(output_dir, summary):
    methods = list(METHODS)
    display = [DISPLAY_NAMES[label] for label in methods]
    x = np.arange(len(methods))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    means = [
        summary["method_aggregates"][label]["mean_finish_time"]
        for label in methods
    ]
    p95 = [
        summary["method_aggregates"][label]["mean_p95_finish_time"]
        for label in methods
    ]
    axes[0, 0].bar(x, means, color=[COLORS[label] for label in methods])
    axes[0, 0].axhline(
        summary["oracle"]["capacity_aware_diagnostic"]["mean"],
        color=COLORS["oracle"],
        linestyle="--",
        label="Capacity-aware oracle (diagnostic)",
    )
    axes[0, 0].axhline(
        summary["oracle"]["certified_perfect_cache_floor"]["mean"],
        color=COLORS["perfect_cache"],
        linestyle=":",
        label="Certified perfect-cache floor",
    )
    axes[0, 0].set_xticks(x, display, rotation=12)
    axes[0, 0].set_ylabel("Mean completion time (s)")
    axes[0, 0].set_title("P-Scale mean completion")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x, p95, color=[COLORS[label] for label in methods])
    axes[0, 1].set_xticks(x, display, rotation=12)
    axes[0, 1].set_ylabel("P95 completion time (s)")
    axes[0, 1].set_title("P-Scale tail latency")

    family_x = np.arange(len(FAMILIES))
    width = 0.25
    for index, label in enumerate(methods):
        values = [
            summary["family_aggregates"][label][family][
                "mean_finish_time"
            ]
            for family in FAMILIES
        ]
        axes[1, 0].bar(
            family_x + (index - 1) * width,
            values,
            width,
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
        )
    axes[1, 0].set_xticks(family_x, FAMILIES, rotation=15)
    axes[1, 0].set_ylabel("Mean completion time (s)")
    axes[1, 0].set_title("Per-family completion")
    axes[1, 0].legend(frameon=False)

    hit = [
        summary["method_aggregates"][label]["mean_cache_hit_rate"]
        for label in methods
    ]
    remote = [
        summary["method_aggregates"][label]["mean_remote_loading_rate"]
        for label in methods
    ]
    axes[1, 1].bar(x - 0.18, hit, 0.36, label="Cache hit")
    axes[1, 1].bar(x + 0.18, remote, 0.36, label="Remote load")
    axes[1, 1].set_xticks(x, display, rotation=12)
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Cache mechanism")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"pegasus_pscale_results.{suffix}", dpi=180)
    plt.close(figure)


def render_report(output_dir, summary):
    aggregates = summary["method_aggregates"]
    daoc = summary["paired_comparisons"]["our_vs_guided_full"]
    central = summary["paired_comparisons"][
        "our_vs_centralized_greedy_daoc"
    ]
    p95 = summary["p95_our_vs_central"]
    family_lines = []
    for family in FAMILIES:
        family_lines.append(
            "| "
            + family
            + " | "
            + " | ".join(
                f"{summary['family_aggregates'][label][family]['mean_finish_time']:.6f}"
                for label in METHODS
            )
            + " |"
        )
    report = f"""# Pegasus P-Scale 三 seed 主实验

> 这是修正多出口终止语义后的受控规模实验，不是无偏 MEC holdout。

## 完整性

- 所有真实任务恰好执行一次：`{summary['integrity']['all_real_tasks_executed_once']}`。
- 每 seed 的 100 个基础场景均不同，五类各 20 个；服务器基础设施固定，用户位置/入口按场景配对重采样。
- 各方法先用 5,000 轮因果历史校准缓存，随后固定缓存并继续训练到平台收敛。
- 数据、收敛、容量、场景配对与五类各 20 场景审计：`{all(summary['integrity'].values())}`。
- Seeds：`{summary['seeds']}`；容量：`{summary['capacity_multiset']}`。

## 总体结果

| 方法 | 平均完成时间 | P95 | 缓存命中率 | 远程加载率 | 等待时延 |
|---|---:|---:|---:|---:|---:|
| DAOC | {aggregates['guided_full']['mean_finish_time']:.6f} | {aggregates['guided_full']['mean_p95_finish_time']:.6f} | {aggregates['guided_full']['mean_cache_hit_rate']:.4f} | {aggregates['guided_full']['mean_remote_loading_rate']:.4f} | {aggregates['guided_full']['mean_waiting_latency']:.6f} |
| Centralized-Greedy-DQN | {aggregates['centralized_greedy_daoc']['mean_finish_time']:.6f} | {aggregates['centralized_greedy_daoc']['mean_p95_finish_time']:.6f} | {aggregates['centralized_greedy_daoc']['mean_cache_hit_rate']:.4f} | {aggregates['centralized_greedy_daoc']['mean_remote_loading_rate']:.4f} | {aggregates['centralized_greedy_daoc']['mean_waiting_latency']:.6f} |
| OUR | {aggregates['lean_our']['mean_finish_time']:.6f} | {aggregates['lean_our']['mean_p95_finish_time']:.6f} | {aggregates['lean_our']['mean_cache_hit_rate']:.4f} | {aggregates['lean_our']['mean_remote_loading_rate']:.4f} | {aggregates['lean_our']['mean_waiting_latency']:.6f} |
| Capacity-aware clairvoyant Oracle (diagnostic) | {summary['oracle']['capacity_aware_diagnostic']['mean']:.6f} | - | - | - | - |
| Certified perfect-cache recurrence floor | {summary['oracle']['certified_perfect_cache_floor']['mean']:.6f} | - | - | - | - |

- OUR vs DAOC：`{daoc['mean_improvement_percent']:.3f}%`，`{daoc['wins']}/3` seed 胜出。
- OUR vs Centralized-Greedy-DQN：`{central['mean_improvement_percent']:.3f}%`，`{central['wins']}/3` seed 胜出。
- P95 vs Centralized-Greedy-DQN：`{p95['mean_improvement_percent']:.3f}%`，`{p95['wins']}/3` seed 胜出。
- OUR 高于容量感知 clairvoyant 诊断值：`{summary['oracle']['capacity_aware_diagnostic']['our_gap_sec']:.6f} s`（`{summary['oracle']['capacity_aware_diagnostic']['our_gap_percent']:.3f}%`）。
- OUR 高于认证 perfect-cache recurrence 严格下界：`{summary['oracle']['certified_perfect_cache_floor']['our_gap_sec']:.6f} s`（`{summary['oracle']['certified_perfect_cache_floor']['our_gap_percent']:.3f}%`）。

容量感知 clairvoyant Oracle 使用已知场景做放松分配，只用于诊断，不称为认证全局下界。
`perfect-cache recurrence` 才是本报告中的严格认证下界。

## 分工作流结果

| Family | DAOC | Centralized-Greedy-DQN | OUR |
|---|---:|---:|---:|
{chr(10).join(family_lines)}

## 三 seed 门槛

- 完整性：`{summary['gate']['integrity']}`。
- OUR 优于 DAOC：`{summary['gate']['our_beats_daoc_mean_and_two_of_three']}`。
- OUR 优于 Centralized-Greedy-DQN：`{summary['gate']['our_beats_central_mean_and_two_of_three']}`。
- P95 优于 Centralized-Greedy-DQN：`{summary['gate']['p95_beats_central_mean_and_two_of_three']}`。
- 主实验门槛：`{summary['gate']['passed']}`。

三 seed 只用于判断是否进入 B5/B8/B10 敏感性和十 seed 阶段，不用于正式显著性宣称。
"""
    (output_dir / "PEGASUS_PSCALE_REPORT_ZH.md").write_text(
        report,
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect(args)
    write_json(args.output_dir / "pegasus_pscale_summary.json", summary)
    plot_results(args.output_dir, summary)
    render_report(args.output_dir, summary)
    print(f"Pegasus P-Scale analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
