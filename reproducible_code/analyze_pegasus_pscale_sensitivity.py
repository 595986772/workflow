#!/usr/bin/env python3
"""Analyze B5/B8/B10 Pegasus P-Scale cache-budget sensitivity."""

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
    CAPACITY_PROFILES,
    DEVELOPMENT_SEEDS,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


METHODS = ("guided_full", "centralized_greedy_daoc", "lean_our")
PROFILES = ("B5", "B8", "B10")
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
PROTOCOL_VERSION = "pegasus_pscale_p2_budget_sensitivity_v1"
COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
    "oracle": "#2A9D8F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--main-suite-dir", type=Path, required=True)
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


def profile_dir(args, profile):
    if profile == "B8":
        return args.main_suite_dir
    return args.sensitivity_dir / profile


def run_dir(args, profile, label, seed):
    return profile_dir(args, profile) / "runs" / label / f"seed_{seed}"


def workload_view(bank):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def infrastructure_view(snapshot):
    return {
        "servers": {
            key: {
                field: server[field]
                for field in ("position", "frequency", "load", "rate_to_cloud")
            }
            for key, server in snapshot["servers"].items()
        },
        "between_server_costs": snapshot["between_server_costs"],
        "service_data_length": snapshot["service_data_length"],
    }


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


def completion_audit(rows):
    return all(
        int(row["real_task_count"]) == int(row["completed_task_count"])
        and int(row["all_tasks_executed_once"]) == 1
        for row in rows
    )


def collect(args):
    integrity = {
        "main_B8_gate_passed": True,
        "all_methods_converged": True,
        "dataset_hash_exact": True,
        "completion_protocol_exact": True,
        "all_real_tasks_executed_once": True,
        "capacity_assignments_exact": True,
        "total_budgets_exact": True,
        "scenario_banks_paired_within_budget": True,
        "workloads_paired_across_budgets": True,
        "infrastructure_paired_across_budgets": True,
        "effective_scenarios_unique": True,
        "family_bank_exact_20_each": True,
        "environment_20_10_10_31_15khz": True,
        "evaluation_state_frozen": True,
    }
    main_summary = read_json(
        args.main_suite_dir / "analysis/pegasus_pscale_summary.json"
    )
    integrity["main_B8_gate_passed"] = bool(
        main_summary.get("gate", {}).get("passed")
    )
    per_profile_seed = {profile: [] for profile in PROFILES}
    cross_budget_banks = {seed: {} for seed in DEVELOPMENT_SEEDS}
    cross_budget_infrastructure = {seed: {} for seed in DEVELOPMENT_SEEDS}

    for profile in PROFILES:
        capacities = CAPACITY_PROFILES[profile]
        budget = int(profile[1:])
        for seed in DEVELOPMENT_SEEDS:
            expected_capacity = deterministic_capacity_assignment(
                capacities,
                number_of_servers=10,
                number_of_services=10,
                seed=seed,
                assignment_namespace=CAPACITY_NAMESPACE,
            )
            methods = {}
            method_banks = []
            for label in METHODS:
                directory = run_dir(args, profile, label, seed)
                summary = read_json(directory / "summary.json")
                config = read_json(directory / "config.json")["arguments"]
                rows = evaluation_rows(directory / "episodes.csv")
                bank = read_json(directory / "evaluation_scenarios.json")
                methods[label] = aggregate_run(rows)

                integrity["all_methods_converged"] &= bool(
                    summary.get("eligible_for_comparison")
                    and summary.get("convergence", {}).get("reached")
                )
                integrity["dataset_hash_exact"] &= (
                    summary.get("dag_dataset", {}).get("sha256")
                    == EXPECTED_DATASET_SHA256
                )
                integrity["completion_protocol_exact"] &= (
                    summary.get("dag_completion_protocol_version")
                    == DAG_COMPLETION_PROTOCOL_VERSION
                )
                integrity["all_real_tasks_executed_once"] &= (
                    len(rows) == EVALUATION_EPISODES
                    and completion_audit(rows)
                )
                observed_capacity = {
                    int(key): int(value)
                    for key, value in summary["server_capacities"].items()
                }
                integrity["capacity_assignments_exact"] &= (
                    observed_capacity == expected_capacity
                )
                integrity["total_budgets_exact"] &= (
                    sum(observed_capacity.values()) == budget
                )
                integrity["effective_scenarios_unique"] &= (
                    summary.get("evaluation_scenario_count")
                    == EVALUATION_EPISODES
                    and summary.get("evaluation_unique_base_scenarios")
                    == EVALUATION_EPISODES
                    and len({row["base_fingerprint"] for row in bank})
                    == EVALUATION_EPISODES
                )
                family_counts = Counter(
                    row.get("workflow_family") for row in bank
                )
                integrity["family_bank_exact_20_each"] &= (
                    family_counts
                    == Counter({family: 20 for family in FAMILIES})
                )
                integrity["environment_20_10_10_31_15khz"] &= (
                    config.get("num_users") == 20
                    and config.get("num_servers") == 10
                    and config.get("num_services") == 10
                    and config.get("num_tasks")
                    == TASK_LIMIT_INCLUDING_DUMMY
                    and config.get("bandwidth") == 15000
                )
                integrity["evaluation_state_frozen"] &= bool(
                    summary.get("evaluation_state_frozen")
                )
                method_banks.append(workload_view(bank))
            integrity["scenario_banks_paired_within_budget"] &= all(
                bank == method_banks[0] for bank in method_banks[1:]
            )
            cross_budget_banks[seed][profile] = method_banks[0]
            initial = read_json(
                run_dir(args, profile, "lean_our", seed)
                / "scenario_initial.json"
            )
            cross_budget_infrastructure[seed][profile] = (
                infrastructure_view(initial)
            )
            per_profile_seed[profile].append(
                {"seed": seed, "methods": methods}
            )

    for seed in DEVELOPMENT_SEEDS:
        banks = [cross_budget_banks[seed][profile] for profile in PROFILES]
        integrity["workloads_paired_across_budgets"] &= all(
            bank == banks[0] for bank in banks[1:]
        )
        deployments = [
            cross_budget_infrastructure[seed][profile]
            for profile in PROFILES
        ]
        integrity["infrastructure_paired_across_budgets"] &= all(
            deployment == deployments[0]
            for deployment in deployments[1:]
        )

    aggregates = {
        profile: {
            label: {
                metric: float(np.mean([
                    row["methods"][label][metric]
                    for row in per_profile_seed[profile]
                ]))
                for metric in per_profile_seed[profile][0]["methods"][label]
            }
            for label in METHODS
        }
        for profile in PROFILES
    }
    comparisons = {}
    oracle = {}
    for profile in PROFILES:
        rows = per_profile_seed[profile]
        comparisons[profile] = {
            f"our_vs_{reference}": paired_superiority(
                [
                    row["methods"][reference]["mean_finish_time"]
                    for row in rows
                ],
                [
                    row["methods"]["lean_our"]["mean_finish_time"]
                    for row in rows
                ],
                formal=False,
            )
            for reference in ("guided_full", "centralized_greedy_daoc")
        }
        comparisons[profile]["p95_our_vs_central"] = paired_superiority(
            [
                row["methods"]["centralized_greedy_daoc"][
                    "mean_p95_finish_time"
                ]
                for row in rows
            ],
            [
                row["methods"]["lean_our"]["mean_p95_finish_time"]
                for row in rows
            ],
            formal=False,
        )
        capacity_floor, perfect_cache_floor = oracle_references(
            profile_dir(args, profile) / "oracle"
        )
        if (
            set(capacity_floor) != set(DEVELOPMENT_SEEDS)
            or set(perfect_cache_floor) != set(DEVELOPMENT_SEEDS)
        ):
            raise RuntimeError(f"Oracle seed mismatch for {profile}")
        capacity_mean = float(np.mean(list(capacity_floor.values())))
        perfect_cache_mean = float(
            np.mean(list(perfect_cache_floor.values()))
        )
        our_mean = aggregates[profile]["lean_our"]["mean_finish_time"]
        oracle[profile] = {
            "capacity_aware_diagnostic": {
                "mean": capacity_mean,
                "our_gap_sec": our_mean - capacity_mean,
                "our_gap_percent": (
                    100 * (our_mean - capacity_mean) / capacity_mean
                ),
                "certified_global_floor": False,
            },
            "certified_perfect_cache_floor": {
                "mean": perfect_cache_mean,
                "our_gap_sec": our_mean - perfect_cache_mean,
                "our_gap_percent": (
                    100
                    * (our_mean - perfect_cache_mean)
                    / perfect_cache_mean
                ),
                "certified_global_floor": True,
            },
        }

    gate = {
        "integrity": all(integrity.values()),
        "our_beats_daoc_at_all_budgets": all(
            comparisons[profile]["our_vs_guided_full"]["passed"]
            for profile in PROFILES
        ),
        "our_beats_central_at_all_budgets": all(
            comparisons[profile][
                "our_vs_centralized_greedy_daoc"
            ]["passed"]
            for profile in PROFILES
        ),
        "p95_beats_central_at_all_budgets": all(
            comparisons[profile]["p95_our_vs_central"]["passed"]
            for profile in PROFILES
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": "controlled_pegasus_pscale_three_seed_sensitivity",
        "profiles": list(PROFILES),
        "capacity_multisets": {
            profile: list(CAPACITY_PROFILES[profile])
            for profile in PROFILES
        },
        "seeds": list(DEVELOPMENT_SEEDS),
        "integrity": integrity,
        "method_aggregates": aggregates,
        "paired_comparisons": comparisons,
        "oracle": oracle,
        "per_profile_seed": per_profile_seed,
        "gate": gate,
        "ten_seed_recommended": bool(gate["passed"]),
    }


def plot_results(output_dir, summary):
    budgets = [5, 8, 10]
    x = np.arange(len(budgets))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label in METHODS:
        axes[0, 0].plot(
            x,
            [
                summary["method_aggregates"][profile][label][
                    "mean_finish_time"
                ]
                for profile in PROFILES
            ],
            marker="o",
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
        )
        axes[0, 1].plot(
            x,
            [
                summary["method_aggregates"][profile][label][
                    "mean_p95_finish_time"
                ]
                for profile in PROFILES
            ],
            marker="o",
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
        )
        axes[1, 0].plot(
            x,
            [
                summary["method_aggregates"][profile][label][
                    "mean_cache_hit_rate"
                ]
                for profile in PROFILES
            ],
            marker="o",
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
        )
        axes[1, 1].plot(
            x,
            [
                summary["method_aggregates"][profile][label][
                    "mean_waiting_latency"
                ]
                for profile in PROFILES
            ],
            marker="o",
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
        )
    axes[0, 0].plot(
        x,
        [
            summary["oracle"][profile]["capacity_aware_diagnostic"][
                "mean"
            ]
            for profile in PROFILES
        ],
        marker="o",
        linestyle="--",
        label="Capacity-aware oracle (diagnostic)",
        color=COLORS["oracle"],
    )
    axes[0, 0].plot(
        x,
        [
            summary["oracle"][profile][
                "certified_perfect_cache_floor"
            ]["mean"]
            for profile in PROFILES
        ],
        marker="x",
        linestyle=":",
        label="Certified perfect-cache floor",
        color="#7B2CBF",
    )
    titles = (
        "Mean completion time",
        "P95 completion time",
        "Cache hit rate",
        "Waiting latency",
    )
    ylabels = ("Seconds", "Seconds", "Rate", "Seconds")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_xticks(x, budgets)
        axis.set_xlabel("Total cache budget")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.savefig(output_dir / "pegasus_pscale_budget_sensitivity.png", dpi=180)
    figure.savefig(output_dir / "pegasus_pscale_budget_sensitivity.pdf")
    plt.close(figure)


def write_report(output_dir, summary):
    rows = []
    for profile in PROFILES:
        for label in METHODS:
            metrics = summary["method_aggregates"][profile][label]
            rows.append(
                f"| {profile} | {DISPLAY_NAMES[label]} | "
                f"{metrics['mean_finish_time']:.6f} | "
                f"{metrics['mean_p95_finish_time']:.6f} | "
                f"{metrics['mean_cache_hit_rate']:.4f} | "
                f"{metrics['mean_remote_loading_rate']:.4f} | "
                f"{metrics['mean_waiting_latency']:.6f} |"
            )
    comparisons = []
    for profile in PROFILES:
        daoc = summary["paired_comparisons"][profile]["our_vs_guided_full"]
        central = summary["paired_comparisons"][profile][
            "our_vs_centralized_greedy_daoc"
        ]
        tail = summary["paired_comparisons"][profile][
            "p95_our_vs_central"
        ]
        capacity_oracle = summary["oracle"][profile][
            "capacity_aware_diagnostic"
        ]
        strict_floor = summary["oracle"][profile][
            "certified_perfect_cache_floor"
        ]
        comparisons.append(
            f"- {profile}: OUR vs DAOC `{daoc['mean_improvement_percent']:.3f}%` "
            f"({daoc['wins']}/3); vs Centralized-Greedy "
            f"`{central['mean_improvement_percent']:.3f}%` "
            f"({central['wins']}/3); P95 `{tail['mean_improvement_percent']:.3f}%` "
            f"({tail['wins']}/3); capacity-aware Oracle gap "
            f"`{capacity_oracle['our_gap_percent']:.3f}%`; certified "
            f"perfect-cache floor gap `{strict_floor['our_gap_percent']:.3f}%`."
        )
    report = f"""# Pegasus P-Scale 缓存预算敏感性

> B8 复用已锁定主实验；B5/B10 中所有学习方法均从零训练至收敛。

## 完整性

- 容量、收敛、任务 exact-once、五类各 20 场景、方法内配对：`{all(summary['integrity'].values())}`。
- 不同预算间服务器部署、用户位置和 DAG 序列配对：`{summary['integrity']['workloads_paired_across_budgets'] and summary['integrity']['infrastructure_paired_across_budgets']}`。
- Seeds: `{summary['seeds']}`。

## 结果

| 预算 | 方法 | 平均完成时间 | P95 | 命中率 | 远程加载率 | 等待时延 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 配对比较

{chr(10).join(comparisons)}

容量感知 clairvoyant Oracle 使用已知场景做放松分配，只用于诊断，不称为认证全局下界。
`perfect-cache recurrence` 才是本报告中的严格认证下界。

## 十 Seed 决策

- 完整性：`{summary['gate']['integrity']}`。
- OUR 在三个预算上均优于 DAOC：`{summary['gate']['our_beats_daoc_at_all_budgets']}`。
- OUR 在三个预算上均优于 Centralized-Greedy-DQN：`{summary['gate']['our_beats_central_at_all_budgets']}`。
- OUR 在三个预算上 P95 均优于 Centralized-Greedy-DQN：`{summary['gate']['p95_beats_central_at_all_budgets']}`。
- 建议扩展至十 seed：`{summary['ten_seed_recommended']}`。

该三 seed 结果仅用于预算稳健性与扩展决策，不作正式显著性声明。
"""
    (output_dir / "PEGASUS_PSCALE_SENSITIVITY_REPORT_ZH.md").write_text(
        report,
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect(args)
    write_json(
        args.output_dir / "pegasus_pscale_sensitivity_summary.json",
        summary,
    )
    plot_results(args.output_dir, summary)
    write_report(args.output_dir, summary)
    print(f"P-Scale sensitivity analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
