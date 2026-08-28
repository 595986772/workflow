#!/usr/bin/env python3
"""Analyze the three-seed Pegasus external-workflow validation for h8v1."""

import argparse
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
    oracle_per_seed,
    paired_superiority,
)
from capacity_protocol import deterministic_capacity_assignment


SEEDS = (31, 32, 33)
METHODS = ("guided_full", "centralized_greedy_daoc", "lean_our")
CAPACITY_MULTISET = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
CAPACITY_NAMESPACE = "h8v1_pegasus_cross_topology_v1"
COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
    "oracle": "#2A9D8F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_dir(suite_dir, label, seed):
    return suite_dir / "runs" / label / f"seed_{seed}"


def workload_view(bank):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def collect(args):
    per_seed = []
    integrity = {
        "dataset_hash_exact": True,
        "all_methods_converged": True,
        "scenario_banks_paired": True,
        "capacity_assignments_exact": True,
        "total_budget_eight": True,
        "networks_frozen_in_evaluation": True,
        "generalization_seed_partition": True,
        "h8v1_modules_exact": True,
        "baseline_did_not_receive_coverage_constraint": True,
    }
    for seed in SEEDS:
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
            methods[label] = aggregate_run(rows)
            checkpoints[label] = summary["selected_checkpoint_episode"]
            integrity["dataset_hash_exact"] &= (
                summary.get("dag_dataset", {}).get("sha256")
                == args.dataset_sha256
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
            integrity["total_budget_eight"] &= sum(capacities.values()) == 8
            integrity["networks_frozen_in_evaluation"] &= bool(
                summary.get("evaluation_state_frozen")
                and summary.get("evaluation_unique_scenarios") == 100
            )
            integrity["generalization_seed_partition"] &= (
                summary.get("revision", {}).get("seed_partition")
                == "generalization"
            )
            arguments = config.get("arguments", {})
            if arguments.get("bandwidth") != 15000:
                raise RuntimeError(f"Wrong bandwidth in {directory}")
            if label == "lean_our":
                modules = set(summary.get("method_modules", {}).get("active", []))
                expected = {
                    "pairwise_dueling_double_dqn",
                    "causal_history_telemetry",
                    "causal_dependency_aware_joint_cache",
                    "scarcity_aware_service_coverage_constraint",
                }
                integrity["h8v1_modules_exact"] &= expected.issubset(modules)
                integrity["h8v1_modules_exact"] &= (
                    arguments.get("cache_coverage_constraint") is True
                )
            else:
                integrity["baseline_did_not_receive_coverage_constraint"] &= not bool(
                    arguments.get("cache_coverage_constraint", False)
                )
            banks.append(
                workload_view(
                    read_json(directory / "evaluation_scenarios.json")
                )
            )
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
            row["methods"]["centralized_greedy_daoc"]["mean_p95_finish_time"]
            for row in per_seed
        ],
        [row["methods"]["lean_our"]["mean_p95_finish_time"] for row in per_seed],
        formal=False,
    )
    oracle = oracle_per_seed(args.suite_dir / "oracle")
    if set(oracle) != set(SEEDS):
        raise RuntimeError("Oracle seed mismatch")
    oracle_mean = float(np.mean(list(oracle.values())))
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
        "claim_scope": (
            "posthoc_external_scientific_workflow_three_seed_diagnostic"
        ),
        "dataset": {
            "path": str(args.dataset_path.resolve()),
            "sha256": args.dataset_sha256,
            "unbiased_mec_holdout": False,
            "topology_only_intervention": False,
            "service_mapping": "stable_hash_of_pegasus_job_type",
        },
        "seeds": list(SEEDS),
        "capacity_multiset": CAPACITY_MULTISET,
        "integrity": integrity,
        "method_aggregates": aggregates,
        "paired_comparisons": comparisons,
        "p95_our_vs_central": p95,
        "oracle": {
            "mean": oracle_mean,
            "our_gap_sec": our_mean - oracle_mean,
            "our_gap_percent": 100 * (our_mean - oracle_mean) / oracle_mean,
            "diagnostic_only": True,
        },
        "gate": gate,
        "per_seed": per_seed,
    }


def plot_results(path, summary):
    methods = list(METHODS) + ["oracle"]
    means = [
        summary["method_aggregates"][label]["mean_finish_time"]
        if label != "oracle"
        else summary["oracle"]["mean"]
        for label in methods
    ]
    display = [DISPLAY_NAMES.get(label, "Oracle") for label in methods]
    seeds = np.asarray(SEEDS)
    daoc_gain = np.asarray(
        [
            row["methods"]["guided_full"]["mean_finish_time"]
            - row["methods"]["lean_our"]["mean_finish_time"]
            for row in summary["per_seed"]
        ]
    )
    central_gain = np.asarray(
        [
            row["methods"]["centralized_greedy_daoc"]["mean_finish_time"]
            - row["methods"]["lean_our"]["mean_finish_time"]
            for row in summary["per_seed"]
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(
        display,
        means,
        color=[COLORS[label] for label in methods],
    )
    axes[0].set_ylabel("Mean DAG completion time (s)")
    axes[0].set_title("Pegasus external-workflow benchmark")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.35
    x = np.arange(len(seeds))
    axes[1].bar(x - width / 2, daoc_gain, width, label="OUR vs DAOC", color="#59636E")
    axes[1].bar(x + width / 2, central_gain, width, label="OUR vs Central", color="#E09F3E")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(x, seeds)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Reference - OUR (s)")
    axes[1].set_title("Paired seed improvements")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(path / f"cross_topology_results.{suffix}", dpi=180)
    plt.close(figure)


def render_report(path, summary):
    aggregates = summary["method_aggregates"]
    daoc = summary["paired_comparisons"]["our_vs_guided_full"]
    central = summary["paired_comparisons"][
        "our_vs_centralized_greedy_daoc"
    ]
    p95 = summary["p95_our_vs_central"]
    report = f"""# h8v1 Pegasus 外部工作流三 seed 验证

> 这是查看 A0 最终结果后新增的事后外部工作流诊断，不是预注册的无偏 MEC holdout，三 seed 也不用于正式显著性宣称。

Pegasus 作业类型被稳定映射为 10 种服务，因此本实验同时改变了 DAG 拓扑和任务类型--服务请求相关性，不是隔离“拓扑”单一因素的因果实验。

## 完整性

- 数据集、场景配对、容量、收敛、冻结评估与 h8v1 模块审计：`{all(summary['integrity'].values())}`。
- Seeds：`{summary['seeds']}`，每 seed 100 个冻结配对场景。
- 容量：`{summary['capacity_multiset']}`，总预算 8。

## 平均结果

| 方法 | 平均 DAG 完成时间 |
|---|---:|
| DAOC | {aggregates['guided_full']['mean_finish_time']:.6f} s |
| Centralized-Greedy | {aggregates['centralized_greedy_daoc']['mean_finish_time']:.6f} s |
| OUR | {aggregates['lean_our']['mean_finish_time']:.6f} s |
| Oracle | {summary['oracle']['mean']:.6f} s |

- OUR vs DAOC：`{daoc['mean_improvement_percent']:.3f}%`，胜出 `{daoc['wins']}/3` seed。
- OUR vs Centralized-Greedy：`{central['mean_improvement_percent']:.3f}%`，胜出 `{central['wins']}/3` seed。
- P95 vs Centralized-Greedy：`{p95['mean_improvement_percent']:.3f}%`，胜出 `{p95['wins']}/3` seed。
- OUR--Oracle gap：`{summary['oracle']['our_gap_percent']:.3f}%`。
- OUR vs Centralized-Greedy 的 95% 配对 CI：`[{central['ci95_lower_sec']:.6f}, {central['ci95_upper_sec']:.6f}] s`；单侧 Wilcoxon `p={central['wilcoxon_one_sided_p']:.6f}`。

## 三 seed 筛选门槛

- OUR 优于 DAOC：`{summary['gate']['our_beats_daoc_mean_and_two_of_three']}`。
- OUR 优于 Centralized-Greedy：`{summary['gate']['our_beats_central_mean_and_two_of_three']}`。
- P95 优于 Centralized-Greedy：`{summary['gate']['p95_beats_central_mean_and_two_of_three']}`。
- 总门槛：`{summary['gate']['passed']}`。

若总门槛通过，只能说 h8v1 在新科学工作流设置上保持了方向一致性；不能宣称结果由拓扑单独导致，也不能用三 seed 做正式显著性结论。
"""
    Path(path).write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect(args)
    write_json(args.output_dir / "cross_topology_summary.json", summary)
    plot_results(args.output_dir, summary)
    render_report(args.output_dir / "CROSS_TOPOLOGY_REPORT_ZH.md", summary)
    print(f"Cross-topology analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
