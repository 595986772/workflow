#!/usr/bin/env python3
"""Analyze Pegasus DAOC-paper, ablation, and final confirmation runs."""

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
    aggregate_run,
    paired_superiority,
)
from capacity_protocol import deterministic_capacity_assignment
from pegasus_paper_closure_protocol import (
    ABLATION_METHODS,
    CACHE_CALIBRATION_EPISODES,
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DEVELOPMENT_METHODS,
    DEVELOPMENT_SEEDS,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_METHODS,
    FINAL_SEEDS,
    P2_DEVELOPMENT_SUITE,
    P2_REFERENCE_METHODS,
    PROTOCOL_VERSION,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


DISPLAY_NAMES = {
    "guided_full": "DAOC-code",
    "daoc_paper": "DAOC-paper",
    "centralized_greedy_daoc": "Centralized-Greedy-DQN",
    "lean_our": "OUR",
    "our_dqn": "OUR-DQN",
    "our_no_telemetry": "OUR-noTelemetry",
    "our_no_coord_cache": "OUR-noCoordCache",
}
COLORS = {
    "guided_full": "#68737D",
    "daoc_paper": "#3D405B",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
    "our_dqn": "#8E6C88",
    "our_no_telemetry": "#5B8E7D",
    "our_no_coord_cache": "#C1666B",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("development", "final"),
        required=True,
    )
    parser.add_argument(
        "--p2-suite-dir",
        type=Path,
        default=P2_DEVELOPMENT_SUITE,
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) != EVALUATION_EPISODES:
        raise RuntimeError(f"Expected 100 evaluation rows in {path}")
    return rows


def run_directory(args, label, seed):
    if args.mode == "development" and label in P2_REFERENCE_METHODS:
        root = args.p2_suite_dir
    else:
        root = args.suite_dir
    return root / "runs" / label / f"seed_{seed}"


def comparable_bank(bank):
    return [
        {
            "episode": record["episode"],
            "seed": record["seed"],
            "base_fingerprint": record["base_fingerprint"],
            "workflow_family": record.get("workflow_family"),
            "user_initial_positions": record["user_initial_positions"],
            "user_graph_keys": record["user_graph_keys"],
        }
        for record in bank
    ]


def family_aggregate(rows, bank):
    family_by_episode = {
        int(record["episode"]): record.get("workflow_family")
        for record in bank
    }
    output = {}
    for family in FAMILIES:
        selected = [
            row
            for row in rows
            if family_by_episode[int(row["episode"])] == family
        ]
        if len(selected) != 20:
            raise RuntimeError(f"Expected 20 {family} scenarios")
        output[family] = aggregate_run(selected)
    return output


def oracle_summary(args, aggregates, seeds):
    oracle_dir = (
        args.p2_suite_dir / "oracle"
        if args.mode == "development"
        else args.suite_dir / "oracle"
    )
    path = oracle_dir / "oracle_floor_per_seed.csv"
    if not path.exists():
        raise RuntimeError(f"Oracle result is missing: {path}")
    capacity_aware = {}
    perfect_cache = {}
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            seed = int(row["seed"])
            capacity_aware[seed] = float(row["oracle_floor"])
            perfect_cache[seed] = float(row["perfect_cache_floor"])
    if set(capacity_aware) != set(seeds):
        raise RuntimeError("Oracle seeds do not match the analysis seeds")
    capacity_mean = float(np.mean(list(capacity_aware.values())))
    perfect_mean = float(np.mean(list(perfect_cache.values())))
    our_mean = aggregates["lean_our"]["mean_finish_time"]
    return {
        "capacity_aware_diagnostic": {
            "mean": capacity_mean,
            "our_gap_sec": our_mean - capacity_mean,
            "our_gap_percent": 100.0 * (our_mean - capacity_mean) / capacity_mean,
            "certified_global_floor": False,
        },
        "certified_perfect_cache_floor": {
            "mean": perfect_mean,
            "our_gap_sec": our_mean - perfect_mean,
            "our_gap_percent": 100.0 * (our_mean - perfect_mean) / perfect_mean,
            "certified_global_floor": True,
        },
    }


def expected_method_protocol(label, arguments):
    if label == "daoc_paper":
        return (
            arguments.get("algorithm")
            == "prev_servers_plus_service_per_serverDQN"
            and arguments.get("reward_mode") == "terminal_binary"
            and arguments.get("cache_policy")
            == "paper_popularity_cost_ema"
            and arguments.get("beta") == 0.9
            and arguments.get("beta_min") == 0.1
            and arguments.get("beta_decay") == 0.995
        )
    if label == "our_dqn":
        return arguments.get("algorithm") == "causal_telemetryDDQN"
    if label == "our_no_telemetry":
        return (
            arguments.get("algorithm") == "causal_task_serverPD3QN"
            and arguments.get("cache_server_quality") is False
        )
    if label == "our_no_coord_cache":
        return arguments.get("cache_policy") == "popularity_ema"
    if label == "lean_our":
        return (
            arguments.get("algorithm") == "causal_telemetryPD3QN"
            and arguments.get("cache_policy") == "critical_path_joint"
            and arguments.get("cache_coverage_constraint") is True
        )
    if label == "centralized_greedy_daoc":
        return arguments.get("cache_policy") == "popularity_coordinated"
    return True


def development_claim_tier(comparison):
    if (
        comparison["wins"] == 3
        and comparison["mean_improvement_percent"] >= 5.0
    ):
        return "primary"
    if comparison["passed"]:
        return "secondary"
    return "unsupported"


def collect(args):
    seeds = DEVELOPMENT_SEEDS if args.mode == "development" else FINAL_SEEDS
    methods = (
        P2_REFERENCE_METHODS + DEVELOPMENT_METHODS
        if args.mode == "development"
        else FINAL_METHODS
    )
    integrity = {
        "all_runs_complete_and_converged": True,
        "dataset_and_completion_protocol_exact": True,
        "all_real_tasks_executed_once": True,
        "scenario_banks_paired": True,
        "family_balance_exact": True,
        "capacity_assignments_exact": True,
        "training_and_evaluation_protocol_exact": True,
        "method_protocols_exact": True,
    }
    per_seed = []
    family_records = {
        label: {family: [] for family in FAMILIES}
        for label in methods
    }

    for seed in seeds:
        reference_bank = None
        expected_capacities = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        method_records = {}
        checkpoints = {}
        for label in methods:
            directory = run_directory(args, label, seed)
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            rows = evaluation_rows(directory / "episodes.csv")
            bank = read_json(directory / "evaluation_scenarios.json")
            arguments = config["arguments"]
            method_records[label] = aggregate_run(rows)
            checkpoints[label] = summary["selected_checkpoint_episode"]
            by_family = family_aggregate(rows, bank)
            for family in FAMILIES:
                family_records[label][family].append(by_family[family])

            integrity["all_runs_complete_and_converged"] &= bool(
                summary.get("status") == "complete"
                and summary.get("eligible_for_comparison")
                and summary.get("convergence", {}).get("reached")
            )
            integrity["dataset_and_completion_protocol_exact"] &= bool(
                summary.get("dag_dataset", {}).get("sha256")
                == EXPECTED_DATASET_SHA256
                and summary.get("dag_completion_protocol_version")
                == DAG_COMPLETION_PROTOCOL_VERSION
            )
            integrity["all_real_tasks_executed_once"] &= all(
                int(row["real_task_count"])
                == int(row["completed_task_count"])
                and int(row["all_tasks_executed_once"]) == 1
                for row in rows
            )
            observed_capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            integrity["capacity_assignments_exact"] &= (
                observed_capacities == expected_capacities
                and sum(observed_capacities.values()) == 8
            )
            integrity["training_and_evaluation_protocol_exact"] &= bool(
                arguments.get("num_users") == 20
                and arguments.get("num_servers") == 10
                and arguments.get("num_services") == 10
                and arguments.get("num_tasks") == 31
                and arguments.get("bandwidth") == 15000
                and arguments.get("cache_freeze_episode")
                == CACHE_CALIBRATION_EPISODES
                and arguments.get("validation_scenarios") == 50
                and arguments.get("eval_bank_scope") == "infrastructure"
                and summary.get("evaluation_state_frozen")
                and summary.get("evaluation_unique_base_scenarios") == 100
            )
            integrity["method_protocols_exact"] &= expected_method_protocol(
                label,
                arguments,
            )
            counts = Counter(
                record.get("workflow_family") for record in bank
            )
            integrity["family_balance_exact"] &= (
                counts == Counter({family: 20 for family in FAMILIES})
            )
            bank_view = comparable_bank(bank)
            if reference_bank is None:
                reference_bank = bank_view
            else:
                integrity["scenario_banks_paired"] &= (
                    bank_view == reference_bank
                )
        per_seed.append(
            {
                "seed": seed,
                "methods": method_records,
                "selected_checkpoint_episode": checkpoints,
            }
        )

    aggregates = {
        label: {
            metric: float(np.mean([
                record["methods"][label][metric]
                for record in per_seed
            ]))
            for metric in per_seed[0]["methods"][label]
        }
        for label in methods
    }
    family_aggregates = {
        label: {
            family: {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in rows[0]
            }
            for family, rows in family_values.items()
        }
        for label, family_values in family_records.items()
    }
    formal = args.mode == "final"
    references = (
        ("daoc_paper", "centralized_greedy_daoc")
        if formal
        else (
            "guided_full",
            "daoc_paper",
            "centralized_greedy_daoc",
        )
    )
    comparisons = {
        f"our_vs_{label}": paired_superiority(
            [row["methods"][label]["mean_finish_time"] for row in per_seed],
            [row["methods"]["lean_our"]["mean_finish_time"] for row in per_seed],
            formal=formal,
        )
        for label in references
    }
    p95_comparisons = {
        f"our_vs_{label}": paired_superiority(
            [
                row["methods"][label]["mean_p95_finish_time"]
                for row in per_seed
            ],
            [
                row["methods"]["lean_our"]["mean_p95_finish_time"]
                for row in per_seed
            ],
            formal=formal,
        )
        for label in references
    }
    output = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "mode": args.mode,
        "claim_scope": (
            "controlled_three_seed_mechanism_closure"
            if args.mode == "development"
            else "frozen_ten_seed_final_confirmation"
        ),
        "seeds": list(seeds),
        "methods": list(methods),
        "integrity": integrity,
        "method_aggregates": aggregates,
        "family_aggregates": family_aggregates,
        "paired_comparisons": comparisons,
        "p95_paired_comparisons": p95_comparisons,
        "per_seed": per_seed,
    }
    output["oracle"] = oracle_summary(args, aggregates, seeds)
    if args.mode == "development":
        ablations = {
            f"our_vs_{label}": paired_superiority(
                [
                    row["methods"][label]["mean_finish_time"]
                    for row in per_seed
                ],
                [
                    row["methods"]["lean_our"]["mean_finish_time"]
                    for row in per_seed
                ],
                formal=False,
            )
            for label in ABLATION_METHODS
        }
        support = {
            "pairwise_pd3qn": ablations["our_vs_our_dqn"]["passed"],
            "causal_telemetry": ablations[
                "our_vs_our_no_telemetry"
            ]["passed"],
            "coordinated_cache": ablations[
                "our_vs_our_no_coord_cache"
            ]["passed"],
        }
        claim_tiers = {
            "pairwise_pd3qn": development_claim_tier(
                ablations["our_vs_our_dqn"]
            ),
            "causal_telemetry": development_claim_tier(
                ablations["our_vs_our_no_telemetry"]
            ),
            "coordinated_cache": development_claim_tier(
                ablations["our_vs_our_no_coord_cache"]
            ),
        }
        gate = {
            "integrity": all(integrity.values()),
            "our_beats_daoc_paper": comparisons[
                "our_vs_daoc_paper"
            ]["passed"],
            "our_beats_centralized_greedy": comparisons[
                "our_vs_centralized_greedy_daoc"
            ]["passed"],
            "at_least_one_primary_algorithmic_contribution": any(
                tier == "primary" for tier in claim_tiers.values()
            ),
        }
        gate["passed"] = all(gate.values())
        output["ablation_comparisons"] = ablations
        output["module_support"] = support
        output["module_claim_tiers"] = claim_tiers
        output["gate"] = gate
    else:
        gate = {
            "integrity": all(integrity.values()),
            "our_formally_beats_daoc_paper": comparisons[
                "our_vs_daoc_paper"
            ]["passed"],
            "our_formally_beats_centralized_greedy": comparisons[
                "our_vs_centralized_greedy_daoc"
            ]["passed"],
        }
        gate["passed"] = all(gate.values())
        output["gate"] = gate
    return output


def plot_summary(output_dir, summary):
    methods = summary["methods"]
    x = np.arange(len(methods))
    means = [
        summary["method_aggregates"][label]["mean_finish_time"]
        for label in methods
    ]
    p95 = [
        summary["method_aggregates"][label]["mean_p95_finish_time"]
        for label in methods
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(x, means, color=[COLORS[label] for label in methods])
    axes[0].axhline(
        summary["oracle"]["capacity_aware_diagnostic"]["mean"],
        color="#2A9D8F",
        linestyle="--",
        label="Capacity-aware oracle (diagnostic)",
    )
    axes[0].set_ylabel("Mean DAG completion time (s)")
    axes[0].set_title("Pegasus B8 mean completion")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, p95, color=[COLORS[label] for label in methods])
    axes[1].set_ylabel("P95 DAG completion time (s)")
    axes[1].set_title("Pegasus B8 tail latency")
    for axis in axes:
        axis.set_xticks(
            x,
            [DISPLAY_NAMES[label] for label in methods],
            rotation=24,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"pegasus_paper_closure_{summary['mode']}.{suffix}",
            dpi=180,
        )
    plt.close(figure)


def render_report(output_dir, summary):
    rows = []
    for label in summary["methods"]:
        aggregate = summary["method_aggregates"][label]
        rows.append(
            f"| {DISPLAY_NAMES[label]} | "
            f"{aggregate['mean_finish_time']:.6f} | "
            f"{aggregate['mean_p95_finish_time']:.6f} | "
            f"{aggregate['mean_cache_hit_rate']:.4f} | "
            f"{aggregate['mean_remote_loading_rate']:.4f} | "
            f"{aggregate['mean_waiting_latency']:.6f} |"
        )
    comparison_lines = []
    for key, comparison in summary["paired_comparisons"].items():
        comparison_lines.append(
            f"- `{key}`: {comparison['mean_improvement_percent']:.3f}%, "
            f"{comparison['wins']}/{comparison['pairs']} seed 胜出, "
            f"95% CI [{comparison['ci95_lower_sec']:.6f}, "
            f"{comparison['ci95_upper_sec']:.6f}], "
            f"p={comparison['wilcoxon_one_sided_p']:.6g}, "
            f"pass={comparison['passed']}。"
        )
    mechanism = ""
    if summary["mode"] == "development":
        names = {
            "pairwise_pd3qn": "Pairwise PD3QN",
            "causal_telemetry": "因果遥测",
            "coordinated_cache": "协调缓存",
        }
        tier_text = {
            "primary": "保留为主要创新",
            "secondary": "仅作为次要辅助机制，不作独立主创新",
            "unsupported": "删除创新表述",
        }
        decisions = [
            f"- {names[key]}：{tier_text[tier]}。"
            for key, tier in summary["module_claim_tiers"].items()
        ]
        mechanism = "\n## 消融决策\n\n" + "\n".join(decisions) + "\n"
    report = f"""# Pegasus B8 论文闭环实验

> 阶段：`{summary['mode']}`；范围：`{summary['claim_scope']}`。

## 完整性

- Seeds：`{summary['seeds']}`。
- 数据、容量、信息、收敛、场景配对和任务完成审计：`{all(summary['integrity'].values())}`。
- 门槛结论：`{summary['gate']['passed']}`。

## 结果

| 方法 | 平均完成时间 | P95 | 缓存命中率 | 远程加载率 | 等待时延 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 配对比较

{chr(10).join(comparison_lines)}

- OUR 高于容量感知 Oracle 诊断值：`{summary['oracle']['capacity_aware_diagnostic']['our_gap_sec']:.6f} s`（`{summary['oracle']['capacity_aware_diagnostic']['our_gap_percent']:.3f}%`）。
- OUR 高于认证 perfect-cache 递推下界：`{summary['oracle']['certified_perfect_cache_floor']['our_gap_sec']:.6f} s`（`{summary['oracle']['certified_perfect_cache_floor']['our_gap_percent']:.3f}%`）。
{mechanism}
## 写作边界

- `DAOC-paper` 按原文 Eq. (14) 的请求频率×服务加载时延 EMA，并使用 0.9→0.1 引导衰减。
- `DAOC-code` 仅代表当前开源实现的纯流行度 EMA，不再单独充当论文 DAOC 的唯一实现。
- 三 seed 阶段只用于机制判断；只有冻结后的十 seed 阶段才允许正式显著性声明。
"""
    (output_dir / "PEGASUS_PAPER_CLOSURE_REPORT_ZH.md").write_text(
        report,
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect(args)
    write_json(args.output_dir / "pegasus_paper_closure_summary.json", summary)
    plot_summary(args.output_dir, summary)
    render_report(args.output_dir, summary)
    print(f"Pegasus paper-closure analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
