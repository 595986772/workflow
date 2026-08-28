#!/usr/bin/env python3
"""Analyze post-lock Pegasus-B8 heuristic and Discrete SAC baselines."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_a0_coordination import aggregate_run, paired_superiority
from capacity_protocol import deterministic_capacity_assignment
from pegasus_baseline_extension_protocol import (
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    EVALUATION_EPISODES,
    FINAL_SEEDS,
    HEURISTIC_METHODS,
    P3_FINAL_DIR,
    PROTOCOL_VERSION,
    REFERENCE_METHODS,
    SAC_CONFIG,
    SAC_METHOD,
    validate_protocol,
)


DISPLAY_NAMES = {
    "daoc_paper": "DAOC-paper",
    "centralized_greedy_daoc": "Centralized-Greedy-DQN",
    "lean_our": "OUR",
    "coord_cache_random": "Random",
    "coord_cache_nearest": "Nearest",
    "coord_cache_nearest_service": "Nearest-with-Service",
    "coord_cache_discrete_sac": "CoordCache-DiscreteSAC",
}
COLORS = {
    "daoc_paper": "#59636E",
    "centralized_greedy_daoc": "#D18B35",
    "lean_our": "#247BA0",
    "coord_cache_random": "#A5A5A5",
    "coord_cache_nearest": "#7A9E9F",
    "coord_cache_nearest_service": "#70A288",
    "coord_cache_discrete_sac": "#9B5DE5",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-dir", type=Path, required=True)
    parser.add_argument("--sac-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=P3_FINAL_DIR,
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_dir(args, label, seed):
    if label in REFERENCE_METHODS:
        root = args.reference_dir
    elif label in HEURISTIC_METHODS:
        root = args.heuristic_dir
    else:
        root = args.sac_dir
    return root / "runs" / label / f"seed_{seed}"


def evaluation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) != EVALUATION_EPISODES:
        raise RuntimeError(
            f"Expected {EVALUATION_EPISODES} evaluation rows: {path}"
        )
    return rows


def comparable_bank(bank):
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


def expected_method_config(label, arguments):
    if label in HEURISTIC_METHODS:
        algorithms = {
            "coord_cache_random": "random",
            "coord_cache_nearest": "nearest_server",
            "coord_cache_nearest_service": "nearest_with_service",
        }
        return (
            arguments["algorithm"] == algorithms[label]
            and arguments["cache_policy"] == "critical_path_joint"
            and arguments["cache_coverage_constraint"] is True
            and arguments["train_episodes"] == 5000
            and arguments["checkpoint_every"] == 0
        )
    if label == SAC_METHOD:
        return (
            arguments["algorithm"] == SAC_CONFIG["algorithm"]
            and arguments["reward_mode"] == SAC_CONFIG["reward_mode"]
            and arguments["cache_policy"] == SAC_CONFIG["cache_policy"]
            and arguments["cache_coverage_constraint"] is True
            and math.isclose(
                arguments["entropy_coefficient"],
                SAC_CONFIG["initial_alpha"],
            )
            and math.isclose(
                arguments["sac_target_entropy_ratio"],
                SAC_CONFIG["target_entropy_ratio"],
            )
            and math.isclose(
                arguments["sac_target_tau"],
                SAC_CONFIG["target_tau"],
            )
        )
    return True


def checkpoint_alpha(directory):
    checkpoint = torch.load(
        directory / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    weights = checkpoint["frozen_state"]["weights"]
    values = []
    for server_weights in weights.values():
        log_alpha = server_weights.get("log_alpha")
        if log_alpha is not None:
            values.append(float(torch.exp(log_alpha).item()))
    return float(np.mean(values)) if values else None


def collect(args):
    protocol = validate_protocol()
    methods = REFERENCE_METHODS + HEURISTIC_METHODS + (SAC_METHOD,)
    integrity = {
        "all_runs_complete": True,
        "all_learning_runs_converged": True,
        "scenario_banks_paired": True,
        "capacity_assignments_exact": True,
        "method_protocols_exact": True,
        "causal_information_protocol_exact": True,
    }
    per_seed = []
    values = {label: [] for label in methods}
    p95_values = {label: [] for label in methods}
    alpha_values = []

    for seed in FINAL_SEEDS:
        reference_bank = None
        expected_capacities = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        seed_record = {"seed": seed, "methods": {}}
        for label in methods:
            directory = run_dir(args, label, seed)
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            rows = evaluation_rows(directory / "episodes.csv")
            bank = comparable_bank(
                read_json(directory / "evaluation_scenarios.json")
            )
            arguments = config["arguments"]
            capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            if summary.get("status") != "complete":
                integrity["all_runs_complete"] = False
            if (
                label not in HEURISTIC_METHODS
                and not summary.get("eligible_for_comparison", False)
            ):
                integrity["all_learning_runs_converged"] = False
            if capacities != expected_capacities:
                integrity["capacity_assignments_exact"] = False
            if not expected_method_config(label, arguments):
                integrity["method_protocols_exact"] = False
            if (
                summary.get("information_protocol_version")
                != "causal_cache_v1"
            ):
                integrity["causal_information_protocol_exact"] = False
            if reference_bank is None:
                reference_bank = bank
            elif bank != reference_bank:
                integrity["scenario_banks_paired"] = False

            aggregate = aggregate_run(rows)
            values[label].append(aggregate["mean_finish_time"])
            p95_values[label].append(
                aggregate["mean_p95_finish_time"]
            )
            seed_record["methods"][label] = aggregate
            if label == SAC_METHOD:
                alpha_values.append(checkpoint_alpha(directory))
        per_seed.append(seed_record)

    aggregates = {}
    for label in methods:
        method_records = [
            record["methods"][label]
            for record in per_seed
        ]
        aggregates[label] = {
            key: float(np.mean([record[key] for record in method_records]))
            for key in method_records[0]
        }

    our = values["lean_our"]
    comparisons = {
        f"our_vs_{label}": paired_superiority(
            values[label],
            our,
            formal=True,
        )
        for label in HEURISTIC_METHODS + (SAC_METHOD,)
    }
    p95_comparisons = {
        f"our_vs_{label}": paired_superiority(
            p95_values[label],
            p95_values["lean_our"],
            formal=True,
        )
        for label in HEURISTIC_METHODS + (SAC_METHOD,)
    }
    sac_comparisons = {
        f"sac_vs_{label}": paired_superiority(
            values[label],
            values[SAC_METHOD],
            formal=True,
        )
        for label in ("daoc_paper", "centralized_greedy_daoc")
    }
    gate = {
        "integrity": all(integrity.values()),
        "our_beats_all_new_baselines": all(
            comparison["passed"]
            for comparison in comparisons.values()
        ),
        "sac_is_stronger_than_daoc": sac_comparisons[
            "sac_vs_daoc_paper"
        ]["passed"],
    }
    gate["passed"] = (
        gate["integrity"]
        and gate["our_beats_all_new_baselines"]
    )
    return {
        "protocol": protocol,
        "integrity": integrity,
        "aggregates": aggregates,
        "per_seed": per_seed,
        "comparisons": comparisons,
        "p95_comparisons": p95_comparisons,
        "sac_comparisons": sac_comparisons,
        "sac_temperature": {
            "per_seed": alpha_values,
            "mean": float(np.mean(alpha_values)),
        },
        "gate": gate,
    }


def plot_results(result, output_dir):
    methods = list(result["aggregates"])
    means = [
        result["aggregates"][label]["mean_finish_time"]
        for label in methods
    ]
    seed_values = {
        label: np.asarray(
            [
                record["methods"][label]["mean_finish_time"]
                for record in result["per_seed"]
            ],
            dtype=float,
        )
        for label in methods
    }
    errors = [
        1.96 * seed_values[label].std(ddof=1) / math.sqrt(len(FINAL_SEEDS))
        for label in methods
    ]
    figure, axis = plt.subplots(figsize=(11, 5.8))
    positions = np.arange(len(methods))
    axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=4,
        color=[COLORS[label] for label in methods],
        edgecolor="white",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [DISPLAY_NAMES[label] for label in methods],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.set_title("Pegasus-B8 Post-lock Baseline Comparison", loc="left")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_comparison.png", dpi=220)
    figure.savefig(output_dir / "baseline_comparison.pdf")
    plt.close(figure)


def write_report(result, output_dir):
    lines = [
        "# Pegasus-B8论文级Baseline补充报告",
        "",
        f"- 协议：`{PROTOCOL_VERSION}`。",
        "- 范围：P3算法锁定后的强基线扩展，不称为新的独立holdout。",
        f"- 完整性：`{result['gate']['integrity']}`。",
        "",
        "## 主结果",
        "",
        "| 方法 | Mean | P95 | 命中率 | 远程加载率 | 等待时延 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, record in result["aggregates"].items():
        lines.append(
            f"| {DISPLAY_NAMES[label]} "
            f"| {record['mean_finish_time']:.6f} "
            f"| {record['mean_p95_finish_time']:.6f} "
            f"| {record['mean_cache_hit_rate']:.4f} "
            f"| {record['mean_remote_loading_rate']:.4f} "
            f"| {record['mean_waiting_latency']:.6f} |"
        )
    lines.extend(["", "## OUR与新增基线的配对统计", ""])
    for key, comparison in result["comparisons"].items():
        label = key.removeprefix("our_vs_")
        lines.append(
            f"- OUR vs {DISPLAY_NAMES[label]}：改善 "
            f"`{comparison['mean_improvement_percent']:.3f}%`，"
            f"胜出 `{comparison['wins']}/10`，95% CI "
            f"`[{comparison['ci95_lower_sec']:.6f}, "
            f"{comparison['ci95_upper_sec']:.6f}] s`，"
            f"p=`{comparison['wilcoxon_one_sided_p']:.9f}`，"
            f"pass=`{comparison['passed']}`。"
        )
    lines.extend(
        [
            "",
            "## Discrete SAC诊断",
            "",
            f"- 最终自动温度均值：`{result['sac_temperature']['mean']:.6f}`。",
            f"- 强于DAOC：`{result['gate']['sac_is_stronger_than_daoc']}`。",
            f"- OUR强于全部新增基线：`{result['gate']['our_beats_all_new_baselines']}`。",
            "",
            "三个启发式没有训练神经网络，但均进行了5000轮因果协调缓存校准；"
            "Discrete SAC与OUR使用相同状态、奖励、缓存和离散动作空间。",
        ]
    )
    (output_dir / "BASELINE_EXTENSION_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = collect(args)
    write_json(args.output_dir / "baseline_extension_summary.json", result)
    plot_results(result, args.output_dir)
    write_report(result, args.output_dir)
    print(args.output_dir / "BASELINE_EXTENSION_REPORT_ZH.md")


if __name__ == "__main__":
    main()
