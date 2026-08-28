#!/usr/bin/env python3
"""Analyze Pegasus-B8 baselines, the 2x2 design and mechanism ablations."""

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

from analyze_a0_coordination import aggregate_run, paired_superiority
from capacity_protocol import deterministic_capacity_assignment
from pegasus_p6_protocol import (
    BASE_DDQN_STD_LABEL,
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FACTORIAL_METHODS,
    FINAL_SEEDS,
    HEURISTIC_METHODS,
    MAIN_COMPARISON_METHODS,
    MECHANISM_ABLATIONS,
    OUR_FLAT_DDQN_LABEL,
    OUR_NO_DEPENDENCY_CACHE_LABEL,
    OUR_NO_TASK_DEPENDENCY_LABEL,
    OUR_TERMINAL_REWARD_LABEL,
    P3_FINAL_DIR,
    P4_ABLATION_DIR,
    P5_SAC_DIR,
    PROTOCOL_VERSION,
    REFERENCE_METHODS,
    validate_protocol,
)


DISPLAY = {
    "random": "Random",
    "nearest": "Nearest",
    "greedy": "Nearest-with-Service",
    "dqn_wdsa_std_cache": "DQN-WDSA",
    "daoc_paper": "DAOC-paper",
    "centralized_greedy_daoc": "Centralized-Greedy-DQN",
    "coord_cache_discrete_sac": "CoordCache-DiscreteSAC",
    "lean_our": "OUR",
    "base_ddqn_std_cache": "Flat DDQN + StdCache",
    "our_flat_ddqn": "Flat DDQN + CoordCache",
    "our_no_coord_cache": "Pairwise PD3QN + StdCache",
    "our_no_task_dependency": "OUR-noTaskDependency",
    "our_no_dependency_cache": "OUR-noDependencyCache",
    "our_terminal_reward": "OUR-TerminalReward",
}
COLORS = {
    "random": "#9EA3A8",
    "nearest": "#6C8EAD",
    "greedy": "#63A375",
    "dqn_wdsa_std_cache": "#A47C48",
    "daoc_paper": "#4D5661",
    "centralized_greedy_daoc": "#D38B35",
    "coord_cache_discrete_sac": "#8B6BB1",
    "lean_our": "#147D92",
    "base_ddqn_std_cache": "#7B8794",
    "our_flat_ddqn": "#D38B35",
    "our_no_coord_cache": "#5D8AA8",
    "our_no_task_dependency": "#B56B45",
    "our_no_dependency_cache": "#7A8F62",
    "our_terminal_reward": "#9B6A8F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-dir", type=Path, required=True)
    parser.add_argument("--learning-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_directory(args, label, seed):
    if label in HEURISTIC_METHODS:
        suite = args.heuristic_dir
    elif label in REFERENCE_METHODS:
        if label in {"daoc_paper", "centralized_greedy_daoc", "lean_our"}:
            suite = P3_FINAL_DIR
        elif label == "our_no_coord_cache":
            suite = P4_ABLATION_DIR
        else:
            suite = P5_SAC_DIR
    else:
        suite = args.learning_dir
    return suite / "runs" / label / f"seed_{seed}"


def evaluation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) != EVALUATION_EPISODES:
        raise RuntimeError(f"Wrong evaluation-row count: {path}")
    return rows


def comparable_bank(path):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in read_json(path)
    ]


def method_protocol_exact(label, arguments):
    common_cp = (
        arguments["reward_mode"]
        in {"causal_makespan_increment", "terminal_binary"}
        and arguments["gamma"] == 1.0
        and arguments["n_step"] == 3
    )
    if label == "random":
        return arguments["algorithm"] == "random" and (
            arguments["cache_policy"] == "popularity_ema"
        )
    if label == "nearest":
        return arguments["algorithm"] == "nearest_server" and (
            arguments["cache_policy"] == "popularity_ema"
        )
    if label == "greedy":
        return arguments["algorithm"] == "nearest_with_service" and (
            arguments["cache_policy"] == "popularity_ema"
        )
    if label == "dqn_wdsa_std_cache":
        return (
            arguments["algorithm"] == "simpleDQN"
            and arguments["reward_mode"] == "terminal_binary"
            and arguments["cache_policy"] == "popularity_ema"
        )
    if label == BASE_DDQN_STD_LABEL:
        return (
            common_cp
            and arguments["algorithm"] == "causal_telemetryDDQN"
            and arguments["reward_mode"] == "causal_makespan_increment"
            and arguments["cache_policy"] == "popularity_ema"
        )
    if label == OUR_FLAT_DDQN_LABEL:
        return (
            common_cp
            and arguments["algorithm"] == "causal_telemetryDDQN"
            and arguments["reward_mode"] == "causal_makespan_increment"
            and arguments["cache_policy"] == "critical_path_joint"
            and arguments["cache_coverage_constraint"] is True
        )
    if label == OUR_NO_TASK_DEPENDENCY_LABEL:
        return (
            common_cp
            and arguments["algorithm"] == "causal_telemetryPD3QN"
            and arguments["cache_policy"] == "critical_path_joint"
            and arguments["cache_coverage_constraint"] is True
            and arguments["task_dependency_features"] is False
            and arguments["cache_dependency_awareness"] is True
        )
    if label == OUR_NO_DEPENDENCY_CACHE_LABEL:
        return (
            common_cp
            and arguments["algorithm"] == "causal_telemetryPD3QN"
            and arguments["cache_policy"] == "critical_path_joint"
            and arguments["cache_coverage_constraint"] is True
            and arguments["task_dependency_features"] is True
            and arguments["cache_dependency_awareness"] is False
        )
    if label == OUR_TERMINAL_REWARD_LABEL:
        return (
            common_cp
            and arguments["algorithm"] == "causal_telemetryPD3QN"
            and arguments["reward_mode"] == "terminal_binary"
            and arguments["cache_policy"] == "critical_path_joint"
            and arguments["cache_coverage_constraint"] is True
            and arguments["task_dependency_features"] is True
            and arguments["cache_dependency_awareness"] is True
        )
    return True


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, half


def collect(args):
    protocol = validate_protocol()
    methods = tuple(
        dict.fromkeys(
            MAIN_COMPARISON_METHODS
            + FACTORIAL_METHODS
            + MECHANISM_ABLATIONS
        )
    )
    integrity = {
        "all_runs_complete": True,
        "all_learning_runs_converged": True,
        "scenario_banks_paired": True,
        "capacity_assignments_exact": True,
        "dataset_and_environment_exact": True,
        "method_protocols_exact": True,
        "evaluation_state_frozen": True,
        "all_tasks_executed_once": True,
    }
    per_seed = []
    for seed in FINAL_SEEDS:
        expected_capacities = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        reference_bank = None
        seed_record = {"seed": seed, "methods": {}}
        for label in methods:
            directory = run_directory(args, label, seed)
            summary = read_json(directory / "summary.json")
            arguments = read_json(directory / "config.json")["arguments"]
            rows = evaluation_rows(directory / "episodes.csv")
            bank = comparable_bank(directory / "evaluation_scenarios.json")
            capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            integrity["all_runs_complete"] &= summary.get("status") == "complete"
            if label not in HEURISTIC_METHODS:
                integrity["all_learning_runs_converged"] &= bool(
                    summary.get("eligible_for_comparison", False)
                    and summary.get("convergence", {}).get("reached", False)
                )
            integrity["capacity_assignments_exact"] &= (
                capacities == expected_capacities
            )
            integrity["dataset_and_environment_exact"] &= (
                summary.get("dag_dataset", {}).get("sha256")
                == EXPECTED_DATASET_SHA256
                and arguments.get("bandwidth") == 15000
                and arguments.get("num_users") == 20
                and arguments.get("num_servers") == 10
                and arguments.get("num_services") == 10
                and arguments.get("num_tasks") == 31
                and arguments.get("eval_bank_scope") == "infrastructure"
            )
            integrity["method_protocols_exact"] &= method_protocol_exact(
                label, arguments
            )
            integrity["evaluation_state_frozen"] &= bool(
                summary.get("evaluation_state_frozen", False)
            )
            integrity["all_tasks_executed_once"] &= all(
                row["all_tasks_executed_once"] in {"1", "True", "true"}
                for row in rows
            )
            if reference_bank is None:
                reference_bank = bank
            elif bank != reference_bank:
                integrity["scenario_banks_paired"] = False
            seed_record["methods"][label] = aggregate_run(rows)
        per_seed.append(seed_record)

    aggregates = {
        label: {
            metric: float(
                np.mean(
                    [row["methods"][label][metric] for row in per_seed]
                )
            )
            for metric in per_seed[0]["methods"][label]
        }
        for label in methods
    }

    def values(label, metric="mean_finish_time"):
        return [row["methods"][label][metric] for row in per_seed]

    main_comparisons = {
        f"our_vs_{label}": paired_superiority(
            values(label), values("lean_our"), formal=True
        )
        for label in MAIN_COMPARISON_METHODS
        if label != "lean_our"
    }
    main_p95_comparisons = {
        f"our_vs_{label}": paired_superiority(
            values(label, "mean_p95_finish_time"),
            values("lean_our", "mean_p95_finish_time"),
            formal=True,
        )
        for label in MAIN_COMPARISON_METHODS
        if label != "lean_our"
    }
    factorial = {
        "coord_cache_effect_flat_ddqn": paired_superiority(
            values(BASE_DDQN_STD_LABEL),
            values(OUR_FLAT_DDQN_LABEL),
            formal=True,
        ),
        "coord_cache_effect_pairwise_pd3qn": paired_superiority(
            values("our_no_coord_cache"),
            values("lean_our"),
            formal=True,
        ),
        "pairwise_effect_standard_cache": paired_superiority(
            values(BASE_DDQN_STD_LABEL),
            values("our_no_coord_cache"),
            formal=True,
        ),
        "pairwise_effect_coordinated_cache": paired_superiority(
            values(OUR_FLAT_DDQN_LABEL),
            values("lean_our"),
            formal=True,
        ),
    }
    mechanism = {
        f"our_vs_{label}": paired_superiority(
            values(label), values("lean_our"), formal=True
        )
        for label in MECHANISM_ABLATIONS
    }
    mechanism_p95 = {
        f"our_vs_{label}": paired_superiority(
            values(label, "mean_p95_finish_time"),
            values("lean_our", "mean_p95_finish_time"),
            formal=True,
        )
        for label in MECHANISM_ABLATIONS
    }

    oracle_path = P3_FINAL_DIR / "oracle/oracle_floor_per_seed.csv"
    with oracle_path.open(newline="", encoding="utf-8") as input_file:
        oracle_by_seed = {
            int(row["seed"]): float(row["oracle_floor"])
            for row in csv.DictReader(input_file)
        }
    oracle_gaps = {}
    for label in methods:
        gaps = [
            value - oracle_by_seed[seed]
            for seed, value in zip(FINAL_SEEDS, values(label))
        ]
        ratios = [
            value / oracle_by_seed[seed]
            for seed, value in zip(FINAL_SEEDS, values(label))
        ]
        oracle_gaps[label] = {
            "mean_absolute_gap_sec": float(np.mean(gaps)),
            "mean_ratio_to_capacity_oracle": float(np.mean(ratios)),
        }

    evidence = {
        "integrity_passed": bool(all(integrity.values())),
        "our_beats_daoc": main_comparisons["our_vs_daoc_paper"]["passed"],
        "our_beats_centralized_greedy": main_comparisons[
            "our_vs_centralized_greedy_daoc"
        ]["passed"],
        "our_beats_coord_cache_discrete_sac": main_comparisons[
            "our_vs_coord_cache_discrete_sac"
        ]["passed"],
        "coordinated_cache_supported_in_both_learners": bool(
            factorial["coord_cache_effect_flat_ddqn"]["passed"]
            and factorial["coord_cache_effect_pairwise_pd3qn"]["passed"]
        ),
        "pairwise_pd3qn_supported_in_both_cache_regimes": bool(
            factorial["pairwise_effect_standard_cache"]["passed"]
            and factorial["pairwise_effect_coordinated_cache"]["passed"]
        ),
        "task_dependency_state_supported": mechanism[
            f"our_vs_{OUR_NO_TASK_DEPENDENCY_LABEL}"
        ]["passed"],
        "cache_dependency_weighting_supported": mechanism[
            f"our_vs_{OUR_NO_DEPENDENCY_CACHE_LABEL}"
        ]["passed"],
        "causal_makespan_reward_supported": mechanism[
            f"our_vs_{OUR_TERMINAL_REWARD_LABEL}"
        ]["passed"],
    }
    return {
        "protocol": protocol,
        "integrity": integrity,
        "aggregates": aggregates,
        "per_seed": per_seed,
        "main_comparisons": main_comparisons,
        "main_p95_comparisons": main_p95_comparisons,
        "factorial_comparisons": factorial,
        "mechanism_comparisons": mechanism,
        "mechanism_p95_comparisons": mechanism_p95,
        "oracle_gaps": oracle_gaps,
        "evidence": evidence,
    }


def save_aggregate_csv(result, output_dir):
    path = output_dir / "method_aggregates.csv"
    metrics = tuple(next(iter(result["aggregates"].values())).keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("method",) + metrics)
        writer.writeheader()
        for label, values in result["aggregates"].items():
            writer.writerow({"method": label, **values})


def bar_panel(axis, result, methods, metric, ylabel):
    positions = np.arange(len(methods))
    values = {
        label: [
            row["methods"][label][metric]
            for row in result["per_seed"]
        ]
        for label in methods
    }
    means = []
    errors = []
    for label in methods:
        mean, half = mean_ci95(values[label])
        means.append(mean)
        errors.append(half)
    axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=3,
        color=[COLORS[label] for label in methods],
        edgecolor="white",
        linewidth=0.8,
    )
    rng = np.random.default_rng(20260809)
    for position, label in zip(positions, methods):
        jitter = rng.uniform(-0.07, 0.07, len(values[label]))
        axis.scatter(
            position + jitter,
            values[label],
            s=14,
            facecolor="white",
            edgecolor=COLORS[label],
            linewidth=0.8,
            zorder=3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [DISPLAY[label] for label in methods],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22)


def plot_main(result, output_dir):
    methods = MAIN_COMPARISON_METHODS
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 4.9), constrained_layout=True
    )
    bar_panel(
        axes[0], result, methods, "mean_finish_time", "Mean completion time (s)"
    )
    bar_panel(
        axes[1],
        result,
        methods,
        "mean_p95_finish_time",
        "P95 completion time (s)",
    )
    axes[0].text(
        -0.08,
        1.03,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
    )
    axes[1].text(
        -0.08,
        1.03,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"p6_main_baselines.{suffix}", dpi=300)
    plt.close(figure)


def plot_factorial(result, output_dir):
    x = np.arange(2)
    flat = (
        result["aggregates"][BASE_DDQN_STD_LABEL]["mean_finish_time"],
        result["aggregates"][OUR_FLAT_DDQN_LABEL]["mean_finish_time"],
    )
    pairwise = (
        result["aggregates"]["our_no_coord_cache"]["mean_finish_time"],
        result["aggregates"]["lean_our"]["mean_finish_time"],
    )
    figure, axis = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
    axis.plot(x, flat, marker="o", linewidth=2, color="#7B8794", label="Flat DDQN")
    axis.plot(
        x,
        pairwise,
        marker="s",
        linewidth=2,
        color="#147D92",
        label="Pairwise PD3QN",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(("Independent StdCache", "Coordinated Cache"))
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"p6_factorial_2x2.{suffix}", dpi=300)
    plt.close(figure)


def plot_mechanisms(result, output_dir):
    methods = MECHANISM_ABLATIONS + ("lean_our",)
    figure, axes = plt.subplots(
        1, 2, figsize=(10.7, 4.6), constrained_layout=True
    )
    bar_panel(
        axes[0], result, methods, "mean_finish_time", "Mean completion time (s)"
    )
    hit = [result["aggregates"][label]["mean_cache_hit_rate"] for label in methods]
    remote = [
        result["aggregates"][label]["mean_remote_loading_rate"]
        for label in methods
    ]
    x = np.arange(len(methods))
    width = 0.36
    axes[1].bar(x - width / 2, hit, width, label="Cache hit", color="#4C956C")
    axes[1].bar(
        x + width / 2,
        remote,
        width,
        label="Remote loading",
        color="#D17A4A",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [DISPLAY[label] for label in methods], rotation=22, ha="right"
    )
    axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"p6_mechanism_ablation.{suffix}", dpi=300)
    plt.close(figure)


def comparison_line(label, comparison):
    return (
        f"- {label}: improvement `{comparison['mean_improvement_percent']:.3f}%`, "
        f"wins `{comparison['wins']}/10`, 95% CI "
        f"`[{comparison['ci95_lower_sec']:.6f}, "
        f"{comparison['ci95_upper_sec']:.6f}] s`, "
        f"p=`{comparison['wilcoxon_one_sided_p']:.9f}`, "
        f"pass=`{comparison['passed']}`."
    )


def write_report(result, output_dir):
    lines = [
        "# Pegasus-B8 Baseline and Mechanism Ablation Report",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`.",
        "- Seeds 51-60 are paired final confirmation seeds, not a new holdout.",
        f"- Integrity passed: `{result['evidence']['integrity_passed']}`.",
        "",
        "## Main Baselines",
        "",
        "| Method | Mean (s) | P95 (s) | Cache hit | Coverage | Remote load | Waiting (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in MAIN_COMPARISON_METHODS:
        row = result["aggregates"][label]
        lines.append(
            f"| {DISPLAY[label]} | {row['mean_finish_time']:.6f} "
            f"| {row['mean_p95_finish_time']:.6f} "
            f"| {row['mean_cache_hit_rate']:.4f} "
            f"| {row['mean_cache_service_coverage']:.4f} "
            f"| {row['mean_remote_loading_rate']:.4f} "
            f"| {row['mean_waiting_latency']:.6f} |"
        )
    lines.extend(["", "## OUR Paired Comparisons", ""])
    for key, comparison in result["main_comparisons"].items():
        label = key.removeprefix("our_vs_")
        lines.append(comparison_line(f"OUR vs {DISPLAY[label]}", comparison))
    lines.extend(["", "## Core 2x2", ""])
    for label, comparison in result["factorial_comparisons"].items():
        lines.append(comparison_line(label, comparison))
    lines.extend(["", "## Mechanism Ablations", ""])
    for key, comparison in result["mechanism_comparisons"].items():
        label = key.removeprefix("our_vs_")
        lines.append(comparison_line(f"OUR vs {DISPLAY[label]}", comparison))
    lines.extend(["", "## Evidence Decision", ""])
    for key, value in result["evidence"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "A mechanism is retained as a primary paper claim only when its paired "
            "test passes. Failed ablations are reported as unsupported rather than "
            "being hidden or retuned on these seeds.",
        ]
    )
    (output_dir / "PEGASUS_P6_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = collect(args)
    write_json(args.output_dir / "pegasus_p6_summary.json", result)
    save_aggregate_csv(result, args.output_dir)
    plot_main(result, args.output_dir)
    plot_factorial(result, args.output_dir)
    plot_mechanisms(result, args.output_dir)
    write_report(result, args.output_dir)
    print(args.output_dir / "PEGASUS_P6_REPORT.md")


if __name__ == "__main__":
    main()
