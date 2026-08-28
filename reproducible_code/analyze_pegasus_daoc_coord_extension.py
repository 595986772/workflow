#!/usr/bin/env python3
"""Analyze DAOC with the OUR coordinated cache and merge P6-P8 evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze_a0_coordination import aggregate_run, paired_superiority
from capacity_protocol import deterministic_capacity_assignment
from pegasus_daoc_coord_extension_protocol import (
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DAOC_COORD_LABEL,
    DAOC_LABEL,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FINAL_DIR,
    FINAL_SEEDS,
    P3_FINAL_DIR,
    P6_SUMMARY,
    P7_SUMMARY,
    PROTOCOL_VERSION,
    validate_protocol,
)


STD_SAC_LABEL = "discrete_sac_std_cache"
COORD_SAC_LABEL = "coord_cache_discrete_sac"
DISPLAY = {
    "random": "Random",
    "nearest": "Nearest",
    "greedy": "Nearest-with-Service",
    "dqn_wdsa_std_cache": "DQN-WDSA",
    DAOC_LABEL: "DAOC-paper",
    "centralized_greedy_daoc": "Centralized-Greedy-DQN",
    DAOC_COORD_LABEL: "DAOC+OUR-CoordCache",
    STD_SAC_LABEL: "DiscreteSAC-StdCache",
    COORD_SAC_LABEL: "CoordCache-DiscreteSAC",
    "lean_our": "OUR",
}
MAIN_ORDER = tuple(DISPLAY)
FOCUS_ORDER = (
    DAOC_LABEL,
    DAOC_COORD_LABEL,
    STD_SAC_LABEL,
    COORD_SAC_LABEL,
    "lean_our",
)
COLORS = {
    DAOC_LABEL: "#6B7280",
    DAOC_COORD_LABEL: "#E69F00",
    STD_SAC_LABEL: "#CC79A7",
    COORD_SAC_LABEL: "#8B6BB1",
    "lean_our": "#007C91",
}
MARKERS = {
    DAOC_LABEL: "o",
    DAOC_COORD_LABEL: "P",
    STD_SAC_LABEL: "s",
    COORD_SAC_LABEL: "D",
    "lean_our": "^",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
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


def bool_value(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, half


def aggregate_seed_records(per_seed, label):
    metrics = per_seed[0]["methods"][label].keys()
    return {
        metric: float(
            np.mean([row["methods"][label][metric] for row in per_seed])
        )
        for metric in metrics
    }


def matched_daoc_training_config(control, daoc):
    shared_keys = (
        "algorithm",
        "train_episodes",
        "eval_episodes",
        "num_users",
        "num_servers",
        "num_services",
        "num_tasks",
        "dag_dataset_sha256",
        "dag_depth_increment",
        "dependency_data_scale",
        "server_capacity_multiset",
        "capacity_assignment_namespace",
        "batch_size",
        "min_experiences",
        "filling_steps",
        "steps_to_updates",
        "max_explore",
        "gamma",
        "learning_rate",
        "learning_rate_schedule",
        "epsilon",
        "hidden_units",
        "bandwidth",
        "beta",
        "beta_min",
        "beta_decay",
        "reward_mode",
        "potential_reward_weight",
        "cache_score_alpha",
        "cache_history_alpha",
        "cache_update_interval",
        "cache_freeze_episode",
        "eval_scenario_bank",
        "eval_bank_scope",
        "eval_dag_families",
        "checkpoint_every",
        "validation_scenarios",
        "convergence_mode",
        "convergence_min_episodes",
        "convergence_window",
        "convergence_patience",
    )
    return all(control.get(key) == daoc.get(key) for key in shared_keys)


def collect():
    protocol = validate_protocol(require_p7=True)
    p6 = read_json(P6_SUMMARY)
    p7 = read_json(P7_SUMMARY)
    p6_by_seed = {row["seed"]: row for row in p6["per_seed"]}
    p7_by_seed = {row["seed"]: row for row in p7["per_seed"]}
    integrity = {
        "all_new_runs_complete": True,
        "all_new_runs_converged": True,
        "new_control_trained_from_scratch": True,
        "scenario_banks_paired": True,
        "capacity_assignments_exact": True,
        "dataset_and_environment_exact": True,
        "daoc_state_scheduler_guidance_reward_training_matched": True,
        "only_cache_subsystem_changed": True,
        "evaluation_state_frozen": True,
        "all_tasks_executed_once": True,
        "p6_parent_integrity_passed": all(p6["integrity"].values()),
        "p7_parent_integrity_passed": all(p7["integrity"].values()),
    }
    per_seed = []
    for seed in FINAL_SEEDS:
        control_dir = FINAL_DIR / "runs" / DAOC_COORD_LABEL / f"seed_{seed}"
        daoc_dir = P3_FINAL_DIR / "runs" / DAOC_LABEL / f"seed_{seed}"
        summary = read_json(control_dir / "summary.json")
        daoc_summary = read_json(daoc_dir / "summary.json")
        config = read_json(control_dir / "config.json")
        daoc_config = read_json(daoc_dir / "config.json")
        arguments = config["arguments"]
        daoc_arguments = daoc_config["arguments"]
        rows = evaluation_rows(control_dir / "episodes.csv")

        integrity["all_new_runs_complete"] &= summary.get("status") == "complete"
        integrity["all_new_runs_converged"] &= bool(
            summary.get("eligible_for_comparison", False)
            and summary.get("convergence", {}).get("reached", False)
        )
        integrity["new_control_trained_from_scratch"] &= (
            summary.get("selected_checkpoint_sha256")
            != daoc_summary.get("selected_checkpoint_sha256")
            and arguments.get("revision_id") == PROTOCOL_VERSION
        )
        integrity["evaluation_state_frozen"] &= bool(
            summary.get("evaluation_state_frozen", False)
        ) and all(not bool_value(row["cache_updates_enabled"]) for row in rows)
        integrity["all_tasks_executed_once"] &= all(
            bool_value(row["all_tasks_executed_once"]) for row in rows
        )
        integrity["dataset_and_environment_exact"] &= (
            arguments.get("dag_dataset_sha256") == EXPECTED_DATASET_SHA256
            and arguments.get("bandwidth") == 15000
            and arguments.get("num_users") == 20
            and arguments.get("num_servers") == 10
            and arguments.get("num_services") == 10
            and arguments.get("num_tasks") == 31
        )
        integrity[
            "daoc_state_scheduler_guidance_reward_training_matched"
        ] &= matched_daoc_training_config(arguments, daoc_arguments)
        integrity["only_cache_subsystem_changed"] &= (
            arguments.get("cache_policy") == "critical_path_joint"
            and arguments.get("cache_server_quality") is True
            and arguments.get("cache_coverage_constraint") is True
            and arguments.get("cache_dependency_awareness") is True
            and daoc_arguments.get("cache_policy")
            == "paper_popularity_cost_ema"
            and daoc_arguments.get("cache_coverage_constraint") is False
        )

        expected_capacities = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        capacities = {
            int(key): int(value)
            for key, value in summary["server_capacities"].items()
        }
        integrity["capacity_assignments_exact"] &= capacities == expected_capacities

        control_bank = comparable_bank(
            control_dir / "evaluation_scenarios.json"
        )
        for label in (DAOC_LABEL, "lean_our", "centralized_greedy_daoc"):
            reference_dir = P3_FINAL_DIR / "runs" / label / f"seed_{seed}"
            integrity["scenario_banks_paired"] &= control_bank == comparable_bank(
                reference_dir / "evaluation_scenarios.json"
            )

        p6_methods = p6_by_seed[seed]["methods"]
        p7_methods = p7_by_seed[seed]["methods"]
        per_seed.append(
            {
                "seed": seed,
                "selected_checkpoint_episode": summary[
                    "selected_checkpoint_episode"
                ],
                "methods": {
                    DAOC_LABEL: p6_methods[DAOC_LABEL],
                    DAOC_COORD_LABEL: aggregate_run(rows),
                    "centralized_greedy_daoc": p6_methods[
                        "centralized_greedy_daoc"
                    ],
                    STD_SAC_LABEL: p7_methods[STD_SAC_LABEL],
                    COORD_SAC_LABEL: p6_methods[COORD_SAC_LABEL],
                    "lean_our": p6_methods["lean_our"],
                },
            }
        )

    if not all(integrity.values()):
        failed = [key for key, value in integrity.items() if not value]
        raise RuntimeError(f"P8 integrity audit failed: {failed}")

    labels = tuple(per_seed[0]["methods"])
    aggregates = {
        label: aggregate_seed_records(per_seed, label) for label in labels
    }
    values = {
        label: np.asarray(
            [row["methods"][label]["mean_finish_time"] for row in per_seed],
            dtype=float,
        )
        for label in labels
    }
    p95_values = {
        label: np.asarray(
            [
                row["methods"][label]["mean_p95_finish_time"]
                for row in per_seed
            ],
            dtype=float,
        )
        for label in labels
    }
    comparisons = {
        "daoc_coord_vs_daoc": paired_superiority(
            values[DAOC_LABEL], values[DAOC_COORD_LABEL], formal=True
        ),
        "our_vs_daoc_coord": paired_superiority(
            values[DAOC_COORD_LABEL], values["lean_our"], formal=True
        ),
        "coord_sac_vs_daoc_coord": paired_superiority(
            values[DAOC_COORD_LABEL], values[COORD_SAC_LABEL], formal=True
        ),
        "daoc_coord_vs_centralized_greedy": paired_superiority(
            values["centralized_greedy_daoc"],
            values[DAOC_COORD_LABEL],
            formal=True,
        ),
    }
    p95_comparisons = {
        "daoc_coord_vs_daoc": paired_superiority(
            p95_values[DAOC_LABEL],
            p95_values[DAOC_COORD_LABEL],
            formal=True,
        ),
        "our_vs_daoc_coord": paired_superiority(
            p95_values[DAOC_COORD_LABEL],
            p95_values["lean_our"],
            formal=True,
        ),
        "coord_sac_vs_daoc_coord": paired_superiority(
            p95_values[DAOC_COORD_LABEL],
            p95_values[COORD_SAC_LABEL],
            formal=True,
        ),
    }
    evidence = {
        "daoc_coord_cache_control_valid": True,
        "our_coord_cache_improves_daoc_mean": comparisons[
            "daoc_coord_vs_daoc"
        ]["passed"],
        "our_coord_cache_improves_daoc_p95": p95_comparisons[
            "daoc_coord_vs_daoc"
        ]["passed"],
        "our_beats_daoc_with_same_coord_cache_mean": comparisons[
            "our_vs_daoc_coord"
        ]["passed"],
        "our_beats_daoc_with_same_coord_cache_p95": p95_comparisons[
            "our_vs_daoc_coord"
        ]["passed"],
    }
    return {
        "protocol": protocol,
        "integrity": integrity,
        "evidence": evidence,
        "per_seed": per_seed,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "p95_comparisons": p95_comparisons,
        "parent_summary_hashes": {
            "p6": hashlib.sha256(Path(P6_SUMMARY).read_bytes()).hexdigest(),
            "p7": hashlib.sha256(Path(P7_SUMMARY).read_bytes()).hexdigest(),
        },
    }


def augmented_aggregates(summary):
    p6 = read_json(P6_SUMMARY)
    p7 = read_json(P7_SUMMARY)
    aggregates = {
        label: dict(p6["aggregates"][label])
        for label in MAIN_ORDER
        if label not in {STD_SAC_LABEL, DAOC_COORD_LABEL}
    }
    aggregates[STD_SAC_LABEL] = p7["aggregates"][STD_SAC_LABEL]
    aggregates[DAOC_COORD_LABEL] = summary["aggregates"][DAOC_COORD_LABEL]
    return aggregates


def write_tables(output_dir, summary):
    aggregates = augmented_aggregates(summary)
    fields = (
        "mean_finish_time",
        "mean_p95_finish_time",
        "mean_waiting_latency",
        "mean_cache_hit_rate",
        "mean_cache_service_coverage",
        "mean_remote_loading_rate",
        "mean_inference_time_per_decision_ms",
    )
    with (output_dir / "augmented_main_baselines_p8.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=("method", *fields))
        writer.writeheader()
        for label in MAIN_ORDER:
            writer.writerow(
                {
                    "method": DISPLAY[label],
                    **{field: aggregates[label][field] for field in fields},
                }
            )
    with (output_dir / "daoc_coord_cache_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "seed",
                "daoc_mean",
                "daoc_coord_mean",
                "our_mean",
                "daoc_p95",
                "daoc_coord_p95",
                "our_p95",
                "selected_checkpoint_episode",
            )
        )
        for row in summary["per_seed"]:
            writer.writerow(
                (
                    row["seed"],
                    row["methods"][DAOC_LABEL]["mean_finish_time"],
                    row["methods"][DAOC_COORD_LABEL]["mean_finish_time"],
                    row["methods"]["lean_our"]["mean_finish_time"],
                    row["methods"][DAOC_LABEL]["mean_p95_finish_time"],
                    row["methods"][DAOC_COORD_LABEL]["mean_p95_finish_time"],
                    row["methods"]["lean_our"]["mean_p95_finish_time"],
                    row["selected_checkpoint_episode"],
                )
            )


def plot_focus(output_dir, summary):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.0), layout="constrained"
    )
    metric_specs = (
        ("mean_finish_time", "Mean DAG completion time (s)", "(a) Mean"),
        ("mean_p95_finish_time", "P95 DAG completion time (s)", "(b) P95"),
    )
    rng = np.random.default_rng(20260811)
    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        for index, label in enumerate(FOCUS_ORDER):
            values = np.asarray(
                [row["methods"][label][metric] for row in summary["per_seed"]],
                dtype=float,
            )
            mean, half = mean_ci95(values)
            ax.bar(
                index,
                mean,
                width=0.68,
                color=COLORS[label],
                edgecolor="#333333",
                linewidth=0.6,
                zorder=2,
            )
            ax.errorbar(
                index,
                mean,
                yerr=half,
                color="#222222",
                linewidth=0.9,
                capsize=2.5,
                zorder=4,
            )
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                marker=MARKERS[label],
                s=15,
                color="white",
                edgecolor="#222222",
                linewidth=0.55,
                zorder=5,
            )
        ax.set_xticks(range(len(FOCUS_ORDER)))
        ax.set_xticklabels(
            [DISPLAY[label].replace("+", "+\n").replace("-", "-\n", 1) for label in FOCUS_ORDER]
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, zorder=0)
    fig.suptitle(
        "Controlled cache and scheduler comparisons",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.savefig(output_dir / "p8_controlled_comparison.pdf", facecolor="white")
    fig.savefig(
        output_dir / "p8_controlled_comparison.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def result_line(name, result):
    conclusion = "passes" if result["passed"] else "does not pass"
    return (
        f"{name}: {result['mean_improvement_percent']:.2f}% "
        f"({result['wins']}/10 seeds, 95% CI "
        f"[{result['ci95_lower_sec']:.4f}, {result['ci95_upper_sec']:.4f}] s, "
        f"p={result['wilcoxon_one_sided_p']:.5f}); {conclusion}."
    )


def write_reports(output_dir, summary):
    aggregates = summary["aggregates"]
    comparisons = summary["comparisons"]
    p95 = summary["p95_comparisons"]
    rows = []
    for label in FOCUS_ORDER:
        values = aggregates[label]
        rows.append(
            f"| {DISPLAY[label]} | {values['mean_finish_time']:.4f} | "
            f"{values['mean_p95_finish_time']:.4f} | "
            f"{values['mean_cache_hit_rate']:.4f} | "
            f"{values['mean_cache_service_coverage']:.4f} | "
            f"{values['mean_remote_loading_rate']:.4f} |"
        )
    evidence_lines = [
        result_line(
            "DAOC+CoordCache versus DAOC",
            comparisons["daoc_coord_vs_daoc"],
        ),
        result_line(
            "OUR versus DAOC+CoordCache",
            comparisons["our_vs_daoc_coord"],
        ),
        result_line(
            "CoordCache-SAC versus DAOC+CoordCache",
            comparisons["coord_sac_vs_daoc_coord"],
        ),
        "P95: "
        + result_line(
            "DAOC+CoordCache versus DAOC",
            p95["daoc_coord_vs_daoc"],
        ),
        "P95: "
        + result_line(
            "OUR versus DAOC+CoordCache",
            p95["our_vs_daoc_coord"],
        ),
    ]
    english = [
        "# Pegasus-B8 DAOC + OUR Coordinated-Cache Control",
        "",
        "DAOC+OUR-CoordCache keeps the DAOC state, DQN scheduler, decaying action "
        "guidance, terminal reward and convergence profile. Only the independent "
        "DAOC cache is replaced by the complete OUR coordinated-cache subsystem.",
        "",
        "| Method | Mean (s) | P95 (s) | Cache hit | Coverage | Remote load |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Paired Evidence",
        "",
        *[f"- {line}" for line in evidence_lines],
        "",
        "All values use seeds 51-60 and 100 fully paired frozen scenarios per seed. "
        "These seeds are paired final confirmation, not an independent holdout.",
    ]
    (output_dir / "PEGASUS_P8_DAOC_COORD_REPORT.md").write_text(
        "\n".join(english) + "\n", encoding="utf-8"
    )

    chinese = [
        "# Pegasus-B8 DAOC + OUR 中央协调缓存受控实验",
        "",
        "DAOC+OUR-CoordCache 完整保留 DAOC 的状态、DQN 调度器、衰减动作引导、"
        "终止奖励和收敛超参数，只将 DAOC 独立缓存替换为 OUR 的完整中央协调缓存。",
        "",
        "| 方法 | 平均完成时间 (s) | P95 (s) | 命中率 | 覆盖率 | 远程加载率 |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 配对统计",
        "",
        *[f"- {line}" for line in evidence_lines],
        "",
        "所有结果使用 seeds 51–60，每个 seed 100 个完全配对冻结场景。"
        "这些 seed 属于配对最终确认，不称为独立 holdout。",
    ]
    (output_dir / "PEGASUS_P8_DAOC_COORD_REPORT_ZH.md").write_text(
        "\n".join(chinese) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect()
    write_json(
        args.output_dir / "daoc_coord_cache_extension_summary.json",
        summary,
    )
    write_tables(args.output_dir, summary)
    plot_focus(args.output_dir, summary)
    write_reports(args.output_dir, summary)
    print(json.dumps(summary["evidence"], indent=2))


if __name__ == "__main__":
    main()
