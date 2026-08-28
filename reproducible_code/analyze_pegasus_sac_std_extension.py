#!/usr/bin/env python3
"""Analyze the P7 standard-cache Discrete SAC extension."""

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
from pegasus_sac_std_extension_protocol import (
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    COORD_SAC_LABEL,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FINAL_DIR,
    FINAL_SEEDS,
    P3_FINAL_DIR,
    P5_SAC_DIR,
    P6_SUMMARY,
    PROTOCOL_VERSION,
    STD_SAC_LABEL,
    validate_protocol,
)


DISPLAY = {
    "random": "Random",
    "nearest": "Nearest",
    "greedy": "Nearest-with-Service",
    "dqn_wdsa_std_cache": "DQN-WDSA",
    "daoc_paper": "DAOC-paper",
    "centralized_greedy_daoc": "Centralized-Greedy-DQN",
    STD_SAC_LABEL: "DiscreteSAC-StdCache",
    COORD_SAC_LABEL: "CoordCache-DiscreteSAC",
    "lean_our": "OUR",
}
MAIN_ORDER = tuple(DISPLAY)
FOCUS_ORDER = (
    "daoc_paper",
    STD_SAC_LABEL,
    COORD_SAC_LABEL,
    "lean_our",
)
COLORS = {
    "daoc_paper": "#6B7280",
    STD_SAC_LABEL: "#CC79A7",
    COORD_SAC_LABEL: "#8B6BB1",
    "lean_our": "#007C91",
}
MARKERS = {
    "daoc_paper": "o",
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


def reference_dir(label, seed):
    if label == COORD_SAC_LABEL:
        return P5_SAC_DIR / "runs" / label / f"seed_{seed}"
    return P3_FINAL_DIR / "runs" / label / f"seed_{seed}"


def matched_training_config(standard, coordinated):
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
        "n_step",
        "entropy_coefficient",
        "sac_target_entropy_ratio",
        "sac_target_tau",
        "learning_rate",
        "learning_rate_schedule",
        "epsilon",
        "hidden_units",
        "bandwidth",
        "reward_mode",
        "potential_reward_weight",
        "cache_score_alpha",
        "cache_history_alpha",
        "cache_update_interval",
        "cache_freeze_episode",
        "telemetry_min_samples",
        "telemetry_freshness_half_life",
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
    return all(standard.get(key) == coordinated.get(key) for key in shared_keys)


def collect():
    protocol = validate_protocol()
    p6 = read_json(P6_SUMMARY)
    p6_by_seed = {row["seed"]: row for row in p6["per_seed"]}
    integrity = {
        "all_new_runs_complete": True,
        "all_new_runs_converged": True,
        "new_method_trained_from_scratch": True,
        "scenario_banks_paired": True,
        "capacity_assignments_exact": True,
        "dataset_and_environment_exact": True,
        "sac_state_reward_action_training_matched": True,
        "only_cache_subsystem_changed": True,
        "evaluation_state_frozen": True,
        "all_tasks_executed_once": True,
        "p6_parent_integrity_passed": all(p6["integrity"].values()),
    }
    per_seed = []
    for seed in FINAL_SEEDS:
        standard_dir = FINAL_DIR / "runs" / STD_SAC_LABEL / f"seed_{seed}"
        coordinated_dir = reference_dir(COORD_SAC_LABEL, seed)
        summary = read_json(standard_dir / "summary.json")
        standard_config = read_json(standard_dir / "config.json")
        coordinated_config = read_json(coordinated_dir / "config.json")
        arguments = standard_config["arguments"]
        coordinated_arguments = coordinated_config["arguments"]
        rows = evaluation_rows(standard_dir / "episodes.csv")

        integrity["all_new_runs_complete"] &= summary.get("status") == "complete"
        integrity["all_new_runs_converged"] &= bool(
            summary.get("eligible_for_comparison", False)
            and summary.get("convergence", {}).get("reached", False)
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
        integrity["sac_state_reward_action_training_matched"] &= (
            matched_training_config(arguments, coordinated_arguments)
        )
        integrity["only_cache_subsystem_changed"] &= (
            arguments.get("cache_policy") == "popularity_ema"
            and arguments.get("cache_server_quality") is False
            and arguments.get("cache_coverage_constraint") is False
            and arguments.get("cache_dependency_awareness") is False
            and coordinated_arguments.get("cache_policy")
            == "critical_path_joint"
            and coordinated_arguments.get("cache_coverage_constraint") is True
        )
        integrity["new_method_trained_from_scratch"] &= (
            summary.get("selected_checkpoint_sha256")
            != read_json(coordinated_dir / "summary.json").get(
                "selected_checkpoint_sha256"
            )
            and arguments.get("revision_id") == PROTOCOL_VERSION
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

        reference_bank = comparable_bank(
            standard_dir / "evaluation_scenarios.json"
        )
        for label in (COORD_SAC_LABEL, "lean_our", "daoc_paper"):
            integrity["scenario_banks_paired"] &= reference_bank == comparable_bank(
                reference_dir(label, seed) / "evaluation_scenarios.json"
            )

        seed_record = {
            "seed": seed,
            "selected_checkpoint_episode": summary[
                "selected_checkpoint_episode"
            ],
            "methods": {
                STD_SAC_LABEL: aggregate_run(rows),
                COORD_SAC_LABEL: p6_by_seed[seed]["methods"][COORD_SAC_LABEL],
                "lean_our": p6_by_seed[seed]["methods"]["lean_our"],
                "daoc_paper": p6_by_seed[seed]["methods"]["daoc_paper"],
            },
        }
        per_seed.append(seed_record)

    if not all(integrity.values()):
        failed = [key for key, value in integrity.items() if not value]
        raise RuntimeError(f"P7 integrity audit failed: {failed}")

    aggregates = {
        label: aggregate_seed_records(per_seed, label)
        for label in (
            STD_SAC_LABEL,
            COORD_SAC_LABEL,
            "lean_our",
            "daoc_paper",
        )
    }
    values = {
        label: np.asarray(
            [
                row["methods"][label]["mean_finish_time"]
                for row in per_seed
            ],
            dtype=float,
        )
        for label in aggregates
    }
    p95_values = {
        label: np.asarray(
            [
                row["methods"][label]["mean_p95_finish_time"]
                for row in per_seed
            ],
            dtype=float,
        )
        for label in aggregates
    }
    comparisons = {
        "coord_sac_vs_std_sac": paired_superiority(
            values[STD_SAC_LABEL],
            values[COORD_SAC_LABEL],
            formal=True,
        ),
        "our_vs_std_sac": paired_superiority(
            values[STD_SAC_LABEL],
            values["lean_our"],
            formal=True,
        ),
        "our_vs_coord_sac": paired_superiority(
            values[COORD_SAC_LABEL],
            values["lean_our"],
            formal=True,
        ),
    }
    p95_comparisons = {
        "coord_sac_vs_std_sac": paired_superiority(
            p95_values[STD_SAC_LABEL],
            p95_values[COORD_SAC_LABEL],
            formal=True,
        ),
        "our_vs_std_sac": paired_superiority(
            p95_values[STD_SAC_LABEL],
            p95_values["lean_our"],
            formal=True,
        ),
        "our_vs_coord_sac": paired_superiority(
            p95_values[COORD_SAC_LABEL],
            p95_values["lean_our"],
            formal=True,
        ),
    }
    evidence = {
        "standard_cache_sac_is_valid_baseline": True,
        "coord_cache_effect_under_sac_supported": comparisons[
            "coord_sac_vs_std_sac"
        ]["passed"],
        "our_beats_standard_cache_sac": comparisons[
            "our_vs_std_sac"
        ]["passed"],
        "our_beats_coordinated_sac_mean": comparisons[
            "our_vs_coord_sac"
        ]["passed"],
        "our_beats_coordinated_sac_p95": p95_comparisons[
            "our_vs_coord_sac"
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
        "p6_summary_sha256": __import__("hashlib").sha256(
            Path(P6_SUMMARY).read_bytes()
        ).hexdigest(),
    }


def augmented_aggregates(summary):
    p6 = read_json(P6_SUMMARY)
    aggregates = {
        label: dict(p6["aggregates"][label])
        for label in MAIN_ORDER
        if label != STD_SAC_LABEL
    }
    aggregates[STD_SAC_LABEL] = summary["aggregates"][STD_SAC_LABEL]
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
    with (output_dir / "augmented_main_baselines.csv").open(
        "w",
        newline="",
        encoding="utf-8",
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

    with (output_dir / "sac_cache_per_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "seed",
                "std_sac_mean",
                "coord_sac_mean",
                "our_mean",
                "std_sac_p95",
                "coord_sac_p95",
                "our_p95",
                "selected_checkpoint_episode",
            )
        )
        for row in summary["per_seed"]:
            writer.writerow(
                (
                    row["seed"],
                    row["methods"][STD_SAC_LABEL]["mean_finish_time"],
                    row["methods"][COORD_SAC_LABEL]["mean_finish_time"],
                    row["methods"]["lean_our"]["mean_finish_time"],
                    row["methods"][STD_SAC_LABEL]["mean_p95_finish_time"],
                    row["methods"][COORD_SAC_LABEL]["mean_p95_finish_time"],
                    row["methods"]["lean_our"]["mean_p95_finish_time"],
                    row["selected_checkpoint_episode"],
                )
            )


def plot_focus(output_dir, summary):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
        layout="constrained",
    )
    metric_specs = (
        ("mean_finish_time", "Mean DAG completion time (s)", "(a) Mean"),
        ("mean_p95_finish_time", "P95 DAG completion time (s)", "(b) P95"),
    )
    rng = np.random.default_rng(20260810)
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
                color=COLORS[label],
                edgecolor="#333333",
                linewidth=0.6,
                width=0.68,
                zorder=2,
            )
            ax.errorbar(
                index,
                mean,
                yerr=half,
                color="#222222",
                capsize=2.5,
                linewidth=0.9,
                zorder=4,
            )
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                color="white",
                edgecolor="#222222",
                linewidth=0.55,
                s=15,
                marker=MARKERS[label],
                zorder=5,
            )
        ax.set_xticks(range(len(FOCUS_ORDER)))
        ax.set_xticklabels(
            [DISPLAY[label].replace("-", "-\n", 1) for label in FOCUS_ORDER]
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, zorder=0)
    fig.suptitle(
        "Discrete SAC cache control and OUR comparison",
        fontsize=10.5,
        fontweight="bold",
    )
    for suffix, kwargs in (
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(
            output_dir / f"p7_sac_cache_comparison.{suffix}",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def comparison_sentence(name, comparison):
    status = "passes" if comparison["passed"] else "does not pass"
    return (
        f"{name}: {comparison['mean_improvement_percent']:.2f}% "
        f"({comparison['wins']}/10 seeds, 95% CI "
        f"[{comparison['ci95_lower_sec']:.4f}, "
        f"{comparison['ci95_upper_sec']:.4f}] s, "
        f"p={comparison['wilcoxon_one_sided_p']:.5f}); {status}."
    )


def write_reports(output_dir, summary):
    aggregates = summary["aggregates"]
    comparisons = summary["comparisons"]
    p95 = summary["p95_comparisons"]
    english = [
        "# Pegasus-B8 Standard-Cache Discrete SAC Extension",
        "",
        "All methods use seeds 51-60 and 100 fully paired frozen scenarios per seed. "
        "DiscreteSAC-StdCache and CoordCache-DiscreteSAC share the same causal state, "
        "dense reward, discrete action space and training settings; only the cache "
        "subsystem differs.",
        "",
        "| Method | Mean (s) | P95 (s) | Cache hit | Coverage | Remote load |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in FOCUS_ORDER:
        values = aggregates[label]
        english.append(
            f"| {DISPLAY[label]} | {values['mean_finish_time']:.4f} | "
            f"{values['mean_p95_finish_time']:.4f} | "
            f"{values['mean_cache_hit_rate']:.4f} | "
            f"{values['mean_cache_service_coverage']:.4f} | "
            f"{values['mean_remote_loading_rate']:.4f} |"
        )
    english.extend(
        [
            "",
            "## Paired Evidence",
            "",
            "- " + comparison_sentence(
                "CoordCache-SAC versus StdCache-SAC",
                comparisons["coord_sac_vs_std_sac"],
            ),
            "- " + comparison_sentence(
                "OUR versus StdCache-SAC",
                comparisons["our_vs_std_sac"],
            ),
            "- " + comparison_sentence(
                "OUR versus CoordCache-SAC",
                comparisons["our_vs_coord_sac"],
            ),
            "- P95: " + comparison_sentence(
                "OUR versus StdCache-SAC",
                p95["our_vs_std_sac"],
            ),
            "- P95: " + comparison_sentence(
                "OUR versus CoordCache-SAC",
                p95["our_vs_coord_sac"],
            ),
            "",
            "Seeds 51-60 are paired final-confirmation seeds, not an independent holdout. "
            "The standard-cache SAC is a controlled project baseline rather than a claimed "
            "verbatim reproduction of an external SAC paper.",
        ]
    )
    (output_dir / "PEGASUS_P7_SAC_STD_REPORT.md").write_text(
        "\n".join(english) + "\n",
        encoding="utf-8",
    )

    chinese = [
        "# Pegasus-B8 标准缓存 Discrete SAC 补充实验",
        "",
        "DiscreteSAC-StdCache 与 CoordCache-DiscreteSAC 的状态、因果稠密奖励、"
        "离散动作空间和训练超参数完全相同，只将协调缓存换为逐服务器独立"
        " popularity EMA 缓存。全部结果使用 seeds 51–60，每个 seed 100 个完全配对场景。",
        "",
        "| 方法 | 平均完成时间 (s) | P95 (s) | 命中率 | 覆盖率 | 远程加载率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in FOCUS_ORDER:
        values = aggregates[label]
        chinese.append(
            f"| {DISPLAY[label]} | {values['mean_finish_time']:.4f} | "
            f"{values['mean_p95_finish_time']:.4f} | "
            f"{values['mean_cache_hit_rate']:.4f} | "
            f"{values['mean_cache_service_coverage']:.4f} | "
            f"{values['mean_remote_loading_rate']:.4f} |"
        )
    chinese.extend(
        [
            "",
            "## 配对统计",
            "",
            "- " + comparison_sentence(
                "CoordCache-SAC 相对 StdCache-SAC",
                comparisons["coord_sac_vs_std_sac"],
            ),
            "- " + comparison_sentence(
                "OUR 相对 StdCache-SAC",
                comparisons["our_vs_std_sac"],
            ),
            "- " + comparison_sentence(
                "OUR 相对 CoordCache-SAC",
                comparisons["our_vs_coord_sac"],
            ),
            "- P95：" + comparison_sentence(
                "OUR 相对 StdCache-SAC",
                p95["our_vs_std_sac"],
            ),
            "- P95：" + comparison_sentence(
                "OUR 相对 CoordCache-SAC",
                p95["our_vs_coord_sac"],
            ),
            "",
            "Seeds 51–60 属于配对最终确认，不称为独立 holdout。"
            "DiscreteSAC-StdCache 是当前项目中的受控基线，不声称为某篇外部 SAC "
            "论文的逐字复现。",
        ]
    )
    (output_dir / "PEGASUS_P7_SAC_STD_REPORT_ZH.md").write_text(
        "\n".join(chinese) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect()
    write_json(args.output_dir / "sac_std_cache_extension_summary.json", summary)
    write_tables(args.output_dir, summary)
    plot_focus(args.output_dir, summary)
    write_reports(args.output_dir, summary)
    print(json.dumps(summary["evidence"], indent=2))


if __name__ == "__main__":
    main()
