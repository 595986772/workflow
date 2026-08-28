#!/usr/bin/env python3
"""Analyze the Pegasus-B8 26k common-horizon online-training rerun."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pegasus_common_horizon_protocol import (
    ALL_MAIN_METHODS,
    ANALYSIS_DIR,
    EVALUATION_EPISODES,
    FINAL_DIR,
    FINAL_SEEDS,
    FIXED_TRAIN_EPISODES,
    OUR_LABEL,
    P3_FINAL_DIR,
    P6_LEARNING_DIR,
    P8_FINAL_DIR,
    RERUN_METHODS,
    TAIL_EPISODES,
    TAIL_START_EPISODE,
)


DISPLAY_NAMES = {
    "daoc_paper": "DAOC",
    "dqn_wdsa_std_cache": "DQN-WDSA",
    "daoc_our_coord_cache": "DAOC+CoordCache",
    "discrete_sac_std_cache": "DiscreteSAC+Std",
    "coord_cache_discrete_sac": "CoordCache-SAC",
    "lean_our": "OUR",
}
COLORS = {
    "daoc_paper": "#4C78A8",
    "dqn_wdsa_std_cache": "#72B7B2",
    "daoc_our_coord_cache": "#F2CF5B",
    "discrete_sac_std_cache": "#B279A2",
    "coord_cache_discrete_sac": "#E07B54",
    "lean_our": "#2F7D4A",
}
RUN_ROOTS = {
    "daoc_paper": P3_FINAL_DIR,
    "dqn_wdsa_std_cache": P6_LEARNING_DIR,
    "daoc_our_coord_cache": P8_FINAL_DIR,
    **{label: FINAL_DIR for label in RERUN_METHODS},
}
METRICS = (
    "average_finish_time",
    "p95_finish_time",
    "waiting_latency",
    "service_latency",
    "cache_hit_rate",
    "cache_remote_loading_rate",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def method_run_dir(label, seed):
    return RUN_ROOTS[label] / "runs" / label / f"seed_{seed}"


def phase_rows(path, phase):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        return [
            row for row in csv.DictReader(input_file)
            if row["phase"] == phase
        ]


def aggregate(rows):
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return [mean, mean]
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return [mean - half, mean + half]


def paired_lower_is_better(reference, ours):
    reference = np.asarray(reference, dtype=float)
    ours = np.asarray(ours, dtype=float)
    improvements = reference - ours
    nonzero = improvements[np.abs(improvements) > 1e-12]
    interval = ci95(improvements)
    p_value = (
        float(stats.wilcoxon(nonzero, alternative="greater").pvalue)
        if len(nonzero)
        else 1.0
    )
    return {
        "pairs": int(len(improvements)),
        "reference_mean": float(reference.mean()),
        "our_mean": float(ours.mean()),
        "improvement_seconds": float(improvements.mean()),
        "improvement_percent": float(
            100.0 * improvements.mean() / reference.mean()
        ),
        "ci95_seconds": interval,
        "wilcoxon_one_sided_p": p_value,
        "wins": int(np.sum(improvements > 0)),
        "formal_superiority": bool(
            interval[0] > 0
            and p_value < 0.05
            and np.sum(improvements > 0) >= 7
        ),
    }


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


def collect():
    per_seed = []
    integrity = {
        "all_runs_complete": True,
        "all_training_ends_at_26000": True,
        "all_tail_windows_have_1000_rows": True,
        "all_evaluations_have_100_rows": True,
        "all_evaluation_states_frozen": True,
        "all_scenario_banks_paired": True,
        "new_methods_use_fixed_final_checkpoint": True,
    }

    for seed in FINAL_SEEDS:
        seed_banks = []
        for label in ALL_MAIN_METHODS:
            directory = method_run_dir(label, seed)
            summary = read_json(directory / "summary.json")
            train_rows = phase_rows(directory / "episodes.csv", "train")
            tail_rows = [
                row for row in train_rows
                if TAIL_START_EPISODE
                <= int(row["episode"])
                <= FIXED_TRAIN_EPISODES
            ]
            eval_rows = phase_rows(directory / "episodes.csv", "eval")
            integrity["all_runs_complete"] &= (
                summary.get("status") == "complete"
            )
            integrity["all_training_ends_at_26000"] &= (
                len(train_rows) == FIXED_TRAIN_EPISODES
                and int(train_rows[-1]["episode"])
                == FIXED_TRAIN_EPISODES
            )
            integrity["all_tail_windows_have_1000_rows"] &= (
                len(tail_rows) == TAIL_EPISODES
            )
            integrity["all_evaluations_have_100_rows"] &= (
                len(eval_rows) == EVALUATION_EPISODES
            )
            integrity["all_evaluation_states_frozen"] &= bool(
                summary.get("evaluation_state_frozen")
            )
            if label in RERUN_METHODS:
                integrity["new_methods_use_fixed_final_checkpoint"] &= (
                    summary.get("checkpoint_strategy")
                    == "fixed_budget_final"
                    and summary.get("selected_checkpoint_episode")
                    == FIXED_TRAIN_EPISODES
                )
            if len(tail_rows) != TAIL_EPISODES:
                raise RuntimeError(
                    f"Invalid common tail for {label} seed {seed}: "
                    f"{len(tail_rows)} rows"
                )
            if len(eval_rows) != EVALUATION_EPISODES:
                raise RuntimeError(
                    f"Invalid evaluation for {label} seed {seed}: "
                    f"{len(eval_rows)} rows"
                )
            seed_banks.append(
                comparable_bank(directory / "evaluation_scenarios.json")
            )
            per_seed.append(
                {
                    "seed": seed,
                    "method": label,
                    "display_name": DISPLAY_NAMES[label],
                    "tail": aggregate(tail_rows),
                    "evaluation": aggregate(eval_rows),
                    "training_wall_time_sec": float(
                        summary["total_wall_time_sec"]
                    ),
                }
            )
        integrity["all_scenario_banks_paired"] &= all(
            bank == seed_banks[0] for bank in seed_banks[1:]
        )

    if not all(integrity.values()):
        failed = [key for key, value in integrity.items() if not value]
        raise RuntimeError(f"Common-horizon integrity failure: {failed}")
    return per_seed, integrity


def summarize(per_seed, integrity):
    aggregate_summary = {"tail": {}, "evaluation": {}}
    comparisons = {"tail": {}, "evaluation": {}}
    for phase in ("tail", "evaluation"):
        for label in ALL_MAIN_METHODS:
            rows = [row for row in per_seed if row["method"] == label]
            aggregate_summary[phase][label] = {
                metric: {
                    "mean": float(
                        np.mean([row[phase][metric] for row in rows])
                    ),
                    "ci95": ci95(
                        [row[phase][metric] for row in rows]
                    ),
                }
                for metric in METRICS
            }
        ours = [
            row for row in per_seed if row["method"] == OUR_LABEL
        ]
        for reference in ALL_MAIN_METHODS:
            if reference == OUR_LABEL:
                continue
            baseline = [
                row for row in per_seed if row["method"] == reference
            ]
            comparisons[phase][f"our_vs_{reference}"] = {
                metric: paired_lower_is_better(
                    [row[phase][metric] for row in baseline],
                    [row[phase][metric] for row in ours],
                )
                for metric in (
                    "average_finish_time",
                    "p95_finish_time",
                    "waiting_latency",
                    "service_latency",
                )
            }

    return {
        "protocol": {
            "training_horizon": FIXED_TRAIN_EPISODES,
            "tail_window": [
                TAIL_START_EPISODE,
                FIXED_TRAIN_EPISODES,
            ],
            "seeds": list(FINAL_SEEDS),
            "methods": list(ALL_MAIN_METHODS),
        },
        "integrity": integrity,
        "aggregate": aggregate_summary,
        "comparisons": comparisons,
        "per_seed": per_seed,
    }


def write_csv_outputs(summary):
    per_seed_path = ANALYSIS_DIR / "common_horizon_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as output:
        fields = ["seed", "method", "phase", *METRICS]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_seed"]:
            for phase in ("tail", "evaluation"):
                writer.writerow(
                    {
                        "seed": row["seed"],
                        "method": row["method"],
                        "phase": phase,
                        **row[phase],
                    }
                )

    aggregate_path = ANALYSIS_DIR / "common_horizon_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as output:
        fields = [
            "phase",
            "method",
            "metric",
            "mean",
            "ci95_lower",
            "ci95_upper",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for phase, methods in summary["aggregate"].items():
            for label, metrics in methods.items():
                for metric, values in metrics.items():
                    writer.writerow(
                        {
                            "phase": phase,
                            "method": label,
                            "metric": metric,
                            "mean": values["mean"],
                            "ci95_lower": values["ci95"][0],
                            "ci95_upper": values["ci95"][1],
                        }
                    )


def plot_summary(summary):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    rng = np.random.default_rng(20260811)
    for axis, phase, title in zip(
        axes,
        ("tail", "evaluation"),
        ("Online steady-state tail", "Frozen paired evaluation"),
    ):
        means = [
            summary["aggregate"][phase][label][
                "average_finish_time"
            ]["mean"]
            for label in ALL_MAIN_METHODS
        ]
        intervals = [
            summary["aggregate"][phase][label][
                "average_finish_time"
            ]["ci95"]
            for label in ALL_MAIN_METHODS
        ]
        errors = [
            [mean - interval[0] for mean, interval in zip(means, intervals)],
            [interval[1] - mean for mean, interval in zip(means, intervals)],
        ]
        positions = np.arange(len(ALL_MAIN_METHODS))
        axis.bar(
            positions,
            means,
            color=[COLORS[label] for label in ALL_MAIN_METHODS],
            edgecolor="#202124",
            linewidth=0.7,
            yerr=errors,
            capsize=3,
            zorder=2,
        )
        for index, label in enumerate(ALL_MAIN_METHODS):
            values = [
                row[phase]["average_finish_time"]
                for row in summary["per_seed"]
                if row["method"] == label
            ]
            jitter = rng.uniform(-0.11, 0.11, size=len(values))
            axis.scatter(
                index + jitter,
                values,
                s=22,
                facecolor="white",
                edgecolor="#202124",
                linewidth=0.65,
                zorder=3,
            )
        axis.set_xticks(
            positions,
            [DISPLAY_NAMES[label] for label in ALL_MAIN_METHODS],
            rotation=24,
            ha="right",
        )
        axis.set_title(title)
        axis.set_ylabel("Mean DAG completion time (s)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        ANALYSIS_DIR / "common_horizon_main_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        ANALYSIS_DIR / "common_horizon_main_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def write_report(summary):
    lines = [
        "# Pegasus-B8 26,000轮统一训练预算报告",
        "",
        "- 全部主对比学习方法统一到26,000轮。",
        "- 在线稳态指标固定使用第25,001–26,000轮原始日志。",
        "- 冻结评估使用每seed 100个完全配对场景。",
        "- Seeds 51–60是最终确认seed，不称为独立holdout。",
        "",
        "## 完整性审计",
        "",
    ]
    for key, value in summary["integrity"].items():
        lines.append(f"- `{key}`: `{value}`")
    for phase, title in (
        ("tail", "在线训练稳态尾部"),
        ("evaluation", "冻结配对评估"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 方法 | 平均DAG完成时间 (s) | 95% CI | P95 (s) |",
                "|---|---:|---:|---:|",
            ]
        )
        for label in ALL_MAIN_METHODS:
            values = summary["aggregate"][phase][label]
            finish = values["average_finish_time"]
            p95 = values["p95_finish_time"]["mean"]
            lines.append(
                f"| {DISPLAY_NAMES[label]} | {finish['mean']:.6f} | "
                f"[{finish['ci95'][0]:.6f}, {finish['ci95'][1]:.6f}] | "
                f"{p95:.6f} |"
            )
        lines.extend(["", "### OUR配对优势", ""])
        for reference in ALL_MAIN_METHODS:
            if reference == OUR_LABEL:
                continue
            comparison = summary["comparisons"][phase][
                f"our_vs_{reference}"
            ]["average_finish_time"]
            lines.append(
                f"- OUR vs {DISPLAY_NAMES[reference]}："
                f"{comparison['improvement_percent']:.3f}%，"
                f"{comparison['wins']}/10 seed获胜，"
                f"p={comparison['wilcoxon_one_sided_p']:.6g}，"
                f"formal={comparison['formal_superiority']}。"
            )
    (ANALYSIS_DIR / "COMMON_HORIZON_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed, integrity = collect()
    summary = summarize(per_seed, integrity)
    write_json(ANALYSIS_DIR / "common_horizon_summary.json", summary)
    write_csv_outputs(summary)
    plot_summary(summary)
    write_report(summary)
    print(ANALYSIS_DIR / "COMMON_HORIZON_REPORT_ZH.md")


if __name__ == "__main__":
    main()
