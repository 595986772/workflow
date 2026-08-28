#!/usr/bin/env python3
"""Analyze Alibaba-CP100 OUR-DQN and centralized-cache extensions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_strict_environment_suite import paired_statistics


LABELS = (
    "guided_full",
    "centralized_greedy_daoc",
    "our_dqn",
    "lean_our",
)
DISPLAY_NAMES = {
    "guided_full": "DAOC",
    "centralized_greedy_daoc": "Centralized-Greedy",
    "our_dqn": "OUR-DQN",
    "lean_our": "OUR",
}


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--extension-suite-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    return parser.parse_args()


def run_dir(source_suite, extension_suite, label, seed):
    suite = (
        source_suite
        if label in {"guided_full", "lean_our"}
        else extension_suite
    )
    return suite / "runs" / label / f"seed_{seed}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def mean_metric(rows, metric):
    return float(np.mean([float(row[metric]) for row in rows]))


def collect(source_suite, extension_suite, seeds):
    per_seed = []
    all_paired = True
    all_converged = True
    for seed in seeds:
        rows_by_label = {}
        summaries = {}
        for label in LABELS:
            current = run_dir(
                source_suite,
                extension_suite,
                label,
                seed,
            )
            rows_by_label[label] = read_rows(
                current / "episodes.csv"
            )
            summaries[label] = read_json(current / "summary.json")
            all_converged = all_converged and (
                summaries[label].get("eligible_for_comparison") is True
            )
        fingerprints = {
            label: [
                row["scenario_fingerprint"]
                for row in rows
            ]
            for label, rows in rows_by_label.items()
        }
        paired = (
            len(
                {
                    tuple(values)
                    for values in fingerprints.values()
                }
            )
            == 1
        )
        all_paired = all_paired and paired
        entry = {
            "seed": seed,
            "scenario_paired": paired,
            "selected_checkpoint_episode": {
                label: summaries[label][
                    "selected_checkpoint_episode"
                ]
                for label in LABELS
            },
        }
        for label in LABELS:
            rows = rows_by_label[label]
            entry[label] = {
                "mean_finish_time": mean_metric(
                    rows,
                    "average_finish_time",
                ),
                "mean_p95_finish_time": mean_metric(
                    rows,
                    "p95_finish_time",
                ),
                "cache_hit_rate": mean_metric(
                    rows,
                    "cache_hit_rate",
                ),
                "cache_service_coverage": mean_metric(
                    rows,
                    "cache_service_coverage",
                ),
                "remote_loading_rate": mean_metric(
                    rows,
                    "cache_remote_loading_rate",
                ),
                "computing_latency": mean_metric(
                    rows,
                    "computing_latency",
                ),
                "waiting_latency": mean_metric(
                    rows,
                    "waiting_latency",
                ),
                "service_latency": mean_metric(
                    rows,
                    "service_latency",
                ),
                "predecessor_latency": mean_metric(
                    rows,
                    "predecessor_latency",
                ),
                "data_transfer_latency": mean_metric(
                    rows,
                    "data_transfer_latency",
                ),
                "zero_capacity_assignment_rate": mean_metric(
                    rows,
                    "cache_zero_capacity_assignment_rate",
                ),
            }
        per_seed.append(entry)
    return per_seed, all_paired, all_converged


def compare(per_seed, reference, candidate, metric):
    return paired_statistics(
        [row[reference][metric] for row in per_seed],
        [row[candidate][metric] for row in per_seed],
        lower_is_better=True,
    )


def comparisons(per_seed):
    pairs = {
        "our_vs_our_dqn": ("our_dqn", "lean_our"),
        "our_vs_centralized_greedy": (
            "centralized_greedy_daoc",
            "lean_our",
        ),
        "centralized_greedy_vs_daoc": (
            "guided_full",
            "centralized_greedy_daoc",
        ),
        "our_dqn_vs_daoc": ("guided_full", "our_dqn"),
    }
    return {
        name: {
            "finish_time": compare(
                per_seed,
                reference,
                candidate,
                "mean_finish_time",
            ),
            "p95_finish_time": compare(
                per_seed,
                reference,
                candidate,
                "mean_p95_finish_time",
            ),
        }
        for name, (reference, candidate) in pairs.items()
    }


def method_aggregates(per_seed):
    metrics = (
        "mean_finish_time",
        "mean_p95_finish_time",
        "computing_latency",
        "waiting_latency",
        "service_latency",
        "predecessor_latency",
        "data_transfer_latency",
        "cache_hit_rate",
        "cache_service_coverage",
        "remote_loading_rate",
        "zero_capacity_assignment_rate",
    )
    return {
        label: {
            metric: float(
                np.mean(
                    [row[label][metric] for row in per_seed]
                )
            )
            for metric in metrics
        }
        for label in LABELS
    }


def plot_results(path, per_seed):
    means = {
        label: float(
            np.mean(
                [row[label]["mean_finish_time"] for row in per_seed]
            )
        )
        for label in LABELS
    }
    p95 = {
        label: float(
            np.mean(
                [
                    row[label]["mean_p95_finish_time"]
                    for row in per_seed
                ]
            )
        )
        for label in LABELS
    }
    x = np.arange(len(LABELS))
    colors = ("#59636E", "#3A7D78", "#D89B45", "#D65F45")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.5),
        constrained_layout=True,
    )
    axes[0].bar(
        x,
        [means[label] for label in LABELS],
        color=colors,
    )
    axes[1].bar(
        x,
        [p95[label] for label in LABELS],
        color=colors,
    )
    for axis, title in zip(
        axes,
        ("Mean completion time", "Mean episode P95"),
    ):
        axis.set_xticks(
            x,
            [DISPLAY_NAMES[label] for label in LABELS],
            rotation=15,
            ha="right",
        )
        axis.set_ylabel("DAG completion time (s)")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path, summary):
    dqn = summary["comparisons"]["our_vs_our_dqn"]
    central = summary["comparisons"][
        "centralized_greedy_vs_daoc"
    ]
    architecture = summary["comparisons"][
        "our_vs_centralized_greedy"
    ]
    lines = [
        "# Alibaba-CP100 三Seed算法扩展实验",
        "",
        f"- Seeds：`{summary['seeds']}`，每个方法每 seed "
        "`100` 个冻结配对场景。",
        f"- 四种方法场景完全配对："
        f"`{summary['all_methods_scenario_paired']}`。",
        f"- 四种学习方法均满足收敛比较资格："
        f"`{summary['all_methods_converged']}`。",
        "",
        "## 关键比较",
        "",
        f"- OUR 相对 OUR-DQN 的平均时延改善："
        f"`{dqn['finish_time']['mean_paired_improvement_percent']:.3f}%`，"
        f"胜场 `{dqn['finish_time']['wins']}/3`。",
        f"- 中央流行度贪心相对 DAOC 的平均时延改善："
        f"`{central['finish_time']['mean_paired_improvement_percent']:.3f}%`，"
        f"胜场 `{central['finish_time']['wins']}/3`。",
        f"- OUR 相对同样具有中央协调器的贪心基线改善："
        f"`{architecture['finish_time']['mean_paired_improvement_percent']:.3f}%`，"
        f"胜场 `{architecture['finish_time']['wins']}/3`。",
        "",
        "## 机制分量",
        "",
        "| 方法 | 缓存命中率 | 服务覆盖率 | 服务加载时延 | "
        "计算时延 | 等待时延 | 前驱时延 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        metrics = summary["method_aggregates"][label]
        lines.append(
            f"| {DISPLAY_NAMES[label]} | "
            f"{metrics['cache_hit_rate']:.4f} | "
            f"{metrics['cache_service_coverage']:.4f} | "
            f"{metrics['service_latency']:.4f} | "
            f"{metrics['computing_latency']:.4f} | "
            f"{metrics['waiting_latency']:.4f} | "
            f"{metrics['predecessor_latency']:.4f} |"
        )
    lines.extend(
        [
        "",
        "三 seed 结果用于机制拆分，不单独作正式显著性结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    source_suite = args.source_suite_dir.resolve()
    extension_suite = args.extension_suite_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, paired, converged = collect(
        source_suite,
        extension_suite,
        args.seeds,
    )
    summary = {
        "status": "complete",
        "seeds": args.seeds,
        "labels": list(LABELS),
        "all_methods_scenario_paired": paired,
        "all_methods_converged": converged,
        "per_seed": per_seed,
        "comparisons": comparisons(per_seed),
        "method_aggregates": method_aggregates(per_seed),
    }
    write_json(output_dir / "algorithm_extensions_summary.json", summary)
    plot_results(output_dir / "algorithm_extensions", per_seed)
    write_report(
        output_dir / "ALGORITHM_EXTENSIONS_REPORT.md",
        summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
