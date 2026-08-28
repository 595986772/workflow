#!/usr/bin/env python3
"""Analyze paired DAOC/OUR runs on the Alibaba-CP100 stress set."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alibaba_cp100_protocol import (
    ALIBABA_CP100_PROTOCOL_VERSION,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_MULTISET,
    EXPECTED_DATASET_SHA256,
    EXPECTED_GRAPH_COUNT,
    TOTAL_CACHE_BUDGET,
    validate_protocol,
)
from analyze_e2_e3_results import (
    collect_pair,
    metric_comparisons,
    plot_cache_heatmaps,
    read_json,
    run_dir,
)
from capacity_protocol import deterministic_capacity_assignment


DAOC_LABEL = "guided_full"
OUR_LABEL = "lean_our"


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the Alibaba-CP100 budget-20 experiment."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "converged"), required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def collect_dataset_pairs(suite_dir, seeds):
    pairs = []
    unique_graph_keys = set()
    for seed in seeds:
        pair = collect_pair(
            suite_dir=suite_dir,
            seed=seed,
            daoc_label=DAOC_LABEL,
            our_label=OUR_LABEL,
            expected_capacities=CAPACITY_MULTISET,
        )
        expected_assignment = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
        )
        pair["protocol_checks"]["capacity_assignment_exact"] = (
            pair["capacities"] == expected_assignment
        )
        pair["protocol_checks"]["total_budget_exact"] = (
            sum(pair["capacities"].values()) == TOTAL_CACHE_BUDGET
        )
        for method in ("daoc", "our"):
            summary = pair["summaries"][method]
            config = pair["configs"][method]
            pair["protocol_checks"][
                f"{method}_dataset_hash"
            ] = (
                summary.get("dag_dataset", {}).get("sha256")
                == EXPECTED_DATASET_SHA256
                and config["arguments"].get("dag_dataset_sha256")
                == EXPECTED_DATASET_SHA256
            )
            pair["protocol_checks"][
                f"{method}_dataset_graph_count"
            ] = (
                summary.get("dag_dataset", {}).get("graph_count")
                == EXPECTED_GRAPH_COUNT
                and summary.get("dag_dataset", {}).get(
                    "eligible_graph_count"
                )
                == EXPECTED_GRAPH_COUNT
            )
        scenario_files = {
            method: read_json(
                run_dir(suite_dir, label, seed)
                / "evaluation_scenarios.json"
            )
            for method, label in (
                ("daoc", DAOC_LABEL),
                ("our", OUR_LABEL),
            )
        }
        pair["protocol_checks"]["scenario_metadata_paired"] = (
            scenario_files["daoc"] == scenario_files["our"]
        )
        for scenario in scenario_files["our"]:
            unique_graph_keys.update(
                scenario["user_graph_keys"].values()
            )
        pairs.append(pair)
    return pairs, sorted(unique_graph_keys)


def plot_results(path, per_seed, comparisons):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
        constrained_layout=True,
    )
    labels = ("Mean", "P95")
    daoc = [
        comparisons["finish_time"]["reference_mean"],
        comparisons["p95_finish_time"]["reference_mean"],
    ]
    ours = [
        comparisons["finish_time"]["candidate_mean"],
        comparisons["p95_finish_time"]["candidate_mean"],
    ]
    x = np.arange(2)
    width = 0.36
    axes[0].bar(
        x - width / 2,
        daoc,
        width,
        label="DAOC",
        color="#59636E",
    )
    axes[0].bar(
        x + width / 2,
        ours,
        width,
        label="OUR",
        color="#D65F45",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("DAG completion time (s)")
    axes[0].legend(frameon=False)
    seeds = [row["seed"] for row in per_seed]
    gains = [
        100.0
        * (
            row["daoc_average_finish_time"]
            - row["our_average_finish_time"]
        )
        / row["daoc_average_finish_time"]
        for row in per_seed
    ]
    axes[1].bar(
        [str(seed) for seed in seeds],
        gains,
        color="#277DA1",
    )
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("OUR improvement over DAOC (%)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def plot_convergence(path, suite_dir, seeds):
    figure, axis = plt.subplots(
        figsize=(7.6, 4.5),
        constrained_layout=True,
    )
    methods = (
        (DAOC_LABEL, "DAOC", "#59636E"),
        (OUR_LABEL, "OUR", "#D65F45"),
    )
    for method, display_name, color in methods:
        series = []
        for seed in seeds:
            checkpoint_data = read_json(
                run_dir(suite_dir, method, seed)
                / "checkpoint_validation.json"
            )
            series.append(
                {
                    int(record["episode"]): float(
                        record["mean_average_finish_time"]
                    )
                    for record in checkpoint_data["records"]
                }
            )
        common_episodes = sorted(
            set.intersection(
                *(set(seed_series) for seed_series in series)
            )
        )
        values = np.asarray(
            [
                [seed_series[episode] for episode in common_episodes]
                for seed_series in series
            ],
            dtype=float,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        axis.plot(
            common_episodes,
            mean,
            linewidth=2.2,
            label=display_name,
            color=color,
        )
        axis.fill_between(
            common_episodes,
            mean - std,
            mean + std,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.scatter(
            [common_episodes[-1]],
            [mean[-1]],
            color=color,
            s=28,
            zorder=3,
        )
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Validation DAG completion time (s)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def percent_reduction(reference, candidate):
    if reference == 0:
        return 0.0 if candidate == 0 else -100.0
    return 100.0 * (reference - candidate) / reference


def mechanism_diagnosis(pairs, comparisons):
    row_metrics = (
        "service_latency",
        "waiting_latency",
        "computing_latency",
        "data_transfer_latency",
        "predecessor_latency",
    )
    means = {}
    overhead = {}
    for method in ("daoc", "our"):
        rows = [
            row
            for pair in pairs
            for row in pair["rows"][method]
        ]
        means[method] = {
            metric: float(
                np.mean([float(row[metric]) for row in rows])
            )
            for metric in row_metrics
        }
        total_wall = sum(
            float(row["episode_wall_time_sec"]) for row in rows
        )
        total_decisions = sum(
            int(row["decision_count"]) for row in rows
        )
        overhead[method] = {
            "total_eval_wall_time_sec": total_wall,
            "total_decisions": total_decisions,
            "milliseconds_per_decision": (
                1000.0 * total_wall / total_decisions
            ),
        }

    finish = comparisons["finish_time"]
    service = percent_reduction(
        means["daoc"]["service_latency"],
        means["our"]["service_latency"],
    )
    predecessor = percent_reduction(
        means["daoc"]["predecessor_latency"],
        means["our"]["predecessor_latency"],
    )
    return {
        "primary_result": {
            "mean_finish_time_improvement_percent": (
                finish["mean_paired_improvement_percent"]
            ),
            "seed_wins": finish["wins"],
            "seed_pairs": finish["pairs"],
        },
        "latency_component_means": means,
        "service_latency_reduction_percent": service,
        "predecessor_latency_reduction_percent": predecessor,
        "cache_mechanism": {
            "hit_rate_absolute_gain": (
                comparisons["cache_hit_rate"]["candidate_mean"]
                - comparisons["cache_hit_rate"]["reference_mean"]
            ),
            "coverage_absolute_gain": (
                comparisons["service_coverage"]["candidate_mean"]
                - comparisons["service_coverage"]["reference_mean"]
            ),
            "remote_loading_reduction_percent": (
                comparisons["remote_loading_rate"][
                    "ratio_of_means_improvement_percent"
                ]
            ),
            "our_zero_capacity_assignment_rate": (
                comparisons["zero_capacity_assignment_rate"][
                    "candidate_mean"
                ]
            ),
        },
        "execution_overhead": {
            **overhead,
            "our_extra_milliseconds_per_decision": (
                overhead["our"]["milliseconds_per_decision"]
                - overhead["daoc"]["milliseconds_per_decision"]
            ),
        },
        "interpretation": (
            "The dominant measured mechanism is broader coordinated "
            "service coverage and fewer remote service loads; compute "
            "and radio-transfer metrics remain nearly unchanged."
        ),
    }


def write_report(path, summary):
    finish = summary["comparisons"]["finish_time"]
    p95 = summary["comparisons"]["p95_finish_time"]
    mechanism = summary["mechanism_diagnosis"]
    cache = mechanism["cache_mechanism"]
    inference = summary["statistical_scope"]
    lines = [
        "# Alibaba-CP100 总缓存预算20实验报告",
        "",
        "- 定位：机制压力测试，不是无偏 Alibaba holdout。",
        f"- 数据集 SHA-256：`{EXPECTED_DATASET_SHA256}`",
        f"- 容量向量：`{CAPACITY_MULTISET}`，总预算 "
        f"`{TOTAL_CACHE_BUDGET}`。",
        f"- Seeds：`{summary['seeds']}`。",
        f"- 配对与公平性审计：`{summary['integrity_passed']}`。",
        f"- 双方收敛：`{summary['all_learning_methods_converged']}`。",
        "",
        "## 结果",
        "",
        f"- OUR 平均完成时间改善："
        f"`{finish['mean_paired_improvement_percent']:.3f}%`，"
        f"胜场 `{finish['wins']}/{finish['pairs']}`。",
        f"- OUR P95 改善："
        f"`{p95['mean_paired_improvement_percent']:.3f}%`。",
        f"- 评估实际覆盖 `{summary['unique_dataset_dags_used']}/"
        f"{EXPECTED_GRAPH_COUNT}` 个 DAG。",
        f"- 预注册机制门槛：`{summary['gate']['passed']}`。",
        "",
        "## 机制诊断",
        "",
        f"- 服务加载时延指标降低："
        f"`{mechanism['service_latency_reduction_percent']:.3f}%`。",
        f"- 服务覆盖率绝对提高："
        f"`{100.0 * cache['coverage_absolute_gain']:.2f}` 个百分点；"
        f"缓存命中率提高 "
        f"`{100.0 * cache['hit_rate_absolute_gain']:.2f}` 个百分点。",
        f"- 远程服务加载率降低："
        f"`{cache['remote_loading_reduction_percent']:.3f}%`。",
        f"- OUR 额外决策计算开销："
        f"`{mechanism['execution_overhead']['our_extra_milliseconds_per_decision']:.3f}`"
        " ms/decision。",
        "",
        "## 证据边界",
        "",
        f"- seed 级 95% CI："
        f"`[{finish['paired_improvement_ci95_lower']:.3f}%, "
        f"{finish['paired_improvement_ci95_upper']:.3f}%]`。",
        f"- 单侧 Wilcoxon `p={finish['wilcoxon_one_sided_p']:.3f}`；"
        f"正式显著性结论：`{inference['formal_superiority_supported']}`。",
        f"- 原因：{inference['reason_zh']}",
        "",
        "该结果只能说明在刻意强化依赖局部性、关键路径和服务关联的"
        "压力集上，算法机制是否按预期工作；不能单独支持真实 Alibaba "
        "工作负载上的一般优越性结论。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    protocol = validate_protocol()
    suite_dir = args.suite_dir.resolve()
    pairs, unique_graph_keys = collect_dataset_pairs(
        suite_dir,
        args.seeds,
    )
    per_seed = [pair["metrics"] for pair in pairs]
    comparisons = metric_comparisons(per_seed)
    integrity = all(
        value
        for pair in pairs
        for value in pair["protocol_checks"].values()
    )
    all_converged = all(
        row["daoc_converged"] and row["our_converged"]
        for row in per_seed
    )
    finish = comparisons["finish_time"]
    p95 = comparisons["p95_finish_time"]
    if args.mode == "smoke":
        gate = {
            "passed": integrity,
            "definition": "implementation_and_pairing_integrity_only",
        }
    else:
        gate = {
            "passed": (
                integrity
                and all_converged
                and finish["mean_paired_improvement_percent"] > 0
                and finish["wins"] >= 2
                and p95["candidate_mean"] < p95["reference_mean"]
            ),
            "definition": (
                "all_converged_positive_mean_gain_at_least_2_of_3_"
                "wins_and_lower_mean_p95"
            ),
        }
    summary = {
        "status": "complete",
        "protocol_version": ALIBABA_CP100_PROTOCOL_VERSION,
        "mode": args.mode,
        "claim_scope": "mechanism_stress_test_only",
        "protocol": protocol,
        "seeds": args.seeds,
        "integrity_passed": integrity,
        "all_learning_methods_converged": all_converged,
        "unique_dataset_dags_used": len(unique_graph_keys),
        "dataset_dags_used": unique_graph_keys,
        "per_seed": per_seed,
        "comparisons": comparisons,
        "protocol_checks": [
            {
                "seed": pair["metrics"]["seed"],
                **pair["protocol_checks"],
            }
            for pair in pairs
        ],
        "gate": gate,
    }
    summary["mechanism_diagnosis"] = mechanism_diagnosis(
        pairs,
        comparisons,
    )
    summary["statistical_scope"] = {
        "seed_level_sample_size": len(args.seeds),
        "paired_ci95_excludes_zero": (
            finish["paired_improvement_ci95_lower"] > 0
        ),
        "wilcoxon_one_sided_p": finish["wilcoxon_one_sided_p"],
        "formal_superiority_supported": (
            len(args.seeds) >= 10
            and finish["paired_improvement_ci95_lower"] > 0
            and finish["wilcoxon_one_sided_p"] < 0.05
        ),
        "reason_zh": (
            "当前只有3个独立训练seed，Wilcoxon单侧检验可达到的"
            "最小p值为0.125；应将其作为机制验证，而不是新的正式"
            "显著性主结论。"
        ),
    }
    write_json(suite_dir / "alibaba_cp100_analysis.json", summary)
    with (suite_dir / "alibaba_cp100_per_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    plot_results(
        suite_dir / "alibaba_cp100_main_result",
        per_seed,
        comparisons,
    )
    plot_convergence(
        suite_dir / "alibaba_cp100_convergence",
        suite_dir,
        args.seeds,
    )
    plot_cache_heatmaps(
        suite_dir / "alibaba_cp100_cache_heatmap",
        pairs,
    )
    write_report(
        suite_dir / "ALIBABA_CP100_REPORT_ZH.md",
        summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
