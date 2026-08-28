#!/usr/bin/env python3
"""Analyze the governed static cache-heterogeneity experiments."""

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

from analyze_e2_e3_results import (
    collect_pair,
    evaluation_rows,
    mean_metric,
    metric_comparisons,
    plot_cache_heatmaps,
    read_json,
    run_dir,
)
from analyze_strict_environment_suite import paired_statistics
from capacity_protocol import deterministic_capacity_assignment
from static_heterogeneity_protocol import (
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    CAPACITY_VARIANCES,
    MAIN_PROFILE,
    STATIC_HETEROGENEITY_PROTOCOL_VERSION,
    TOTAL_CACHE_BUDGET,
)


DAOC_LABEL = "guided_full"
OUR_LABEL = "lean_our"
ABLATION_LABELS = (
    "our_no_telemetry",
    "our_no_coord_cache",
)
PRIMARY_METRICS = (
    "average_finish_time",
    "p95_finish_time",
    "cache_hit_rate",
    "cache_remote_loading_rate",
    "cache_zero_capacity_assignment_rate",
    "cache_capacity_utilization",
    "cache_service_coverage",
    "cache_migration_time_sec",
    "cache_migration_events",
)


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze static heterogeneous-cache experiments."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "screen",
            "converged",
            "sweep",
            "ablation",
            "final",
        ),
        required=True,
    )
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    parser.add_argument(
        "--profile",
        choices=CAPACITY_PROFILES,
        default=MAIN_PROFILE,
    )
    parser.add_argument("--daoc-label", default=DAOC_LABEL)
    parser.add_argument("--our-label", default=OUR_LABEL)
    return parser.parse_args()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def capacity_assignment_check(pair, profile, seed):
    expected = deterministic_capacity_assignment(
        CAPACITY_PROFILES[profile],
        number_of_servers=10,
        number_of_services=10,
        seed=seed,
        assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
    )
    return pair["capacities"] == expected


def collect_static_pairs(
    suite_dir,
    seeds,
    profile,
    daoc_label,
    our_label,
):
    pairs = []
    for seed in seeds:
        pair = collect_pair(
            suite_dir=suite_dir,
            seed=seed,
            daoc_label=daoc_label,
            our_label=our_label,
            expected_capacities=CAPACITY_PROFILES[profile],
        )
        pair["suite_dir"] = suite_dir
        pair["our_label"] = our_label
        pair["protocol_checks"][
            "static_capacity_assignment_exact"
        ] = capacity_assignment_check(pair, profile, seed)
        pair["protocol_checks"][
            "static_total_budget_exact"
        ] = sum(pair["capacities"].values()) == TOTAL_CACHE_BUDGET
        pairs.append(pair)
    return pairs


def pair_integrity(pairs):
    return all(
        value
        for pair in pairs
        for value in pair["protocol_checks"].values()
    )


def convergence_status(pairs):
    return all(
        pair["metrics"]["daoc_converged"]
        and pair["metrics"]["our_converged"]
        for pair in pairs
    )


def stage_gate(mode, pairs, comparisons):
    integrity = pair_integrity(pairs)
    converged = convergence_status(pairs)
    finish = comparisons["finish_time"]
    p95 = comparisons["p95_finish_time"]
    if mode == "smoke":
        return {
            "passed": integrity,
            "definition": "integrity_only",
            "required_wins": 0,
        }
    if mode == "screen":
        passed = (
            integrity
            and finish["mean_paired_improvement_percent"] > 0
            and finish["wins"] >= 2
        )
        return {
            "passed": passed,
            "definition": (
                "positive_mean_finish_improvement_and_2_of_3_wins"
            ),
            "required_wins": 2,
        }
    if mode == "converged":
        passed = (
            integrity
            and converged
            and finish["wins"] == len(pairs)
            and finish["mean_paired_improvement_percent"] >= 5.0
            and p95["candidate_mean"] < p95["reference_mean"]
        )
        return {
            "passed": passed,
            "definition": (
                "all_converged_all_seed_wins_mean_finish_gain_at_"
                "least_5pct_and_better_mean_p95"
            ),
            "required_wins": len(pairs),
        }
    if mode == "final":
        passed = (
            integrity
            and converged
            and finish["wins"] >= 7
            and finish["paired_improvement_ci95_lower"] > 0
            and finish["wilcoxon_one_sided_p"] < 0.05
        )
        return {
            "passed": passed,
            "definition": (
                "all_converged_ci95_lower_positive_one_sided_"
                "wilcoxon_below_0.05_and_7_of_10_wins"
            ),
            "required_wins": 7,
        }
    raise ValueError(f"No paired gate for mode {mode}")


def plot_main_metrics(path, comparisons):
    names = ("finish_time", "p95_finish_time")
    labels = ("Mean DAG completion", "P95 DAG completion")
    daoc = [comparisons[name]["reference_mean"] for name in names]
    ours = [comparisons[name]["candidate_mean"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(
        figsize=(7.2, 4.4),
        constrained_layout=True,
    )
    axis.bar(
        x - width / 2,
        daoc,
        width,
        label="DAOC",
        color="#566573",
    )
    axis.bar(
        x + width / 2,
        ours,
        width,
        label="OUR",
        color="#D95F45",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Time (s)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def basic_diagnosis(gate, comparisons, all_converged):
    if not all_converged:
        return {
            "status": "failed",
            "reason": "learning_method_not_converged",
            "next_action": (
                "inspect validation curves before changing the method"
            ),
        }
    finish = comparisons["finish_time"]
    p95 = comparisons["p95_finish_time"]
    if finish["mean_paired_improvement_percent"] <= 0:
        reason = "mean_finish_time_not_improved"
    elif p95["candidate_mean"] >= p95["reference_mean"]:
        reason = "p95_not_improved"
    elif not gate["passed"]:
        reason = "seed_consistency_or_effect_size_gate_failed"
    else:
        reason = "none"
    return {
        "status": "passed" if gate["passed"] else "failed",
        "reason": reason,
        "next_action": (
            "continue_to_the_next_frozen_stage"
            if gate["passed"]
            else "inspect_action_cache_and_oracle_headroom_metrics"
        ),
    }


def write_report(path, summary):
    finish = summary["comparisons"]["finish_time"]
    p95 = summary["comparisons"]["p95_finish_time"]
    lines = [
        "# 静态异构缓存实验报告",
        "",
        f"- 阶段：`{summary['mode']}`",
        f"- 容量环境：`{summary['profile']}` "
        f"`{summary['capacity_multiset']}`",
        f"- Seeds：`{summary['seeds']}`",
        f"- 完整性审计：`{summary['integrity_passed']}`",
        f"- 收敛审计：`{summary['all_learning_methods_converged']}`",
        f"- 阶段门槛：`{summary['gate']['passed']}`",
        "",
        "## 主要结果",
        "",
        f"- OUR 平均完成时间改善："
        f"`{finish['mean_paired_improvement_percent']:.3f}%`，"
        f"seed 胜场 `{finish['wins']}/{finish['pairs']}`。",
        f"- OUR P95 改善："
        f"`{p95['mean_paired_improvement_percent']:.3f}%`。",
        f"- 诊断：`{summary['diagnosis']['reason']}`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_paired(args):
    suite_dir = args.suite_dir.resolve()
    pairs = collect_static_pairs(
        suite_dir,
        args.seeds,
        args.profile,
        args.daoc_label,
        args.our_label,
    )
    per_seed = [pair["metrics"] for pair in pairs]
    comparisons = metric_comparisons(per_seed)
    gate = stage_gate(args.mode, pairs, comparisons)
    all_converged = convergence_status(pairs)
    summary = {
        "status": "complete",
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "mode": args.mode,
        "profile": args.profile,
        "capacity_multiset": CAPACITY_PROFILES[args.profile],
        "capacity_variance": CAPACITY_VARIANCES[args.profile],
        "seeds": args.seeds,
        "integrity_passed": pair_integrity(pairs),
        "all_learning_methods_converged": all_converged,
        "per_seed": per_seed,
        "comparisons": comparisons,
        "gate": gate,
    }
    if args.mode == "final":
        finish = comparisons["finish_time"]
        summary["formal_superiority"] = {
            "untouched_seed_partition": True,
            "ci95_lower_positive": (
                finish["paired_improvement_ci95_lower"] > 0
            ),
            "wilcoxon_one_sided_p_below_0_05": (
                finish["wilcoxon_one_sided_p"] < 0.05
            ),
            "at_least_7_of_10_seed_wins": (
                finish["wins"] >= 7
            ),
            "passed": gate["passed"],
        }
    summary["diagnosis"] = basic_diagnosis(
        gate,
        comparisons,
        all_converged,
    )
    write_json(suite_dir / "static_analysis.json", summary)
    write_csv(suite_dir / "static_per_seed.csv", per_seed)
    plot_main_metrics(
        suite_dir / "static_main_result",
        comparisons,
    )
    plot_cache_heatmaps(
        suite_dir / "static_cache_heatmap",
        pairs,
    )
    write_report(
        suite_dir / "STATIC_DIAGNOSTIC_REPORT_ZH.md",
        summary,
    )
    print(json.dumps(summary, indent=2))
    return summary


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean
    half = float(
        stats.t.ppf(0.975, values.size - 1)
        * stats.sem(values)
    )
    return mean - half, mean + half


def analyze_sweep(args):
    root = args.suite_dir.resolve()
    by_profile = {}
    per_seed_profile = []
    base_fingerprints = {}
    for profile in CAPACITY_PROFILES:
        suite = root / profile
        pairs = collect_static_pairs(
            suite,
            args.seeds,
            profile,
            args.daoc_label,
            args.our_label,
        )
        metrics = [pair["metrics"] for pair in pairs]
        comparisons = metric_comparisons(metrics)
        finish = comparisons["finish_time"]
        by_profile[profile] = {
            "capacity_multiset": CAPACITY_PROFILES[profile],
            "variance": CAPACITY_VARIANCES[profile],
            "integrity_passed": pair_integrity(pairs),
            "all_learning_methods_converged": (
                convergence_status(pairs)
            ),
            "finish_time": finish,
            "p95_finish_time": comparisons["p95_finish_time"],
            "remote_loading_rate": comparisons[
                "remote_loading_rate"
            ],
            "service_coverage": comparisons[
                "service_coverage"
            ],
        }
        for pair in pairs:
            seed = pair["metrics"]["seed"]
            gain = (
                100.0
                * (
                    pair["metrics"]["daoc_average_finish_time"]
                    - pair["metrics"]["our_average_finish_time"]
                )
                / pair["metrics"]["daoc_average_finish_time"]
            )
            per_seed_profile.append(
                {
                    "seed": seed,
                    "profile": profile,
                    "variance": CAPACITY_VARIANCES[profile],
                    "finish_gain_percent": gain,
                }
            )
            base_fingerprints[(profile, seed)] = [
                row["base_scenario_fingerprint"]
                for row in pair["rows"]["our"]
            ]

    paired_across_profiles = all(
        base_fingerprints[(profile, seed)]
        == base_fingerprints[(MAIN_PROFILE, seed)]
        for profile in CAPACITY_PROFILES
        for seed in args.seeds
    )
    slopes = []
    for seed in args.seeds:
        rows = [
            row
            for row in per_seed_profile
            if row["seed"] == seed
        ]
        slopes.append(
            float(
                np.polyfit(
                    [row["variance"] for row in rows],
                    [row["finish_gain_percent"] for row in rows],
                    1,
                )[0]
            )
        )
    slope_interval = confidence_interval(slopes)
    all_integrity = (
        paired_across_profiles
        and all(
            profile["integrity_passed"]
            for profile in by_profile.values()
        )
    )
    all_converged = all(
        profile["all_learning_methods_converged"]
        for profile in by_profile.values()
    )
    h3_wins = by_profile["H3"]["finish_time"]["wins"]
    h4_wins = by_profile["H4"]["finish_time"]["wins"]
    gate = {
        "passed": (
            all_integrity
            and all_converged
            and float(np.mean(slopes)) > 0
            and h3_wins >= math.ceil(len(args.seeds) / 2)
            and h4_wins >= math.ceil(len(args.seeds) / 2)
        ),
        "definition": (
            "all_profiles_paired_and_converged_positive_mean_seed_"
            "slope_and_majority_wins_in_H3_H4"
        ),
        "h3_wins": h3_wins,
        "h4_wins": h4_wins,
    }
    summary = {
        "status": "complete",
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "mode": "sweep",
        "seeds": args.seeds,
        "profiles": by_profile,
        "base_scenarios_paired_across_profiles": (
            paired_across_profiles
        ),
        "integrity_passed": all_integrity,
        "all_learning_methods_converged": all_converged,
        "per_seed_gain_slopes": slopes,
        "mean_gain_slope_per_variance": float(np.mean(slopes)),
        "gain_slope_ci95_lower": slope_interval[0],
        "gain_slope_ci95_upper": slope_interval[1],
        "gate": gate,
    }
    write_json(root / "heterogeneity_analysis.json", summary)
    write_csv(
        root / "heterogeneity_per_seed_profile.csv",
        per_seed_profile,
    )
    plot_sweep(root / "heterogeneity_curve", by_profile)
    write_sweep_report(
        root / "HETEROGENEITY_REPORT_ZH.md",
        summary,
    )
    print(json.dumps(summary, indent=2))
    return summary


def plot_sweep(path, by_profile):
    ordered = list(CAPACITY_PROFILES)
    x = [CAPACITY_VARIANCES[name] for name in ordered]
    daoc = [
        by_profile[name]["finish_time"]["reference_mean"]
        for name in ordered
    ]
    ours = [
        by_profile[name]["finish_time"]["candidate_mean"]
        for name in ordered
    ]
    gains = [
        by_profile[name]["finish_time"][
            "mean_paired_improvement_percent"
        ]
        for name in ordered
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.3),
        constrained_layout=True,
    )
    axes[0].plot(
        x,
        daoc,
        marker="o",
        label="DAOC",
        color="#566573",
    )
    axes[0].plot(
        x,
        ours,
        marker="s",
        label="OUR",
        color="#D95F45",
    )
    axes[0].set_ylabel("Mean DAG completion time (s)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        x,
        gains,
        marker="o",
        color="#2471A3",
    )
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_ylabel("OUR improvement over DAOC (%)")
    for axis in axes:
        axis.set_xlabel("Server cache-capacity variance")
        axis.grid(alpha=0.25)
        axis.set_xticks(x, ordered)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_sweep_report(path, summary):
    lines = [
        "# 静态缓存异构性曲线报告",
        "",
        f"- Seeds：`{summary['seeds']}`",
        f"- 跨环境物理场景严格配对："
        f"`{summary['base_scenarios_paired_across_profiles']}`",
        f"- 全部模型收敛："
        f"`{summary['all_learning_methods_converged']}`",
        f"- 每单位容量方差的平均增益斜率："
        f"`{summary['mean_gain_slope_per_variance']:.4f}`。",
        f"- 故事门槛：`{summary['gate']['passed']}`。",
        "",
        "## 各环境",
        "",
    ]
    for name, result in summary["profiles"].items():
        finish = result["finish_time"]
        lines.append(
            f"- {name}（方差 {result['variance']:.1f}）："
            f"完成时间改善 "
            f"`{finish['mean_paired_improvement_percent']:.3f}%`，"
            f"胜场 `{finish['wins']}/{finish['pairs']}`。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_ablation(args):
    suite = args.suite_dir.resolve()
    labels = (args.our_label,) + ABLATION_LABELS
    rows_by_label = {
        label: {
            seed: evaluation_rows(run_dir(suite, label, seed))
            for seed in args.seeds
        }
        for label in labels
    }
    summaries = {
        label: {
            seed: read_json(
                run_dir(suite, label, seed) / "summary.json"
            )
            for seed in args.seeds
        }
        for label in labels
    }
    fingerprints_match = all(
        [
            row["scenario_fingerprint"]
            for row in rows_by_label[label][seed]
        ]
        == [
            row["scenario_fingerprint"]
            for row in rows_by_label[args.our_label][seed]
        ]
        for label in ABLATION_LABELS
        for seed in args.seeds
    )
    per_seed = []
    for seed in args.seeds:
        row = {"seed": seed}
        for label in labels:
            for metric in PRIMARY_METRICS:
                row[f"{label}_{metric}"] = mean_metric(
                    rows_by_label[label][seed],
                    metric,
                )
        per_seed.append(row)
    comparisons = {}
    for label in ABLATION_LABELS:
        comparisons[label] = {}
        for metric, lower in (
            ("average_finish_time", True),
            ("p95_finish_time", True),
            ("cache_remote_loading_rate", True),
            ("cache_service_coverage", False),
        ):
            # Candidate is full OUR, so positive improvement means the
            # removed module mattered.
            comparisons[label][metric] = paired_statistics(
                [row[f"{label}_{metric}"] for row in per_seed],
                [
                    row[f"{args.our_label}_{metric}"]
                    for row in per_seed
                ],
                lower_is_better=lower,
            )
    all_converged = all(
        summaries[label][seed]["eligible_for_comparison"]
        for label in labels
        for seed in args.seeds
    )
    summary = {
        "status": "complete",
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "mode": "ablation",
        "profile": args.profile,
        "seeds": args.seeds,
        "scenario_fingerprints_match": fingerprints_match,
        "all_learning_methods_converged": all_converged,
        "per_seed": per_seed,
        "comparisons_full_our_vs_ablation": comparisons,
        "gate": {
            "passed": fingerprints_match and all_converged,
            "definition": (
                "integrity_and_convergence_only_ablation_not_used_for_"
                "model_selection"
            ),
        },
    }
    write_json(suite / "static_ablation_analysis.json", summary)
    write_csv(suite / "static_ablation_per_seed.csv", per_seed)
    plot_ablation(suite / "static_ablation", per_seed, labels)
    print(json.dumps(summary, indent=2))
    return summary


def plot_ablation(path, rows, labels):
    means = [
        float(
            np.mean(
                [
                    row[f"{label}_average_finish_time"]
                    for row in rows
                ]
            )
        )
        for label in labels
    ]
    names = ("OUR", "No telemetry", "No coordinated cache")
    figure, axis = plt.subplots(
        figsize=(7.2, 4.3),
        constrained_layout=True,
    )
    axis.bar(
        names,
        means,
        color=("#D95F45", "#7F8C8D", "#2471A3"),
    )
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    args = parse_args()
    if args.mode == "sweep":
        analyze_sweep(args)
    elif args.mode == "ablation":
        analyze_ablation(args)
    else:
        analyze_paired(args)


if __name__ == "__main__":
    main()
