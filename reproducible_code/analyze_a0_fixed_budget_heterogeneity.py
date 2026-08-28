#!/usr/bin/env python3
"""Analyze the A0 fixed-budget cache-heterogeneity experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a0_fixed_budget_heterogeneity_protocol import (
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    CAPACITY_VARIANCES,
    DEVELOPMENT_SEEDS,
    METHOD_LABELS,
    PROFILE_NAMES,
    PROFILE_ORDER,
    TOTAL_CACHE_BUDGET,
)
from a0_coordination_protocol import EXPECTED_DATASET_SHA256
from analyze_a0_coordination import (
    DISPLAY_NAMES,
    aggregate_run,
    evaluation_rows,
    oracle_per_seed,
    paired_superiority,
)
from capacity_protocol import deterministic_capacity_assignment


COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_dir(root, profile, label, seed):
    return root / profile / "runs" / label / f"seed_{seed}"


def detect_our_revision(root):
    revisions = {
        read_json(
            run_dir(root, profile, "lean_our", seed) / "summary.json"
        ).get("revision", {}).get("id")
        for profile in PROFILE_ORDER
        for seed in DEVELOPMENT_SEEDS
    }
    if len(revisions) != 1 or None in revisions:
        raise RuntimeError(f"Inconsistent OUR revisions: {revisions}")
    return revisions.pop()


def workload_scenario_view(bank):
    """Return only fields that must remain fixed across capacities."""
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def collect_profile(root, profile, banks_by_seed):
    expected_multiset = CAPACITY_PROFILES[profile]
    per_seed = []
    integrity = {
        "dataset_hash": True,
        "all_methods_converged": True,
        "scenario_banks_paired_within_profile": True,
        "capacity_assignments_exact": True,
        "total_budget_exact": True,
        "networks_frozen_in_evaluation": True,
    }
    for seed in DEVELOPMENT_SEEDS:
        methods = {}
        scenario_banks = []
        expected_capacity = deterministic_capacity_assignment(
            expected_multiset,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
        )
        for label in METHOD_LABELS:
            directory = run_dir(root, profile, label, seed)
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            rows = evaluation_rows(directory / "episodes.csv")
            methods[label] = aggregate_run(rows)
            integrity["dataset_hash"] &= (
                summary.get("dag_dataset", {}).get("sha256")
                == EXPECTED_DATASET_SHA256
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
            integrity["total_budget_exact"] &= (
                sum(capacities.values()) == TOTAL_CACHE_BUDGET
            )
            integrity["networks_frozen_in_evaluation"] &= bool(
                summary.get("evaluation_state_frozen")
            )
            if config["arguments"].get("bandwidth") != 15000:
                raise RuntimeError(f"Wrong bandwidth in {directory}")
            bank = read_json(directory / "evaluation_scenarios.json")
            scenario_banks.append(bank)
            banks_by_seed.setdefault(seed, []).append(
                workload_scenario_view(bank)
            )
        integrity["scenario_banks_paired_within_profile"] &= all(
            bank == scenario_banks[0] for bank in scenario_banks[1:]
        )
        per_seed.append(
            {
                "seed": seed,
                "methods": methods,
                "selected_checkpoint_episode": {
                    label: read_json(
                        run_dir(root, profile, label, seed) / "summary.json"
                    )["selected_checkpoint_episode"]
                    for label in METHOD_LABELS
                },
            }
        )
    aggregates = {
        label: {
            metric: float(
                np.mean([row["methods"][label][metric] for row in per_seed])
            )
            for metric in per_seed[0]["methods"][label]
        }
        for label in METHOD_LABELS
    }
    comparisons = {}
    for reference in ("guided_full", "centralized_greedy_daoc"):
        comparisons[f"our_vs_{reference}"] = paired_superiority(
            [row["methods"][reference]["mean_finish_time"] for row in per_seed],
            [row["methods"]["lean_our"]["mean_finish_time"] for row in per_seed],
            formal=False,
        )
    p95_central = paired_superiority(
        [
            row["methods"]["centralized_greedy_daoc"][
                "mean_p95_finish_time"
            ]
            for row in per_seed
        ],
        [row["methods"]["lean_our"]["mean_p95_finish_time"] for row in per_seed],
        formal=False,
    )
    oracle = oracle_per_seed(root / profile / "oracle")
    if set(oracle) != set(DEVELOPMENT_SEEDS):
        raise RuntimeError(f"Oracle seeds do not match {profile}")
    oracle_mean = float(np.mean(list(oracle.values())))
    our_mean = aggregates["lean_our"]["mean_finish_time"]
    return {
        "profile": profile,
        "profile_name": PROFILE_NAMES[profile],
        "capacity_multiset": expected_multiset,
        "capacity_variance": CAPACITY_VARIANCES[profile],
        "integrity": integrity,
        "method_aggregates": aggregates,
        "paired_superiority": comparisons,
        "p95_our_vs_central": p95_central,
        "oracle": {
            "mean": oracle_mean,
            "our_gap_sec": our_mean - oracle_mean,
            "our_gap_percent": 100.0 * (our_mean - oracle_mean) / oracle_mean,
            "clairvoyant_diagnostic_only": True,
        },
        "per_seed": per_seed,
    }


def plot_results(output_dir, profiles, trend):
    variances = [CAPACITY_VARIANCES[name] for name in PROFILE_ORDER]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.0),
        constrained_layout=True,
    )
    axes = axes.ravel()
    for label in METHOD_LABELS:
        axes[0].plot(
            variances,
            [
                profiles[name]["method_aggregates"][label]["mean_finish_time"]
                for name in PROFILE_ORDER
            ],
            marker="o",
            linewidth=2,
            color=COLORS[label],
            label=DISPLAY_NAMES[label],
        )
    axes[0].set_ylabel("Mean DAG completion time (s)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        variances,
        trend["our_vs_daoc_improvement_percent"],
        marker="o",
        linewidth=2,
        color=COLORS["guided_full"],
        label="OUR vs DAOC",
    )
    axes[1].plot(
        variances,
        trend["our_vs_central_improvement_percent"],
        marker="o",
        linewidth=2,
        color=COLORS["centralized_greedy_daoc"],
        label="OUR vs Centralized-Greedy",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Mean completion improvement (%)")
    axes[1].legend(frameon=False)
    axes[2].plot(
        variances,
        trend["p95_vs_central_improvement_percent"],
        marker="o",
        linewidth=2,
        color=COLORS["lean_our"],
        label="OUR vs Centralized-Greedy",
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("P95 completion improvement (%)")
    axes[2].legend(frameon=False)
    for label in ("centralized_greedy_daoc", "lean_our"):
        axes[3].plot(
            variances,
            [
                profiles[name]["method_aggregates"][label][
                    "mean_cache_hit_rate"
                ]
                for name in PROFILE_ORDER
            ],
            marker="o",
            linewidth=2,
            color=COLORS[label],
            label=DISPLAY_NAMES[label],
        )
    axes[3].set_ylabel("Cache hit rate")
    axes[3].set_ylim(0, 1)
    axes[3].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Capacity variance")
        axis.set_xticks(
            variances,
            [f"{name}\n{CAPACITY_VARIANCES[name]:.2f}" for name in PROFILE_ORDER],
        )
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "heterogeneity_trend.png", dpi=220)
    figure.savefig(output_dir / "heterogeneity_trend.pdf")
    plt.close(figure)


def write_report(path, result):
    lines = [
        "# A0固定预算异构性开发实验",
        "",
        "> A0仅用于受控机制分析，不是无偏Alibaba holdout。",
        "",
        "| 环境 | 容量方差 | DAOC | Centralized-Greedy | OUR | Oracle | 均值增益 | P95增益 | Oracle gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILE_ORDER:
        row = result["profiles"][profile]
        methods = row["method_aggregates"]
        comparison = row["paired_superiority"][
            "our_vs_centralized_greedy_daoc"
        ]
        lines.append(
            f"| {profile} | {row['capacity_variance']:.2f} | "
            f"{methods['guided_full']['mean_finish_time']:.6f} | "
            f"{methods['centralized_greedy_daoc']['mean_finish_time']:.6f} | "
            f"{methods['lean_our']['mean_finish_time']:.6f} | "
            f"{row['oracle']['mean']:.6f} | "
            f"{comparison['mean_improvement_percent']:.3f}% | "
            f"{row['p95_our_vs_central']['mean_improvement_percent']:.3f}% | "
            f"{row['oracle']['our_gap_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## 协议诊断",
            "",
            f"- 分析协议版本：`{result['revision_id']}`。",
            f"- OUR对Centralized-Greedy的收益斜率："
            f"`{result['trend']['central_gain_slope_sec_per_variance']:.6f}`。",
            f"- 收益随异构性单调不减："
            f"`{result['trend']['central_gain_monotonic_non_decreasing']}`。",
            f"- 各环境均值门槛："
            f"`{result['gate'].get('our_beats_central_each_profile', result['gate'].get('our_beats_central_in_strong_profile'))}`。",
            f"- P95门槛："
            f"`{result['gate'].get('p95_beats_central_each_profile', result['gate'].get('strong_profile_p95_not_worse'))}`。",
            f"- 开发门槛通过：`{result['gate']['passed']}`。",
            "",
            "三seed只用于机制筛选，不能作为正式显著性结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(profiles, cross_profile_paired, slope, revision_id):
    integrity = bool(
        cross_profile_paired
        and all(
            all(profile["integrity"].values())
            for profile in profiles.values()
        )
    )
    beats_daoc = all(
        profile["paired_superiority"]["our_vs_guided_full"]["passed"]
        for profile in profiles.values()
    )
    if revision_id == "h8v1":
        gate = {
            "integrity": integrity,
            "our_beats_daoc_each_profile": beats_daoc,
            "our_beats_central_each_profile": all(
                profile["paired_superiority"][
                    "our_vs_centralized_greedy_daoc"
                ]["passed"]
                for profile in profiles.values()
            ),
            "p95_beats_central_each_profile": all(
                profile["p95_our_vs_central"]["passed"]
                for profile in profiles.values()
            ),
            "coverage_constraint_satisfied_each_profile": all(
                profile["method_aggregates"]["lean_our"][
                    "mean_cache_service_coverage"
                ]
                >= (TOTAL_CACHE_BUDGET / 10.0) - 1e-12
                for profile in profiles.values()
            ),
        }
    else:
        strong = profiles["S8"]
        gate = {
            "integrity": integrity,
            "our_beats_daoc_each_profile": beats_daoc,
            "our_beats_central_in_strong_profile": strong[
                "paired_superiority"
            ]["our_vs_centralized_greedy_daoc"]["passed"],
            "central_advantage_slope_positive": slope > 0,
            "strong_profile_p95_not_worse": (
                strong["method_aggregates"]["lean_our"][
                    "mean_p95_finish_time"
                ]
                <= strong["method_aggregates"][
                    "centralized_greedy_daoc"
                ]["mean_p95_finish_time"]
            ),
        }
    gate["passed"] = bool(all(gate.values()))
    return gate


def analyze(args):
    root = args.suite_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    banks_by_seed = {}
    profiles = {
        profile: collect_profile(root, profile, banks_by_seed)
        for profile in PROFILE_ORDER
    }
    revision_id = detect_our_revision(root)
    cross_profile_paired = all(
        all(bank == banks[0] for bank in banks[1:])
        for banks in banks_by_seed.values()
    )
    central_gains = [
        profiles[name]["paired_superiority"][
            "our_vs_centralized_greedy_daoc"
        ]["mean_improvement_sec"]
        for name in PROFILE_ORDER
    ]
    variances = [CAPACITY_VARIANCES[name] for name in PROFILE_ORDER]
    slope = float(np.polyfit(variances, central_gains, 1)[0])
    trend = {
        "central_gain_sec": central_gains,
        "central_gain_slope_sec_per_variance": slope,
        "central_gain_monotonic_non_decreasing": all(
            later >= earlier
            for earlier, later in zip(central_gains, central_gains[1:])
        ),
        "strong_gain_exceeds_balanced": central_gains[-1] > central_gains[0],
        "our_vs_daoc_improvement_percent": [
            profiles[name]["paired_superiority"]["our_vs_guided_full"][
                "mean_improvement_percent"
            ]
            for name in PROFILE_ORDER
        ],
        "our_vs_central_improvement_percent": [
            profiles[name]["paired_superiority"][
                "our_vs_centralized_greedy_daoc"
            ]["mean_improvement_percent"]
            for name in PROFILE_ORDER
        ],
        "p95_vs_central_improvement_percent": [
            profiles[name]["p95_our_vs_central"][
                "mean_improvement_percent"
            ]
            for name in PROFILE_ORDER
        ],
    }
    gate = build_gate(
        profiles=profiles,
        cross_profile_paired=cross_profile_paired,
        slope=slope,
        revision_id=revision_id,
    )
    result = {
        "status": "complete",
        "claim_scope": "A0_controlled_mechanism_development_only",
        "revision_id": revision_id,
        "seeds": list(DEVELOPMENT_SEEDS),
        "total_cache_budget": TOTAL_CACHE_BUDGET,
        "cross_profile_scenario_banks_paired": cross_profile_paired,
        "profiles": profiles,
        "trend": trend,
        "gate": gate,
    }
    write_json(output / "heterogeneity_summary.json", result)
    plot_results(output, profiles, trend)
    write_report(output / "HETEROGENEITY_REPORT_ZH.md", result)
    return result


def main():
    result = analyze(parse_args())
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
