#!/usr/bin/env python3
"""Analyze the frozen A0 fixed-budget heterogeneity confirmation study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a0_coordination_protocol import EXPECTED_DATASET_SHA256
from a0_fixed_budget_heterogeneity_protocol import (
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    CAPACITY_VARIANCES,
    METHOD_LABELS,
    PROFILE_NAMES,
    PROFILE_ORDER,
    TOTAL_CACHE_BUDGET,
)
from analyze_a0_coordination import (
    DISPLAY_NAMES,
    aggregate_run,
    evaluation_rows,
    oracle_per_seed,
    paired_superiority,
)
from capacity_protocol import deterministic_capacity_assignment


FINAL_SEEDS = tuple(range(11, 21))
COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in FINAL_SEEDS),
    )
    args = parser.parse_args()
    args.seeds = tuple(
        int(item.strip()) for item in args.seeds.split(",") if item.strip()
    )
    if args.seeds != FINAL_SEEDS:
        raise ValueError(f"Formal seeds must be exactly {FINAL_SEEDS}")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_dir(root, profile, label, seed):
    return root / profile / "runs" / label / f"seed_{seed}"


def workload_scenario_view(bank):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def collect_profile(root, profile, seeds, banks_by_seed):
    expected_multiset = CAPACITY_PROFILES[profile]
    per_seed = []
    integrity = {
        "dataset_hash": True,
        "all_methods_converged": True,
        "scenario_banks_paired_within_profile": True,
        "capacity_assignments_exact": True,
        "total_budget_exact": True,
        "networks_frozen_in_evaluation": True,
        "final_seed_partition": True,
        "frozen_revision_exact": True,
    }
    for seed in seeds:
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
                and summary.get("evaluation_unique_scenarios") == 100
            )
            revision = summary.get("revision", {})
            integrity["final_seed_partition"] &= (
                revision.get("seed_partition") == "final"
            )
            integrity["frozen_revision_exact"] &= (
                revision.get("id") == "h8v1"
                and revision.get("changed_module")
                == "scarcity_aware_service_coverage_constraint"
            )
            arguments = config.get("arguments", {})
            if arguments.get("bandwidth") != 15000:
                raise RuntimeError(f"Wrong bandwidth in {directory}")
            if label == "lean_our":
                if arguments.get("cache_coverage_constraint") is not True:
                    raise RuntimeError(
                        f"Missing frozen coverage constraint in {directory}"
                    )
            elif arguments.get("cache_coverage_constraint", False):
                raise RuntimeError(
                    f"Baseline received OUR coverage constraint: {directory}"
                )
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
    comparisons = {
        f"our_vs_{reference}": paired_superiority(
            [
                row["methods"][reference]["mean_finish_time"]
                for row in per_seed
            ],
            [
                row["methods"]["lean_our"]["mean_finish_time"]
                for row in per_seed
            ],
            formal=True,
        )
        for reference in ("guided_full", "centralized_greedy_daoc")
    }
    p95_central = paired_superiority(
        [
            row["methods"]["centralized_greedy_daoc"][
                "mean_p95_finish_time"
            ]
            for row in per_seed
        ],
        [
            row["methods"]["lean_our"]["mean_p95_finish_time"]
            for row in per_seed
        ],
        formal=True,
    )
    oracle = oracle_per_seed(root / profile / "oracle")
    if set(oracle) != set(seeds):
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


def build_formal_gate(profiles, cross_profile_paired):
    gate = {
        "integrity": bool(
            cross_profile_paired
            and all(
                all(profile["integrity"].values())
                for profile in profiles.values()
            )
        ),
        "our_beats_daoc_each_profile_formally": all(
            profile["paired_superiority"]["our_vs_guided_full"]["passed"]
            for profile in profiles.values()
        ),
        "our_beats_central_each_profile_formally": all(
            profile["paired_superiority"][
                "our_vs_centralized_greedy_daoc"
            ]["passed"]
            for profile in profiles.values()
        ),
        "p95_not_worse_each_profile": all(
            profile["p95_our_vs_central"]["mean_improvement_sec"] > 0
            and profile["p95_our_vs_central"]["wins"] >= 5
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
    gate["passed"] = bool(all(gate.values()))
    return gate


def plot_results(output_dir, profiles):
    variances = [CAPACITY_VARIANCES[name] for name in PROFILE_ORDER]
    figure, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.0), constrained_layout=True
    )
    axes = axes.ravel()
    for label in METHOD_LABELS:
        axes[0].plot(
            variances,
            [
                profiles[name]["method_aggregates"][label][
                    "mean_finish_time"
                ]
                for name in PROFILE_ORDER
            ],
            marker="o",
            linewidth=2,
            color=COLORS[label],
            label=DISPLAY_NAMES[label],
        )
    axes[0].set_ylabel("Mean DAG completion time (s)")
    axes[0].legend(frameon=False)

    for reference, color, display in (
        ("guided_full", COLORS["guided_full"], "OUR vs DAOC"),
        (
            "centralized_greedy_daoc",
            COLORS["centralized_greedy_daoc"],
            "OUR vs Centralized-Greedy",
        ),
    ):
        axes[1].plot(
            variances,
            [
                profiles[name]["paired_superiority"][
                    f"our_vs_{reference}"
                ]["mean_improvement_percent"]
                for name in PROFILE_ORDER
            ],
            marker="o",
            linewidth=2,
            color=color,
            label=display,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Mean completion improvement (%)")
    axes[1].legend(frameon=False)

    axes[2].plot(
        variances,
        [
            profiles[name]["p95_our_vs_central"][
                "mean_improvement_percent"
            ]
            for name in PROFILE_ORDER
        ],
        marker="o",
        linewidth=2,
        color=COLORS["lean_our"],
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("P95 improvement vs Centralized-Greedy (%)")

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
            [
                f"{name}\n{CAPACITY_VARIANCES[name]:.2f}"
                for name in PROFILE_ORDER
            ],
        )
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "final_heterogeneity.png", dpi=220)
    figure.savefig(output_dir / "final_heterogeneity.pdf")
    plt.close(figure)


def write_report(path, result):
    lines = [
        "# A0固定预算异构性十seed最终确认",
        "",
        "> A0仅用于受控机制确认，不是无偏Alibaba holdout。",
        "",
        "| 环境 | DAOC | Centralized-Greedy | OUR | Oracle | OUR vs Central | 95% CI | p | 胜出seed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PROFILE_ORDER:
        row = result["profiles"][name]
        methods = row["method_aggregates"]
        comparison = row["paired_superiority"][
            "our_vs_centralized_greedy_daoc"
        ]
        lines.append(
            f"| {name} | "
            f"{methods['guided_full']['mean_finish_time']:.6f} | "
            f"{methods['centralized_greedy_daoc']['mean_finish_time']:.6f} | "
            f"{methods['lean_our']['mean_finish_time']:.6f} | "
            f"{row['oracle']['mean']:.6f} | "
            f"{comparison['mean_improvement_percent']:.3f}% | "
            f"[{comparison['ci95_lower_sec']:.6f}, "
            f"{comparison['ci95_upper_sec']:.6f}] | "
            f"{comparison['wilcoxon_one_sided_p']:.6f} | "
            f"{comparison['wins']}/10 |"
        )
    lines.extend(
        [
            "",
            "## 最终门槛",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in result["gate"].items()
            ],
            "",
            "算法在查看seeds 11–20结果前已锁定，该批seed不得用于后续调参。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args):
    root = args.suite_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    banks_by_seed = {}
    profiles = {
        profile: collect_profile(root, profile, args.seeds, banks_by_seed)
        for profile in PROFILE_ORDER
    }
    cross_profile_paired = all(
        all(bank == banks[0] for bank in banks[1:])
        for banks in banks_by_seed.values()
    )
    gate = build_formal_gate(profiles, cross_profile_paired)
    result = {
        "status": "complete",
        "claim_scope": "A0_controlled_mechanism_final_confirmation",
        "revision_id": "h8v1",
        "seeds": list(args.seeds),
        "total_cache_budget": TOTAL_CACHE_BUDGET,
        "cross_profile_scenario_banks_paired": cross_profile_paired,
        "profiles": profiles,
        "gate": gate,
    }
    write_json(output / "final_heterogeneity_summary.json", result)
    plot_results(output, profiles)
    write_report(output / "FINAL_HETEROGENEITY_REPORT_ZH.md", result)
    return result


def main():
    result = analyze(parse_args())
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
