#!/usr/bin/env python3
"""Audit and summarize strictly paired environment stress results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_DAOC_LABEL = "guided_full"
DEFAULT_OUR_LABEL = "lean_our"


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value):
    return [int(item) for item in parse_csv(value)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze strictly paired DAOC stress experiments."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument(
        "--environments",
        type=parse_csv,
        required=True,
    )
    parser.add_argument("--seeds", type=parse_seed_list, required=True)
    parser.add_argument("--daoc-label", default=DEFAULT_DAOC_LABEL)
    parser.add_argument("--our-label", default=DEFAULT_OUR_LABEL)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean
    half_width = float(
        stats.t.ppf(0.975, values.size - 1)
        * stats.sem(values)
    )
    return mean - half_width, mean + half_width


def paired_statistics(reference, candidate, lower_is_better):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    direction = -1.0 if lower_is_better else 1.0
    differences = candidate - reference
    denominator = np.where(
        np.abs(reference) > 1e-12,
        np.abs(reference),
        np.where(np.abs(candidate) > 1e-12, np.abs(candidate), 1.0),
    )
    improvements = (
        100.0 * direction * differences / denominator
    )
    ties = np.isclose(
        candidate,
        reference,
        rtol=1e-12,
        atol=1e-12,
    )
    if lower_is_better:
        wins = int(np.sum((candidate < reference) & ~ties))
        losses = int(np.sum((candidate > reference) & ~ties))
        alternative = "less"
    else:
        wins = int(np.sum((candidate > reference) & ~ties))
        losses = int(np.sum((candidate < reference) & ~ties))
        alternative = "greater"
    non_ties = wins + losses
    interval = confidence_interval(improvements)
    if reference.size < 2 or np.allclose(differences, 0.0):
        paired_t_p = 1.0
        wilcoxon_p = 1.0
    else:
        paired_t_p = float(
            stats.ttest_rel(candidate, reference).pvalue
        )
        wilcoxon_p = float(
            stats.wilcoxon(
                candidate,
                reference,
                alternative=alternative,
            ).pvalue
        )
    return {
        "pairs": int(reference.size),
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(ties)),
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "ratio_of_means_improvement_percent": float(
            100.0
            * direction
            * (candidate.mean() - reference.mean())
            / (
                abs(reference.mean())
                if abs(reference.mean()) > 1e-12
                else (
                    abs(candidate.mean())
                    if abs(candidate.mean()) > 1e-12
                    else 1.0
                )
            )
        ),
        "mean_paired_improvement_percent": float(
            improvements.mean()
        ),
        "median_paired_improvement_percent": float(
            np.median(improvements)
        ),
        "paired_improvement_ci95_lower": interval[0],
        "paired_improvement_ci95_upper": interval[1],
        "paired_t_two_sided_p": paired_t_p,
        "wilcoxon_one_sided_p": wilcoxon_p,
        "sign_test_one_sided_p": (
            float(
                stats.binomtest(
                    wins,
                    non_ties,
                    p=0.5,
                    alternative="greater",
                ).pvalue
            )
            if non_ties
            else 1.0
        ),
    }


def evaluation_rows(run_dir):
    with (run_dir / "episodes.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        return [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def run_dir(suite_dir, environment, label, seed):
    return (
        suite_dir
        / environment
        / "runs"
        / label
        / f"seed_{seed}"
    )


def audit_pairing(
    suite_dir,
    environments,
    seeds,
    daoc_label,
    our_label,
):
    reference_environment = (
        "e0_original"
        if "e0_original" in environments
        else environments[0]
    )
    checks = []
    for seed in seeds:
        reference = evaluation_rows(
            run_dir(
                suite_dir,
                reference_environment,
                daoc_label,
                seed,
            )
        )
        reference_base = [
            row["base_scenario_fingerprint"]
            for row in reference
        ]
        reference_seeds = [
            row["scenario_seed"]
            for row in reference
        ]
        for environment in environments:
            daoc = evaluation_rows(
                run_dir(
                    suite_dir,
                    environment,
                    daoc_label,
                    seed,
                )
            )
            our = evaluation_rows(
                run_dir(
                    suite_dir,
                    environment,
                    our_label,
                    seed,
                )
            )
            check = {
                "environment": environment,
                "seed": seed,
                "episodes": len(daoc),
                "methods_share_stressed_scenarios": (
                    [
                        row["scenario_fingerprint"]
                        for row in daoc
                    ]
                    == [
                        row["scenario_fingerprint"]
                        for row in our
                    ]
                ),
                "methods_share_base_scenarios": (
                    [
                        row["base_scenario_fingerprint"]
                        for row in daoc
                    ]
                    == [
                        row["base_scenario_fingerprint"]
                        for row in our
                    ]
                ),
                "environment_shares_e0_base_scenarios": (
                    [
                        row["base_scenario_fingerprint"]
                        for row in daoc
                    ]
                    == reference_base
                ),
                "environment_shares_e0_scenario_seeds": (
                    [row["scenario_seed"] for row in daoc]
                    == reference_seeds
                ),
            }
            checks.append(check)

    boolean_keys = [
        key
        for key in checks[0]
        if key not in {"environment", "seed", "episodes"}
    ]
    all_passed = all(
        check[key]
        for check in checks
        for key in boolean_keys
    )
    if not all_passed:
        failed = [
            check
            for check in checks
            if not all(check[key] for key in boolean_keys)
        ]
        raise RuntimeError(
            f"Strict scenario pairing failed: {failed}"
        )
    return {"all_passed": True, "checks": checks}


def audit_convergence(
    suite_dir,
    environments,
    seeds,
    required,
    daoc_label,
    our_label,
):
    if not required:
        return {
            "required": False,
            "all_passed": None,
            "checks": [],
        }

    shared_keys = (
        "num_users",
        "num_servers",
        "num_services",
        "num_tasks",
        "dag_depth_increment",
        "dependency_data_scale",
        "server_capacity",
        "bandwidth",
        "epsilon",
        "hidden_units",
        "eval_episodes",
        "validation_scenarios",
        "convergence_window",
        "convergence_patience",
        "convergence_relative_mean_change",
        "convergence_relative_slope",
        "eval_seed_offset",
        "validation_seed_offset",
        "eval_bank_scope",
    )
    checks = []
    for environment in environments:
        environment_manifest = read_json(
            suite_dir / environment / "suite_manifest.json"
        )
        manifest_passed = (
            environment_manifest.get("status") == "complete"
            and not environment_manifest.get("failed_runs")
            and not environment_manifest.get("nonconverged_runs")
        )
        for seed in seeds:
            daoc_dir = run_dir(
                suite_dir,
                environment,
                daoc_label,
                seed,
            )
            our_dir = run_dir(
                suite_dir,
                environment,
                our_label,
                seed,
            )
            daoc_summary = read_json(daoc_dir / "summary.json")
            our_summary = read_json(our_dir / "summary.json")
            daoc_args = read_json(daoc_dir / "config.json")[
                "arguments"
            ]
            our_args = read_json(our_dir / "config.json")[
                "arguments"
            ]
            shared_protocol = all(
                daoc_args[key] == our_args[key]
                for key in shared_keys
            )
            daoc_budget_not_smaller = (
                daoc_args["train_episodes"]
                >= our_args["train_episodes"]
            )
            daoc_native = (
                daoc_args["batch_size"] == 1024
                and daoc_args["min_experiences"] == 1024
                and daoc_args["filling_steps"] == 500
                and daoc_args["steps_to_updates"] == 100
                and daoc_args["max_explore"] == 20000
                and daoc_args["learning_rate"] == 0.001
                and daoc_args["learning_rate_schedule"] == "cosine"
                and daoc_args["checkpoint_every"] == 1000
                and daoc_args["convergence_min_episodes"] == 15000
                and daoc_args["reward_mode"] == "terminal_binary"
                and daoc_args["cache_policy"] == "popularity_ema"
            )
            lean_candidate = our_label == DEFAULT_OUR_LABEL
            lean_configuration = (
                our_args["algorithm"] == "causal_telemetryPD3QN"
                and our_args["reward_mode"]
                == "causal_makespan_increment"
                and our_args["priority_alpha"] == 0.0
                and our_args["criticality_boost"] == 0.0
                and our_args["num_quantiles"] == 1
                and our_args["risk_tail_fraction"] == 1.0
                and our_args["entropy_coefficient"] == 0.0
                and not our_args["historical_feedback_guidance"]
                and not our_args["adaptive_guidance_gate"]
            )
            our_native = (
                our_args["batch_size"] == 64
                and our_args["min_experiences"] == 64
                and our_args["filling_steps"] == 20
                and our_args["steps_to_updates"] == 5
                and our_args["max_explore"] == 150
                and our_args["learning_rate"] == 0.0005
                and our_args["learning_rate_schedule"] == "constant"
                and our_args["checkpoint_every"] == 500
                and our_args["convergence_min_episodes"] == 2000
                and our_args["reward_mode"]
                in {
                    "causal_critical_path",
                    "causal_makespan_increment",
                }
                and our_args["cache_policy"] == "critical_path_joint"
                and (
                    lean_configuration
                    if lean_candidate
                    else True
                )
            )
            daoc_converged = (
                daoc_summary.get("eligible_for_comparison") is True
                and daoc_summary["convergence"]["reached"] is True
                and daoc_summary["convergence"]["stop_reason"]
                == "criterion_met"
                and daoc_summary["convergence"]["final_diagnostics"][
                    "converged"
                ]
            )
            our_converged = (
                our_summary.get("eligible_for_comparison") is True
                and our_summary["convergence"]["reached"] is True
                and our_summary["convergence"]["stop_reason"]
                == "criterion_met"
                and our_summary["convergence"]["final_diagnostics"][
                    "converged"
                ]
            )
            checks.append(
                {
                    "environment": environment,
                    "seed": seed,
                    "environment_suite_complete": manifest_passed,
                    "shared_environment_and_convergence_protocol": (
                        shared_protocol
                    ),
                    "daoc_budget_not_smaller": (
                        daoc_budget_not_smaller
                    ),
                    "daoc_native_training_profile": daoc_native,
                    "our_native_training_profile": our_native,
                    "daoc_converged": daoc_converged,
                    "our_converged": our_converged,
                    "daoc_convergence_episode": daoc_summary[
                        "convergence"
                    ]["actual_train_episodes"],
                    "our_convergence_episode": our_summary[
                        "convergence"
                    ]["actual_train_episodes"],
                }
            )

    boolean_keys = (
        "environment_suite_complete",
        "shared_environment_and_convergence_protocol",
        "daoc_budget_not_smaller",
        "daoc_native_training_profile",
        "our_native_training_profile",
        "daoc_converged",
        "our_converged",
    )
    all_passed = all(
        check[key]
        for check in checks
        for key in boolean_keys
    )
    if not all_passed:
        failed = [
            check
            for check in checks
            if not all(check[key] for key in boolean_keys)
        ]
        raise RuntimeError(
            f"Strict convergence audit failed: {failed}"
        )
    return {
        "required": True,
        "all_passed": True,
        "checks": checks,
    }


def environment_row(
    suite_dir,
    environment,
    seeds,
    daoc_label,
    our_label,
):
    oracle = read_json(
        suite_dir
        / environment
        / "oracle"
        / "oracle_floor_summary.json"
    )
    config = read_json(
        next(
            (
                suite_dir
                / environment
                / "runs"
                / our_label
            ).glob("seed_*/config.json")
        )
    )
    arguments = config["arguments"]
    aggregates = oracle["method_aggregates"]
    gap = oracle["gap_analysis"]
    daoc_summaries = [
        read_json(
            run_dir(
                suite_dir,
                environment,
                daoc_label,
                seed,
            )
            / "summary.json"
        )
        for seed in seeds
    ]
    our_summaries = [
        read_json(
            run_dir(
                suite_dir,
                environment,
                our_label,
                seed,
            )
            / "summary.json"
        )
        for seed in seeds
    ]
    paired = {}
    for output_name, metric, lower_is_better in (
        ("finish_time", "mean_average_finish_time", True),
        ("p95_finish_time", "mean_p95_finish_time", True),
        ("cache_hit_rate", "mean_cache_hit_rate", False),
    ):
        paired[output_name] = paired_statistics(
            [
                summary["eval"][metric]
                for summary in daoc_summaries
            ],
            [
                summary["eval"][metric]
                for summary in our_summaries
            ],
            lower_is_better=lower_is_better,
        )
    return {
        "environment": environment,
        "dag_depth_increment": arguments["dag_depth_increment"],
        "dependency_data_scale": arguments[
            "dependency_data_scale"
        ],
        "server_capacity": arguments["server_capacity"],
        "oracle_floor": aggregates["oracle_floor"]["mean"],
        "our_finish_time": aggregates["our"]["mean"],
        "daoc_finish_time": aggregates["daoc"]["mean"],
        "our_vs_daoc_percent": gap["our_vs_daoc_percent"],
        "our_reducible_fraction_percent": gap[
            "our_reducible_fraction_percent"
        ],
        "daoc_to_floor_gap_closed_percent": gap[
            "daoc_to_floor_gap_closed_percent"
        ],
        "paired_statistics": paired,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows):
    output = []
    for row in rows:
        for metric, values in row["paired_statistics"].items():
            output.append(
                {
                    "environment": row["environment"],
                    "metric": metric,
                    **values,
                }
            )
    return output


def plot_summary(path, rows, our_display_name):
    labels = [row["environment"] for row in rows]
    positions = np.arange(len(labels))
    width = 0.24
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.4),
        constrained_layout=True,
    )
    methods = (
        ("oracle_floor", "Oracle floor", "#059669"),
        ("our_finish_time", our_display_name, "#2563EB"),
        ("daoc_finish_time", "DAOC", "#4B5563"),
    )
    for offset, (key, label, color) in zip(
        (-width, 0.0, width),
        methods,
    ):
        axes[0].bar(
            positions + offset,
            [row[key] for row in rows],
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(positions, labels, rotation=20)
    axes[0].set_ylabel("Mean application finish time (s)")
    axes[0].set_title("Nested environment stress", loc="left")
    axes[0].legend(frameon=False)

    axes[1].plot(
        positions,
        [
            row["our_reducible_fraction_percent"]
            for row in rows
        ],
        marker="o",
        linewidth=2.0,
        color="#2563EB",
        label="OUR to Oracle headroom",
    )
    axes[1].plot(
        positions,
        [row["our_vs_daoc_percent"] for row in rows],
        marker="s",
        linewidth=2.0,
        color="#DC2626",
        label="OUR improvement over DAOC",
    )
    axes[1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Diagnostic gaps", loc="left")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(
    path,
    rows,
    integrity,
    stage,
    daoc_label,
    our_label,
):
    converged = stage == "converged"
    lines = [
        "# Strict Nested Environment Stress",
        "",
        (
            "This is a convergence-controlled three-seed candidate "
            "experiment."
            if converged
            else (
                "This suite is diagnostic. Smoke and screen stages are "
                "not converged-paper comparisons."
            )
        ),
        "",
        "## Pairing integrity",
        "",
        f"- DAOC label: `{daoc_label}`.",
        f"- OUR label: `{our_label}`.",
        f"- All strict pairing checks passed: {integrity['all_passed']}.",
        "- Every stress level reuses the same base scenario seeds, DAG "
        "keys, deployment, workload, and server telemetry.",
        "",
        "## Results",
        "",
        "| Environment | Depth + | Dependency data x | K | "
        "Oracle | OUR | DAOC | OUR vs DAOC | OUR headroom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['environment']} | "
            f"{row['dag_depth_increment']} | "
            f"{row['dependency_data_scale']:.1f} | "
            f"{row['server_capacity']} | "
            f"{row['oracle_floor']:.6f} | "
            f"{row['our_finish_time']:.6f} | "
            f"{row['daoc_finish_time']:.6f} | "
            f"{row['our_vs_daoc_percent']:.2f}% | "
            f"{row['our_reducible_fraction_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Seed-paired inference",
            "",
            "| Environment | Mean paired gain | 95% CI | Wins | "
            "Paired t p | One-sided Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        paired = row["paired_statistics"]["finish_time"]
        lines.append(
            f"| {row['environment']} | "
            f"{paired['mean_paired_improvement_percent']:.2f}% | "
            f"[{paired['paired_improvement_ci95_lower']:.2f}%, "
            f"{paired['paired_improvement_ci95_upper']:.2f}%] | "
            f"{paired['wins']}/{paired['pairs']} | "
            f"{paired['paired_t_two_sided_p']:.4f} | "
            f"{paired['wilcoxon_one_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Every run passed the shared convergence gate while DAOC "
                "and OUR retained their pre-calibrated method-native "
                "optimizer, validation cadence, and burn-in settings. "
                "DAOC was allowed an equal or larger maximum training "
                "budget than OUR. "
                "Three seeds are suitable for candidate screening; expand "
                "the retained environments to 10 seeds for final "
                "inferential claims."
                if converged
                else (
                    "Only a converged multi-seed retraining may be used "
                    "for paper performance claims."
                )
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    suite_manifest = read_json(
        suite_dir / "strict_suite_manifest.json"
    )
    stage = suite_manifest.get("stage", "unknown")
    pairing = audit_pairing(
        suite_dir,
        args.environments,
        args.seeds,
        daoc_label=args.daoc_label,
        our_label=args.our_label,
    )
    convergence = audit_convergence(
        suite_dir,
        args.environments,
        args.seeds,
        required=stage == "converged",
        daoc_label=args.daoc_label,
        our_label=args.our_label,
    )
    integrity = {
        "all_passed": (
            pairing["all_passed"]
            and (
                convergence["all_passed"]
                if convergence["required"]
                else True
            )
        ),
        "checks": pairing["checks"],
        "convergence": convergence,
    }
    rows = [
        environment_row(
            suite_dir,
            environment,
            args.seeds,
            daoc_label=args.daoc_label,
            our_label=args.our_label,
        )
        for environment in args.environments
    ]
    write_csv(
        suite_dir / "strict_environment_summary.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key != "paired_statistics"
            }
            for row in rows
        ],
    )
    write_csv(
        suite_dir / "strict_environment_paired_statistics.csv",
        paired_rows(rows),
    )
    summary = {
        "status": "complete",
        "daoc_label": args.daoc_label,
        "our_label": args.our_label,
        "integrity": integrity,
        "environments": rows,
    }
    (suite_dir / "strict_environment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(
        suite_dir / "STRICT_ENVIRONMENT_REPORT.md",
        rows,
        integrity,
        stage,
        args.daoc_label,
        args.our_label,
    )
    plot_summary(
        suite_dir / "strict_environment_comparison",
        rows,
        (
            "Lean OUR"
            if args.our_label == DEFAULT_OUR_LABEL
            else "OUR"
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
