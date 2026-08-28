#!/usr/bin/env python3
"""Evaluate a certified clairvoyant latency floor on frozen scenarios."""

import argparse
import copy
import csv
import io
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_pd3qn_methods import confidence_interval, paired_statistics
from oracle_latency_bound import (
    assignment_problem_from_simulator,
    exact_optimistic_assignment_oracle,
    relaxed_assignment_lower_bound,
    scenario_capacity_aware_oracle_bounds,
    scenario_relaxed_oracle_bounds,
)
from run_independent_experiment import (
    apply_deployment_state,
    base_scenario_fingerprint,
    capture_deployment_state,
    scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
)
from simulator import MEC_Simulator


def parse_seed_list(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or any(seed < 1 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "--seeds must contain positive integers"
        )
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the clairvoyant optimistic latency floor."
    )
    parser.add_argument(
        "--our-suite-dir",
        type=Path,
        default=Path("results/hcpr_telemetry_calibration_3seed"),
    )
    parser.add_argument(
        "--our-label",
        default="hcpr_telemetry_pd3qn",
    )
    parser.add_argument(
        "--daoc-suite-dir",
        type=Path,
        default=Path("results/paper_dual_main_10seed"),
    )
    parser.add_argument("--daoc-label", default="guided_full")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/oracle_latency_floor_10seed"),
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=list(range(1, 11)),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--exact-check-scenarios",
        type=int,
        default=1,
        help="Number of scenarios across the whole run checked by MILP.",
    )
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=30.0,
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.exact_check_scenarios < 0:
        raise ValueError(
            "--exact-check-scenarios must be non-negative"
        )
    if args.exact_time_limit <= 0:
        raise ValueError("--exact-time-limit must be positive")
    return args


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_rows(run_dir, episodes):
    with (run_dir / "episodes.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) < episodes:
        raise RuntimeError(
            f"Expected at least {episodes} eval rows in {run_dir}"
        )
    return rows[:episodes]


def aggregate(values):
    array = np.asarray(values, dtype=float)
    lower, upper, half_width = confidence_interval(array)
    return {
        "mean": float(array.mean()),
        "ci95_half_width": half_width,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def recreate_scenario(
    input_config,
    learning_config,
    deployment_state,
    eval_seed,
    output_dir,
    workflow_family=None,
    eval_bank_scope="workload",
):
    seed_everything(eval_seed)
    eval_config = copy.deepcopy(input_config)
    eval_config["seed"] = eval_seed
    eval_config["save topology figure"] = False
    if workflow_family is not None:
        eval_config["application graph family"] = workflow_family
    simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=eval_config,
        learning_arguments=learning_config,
        filename_png=str(output_dir),
    )
    apply_deployment_state(
        simulator,
        deployment_state,
        include_users=eval_bank_scope == "workload",
    )
    fingerprint = scenario_fingerprint(
        scenario_snapshot(simulator)
    )
    return simulator, fingerprint, base_scenario_fingerprint(simulator)


def exact_check(simulator, relaxed_per_user, time_limit_sec):
    exact_values = []
    wall_time = 0.0
    for user_id, user in simulator.users.items():
        result = exact_optimistic_assignment_oracle(
            assignment_problem_from_simulator(simulator, user),
            time_limit_sec=time_limit_sec,
        )
        relaxed = relaxed_per_user[user_id]
        if relaxed > result.objective + 1e-6:
            raise RuntimeError(
                "Relaxed floor exceeded exact optimistic oracle"
            )
        exact_values.append(result.objective)
        wall_time += result.wall_time_sec
    relaxed_mean = float(np.mean(relaxed_per_user))
    exact_mean = float(np.mean(exact_values))
    signed_gap = 100.0 * (exact_mean - relaxed_mean) / exact_mean
    return {
        "relaxed_mean": relaxed_mean,
        "exact_mean": exact_mean,
        "relaxation_gap_percent": max(0.0, signed_gap),
        "signed_numeric_gap_percent": signed_gap,
        "wall_time_sec": wall_time,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary):
    floor = summary["method_aggregates"]["oracle_floor"]
    our = summary["method_aggregates"]["our"]
    daoc = summary["method_aggregates"]["daoc"]
    gap = summary["gap_analysis"]
    exact = summary["exact_validation"]
    exact_gap = exact["mean_relaxation_gap_percent"]
    exact_gap_text = (
        f"{exact_gap:.3f}%"
        if exact_gap is not None
        else "not run"
    )
    lines = [
        "# Clairvoyant Capacity-Aware Reference",
        "",
        "## Protocol",
        "",
        "- Recreated all frozen evaluation scenarios from their deterministic seeds.",
        "- Verified every reconstructed fingerprint against both DAOC and OUR.",
        "- The primary Oracle sees the complete workload and true server/link parameters.",
        "- Its service placement enforces every per-server K_s constraint.",
        "- Assignment branches and queue coupling are relaxed.",
        "- Cache placement is a capacity-feasible clairvoyant greedy solution, not a globally certified joint optimum.",
        "- The separate perfect-cache recurrence is the certified latency floor.",
        "",
        "## Result",
        "",
        "| Method | Mean finish time (s) |",
        "|---|---:|",
        f"| DAOC | {daoc['mean']:.6f} |",
        f"| Best validated OUR | {our['mean']:.6f} |",
        f"| Capacity-feasible clairvoyant reference | {floor['mean']:.6f} |",
        "",
        f"- OUR improves over DAOC by {gap['our_vs_daoc_percent']:.2f}%.",
        f"- OUR remains {gap['our_above_floor_seconds']:.6f} s above this diagnostic reference.",
        f"- The observed OUR-to-reference gap is {gap['our_reducible_fraction_percent']:.2f}% of OUR latency.",
        f"- OUR has closed {gap['daoc_to_floor_gap_closed_percent']:.2f}% of the DAOC-to-reference gap.",
        "",
        "## Exact Check",
        "",
        f"- MILP-validated scenarios: {exact['scenarios']}.",
        "- Mean DP relaxation gap to the exact optimistic MILP: "
        f"{exact_gap_text}.",
        "",
        "The capacity-aware result is clairvoyant and diagnostic only. It uses future workload, greedy cache placement, and relaxed queue coupling, so it must not be presented as an online method or a certified global lower bound.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(path, seed_rows, summary):
    seeds = np.asarray([row["seed"] for row in seed_rows])
    floor = np.asarray(
        [row["oracle_floor"] for row in seed_rows],
        dtype=float,
    )
    our = np.asarray(
        [row["our_finish_time"] for row in seed_rows],
        dtype=float,
    )
    daoc = np.asarray(
        [row["daoc_finish_time"] for row in seed_rows],
        dtype=float,
    )
    aggregates = summary["method_aggregates"]
    methods = ("oracle_floor", "our", "daoc")
    labels = ("Oracle floor", "Best OUR", "DAOC")
    colors = ("#059669", "#2563EB", "#4B5563")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        constrained_layout=True,
    )
    for values, label, color in zip(
        (floor, our, daoc),
        labels,
        colors,
    ):
        axes[0].plot(
            seeds,
            values,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=label,
            color=color,
        )
    axes[0].fill_between(
        seeds,
        floor,
        our,
        color="#93C5FD",
        alpha=0.25,
        label="Maximum remaining headroom",
    )
    axes[0].set_xticks(seeds)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Mean application finish time (s)")
    axes[0].set_title("Paired frozen scenarios", loc="left")
    axes[0].legend(frameon=False, fontsize=9)

    means = [aggregates[method]["mean"] for method in methods]
    errors = [
        aggregates[method]["ci95_half_width"]
        for method in methods
    ]
    axes[1].bar(
        labels,
        means,
        yerr=errors,
        capsize=4,
        color=colors,
        width=0.64,
    )
    lower_limit = min(means) - 0.025
    upper_limit = max(means) + 0.025
    axes[1].set_ylim(lower_limit, upper_limit)
    axes[1].set_ylabel("Mean application finish time (s)")
    axes[1].set_title("10-seed mean and 95% CI", loc="left")
    for index, value in enumerate(means):
        axes[1].text(
            index,
            value + errors[index] + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Clairvoyant Capacity-Aware Reference",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    our_suite = args.our_suite_dir.resolve()
    daoc_suite = args.daoc_suite_dir.resolve()

    scenario_rows = []
    seed_rows = []
    exact_rows = []
    exact_remaining = args.exact_check_scenarios
    started = time.perf_counter()

    for seed in args.seeds:
        our_run = (
            our_suite
            / "runs"
            / args.our_label
            / f"seed_{seed}"
        )
        daoc_run = (
            daoc_suite
            / "runs"
            / args.daoc_label
            / f"seed_{seed}"
        )
        config = read_json(our_run / "config.json")
        arguments = config["arguments"]
        if arguments["eval_bank_scope"] not in (
            "workload",
            "infrastructure",
        ):
            raise RuntimeError(
                "Oracle evaluator requires workload or infrastructure banks"
            )
        our_rows = evaluation_rows(our_run, args.episodes)
        daoc_rows = evaluation_rows(daoc_run, args.episodes)
        if [
            row["base_scenario_fingerprint"] for row in our_rows
        ] != [
            row["base_scenario_fingerprint"] for row in daoc_rows
        ]:
            raise RuntimeError(
                f"DAOC/OUR fingerprint mismatch for seed {seed}"
            )

        seed_everything(seed)
        initial_config = copy.deepcopy(config["input_config"])
        initial_config["save topology figure"] = False
        initial_simulator = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=initial_config,
            learning_arguments=config["learning_config"],
            filename_png=str(output_dir),
        )
        deployment_state = capture_deployment_state(
            initial_simulator
        )

        current_seed_rows = []
        eval_dag_families = arguments.get("eval_dag_families")
        for episode_index, (our_row, daoc_row) in enumerate(
            zip(our_rows, daoc_rows),
            start=1,
        ):
            eval_seed = (
                seed
                + int(arguments["eval_seed_offset"])
                + episode_index
                - 1
            )
            workflow_family = None
            if eval_dag_families:
                workflow_family = eval_dag_families[
                    (episode_index - 1) % len(eval_dag_families)
                ]
            simulator, fingerprint, base_fingerprint = recreate_scenario(
                input_config=config["input_config"],
                learning_config=config["learning_config"],
                deployment_state=deployment_state,
                eval_seed=eval_seed,
                output_dir=output_dir,
                workflow_family=workflow_family,
                eval_bank_scope=arguments["eval_bank_scope"],
            )
            expected_base_fingerprint = our_row[
                "base_scenario_fingerprint"
            ]
            if base_fingerprint != expected_base_fingerprint:
                raise RuntimeError(
                    "Reconstructed base fingerprint mismatch: "
                    f"seed={seed}, episode={episode_index}"
                )

            perfect_bound = scenario_relaxed_oracle_bounds(
                simulator
            )
            bound = scenario_capacity_aware_oracle_bounds(
                simulator
            )
            our_finish = float(our_row["average_finish_time"])
            daoc_finish = float(daoc_row["average_finish_time"])
            if (
                perfect_bound["mean"]
                > min(our_finish, daoc_finish) + 1e-8
            ):
                raise RuntimeError(
                    "Oracle floor exceeded an observed policy latency"
                )

            row = {
                "seed": seed,
                "episode": episode_index,
                "scenario_seed": eval_seed,
                "scenario_fingerprint": fingerprint,
                "oracle_floor": bound["mean"],
                "oracle_floor_p95": bound["p95"],
                "perfect_cache_floor": perfect_bound["mean"],
                "oracle_capacity_constraints_satisfied": int(
                    bound["capacity_constraints_satisfied"]
                ),
                "oracle_cache_placement_json": json.dumps(
                    bound["cache_placement"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "our_finish_time": our_finish,
                "daoc_finish_time": daoc_finish,
                "our_above_floor_seconds": (
                    our_finish - bound["mean"]
                ),
                "daoc_above_floor_seconds": (
                    daoc_finish - bound["mean"]
                ),
            }
            scenario_rows.append(row)
            current_seed_rows.append(row)

            if exact_remaining > 0:
                exact_rows.append(
                    {
                        "seed": seed,
                        "episode": episode_index,
                        **exact_check(
                            simulator,
                            perfect_bound["per_user"],
                            args.exact_time_limit,
                        ),
                    }
                )
                exact_remaining -= 1

        seed_rows.append(
            {
                "seed": seed,
                "oracle_floor": float(
                    np.mean(
                        [
                            row["oracle_floor"]
                            for row in current_seed_rows
                        ]
                    )
                ),
                "perfect_cache_floor": float(
                    np.mean(
                        [
                            row["perfect_cache_floor"]
                            for row in current_seed_rows
                        ]
                    )
                ),
                "our_finish_time": float(
                    np.mean(
                        [
                            row["our_finish_time"]
                            for row in current_seed_rows
                        ]
                    )
                ),
                "daoc_finish_time": float(
                    np.mean(
                        [
                            row["daoc_finish_time"]
                            for row in current_seed_rows
                        ]
                    )
                ),
            }
        )
        print(
            f"seed {seed}: floor={seed_rows[-1]['oracle_floor']:.6f}, "
            f"our={seed_rows[-1]['our_finish_time']:.6f}, "
            f"daoc={seed_rows[-1]['daoc_finish_time']:.6f}"
        )

    floor_values = [row["oracle_floor"] for row in seed_rows]
    perfect_floor_values = [
        row["perfect_cache_floor"] for row in seed_rows
    ]
    our_values = [row["our_finish_time"] for row in seed_rows]
    daoc_values = [row["daoc_finish_time"] for row in seed_rows]
    floor_mean = float(np.mean(floor_values))
    our_mean = float(np.mean(our_values))
    daoc_mean = float(np.mean(daoc_values))
    daoc_to_floor = daoc_mean - floor_mean

    summary = {
        "status": "complete",
        "oracle": (
            "clairvoyant_capacity_feasible_relaxed_assignment_v1"
        ),
        "capacity_reference_is_certified_global_floor": False,
        "perfect_cache_reference_is_certified_floor": True,
        "integrity": {
            "seeds": len(args.seeds),
            "scenarios_per_seed": args.episodes,
            "total_scenarios": len(scenario_rows),
            "all_fingerprints_match": True,
            "perfect_cache_floor_never_exceeds_observed_latency": (
                True
            ),
            "capacity_reference_never_exceeds_observed_latency": all(
                row["oracle_floor"]
                <= min(
                    row["our_finish_time"],
                    row["daoc_finish_time"],
                )
                + 1e-8
                for row in scenario_rows
            ),
            "capacity_constraints_satisfied": all(
                row["oracle_capacity_constraints_satisfied"]
                for row in scenario_rows
            ),
        },
        "method_aggregates": {
            "oracle_floor": aggregate(floor_values),
            "perfect_cache_floor": aggregate(
                perfect_floor_values
            ),
            "our": aggregate(our_values),
            "daoc": aggregate(daoc_values),
        },
        "paired_comparisons": {
            "our_vs_daoc": paired_statistics(
                daoc_values,
                our_values,
                lower_is_better=True,
            ),
            "our_vs_floor": paired_statistics(
                our_values,
                floor_values,
                lower_is_better=True,
            ),
        },
        "gap_analysis": {
            "our_vs_daoc_percent": (
                100.0 * (daoc_mean - our_mean) / daoc_mean
            ),
            "our_above_floor_seconds": our_mean - floor_mean,
            "our_above_floor_percent_of_floor": (
                100.0 * (our_mean - floor_mean) / floor_mean
            ),
            "our_reducible_fraction_percent": (
                100.0 * (our_mean - floor_mean) / our_mean
            ),
            "daoc_to_floor_gap_closed_percent": (
                100.0 * (daoc_mean - our_mean) / daoc_to_floor
            ),
        },
        "exact_validation": {
            "scenarios": len(exact_rows),
            "mean_relaxation_gap_percent": (
                float(
                    np.mean(
                        [
                            row["relaxation_gap_percent"]
                            for row in exact_rows
                        ]
                    )
                )
                if exact_rows
                else None
            ),
            "details": exact_rows,
        },
        "wall_time_sec": time.perf_counter() - started,
    }

    write_csv(output_dir / "oracle_floor_per_scenario.csv", scenario_rows)
    write_csv(output_dir / "oracle_floor_per_seed.csv", seed_rows)
    (output_dir / "oracle_floor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        output_dir / "ORACLE_LATENCY_FLOOR_REPORT.md",
        summary,
    )
    plot_results(
        output_dir / "oracle_latency_floor",
        seed_rows,
        summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
