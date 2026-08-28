#!/usr/bin/env python3
"""Audit and compare PD3QN with converged DAOC and the previous OUR."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from information_protocol import (
    CAUSAL_CACHE_INFORMATION_REGIME,
    INFORMATION_PROTOCOL_VERSION,
)


METHODS = {
    "daoc": {
        "label": "guided_full",
        "display": "DAOC",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "cache_policy": "popularity_ema",
        "reward_mode": "terminal_binary",
    },
    "previous_our": {
        "label": "our",
        "display": "Previous OUR",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "cache_policy": "critical_path_joint",
        "reward_mode": "terminal_plus_potential",
    },
    "pd3qn": {
        "label": "pd3qn",
        "display": "PD3QN",
        "algorithm": "causal_task_serverPD3QN",
        "cache_policy": "critical_path_joint",
        "reward_mode": "causal_critical_path",
    },
}

PHYSICAL_CONFIG_KEYS = (
    "Bandwidth",
    "Max rate to cloud",
    "Min rate to cloud",
    "Number of servers",
    "Number of services",
    "Number of tasks for each user",
    "Number of users",
    "Power",
    "SimulationTime",
    "cache compute weight",
    "cache history alpha",
    "cache hysteresis factor",
    "cache locality weight",
    "cache min residence updates",
    "cache score alpha",
    "cache update interval",
    "caching decision enabled",
    "deadline",
    "max_cpu_cycles",
    "max_cpu_frquency",
    "max_data_length",
    "max_load_on_server",
    "max_rate_between_servers",
    "max_service_data_length",
    "min_cpu_frquency",
    "min_load_on_server",
    "min_rate_between_servers",
    "min_service_data_length",
    "server capacity",
    "update deadline",
    "velocity",
    "xlim",
    "ylim",
)

EVALUATION_CONFIG_KEYS = (
    "eval_bank_scope",
    "eval_episodes",
    "eval_scenario_bank",
    "eval_seed_offset",
    "eval_update_caching",
    "validation_scenarios",
    "validation_seed_offset",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare converged PD3QN with converged DAOC and previous OUR "
            "on exactly paired validation and evaluation scenario banks."
        )
    )
    parser.add_argument("--reference-suite-dir", type=Path, required=True)
    parser.add_argument("--pd3qn-suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def discover_seeds(suite_dir, label):
    return {
        int(path.name.split("_", 1)[1])
        for path in (suite_dir / "runs" / label).glob("seed_*")
        if (path / "summary.json").exists()
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


def validation_fingerprints(run_dir):
    records = read_json(
        run_dir / "checkpoint_validation.json"
    )["records"]
    require(records, f"No validation records in {run_dir}")
    fingerprints = tuple(records[0]["scenario_fingerprints"])
    require(
        all(
            tuple(record["scenario_fingerprints"]) == fingerprints
            for record in records
        ),
        f"Validation bank changed within {run_dir}",
    )
    return fingerprints


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, 0.0
    half_width = float(
        stats.t.ppf(0.975, values.size - 1)
        * stats.sem(values)
    )
    return mean - half_width, mean + half_width, half_width


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
    tie_mask = np.isclose(
        candidate,
        reference,
        rtol=1e-12,
        atol=1e-12,
    )
    if lower_is_better:
        win_mask = (candidate < reference) & ~tie_mask
        loss_mask = (candidate > reference) & ~tie_mask
    else:
        win_mask = (candidate > reference) & ~tie_mask
        loss_mask = (candidate < reference) & ~tie_mask
    wins = int(np.sum(win_mask))
    losses = int(np.sum(loss_mask))
    ties = int(np.sum(tie_mask))
    reference_ci = confidence_interval(reference)
    candidate_ci = confidence_interval(candidate)
    improvement_ci = confidence_interval(improvements)
    difference_ci = confidence_interval(differences)
    non_ties = wins + losses
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
                alternative=(
                    "less" if lower_is_better else "greater"
                ),
            ).pvalue
        )
    return {
        "pairs": int(reference.size),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "reference_mean": float(reference.mean()),
        "reference_ci95_half_width": reference_ci[2],
        "candidate_mean": float(candidate.mean()),
        "candidate_ci95_half_width": candidate_ci[2],
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
        "paired_improvement_ci95_lower": improvement_ci[0],
        "paired_improvement_ci95_upper": improvement_ci[1],
        "mean_paired_difference": float(differences.mean()),
        "paired_difference_ci95_lower": difference_ci[0],
        "paired_difference_ci95_upper": difference_ci[1],
        "paired_t_two_sided_p": paired_t_p,
        "wilcoxon_one_sided_p": wilcoxon_p,
        "sign_test_one_sided_p": float(
            stats.binomtest(
                wins,
                non_ties,
                p=0.5,
                alternative="greater",
            ).pvalue
        )
        if non_ties
        else 1.0,
    }


def assert_run_integrity(method_key, run_dir):
    expected = METHODS[method_key]
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    arguments = config["arguments"]
    require(
        summary.get("status") == "complete",
        f"Incomplete run: {run_dir}",
    )
    require(
        summary.get("eligible_for_comparison") is True,
        f"Ineligible run: {run_dir}",
    )
    require(
        summary["convergence"]["reached"] is True,
        f"Non-converged run: {run_dir}",
    )
    require(
        summary.get("evaluation_state_frozen") is True
        and arguments["eval_update_caching"] is False,
        f"Evaluation state was not frozen: {run_dir}",
    )
    require(
        arguments["algorithm"] == expected["algorithm"]
        and arguments["cache_policy"] == expected["cache_policy"]
        and arguments["reward_mode"] == expected["reward_mode"],
        f"Unexpected method configuration: {run_dir}",
    )
    require(
        summary.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION,
        f"Wrong information protocol: {run_dir}",
    )
    if method_key != "daoc":
        require(
            summary.get("cache_information_regime")
            == CAUSAL_CACHE_INFORMATION_REGIME,
            f"Non-causal cache information regime: {run_dir}",
        )
    return summary, config


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path, rows, summary):
    seeds = np.asarray([row["seed"] for row in rows])
    daoc = np.asarray([row["daoc_finish_time"] for row in rows])
    previous = np.asarray(
        [row["previous_our_finish_time"] for row in rows]
    )
    pd3qn = np.asarray([row["pd3qn_finish_time"] for row in rows])
    daoc_improvement = np.asarray(
        [row["pd3qn_vs_daoc_finish_improvement_percent"] for row in rows]
    )
    previous_improvement = np.asarray(
        [
            row["pd3qn_vs_previous_our_finish_improvement_percent"]
            for row in rows
        ]
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.0),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")

    positions = [0, 1, 2]
    for index in range(len(seeds)):
        axes[0, 0].plot(
            positions,
            [daoc[index], previous[index], pd3qn[index]],
            color="#9CA3AF",
            linewidth=1.0,
            alpha=0.65,
        )
    axes[0, 0].scatter(
        np.repeat(positions[0], len(seeds)),
        daoc,
        color="#4B5563",
        edgecolor="white",
        linewidth=0.6,
        s=38,
        zorder=3,
    )
    axes[0, 0].scatter(
        np.repeat(positions[1], len(seeds)),
        previous,
        color="#D97706",
        edgecolor="white",
        linewidth=0.6,
        s=38,
        zorder=3,
    )
    axes[0, 0].scatter(
        np.repeat(positions[2], len(seeds)),
        pd3qn,
        color="#2563EB",
        edgecolor="white",
        linewidth=0.6,
        s=38,
        zorder=3,
    )
    axes[0, 0].set_xticks(
        positions,
        ["DAOC", "Previous OUR", "PD3QN"],
    )
    axes[0, 0].set_ylabel("Mean application finish time (s)")
    axes[0, 0].set_title("Paired held-out performance", loc="left")

    width = 0.36
    axes[0, 1].bar(
        seeds - width / 2,
        daoc_improvement,
        width=width,
        color="#2563EB",
        label="vs DAOC",
    )
    axes[0, 1].bar(
        seeds + width / 2,
        previous_improvement,
        width=width,
        color="#D97706",
        label="vs Previous OUR",
    )
    axes[0, 1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[0, 1].set_xticks(seeds)
    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("PD3QN improvement (%)")
    axes[0, 1].set_title("Per-seed relative effect", loc="left")
    axes[0, 1].legend(frameon=False)

    metric_names = ["Mean", "P95"]
    daoc_means = [
        summary["method_aggregates"]["daoc"]["finish_time_mean"],
        summary["method_aggregates"]["daoc"]["p95_finish_time_mean"],
    ]
    previous_means = [
        summary["method_aggregates"]["previous_our"][
            "finish_time_mean"
        ],
        summary["method_aggregates"]["previous_our"][
            "p95_finish_time_mean"
        ],
    ]
    pd3qn_means = [
        summary["method_aggregates"]["pd3qn"]["finish_time_mean"],
        summary["method_aggregates"]["pd3qn"][
            "p95_finish_time_mean"
        ],
    ]
    x_values = np.arange(len(metric_names))
    axes[1, 0].bar(
        x_values - 0.25,
        daoc_means,
        width=0.25,
        color="#4B5563",
        label="DAOC",
    )
    axes[1, 0].bar(
        x_values,
        previous_means,
        width=0.25,
        color="#D97706",
        label="Previous OUR",
    )
    axes[1, 0].bar(
        x_values + 0.25,
        pd3qn_means,
        width=0.25,
        color="#2563EB",
        label="PD3QN",
    )
    axes[1, 0].set_xticks(x_values, metric_names)
    axes[1, 0].set_ylabel("Application finish time (s)")
    axes[1, 0].set_title("Mean and tail latency", loc="left")
    axes[1, 0].legend(frameon=False)

    cost_names = ["Episodes", "Wall time"]
    daoc_cost = [
        summary["method_aggregates"]["daoc"][
            "convergence_episode_mean"
        ],
        summary["method_aggregates"]["daoc"]["wall_time_sec_mean"],
    ]
    previous_cost = [
        summary["method_aggregates"]["previous_our"][
            "convergence_episode_mean"
        ],
        summary["method_aggregates"]["previous_our"][
            "wall_time_sec_mean"
        ],
    ]
    pd3qn_cost = [
        summary["method_aggregates"]["pd3qn"][
            "convergence_episode_mean"
        ],
        summary["method_aggregates"]["pd3qn"][
            "wall_time_sec_mean"
        ],
    ]
    normalized_cost = np.asarray(
        [daoc_cost, previous_cost, pd3qn_cost],
        dtype=float,
    )
    normalized_cost /= normalized_cost[0]
    x_values = np.arange(len(cost_names))
    axes[1, 1].bar(
        x_values - 0.25,
        normalized_cost[0],
        width=0.25,
        color="#4B5563",
        label="DAOC",
    )
    axes[1, 1].bar(
        x_values,
        normalized_cost[1],
        width=0.25,
        color="#D97706",
        label="Previous OUR",
    )
    axes[1, 1].bar(
        x_values + 0.25,
        normalized_cost[2],
        width=0.25,
        color="#2563EB",
        label="PD3QN",
    )
    axes[1, 1].axhline(
        1.0,
        color="#6B7280",
        linewidth=1.0,
        linestyle="--",
    )
    axes[1, 1].set_xticks(x_values, cost_names)
    axes[1, 1].set_ylabel("Cost normalized to DAOC")
    axes[1, 1].set_title("Observed training cost", loc="left")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)

    figure.suptitle(
        "PD3QN Main Experiment: 10 Paired Converged Seeds",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def aggregate_method(rows, prefix):
    fields = {
        "finish_time": f"{prefix}_finish_time",
        "p95_finish_time": f"{prefix}_p95_finish_time",
        "cache_hit_rate": f"{prefix}_cache_hit_rate",
        "convergence_episode": f"{prefix}_convergence_episode",
        "wall_time_sec": f"{prefix}_wall_time_sec",
        "eval_episode_wall_time_sec": (
            f"{prefix}_eval_episode_wall_time_sec"
        ),
    }
    result = {}
    for name, field in fields.items():
        values = np.asarray([row[field] for row in rows], dtype=float)
        interval = confidence_interval(values)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_ci95_half_width"] = interval[2]
    return result


def write_report(path, summary, rows):
    daoc = summary["comparisons"]["pd3qn_vs_daoc"]
    previous = summary["comparisons"]["pd3qn_vs_previous_our"]
    aggregates = summary["method_aggregates"]
    row_lines = [
        (
            f"| {row['seed']} | {row['daoc_finish_time']:.6f} | "
            f"{row['previous_our_finish_time']:.6f} | "
            f"{row['pd3qn_finish_time']:.6f} | "
            f"{row['pd3qn_vs_daoc_finish_improvement_percent']:.2f}% | "
            f"{row['pd3qn_vs_previous_our_finish_improvement_percent']:.2f}% |"
        )
        for row in rows
    ]
    text = "\n".join(
        [
            "# PD3QN Converged Main Experiment",
            "",
            "## Protocol integrity",
            "",
            "- Scale: 20 users, 10 servers, 10 services, and 10 tasks per user.",
            "- Independent statistical units: 10 training seeds.",
            "- Every method passed its preset convergence gate.",
            "- Each seed uses exactly the same 10 validation workloads and "
            "100 frozen held-out workloads across all three methods.",
            "- All workload fingerprints match exactly.",
            "- PD3QN and Previous OUR both use the causal history-only joint "
            "cache, so their comparison isolates the new RL/reward stack.",
            "- DAOC uses its native popularity-EMA cache and terminal reward; "
            "the DAOC comparison is full method against full method.",
            "- No unfinished-task CPU demand or exact global server state is "
            "available to PD3QN.",
            "",
            "## Primary result",
            "",
            "| Method | Mean finish time (s) | P95 finish time (s) | "
            "Cache hit rate | Convergence episodes |",
            "|---|---:|---:|---:|---:|",
            (
                f"| DAOC | {aggregates['daoc']['finish_time_mean']:.6f} | "
                f"{aggregates['daoc']['p95_finish_time_mean']:.6f} | "
                f"{aggregates['daoc']['cache_hit_rate_mean']:.4f} | "
                f"{aggregates['daoc']['convergence_episode_mean']:.0f} |"
            ),
            (
                "| Previous OUR | "
                f"{aggregates['previous_our']['finish_time_mean']:.6f} | "
                f"{aggregates['previous_our']['p95_finish_time_mean']:.6f} | "
                f"{aggregates['previous_our']['cache_hit_rate_mean']:.4f} | "
                f"{aggregates['previous_our']['convergence_episode_mean']:.0f} |"
            ),
            (
                f"| PD3QN | {aggregates['pd3qn']['finish_time_mean']:.6f} | "
                f"{aggregates['pd3qn']['p95_finish_time_mean']:.6f} | "
                f"{aggregates['pd3qn']['cache_hit_rate_mean']:.4f} | "
                f"{aggregates['pd3qn']['convergence_episode_mean']:.0f} |"
            ),
            "",
            "Against DAOC:",
            "",
            "- Mean finish time: "
            f"{daoc['finish_time']['mean_paired_improvement_percent']:.2f}% "
            "lower (95% CI "
            f"[{daoc['finish_time']['paired_improvement_ci95_lower']:.2f}%, "
            f"{daoc['finish_time']['paired_improvement_ci95_upper']:.2f}%]); "
            f"{daoc['finish_time']['wins']}/10 wins; paired t p="
            f"{daoc['finish_time']['paired_t_two_sided_p']:.6g}; "
            "one-sided Wilcoxon p="
            f"{daoc['finish_time']['wilcoxon_one_sided_p']:.6g}.",
            "- P95 finish time: "
            f"{daoc['p95_finish_time']['mean_paired_improvement_percent']:.2f}% "
            f"lower; {daoc['p95_finish_time']['wins']}/10 wins.",
            "",
            "Against Previous OUR:",
            "",
            "- Mean finish time: "
            f"{previous['finish_time']['mean_paired_improvement_percent']:.2f}% "
            "lower (95% CI "
            f"[{previous['finish_time']['paired_improvement_ci95_lower']:.2f}%, "
            f"{previous['finish_time']['paired_improvement_ci95_upper']:.2f}%]); "
            f"{previous['finish_time']['wins']}/10 wins; paired t p="
            f"{previous['finish_time']['paired_t_two_sided_p']:.6g}; "
            "one-sided Wilcoxon p="
            f"{previous['finish_time']['wilcoxon_one_sided_p']:.6g}.",
            "- P95 finish time: "
            f"{previous['p95_finish_time']['mean_paired_improvement_percent']:.2f}% "
            f"lower; {previous['p95_finish_time']['wins']}/10 wins.",
            "",
            "## Per-seed result",
            "",
            "| Seed | DAOC | Previous OUR | PD3QN | vs DAOC | vs Previous OUR |",
            "|---:|---:|---:|---:|---:|---:|",
            *row_lines,
            "",
            "## Cost and claim boundary",
            "",
            "- PD3QN reached its configured convergence criterion in "
            f"{aggregates['pd3qn']['convergence_episode_mean']:.0f} episodes, "
            "while DAOC and Previous OUR required approximately "
            f"{aggregates['daoc']['convergence_episode_mean']:.0f} and "
            f"{aggregates['previous_our']['convergence_episode_mean']:.0f}.",
            "- These convergence counts use method-specific checkpoint "
            "intervals and minimum-training settings; they demonstrate lower "
            "observed cost, not a controlled sample-efficiency theorem.",
            "- PD3QN evaluation wall time is "
            f"{1000.0 * aggregates['pd3qn']['eval_episode_wall_time_sec_mean']:.2f} "
            "ms per simulated workload on this machine, versus "
            f"{1000.0 * aggregates['daoc']['eval_episode_wall_time_sec_mean']:.2f} "
            "ms for DAOC. This includes simulator execution and is not a "
            "pure neural-network inference benchmark.",
            "- PD3QN has a substantially higher frozen-evaluation cache hit "
            "rate, but the final cache state depends on the learned routing "
            "trajectory and the method-specific training duration. A "
            "fixed-cache counterfactual is required before attributing that "
            "increase solely to the new RL architecture.",
            "- This experiment supports the stationary 20/10/10/10 result. "
            "Ablations and dynamic/stale-telemetry tests are still required "
            "before attributing the gain to each component or making a "
            "non-stationary deployment claim.",
            "",
            "## Artifacts",
            "",
            "- `pd3qn_main_per_seed.csv`",
            "- `pd3qn_main_summary.json`",
            "- `pd3qn_main_comparison.png` and `.pdf`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main():
    args = parse_args()
    reference_suite = args.reference_suite_dir.resolve()
    pd3qn_suite = args.pd3qn_suite_dir.resolve()
    suite_for_method = {
        "daoc": reference_suite,
        "previous_our": reference_suite,
        "pd3qn": pd3qn_suite,
    }

    for suite in {reference_suite, pd3qn_suite}:
        manifest = read_json(suite / "suite_manifest.json")
        require(
            manifest.get("status") == "complete"
            and not manifest.get("failed_runs")
            and not manifest.get("nonconverged_runs"),
            f"Suite is incomplete or contains failed runs: {suite}",
        )
        require(
            manifest.get("information_protocol_version")
            == INFORMATION_PROTOCOL_VERSION,
            f"Wrong suite information protocol: {suite}",
        )

    seed_sets = {
        method_key: discover_seeds(
            suite_for_method[method_key],
            method["label"],
        )
        for method_key, method in METHODS.items()
    }
    require(
        seed_sets["daoc"]
        == seed_sets["previous_our"]
        == seed_sets["pd3qn"]
        and seed_sets["daoc"],
        "The three methods do not have identical non-empty seed sets",
    )

    rows = []
    for seed in sorted(seed_sets["daoc"]):
        artifacts = {}
        for method_key, method in METHODS.items():
            run_dir = (
                suite_for_method[method_key]
                / "runs"
                / method["label"]
                / f"seed_{seed}"
            )
            summary, config = assert_run_integrity(
                method_key,
                run_dir,
            )
            artifacts[method_key] = {
                "run_dir": run_dir,
                "summary": summary,
                "config": config,
                "validation": validation_fingerprints(run_dir),
                "eval_rows": evaluation_rows(run_dir),
            }

        reference_config = artifacts["daoc"]["config"]
        for method_key in ("previous_our", "pd3qn"):
            candidate_config = artifacts[method_key]["config"]
            require(
                all(
                    reference_config["input_config"][key]
                    == candidate_config["input_config"][key]
                    for key in PHYSICAL_CONFIG_KEYS
                ),
                f"Physical environment differs for {method_key} seed {seed}",
            )
            require(
                all(
                    reference_config["arguments"][key]
                    == candidate_config["arguments"][key]
                    for key in EVALUATION_CONFIG_KEYS
                ),
                f"Evaluation protocol differs for {method_key} seed {seed}",
            )
            require(
                artifacts["daoc"]["validation"]
                == artifacts[method_key]["validation"],
                f"Validation fingerprints differ for {method_key} seed {seed}",
            )
            reference_fingerprints = tuple(
                row["scenario_fingerprint"]
                for row in artifacts["daoc"]["eval_rows"]
            )
            candidate_fingerprints = tuple(
                row["scenario_fingerprint"]
                for row in artifacts[method_key]["eval_rows"]
            )
            require(
                reference_fingerprints == candidate_fingerprints,
                f"Evaluation fingerprints differ for {method_key} seed {seed}",
            )
            require(
                len(reference_fingerprints)
                == reference_config["arguments"]["eval_episodes"],
                f"Unexpected evaluation count for seed {seed}",
            )

        row = {
            "seed": seed,
            "validation_fingerprints_match": 1,
            "evaluation_fingerprints_match": 1,
        }
        for method_key, prefix in (
            ("daoc", "daoc"),
            ("previous_our", "previous_our"),
            ("pd3qn", "pd3qn"),
        ):
            summary = artifacts[method_key]["summary"]
            row.update(
                {
                    f"{prefix}_convergence_episode": summary[
                        "convergence"
                    ]["actual_train_episodes"],
                    f"{prefix}_finish_time": summary["eval"][
                        "mean_average_finish_time"
                    ],
                    f"{prefix}_p95_finish_time": summary["eval"][
                        "mean_p95_finish_time"
                    ],
                    f"{prefix}_cache_hit_rate": summary["eval"][
                        "mean_cache_hit_rate"
                    ],
                    f"{prefix}_wall_time_sec": summary[
                        "total_wall_time_sec"
                    ],
                    f"{prefix}_eval_episode_wall_time_sec": summary[
                        "eval"
                    ]["mean_episode_wall_time_sec"],
                }
            )
        row["pd3qn_vs_daoc_finish_improvement_percent"] = (
            100.0
            * (row["daoc_finish_time"] - row["pd3qn_finish_time"])
            / row["daoc_finish_time"]
        )
        row[
            "pd3qn_vs_previous_our_finish_improvement_percent"
        ] = (
            100.0
            * (
                row["previous_our_finish_time"]
                - row["pd3qn_finish_time"]
            )
            / row["previous_our_finish_time"]
        )
        rows.append(row)

    comparisons = {}
    for reference_key, reference_prefix in (
        ("daoc", "daoc"),
        ("previous_our", "previous_our"),
    ):
        comparisons[f"pd3qn_vs_{reference_key}"] = {
            "finish_time": paired_statistics(
                [row[f"{reference_prefix}_finish_time"] for row in rows],
                [row["pd3qn_finish_time"] for row in rows],
                lower_is_better=True,
            ),
            "p95_finish_time": paired_statistics(
                [
                    row[f"{reference_prefix}_p95_finish_time"]
                    for row in rows
                ],
                [row["pd3qn_p95_finish_time"] for row in rows],
                lower_is_better=True,
            ),
            "cache_hit_rate": paired_statistics(
                [
                    row[f"{reference_prefix}_cache_hit_rate"]
                    for row in rows
                ],
                [row["pd3qn_cache_hit_rate"] for row in rows],
                lower_is_better=False,
            ),
        }

    summary = {
        "status": "complete",
        "comparison": "pd3qn_vs_converged_daoc_and_previous_our",
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "integrity": {
            "seeds": len(rows),
            "validation_scenarios_per_seed": len(
                artifacts["daoc"]["validation"]
            ),
            "evaluation_scenarios_per_seed": len(
                artifacts["daoc"]["eval_rows"]
            ),
            "all_runs_converged": True,
            "all_validation_fingerprints_match": True,
            "all_evaluation_fingerprints_match": True,
            "all_evaluations_frozen": True,
            "physical_environment_match": True,
            "evaluation_protocol_match": True,
        },
        "method_aggregates": {
            "daoc": aggregate_method(rows, "daoc"),
            "previous_our": aggregate_method(rows, "previous_our"),
            "pd3qn": aggregate_method(rows, "pd3qn"),
        },
        "comparisons": comparisons,
    }

    write_csv(pd3qn_suite / "pd3qn_main_per_seed.csv", rows)
    (pd3qn_suite / "pd3qn_main_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_comparison(
        pd3qn_suite / "pd3qn_main_comparison",
        rows,
        summary,
    )
    write_report(
        pd3qn_suite / "PD3QN_MAIN_EXPERIMENT_REPORT.md",
        summary,
        rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
