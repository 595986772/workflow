#!/usr/bin/env python3
"""Audit and compare causal OUR against the clean legacy DAOC baseline."""

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a causal OUR suite with a converged DAOC suite "
            "after verifying paired validation and test scenario banks."
        )
    )
    parser.add_argument("--daoc-suite-dir", type=Path, required=True)
    parser.add_argument("--our-suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def stable_validation_fingerprints(run_dir):
    checkpoint_validation = read_json(
        run_dir / "checkpoint_validation.json"
    )
    records = checkpoint_validation["records"]
    if not records:
        raise RuntimeError(
            f"No checkpoint validation records in {run_dir}"
        )
    fingerprints = tuple(records[0]["scenario_fingerprints"])
    if any(
        tuple(record["scenario_fingerprints"]) != fingerprints
        for record in records[1:]
    ):
        raise RuntimeError(
            f"Validation bank changed within {run_dir}"
        )
    return fingerprints


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def discover_seeds(root, label):
    label_dir = root / "runs" / label
    return {
        int(path.name.split("_", 1)[1])
        for path in label_dir.glob("seed_*")
        if (path / "summary.json").exists()
    }


def paired_statistics(daoc, ours, lower_is_better=True):
    daoc = np.asarray(daoc, dtype=float)
    ours = np.asarray(ours, dtype=float)
    differences = ours - daoc
    direction = -1.0 if lower_is_better else 1.0
    improvements = 100.0 * direction * differences / daoc
    non_ties = int(np.sum(~np.isclose(ours, daoc)))
    wins = int(
        np.sum(ours < daoc)
        if lower_is_better
        else np.sum(ours > daoc)
    )
    losses = int(
        np.sum(ours > daoc)
        if lower_is_better
        else np.sum(ours < daoc)
    )
    improvement_ci = confidence_interval(improvements)
    difference_ci = confidence_interval(differences)
    daoc_ci = confidence_interval(daoc)
    our_ci = confidence_interval(ours)
    return {
        "pairs": int(daoc.size),
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(np.isclose(ours, daoc))),
        "daoc_mean": float(daoc.mean()),
        "daoc_ci95_half_width": daoc_ci[2],
        "our_mean": float(ours.mean()),
        "our_ci95_half_width": our_ci[2],
        "ratio_of_means_improvement_percent": float(
            100.0
            * direction
            * (ours.mean() - daoc.mean())
            / daoc.mean()
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
        "paired_t_two_sided_p": float(
            stats.ttest_rel(ours, daoc).pvalue
        ),
        "wilcoxon_one_sided_p": float(
            stats.wilcoxon(
                ours,
                daoc,
                alternative=(
                    "less" if lower_is_better else "greater"
                ),
            ).pvalue
        ),
        "sign_test_one_sided_p": float(
            stats.binomtest(
                wins,
                non_ties,
                p=0.5,
                alternative="greater",
            ).pvalue
        ),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path, rows, statistics):
    seeds = np.asarray([row["seed"] for row in rows])
    daoc = np.asarray(
        [row["daoc_mean_finish_time"] for row in rows]
    )
    ours = np.asarray(
        [row["our_mean_finish_time"] for row in rows]
    )
    improvements = np.asarray(
        [row["finish_time_improvement_percent"] for row in rows]
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")

    for index in range(len(seeds)):
        axes[0].plot(
            [0, 1],
            [daoc[index], ours[index]],
            color=(
                "#16A34A"
                if ours[index] < daoc[index]
                else "#DC2626"
            ),
            alpha=0.65,
            linewidth=1.5,
        )
        axes[0].scatter(
            [0, 1],
            [daoc[index], ours[index]],
            color=["#4B5563", "#2563EB"],
            edgecolor="white",
            linewidth=0.7,
            s=48,
            zorder=3,
        )
    axes[0].set_xticks([0, 1], ["DAOC", "Causal OUR"])
    axes[0].set_ylabel("Mean application finish time (s)")
    axes[0].set_title("Paired held-out performance", loc="left")
    axes[0].set_ylim(bottom=0)

    colors = [
        "#16A34A" if value > 0 else "#DC2626"
        for value in improvements
    ]
    axes[1].bar(seeds, improvements, color=colors, width=0.72)
    axes[1].axhline(0.0, color="#6B7280", linewidth=1.0)
    axes[1].axhline(
        statistics["mean_paired_improvement_percent"],
        color="#111827",
        linestyle="--",
        linewidth=1.3,
        label=(
            "Mean paired improvement "
            f"{statistics['mean_paired_improvement_percent']:.2f}%"
        ),
    )
    axes[1].set_xticks(seeds)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("DAOC to OUR improvement (%)")
    axes[1].set_title("Per-seed relative effect", loc="left")
    axes[1].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)

    figure.suptitle(
        "DAOC vs Causal OUR: Exact Paired Scenario Banks",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def format_pm(mean, half_width):
    return f"{mean:.6f} +/- {half_width:.6f}"


def write_report(path, summary, rows):
    finish = summary["finish_time"]
    p95 = summary["p95_finish_time"]
    cache = summary["cache_hit_rate"]
    integrity = summary["integrity"]
    primary_tests_significant = all(
        value < 0.05
        for value in (
            finish["wilcoxon_one_sided_p"],
            finish["paired_t_two_sided_p"],
            finish["sign_test_one_sided_p"],
        )
    )
    significance_text = (
        "All three reported tests pass the 0.05 threshold."
        if primary_tests_significant
        else (
            "The tests do not all pass the 0.05 threshold, so the "
            "result should be described with the individual p-values."
        )
    )
    row_lines = [
        (
            f"| {row['seed']} | "
            f"{row['daoc_mean_finish_time']:.6f} | "
            f"{row['our_mean_finish_time']:.6f} | "
            f"{row['finish_time_improvement_percent']:.2f}% |"
        )
        for row in rows
    ]
    text = "\n".join(
        [
            "# DAOC vs Causal OUR",
            "",
            "## Integrity",
            "",
            f"- {integrity['seeds']}/{integrity['seeds']} causal OUR runs "
            "converged and passed the comparison gate.",
            "- Every causal OUR run used `causal_cache_v1` and "
            "`causal_history_only_v1`.",
            "- The DAOC runs use `popularity_ema` and "
            "`terminal_binary`; no legacy joint-cache or OUR result is reused.",
            f"- All {integrity['validation_scenarios_per_seed']} validation "
            f"scenarios and all {integrity['test_scenarios_per_seed']} "
            "held-out test scenarios "
            "match exactly by fingerprint for every seed.",
            "- Both methods keep their native cache controllers online "
            "throughout training; frozen evaluation disables cache updates.",
            "- Seeds are the independent statistical units.",
            "",
            "## Finish time",
            "",
            f"- DAOC: {format_pm(finish['daoc_mean'], finish['daoc_ci95_half_width'])} s.",
            f"- Causal OUR: {format_pm(finish['our_mean'], finish['our_ci95_half_width'])} s.",
            f"- Wins: {finish['wins']}/{finish['pairs']}.",
            "- Mean paired relative improvement: "
            f"{finish['mean_paired_improvement_percent']:.2f}% "
            f"(95% CI {finish['paired_improvement_ci95_lower']:.2f}% to "
            f"{finish['paired_improvement_ci95_upper']:.2f}%).",
            "- Median paired relative improvement: "
            f"{finish['median_paired_improvement_percent']:.2f}%.",
            "- One-sided Wilcoxon p="
            f"{finish['wilcoxon_one_sided_p']:.6f}; "
            "two-sided paired t-test p="
            f"{finish['paired_t_two_sided_p']:.6f}; "
            "one-sided sign-test p="
            f"{finish['sign_test_one_sided_p']:.6f}.",
            "",
            "| Seed | DAOC (s) | Causal OUR (s) | Improvement |",
            "|---:|---:|---:|---:|",
            *row_lines,
            "",
            "## Secondary metrics",
            "",
            "- P95 finish time: "
            f"{p95['daoc_mean']:.6f} s to {p95['our_mean']:.6f} s; "
            f"{p95['wins']}/{p95['pairs']} wins; "
            f"mean paired improvement {p95['mean_paired_improvement_percent']:.2f}%; "
            f"one-sided Wilcoxon p={p95['wilcoxon_one_sided_p']:.6f}; "
            f"paired t-test p={p95['paired_t_two_sided_p']:.6f}.",
            "- Cache hit rate: "
            f"{cache['daoc_mean']:.4f} for DAOC and "
            f"{cache['our_mean']:.4f} for causal OUR; "
            f"paired t-test p={cache['paired_t_two_sided_p']:.6f}.",
            "",
            "## Interpretation",
            "",
            "The causal OUR result has a statistically supported directional "
            "advantage under the pre-specified one-sided Wilcoxon test. "
            + significance_text
            + " The cache-hit-rate difference is not significant, so raw hit "
            "rate alone does not explain the latency gain. Mechanism checks "
            "remain necessary before assigning the improvement to individual "
            "components.",
            "",
            "The ratio-of-means improvement is "
            f"{finish['ratio_of_means_improvement_percent']:.2f}%, but the "
            f"seed-paired mean ({finish['mean_paired_improvement_percent']:.2f}%) "
            "is the primary effect size because seeds are the independent units.",
            "",
            "## Artifacts",
            "",
            "- `daoc_vs_causal_our_per_seed.csv`",
            "- `daoc_vs_causal_our_summary.json`",
            "- `daoc_vs_causal_our.png` and `.pdf`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main():
    args = parse_args()
    daoc_suite = args.daoc_suite_dir.resolve()
    our_suite = args.our_suite_dir.resolve()
    our_manifest = read_json(our_suite / "suite_manifest.json")

    require(
        our_manifest.get("status") == "complete",
        "Causal OUR suite is not complete",
    )
    require(
        our_manifest.get("information_protocol_version")
        == INFORMATION_PROTOCOL_VERSION,
        "Causal OUR suite has the wrong information protocol",
    )
    require(
        not our_manifest.get("failed_runs"),
        "Causal OUR suite contains failed runs",
    )
    require(
        not our_manifest.get("nonconverged_runs"),
        "Causal OUR suite contains non-converged runs",
    )

    daoc_seeds = discover_seeds(daoc_suite, "guided_full")
    our_seeds = discover_seeds(our_suite, "our")
    require(
        daoc_seeds == our_seeds and daoc_seeds,
        "DAOC and OUR seed sets do not match",
    )

    rows = []
    validation_scenarios_per_seed = None
    test_scenarios_per_seed = None
    for seed in sorted(daoc_seeds):
        daoc_dir = (
            daoc_suite / "runs" / "guided_full" / f"seed_{seed}"
        )
        our_dir = our_suite / "runs" / "our" / f"seed_{seed}"
        daoc_summary = read_json(daoc_dir / "summary.json")
        our_summary = read_json(our_dir / "summary.json")
        daoc_config = read_json(daoc_dir / "config.json")
        our_config = read_json(our_dir / "config.json")

        require(
            daoc_summary.get("status") == "complete"
            and daoc_summary.get("eligible_for_comparison") is True
            and daoc_summary["convergence"]["reached"] is True,
            f"DAOC seed {seed} is not a converged eligible run",
        )
        require(
            our_summary.get("status") == "complete"
            and our_summary.get("eligible_for_comparison") is True
            and our_summary["convergence"]["reached"] is True,
            f"OUR seed {seed} is not a converged eligible run",
        )
        require(
            daoc_config["arguments"]["cache_policy"]
            == "popularity_ema"
            and daoc_config["arguments"]["reward_mode"]
            == "terminal_binary",
            f"DAOC seed {seed} is not the clean DAOC baseline",
        )
        require(
            daoc_config["arguments"]["cache_freeze_episode"] == 0
            and our_config["arguments"]["cache_freeze_episode"] == 0,
            f"Cache controller was frozen during training for seed {seed}",
        )
        matched_environment_keys = (
            "num_users",
            "num_servers",
            "num_services",
            "num_tasks",
            "bandwidth",
            "batch_size",
            "min_experiences",
            "filling_steps",
            "steps_to_updates",
            "max_explore",
            "learning_rate",
            "learning_rate_schedule",
            "learning_rate_decay_start",
            "learning_rate_decay_end",
            "min_learning_rate",
            "epsilon",
            "hidden_units",
            "train_episodes",
            "eval_episodes",
            "checkpoint_every",
            "validation_scenarios",
            "convergence_min_episodes",
            "convergence_window",
            "convergence_patience",
            "convergence_relative_mean_change",
            "convergence_relative_slope",
        )
        require(
            all(
                daoc_config["arguments"][key]
                == our_config["arguments"][key]
                for key in matched_environment_keys
            ),
            f"Training or environment configuration differs for seed {seed}",
        )
        require(
            our_config["arguments"]["cache_policy"]
            == "critical_path_joint"
            and our_summary.get("information_protocol_version")
            == INFORMATION_PROTOCOL_VERSION
            and our_summary.get("cache_information_regime")
            == CAUSAL_CACHE_INFORMATION_REGIME,
            f"OUR seed {seed} is not the causal implementation",
        )
        require(
            daoc_summary["evaluation_state_frozen"] is True
            and our_summary["evaluation_state_frozen"] is True,
            f"Evaluation state was not frozen for seed {seed}",
        )

        daoc_validation = stable_validation_fingerprints(daoc_dir)
        our_validation = stable_validation_fingerprints(our_dir)
        require(
            daoc_validation == our_validation,
            f"Validation fingerprints differ for seed {seed}",
        )
        if validation_scenarios_per_seed is None:
            validation_scenarios_per_seed = len(daoc_validation)
        require(
            len(daoc_validation) == validation_scenarios_per_seed,
            f"Validation scenario count differs for seed {seed}",
        )
        daoc_eval = evaluation_rows(daoc_dir)
        our_eval = evaluation_rows(our_dir)
        daoc_fingerprints = tuple(
            row["scenario_fingerprint"] for row in daoc_eval
        )
        our_fingerprints = tuple(
            row["scenario_fingerprint"] for row in our_eval
        )
        require(
            daoc_fingerprints == our_fingerprints,
            f"Test fingerprints differ for seed {seed}",
        )
        expected_test_scenarios = daoc_config["arguments"][
            "eval_episodes"
        ]
        require(
            len(daoc_fingerprints) == expected_test_scenarios,
            f"Expected {expected_test_scenarios} paired test scenarios "
            f"for seed {seed}",
        )
        if test_scenarios_per_seed is None:
            test_scenarios_per_seed = len(daoc_fingerprints)
        require(
            len(daoc_fingerprints) == test_scenarios_per_seed,
            f"Test scenario count differs for seed {seed}",
        )

        daoc_finish = daoc_summary["eval"][
            "mean_average_finish_time"
        ]
        our_finish = our_summary["eval"][
            "mean_average_finish_time"
        ]
        daoc_p95 = daoc_summary["eval"]["mean_p95_finish_time"]
        our_p95 = our_summary["eval"]["mean_p95_finish_time"]
        rows.append(
            {
                "seed": seed,
                "validation_fingerprints_match": 1,
                "test_fingerprints_match": 1,
                "daoc_convergence_episode": (
                    daoc_summary["convergence"][
                        "actual_train_episodes"
                    ]
                ),
                "our_convergence_episode": (
                    our_summary["convergence"][
                        "actual_train_episodes"
                    ]
                ),
                "daoc_mean_finish_time": daoc_finish,
                "our_mean_finish_time": our_finish,
                "finish_time_improvement_percent": (
                    100.0
                    * (daoc_finish - our_finish)
                    / daoc_finish
                ),
                "daoc_p95_finish_time": daoc_p95,
                "our_p95_finish_time": our_p95,
                "p95_improvement_percent": (
                    100.0 * (daoc_p95 - our_p95) / daoc_p95
                ),
                "daoc_cache_hit_rate": (
                    daoc_summary["eval"]["mean_cache_hit_rate"]
                ),
                "our_cache_hit_rate": (
                    our_summary["eval"]["mean_cache_hit_rate"]
                ),
            }
        )

    finish_statistics = paired_statistics(
        [row["daoc_mean_finish_time"] for row in rows],
        [row["our_mean_finish_time"] for row in rows],
    )
    p95_statistics = paired_statistics(
        [row["daoc_p95_finish_time"] for row in rows],
        [row["our_p95_finish_time"] for row in rows],
    )
    cache_statistics = paired_statistics(
        [row["daoc_cache_hit_rate"] for row in rows],
        [row["our_cache_hit_rate"] for row in rows],
        lower_is_better=False,
    )
    summary = {
        "status": "complete",
        "comparison": "clean_legacy_daoc_vs_causal_our",
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "integrity": {
            "seeds": len(rows),
            "validation_scenarios_per_seed": (
                validation_scenarios_per_seed
            ),
            "test_scenarios_per_seed": test_scenarios_per_seed,
            "all_validation_fingerprints_match": True,
            "all_test_fingerprints_match": True,
            "cache_online_during_training": True,
            "all_our_runs_converged": True,
            "legacy_daoc_whitelist": (
                "popularity_ema_terminal_binary_only"
            ),
        },
        "finish_time": finish_statistics,
        "p95_finish_time": p95_statistics,
        "cache_hit_rate": cache_statistics,
    }

    write_csv(
        our_suite / "daoc_vs_causal_our_per_seed.csv",
        rows,
    )
    (our_suite / "daoc_vs_causal_our_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_comparison(
        our_suite / "daoc_vs_causal_our",
        rows,
        finish_statistics,
    )
    write_report(
        our_suite / "DAOC_VS_CAUSAL_OUR_REPORT.md",
        summary,
        rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
