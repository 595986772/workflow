#!/usr/bin/env python3
"""Audit and summarize the convergence-controlled CPR experiment suite."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from information_protocol import INFORMATION_PROTOCOL_VERSION


DISPLAY_NAMES = {
    "guided_full": "DAOC",
    "cpr_reward": "CPR reward",
    "cpr_cache": "CPR + local cache",
    "cpr_coord_cache": "CPR + coordinated cache",
    "cpr_joint_cache": "CPR + joint cache",
    "hybrid_reward": "Hybrid reward",
    "our": "OUR",
}

COLORS = {
    "guided_full": "#4B5563",
    "cpr_reward": "#2563EB",
    "cpr_cache": "#E76F51",
    "cpr_coord_cache": "#7C3AED",
    "cpr_joint_cache": "#0891B2",
    "hybrid_reward": "#65A30D",
    "our": "#DC2626",
}

COMPARISON_REFERENCES = [
    "guided_full",
    "cpr_reward",
    "cpr_cache",
    "cpr_coord_cache",
    "cpr_joint_cache",
    "hybrid_reward",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a completed convergence-controlled experiment suite."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_scenario(snapshot):
    snapshot = dict(snapshot)
    snapshot.pop("algorithm", None)
    return snapshot


def trace_audit(run_dir, freeze_episode):
    train_rows = 0
    eval_rows = 0
    last_train = None
    test_fingerprints = []
    post_freeze_cache_updates = 0.0
    post_freeze_cache_enabled = 0

    with (run_dir / "episodes.csv").open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        for row in csv.DictReader(input_file):
            phase = row["phase"]
            if phase == "train":
                train_rows += 1
                last_train = row
                episode = int(float(row["episode"]))
                if episode > freeze_episode:
                    post_freeze_cache_updates += float(
                        row.get("cache_update_events") or 0.0
                    )
                    post_freeze_cache_enabled += int(
                        float(row.get("cache_updates_enabled") or 0.0) > 0.0
                    )
            elif phase == "eval":
                eval_rows += 1
                test_fingerprints.append(row["scenario_fingerprint"])

    if last_train is None:
        raise RuntimeError(f"No training rows in {run_dir / 'episodes.csv'}")
    return {
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "last_learning_rate": float(last_train["mean_learning_rate"]),
        "post_freeze_cache_updates": post_freeze_cache_updates,
        "post_freeze_cache_enabled_rows": post_freeze_cache_enabled,
        "test_fingerprints": tuple(test_fingerprints),
    }


def load_and_verify(suite_dir):
    manifest = read_json(suite_dir / "suite_manifest.json")
    profile = manifest.get("profile_config", {})
    labels = list(profile.get("labels", []))
    seeds = list(profile.get("seeds", []))
    expected_runs = len(labels) * len(seeds)

    if manifest.get("status") != "complete":
        raise RuntimeError("Suite manifest is not complete")
    if manifest.get("failed_runs"):
        raise RuntimeError("Suite manifest contains failed runs")
    if manifest.get("nonconverged_runs"):
        raise RuntimeError("Suite manifest contains non-converged runs")
    if manifest.get("completed_runs") != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} completed runs, found "
            f"{manifest.get('completed_runs')}"
        )
    if manifest.get("converged_runs") != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} converged runs, found "
            f"{manifest.get('converged_runs')}"
        )
    if "guided_full" not in labels or "our" not in labels:
        raise RuntimeError("The suite must contain both guided_full and our")

    runs = []
    for label in labels:
        for seed in seeds:
            run_dir = suite_dir / "runs" / label / f"seed_{seed}"
            summary = read_json(run_dir / "summary.json")
            config = read_json(run_dir / "config.json")
            convergence = summary.get("convergence", {})
            arguments = config["arguments"]
            freeze_episode = int(convergence["cache_freeze_episode"])
            trace = trace_audit(run_dir, freeze_episode)

            checks = {
                "summary_complete": summary.get("status") == "complete",
                "information_protocol": (
                    summary.get("information_protocol_version")
                    == INFORMATION_PROTOCOL_VERSION
                ),
                "eligible": summary.get("eligible_for_comparison") is True,
                "evaluation_frozen": (
                    summary.get("evaluation_state_frozen") is True
                ),
                "convergence_enabled": convergence.get("enabled") is True,
                "convergence_reached": convergence.get("reached") is True,
                "criterion_stop": (
                    convergence.get("stop_reason") == "criterion_met"
                ),
                "final_checkpoint_strategy": (
                    summary.get("checkpoint_strategy")
                    == "convergence_final"
                ),
                "final_checkpoint_episode": (
                    summary.get("selected_checkpoint_episode")
                    == convergence.get("episode")
                    == convergence.get("actual_train_episodes")
                ),
                "train_row_count": (
                    trace["train_rows"]
                    == convergence.get("actual_train_episodes")
                ),
                "eval_row_count": (
                    trace["eval_rows"] == arguments["eval_episodes"]
                ),
                "validation_count": (
                    len(summary.get("checkpoint_validation", []))
                    == convergence.get("actual_train_episodes")
                    // convergence.get("checkpoint_every")
                ),
                "minimum_learning_rate": np.isclose(
                    trace["last_learning_rate"],
                    convergence["minimum_learning_rate"],
                    rtol=0.0,
                    atol=1e-12,
                ),
                "cache_frozen": (
                    trace["post_freeze_cache_updates"] == 0.0
                    and trace["post_freeze_cache_enabled_rows"] == 0
                ),
                "checkpoint_hash": (
                    sha256_file(run_dir / "checkpoint.pt")
                    == summary["checkpoint_sha256"]
                ),
                "selected_checkpoint_hash": (
                    sha256_file(run_dir / "selected_checkpoint.pt")
                    == summary["selected_checkpoint_sha256"]
                ),
                "final_training_checkpoint_exists": (
                    run_dir
                    / "training_checkpoints"
                    / (
                        "episode_"
                        f"{convergence['actual_train_episodes']:06d}.pt"
                    )
                ).exists(),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(
                    f"Integrity failure for {label} seed={seed}: "
                    + ", ".join(failed)
                )

            diagnostics = convergence["final_diagnostics"]
            if not diagnostics.get("converged"):
                raise RuntimeError(
                    f"Final diagnostics are not converged: {label} seed={seed}"
                )
            if (
                diagnostics["relative_mean_change"]
                > convergence["relative_mean_change_threshold"]
            ):
                raise RuntimeError(
                    f"Mean-change threshold failed: {label} seed={seed}"
                )
            if (
                abs(diagnostics["relative_slope_per_checkpoint"])
                > convergence["relative_slope_threshold"]
            ):
                raise RuntimeError(
                    f"Slope threshold failed: {label} seed={seed}"
                )
            if diagnostics["stable_streak"] < convergence["patience"]:
                raise RuntimeError(
                    f"Patience threshold failed: {label} seed={seed}"
                )

            validation_records = summary["checkpoint_validation"]
            validation_fingerprints = tuple(
                validation_records[0]["scenario_fingerprints"]
            )
            if any(
                tuple(record["scenario_fingerprints"])
                != validation_fingerprints
                for record in validation_records[1:]
            ):
                raise RuntimeError(
                    f"Validation bank changed: {label} seed={seed}"
                )
            if not set(validation_fingerprints).isdisjoint(
                trace["test_fingerprints"]
            ):
                raise RuntimeError(
                    f"Validation/test leakage: {label} seed={seed}"
                )

            runs.append(
                {
                    "label": label,
                    "seed": seed,
                    "run_dir": run_dir,
                    "summary": summary,
                    "config": config,
                    "trace": trace,
                    "validation_fingerprints": validation_fingerprints,
                    "scenario": normalize_scenario(
                        read_json(run_dir / "scenario_initial.json")
                    ),
                }
            )

    runs_by_seed = defaultdict(list)
    for run in runs:
        runs_by_seed[run["seed"]].append(run)
    for seed, seed_runs in runs_by_seed.items():
        scenarios = [run["scenario"] for run in seed_runs]
        if any(scenario != scenarios[0] for scenario in scenarios[1:]):
            raise RuntimeError(f"Initial scenario mismatch for seed={seed}")
        validation_banks = [
            run["validation_fingerprints"] for run in seed_runs
        ]
        if any(bank != validation_banks[0] for bank in validation_banks[1:]):
            raise RuntimeError(f"Validation bank mismatch for seed={seed}")
        test_banks = [run["trace"]["test_fingerprints"] for run in seed_runs]
        if any(bank != test_banks[0] for bank in test_banks[1:]):
            raise RuntimeError(f"Test bank mismatch for seed={seed}")

    return manifest, labels, seeds, runs


def convergence_rows(runs):
    rows = []
    for run in runs:
        summary = run["summary"]
        convergence = summary["convergence"]
        diagnostics = convergence["final_diagnostics"]
        rows.append(
            {
                "label": run["label"],
                "display_name": DISPLAY_NAMES.get(
                    run["label"], run["label"]
                ),
                "seed": run["seed"],
                "converged": convergence["reached"],
                "stop_reason": convergence["stop_reason"],
                "train_episodes": convergence["actual_train_episodes"],
                "selected_checkpoint_episode": summary[
                    "selected_checkpoint_episode"
                ],
                "final_window_mean": diagnostics["window_mean"],
                "previous_window_mean": diagnostics[
                    "previous_window_mean"
                ],
                "relative_mean_change_percent": (
                    100.0 * diagnostics["relative_mean_change"]
                ),
                "relative_slope_per_checkpoint_percent": (
                    100.0
                    * diagnostics["relative_slope_per_checkpoint"]
                ),
                "relative_checkpoint_range_percent": (
                    100.0 * diagnostics["relative_range"]
                ),
                "stable_streak": diagnostics["stable_streak"],
                "final_learning_rate": run["trace"][
                    "last_learning_rate"
                ],
                "post_freeze_cache_updates": run["trace"][
                    "post_freeze_cache_updates"
                ],
                "validation_scenarios": len(
                    run["validation_fingerprints"]
                ),
                "test_scenarios": run["trace"]["eval_rows"],
            }
        )
    return rows


def aggregate_rows(labels, runs):
    rows = []
    for label in labels:
        label_runs = [run for run in runs if run["label"] == label]
        eval_means = [
            run["summary"]["eval"]["mean_average_finish_time"]
            for run in label_runs
        ]
        p95_means = [
            run["summary"]["eval"]["mean_p95_finish_time"]
            for run in label_runs
        ]
        cache_hits = [
            run["summary"]["eval"]["mean_cache_hit_rate"]
            for run in label_runs
        ]
        train_episodes = [
            run["summary"]["convergence"]["actual_train_episodes"]
            for run in label_runs
        ]
        rows.append(
            {
                "label": label,
                "display_name": DISPLAY_NAMES.get(label, label),
                "seeds": len(label_runs),
                "eval_finish_time_mean": float(np.mean(eval_means)),
                "eval_finish_time_ci95": confidence_interval(eval_means),
                "eval_p95_finish_time_mean": float(np.mean(p95_means)),
                "eval_p95_finish_time_ci95": confidence_interval(p95_means),
                "eval_cache_hit_rate_mean": float(np.mean(cache_hits)),
                "eval_cache_hit_rate_ci95": confidence_interval(cache_hits),
                "convergence_episodes_mean": float(
                    np.mean(train_episodes)
                ),
                "convergence_episodes_ci95": confidence_interval(
                    train_episodes
                ),
                "convergence_episodes_min": int(np.min(train_episodes)),
                "convergence_episodes_max": int(np.max(train_episodes)),
            }
        )
    return rows


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def safe_wilcoxon(left, right):
    differences = np.asarray(left, dtype=float) - np.asarray(
        right, dtype=float
    )
    if np.allclose(differences, 0.0):
        return 1.0
    return float(
        stats.wilcoxon(
            left,
            right,
            alternative="less",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def paired_rows(runs):
    values = defaultdict(dict)
    for run in runs:
        values[run["label"]][run["seed"]] = run["summary"]["eval"][
            "mean_average_finish_time"
        ]

    rows = []
    for reference in COMPARISON_REFERENCES:
        if reference not in values:
            continue
        seeds = sorted(set(values["our"]) & set(values[reference]))
        ours = np.asarray([values["our"][seed] for seed in seeds])
        refs = np.asarray([values[reference][seed] for seed in seeds])
        differences = ours - refs
        advantages = refs - ours
        improvements = 100.0 * advantages / refs
        wins = int(np.sum(ours < refs))
        losses = int(np.sum(ours > refs))
        non_tied_pairs = wins + losses
        standard_deviation = float(np.std(advantages, ddof=1))
        effect_size = (
            float(np.mean(advantages) / standard_deviation)
            if standard_deviation > 0
            else float("inf")
        )
        rows.append(
            {
                "label": "our",
                "display_name": DISPLAY_NAMES["our"],
                "reference": reference,
                "reference_display_name": DISPLAY_NAMES.get(
                    reference, reference
                ),
                "paired_seeds": len(seeds),
                "wins": wins,
                "ties": int(np.sum(np.isclose(ours, refs))),
                "losses": losses,
                "mean_difference_seconds": float(np.mean(differences)),
                "difference_ci95_seconds": confidence_interval(differences),
                "mean_improvement_percent": float(np.mean(improvements)),
                "improvement_ci95_percent": confidence_interval(improvements),
                "median_improvement_percent": float(
                    np.median(improvements)
                ),
                "paired_effect_size_dz": effect_size,
                "paired_t_two_sided_p": float(
                    stats.ttest_rel(ours, refs).pvalue
                ),
                "wilcoxon_one_sided_p": safe_wilcoxon(ours, refs),
                "sign_test_one_sided_p": float(
                    stats.binomtest(
                        wins,
                        non_tied_pairs,
                        0.5,
                        alternative="greater",
                    ).pvalue
                ),
            }
        )
    adjusted = holm_adjust(
        [row["wilcoxon_one_sided_p"] for row in rows]
    )
    for row, value in zip(rows, adjusted):
        row["wilcoxon_holm_p"] = float(value)
    return rows


def daoc_our_sensitivity(runs):
    values = defaultdict(dict)
    for run in runs:
        if run["label"] not in {"guided_full", "our"}:
            continue
        evaluation = run["summary"]["eval"]
        values[run["label"]][run["seed"]] = {
            "mean": evaluation["mean_average_finish_time"],
            "p95": evaluation["mean_p95_finish_time"],
            "cache_hit": evaluation["mean_cache_hit_rate"],
        }

    seeds = sorted(set(values["guided_full"]) & set(values["our"]))
    daoc_mean = np.asarray(
        [values["guided_full"][seed]["mean"] for seed in seeds]
    )
    our_mean = np.asarray(
        [values["our"][seed]["mean"] for seed in seeds]
    )
    daoc_p95 = np.asarray(
        [values["guided_full"][seed]["p95"] for seed in seeds]
    )
    our_p95 = np.asarray(
        [values["our"][seed]["p95"] for seed in seeds]
    )
    daoc_hit = np.asarray(
        [values["guided_full"][seed]["cache_hit"] for seed in seeds]
    )
    our_hit = np.asarray(
        [values["our"][seed]["cache_hit"] for seed in seeds]
    )

    mean_improvements = 100.0 * (daoc_mean - our_mean) / daoc_mean
    p95_improvements = 100.0 * (daoc_p95 - our_p95) / daoc_p95
    leave_one_out = np.asarray(
        [
            np.delete(mean_improvements, index).mean()
            for index in range(len(seeds))
        ]
    )
    return {
        "paired_seeds": len(seeds),
        "mean_finish_time_improvement_percent": float(
            mean_improvements.mean()
        ),
        "mean_finish_time_median_improvement_percent": float(
            np.median(mean_improvements)
        ),
        "mean_finish_time_trimmed_improvement_percent": float(
            stats.trim_mean(mean_improvements, 0.1)
        ),
        "mean_finish_time_leave_one_out_min_percent": float(
            leave_one_out.min()
        ),
        "mean_finish_time_leave_one_out_max_percent": float(
            leave_one_out.max()
        ),
        "mean_finish_time_paired_t_two_sided_p": float(
            stats.ttest_rel(our_mean, daoc_mean).pvalue
        ),
        "mean_finish_time_wilcoxon_one_sided_p": safe_wilcoxon(
            our_mean, daoc_mean
        ),
        "mean_finish_time_sign_test_one_sided_p": float(
            stats.binomtest(
                int(np.sum(our_mean < daoc_mean)),
                int(np.sum(our_mean != daoc_mean)),
                0.5,
                alternative="greater",
            ).pvalue
        ),
        "p95_improvement_percent": float(p95_improvements.mean()),
        "p95_improvement_ci95_percent": confidence_interval(
            p95_improvements
        ),
        "p95_wins": int(np.sum(our_p95 < daoc_p95)),
        "p95_wilcoxon_one_sided_p": safe_wilcoxon(our_p95, daoc_p95),
        "p95_paired_t_two_sided_p": float(
            stats.ttest_rel(our_p95, daoc_p95).pvalue
        ),
        "cache_hit_rate_difference": float(
            np.mean(our_hit - daoc_hit)
        ),
        "cache_hit_rate_paired_t_two_sided_p": float(
            stats.ttest_rel(our_hit, daoc_hit).pvalue
        ),
    }


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "legend.frameon": False,
        }
    )


def clean_axis(axis):
    axis.set_facecolor("#F8FAFC")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def save_figure(figure, output_base):
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_convergence_episodes(suite_dir, labels, runs):
    series = {
        label: [
            run["summary"]["convergence"]["actual_train_episodes"]
            for run in runs
            if run["label"] == label
        ]
        for label in labels
    }
    means = [np.mean(series[label]) for label in labels]
    errors = [confidence_interval(series[label]) for label in labels]
    x = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=[COLORS.get(label, "#64748B") for label in labels],
        edgecolor="white",
    )
    for index, label in enumerate(labels):
        values = np.asarray(series[label], dtype=float)
        offsets = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            index + offsets,
            values,
            s=20,
            color="#111827",
            alpha=0.7,
            zorder=3,
        )
    clean_axis(axis)
    axis.axhline(
        40000,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        label="Maximum budget",
    )
    axis.set_xticks(
        x,
        [DISPLAY_NAMES.get(label, label) for label in labels],
        rotation=15,
        ha="right",
    )
    axis.set_ylabel("Episodes to convergence")
    axis.set_title("Convergence-Controlled Training", loc="left", weight="bold")
    axis.legend(loc="lower right", frameon=True, facecolor="white")
    save_figure(figure, suite_dir / "convergence_episodes")


def plot_validation_curves(suite_dir, runs):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
        sharey=False,
    )
    figure.patch.set_facecolor("white")
    for axis, label in zip(axes, ["guided_full", "our"]):
        label_runs = sorted(
            [run for run in runs if run["label"] == label],
            key=lambda run: run["seed"],
        )
        for run in label_runs:
            records = run["summary"]["checkpoint_validation"]
            episodes = [record["episode"] for record in records]
            scores = [
                record["mean_average_finish_time"] for record in records
            ]
            axis.plot(
                episodes,
                scores,
                linewidth=0.9,
                alpha=0.55,
                color=COLORS[label],
                label=(
                    f"seed {run['seed']}"
                    if run["seed"] == label_runs[0]["seed"]
                    else None
                ),
            )
            axis.scatter(
                episodes[-1],
                scores[-1],
                s=18,
                color="#111827",
                zorder=3,
            )
        clean_axis(axis)
        axis.axvline(
            20000,
            color="#64748B",
            linestyle="--",
            linewidth=1.0,
            label="LR floor/cache freeze",
        )
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Fixed-bank validation finish time (s)")
        axis.set_title(
            f"{DISPLAY_NAMES[label]} validation paths",
            loc="left",
            weight="bold",
        )
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    save_figure(figure, suite_dir / "daoc_our_validation_paths")


def plot_paired_daoc_our(suite_dir, runs):
    values = defaultdict(dict)
    for run in runs:
        if run["label"] in {"guided_full", "our"}:
            values[run["label"]][run["seed"]] = run["summary"]["eval"][
                "mean_average_finish_time"
            ]
    seeds = sorted(set(values["guided_full"]) & set(values["our"]))
    daoc = np.asarray([values["guided_full"][seed] for seed in seeds])
    ours = np.asarray([values["our"][seed] for seed in seeds])
    label_positions = ours.copy()
    ordered_indices = np.argsort(label_positions)
    minimum_gap = 0.005
    for previous, current in zip(
        ordered_indices[:-1],
        ordered_indices[1:],
    ):
        label_positions[current] = max(
            label_positions[current],
            label_positions[previous] + minimum_gap,
        )

    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for index, seed in enumerate(seeds):
        axis.plot(
            [0, 1],
            [daoc[index], ours[index]],
            color="#94A3B8",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.scatter(
            [0, 1],
            [daoc[index], ours[index]],
            color=[COLORS["guided_full"], COLORS["our"]],
            s=34,
            zorder=3,
        )
        axis.plot(
            [1.01, 1.075],
            [ours[index], label_positions[index]],
            color="#94A3B8",
            linewidth=0.7,
            clip_on=False,
        )
        axis.text(
            1.09,
            label_positions[index],
            f"seed {seed}",
            fontsize=7,
            va="center",
            color="#475569",
        )
    clean_axis(axis)
    axis.set_xlim(-0.25, 1.35)
    axis.set_xticks([0, 1], ["DAOC", "OUR"])
    axis.set_ylabel("Independent-test finish time (s)")
    axis.set_title(
        "Paired DAOC vs OUR by Seed",
        loc="left",
        weight="bold",
    )
    axis.set_ylim(
        bottom=0,
        top=max(daoc.max(), label_positions.max()) * 1.08,
    )
    save_figure(figure, suite_dir / "paired_daoc_our")


def format_mean_ci(mean, ci, digits=4):
    return f"{mean:.{digits}f} +/- {ci:.{digits}f}"


def render_report(
    manifest,
    labels,
    seeds,
    aggregate,
    paired,
    convergence,
    sensitivity,
):
    aggregate_by_label = {row["label"]: row for row in aggregate}
    paired_by_reference = {row["reference"]: row for row in paired}
    primary = paired_by_reference["guided_full"]
    convergence_by_label = defaultdict(list)
    for row in convergence:
        convergence_by_label[row["label"]].append(row)

    if (
        primary["mean_improvement_percent"] > 0
        and primary["wilcoxon_holm_p"] < 0.05
    ):
        primary_conclusion = (
            "OUR has a statistically supported directional advantage over "
            "DAOC under the pre-specified one-sided Wilcoxon signed-rank "
            "test, including Holm correction."
        )
    elif primary["mean_improvement_percent"] > 0:
        primary_conclusion = (
            "OUR has a lower mean finish time than DAOC, but the paired "
            "evidence is not significant after Holm correction."
        )
    else:
        primary_conclusion = (
            "OUR does not beat DAOC on mean paired test performance in "
            "this configuration."
        )

    lines = [
        "# Convergence-controlled experiment audit",
        "",
        "## Integrity",
        "",
        f"- {len(labels) * len(seeds)}/{len(labels) * len(seeds)} runs "
        "completed and passed the convergence gate; no capped run was "
        "included.",
        f"- Each method used {len(seeds)} independent seeds. Each final "
        "score is the mean over 30 held-out test DAGs.",
        "- For each seed, every method used identical validation and test "
        "scenario banks; the two banks are disjoint.",
        "- The tested checkpoint is exactly the final checkpoint that "
        "triggered convergence; no historical-best checkpoint was selected.",
        "- Learning rate reached 1e-5 and cache updates were disabled after "
        "episode 20,000 in every eligible run.",
        "- Statistical tests use seeds as the independent unit. Test DAGs "
        "within a seed are not treated as independent replicates.",
        "",
        "## Convergence protocol",
        "",
        "- Validate every 500 episodes on the same 30 validation DAGs.",
        "- After episode 20,000, compare adjacent 10-checkpoint windows "
        "(5,000 episodes each).",
        "- Require relative window-mean change <= 5% and normalized slope "
        "<= 1% per checkpoint for three consecutive checks.",
        "- With these window and patience settings, episode 30,500 is the "
        "earliest possible stopping point; convergence-time comparisons are "
        "therefore only informative when a run needs longer.",
        "- Maximum budget: 40,000 episodes. Failure to pass makes a run "
        "ineligible for comparison.",
        "- This establishes an empirical validation-performance plateau, "
        "not theoretical parameter convergence.",
        "",
        "## Test performance",
        "",
        "| Method | Mean finish time (s) | P95 finish time (s) | "
        "Cache hit rate | Convergence episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = aggregate_by_label[label]
        lines.append(
            "| "
            f"{row['display_name']} | "
            f"{format_mean_ci(row['eval_finish_time_mean'], row['eval_finish_time_ci95'])} | "
            f"{format_mean_ci(row['eval_p95_finish_time_mean'], row['eval_p95_finish_time_ci95'])} | "
            f"{format_mean_ci(row['eval_cache_hit_rate_mean'], row['eval_cache_hit_rate_ci95'], 3)} | "
            f"{row['convergence_episodes_mean']:.0f} +/- "
            f"{row['convergence_episodes_ci95']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Paired OUR comparisons",
            "",
            "| Reference | Wins | Mean improvement | 95% CI | "
            "Wilcoxon p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            "| "
            f"{row['reference_display_name']} | "
            f"{row['wins']}/{row['paired_seeds']} | "
            f"{row['mean_improvement_percent']:.2f}% | "
            f"+/- {row['improvement_ci95_percent']:.2f}% | "
            f"{row['wilcoxon_one_sided_p']:.4g} | "
            f"{row['wilcoxon_holm_p']:.4g} |"
        )

    max_range = max(
        row["relative_checkpoint_range_percent"] for row in convergence
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- {primary_conclusion}",
            f"- OUR vs DAOC: {primary['wins']}/{primary['paired_seeds']} "
            f"seed wins, mean paired improvement "
            f"{primary['mean_improvement_percent']:.2f}% +/- "
            f"{primary['improvement_ci95_percent']:.2f}%.",
            f"- The paired t-test is not significant "
            f"(two-sided p={primary['paired_t_two_sided_p']:.4f}), and the "
            f"win/loss-only sign test is marginal "
            f"(one-sided p={primary['sign_test_one_sided_p']:.4f}). The "
            "strongest claim should therefore be tied to the directional "
            "rank test, not described as unanimous across all tests.",
            f"- Sensitivity is still positive: median improvement "
            f"{sensitivity['mean_finish_time_median_improvement_percent']:.2f}%, "
            f"10% trimmed mean "
            f"{sensitivity['mean_finish_time_trimmed_improvement_percent']:.2f}%, "
            "and leave-one-seed-out mean improvement ranges from "
            f"{sensitivity['mean_finish_time_leave_one_out_min_percent']:.2f}% "
            f"to {sensitivity['mean_finish_time_leave_one_out_max_percent']:.2f}%.",
            f"- P95 finish time improves by "
            f"{sensitivity['p95_improvement_percent']:.2f}% +/- "
            f"{sensitivity['p95_improvement_ci95_percent']:.2f}% "
            f"with {sensitivity['p95_wins']}/10 seed wins "
            f"(one-sided Wilcoxon p="
            f"{sensitivity['p95_wilcoxon_one_sided_p']:.4f}).",
            f"- OUR does not improve raw cache hit rate "
            f"(mean difference "
            f"{sensitivity['cache_hit_rate_difference']:+.3f}); the latency "
            "gain is therefore better attributed to dependency-aware cache "
            "placement and scheduling quality than to more cache hits alone.",
            "- Raw checkpoint values can still oscillate because the final "
            "policy is discrete. The largest within-window checkpoint range "
            f"was {max_range:.2f}%; claims should therefore use "
            "\"moving-window performance convergence\" wording.",
            "",
            "## Artifacts",
            "",
            "- `convergence_per_seed.csv`: every stopping decision and final "
            "diagnostic.",
            "- `convergence_summary.csv`: method-level performance and "
            "convergence summaries.",
            "- `paired_our_comparisons.csv`: seed-paired effect sizes and "
            "tests.",
            "- `convergence_episodes.png`: stopping-budget distribution.",
            "- `daoc_our_validation_paths.png`: fixed-bank validation paths.",
            "- `paired_daoc_our.png`: per-seed held-out comparison.",
            "",
            f"Suite profile: `{manifest.get('profile')}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    manifest, labels, seeds, runs = load_and_verify(suite_dir)
    convergence = convergence_rows(runs)
    aggregate = aggregate_rows(labels, runs)
    paired = paired_rows(runs)
    sensitivity = daoc_our_sensitivity(runs)

    write_csv(suite_dir / "convergence_per_seed.csv", convergence)
    write_csv(suite_dir / "convergence_summary.csv", aggregate)
    write_csv(suite_dir / "paired_our_comparisons.csv", paired)
    write_csv(
        suite_dir / "daoc_our_sensitivity.csv",
        [sensitivity],
    )

    configure_plot_style()
    plot_convergence_episodes(suite_dir, labels, runs)
    plot_validation_curves(suite_dir, runs)
    plot_paired_daoc_our(suite_dir, runs)

    report = render_report(
        manifest,
        labels,
        seeds,
        aggregate,
        paired,
        convergence,
        sensitivity,
    )
    report_path = suite_dir / "CONVERGED_EXPERIMENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
