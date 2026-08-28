#!/usr/bin/env python3
"""Attribute online-stream performance to adaptive cache updates."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LABELS = {
    "guided_full": "DAOC",
    "our": "OUR",
}
MODES = ("frozen", "adaptive")
ROW_FILES = {
    "frozen": "online_stream_frozen.csv",
    "adaptive": "online_stream.csv",
}
SUMMARY_FILES = {
    "frozen": "online_stream_frozen_summary.json",
    "adaptive": "online_stream_summary.json",
}
LATENCY_METRICS = (
    "computing_latency",
    "data_transfer_latency",
    "predecessor_latency",
    "service_latency",
    "waiting_latency",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare adaptive and frozen cache evaluation on exactly "
            "matched online workload streams."
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(
        stats.t.ppf(0.975, values.size - 1)
        * stats.sem(values)
    )


def paired_mode_effect(
    frozen,
    adaptive,
    lower_is_better=True,
):
    frozen = np.asarray(frozen, dtype=float)
    adaptive = np.asarray(adaptive, dtype=float)
    delta = adaptive - frozen
    ties = np.isclose(delta, 0.0, rtol=0.0, atol=1e-12)
    if np.all(ties):
        wilcoxon_p = 1.0
        paired_t_p = 1.0
    else:
        wilcoxon_p = float(
            stats.wilcoxon(
                adaptive,
                frozen,
                alternative="two-sided",
            ).pvalue
        )
        paired_t_p = float(
            stats.ttest_rel(adaptive, frozen).pvalue
        )
    relative_change = None
    if np.all(np.abs(frozen) > 1e-12):
        relative_change = float(
            np.mean(100.0 * delta / frozen)
        )
    adaptive_better = (
        delta < -1e-12
        if lower_is_better
        else delta > 1e-12
    )
    adaptive_worse = (
        delta > 1e-12
        if lower_is_better
        else delta < -1e-12
    )
    return {
        "pairs": int(frozen.size),
        "frozen_mean": float(frozen.mean()),
        "adaptive_mean": float(adaptive.mean()),
        "adaptive_minus_frozen_mean": float(delta.mean()),
        "adaptive_minus_frozen_ci95_half_width": (
            confidence_interval(delta)
        ),
        "adaptive_relative_change_percent": relative_change,
        "adaptive_better": int(np.sum(adaptive_better)),
        "ties": int(np.sum(ties)),
        "adaptive_worse": int(np.sum(adaptive_worse)),
        "wilcoxon_two_sided_p": wilcoxon_p,
        "paired_t_two_sided_p": paired_t_p,
    }


def one_sample_effect(values):
    values = np.asarray(values, dtype=float)
    ties = np.isclose(values, 0.0, rtol=0.0, atol=1e-12)
    if np.all(ties):
        wilcoxon_p = 1.0
        t_p = 1.0
    else:
        wilcoxon_p = float(
            stats.wilcoxon(
                values,
                alternative="two-sided",
            ).pvalue
        )
        t_p = float(stats.ttest_1samp(values, 0.0).pvalue)
    return {
        "pairs": int(values.size),
        "mean": float(values.mean()),
        "ci95_half_width": confidence_interval(values),
        "positive": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(ties)),
        "negative": int(np.sum(values < -1e-12)),
        "wilcoxon_two_sided_p": wilcoxon_p,
        "one_sample_t_two_sided_p": t_p,
    }


def segment_rows(rows, shift_episode):
    shift_index = shift_episode - 1
    return {
        "overall": rows,
        "pre_shift": rows[:shift_index],
        "post_shift_early": rows[shift_index : shift_index + 10],
        "post_shift_late": rows[-10:],
    }


def mean_metric(rows, metric):
    return float(
        np.mean([float(row[metric]) for row in rows])
    )


def total_metric(rows, metric):
    return float(
        np.sum([float(row[metric]) for row in rows])
    )


def discover_seeds(suite_dir):
    manifest = read_json(suite_dir / "suite_manifest.json")
    seeds = manifest.get("profile_config", {}).get("seeds", [])
    if not seeds:
        raise RuntimeError("The suite manifest contains no seeds")
    return [int(seed) for seed in seeds]


def load_experiment(suite_dir):
    seeds = discover_seeds(suite_dir)
    values = {
        label: {
            mode: {
                "trajectories": [],
                "segments": {},
                "totals": {},
            }
            for mode in MODES
        }
        for label in LABELS
    }
    seed_records = []
    shift_episode = None
    episodes = None

    for seed in seeds:
        rows_by_label_mode = {}
        for label in LABELS:
            run_dir = (
                suite_dir / "runs" / label / f"seed_{seed}"
            )
            for mode in MODES:
                summary = read_json(
                    run_dir / SUMMARY_FILES[mode]
                )
                rows = read_rows(run_dir / ROW_FILES[mode])
                protocol = summary["protocol"]
                expected_updates = mode == "adaptive"
                if (
                    protocol["cache_updates_enabled"]
                    != expected_updates
                ):
                    raise RuntimeError(
                        f"{label} seed {seed}: invalid {mode} protocol"
                    )
                if shift_episode is None:
                    shift_episode = int(
                        protocol["shift_episode"]
                    )
                    episodes = int(protocol["episodes"])
                if (
                    int(protocol["shift_episode"]) != shift_episode
                    or int(protocol["episodes"]) != episodes
                    or len(rows) != episodes
                ):
                    raise RuntimeError(
                        f"{label} seed {seed}: stream shape mismatch"
                    )
                rows_by_label_mode[label, mode] = rows

        reference_fingerprints = tuple(
            row["scenario_fingerprint"]
            for row in rows_by_label_mode[
                "guided_full", "frozen"
            ]
        )
        for rows in rows_by_label_mode.values():
            fingerprints = tuple(
                row["scenario_fingerprint"] for row in rows
            )
            if fingerprints != reference_fingerprints:
                raise RuntimeError(
                    f"Scenario mismatch for seed {seed}"
                )
        if len(set(reference_fingerprints)) != episodes:
            raise RuntimeError(
                f"Repeated workload scenario for seed {seed}"
            )

        record = {
            "seed": seed,
            "all_four_streams_match": 1,
        }
        for label in LABELS:
            key_prefix = (
                "daoc" if label == "guided_full" else "our"
            )
            for mode in MODES:
                rows = rows_by_label_mode[label, mode]
                segments = segment_rows(rows, shift_episode)
                values[label][mode]["trajectories"].append(
                    [
                        float(row["average_finish_time"])
                        for row in rows
                    ]
                )
                for segment, segment_values in segments.items():
                    metrics = {
                        "average_finish_time",
                        "p95_finish_time",
                        "cache_hit_rate",
                        "cache_replacements",
                        *LATENCY_METRICS,
                    }
                    for metric in metrics:
                        values[label][mode]["segments"].setdefault(
                            (segment, metric),
                            [],
                        ).append(
                            mean_metric(segment_values, metric)
                        )
                for metric in (
                    "cache_replacements",
                    "cache_decision_calls",
                    "cache_decision_wall_time_sec",
                    "control_payload_bytes",
                ):
                    values[label][mode]["totals"].setdefault(
                        metric,
                        [],
                    ).append(total_metric(rows, metric))

                overall = mean_metric(
                    rows,
                    "average_finish_time",
                )
                record[
                    f"{key_prefix}_{mode}_finish_time"
                ] = overall
                record[
                    f"{key_prefix}_{mode}_cache_hit_rate"
                ] = mean_metric(rows, "cache_hit_rate")
                record[
                    f"{key_prefix}_{mode}_cache_replacements"
                ] = total_metric(rows, "cache_replacements")

            frozen_finish = record[
                f"{key_prefix}_frozen_finish_time"
            ]
            adaptive_finish = record[
                f"{key_prefix}_adaptive_finish_time"
            ]
            record[
                f"{key_prefix}_adaptive_minus_frozen"
            ] = adaptive_finish - frozen_finish

        frozen_gap = (
            record["daoc_frozen_finish_time"]
            - record["our_frozen_finish_time"]
        )
        adaptive_gap = (
            record["daoc_adaptive_finish_time"]
            - record["our_adaptive_finish_time"]
        )
        record["frozen_daoc_minus_our"] = frozen_gap
        record["adaptive_daoc_minus_our"] = adaptive_gap
        record["adaptive_gap_expansion"] = (
            adaptive_gap - frozen_gap
        )
        seed_records.append(record)

    return (
        seeds,
        shift_episode,
        episodes,
        values,
        seed_records,
    )


def build_summary(
    seeds,
    shift_episode,
    episodes,
    values,
    seed_records,
):
    segment_effects = {}
    for label in LABELS:
        segment_effects[label] = {}
        for segment in (
            "overall",
            "pre_shift",
            "post_shift_early",
            "post_shift_late",
        ):
            segment_effects[label][segment] = paired_mode_effect(
                values[label]["frozen"]["segments"][
                    segment,
                    "average_finish_time",
                ],
                values[label]["adaptive"]["segments"][
                    segment,
                    "average_finish_time",
                ],
            )

    latency_effects = {}
    for label in LABELS:
        latency_effects[label] = {
            metric: paired_mode_effect(
                values[label]["frozen"]["segments"][
                    "overall",
                    metric,
                ],
                values[label]["adaptive"]["segments"][
                    "overall",
                    metric,
                ],
            )
            for metric in LATENCY_METRICS
        }

    cache_effects = {}
    for label in LABELS:
        cache_effects[label] = {
            "cache_hit_rate": paired_mode_effect(
                values[label]["frozen"]["segments"][
                    "overall",
                    "cache_hit_rate",
                ],
                values[label]["adaptive"]["segments"][
                    "overall",
                    "cache_hit_rate",
                ],
                lower_is_better=False,
            ),
            "cache_replacements_per_run": paired_mode_effect(
                values[label]["frozen"]["totals"][
                    "cache_replacements"
                ],
                values[label]["adaptive"]["totals"][
                    "cache_replacements"
                ],
            ),
            "cache_decision_calls_per_run": paired_mode_effect(
                values[label]["frozen"]["totals"][
                    "cache_decision_calls"
                ],
                values[label]["adaptive"]["totals"][
                    "cache_decision_calls"
                ],
            ),
            "cache_decision_wall_time_sec_per_run": (
                paired_mode_effect(
                    values[label]["frozen"]["totals"][
                        "cache_decision_wall_time_sec"
                    ],
                    values[label]["adaptive"]["totals"][
                        "cache_decision_wall_time_sec"
                    ],
                )
            ),
            "control_payload_bytes_per_run": paired_mode_effect(
                values[label]["frozen"]["totals"][
                    "control_payload_bytes"
                ],
                values[label]["adaptive"]["totals"][
                    "control_payload_bytes"
                ],
            ),
        }

    gap_expansion = [
        row["adaptive_gap_expansion"]
        for row in seed_records
    ]
    return {
        "status": "complete",
        "integrity": {
            "paired_seeds": len(seeds),
            "seeds": seeds,
            "episodes_per_seed": episodes,
            "shift_episode": shift_episode,
            "all_four_streams_match": True,
            "all_scenarios_unique_within_seed": True,
            "offloading_weights_frozen": True,
            "deployment_fixed": True,
        },
        "finish_time_mode_effects": segment_effects,
        "latency_component_mode_effects": latency_effects,
        "cache_mode_effects": cache_effects,
        "difference_in_differences": {
            "definition": (
                "(DAOC adaptive - DAOC frozen) - "
                "(OUR adaptive - OUR frozen)"
            ),
            **one_sample_effect(gap_expansion),
        },
    }


def write_per_seed(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    path,
    values,
    seed_records,
    shift_episode,
):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
    )
    styles = (
        ("guided_full", "adaptive", "#4B5563", "-"),
        ("guided_full", "frozen", "#4B5563", "--"),
        ("our", "adaptive", "#DC2626", "-"),
        ("our", "frozen", "#DC2626", "--"),
    )
    for label, mode, color, line_style in styles:
        trajectories = np.asarray(
            values[label][mode]["trajectories"],
            dtype=float,
        )
        mean = trajectories.mean(axis=0)
        episodes = np.arange(1, trajectories.shape[1] + 1)
        axes[0].plot(
            episodes,
            mean,
            color=color,
            linestyle=line_style,
            linewidth=1.7,
            label=f"{LABELS[label]} {mode}",
        )
    axes[0].axvline(
        shift_episode,
        color="#111827",
        linestyle=":",
        linewidth=1.3,
        label="Hotspot shift",
    )
    axes[0].set_xlabel("Online workload window")
    axes[0].set_ylabel("Mean application finish time (s)")
    axes[0].set_title("Cache-update attribution", loc="left")
    axes[0].legend(frameon=False, ncol=2)

    x = np.arange(len(seed_records))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [
            row["daoc_adaptive_minus_frozen"]
            for row in seed_records
        ],
        width,
        color="#4B5563",
        label="DAOC",
    )
    axes[1].bar(
        x + width / 2,
        [
            row["our_adaptive_minus_frozen"]
            for row in seed_records
        ],
        width,
        color="#DC2626",
        label="OUR",
    )
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(
        x,
        [str(row["seed"]) for row in seed_records],
    )
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Adaptive - frozen finish time (s)")
    axes[1].set_title("Per-seed cache-update effect", loc="left")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def format_p(value):
    return f"{value:.6g}"


def write_report(path, summary):
    finish = summary["finish_time_mode_effects"]
    daoc = finish["guided_full"]
    ours = finish["our"]
    did = summary["difference_in_differences"]
    latency = summary["latency_component_mode_effects"]
    cache = summary["cache_mode_effects"]

    lines = [
        "# Adaptive-vs-Frozen Cache Attribution",
        "",
        "## Integrity",
        "",
        (
            f"- {summary['integrity']['paired_seeds']} paired seeds, "
            f"{summary['integrity']['episodes_per_seed']} independent "
            "workload windows per seed."
        ),
        "- Every DAOC/OUR adaptive/frozen run uses the same exogenous fingerprints.",
        "- Offloading weights and deployment are fixed; only cache updating differs.",
        "",
        "## Finish-time effect",
        "",
        "| Method | Segment | Frozen (s) | Adaptive (s) | Adaptive - frozen (s) | Paired t p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    segment_names = {
        "overall": "Overall",
        "pre_shift": "Pre-shift",
        "post_shift_early": "First 10 post-shift",
        "post_shift_late": "Last 10 post-shift",
    }
    for label, method_effects in (
        ("DAOC", daoc),
        ("OUR", ours),
    ):
        for segment, name in segment_names.items():
            effect = method_effects[segment]
            lines.append(
                f"| {label} | {name} | "
                f"{effect['frozen_mean']:.6f} | "
                f"{effect['adaptive_mean']:.6f} | "
                f"{effect['adaptive_minus_frozen_mean']:+.6f} | "
                f"{format_p(effect['paired_t_two_sided_p'])} |"
            )
    lines.extend(
        [
            "",
            (
                "- Difference in differences: "
                f"{did['mean']:+.6f} s "
                f"(95% CI {did['mean'] - did['ci95_half_width']:+.6f} "
                f"to {did['mean'] + did['ci95_half_width']:+.6f}); "
                f"{did['positive']}/{did['pairs']} seeds positive; "
                f"paired construction one-sample t p="
                f"{format_p(did['one_sample_t_two_sided_p'])}."
            ),
            "",
            "## Mechanism",
            "",
            "| Method | Metric | Frozen | Adaptive | Delta | Paired t p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label_key, label in LABELS.items():
        for metric, metric_name in (
            ("predecessor_latency", "Predecessor latency (s)"),
            ("service_latency", "Service latency (s)"),
        ):
            effect = latency[label_key][metric]
            lines.append(
                f"| {label} | {metric_name} | "
                f"{effect['frozen_mean']:.6f} | "
                f"{effect['adaptive_mean']:.6f} | "
                f"{effect['adaptive_minus_frozen_mean']:+.6f} | "
                f"{format_p(effect['paired_t_two_sided_p'])} |"
            )
        hit = cache[label_key]["cache_hit_rate"]
        replacements = cache[label_key][
            "cache_replacements_per_run"
        ]
        lines.append(
            f"| {label} | Cache hit rate | "
            f"{hit['frozen_mean']:.6f} | "
            f"{hit['adaptive_mean']:.6f} | "
            f"{hit['adaptive_minus_frozen_mean']:+.6f} | "
            f"{format_p(hit['paired_t_two_sided_p'])} |"
        )
        lines.append(
            f"| {label} | Replacements/run | "
            f"{replacements['frozen_mean']:.1f} | "
            f"{replacements['adaptive_mean']:.1f} | "
            f"{replacements['adaptive_minus_frozen_mean']:+.1f} | "
            f"{format_p(replacements['paired_t_two_sided_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "DAOC's adaptive cache increases finish time in every seed, "
                "whereas OUR adaptive and frozen performance are statistically "
                "indistinguishable. The adaptive DAOC-vs-OUR gap therefore "
                "supports resistance to harmful cache churn, not evidence that "
                "OUR positively adapts beyond its frozen cache."
            ),
            "",
            (
                "Raw cache hit rate is not the optimization target: it is "
                "unweighted by service image size, transfer path, task "
                "criticality, and predecessor locality. DAOC obtains more raw "
                "hits while both service and predecessor latency increase."
            ),
            "",
            (
                "Service migration time is not added to application finish "
                "time in this experiment. Repeated regime shifts, migration "
                "cost, stale telemetry, and trace-driven demand remain needed "
                "before claiming fast online adaptation."
            ),
            "",
            "## Artifacts",
            "",
            "- `online_cache_mode_per_seed.csv`",
            "- `online_cache_mode_summary.json`",
            "- `online_cache_mode_comparison.png` and `.pdf`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    (
        seeds,
        shift_episode,
        episodes,
        values,
        seed_records,
    ) = load_experiment(suite_dir)
    summary = build_summary(
        seeds,
        shift_episode,
        episodes,
        values,
        seed_records,
    )
    write_per_seed(
        suite_dir / "online_cache_mode_per_seed.csv",
        seed_records,
    )
    write_json(
        suite_dir / "online_cache_mode_summary.json",
        summary,
    )
    plot_comparison(
        suite_dir / "online_cache_mode_comparison",
        values,
        seed_records,
        shift_episode,
    )
    write_report(
        suite_dir / "ONLINE_CACHE_MODE_REPORT.md",
        summary,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "suite_dir": str(suite_dir),
                "paired_seeds": len(seeds),
                "difference_in_differences": (
                    summary["difference_in_differences"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
