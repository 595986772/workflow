#!/usr/bin/env python3
"""Audit and summarize Pegasus-B8 fixed-budget heterogeneity results."""

import csv
import json
from pathlib import Path

import numpy as np

from pegasus_b8_heterogeneity_protocol import (
    ANALYSIS_DIR,
    CAPACITY_PROFILES,
    DISPLAY_NAMES,
    H2_RUNS,
    METHODS,
    RESULT_ROOT,
    RUN_ROOT,
    SEEDS,
    population_variance,
    validate_protocol,
)
from run_pegasus_b8_heterogeneity import verify_suite


METRICS = (
    "mean_average_finish_time",
    "mean_p95_finish_time",
    "mean_waiting_latency",
    "mean_cache_hit_rate",
    "mean_cache_remote_loading_rate",
    "mean_cache_service_coverage",
    "mean_cache_zero_capacity_assignment_rate",
    "mean_cache_capacity_utilization",
    "mean_policy_inference_time_per_decision_ms",
    "mean_cache_decision_wall_time_sec",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dir(profile, method, seed):
    if profile == "H2":
        return H2_RUNS[method] / f"seed_{seed}"
    return RUN_ROOT / profile / "runs" / method / f"seed_{seed}"


def metric_value(summary, metric):
    if metric in summary["eval"]:
        return float(summary["eval"][metric])
    return float("nan")


def paired_gain(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return 100.0 * (reference - candidate) / reference


def main():
    protocol = validate_protocol(require_h2=True)
    for profile in ("H0", "H1", "H3"):
        verify_suite(profile)

    rows = []
    values = {}
    for profile, capacities in CAPACITY_PROFILES.items():
        values[profile] = {}
        for method in METHODS:
            seed_values = []
            for seed in SEEDS:
                summary = read_json(run_dir(profile, method, seed) / "summary.json")
                record = {
                    "profile": profile,
                    "capacity_multiset": ",".join(map(str, capacities)),
                    "capacity_variance": population_variance(capacities),
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "seed": seed,
                    "convergence_episode": summary["convergence"]["episode"],
                }
                for metric in METRICS:
                    record[metric] = metric_value(summary, metric)
                rows.append(record)
                seed_values.append(record["mean_average_finish_time"])
            values[profile][method] = seed_values

    aggregate = []
    for profile, capacities in CAPACITY_PROFILES.items():
        for method in METHODS:
            method_rows = [
                row
                for row in rows
                if row["profile"] == profile and row["method"] == method
            ]
            item = {
                "profile": profile,
                "capacity_variance": population_variance(capacities),
                "method": method,
                "display_name": DISPLAY_NAMES[method],
            }
            for metric in METRICS:
                data = np.asarray([row[metric] for row in method_rows], dtype=float)
                item[f"{metric}_mean"] = float(np.nanmean(data))
                item[f"{metric}_std"] = float(np.nanstd(data, ddof=1))
            aggregate.append(item)

    gains = {}
    for profile in CAPACITY_PROFILES:
        dcc = paired_gain(
            values[profile][METHODS[0]], values[profile][METHODS[1]]
        )
        pairwise = paired_gain(
            values[profile][METHODS[2]], values[profile][METHODS[3]]
        )
        ours_daoc = paired_gain(
            values[profile][METHODS[0]], values[profile][METHODS[3]]
        )
        gains[profile] = {
            "dcc_gain_percent_per_seed": dcc.tolist(),
            "dcc_gain_percent_mean": float(dcc.mean()),
            "pairwise_gain_percent_per_seed": pairwise.tolist(),
            "pairwise_gain_percent_mean": float(pairwise.mean()),
            "our_vs_daoc_percent_per_seed": ours_daoc.tolist(),
            "our_vs_daoc_percent_mean": float(ours_daoc.mean()),
            "our_vs_daoc_seed_wins": int(np.sum(ours_daoc > 0)),
        }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with (ANALYSIS_DIR / "seed_level_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ANALYSIS_DIR / "aggregate_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    report = {
        "status": "complete",
        "protocol": protocol,
        "scope": (
            "Three-seed mechanism/sensitivity evidence; seed is the "
            "statistical unit and no formal superiority claim is made."
        ),
        "aggregate": aggregate,
        "paired_gains": gains,
    }
    (ANALYSIS_DIR / "heterogeneity_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(gains, indent=2))


if __name__ == "__main__":
    main()
