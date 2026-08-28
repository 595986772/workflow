#!/usr/bin/env python3
"""Post-hoc S8 diagnosis and paper-evidence audit for frozen h8v1.

This script only reads completed experiment artifacts. It never loads a model,
changes an algorithm, or updates FINAL_LOCK.json.
"""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SEEDS = tuple(range(11, 21))
CENTRAL = "centralized_greedy_daoc"
OUR = "lean_our"
METHODS = (CENTRAL, OUR)
DISPLAY_NAMES = {CENTRAL: "Centralized-Greedy", OUR: "OUR"}
COLORS = {CENTRAL: "#E09F3E", OUR: "#277DA1"}
SERVER_COUNT = 10
EXPECTED_EVAL_EPISODES = 100


def parse_args():
    repo_root = Path(__file__).resolve().parent
    revision_root = (
        repo_root / "results/a0_fixed_budget_heterogeneity/h8v1"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, default=revision_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=revision_root / "posthoc_diagnosis",
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) != EXPECTED_EVAL_EPISODES:
        raise RuntimeError(
            f"Expected {EXPECTED_EVAL_EPISODES} eval rows in {path}, "
            f"found {len(rows)}"
        )
    fingerprints = [row["base_scenario_fingerprint"] for row in rows]
    if len(set(fingerprints)) != EXPECTED_EVAL_EPISODES:
        raise RuntimeError(f"Evaluation fingerprints are not unique: {path}")
    return rows


def parse_server_map(value, cast=float):
    parsed = json.loads(value)
    return {
        server_id: cast(parsed.get(str(server_id), 0))
        for server_id in range(SERVER_COUNT)
    }


def normalized_hhi(values):
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values / total
    raw = float(np.sum(probabilities**2))
    uniform = 1.0 / len(values)
    return float((raw - uniform) / (1.0 - uniform))


def normalized_entropy(values):
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(values)))


def normalized_gini(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or float(values.sum()) <= 0:
        return 0.0
    sorted_values = np.sort(values)
    indices = np.arange(1, len(values) + 1)
    raw = float(
        np.sum((2 * indices - len(values) - 1) * sorted_values)
        / (len(values) * sorted_values.sum())
    )
    maximum = (len(values) - 1) / len(values)
    return raw / maximum


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half_width = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return mean - half_width, mean + half_width


def paired_diagnostic(improvements):
    improvements = np.asarray(improvements, dtype=float)
    lower, upper = confidence_interval(improvements)
    nonzero = improvements[np.abs(improvements) > 1e-12]
    p_value = (
        float(stats.wilcoxon(nonzero, alternative="greater").pvalue)
        if len(nonzero)
        else 1.0
    )
    return {
        "pairs": int(len(improvements)),
        "mean_improvement_sec": float(improvements.mean()),
        "ci95_lower_sec": lower,
        "ci95_upper_sec": upper,
        "wilcoxon_one_sided_p": p_value,
        "wins": int(np.sum(improvements > 0)),
        "diagnostic_only": True,
    }


def spearman_record(x_values, y_values):
    result = stats.spearmanr(x_values, y_values)
    return {
        "rho": float(result.statistic),
        "p_two_sided": float(result.pvalue),
        "pairs": len(x_values),
        "unit": "seed",
    }


def run_dir(suite_root, label, seed):
    return suite_root / "runs" / label / f"seed_{seed}"


def server_properties(directory):
    scenario = read_json(directory / "scenario_initial.json")
    servers = {}
    for key, value in scenario["servers"].items():
        server_id = int(key)
        frequency = float(value["frequency"])
        load = float(value["load"])
        servers[server_id] = {
            "frequency_hz": frequency,
            "frequency_ghz": frequency / 1e9,
            "load": load,
            "background_wait_sec": load * 1e6 / frequency,
            "cache_capacity": int(value["cache_capacity"]),
            "position": value["position"],
        }
    if set(servers) != set(range(SERVER_COUNT)):
        raise RuntimeError(f"Invalid server IDs in {directory}")
    return servers


def episode_metrics(row, servers):
    actions = np.asarray(
        list(parse_server_map(row["server_action_histogram_json"]).values()),
        dtype=float,
    )
    cpu_cycles = np.asarray(
        list(parse_server_map(row["server_cpu_cycle_histogram_json"]).values()),
        dtype=float,
    )
    capacities = parse_server_map(row["cache_capacity_vector_json"], int)
    if int(actions.sum()) <= 0:
        raise RuntimeError("Evaluation row has no actions")
    max_capacity = max(capacities.values())
    max_capacity_servers = [
        server_id
        for server_id, capacity in capacities.items()
        if capacity == max_capacity
    ]
    action_probabilities = actions / actions.sum()
    cpu_probabilities = cpu_cycles / cpu_cycles.sum()
    static_wait = np.asarray(
        [servers[server_id]["background_wait_sec"] for server_id in range(SERVER_COUNT)]
    )
    frequencies = np.asarray(
        [servers[server_id]["frequency_hz"] for server_id in range(SERVER_COUNT)]
    )
    return {
        "average_finish_time": float(row["average_finish_time"]),
        "p95_finish_time": float(row["p95_finish_time"]),
        "waiting_latency": float(row["waiting_latency"]),
        "computing_latency": float(row["computing_latency"]),
        "predecessor_latency": float(row["predecessor_latency"]),
        "service_latency": float(row["service_latency"]),
        "cache_hit_rate": float(row["cache_hit_rate"]),
        "remote_loading_rate": float(row["cache_remote_loading_rate"]),
        "action_hhi": normalized_hhi(actions),
        "action_entropy": normalized_entropy(actions),
        "action_gini": normalized_gini(actions),
        "top1_action_share": float(action_probabilities.max()),
        "cpu_hhi": normalized_hhi(cpu_cycles),
        "top1_cpu_share": float(cpu_probabilities.max()),
        "cache_capable_action_share": float(
            sum(
                action_probabilities[server_id]
                for server_id, capacity in capacities.items()
                if capacity > 0
            )
        ),
        "max_capacity_action_share": float(
            sum(action_probabilities[server_id] for server_id in max_capacity_servers)
        ),
        "max_capacity_cpu_share": float(
            sum(cpu_probabilities[server_id] for server_id in max_capacity_servers)
        ),
        "weighted_background_wait_proxy_sec_per_task": float(
            np.sum(action_probabilities * static_wait)
        ),
        "weighted_compute_proxy_sec_per_task": float(
            np.sum(cpu_cycles / frequencies) / actions.sum()
        ),
        "action_counts": actions,
        "cpu_cycles": cpu_cycles,
        "capacities": capacities,
        "max_capacity_servers": max_capacity_servers,
    }


def mean_fields(records, fields):
    return {
        field: float(np.mean([record[field] for record in records]))
        for field in fields
    }


METHOD_FIELDS = (
    "average_finish_time",
    "p95_finish_time",
    "waiting_latency",
    "computing_latency",
    "predecessor_latency",
    "service_latency",
    "cache_hit_rate",
    "remote_loading_rate",
    "action_hhi",
    "action_entropy",
    "action_gini",
    "top1_action_share",
    "cpu_hhi",
    "top1_cpu_share",
    "cache_capable_action_share",
    "max_capacity_action_share",
    "max_capacity_cpu_share",
    "weighted_background_wait_proxy_sec_per_task",
    "weighted_compute_proxy_sec_per_task",
)


def collect_seed(suite_root, seed):
    rows_by_method = {}
    servers_by_method = {}
    summaries = {}
    for label in METHODS:
        directory = run_dir(suite_root, label, seed)
        rows = evaluation_rows(directory / "episodes.csv")
        rows_by_method[label] = {
            row["base_scenario_fingerprint"]: row for row in rows
        }
        servers_by_method[label] = server_properties(directory)
        summaries[label] = read_json(directory / "summary.json")

    fingerprints = list(rows_by_method[CENTRAL])
    if set(fingerprints) != set(rows_by_method[OUR]):
        raise RuntimeError(f"Scenario pairing failed for seed {seed}")
    for server_id in range(SERVER_COUNT):
        left = servers_by_method[CENTRAL][server_id]
        right = servers_by_method[OUR][server_id]
        for key in ("frequency_hz", "load", "cache_capacity", "position"):
            if left[key] != right[key]:
                raise RuntimeError(
                    f"Server property mismatch for seed {seed}, server {server_id}: {key}"
                )

    servers = servers_by_method[OUR]
    method_episode_records = {label: [] for label in METHODS}
    episode_pairs = []
    for fingerprint in fingerprints:
        metrics = {
            label: episode_metrics(rows_by_method[label][fingerprint], servers)
            for label in METHODS
        }
        for label in METHODS:
            method_episode_records[label].append(metrics[label])
        central = metrics[CENTRAL]
        ours = metrics[OUR]
        episode_pairs.append(
            {
                "seed": seed,
                "base_scenario_fingerprint": fingerprint,
                **{
                    f"central_{field}": central[field]
                    for field in METHOD_FIELDS
                },
                **{f"our_{field}": ours[field] for field in METHOD_FIELDS},
                "finish_improvement_sec": (
                    central["average_finish_time"] - ours["average_finish_time"]
                ),
                "waiting_penalty_sec": (
                    ours["waiting_latency"] - central["waiting_latency"]
                ),
                "computing_penalty_sec": (
                    ours["computing_latency"] - central["computing_latency"]
                ),
                "predecessor_penalty_sec": (
                    ours["predecessor_latency"] - central["predecessor_latency"]
                ),
                "service_loading_gain_sec": (
                    central["service_latency"] - ours["service_latency"]
                ),
                "action_hhi_delta": ours["action_hhi"] - central["action_hhi"],
                "max_capacity_action_share_delta": (
                    ours["max_capacity_action_share"]
                    - central["max_capacity_action_share"]
                ),
                "wait_proxy_delta_sec_per_task": (
                    ours["weighted_background_wait_proxy_sec_per_task"]
                    - central["weighted_background_wait_proxy_sec_per_task"]
                ),
                "compute_proxy_delta_sec_per_task": (
                    ours["weighted_compute_proxy_sec_per_task"]
                    - central["weighted_compute_proxy_sec_per_task"]
                ),
            }
        )

    method_aggregates = {
        label: mean_fields(records, METHOD_FIELDS)
        for label, records in method_episode_records.items()
    }
    seed_record = {
        "seed": seed,
        "methods": method_aggregates,
        "finish_improvement_sec": float(
            np.mean([row["finish_improvement_sec"] for row in episode_pairs])
        ),
        "waiting_penalty_sec": float(
            np.mean([row["waiting_penalty_sec"] for row in episode_pairs])
        ),
        "computing_penalty_sec": float(
            np.mean([row["computing_penalty_sec"] for row in episode_pairs])
        ),
        "predecessor_penalty_sec": float(
            np.mean([row["predecessor_penalty_sec"] for row in episode_pairs])
        ),
        "service_loading_gain_sec": float(
            np.mean([row["service_loading_gain_sec"] for row in episode_pairs])
        ),
        "action_hhi_delta": float(
            np.mean([row["action_hhi_delta"] for row in episode_pairs])
        ),
        "max_capacity_action_share_delta": float(
            np.mean(
                [row["max_capacity_action_share_delta"] for row in episode_pairs]
            )
        ),
        "wait_proxy_delta_sec_per_task": float(
            np.mean([row["wait_proxy_delta_sec_per_task"] for row in episode_pairs])
        ),
        "compute_proxy_delta_sec_per_task": float(
            np.mean(
                [row["compute_proxy_delta_sec_per_task"] for row in episode_pairs]
            )
        ),
    }

    max_capacity = max(server["cache_capacity"] for server in servers.values())
    max_capacity_ids = [
        server_id
        for server_id, server in servers.items()
        if server["cache_capacity"] == max_capacity
    ]
    if len(max_capacity_ids) != 1:
        raise RuntimeError(f"Expected one maximum-capacity S8 server for seed {seed}")
    max_server_id = max_capacity_ids[0]
    max_server = dict(servers[max_server_id])
    max_server["server_id"] = max_server_id
    frequencies = [server["frequency_hz"] for server in servers.values()]
    waits = [server["background_wait_sec"] for server in servers.values()]
    max_server["frequency_rank_from_slowest"] = 1 + sum(
        value < max_server["frequency_hz"] for value in frequencies
    )
    max_server["background_wait_rank_from_best"] = 1 + sum(
        value < max_server["background_wait_sec"] for value in waits
    )
    final_our_row = rows_by_method[OUR][fingerprints[-1]]
    qualities = parse_server_map(final_our_row["cache_server_quality_json"])
    max_server["our_causal_quality"] = qualities[max_server_id]
    max_server["central_action_share"] = method_aggregates[CENTRAL][
        "max_capacity_action_share"
    ]
    max_server["our_action_share"] = method_aggregates[OUR][
        "max_capacity_action_share"
    ]
    max_server["central_cpu_share"] = method_aggregates[CENTRAL][
        "max_capacity_cpu_share"
    ]
    max_server["our_cpu_share"] = method_aggregates[OUR][
        "max_capacity_cpu_share"
    ]
    seed_record["max_capacity_server"] = max_server
    seed_record["selected_checkpoint_episode"] = {
        label: summaries[label]["selected_checkpoint_episode"] for label in METHODS
    }
    seed_record["scenario_pairs"] = len(episode_pairs)
    return seed_record, episode_pairs


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_seed_record(record):
    central = record["methods"][CENTRAL]
    ours = record["methods"][OUR]
    maximum = record["max_capacity_server"]
    return {
        "seed": record["seed"],
        "central_finish_time": central["average_finish_time"],
        "our_finish_time": ours["average_finish_time"],
        "finish_improvement_sec": record["finish_improvement_sec"],
        "waiting_penalty_sec": record["waiting_penalty_sec"],
        "computing_penalty_sec": record["computing_penalty_sec"],
        "predecessor_penalty_sec": record["predecessor_penalty_sec"],
        "service_loading_gain_sec": record["service_loading_gain_sec"],
        "central_action_hhi": central["action_hhi"],
        "our_action_hhi": ours["action_hhi"],
        "action_hhi_delta": record["action_hhi_delta"],
        "max_capacity_server_id": maximum["server_id"],
        "max_capacity_server_frequency_ghz": maximum["frequency_ghz"],
        "max_capacity_server_load": maximum["load"],
        "max_capacity_server_background_wait_ms": (
            1000 * maximum["background_wait_sec"]
        ),
        "max_capacity_server_frequency_rank_from_slowest": maximum[
            "frequency_rank_from_slowest"
        ],
        "max_capacity_server_background_wait_rank_from_best": maximum[
            "background_wait_rank_from_best"
        ],
        "max_capacity_server_causal_quality": maximum["our_causal_quality"],
        "central_max_capacity_action_share": maximum["central_action_share"],
        "our_max_capacity_action_share": maximum["our_action_share"],
        "max_capacity_action_share_delta": record[
            "max_capacity_action_share_delta"
        ],
        "central_max_capacity_cpu_share": maximum["central_cpu_share"],
        "our_max_capacity_cpu_share": maximum["our_cpu_share"],
        "wait_proxy_delta_sec_per_task": record[
            "wait_proxy_delta_sec_per_task"
        ],
        "compute_proxy_delta_sec_per_task": record[
            "compute_proxy_delta_sec_per_task"
        ],
    }


def build_correlations(seed_records):
    def values(field):
        return [record[field] for record in seed_records]

    improvements = values("finish_improvement_sec")
    return {
        "finish_improvement_vs_waiting_penalty": spearman_record(
            improvements, values("waiting_penalty_sec")
        ),
        "finish_improvement_vs_computing_penalty": spearman_record(
            improvements, values("computing_penalty_sec")
        ),
        "finish_improvement_vs_predecessor_penalty": spearman_record(
            improvements, values("predecessor_penalty_sec")
        ),
        "finish_improvement_vs_service_loading_gain": spearman_record(
            improvements, values("service_loading_gain_sec")
        ),
        "finish_improvement_vs_action_hhi_delta": spearman_record(
            improvements, values("action_hhi_delta")
        ),
        "finish_improvement_vs_max_capacity_share_delta": spearman_record(
            improvements, values("max_capacity_action_share_delta")
        ),
        "waiting_penalty_vs_static_wait_proxy_delta": spearman_record(
            values("waiting_penalty_sec"),
            values("wait_proxy_delta_sec_per_task"),
        ),
        "computing_penalty_vs_compute_proxy_delta": spearman_record(
            values("computing_penalty_sec"),
            values("compute_proxy_delta_sec_per_task"),
        ),
    }


def robust_z_scores(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0:
        return np.zeros_like(values)
    return 0.67448975 * (values - median) / mad


def build_summary(revision_root, seed_records, episode_pairs):
    final_summary_path = (
        revision_root / "final/analysis/final_heterogeneity_summary.json"
    )
    final_lock_path = revision_root / "FINAL_LOCK.json"
    final_summary = read_json(final_summary_path)
    final_lock = read_json(final_lock_path)
    formal_gate = final_summary.get("formal_gate", final_summary.get("gate"))
    if formal_gate is None:
        raise RuntimeError("Formal summary does not contain a gate")
    official_s8 = final_summary["profiles"]["S8"]
    improvements = np.asarray(
        [record["finish_improvement_sec"] for record in seed_records],
        dtype=float,
    )
    robust_z = robust_z_scores(improvements)
    for record, value in zip(seed_records, robust_z):
        record["finish_improvement_robust_z"] = float(value)

    seed_18 = next(record for record in seed_records if record["seed"] == 18)
    without_seed_18 = [
        record["finish_improvement_sec"]
        for record in seed_records
        if record["seed"] != 18
    ]
    central_hhi = [record["methods"][CENTRAL]["action_hhi"] for record in seed_records]
    our_hhi = [record["methods"][OUR]["action_hhi"] for record in seed_records]
    correlations = build_correlations(seed_records)

    integrity = {
        "final_lock_complete": final_lock.get("status") == "complete",
        "formal_results_not_used_before_lock": (
            final_lock.get("formal_results_seen_before_lock") is False
        ),
        "algorithm_not_retuned_after_final": (
            final_lock.get("algorithm_retuned_after_final") is False
        ),
        "all_ten_final_seeds_present": [record["seed"] for record in seed_records]
        == list(SEEDS),
        "one_hundred_paired_scenarios_per_seed": all(
            record["scenario_pairs"] == EXPECTED_EVAL_EPISODES
            for record in seed_records
        ),
        "official_formal_gate_unchanged": (
            final_lock.get("formal_gate") == formal_gate
        ),
    }
    max_server = seed_18["max_capacity_server"]
    conclusions = {
        "global_action_concentration_hypothesis_supported": bool(
            np.mean(our_hhi) > np.mean(central_hhi)
        ),
        "our_has_lower_action_hhi_in_seed_count": int(
            sum(ours < central for ours, central in zip(our_hhi, central_hhi))
        ),
        "seed18_cache_compute_mismatch_supported": bool(
            max_server["frequency_rank_from_slowest"] <= 2
            and max_server["background_wait_rank_from_best"] == SERVER_COUNT
            and max_server["our_action_share"] - max_server["central_action_share"]
            > 0.5
            and seed_18["computing_penalty_sec"]
            + seed_18["waiting_penalty_sec"]
            > seed_18["service_loading_gain_sec"]
        ),
        "static_waiting_is_background_load_not_endogenous_task_queue": True,
        "seed18_is_influential_not_removable": True,
        "posthoc_results_may_not_change_formal_gate": True,
    }
    return {
        "status": "complete",
        "analysis_type": "posthoc_diagnostic_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "revision": "h8v1",
        "profile": "S8",
        "seeds": list(SEEDS),
        "paired_scenarios": len(episode_pairs),
        "integrity": integrity,
        "official_s8_result": {
            "our_vs_central": official_s8["paired_superiority"][
                "our_vs_centralized_greedy_daoc"
            ],
            "formal_gate_passed": formal_gate["passed"],
        },
        "action_concentration": {
            "central_mean_normalized_hhi": float(np.mean(central_hhi)),
            "our_mean_normalized_hhi": float(np.mean(our_hhi)),
            "central_mean_top1_share": float(
                np.mean(
                    [
                        record["methods"][CENTRAL]["top1_action_share"]
                        for record in seed_records
                    ]
                )
            ),
            "our_mean_top1_share": float(
                np.mean(
                    [
                        record["methods"][OUR]["top1_action_share"]
                        for record in seed_records
                    ]
                )
            ),
        },
        "seed_level_correlations": correlations,
        "seed18": seed_18,
        "leave_seed18_out_influence_diagnostic": paired_diagnostic(
            without_seed_18
        ),
        "conclusions": conclusions,
        "per_seed": seed_records,
        "source_files": {
            "final_lock": {
                "path": str(final_lock_path.resolve()),
                "sha256": sha256_file(final_lock_path),
            },
            "formal_summary": {
                "path": str(final_summary_path.resolve()),
                "sha256": sha256_file(final_summary_path),
            },
            "static_wait_formula": {
                "path": str((Path(__file__).resolve().parent / "server.py").resolve()),
                "sha256": sha256_file(Path(__file__).resolve().parent / "server.py"),
                "formula": "server.load * 1e6 / server.frequency",
            },
        },
    }


def plot_diagnosis(output_dir, seed_records, summary):
    seeds = np.asarray([record["seed"] for record in seed_records])
    improvements = np.asarray(
        [record["finish_improvement_sec"] for record in seed_records]
    )
    waiting_penalties = np.asarray(
        [record["waiting_penalty_sec"] for record in seed_records]
    )
    computing_penalties = np.asarray(
        [record["computing_penalty_sec"] for record in seed_records]
    )
    service_gains = np.asarray(
        [record["service_loading_gain_sec"] for record in seed_records]
    )
    central_hhi = np.asarray(
        [record["methods"][CENTRAL]["action_hhi"] for record in seed_records]
    )
    our_hhi = np.asarray(
        [record["methods"][OUR]["action_hhi"] for record in seed_records]
    )
    central_max_share = np.asarray(
        [record["max_capacity_server"]["central_action_share"] for record in seed_records]
    )
    our_max_share = np.asarray(
        [record["max_capacity_server"]["our_action_share"] for record in seed_records]
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    bar_colors = ["#2A9D8F" if value > 0 else "#C94C4C" for value in improvements]
    axes[0, 0].bar(seeds, improvements, color=bar_colors)
    axes[0, 0].axhline(0, color="#333333", linewidth=1)
    axes[0, 0].set_title("S8 paired completion-time improvement")
    axes[0, 0].set_xlabel("Seed")
    axes[0, 0].set_ylabel("Centralized-Greedy - OUR (s)")
    axes[0, 0].set_xticks(seeds)

    width = 0.25
    x = np.arange(len(seeds))
    axes[0, 1].bar(x - width, service_gains, width, label="Service-loading gain", color="#2A9D8F")
    axes[0, 1].bar(x, computing_penalties, width, label="Computing penalty", color="#E76F51")
    axes[0, 1].bar(x + width, waiting_penalties, width, label="Waiting penalty", color="#F4A261")
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].set_title("Latency trade-off by seed")
    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].set_xticks(x, seeds)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(seeds, central_hhi, "o-", label=DISPLAY_NAMES[CENTRAL], color=COLORS[CENTRAL])
    axes[1, 0].plot(seeds, our_hhi, "o-", label=DISPLAY_NAMES[OUR], color=COLORS[OUR])
    axes[1, 0].set_title("Overall action concentration")
    axes[1, 0].set_xlabel("Seed")
    axes[1, 0].set_ylabel("Normalized HHI")
    axes[1, 0].set_xticks(seeds)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(seeds, central_max_share, "o-", label=DISPLAY_NAMES[CENTRAL], color=COLORS[CENTRAL])
    axes[1, 1].plot(seeds, our_max_share, "o-", label=DISPLAY_NAMES[OUR], color=COLORS[OUR])
    axes[1, 1].set_title("Actions sent to the K=3 server")
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Action share")
    axes[1, 1].set_xticks(seeds)
    axes[1, 1].legend(frameon=False)

    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"s8_posthoc_diagnosis.{suffix}", dpi=180)
    plt.close(figure)


def audit_evidence(repo_root):
    paths = {
        "h8v1_final": repo_root
        / "results/a0_fixed_budget_heterogeneity/h8v1/final/analysis/final_heterogeneity_summary.json",
        "h8v1_development": repo_root
        / "results/a0_fixed_budget_heterogeneity/h8v1/experiment/analysis/heterogeneity_summary.json",
        "h8v0_development": repo_root
        / "results/a0_fixed_budget_heterogeneity/h8v0/experiment/analysis/heterogeneity_summary.json",
        "a0r2_final": repo_root
        / "results/a0_cache_coordination/a0r2/final/FINAL_SUMMARY.json",
        "a0r2_dynamic": repo_root
        / "results/a0_cache_coordination/a0r2/final/dynamic/dynamic_summary.json",
        "hsr1_ablation": repo_root
        / "results/static_heterogeneity/hsr1/ablation/static_ablation_analysis.json",
        "algorithm_extensions": repo_root
        / "results/alibaba_cp100_budget20/a2_paper_extensions/analysis/algorithm_extensions_summary.json",
        "user_scaling": repo_root
        / "results/alibaba_cp100_budget20/a2_paper_extensions/scaling/user_scaling_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Evidence files are missing: {missing}")
    data = {key: read_json(path) for key, path in paths.items()}
    h8v1 = data["h8v1_final"]
    a0r2 = data["a0r2_final"]
    ablation = data["hsr1_ablation"]
    extensions = data["algorithm_extensions"]
    scaling = data["user_scaling"]
    dynamic = data["a0r2_dynamic"]

    entries = [
        {
            "id": "h8v1_fixed_budget_heterogeneity_final",
            "algorithm_revision": "h8v1",
            "dataset": "Alibaba-CP100-A0 controlled mechanism dataset",
            "seeds": list(SEEDS),
            "evidence_level": "A_exact_current_formal",
            "usable_for": "Primary static fixed-budget heterogeneity claim",
            "not_usable_for": "Unbiased Alibaba generalization claim",
            "result": {
                profile: {
                    "our_vs_daoc": value["paired_superiority"]["our_vs_guided_full"],
                    "our_vs_central": value["paired_superiority"]["our_vs_centralized_greedy_daoc"],
                }
                for profile, value in h8v1["profiles"].items()
            },
        },
        {
            "id": "a0r2_budget_sensitivity",
            "algorithm_revision": "a0r2_without_h8v1_coverage_constraint",
            "dataset": "Alibaba-CP100-A0 controlled mechanism dataset",
            "seeds": list(SEEDS),
            "evidence_level": "B_formal_previous_revision",
            "usable_for": "Supporting budget trend for the previous OUR revision",
            "not_usable_for": "Exact h8v1 B5/B10 performance",
            "result": a0r2["static"],
        },
        {
            "id": "h8v0_h8v1_development_comparison",
            "algorithm_revision": "h8v0_vs_h8v1",
            "dataset": "Alibaba-CP100-A0 controlled mechanism dataset",
            "seeds": [1, 2, 3],
            "evidence_level": "C_development_mechanism",
            "usable_for": "Development-only coverage-constraint mechanism discussion",
            "not_usable_for": "Independent ablation significance",
        },
        {
            "id": "hsr1_static_ablation",
            "algorithm_revision": "hsr1_pre_h8v1",
            "dataset": "Earlier static heterogeneous environment",
            "seeds": ablation["seeds"],
            "evidence_level": "C_three_seed_ablation_previous_revision",
            "usable_for": "Qualitative coordinated-cache contribution",
            "not_usable_for": "Exact h8v1 module attribution",
            "result": ablation["comparisons_full_our_vs_ablation"],
        },
        {
            "id": "alibaba_a2_algorithm_extensions",
            "algorithm_revision": "pre_h8v1",
            "dataset": "Original selected Alibaba-CP100",
            "seeds": extensions["seeds"],
            "evidence_level": "C_three_seed_mechanism",
            "usable_for": "PD3QN and strong-central-baseline mechanism discussion",
            "not_usable_for": "Formal h8v1 superiority",
            "result": extensions["comparisons"],
        },
        {
            "id": "alibaba_a2_user_scaling",
            "algorithm_revision": "pre_h8v1_frozen_checkpoint",
            "dataset": "Original selected Alibaba-CP100",
            "seeds": scaling["seeds"],
            "evidence_level": "C_three_seed_scalability_diagnostic",
            "usable_for": "Runtime and user-count trend",
            "not_usable_for": "Formal h8v1 scalability claim",
            "result": {
                users: {
                    "finish_time": value["finish_time"],
                    "p95_finish_time": value["p95_finish_time"],
                }
                for users, value in scaling["results"].items()
            },
        },
        {
            "id": "a0r2_dynamic_nhpp",
            "algorithm_revision": "a0r2_pre_h8v1",
            "dataset": "Alibaba-CP100-A0 controlled dynamic stream",
            "seeds": dynamic["seeds"],
            "evidence_level": "D_diagnostic_failed_gate",
            "usable_for": "Limitation and debugging discussion only",
            "not_usable_for": "Dynamic superiority or recovery claim",
            "result": {
                "gate": dynamic["gate"],
                "empirical_baseline_utilization_range": dynamic["integrity"][
                    "empirical_baseline_utilization_range"
                ],
            },
        },
    ]
    exact_current_cross_dataset = [
        entry
        for entry in entries
        if entry["algorithm_revision"] == "h8v1"
        and "A0" not in entry["dataset"]
    ]
    return {
        "status": "complete",
        "entries": entries,
        "decision": {
            "repeat_budget_sensitivity": False,
            "reason_not_to_repeat_budget": (
                "B5/B8/B10 already exist for a0r2; repeating them is lower priority than external validity."
            ),
            "exact_h8v1_cross_dataset_evidence_exists": bool(
                exact_current_cross_dataset
            ),
            "cross_dataset_validation_recommended": not bool(
                exact_current_cross_dataset
            ),
            "recommended_next_experiment": (
                "Freeze h8v1 and compare DAOC, Centralized-Greedy, OUR, and Oracle "
                "on a workload not used to design A0/h8v1, starting with three fresh development seeds."
            ),
        },
        "source_files": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }


def render_s8_report(path, summary):
    official = summary["official_s8_result"]["our_vs_central"]
    concentration = summary["action_concentration"]
    seed18 = summary["seed18"]
    maximum = seed18["max_capacity_server"]
    correlations = summary["seed_level_correlations"]
    influence = summary["leave_seed18_out_influence_diagnostic"]
    report = f"""# h8v1 S8 事后机制诊断

> 本报告只读取已锁定的 seeds 11--20 结果。它不改变正式门槛，不得用于删除 seed、重新调参或宣称新的独立验证。

## 正式结果保持不变

- OUR 相对 Centralized-Greedy 平均改善：`{official['mean_improvement_percent']:.3f}%`。
- 95% CI：`[{official['ci95_lower_sec']:.6f}, {official['ci95_upper_sec']:.6f}] s`。
- 单侧 Wilcoxon：`p={official['wilcoxon_one_sided_p']:.6f}`，胜出 `{official['wins']}/10` seed。
- 因 CI 跨 0 且 `p>=0.05`，S8 仍然不能宣称显著超过 Centralized-Greedy。

## 诊断一：不是普遍的动作集中

- Centralized-Greedy 的平均归一化 HHI 为 `{concentration['central_mean_normalized_hhi']:.4f}`。
- OUR 的平均归一化 HHI 为 `{concentration['our_mean_normalized_hhi']:.4f}`。
- OUR 在 `{summary['conclusions']['our_has_lower_action_hhi_in_seed_count']}/10` 个 seed 中动作更分散。

因此，“OUR 在 S8 中普遍将任务挤到少数节点”这个简单解释不成立。

## 诊断二：seed 18 是缓存--算力错配

- 容量为 3 的服务器是 server `{maximum['server_id']}`，主频仅 `{maximum['frequency_ghz']:.1f} GHz`，是第 `{maximum['frequency_rank_from_slowest']}` 慢的服务器。
- 它的静态背景等待为 `{1000 * maximum['background_wait_sec']:.2f} ms/task`，在 10 台服务器中最差。
- 因果遥测已将它的质量估计为 `{maximum['our_causal_quality']:.4f}`，但 OUR 仍将 `{100 * maximum['our_action_share']:.2f}%` 任务和 `{100 * maximum['our_cpu_share']:.2f}%` CPU 工作量送给它；Centralized-Greedy 的任务份额仅 `{100 * maximum['central_action_share']:.2f}%`。
- OUR 节省服务加载 `{seed18['service_loading_gain_sec']:.4f} s`，但计算时延增加 `{seed18['computing_penalty_sec']:.4f} s`，等待时延增加 `{seed18['waiting_penalty_sec']:.4f} s`，前驱链时延增加 `{seed18['predecessor_penalty_sec']:.4f} s`。
- 最终 OUR 在 seed 18 中慢 `{abs(seed18['finish_improvement_sec']):.4f} s`。

注意：当前静态环境的 waiting latency 直接按 `load*1e6/frequency` 计算，不是任务之间真实排队造成的。准确的问题是“缓存容量与计算质量错配后，策略过度追随本地缓存”，而不是“动态队列被挤爆”。

## 十 seed 关联证据

| 关联 | Spearman rho | p |
|---|---:|---:|
| 完成时间改善 vs 计算时延惩罚 | {correlations['finish_improvement_vs_computing_penalty']['rho']:.3f} | {correlations['finish_improvement_vs_computing_penalty']['p_two_sided']:.6f} |
| 完成时间改善 vs 前驱时延惩罚 | {correlations['finish_improvement_vs_predecessor_penalty']['rho']:.3f} | {correlations['finish_improvement_vs_predecessor_penalty']['p_two_sided']:.6f} |
| 完成时间改善 vs 服务加载收益 | {correlations['finish_improvement_vs_service_loading_gain']['rho']:.3f} | {correlations['finish_improvement_vs_service_loading_gain']['p_two_sided']:.6f} |
| 完成时间改善 vs 动作 HHI 变化 | {correlations['finish_improvement_vs_action_hhi_delta']['rho']:.3f} | {correlations['finish_improvement_vs_action_hhi_delta']['p_two_sided']:.6f} |

这说明 S8 成败更直接地取决于“服务加载节省”能否盖过“慢节点的计算与前驱链惩罚”，而不是总体动作是否更集中。

## 影响度检查

仅作事后影响度分析，若排除 seed 18，剩余 9 seed 的平均改善为 `{influence['mean_improvement_sec']:.4f} s`，95% CI 为 `[{influence['ci95_lower_sec']:.4f}, {influence['ci95_upper_sec']:.4f}]`。**seed 18 不能从正式结果中删除**；该检查只用于确认不确定性的来源。

## 结论边界

1. 保留正式结论：OUR 在 S8 平均更好，但未达统计显著。
2. 不再将 S8 问题笼统表述为“流量集中导致排队”。
3. 更准确的局限是：极端容量异构下，缓存富集节点可能恰好是慢计算节点，当前策略对这种错配的抑制不足。
4. 不使用这批 seed 调参；若日后开发拥塞感知新版本，必须换新的开发和确认 seed。
"""
    Path(path).write_text(report, encoding="utf-8")


def render_evidence_report(path, audit):
    rows = []
    for entry in audit["entries"]:
        rows.append(
            f"| {entry['id']} | {entry['algorithm_revision']} | "
            f"{entry['evidence_level']} | {entry['usable_for']} |"
        )
    decision = audit["decision"]
    report = """# 边缘卸载与异构缓存实验证据审计

> 本审计用于防止不同算法版本、数据集和 seed 分区的结果被混用。

| 证据 | 算法版本 | 等级 | 可支撑内容 |
|---|---|---|---|
""" + "\n".join(rows) + f"""

## 审计结论

1. `h8v1` 的十 seed U8/M8/S8 是当前唯一组可作正式主结论的精确版本证据。
2. `a0r2` 已经完成 B5/B8/B10，因此不建议重复预算敏感性；但它没有 `h8v1` 的稀缺性覆盖约束，只能作旧版本补充证据。
3. 已有消融和用户规模实验均是三 seed 且属于旧版本，适合机制解释，不能当作 `h8v1` 的正式显著性证据。
4. 旧 NHPP 动态实验未通过利用率和正式统计门槛，只能作诊断。
5. 当前没有“精确 `h8v1` + 非 A0 新工作负载”的跨数据集证据。

## 下一个有信息量的实验

- 是否重跑预算敏感性：`{decision['repeat_budget_sensitivity']}`。
- 是否建议跨数据集验证：`{decision['cross_dataset_validation_recommended']}`。
- 建议：锁定 `h8v1`，先在一个未用于 A0/h8v1 设计的工作负载上使用 3 个全新 seed，从头训练 DAOC、Centralized-Greedy 和 OUR，并计算 Oracle。三 seed 方向一致后再决定是否扩展。
"""
    Path(path).write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    suite_root = args.revision_root / "final/S8"
    if not suite_root.exists():
        raise FileNotFoundError(suite_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_records = []
    episode_pairs = []
    for seed in SEEDS:
        seed_record, pairs = collect_seed(suite_root, seed)
        seed_records.append(seed_record)
        episode_pairs.extend(pairs)

    summary = build_summary(args.revision_root, seed_records, episode_pairs)
    audit = audit_evidence(repo_root)
    write_json(args.output_dir / "s8_concentration_summary.json", summary)
    write_json(args.output_dir / "evidence_audit.json", audit)
    write_csv(
        args.output_dir / "s8_seed_diagnosis.csv",
        [flatten_seed_record(record) for record in seed_records],
    )
    write_csv(args.output_dir / "s8_episode_pairs.csv", episode_pairs)
    plot_diagnosis(args.output_dir, seed_records, summary)
    render_s8_report(args.output_dir / "S8_DIAGNOSIS_ZH.md", summary)
    render_evidence_report(args.output_dir / "EVIDENCE_AUDIT_ZH.md", audit)

    manifest = {
        "status": "complete",
        "analysis_type": "posthoc_read_only",
        "algorithm_modified": False,
        "final_lock_modified": False,
        "formal_gate_modified": False,
        "inputs": {
            "revision_root": str(args.revision_root.resolve()),
            "seeds": list(SEEDS),
            "scenarios_per_seed": EXPECTED_EVAL_EPISODES,
        },
        "outputs": sorted(
            path.name for path in args.output_dir.iterdir() if path.is_file()
        ),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "POSTHOC_MANIFEST.json", manifest)
    print(f"Post-hoc diagnosis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
