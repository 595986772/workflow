#!/usr/bin/env python3
"""Audit, analyze, visualize, and diagnose E2/E3 experiments."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze_strict_environment_suite import paired_statistics
from capacity_protocol import CAPACITY_PROTOCOL_VERSION
from information_protocol import (
    CAUSAL_CACHE_INFORMATION_REGIME,
    INFORMATION_PROTOCOL_VERSION,
)
from run_independent_experiment import SCENARIO_FINGERPRINT_VERSION


EXPECTED_CAPACITIES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
DEFAULT_DAOC_LABEL = "guided_full"
DEFAULT_OUR_LABEL = "lean_our"
REVISION_METRICS = {
    "paired_finish_time": ("e2", "average_finish_time", True),
    "average_finish_time": ("e2", "average_finish_time", True),
    "p95_finish_time": ("e2", "p95_finish_time", True),
    "remote_loading_rate": (
        "e2",
        "cache_remote_loading_rate",
        True,
    ),
    "service_coverage": ("e2", "cache_service_coverage", False),
    "migration_time": ("e2", "cache_migration_time_sec", True),
    "e3_cumulative_oracle_regret": (
        "e3",
        "e3_cumulative_oracle_regret",
        True,
    ),
    "e3_adaptation_delay": (
        "e3",
        "e3_adaptation_delay",
        True,
    ),
}


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze paired heterogeneous-cache E2/E3 results."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--daoc-label", default=DEFAULT_DAOC_LABEL)
    parser.add_argument("--our-label", default=DEFAULT_OUR_LABEL)
    parser.add_argument("--parent-suite-dir", type=Path)
    parser.add_argument(
        "--expected-metric",
        choices=REVISION_METRICS,
        default="paired_finish_time",
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def evaluation_rows(run_dir):
    return [
        row
        for row in read_csv(run_dir / "episodes.csv")
        if row["phase"] == "eval"
    ]


def run_dir(suite_dir, label, seed):
    return suite_dir / "runs" / label / f"seed_{seed}"


def mean_metric(rows, metric):
    return float(np.mean([float(row[metric]) for row in rows]))


def validate_capacity_row(
    row,
    expected_capacities=EXPECTED_CAPACITIES,
):
    capacities = {
        int(server_id): int(value)
        for server_id, value
        in json.loads(row["cache_capacity_vector_json"]).items()
    }
    matrix = {
        int(server_id): [int(service) for service in services]
        for server_id, services
        in json.loads(row["cache_matrix_json"]).items()
    }
    checks = {
        "multiset": (
            sorted(capacities.values())
            == sorted(expected_capacities)
        ),
        "total_slots": (
            sum(capacities.values())
            == sum(expected_capacities)
        ),
        "all_capacity_constraints": all(
            len(matrix[server_id]) <= capacity
            for server_id, capacity in capacities.items()
        ),
        "zero_capacity_empty": all(
            matrix[server_id] == []
            for server_id, capacity in capacities.items()
            if capacity == 0
        ),
        "real_services_only": all(
            1 <= service <= 10
            for services in matrix.values()
            for service in services
        ),
    }
    return capacities, matrix, checks


def collect_pair(
    suite_dir,
    seed,
    daoc_label,
    our_label,
    expected_capacities=EXPECTED_CAPACITIES,
):
    directories = {
        "daoc": run_dir(suite_dir, daoc_label, seed),
        "our": run_dir(suite_dir, our_label, seed),
    }
    summaries = {
        method: read_json(path / "summary.json")
        for method, path in directories.items()
    }
    configs = {
        method: read_json(path / "config.json")
        for method, path in directories.items()
    }
    rows = {
        method: evaluation_rows(path)
        for method, path in directories.items()
    }
    if not rows["daoc"] or len(rows["daoc"]) != len(rows["our"]):
        raise RuntimeError(f"Incomplete paired E2 rows for seed {seed}")
    if [
        row["scenario_fingerprint"] for row in rows["daoc"]
    ] != [
        row["scenario_fingerprint"] for row in rows["our"]
    ]:
        raise RuntimeError(f"E2 scenario mismatch for seed {seed}")

    capacity_checks = []
    reference_vectors = []
    for method in ("daoc", "our"):
        for row in rows[method]:
            capacities, _, checks = validate_capacity_row(
                row,
                expected_capacities=expected_capacities,
            )
            capacity_checks.append(checks)
            reference_vectors.append(capacities)
    if any(
        vector != reference_vectors[0]
        for vector in reference_vectors[1:]
    ):
        raise RuntimeError(
            f"Capacity assignment changed across paired seed {seed}"
        )

    shared_argument_keys = (
        "num_users",
        "num_servers",
        "num_services",
        "num_tasks",
        "dag_dataset_path",
        "dag_dataset_sha256",
        "bandwidth",
        "eval_episodes",
        "eval_seed_offset",
        "server_capacity_multiset",
        "capacity_assignment_namespace",
        "baseline_server_capacity",
    )
    arguments_paired = all(
        configs["daoc"]["arguments"].get(key)
        == configs["our"]["arguments"].get(key)
        for key in shared_argument_keys
    )
    protocol_checks = {
        "information_protocol_match": all(
            summary["information_protocol_version"]
            == INFORMATION_PROTOCOL_VERSION
            for summary in summaries.values()
        ),
        "capacity_protocol_match": all(
            summary["capacity_protocol_version"]
            == CAPACITY_PROTOCOL_VERSION
            for summary in summaries.values()
        ),
        "full_dag_workload_fingerprinted": all(
            summary.get("scenario_fingerprint_version")
            == SCENARIO_FINGERPRINT_VERSION
            for summary in summaries.values()
        ),
        "configuration_hash_present": all(
            summary.get("experiment_config_sha256")
            == configs[method].get("experiment_config_sha256")
            for method, summary in summaries.items()
        ),
        "our_cache_history_only": (
            summaries["our"]["cache_information_regime"]
            == CAUSAL_CACHE_INFORMATION_REGIME
        ),
        "arguments_paired": arguments_paired,
        "capacity_constraints": all(
            all(check.values()) for check in capacity_checks
        ),
    }
    metrics = {"seed": seed}
    for method in ("daoc", "our"):
        for metric in (
            "average_finish_time",
            "p95_finish_time",
            "cache_hit_rate",
            "cache_remote_loading_rate",
            "cache_zero_capacity_assignment_rate",
            "cache_capacity_utilization",
            "cache_service_coverage",
            "cache_migration_time_sec",
            "cache_migration_events",
        ):
            metrics[f"{method}_{metric}"] = mean_metric(
                rows[method],
                metric,
            )
        metrics[f"{method}_converged"] = bool(
            summaries[method]["convergence"]["reached"]
            if summaries[method]["convergence"]["enabled"]
            else True
        )
    return {
        "metrics": metrics,
        "rows": rows,
        "summaries": summaries,
        "configs": configs,
        "protocol_checks": protocol_checks,
        "capacities": reference_vectors[0],
    }


def empirical_capacity_correlations(pairs):
    capacities = []
    properties = {
        "frequency": [],
        "load": [],
        "position_x": [],
        "position_y": [],
        "rate_to_cloud": [],
    }
    for pair in pairs:
        seed = pair["metrics"]["seed"]
        scenario = read_json(
            run_dir(
                pair["suite_dir"],
                pair["our_label"],
                seed,
            )
            / "scenario_initial.json"
        )
        for server_id, capacity in sorted(
            pair["capacities"].items()
        ):
            server = scenario["servers"][str(server_id)]
            capacities.append(capacity)
            properties["frequency"].append(server["frequency"])
            properties["load"].append(server["load"])
            properties["position_x"].append(server["position"][0])
            properties["position_y"].append(server["position"][1])
            properties["rate_to_cloud"].append(
                server["rate_to_cloud"]
            )
    result = {}
    for name, values in properties.items():
        pearson = stats.pearsonr(capacities, values)
        spearman = stats.spearmanr(capacities, values)
        result[name] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    return result


def metric_comparisons(per_seed):
    definitions = {
        "finish_time": ("average_finish_time", True),
        "p95_finish_time": ("p95_finish_time", True),
        "cache_hit_rate": ("cache_hit_rate", False),
        "remote_loading_rate": (
            "cache_remote_loading_rate",
            True,
        ),
        "zero_capacity_assignment_rate": (
            "cache_zero_capacity_assignment_rate",
            True,
        ),
        "service_coverage": ("cache_service_coverage", False),
        "migration_time": ("cache_migration_time_sec", True),
    }
    comparisons = {}
    for name, (metric, lower_is_better) in definitions.items():
        comparisons[name] = paired_statistics(
            [row[f"daoc_{metric}"] for row in per_seed],
            [row[f"our_{metric}"] for row in per_seed],
            lower_is_better=lower_is_better,
        )
    return comparisons


def compare_revision(
    suite_dir,
    parent_suite_dir,
    seeds,
    our_label,
    expected_metric,
):
    """Compare a revision against its parent on identical E2 scenarios."""
    metric_definitions = {
        "average_finish_time": True,
        "p95_finish_time": True,
        "cache_remote_loading_rate": True,
        "cache_service_coverage": False,
        "cache_migration_time_sec": True,
    }
    values = {
        metric: {"parent": [], "candidate": []}
        for metric in metric_definitions
    }
    dynamic_available = all(
        (
            run_dir(root, our_label, seed)
            / "online_stream_e3_load_shift_summary.json"
        ).exists()
        and (
            run_dir(root, our_label, seed)
            / "online_stream_e3_load_shift.csv"
        ).exists()
        for root in (parent_suite_dir, suite_dir)
        for seed in seeds
    )
    if dynamic_available:
        values.update(
            {
                "e3_cumulative_oracle_regret": {
                    "parent": [],
                    "candidate": [],
                },
                "e3_adaptation_delay": {
                    "parent": [],
                    "candidate": [],
                },
            }
        )
    scenario_pairs = 0
    for seed in seeds:
        parent_rows = evaluation_rows(
            run_dir(parent_suite_dir, our_label, seed)
        )
        candidate_rows = evaluation_rows(
            run_dir(suite_dir, our_label, seed)
        )
        parent_fingerprints = [
            row["scenario_fingerprint"] for row in parent_rows
        ]
        candidate_fingerprints = [
            row["scenario_fingerprint"] for row in candidate_rows
        ]
        if parent_fingerprints != candidate_fingerprints:
            raise RuntimeError(
                f"Revision scenarios differ for seed {seed}"
            )
        scenario_pairs += len(parent_rows)
        for metric in metric_definitions:
            values[metric]["parent"].append(
                mean_metric(parent_rows, metric)
            )
            values[metric]["candidate"].append(
                mean_metric(candidate_rows, metric)
            )
        if dynamic_available:
            dynamic_summaries = {}
            for version, root in (
                ("parent", parent_suite_dir),
                ("candidate", suite_dir),
            ):
                online_run = run_dir(root, our_label, seed)
                online_rows = read_csv(
                    online_run / "online_stream_e3_load_shift.csv"
                )
                dynamic_summaries[version] = read_json(
                    online_run
                    / "online_stream_e3_load_shift_summary.json"
                )
                if version == "parent":
                    parent_online_fingerprints = [
                        row["scenario_fingerprint"]
                        for row in online_rows
                    ]
                elif [
                    row["scenario_fingerprint"] for row in online_rows
                ] != parent_online_fingerprints:
                    raise RuntimeError(
                        f"Revision E3 scenarios differ for seed {seed}"
                    )
            for version, summary in dynamic_summaries.items():
                values[
                    "e3_cumulative_oracle_regret"
                ][version].append(
                    float(
                        summary["adaptation"][
                            "cumulative_oracle_regret"
                        ]
                    )
                )
                delay = summary["adaptation"][
                    "adaptation_delay_windows"
                ]
                if delay is None:
                    protocol = summary["protocol"]
                    delay = (
                        protocol["episodes"]
                        - protocol["shift_episode"]
                        + 2
                    )
                values["e3_adaptation_delay"][version].append(
                    float(delay)
                )

    comparisons = {
        metric: paired_statistics(
            metric_values["parent"],
            metric_values["candidate"],
            lower_is_better=(
                metric_definitions.get(metric, True)
            ),
        )
        for metric, metric_values in values.items()
    }
    _, expected_field, _ = REVISION_METRICS[expected_metric]
    if expected_field not in comparisons:
        raise RuntimeError(
            f"Revision metric {expected_metric} requires E3 artifacts"
        )
    expected = comparisons[expected_field]
    required_wins = math.ceil(2 * len(seeds) / 3)
    primary_non_degradation = {
        metric: (
            comparison["candidate_mean"]
            <= 1.01 * comparison["reference_mean"]
        )
        for metric, comparison in comparisons.items()
        if metric in {"average_finish_time", "p95_finish_time"}
        and metric != expected_field
    }
    retained = (
        expected["mean_paired_improvement_percent"] > 0
        and expected["wins"] >= required_wins
        and all(primary_non_degradation.values())
    )
    return {
        "parent_suite_dir": str(parent_suite_dir),
        "scenario_pairs": scenario_pairs,
        "all_scenarios_paired": True,
        "dynamic_scenarios_compared": dynamic_available,
        "expected_metric": expected_metric,
        "expected_metric_field": expected_field,
        "required_seed_wins": required_wins,
        "comparisons": comparisons,
        "primary_non_degradation_within_1pct": (
            primary_non_degradation
        ),
        "retained": retained,
    }


def plot_cache_heatmaps(path, pairs):
    matrices = {"daoc": [], "our": []}
    capacity_rows = []
    for pair in pairs:
        for method in ("daoc", "our"):
            for row in pair["rows"][method]:
                capacities = {
                    int(server_id): int(value)
                    for server_id, value
                    in json.loads(
                        row["cache_capacity_vector_json"]
                    ).items()
                }
                matrix = {
                    int(server_id): services
                    for server_id, services
                    in json.loads(row["cache_matrix_json"]).items()
                }
                order = sorted(
                    capacities,
                    key=lambda server_id: (
                        capacities[server_id],
                        server_id,
                    ),
                )
                occupancy = np.zeros((10, 10), dtype=float)
                for row_index, server_id in enumerate(order):
                    for service_id in matrix[server_id]:
                        occupancy[row_index, int(service_id) - 1] = 1.0
                matrices[method].append(occupancy)
                if method == "our":
                    capacity_rows.append(
                        [capacities[server_id] for server_id in order]
                    )
    mean_matrices = {
        method: np.mean(values, axis=0)
        for method, values in matrices.items()
    }
    mean_capacities = np.mean(capacity_rows, axis=0)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    image = None
    for axis, method, title in zip(
        axes,
        ("daoc", "our"),
        ("DAOC", "OUR"),
    ):
        image = axis.imshow(
            mean_matrices[method],
            vmin=0,
            vmax=1,
            cmap="viridis",
            aspect="auto",
        )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Service")
        axis.set_xticks(range(10), range(1, 11))
        axis.set_yticks(
            range(10),
            [
                f"K={int(round(capacity))}"
                for capacity in mean_capacities
            ],
        )
    axes[0].set_ylabel("Servers sorted by capacity")
    figure.colorbar(
        image,
        ax=axes,
        label="Cache occupancy probability",
        shrink=0.88,
    )
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def plot_e3_sensitivity(path, sensitivity):
    if not sensitivity:
        return
    multipliers = sorted(float(value) for value in sensitivity)
    definitions = (
        (
            "post_shift_early_finish_time",
            "First 10 windows mean",
            "o",
        ),
        ("post_shift_early_p95", "First 10 windows P95", "s"),
        (
            "cumulative_oracle_regret",
            "Cumulative Oracle regret",
            "^",
        ),
    )
    figure, axis = plt.subplots(
        figsize=(7.2, 4.6),
        constrained_layout=True,
    )
    for metric, label, marker in definitions:
        values = [
            sensitivity[str(int(multiplier))][metric][
                "mean_paired_improvement_percent"
            ]
            for multiplier in multipliers
        ]
        axis.plot(
            multipliers,
            values,
            marker=marker,
            linewidth=1.8,
            label=label,
        )
    axis.axhline(0, color="#444444", linewidth=1)
    axis.set_xticks(multipliers)
    axis.set_xlabel("Server load multiplier")
    axis.set_ylabel("OUR improvement over DAOC (%)")
    axis.set_title("E3-S load-shift sensitivity", loc="left")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def choose_diagnosis(per_seed, comparisons, oracle_summary):
    if not all(
        row["daoc_converged"] and row["our_converged"]
        for row in per_seed
    ):
        return {
            "symptom": "model_not_converged",
            "module_id": "optimizer_schedule",
            "check": "validation curve, TD loss, epsilon, update rate",
            "allowed_module": "optimizer schedule only",
            "expected_metric": "paired_finish_time",
            "rejection_condition": "no plateau or worse frozen evaluation",
        }

    daoc_zero = np.mean(
        [
            row["daoc_cache_zero_capacity_assignment_rate"]
            for row in per_seed
        ]
    )
    our_zero = np.mean(
        [
            row["our_cache_zero_capacity_assignment_rate"]
            for row in per_seed
        ]
    )
    if our_zero > max(0.25, daoc_zero + 0.05):
        return {
            "symptom": "excessive_zero_capacity_remote_loading",
            "module_id": "actor_server_features",
            "check": "server action histogram and capacity features",
            "allowed_module": "actor server features",
            "expected_metric": "remote_loading_rate",
            "rejection_condition": "finish time or P95 fails to improve",
        }

    finish = comparisons["finish_time"]
    hit = comparisons["cache_hit_rate"]
    if (
        finish["mean_paired_improvement_percent"] <= 0
        and hit["mean_paired_improvement_percent"] > 0
    ):
        return {
            "symptom": "higher_hit_rate_but_worse_completion",
            "module_id": "cache_value_function",
            "check": "replica location, dependency traffic, slow servers",
            "allowed_module": "cache value function",
            "expected_metric": "paired_finish_time",
            "rejection_condition": "primary latency does not improve",
        }

    coverage = comparisons["service_coverage"]
    if coverage["mean_paired_improvement_percent"] < -5.0:
        return {
            "symptom": "low_service_coverage",
            "module_id": "replica_marginal_gain",
            "check": "replica counts and duplicate popular services",
            "allowed_module": "replica marginal gain",
            "expected_metric": "service_coverage",
            "rejection_condition": "latency or P95 degrades over 1%",
        }

    our_migrations = np.mean(
        [row["our_cache_migration_events"] for row in per_seed]
    )
    if our_migrations > 1.0:
        return {
            "symptom": "cache_churn",
            "module_id": "cache_stability",
            "check": "migration count, residence time, value margin",
            "allowed_module": "hysteresis or minimum residence",
            "expected_metric": "migration_time",
            "rejection_condition": "E2 finish time degrades over 1%",
        }

    if oracle_summary is not None:
        reducible = oracle_summary["gap_analysis"][
            "our_reducible_fraction_percent"
        ]
        if (
            finish["mean_paired_improvement_percent"] <= 0
            and reducible < 3.0
        ):
            return {
                "symptom": "insufficient_oracle_headroom",
                "module_id": "none",
                "check": "OUR-to-Oracle and DAOC-to-Oracle gaps",
                "allowed_module": "none",
                "expected_metric": "paired_finish_time",
                "rejection_condition": "do not add another algorithm module",
            }

    if len(per_seed) > 1:
        values = np.asarray(
            [row["our_average_finish_time"] for row in per_seed]
        )
        if float(values.std(ddof=1) / values.mean()) > 0.1:
            return {
                "symptom": "high_seed_variance",
                "module_id": "none",
                "check": "paired seed trajectories and confidence interval",
                "allowed_module": "none; report uncertainty without tuning",
                "expected_metric": "paired_finish_time",
                "rejection_condition": "never tune to one seed",
            }

    if finish["mean_paired_improvement_percent"] <= 0:
        return {
            "symptom": "primary_metric_not_improved",
            "module_id": "diagnosis_selected_module",
            "check": "latency decomposition and Oracle headroom",
            "allowed_module": "one module selected by decomposition",
            "expected_metric": "paired_finish_time",
            "rejection_condition": "other primary metric degrades over 1%",
        }
    return {
        "symptom": "none",
        "module_id": "none",
        "check": "proceed to the next seed partition",
        "allowed_module": "none",
        "expected_metric": "paired_finish_time",
        "rejection_condition": "the next protocol gate reverses direction",
    }


def choose_e3_diagnosis(suite_dir, seeds, our_label):
    rows = []
    for seed in seeds:
        rows.extend(
            read_csv(
                run_dir(suite_dir, our_label, seed)
                / "online_stream_e3_load_shift.csv"
            )
        )
    post_rows = [
        row for row in rows if row["stream_segment"] == "post_shift"
    ]
    ema_values = []
    observed_values = []
    for row in post_rows:
        ema = json.loads(
            row["pre_window_execution_latency_ema_json"]
        )
        observed = json.loads(
            row["observed_server_execution_latency_json"]
        )
        for server_id, value in ema.items():
            actual = observed.get(server_id)
            if value is not None and actual is not None:
                ema_values.append(float(value))
                observed_values.append(float(actual))
    if (
        len(ema_values) >= 3
        and np.std(ema_values) > 0
        and np.std(observed_values) > 0
    ):
        ema_rank_correlation = float(
            stats.spearmanr(ema_values, observed_values).statistic
        )
        if not np.isfinite(ema_rank_correlation):
            ema_rank_correlation = None
    else:
        ema_rank_correlation = None

    def shifted_action_share(selected_rows):
        assigned = 0
        total = 0
        for row in selected_rows:
            shifted = {
                int(server_id)
                for server_id in row["shifted_server_ids"].split(",")
                if server_id
            }
            histogram = {
                int(server_id): int(count)
                for server_id, count
                in json.loads(
                    row["server_action_histogram_json"]
                ).items()
            }
            assigned += sum(
                histogram.get(server_id, 0)
                for server_id in shifted
            )
            total += sum(histogram.values())
        return assigned / total if total else 0.0

    pre_rows = [
        row for row in rows if row["stream_segment"] == "pre_shift"
    ]
    early_post_rows = []
    for seed in seeds:
        seed_post_rows = [
            row
            for row in read_csv(
                run_dir(suite_dir, our_label, seed)
                / "online_stream_e3_load_shift.csv"
            )
            if row["stream_segment"] == "post_shift"
        ]
        early_post_rows.extend(seed_post_rows[:10])
    pre_share = shifted_action_share(pre_rows)
    post_share = shifted_action_share(early_post_rows)
    post_decisions = sum(
        int(float(row["cache_decision_calls"]))
        for row in post_rows
    )
    post_migrations = sum(
        int(float(row["cache_migration_events"]))
        for row in post_rows
    )
    post_migration_time = sum(
        float(row["cache_migration_critical_time_sec"])
        for row in post_rows
    )
    post_finish_total = sum(
        float(row["migration_adjusted_finish_time"])
        for row in post_rows
    )
    diagnostics = {
        "ema_observed_spearman": ema_rank_correlation,
        "pre_shifted_server_action_share": pre_share,
        "early_post_shifted_server_action_share": post_share,
        "post_cache_decision_calls": post_decisions,
        "post_cache_migration_events": post_migrations,
        "migration_to_finish_ratio": (
            post_migration_time / post_finish_total
            if post_finish_total > 0
            else 0.0
        ),
    }
    if (
        ema_rank_correlation is None
        or ema_rank_correlation < 0.3
    ):
        return {
            "symptom": "e3_telemetry_ranking_is_weak",
            "module_id": "telemetry_estimator",
            "check": "EMA coefficient, sample count, and freshness",
            "allowed_module": "causal telemetry estimator only",
            "expected_metric": "e3_cumulative_oracle_regret",
            "rejection_condition": (
                "E3 regret does not improve or E2 degrades over 1%"
            ),
            "e3_diagnostics": diagnostics,
        }
    if post_share >= 0.95 * pre_share:
        return {
            "symptom": "e3_actor_does_not_respond_to_valid_telemetry",
            "module_id": "actor_telemetry_features",
            "check": "current telemetry channels in the actor state",
            "allowed_module": "actor telemetry features only",
            "expected_metric": "e3_adaptation_delay",
            "rejection_condition": (
                "shifted-server actions do not fall or E2 degrades over 1%"
            ),
            "e3_diagnostics": diagnostics,
        }
    minimum_decisions = max(1, len(post_rows) // 10)
    if post_decisions < minimum_decisions:
        return {
            "symptom": "e3_cache_adaptation_is_too_slow",
            "module_id": "cache_update_interval",
            "check": "cache update interval with migration cost included",
            "allowed_module": "cache update interval only",
            "expected_metric": "e3_adaptation_delay",
            "rejection_condition": (
                "recovery does not improve or E2 degrades over 1%"
            ),
            "e3_diagnostics": diagnostics,
        }
    if post_migrations > 5 * post_decisions:
        return {
            "symptom": "e3_cache_oscillation",
            "module_id": "cache_stability",
            "check": "hysteresis and minimum residence time",
            "allowed_module": "cache stability gate only",
            "expected_metric": "e3_cumulative_oracle_regret",
            "rejection_condition": (
                "migration falls but latency or E2 degrades over 1%"
            ),
            "e3_diagnostics": diagnostics,
        }
    if diagnostics["migration_to_finish_ratio"] > 0.1:
        return {
            "symptom": "e3_adaptation_overhead_is_high",
            "module_id": "coordination_frequency",
            "check": "coordination frequency and greedy solve cost",
            "allowed_module": "coordination frequency only",
            "expected_metric": "e3_cumulative_oracle_regret",
            "rejection_condition": (
                "overhead falls but recovery or E2 degrades over 1%"
            ),
            "e3_diagnostics": diagnostics,
        }
    return {
        "symptom": "e3_primary_gate_failed",
        "module_id": "diagnosis_selected_module",
        "check": "paired load-shift trajectories and latency decomposition",
        "allowed_module": "one module selected by E3 decomposition",
        "expected_metric": "e3_cumulative_oracle_regret",
        "rejection_condition": "E3 does not improve or E2 degrades over 1%",
        "e3_diagnostics": diagnostics,
    }


def write_report(path, summary):
    finish = summary["e2"]["comparisons"]["finish_time"]
    p95 = summary["e2"]["comparisons"]["p95_finish_time"]
    diagnosis = summary["diagnosis"]
    lines = [
        "# E2/E3 Closed-Loop Report",
        "",
        "## Integrity",
        "",
        f"- Capacity/fairness audit: {summary['integrity']['all_passed']}.",
        f"- Capacity multiset: `{EXPECTED_CAPACITIES}`; total slots: 10.",
        "- K=0 servers remain compute-capable and hold no real service.",
        "- Capacity assignment uses an RNG namespace independent of deployment generation.",
        "",
        "## E2",
        "",
        f"- OUR paired finish-time improvement: {finish['mean_paired_improvement_percent']:.3f}%.",
        f"- Seed wins: {finish['wins']}/{finish['pairs']}.",
        f"- OUR paired P95 improvement: {p95['mean_paired_improvement_percent']:.3f}%.",
        f"- Stage gate: {summary['e2']['gate_passed']}.",
    ]
    if summary.get("e3") is not None:
        e3 = summary["e3"]
        early_p95 = e3["summary"]["comparisons"][
            "post_shift_early_p95"
        ]
        lines.extend(
            [
                "",
                "## E3",
                "",
                f"- Dynamic integrity audit: {e3['integrity_passed']}.",
                "- First 10 post-shift-window P95 improvement: "
                f"{early_p95['mean_paired_improvement_percent']:.3f}%.",
                f"- Recovery-delay wins: {e3['recovery_wins']}/{e3['pairs']}.",
                f"- Oracle-regret wins: {e3['oracle_regret_wins']}/{e3['pairs']}.",
                f"- E3 gate: {e3['gate_passed']}.",
            ]
        )
    if summary.get("revision_comparison") is not None:
        revision = summary["revision_comparison"]
        lines.extend(
            [
                "",
                "## Revision Decision",
                "",
                f"- Parent suite: `{revision['parent_suite_dir']}`.",
                f"- Paired scenarios: {revision['scenario_pairs']}.",
                f"- Expected metric: `{revision['expected_metric']}`.",
                f"- Retain revision: {revision['retained']}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- Symptom: `{diagnosis['symptom']}`.",
            f"- Module ID: `{diagnosis['module_id']}`.",
            f"- First check: {diagnosis['check']}.",
            f"- Allowed change: {diagnosis['allowed_module']}.",
            f"- Expected effect: {diagnosis['expected_metric']}.",
            f"- Reject when: {diagnosis['rejection_condition']}.",
            "",
            "Only this first diagnosed module may be changed in the next revision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_zh(path, summary):
    finish = summary["e2"]["comparisons"]["finish_time"]
    p95 = summary["e2"]["comparisons"]["p95_finish_time"]
    diagnosis = summary["diagnosis"]
    lines = [
        "# E2/E3 闭环诊断报告",
        "",
        f"- 阶段：`{summary['stage']}`。",
        f"- 容量与公平性审计：{summary['integrity']['all_passed']}。",
        f"- OUR 平均完成时间配对改善：{finish['mean_paired_improvement_percent']:.3f}%。",
        f"- 完成时间获胜 seed：{finish['wins']}/{finish['pairs']}。",
        f"- OUR P95 配对改善：{p95['mean_paired_improvement_percent']:.3f}%。",
        f"- E2阶段门槛：{summary['e2']['gate_passed']}。",
    ]
    if summary.get("e3") is not None:
        e3 = summary["e3"]
        early = e3["summary"]["comparisons"][
            "post_shift_early_finish_time"
        ]
        lines.extend(
            [
                "",
                "## E3动态结果",
                "",
                f"- 突变后前10窗口平均时延改善：{early['mean_paired_improvement_percent']:.3f}%。",
                f"- 恢复时间获胜：{e3['recovery_wins']}/{e3['pairs']}。",
                f"- 累计Oracle regret获胜：{e3['oracle_regret_wins']}/{e3['pairs']}。",
                f"- E3阶段门槛：{e3['gate_passed']}。",
            ]
        )
    if summary.get("formal_final") is not None:
        formal = summary["formal_final"]
        lines.extend(
            [
                "",
                "## 十Seed最终确认",
                "",
                f"- 95% CI下界大于0：{formal['ci95_lower_positive']}。",
                "- 单侧Wilcoxon p<0.05："
                f"{formal['wilcoxon_one_sided_p_below_0_05']}。",
                f"- 至少7/10 seed获胜：{formal['at_least_7_of_10_seed_wins']}。",
                f"- 正式优势结论通过：{formal['superiority_claim_passed']}。",
                "- Seeds 1–3参与开发，因此该结果不是独立holdout。",
            ]
        )
    lines.extend(
        [
            "",
            "## 诊断",
            "",
            f"- 症状：`{diagnosis['symptom']}`。",
            f"- 允许调整模块：`{diagnosis['module_id']}`。",
            f"- 首要检查：{diagnosis['check']}。",
            f"- 拒绝条件：{diagnosis['rejection_condition']}。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    pairs = []
    for seed in args.seeds:
        pair = collect_pair(
            suite_dir,
            seed,
            args.daoc_label,
            args.our_label,
        )
        pair["suite_dir"] = suite_dir
        pair["our_label"] = args.our_label
        pairs.append(pair)
    per_seed = [pair["metrics"] for pair in pairs]
    comparisons = metric_comparisons(per_seed)
    protocol_checks = [
        value
        for pair in pairs
        for value in pair["protocol_checks"].values()
    ]
    integrity_passed = all(protocol_checks)
    correlations = empirical_capacity_correlations(pairs)
    finish = comparisons["finish_time"]
    p95 = comparisons["p95_finish_time"]
    all_converged = all(
        row["daoc_converged"] and row["our_converged"]
        for row in per_seed
    )
    if args.stage == "smoke":
        required_wins = 0
        e2_gate = integrity_passed
        e2_gate_definition = "integrity_only"
    elif args.stage == "screen":
        required_wins = 2
        e2_gate = (
            integrity_passed
            and finish["mean_paired_improvement_percent"] > 0
            and finish["wins"] >= required_wins
        )
        e2_gate_definition = (
            "positive_mean_finish_improvement_and_2_of_3_wins"
        )
    elif args.stage in {"converged", "dynamic", "ablation"}:
        required_wins = len(per_seed)
        e2_gate = (
            integrity_passed
            and all_converged
            and finish["wins"] == len(per_seed)
            and finish["mean_paired_improvement_percent"] >= 5.0
            and p95["candidate_mean"] < p95["reference_mean"]
        )
        e2_gate_definition = (
            "all_converged_3_of_3_finish_wins_mean_gain_at_least_"
            "5pct_and_better_mean_p95"
        )
    elif args.stage == "final":
        required_wins = 7
        e2_gate = (
            integrity_passed
            and all_converged
            and finish["wins"] >= required_wins
            and finish["paired_improvement_ci95_lower"] > 0
            and finish["wilcoxon_one_sided_p"] < 0.05
        )
        e2_gate_definition = (
            "converged_seed_level_ci_wilcoxon_and_7_of_10_wins"
        )
    else:
        raise ValueError(f"Unsupported E2/E3 analysis stage: {args.stage}")

    oracle_path = suite_dir / "oracle" / "oracle_floor_summary.json"
    oracle_summary = (
        read_json(oracle_path) if oracle_path.exists() else None
    )
    e3_summary_path = (
        suite_dir / "online_stream_e3_load_shift_summary.json"
    )
    e3 = None
    if e3_summary_path.exists():
        online = read_json(e3_summary_path)
        e3_required_wins = math.ceil(2 * len(per_seed) / 3)
        delay = online["comparisons"]["adaptation_delay"]
        regret = online["comparisons"][
            "cumulative_oracle_regret"
        ]
        e3_integrity = (
            online.get("all_scenario_fingerprints_match") is True
            and online.get(
                "all_oracle_capacity_constraints_satisfied"
            ) is True
            and online["protocol"].get(
                "migration_delay_carried_to_next_window"
            ) is True
        )
        e3 = {
            "pairs": delay["pairs"],
            "recovery_wins": delay["wins"],
            "oracle_regret_wins": regret["wins"],
            "required_seed_wins": e3_required_wins,
            "integrity_passed": e3_integrity,
            "gate_passed": (
                e3_integrity
                and delay["wins"] >= e3_required_wins
                and regret["wins"] >= e3_required_wins
            ),
            "summary": online,
        }

    e3_sensitivity = {}
    if args.stage == "final":
        for multiplier, suffix in (
            (2, "_x2"),
            (4, ""),
            (6, "_x6"),
        ):
            path = suite_dir / (
                "online_stream_e3_load_shift"
                f"{suffix}_summary.json"
            )
            if path.exists():
                sensitivity = read_json(path)
                e3_sensitivity[str(multiplier)] = {
                    "overall_finish_time": sensitivity[
                        "comparisons"
                    ]["overall_finish_time"],
                    "post_shift_early_finish_time": sensitivity[
                        "comparisons"
                    ]["post_shift_early_finish_time"],
                    "post_shift_early_p95": sensitivity[
                        "comparisons"
                    ]["post_shift_early_p95"],
                    "cumulative_oracle_regret": sensitivity[
                        "comparisons"
                    ]["cumulative_oracle_regret"],
                    "adaptation_delay": sensitivity[
                        "comparisons"
                    ]["adaptation_delay"],
                }

    revision_comparison = None
    if args.parent_suite_dir is not None:
        revision_comparison = compare_revision(
            suite_dir=suite_dir,
            parent_suite_dir=args.parent_suite_dir.resolve(),
            seeds=args.seeds,
            our_label=args.our_label,
            expected_metric=args.expected_metric,
        )

    formal_final = None
    if args.stage == "final":
        formal_final = {
            "ci95_lower_positive": (
                finish["paired_improvement_ci95_lower"] > 0
            ),
            "wilcoxon_one_sided_p_below_0_05": (
                finish["wilcoxon_one_sided_p"] < 0.05
            ),
            "at_least_7_of_10_seed_wins": finish["wins"] >= 7,
            "all_integrity_checks_passed": integrity_passed,
            "all_learning_methods_converged": all_converged,
            "independent_holdout": False,
        }
        claim_fields = (
            "ci95_lower_positive",
            "wilcoxon_one_sided_p_below_0_05",
            "at_least_7_of_10_seed_wins",
            "all_integrity_checks_passed",
            "all_learning_methods_converged",
        )
        formal_final["superiority_claim_passed"] = all(
            formal_final[field] for field in claim_fields
        )

    summary = {
        "status": "complete",
        "stage": args.stage,
        "seeds": args.seeds,
        "integrity": {
            "all_passed": integrity_passed,
            "checks": [
                pair["protocol_checks"] for pair in pairs
            ],
            "capacity_assignment_independent_by_construction": True,
            "empirical_correlations": correlations,
        },
        "e2": {
            "per_seed": per_seed,
            "comparisons": comparisons,
            "required_seed_wins": required_wins,
            "all_learning_methods_converged": all_converged,
            "gate_definition": e2_gate_definition,
            "gate_passed": e2_gate,
        },
        "e3": e3,
        "e3_sensitivity": e3_sensitivity,
        "revision_comparison": revision_comparison,
        "oracle": oracle_summary,
        "formal_final": formal_final,
        "formal_holdout": None,
    }
    summary["diagnosis"] = choose_diagnosis(
        per_seed,
        comparisons,
        oracle_summary,
    )
    if (
        e2_gate
        and e3 is not None
        and not e3["gate_passed"]
    ):
        summary["diagnosis"] = choose_e3_diagnosis(
            suite_dir,
            args.seeds,
            args.our_label,
        )
    write_json(suite_dir / "e2_e3_analysis.json", summary)
    with (suite_dir / "e2_per_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(per_seed[0]),
        )
        writer.writeheader()
        writer.writerows(per_seed)
    plot_cache_heatmaps(
        suite_dir / "e2_cache_heatmap",
        pairs,
    )
    plot_e3_sensitivity(
        suite_dir / "e3_sensitivity",
        e3_sensitivity,
    )
    write_report(
        suite_dir / "E2_E3_DIAGNOSTIC_REPORT.md",
        summary,
    )
    write_report_zh(
        suite_dir / "E2_E3_DIAGNOSTIC_REPORT_ZH.md",
        summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
