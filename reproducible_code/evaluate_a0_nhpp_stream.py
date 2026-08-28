#!/usr/bin/env python3
"""Evaluate frozen A0 checkpoints on a causal piecewise-NHPP stream."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from a0_coordination_protocol import (
    BASELINE_END_WINDOW,
    BASELINE_EXPECTED_ARRIVALS_PER_WINDOW,
    BASELINE_TARGET_UTILIZATION,
    BURST_END_WINDOW,
    BURST_RATE_MULTIPLIER,
    CAPACITY_PROFILES,
    DYNAMIC_WINDOWS,
    EXPECTED_DATASET_SHA256,
    FINAL_SEEDS,
    INGRESS_HOTSPOT_SHARE,
    MAIN_BUDGET,
    METHOD_LABELS,
    RECOVERY_ROLLING_WINDOWS,
    RECOVERY_TOLERANCE,
)
from oracle_latency_bound import scenario_capacity_aware_oracle_bounds
from run_independent_experiment import (
    apply_frozen_state,
    cache_snapshot,
    capture_deployment_state,
    scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
)
from simulator import MEC_Simulator


DYNAMIC_PROTOCOL_VERSION = "a0_piecewise_nhpp_fcfs_v5"
CALIBRATION_REPLICATES = 5
CALIBRATION_WINDOWS = 40
CALIBRATION_ARRIVALS_PER_WINDOW = 20
CALIBRATION_ARRIVAL_SPACING_SEC = 200.0
CALIBRATION_SEED_STRIDE = 100_003
CALIBRATION_WINDOW_DURATION_SEC = (
    CALIBRATION_ARRIVALS_PER_WINDOW
    * CALIBRATION_ARRIVAL_SPACING_SEC
)
DISPLAY_NAMES = {
    "guided_full": "DAOC",
    "centralized_greedy_daoc": "Centralized-Greedy",
    "lean_our": "OUR",
}
COLORS = {
    "guided_full": "#59636E",
    "centralized_greedy_daoc": "#E09F3E",
    "lean_our": "#277DA1",
    "oracle": "#2A9D8F",
}


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=parse_str_list, default=METHOD_LABELS)
    parser.add_argument("--seeds", type=parse_int_list, default=FINAL_SEEDS)
    parser.add_argument(
        "--mode",
        choices=("development", "final"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if set(args.labels) != set(METHOD_LABELS):
        raise ValueError("A0 dynamic evaluation requires all three methods")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_run(suite_dir, label, seed):
    return suite_dir / "runs" / label / f"seed_{seed}"


def mean_a0_dag_cycles(input_config):
    graphs = read_json(input_config["dag dataset path"])
    maximum = float(input_config["max_cpu_cycles"])
    totals = []
    for graph in graphs.values():
        totals.append(
            sum(
                int(maximum * float(node["cpucycle"])) * 1e6
                for node in graph["nodes"]
                if node.get("id") != "0" and float(node["service"]) > 0
            )
        )
    return float(np.mean(totals))


def closest_hotspot_sources(initial_scenario, seed):
    servers = initial_scenario["servers"]
    users = initial_scenario["users"]
    server_ids = sorted(int(server_id) for server_id in servers)
    target = server_ids[seed % len(server_ids)]
    server_position = np.asarray(
        servers[str(target)]["position"],
        dtype=float,
    )
    ranked = sorted(
        (
            float(
                np.linalg.norm(
                    np.asarray(user["initial_position"], dtype=float)
                    - server_position
                )
            ),
            int(user_id),
        )
        for user_id, user in users.items()
    )
    return target, [user_id for _, user_id in ranked[:3]]


def frozen_routing_fractions(run_dir, number_of_servers):
    """Estimate CPU-work routing shares from frozen static evaluations."""
    path = Path(run_dir) / "episodes.csv"
    counts = np.zeros(number_of_servers, dtype=float)
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]
    if len(rows) != 100:
        raise RuntimeError(
            f"Expected 100 frozen routing scenarios in {path}"
        )
    has_cpu_work = all(
        row.get("server_cpu_cycle_histogram_json")
        for row in rows
    )
    histogram_field = (
        "server_cpu_cycle_histogram_json"
        if has_cpu_work
        else "server_action_histogram_json"
    )
    for row in rows:
        histogram = json.loads(row[histogram_field])
        for server_id, value in histogram.items():
            server_id = int(server_id)
            if not 0 <= server_id < number_of_servers:
                raise RuntimeError(
                    f"Invalid server {server_id} in {path}"
                )
            counts[server_id] += float(value)
    total = float(counts.sum())
    if total <= 0.0:
        raise RuntimeError(f"No frozen routing actions in {path}")
    return counts / total, histogram_field


def bottleneck_rate_calibration(
    run_dirs,
    frequencies,
    mean_cycles,
    routing_profiles=None,
    routing_upper_bounds=None,
):
    """Choose one shared rate with worst-policy bottleneck load at 45%."""
    number_of_servers = len(frequencies)
    method_fractions = {}
    configured_sources = sum(
        source is not None
        for source in (
            run_dirs,
            routing_profiles,
            routing_upper_bounds,
        )
    )
    if configured_sources > 1:
        raise ValueError(
            "Use one calibration source only"
        )
    if routing_upper_bounds is not None:
        for label, profile in routing_upper_bounds.items():
            if isinstance(profile, dict):
                fractions = np.asarray(
                    [profile[str(server_id)] for server_id in range(number_of_servers)],
                    dtype=float,
                )
            else:
                fractions = np.asarray(profile, dtype=float)
            if (
                fractions.shape != (number_of_servers,)
                or np.any(fractions < 0.0)
                or np.any(fractions > 1.0)
                or fractions.sum() <= 0.0
            ):
                raise ValueError(
                    f"Invalid online routing upper bound: {label}"
                )
            method_fractions[label] = fractions
        calibration_source = (
            "independent_multi_stream_online_cache_cpu_work_upper_bound"
        )
        calibration_fields = {
            label: "online_cache_pilot_cpu_work_upper_bound"
            for label in method_fractions
        }
    elif routing_profiles is not None:
        for label, profile in routing_profiles.items():
            if isinstance(profile, dict):
                fractions = np.asarray(
                    [profile[str(server_id)] for server_id in range(number_of_servers)],
                    dtype=float,
                )
            else:
                fractions = np.asarray(profile, dtype=float)
            if (
                fractions.shape != (number_of_servers,)
                or np.any(fractions < 0.0)
                or not np.isclose(fractions.sum(), 1.0)
            ):
                raise ValueError(f"Invalid online routing profile: {label}")
            method_fractions[label] = fractions
        calibration_source = "independent_online_cache_pilot_cpu_work"
        calibration_fields = {
            label: "online_cache_pilot_cpu_work"
            for label in method_fractions
        }
    elif run_dirs is None:
        method_fractions["uniform_reference"] = np.full(
            number_of_servers,
            1.0 / number_of_servers,
            dtype=float,
        )
        calibration_source = "uniform_reference"
        calibration_fields = {
            "uniform_reference": "uniform_reference"
        }
    else:
        routing = {
            label: frozen_routing_fractions(
                run_dir,
                number_of_servers,
            )
            for label, run_dir in run_dirs.items()
        }
        method_fractions = {
            label: values[0]
            for label, values in routing.items()
        }
        calibration_source = (
            "frozen_static_evaluation_cpu_work_histograms"
            if all(
                values[1] == "server_cpu_cycle_histogram_json"
                for values in routing.values()
            )
            else "frozen_static_evaluation_action_histograms_fallback"
        )
        calibration_fields = {
            label: values[1]
            for label, values in routing.items()
        }

    rate_limits = []
    for fractions in method_fractions.values():
        for server_id, fraction in enumerate(fractions):
            if fraction <= 0.0:
                continue
            rate_limits.append(
                BASELINE_TARGET_UTILIZATION
                * frequencies[server_id]
                / (mean_cycles * float(fraction))
            )
    if not rate_limits:
        raise RuntimeError("Could not calibrate a positive NHPP rate")
    lambda0 = float(min(rate_limits))
    method_server_utilization = {
        label: {
            str(server_id): float(
                lambda0
                * mean_cycles
                * float(fraction)
                / frequencies[server_id]
            )
            for server_id, fraction in enumerate(fractions)
        }
        for label, fractions in method_fractions.items()
    }
    method_bottlenecks = {
        label: max(values.values(), default=0.0)
        for label, values in method_server_utilization.items()
    }
    worst_bottleneck = max(method_bottlenecks.values())
    if not 0.4 <= worst_bottleneck <= 0.5:
        raise RuntimeError(
            "Calibrated bottleneck utilization is outside [0.4, 0.5]"
        )
    return {
        "lambda0": lambda0,
        "method_routing_fractions": {
            label: {
                str(server_id): float(fraction)
                for server_id, fraction in enumerate(fractions)
            }
            for label, fractions in method_fractions.items()
        },
        "routing_values_are_upper_bounds": bool(
            routing_upper_bounds is not None
        ),
        "method_server_utilization": method_server_utilization,
        "method_bottleneck_utilization": method_bottlenecks,
        "worst_bottleneck_utilization": worst_bottleneck,
        "calibration_source": calibration_source,
        "calibration_histogram_fields": calibration_fields,
    }


def build_stream_bank(
    run_dir,
    seed,
    calibration_run_dirs=None,
    calibration_routing_profiles=None,
    calibration_routing_upper_bounds=None,
    calibration_metadata=None,
):
    config = read_json(run_dir / "config.json")
    input_config = config["input_config"]
    initial = read_json(run_dir / "scenario_initial.json")
    frequencies = np.asarray(
        [
            float(initial["servers"][str(server_id)]["frequency"])
            for server_id in sorted(
                int(value) for value in initial["servers"]
            )
        ],
        dtype=float,
    )
    total_frequency = float(sum(
        float(server["frequency"])
        for server in initial["servers"].values()
    ))
    mean_cycles = mean_a0_dag_cycles(input_config)
    calibration = bottleneck_rate_calibration(
        calibration_run_dirs,
        frequencies,
        mean_cycles,
        routing_profiles=calibration_routing_profiles,
        routing_upper_bounds=calibration_routing_upper_bounds,
    )
    if calibration_metadata is not None:
        calibration["online_pilots"] = calibration_metadata
        calibration["routing_envelope"] = {
            "pilot_replicates": CALIBRATION_REPLICATES,
            "per_server_aggregation": "maximum",
            "uniform_reference_floor": 1.0 / len(frequencies),
            "renormalized_after_aggregation": False,
        }
    lambda0 = calibration["lambda0"]
    window_duration = (
        BASELINE_EXPECTED_ARRIVALS_PER_WINDOW / lambda0
    )
    target_ingress, hotspot_sources = closest_hotspot_sources(
        initial,
        seed,
    )
    rng = np.random.default_rng(seed + 9_000_019)
    events = []
    counts = []
    event_id = 0
    for window in range(1, DYNAMIC_WINDOWS + 1):
        burst = BASELINE_END_WINDOW < window <= BURST_END_WINDOW
        multiplier = BURST_RATE_MULTIPLIER if burst else 1.0
        count = int(
            rng.poisson(
                BASELINE_EXPECTED_ARRIVALS_PER_WINDOW * multiplier
            )
        )
        counts.append(count)
        offsets = np.sort(rng.uniform(0.0, window_duration, count))
        for offset in offsets:
            if burst and rng.random() < INGRESS_HOTSPOT_SHARE:
                source_id = int(rng.choice(hotspot_sources))
            else:
                source_id = int(rng.integers(0, 20))
            events.append(
                {
                    "event_id": event_id,
                    "window": window,
                    "arrival_time": float(
                        (window - 1) * window_duration + offset
                    ),
                    "source_user_id": source_id,
                }
            )
            event_id += 1
    bank = {
        "protocol_version": DYNAMIC_PROTOCOL_VERSION,
        "seed": seed,
        "stream_seed": seed + 3_000_017,
        "lambda0": lambda0,
        "window_duration": window_duration,
        "estimated_baseline_utilization": calibration[
            "worst_bottleneck_utilization"
        ],
        "estimated_global_offered_utilization": (
            lambda0 * mean_cycles / total_frequency
        ),
        "rate_calibration": calibration,
        "burst_lambda": lambda0 * BURST_RATE_MULTIPLIER,
        "burst_rate_multiplier": BURST_RATE_MULTIPLIER,
        "target_ingress_server": target_ingress,
        "hotspot_source_users": hotspot_sources,
        "hotspot_share": INGRESS_HOTSPOT_SHARE,
        "window_arrival_counts": counts,
        "events": events,
    }
    bank["stream_sha256"] = canonical_hash(bank)
    return bank


def validate_training_run(run_dir, seed, label):
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    if (
        summary.get("status") != "complete"
        or summary.get("eligible_for_comparison") is not True
        or summary.get("convergence", {}).get("reached") is not True
    ):
        raise RuntimeError(f"Unconverged checkpoint: {run_dir}")
    if summary.get("dag_dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Wrong A0 dataset in {run_dir}")
    if int(summary.get("total_server_capacity", -1)) != 8:
        raise RuntimeError(f"Dynamic evaluation requires budget 8: {run_dir}")
    arguments = config["arguments"]
    if int(arguments["seed"]) != seed or arguments["label"] != label:
        raise RuntimeError(f"Run identity mismatch: {run_dir}")
    return summary, config


def reconstruct_deployment(run_dir, config):
    arguments = config["arguments"]
    input_config = copy.deepcopy(config["input_config"])
    learning_config = copy.deepcopy(config["learning_config"])
    seed_everything(int(arguments["seed"]))
    input_config["save topology figure"] = False
    simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=input_config,
        learning_arguments=learning_config,
        filename_png=str(run_dir / "figures"),
    )
    expected = scenario_fingerprint(
        read_json(run_dir / "scenario_initial.json")
    )
    actual = scenario_fingerprint(scenario_snapshot(simulator))
    if actual != expected:
        raise RuntimeError(f"Could not reconstruct deployment: {run_dir}")
    return capture_deployment_state(simulator)


def apply_stream_deployment(simulator, deployment, events):
    for server_id, state in deployment["servers"].items():
        server = simulator.servers[int(server_id)]
        server.pos = tuple(state["position"])
        server.frequency = float(state["frequency"])
        server.load = float(state["load"])
        server.rate_to_cloud = float(state["rate_to_cloud"])
        server.capacity = int(state["capacity"])
        simulator.server_capacities[int(server_id)] = server.capacity
    simulator.total_server_capacity = sum(
        server.capacity for server in simulator.servers.values()
    )
    simulator.service_data_length = copy.deepcopy(
        deployment["service_data_length"]
    )
    simulator.between_server_costs = deployment[
        "between_server_costs"
    ].copy()
    for user_id, event in enumerate(events):
        source_id = int(event["source_user_id"])
        source = deployment["users"][source_id]
        user = simulator.users[user_id]
        user.source_user_id = source_id
        user.arrival_window = int(event["window"])
        user.pos0 = tuple(source["position"])
        user.pos = tuple(source["position"])
        simulator.user_directions[user_id] = source["direction"]
        user.nearest_server = user.find_nearest_server(simulator.servers)


def exogenous_fingerprint(simulator, bank, deployment):
    payload = {
        "protocol": DYNAMIC_PROTOCOL_VERSION,
        "dataset_sha256": simulator.dag_dataset_sha256,
        "stream_sha256": bank["stream_sha256"],
        "user_graph_keys": simulator.user_graph_keys,
        "events": bank["events"],
        "servers": {
            str(server_id): {
                key: (
                    list(value) if key == "position" else value
                )
                for key, value in state.items()
            }
            for server_id, state in deployment["servers"].items()
        },
        "service_data_length": {
            str(key): float(value)
            for key, value in deployment["service_data_length"].items()
        },
        "between_server_costs": deployment[
            "between_server_costs"
        ].tolist(),
    }
    return canonical_hash(payload)


def neural_weight_hash(simulator):
    digest = hashlib.sha256()
    for server_id, server in sorted(simulator.servers.items()):
        digest.update(str(server_id).encode("ascii"))
        for name, tensor in sorted(
            server.agent.agent.TrainNet.model.state_dict().items()
        ):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def calibration_events(seed, replicate=0):
    if not 0 <= replicate < CALIBRATION_REPLICATES:
        raise ValueError("Invalid calibration replicate")
    rng = np.random.default_rng(
        seed + 6_000_043 + replicate * CALIBRATION_SEED_STRIDE
    )
    events = []
    event_id = 0
    for window in range(1, CALIBRATION_WINDOWS + 1):
        for index in range(CALIBRATION_ARRIVALS_PER_WINDOW):
            events.append(
                {
                    "event_id": event_id,
                    "window": window,
                    "arrival_time": float(
                        (window - 1) * CALIBRATION_WINDOW_DURATION_SEC
                        + index * CALIBRATION_ARRIVAL_SPACING_SEC
                    ),
                    "source_user_id": int(rng.integers(0, 20)),
                }
            )
            event_id += 1
    return events


def routing_upper_bounds_from_probes(probes_by_method, number_of_servers):
    """Envelope pilot modes plus a method-independent uniform reference."""
    upper_bounds = {}
    for label, probes in probes_by_method.items():
        if len(probes) != CALIBRATION_REPLICATES:
            raise ValueError(
                f"Expected {CALIBRATION_REPLICATES} pilots for {label}"
            )
        profiles = []
        for probe in probes:
            profile = np.asarray(
                [
                    probe["routing_fractions"][str(server_id)]
                    for server_id in range(number_of_servers)
                ],
                dtype=float,
            )
            if (
                profile.shape != (number_of_servers,)
                or np.any(profile < 0.0)
                or not np.isclose(profile.sum(), 1.0)
            ):
                raise ValueError(f"Invalid calibration probe for {label}")
            profiles.append(profile)
        upper_bounds[label] = np.maximum(
            np.max(np.asarray(profiles, dtype=float), axis=0),
            1.0 / number_of_servers,
        ).tolist()
    return upper_bounds


def calibration_cache_observer(simulator):
    for window in range(1, CALIBRATION_WINDOWS + 1):
        boundary = window * CALIBRATION_WINDOW_DURATION_SEC
        if boundary > simulator.env.now:
            yield simulator.env.timeout(boundary - simulator.env.now)
        simulator.broker.advance_cache_window()


def online_cache_routing_probe(suite_dir, label, seed, replicate):
    """Measure causal routing after native cache adaptation on a disjoint stream."""
    torch.set_num_threads(1)
    source_run = checkpoint_run(suite_dir, label, seed)
    training_summary, config = validate_training_run(
        source_run,
        seed,
        label,
    )
    deployment = reconstruct_deployment(source_run, config)
    checkpoint = torch.load(
        source_run / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    frozen_state = checkpoint["frozen_state"]
    events = calibration_events(seed, replicate)
    pilot_seed = (
        seed + 6_500_047 + replicate * CALIBRATION_SEED_STRIDE
    )
    input_config = copy.deepcopy(config["input_config"])
    input_config.update(
        {
            "seed": pilot_seed,
            "Number of users": len(events),
            "application arrival times": [
                event["arrival_time"] for event in events
            ],
            "dynamic queueing enabled": True,
            "periodic cache updates": True,
            "save topology figure": False,
        }
    )
    seed_everything(pilot_seed)
    simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=input_config,
        learning_arguments=copy.deepcopy(config["learning_config"]),
        filename_png=str(source_run / "figures"),
    )
    apply_stream_deployment(simulator, deployment, events)
    apply_frozen_state(simulator, frozen_state)
    for user_id, event in enumerate(events):
        source_id = int(event["source_user_id"])
        simulator.users[user_id].deadline = frozen_state[
            "deadlines"
        ][source_id]
    simulator.set_training(False, update_caching=True)
    simulator.reset()
    weights_before = neural_weight_hash(simulator)
    simulator.start_processes()
    simulator.env.process(calibration_cache_observer(simulator))
    simulator.env.run()
    weights_after = neural_weight_hash(simulator)
    if weights_before != weights_after:
        raise RuntimeError("Calibration pilot changed frozen weights")

    responses = [
        user.finish_time_of_application - user.arrival_time
        for user in simulator.users.values()
    ]
    if max(responses, default=0.0) >= CALIBRATION_ARRIVAL_SPACING_SEC:
        raise RuntimeError("Calibration pilot was not queue-free")
    cycles = {
        server_id: float(
            sum(
                task.cpu_cycle
                for user in simulator.users.values()
                for task in user.done_tasks.values()
                if task.assigned_server == server_id
            )
        )
        for server_id in simulator.servers
    }
    total_cycles = float(sum(cycles.values()))
    if total_cycles <= 0.0:
        raise RuntimeError("Calibration pilot observed no CPU work")
    routing_fractions = {
        str(server_id): value / total_cycles
        for server_id, value in cycles.items()
    }
    exogenous_sha256 = canonical_hash(
        {
            "protocol": DYNAMIC_PROTOCOL_VERSION,
            "role": "independent_online_cache_calibration",
            "seed": pilot_seed,
            "events": events,
            "user_graph_keys": simulator.user_graph_keys,
            "scenario": scenario_fingerprint(
                read_json(source_run / "scenario_initial.json")
            ),
        }
    )
    return {
        "status": "complete",
        "protocol_version": DYNAMIC_PROTOCOL_VERSION,
        "label": label,
        "seed": seed,
        "replicate": replicate,
        "pilot_seed": pilot_seed,
        "windows": CALIBRATION_WINDOWS,
        "arrivals": len(events),
        "arrival_spacing_sec": CALIBRATION_ARRIVAL_SPACING_SEC,
        "future_test_requests_visible": False,
        "network_frozen": True,
        "native_online_cache_updates": True,
        "maximum_response_time_sec": max(responses, default=0.0),
        "routing_fractions": routing_fractions,
        "total_cpu_cycles": total_cycles,
        "exogenous_sha256": exogenous_sha256,
        "source_selected_checkpoint_sha256": training_summary[
            "selected_checkpoint_sha256"
        ],
    }


def cache_matrix(simulator):
    return {
        str(server_id): [
            int(service)
            for service in server.services
            if service > 0
        ]
        for server_id, server in simulator.servers.items()
    }


def periodic_cache_observer(simulator, bank, records):
    previous = cache_snapshot(simulator)
    duration = float(bank["window_duration"])
    for window in range(1, DYNAMIC_WINDOWS + 1):
        boundary = window * duration
        if boundary > simulator.env.now:
            yield simulator.env.timeout(boundary - simulator.env.now)
        before_update = cache_snapshot(simulator)
        updated = simulator.broker.advance_cache_window()
        after_update = cache_snapshot(simulator)
        coverage = len(
            {
                service
                for services in after_update["cache_matrix"].values()
                for service in services
            }
        ) / simulator.Q
        records.append(
            {
                "window": window,
                "cache_updated": int(updated),
                "completed_cache_hits": (
                    before_update["hits"] - previous["hits"]
                ),
                "completed_cache_misses": (
                    before_update["misses"] - previous["misses"]
                ),
                "cache_service_coverage": coverage,
                "cache_migration_events": (
                    after_update["migration_events"]
                    - previous["migration_events"]
                ),
                "cache_migration_time_sec": (
                    after_update["migration_time_sec"]
                    - previous["migration_time_sec"]
                ),
                "cache_migration_critical_time_sec": (
                    after_update["migration_critical_time_sec"]
                    - previous["migration_critical_time_sec"]
                ),
                "cache_decision_calls": (
                    after_update["decision_calls"]
                    - previous["decision_calls"]
                ),
                "cache_decision_wall_time_sec": (
                    after_update["decision_wall_time_sec"]
                    - previous["decision_wall_time_sec"]
                ),
                "policy_inference_calls": (
                    before_update["policy_inference_calls"]
                    - previous["policy_inference_calls"]
                ),
                "policy_inference_wall_time_sec": (
                    before_update["policy_inference_wall_time_sec"]
                    - previous["policy_inference_wall_time_sec"]
                ),
                "cache_matrix": cache_matrix(simulator),
            }
        )
        previous = after_update


def task_dependency_transfer(user, simulator):
    total = 0.0
    for task in user.done_tasks.values():
        for predecessor_id in task.predecessors:
            predecessor = user.done_tasks[predecessor_id]
            total += (
                simulator.between_server_costs[
                    predecessor.assigned_server,
                    task.assigned_server,
                ]
                * predecessor.outputs_length[task.task_number]
            )
    return total


def control_payload_bytes(simulator, decision_calls):
    if simulator.cache_policy not in {
        "popularity_coordinated",
        "critical_path_coordinated",
        "critical_path_joint",
    }:
        return 0
    values = simulator.S * simulator.Q + 2 * simulator.S
    return int(decision_calls * values * 8)


def recovery_delay(window_rows):
    baseline = float(
        np.mean(
            [
                row["migration_adjusted_mean_finish_time"]
                for row in window_rows
                if 21 <= row["window"] <= BASELINE_END_WINDOW
            ]
        )
    )
    threshold = RECOVERY_TOLERANCE * baseline
    recovery = [
        row["migration_adjusted_mean_finish_time"]
        for row in window_rows
        if row["window"] > BURST_END_WINDOW
    ]
    delay = None
    for index in range(
        len(recovery) - RECOVERY_ROLLING_WINDOWS + 1
    ):
        if (
            float(
                np.mean(
                    recovery[index:index + RECOVERY_ROLLING_WINDOWS]
                )
            )
            <= threshold
        ):
            delay = index
            break
    return {
        "baseline_mean": baseline,
        "threshold": threshold,
        "delay_windows": delay,
        "effective_delay_windows": (
            delay if delay is not None else len(recovery) + 1
        ),
    }


def evaluate_run(suite_dir, output_dir, label, seed, bank, resume, quiet):
    torch.set_num_threads(1)
    source_run = checkpoint_run(suite_dir, label, seed)
    run_output = output_dir / "runs" / label / f"seed_{seed}"
    run_output.mkdir(parents=True, exist_ok=True)
    summary_path = run_output / "summary.json"
    if resume and summary_path.exists():
        summary = read_json(summary_path)
        if (
            summary.get("status") == "complete"
            and summary.get("stream_sha256") == bank["stream_sha256"]
        ):
            return summary
    training_summary, config = validate_training_run(
        source_run,
        seed,
        label,
    )
    deployment = reconstruct_deployment(source_run, config)
    checkpoint = torch.load(
        source_run / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    frozen_state = checkpoint["frozen_state"]
    input_config = copy.deepcopy(config["input_config"])
    learning_config = copy.deepcopy(config["learning_config"])
    events = bank["events"]
    input_config.update(
        {
            "seed": int(bank["stream_seed"]),
            "Number of users": len(events),
            "application arrival times": [
                event["arrival_time"] for event in events
            ],
            "dynamic queueing enabled": True,
            "periodic cache updates": True,
            "save topology figure": False,
        }
    )
    seed_everything(int(bank["stream_seed"]))
    started = time.perf_counter()
    with (run_output / "simulator.log").open(
        "w",
        encoding="utf-8",
    ) as simulator_log:
        simulator = MEC_Simulator(
            outputfile=simulator_log,
            Input_dict=input_config,
            learning_arguments=learning_config,
            filename_png=str(run_output),
        )
        apply_stream_deployment(simulator, deployment, events)
        apply_frozen_state(simulator, frozen_state)
        for user_id, event in enumerate(events):
            source_id = int(event["source_user_id"])
            if source_id in frozen_state["deadlines"]:
                simulator.users[user_id].deadline = frozen_state[
                    "deadlines"
                ][source_id]
        simulator.set_training(False, update_caching=True)
        simulator.reset()
        exogenous_hash = exogenous_fingerprint(
            simulator,
            bank,
            deployment,
        )
        oracle_started = time.perf_counter()
        oracle = scenario_capacity_aware_oracle_bounds(simulator)
        oracle_wall_time = time.perf_counter() - oracle_started
        if not oracle["capacity_constraints_satisfied"]:
            raise RuntimeError("Dynamic Oracle violated cache capacity")
        weights_before = neural_weight_hash(simulator)
        observer_rows = []
        simulator.start_processes()
        simulator.env.process(
            periodic_cache_observer(simulator, bank, observer_rows)
        )
        simulator.env.run()
        weights_after = neural_weight_hash(simulator)
        if weights_before != weights_after:
            raise RuntimeError("Frozen neural network changed online")

    oracle_by_user = np.asarray(oracle["per_user"], dtype=float)
    observer_by_window = {
        int(row["window"]): row for row in observer_rows
    }
    window_rows = []
    pending_migration = 0.0
    for window in range(1, DYNAMIC_WINDOWS + 1):
        user_ids = [
            index
            for index, event in enumerate(events)
            if int(event["window"]) == window
        ]
        if not user_ids:
            raise RuntimeError(
                f"NHPP produced an empty window {window}; change seed bank"
            )
        responses = np.asarray(
            [
                simulator.users[user_id].finish_time_of_application
                - simulator.users[user_id].arrival_time
                for user_id in user_ids
            ],
            dtype=float,
        )
        waiting = np.asarray(
            [
                sum(
                    task.result.waiting_latency
                    for task in simulator.users[user_id].done_tasks.values()
                )
                for user_id in user_ids
            ],
            dtype=float,
        )
        tasks = [
            task
            for user_id in user_ids
            for task in simulator.users[user_id].done_tasks.values()
        ]
        server_cpu_cycles = {
            server_id: float(
                sum(
                    task.cpu_cycle
                    for task in tasks
                    if task.assigned_server == server_id
                )
            )
            for server_id in simulator.servers
        }
        server_offered_utilization = {
            server_id: float(
                server_cpu_cycles[server_id]
                / (
                    simulator.servers[server_id].frequency
                    * bank["window_duration"]
                )
            )
            for server_id in simulator.servers
        }
        hits = sum(task.cache_hit is True for task in tasks)
        misses = sum(task.cache_hit is False for task in tasks)
        service_loading = sum(
            float(task.result.service_latency) for task in tasks
        )
        dependency_transfer = sum(
            task_dependency_transfer(simulator.users[user_id], simulator)
            for user_id in user_ids
        )
        observation = observer_by_window[window]
        oracle_values = oracle_by_user[user_ids]
        adjusted = responses + pending_migration
        regret = np.maximum(adjusted - oracle_values, 0.0)
        row = {
            "label": label,
            "seed": seed,
            "window": window,
            "segment": (
                "baseline"
                if window <= BASELINE_END_WINDOW
                else (
                    "burst"
                    if window <= BURST_END_WINDOW
                    else "recovery"
                )
            ),
            "lambda_multiplier": (
                BURST_RATE_MULTIPLIER
                if BASELINE_END_WINDOW < window <= BURST_END_WINDOW
                else 1.0
            ),
            "arrivals": len(user_ids),
            "mean_finish_time": float(responses.mean()),
            "p95_finish_time": float(np.percentile(responses, 95)),
            "migration_delay_applied_sec": pending_migration,
            "migration_adjusted_mean_finish_time": float(adjusted.mean()),
            "migration_adjusted_p95_finish_time": float(
                np.percentile(adjusted, 95)
            ),
            "mean_waiting_latency": float(waiting.mean()),
            "cache_hit_rate": hits / max(hits + misses, 1),
            "cache_remote_loading_rate": misses / max(hits + misses, 1),
            "service_loading_time_sec": service_loading,
            "dependency_transfer_time_sec": dependency_transfer,
            "max_server_offered_utilization": max(
                server_offered_utilization.values(),
                default=0.0,
            ),
            "server_cpu_cycles_json": json.dumps(
                server_cpu_cycles,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "server_offered_utilization_json": json.dumps(
                server_offered_utilization,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "oracle_mean": float(oracle_values.mean()),
            "oracle_p95": float(np.percentile(oracle_values, 95)),
            "oracle_regret": float(regret.sum()),
            **{
                key: value
                for key, value in observation.items()
                if key not in {"window", "cache_matrix"}
            },
            "control_payload_bytes": control_payload_bytes(
                simulator,
                observation["cache_decision_calls"],
            ),
            "cache_matrix_json": json.dumps(
                observation["cache_matrix"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        window_rows.append(row)
        pending_migration = float(
            observation["cache_migration_critical_time_sec"]
        )

    recovery = recovery_delay(window_rows)
    post_rows = [
        row for row in window_rows if row["window"] > BASELINE_END_WINDOW
    ]
    all_tasks = [
        task
        for user in simulator.users.values()
        for task in user.done_tasks.values()
    ]
    inference_calls = sum(
        server.policy_inference_calls
        for server in simulator.servers.values()
    )
    inference_time = sum(
        server.policy_inference_wall_time_sec
        for server in simulator.servers.values()
    )
    baseline_server_cpu_cycles = {
        server_id: float(
            sum(
                task.cpu_cycle
                for user_id, event in enumerate(events)
                if int(event["window"]) <= BASELINE_END_WINDOW
                for task in simulator.users[user_id].done_tasks.values()
                if task.assigned_server == server_id
            )
        )
        for server_id in simulator.servers
    }
    empirical_baseline_server_utilization = {
        str(server_id): float(
            cycles
            / (
                BASELINE_END_WINDOW
                * bank["window_duration"]
                * simulator.servers[server_id].frequency
            )
        )
        for server_id, cycles in baseline_server_cpu_cycles.items()
    }
    summary = {
        "status": "complete",
        "protocol_version": DYNAMIC_PROTOCOL_VERSION,
        "label": label,
        "display_name": DISPLAY_NAMES[label],
        "seed": seed,
        "source_run": str(source_run.resolve()),
        "source_selected_checkpoint_sha256": training_summary[
            "selected_checkpoint_sha256"
        ],
        "stream_sha256": bank["stream_sha256"],
        "exogenous_scenario_sha256": exogenous_hash,
        "dataset_sha256": simulator.dag_dataset_sha256,
        "capacity_vector": {
            str(server_id): int(server.capacity)
            for server_id, server in simulator.servers.items()
        },
        "total_cache_budget": simulator.total_server_capacity,
        "network_frozen": True,
        "network_weight_sha256_before": weights_before,
        "network_weight_sha256_after": weights_after,
        "native_online_cache_updates": True,
        "arrivals": len(events),
        "lambda0": bank["lambda0"],
        "estimated_baseline_utilization": bank[
            "estimated_baseline_utilization"
        ],
        "estimated_global_offered_utilization": bank[
            "estimated_global_offered_utilization"
        ],
        "rate_calibration": bank["rate_calibration"],
        "empirical_baseline_server_utilization": (
            empirical_baseline_server_utilization
        ),
        "empirical_baseline_bottleneck_utilization": max(
            empirical_baseline_server_utilization.values(),
            default=0.0,
        ),
        "overall_mean_finish_time": float(
            np.average(
                [row["migration_adjusted_mean_finish_time"] for row in window_rows],
                weights=[row["arrivals"] for row in window_rows],
            )
        ),
        "post_change_mean_finish_time": float(
            np.average(
                [row["migration_adjusted_mean_finish_time"] for row in post_rows],
                weights=[row["arrivals"] for row in post_rows],
            )
        ),
        "post_change_p95_window_mean": float(
            np.mean(
                [row["migration_adjusted_p95_finish_time"] for row in post_rows]
            )
        ),
        "post_change_mean_waiting_latency": float(
            np.average(
                [row["mean_waiting_latency"] for row in post_rows],
                weights=[row["arrivals"] for row in post_rows],
            )
        ),
        "post_change_cumulative_oracle_regret": float(
            sum(row["oracle_regret"] for row in post_rows)
        ),
        "recovery": recovery,
        "cache_hit_rate": float(
            np.mean([task.cache_hit is True for task in all_tasks])
        ),
        "cache_remote_loading_rate": float(
            np.mean([task.cache_hit is False for task in all_tasks])
        ),
        "cache_migration_events": int(
            sum(row["cache_migration_events"] for row in window_rows)
        ),
        "cache_migration_time_sec": float(
            sum(row["cache_migration_time_sec"] for row in window_rows)
        ),
        "policy_inference_calls": inference_calls,
        "policy_inference_time_per_task_ms": (
            1000.0 * inference_time / max(inference_calls, 1)
        ),
        "cache_decision_wall_time_sec": float(
            sum(row["cache_decision_wall_time_sec"] for row in window_rows)
        ),
        "control_payload_bytes": int(
            sum(row["control_payload_bytes"] for row in window_rows)
        ),
        "oracle_wall_time_sec": oracle_wall_time,
        "oracle_future_workload_visible": True,
        "oracle_queue_coupling_relaxed": True,
        "wall_time_sec": time.perf_counter() - started,
    }
    with (run_output / "windows.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)
    write_json(run_output / "stream_bank.json", bank)
    write_json(summary_path, summary)
    if not quiet:
        print(
            f"dynamic {label} seed={seed}: "
            f"post={summary['post_change_mean_finish_time']:.6f}",
            flush=True,
        )
    return summary


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return mean - half, mean + half


def paired_superiority(reference, ours, formal):
    reference = np.asarray(reference, dtype=float)
    ours = np.asarray(ours, dtype=float)
    improvements = reference - ours
    lower, upper = confidence_interval(improvements)
    nonzero = improvements[np.abs(improvements) > 1e-12]
    p_value = (
        float(stats.wilcoxon(nonzero, alternative="greater").pvalue)
        if len(nonzero)
        else 1.0
    )
    return {
        "pairs": len(improvements),
        "reference_mean": float(reference.mean()),
        "our_mean": float(ours.mean()),
        "mean_improvement_sec": float(improvements.mean()),
        "mean_improvement_percent": float(
            100.0 * improvements.mean() / reference.mean()
        ),
        "ci95_lower_sec": lower,
        "ci95_upper_sec": upper,
        "wins": int(np.sum(improvements > 0)),
        "wilcoxon_one_sided_p": p_value,
        "passed": bool(
            (
                lower > 0
                and p_value < 0.05
                and np.sum(improvements > 0) >= 7
            )
            if formal
            else (
                improvements.mean() > 0
                and np.sum(improvements > 0) >= 2
            )
        ),
    }


def read_window_rows(output_dir, label, seed):
    path = output_dir / "runs" / label / f"seed_{seed}" / "windows.csv"
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        for key, value in list(row.items()):
            if key in {
                "label",
                "segment",
                "cache_matrix_json",
                "server_cpu_cycles_json",
                "server_offered_utilization_json",
            }:
                continue
            row[key] = float(value)
        row["window"] = int(row["window"])
    return rows


def aggregate_results(output_dir, labels, seeds, summaries, banks, mode):
    by_method = {
        label: [
            summaries[(label, seed)] for seed in seeds
        ]
        for label in labels
    }
    paired = {}
    for reference_label in (
        "guided_full",
        "centralized_greedy_daoc",
    ):
        paired[f"our_vs_{reference_label}"] = paired_superiority(
            [
                row["post_change_mean_finish_time"]
                for row in by_method[reference_label]
            ],
            [
                row["post_change_mean_finish_time"]
                for row in by_method["lean_our"]
            ],
            formal=(mode == "final"),
        )
    integrity = {
        "all_streams_paired": all(
            len(
                {
                    summaries[(label, seed)]["stream_sha256"]
                    for label in labels
                }
            )
            == 1
            for seed in seeds
        ),
        "all_exogenous_scenarios_paired": all(
            len(
                {
                    summaries[(label, seed)][
                        "exogenous_scenario_sha256"
                    ]
                    for label in labels
                }
            )
            == 1
            for seed in seeds
        ),
        "all_networks_frozen": all(
            row["network_weight_sha256_before"]
            == row["network_weight_sha256_after"]
            for row in summaries.values()
        ),
        "baseline_utilization_range": all(
            0.4 <= bank["estimated_baseline_utilization"] <= 0.5
            for bank in banks.values()
        ),
        "robust_online_cpu_work_calibration": all(
            bank["rate_calibration"]["calibration_source"]
            == (
                "independent_multi_stream_online_cache_"
                "cpu_work_upper_bound"
            )
            and bank["rate_calibration"][
                "routing_values_are_upper_bounds"
            ]
            and bank["rate_calibration"]["routing_envelope"]
            == {
                "pilot_replicates": CALIBRATION_REPLICATES,
                "per_server_aggregation": "maximum",
                "uniform_reference_floor": 1.0 / 10,
                "renormalized_after_aggregation": False,
            }
            and set(
                bank["rate_calibration"][
                    "calibration_histogram_fields"
                ].values()
            )
            == {"online_cache_pilot_cpu_work_upper_bound"}
            for bank in banks.values()
        ),
        "calibration_pilots_paired": all(
            all(
                len(
                    {
                        bank["rate_calibration"]["online_pilots"][
                            label
                        ][replicate]["exogenous_sha256"]
                        for label in labels
                    }
                )
                == 1
                for replicate in range(CALIBRATION_REPLICATES)
            )
            for bank in banks.values()
        ),
        "calibration_replicates_independent": all(
            all(
                len(
                    {
                        probe["pilot_seed"]
                        for probe in bank["rate_calibration"][
                            "online_pilots"
                        ][label]
                    }
                )
                == CALIBRATION_REPLICATES
                for label in labels
            )
            for bank in banks.values()
        ),
        "calibration_streams_disjoint_and_causal": all(
            all(
                probe["protocol_version"] == DYNAMIC_PROTOCOL_VERSION
                and probe["future_test_requests_visible"] is False
                and probe["pilot_seed"] != bank["stream_seed"]
                for probes in bank["rate_calibration"][
                    "online_pilots"
                ].values()
                for probe in probes
            )
            for bank in banks.values()
        ),
        "calibration_networks_frozen": all(
            all(
                probe["network_frozen"] is True
                and probe["native_online_cache_updates"] is True
                for probes in bank["rate_calibration"][
                    "online_pilots"
                ].values()
                for probe in probes
            )
            for bank in banks.values()
        ),
        "calibration_queue_free": all(
            all(
                probe["maximum_response_time_sec"]
                < probe["arrival_spacing_sec"]
                for probes in bank["rate_calibration"][
                    "online_pilots"
                ].values()
                for probe in probes
            )
            for bank in banks.values()
        ),
        # The pilot targets 0.45. The independent formal stream may differ
        # slightly through Poisson and DAG sampling, but must remain clearly
        # below saturation and near the stated 40--50% operating region.
        "empirical_baseline_utilization_range": all(
            0.35
            <= max(
                summaries[(label, seed)][
                    "empirical_baseline_bottleneck_utilization"
                ]
                for label in labels
            )
            <= 0.60
            for seed in seeds
        ),
        "no_method_saturated_in_baseline": all(
            summary["empirical_baseline_bottleneck_utilization"]
            <= 0.65
            for summary in summaries.values()
        ),
    }
    aggregate = {
        label: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in (
                "overall_mean_finish_time",
                "post_change_mean_finish_time",
                "post_change_p95_window_mean",
                "post_change_mean_waiting_latency",
                "post_change_cumulative_oracle_regret",
                "cache_hit_rate",
                "cache_remote_loading_rate",
                "cache_migration_events",
                "cache_migration_time_sec",
                "policy_inference_time_per_task_ms",
                "cache_decision_wall_time_sec",
                "control_payload_bytes",
                "empirical_baseline_bottleneck_utilization",
            )
        }
        for label, rows in by_method.items()
    }
    for label, rows in by_method.items():
        aggregate[label]["cache_service_coverage"] = float(
            np.mean(
                [
                    row["cache_service_coverage"]
                    for seed in seeds
                    for row in read_window_rows(output_dir, label, seed)
                ]
            )
        )
        aggregate[label]["mean_recovery_delay_windows"] = float(
            np.mean(
                [
                    row["recovery"]["effective_delay_windows"]
                    for row in rows
                ]
            )
        )
    gate = {
        "our_beats_daoc": paired["our_vs_guided_full"]["passed"],
        "our_beats_centralized_greedy": paired[
            "our_vs_centralized_greedy_daoc"
        ]["passed"],
    }
    gate["passed"] = bool(
        all(integrity.values())
        and gate["our_beats_daoc"]
        and gate["our_beats_centralized_greedy"]
    )
    result = {
        "status": "complete",
        "protocol_version": DYNAMIC_PROTOCOL_VERSION,
        "mode": mode,
        "seeds": seeds,
        "integrity": integrity,
        "method_aggregates": aggregate,
        "paired_superiority": paired,
        "gate": gate,
        "claim_scope": "A0_controlled_mechanism_only",
    }
    write_json(output_dir / "dynamic_summary.json", result)
    plot_trajectory(output_dir, labels, seeds)
    write_report(output_dir / "DYNAMIC_REPORT_ZH.md", result)
    return result


def plot_trajectory(output_dir, labels, seeds):
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    all_axis, detail_axis = axes
    for label in labels:
        values = np.asarray(
            [
                [
                    row["migration_adjusted_mean_finish_time"]
                    for row in read_window_rows(output_dir, label, seed)
                ]
                for seed in seeds
            ],
            dtype=float,
        )
        mean = values.mean(axis=0)
        all_axis.plot(
            np.arange(1, DYNAMIC_WINDOWS + 1),
            mean,
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
            linewidth=2,
        )
        if label != "guided_full":
            detail_axis.plot(
                np.arange(1, DYNAMIC_WINDOWS + 1),
                mean,
                label=DISPLAY_NAMES[label],
                color=COLORS[label],
                linewidth=2,
            )
    for axis in axes:
        axis.axvspan(41, 60, color="#E9C46A", alpha=0.2, label="Burst")
        axis.axvline(40.5, color="#555555", linewidth=0.8)
        axis.axvline(60.5, color="#555555", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.set_ylabel("Mean completion time (s)")
    all_axis.set_yscale("log")
    all_axis.set_title("All methods (log scale)")
    all_axis.legend(frameon=False, ncol=2)
    detail_axis.set_title("Centralized methods (linear detail)")
    detail_axis.set_xlabel("Window")
    detail_axis.legend(frameon=False, ncol=3)
    figure.savefig(output_dir / "dynamic_trajectory.png", dpi=220)
    figure.savefig(output_dir / "dynamic_trajectory.pdf")
    plt.close(figure)


def write_report(path, result):
    aggregate = result["method_aggregates"]
    comparisons = result["paired_superiority"]
    lines = [
        "# A0 NHPP动态实验报告",
        "",
        "- A0仅作为受控机制数据集，不是无偏Alibaba holdout。",
        "- 窗口1–40为基准NHPP，41–60为3倍有限突发并切换入口热点，61–100恢复。",
        "- 神经网络全程冻结；仅保留方法原生的缓存和历史更新。",
        "",
        "| 方法 | 变化后平均完成时间 | P95窗口均值 | 等待时延 | Oracle regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in METHOD_LABELS:
        row = aggregate[label]
        lines.append(
            f"| {DISPLAY_NAMES[label]} | "
            f"{row['post_change_mean_finish_time']:.6f} | "
            f"{row['post_change_p95_window_mean']:.6f} | "
            f"{row['post_change_mean_waiting_latency']:.6f} | "
            f"{row['post_change_cumulative_oracle_regret']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| 方法 | 缓存命中 | 服务覆盖 | 远程加载 | "
            "恢复窗口 | 迁移时延 | 推理ms/任务 | 协调时间 | 通信字节 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in METHOD_LABELS:
        row = aggregate[label]
        lines.append(
            f"| {DISPLAY_NAMES[label]} | "
            f"{row['cache_hit_rate']:.6f} | "
            f"{row['cache_service_coverage']:.6f} | "
            f"{row['cache_remote_loading_rate']:.6f} | "
            f"{row['mean_recovery_delay_windows']:.3f} | "
            f"{row['cache_migration_time_sec']:.6f} | "
            f"{row['policy_inference_time_per_task_ms']:.6f} | "
            f"{row['cache_decision_wall_time_sec']:.6f} | "
            f"{row['control_payload_bytes']:.0f} |"
        )
    gate_title = (
        "正式门槛"
        if result["mode"] == "final"
        else "三Seed开发门槛"
    )
    lines.extend(["", f"## {gate_title}", ""])
    for key, comparison in comparisons.items():
        lines.append(
            f"- `{key}`: 改善 {comparison['mean_improvement_percent']:.3f}%，"
            f"95% CI下界 {comparison['ci95_lower_sec']:.6f} s，"
            f"Wilcoxon p={comparison['wilcoxon_one_sided_p']:.6g}，"
            f"胜出 {comparison['wins']}/{comparison['pairs']}，"
            f"通过={comparison['passed']}。"
        )
    lines.extend(
        [
            "",
            f"- 动态总门槛通过：`{result['gate']['passed']}`。",
            "- Oracle可见完整工作负载并松弛队列耦合，只能作为诊断参考。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    torch.set_num_threads(1)
    suite_dir = args.suite_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    banks = {}
    bank_dir = output_dir / "stream_banks"
    bank_dir.mkdir(exist_ok=True)
    pilot_dir = output_dir / "calibration_pilots"
    pilot_dir.mkdir(exist_ok=True)
    probes = {}
    probe_futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for label in args.labels:
            for seed in args.seeds:
                for replicate in range(CALIBRATION_REPLICATES):
                    directory = pilot_dir / label
                    directory.mkdir(exist_ok=True)
                    probe_path = directory / (
                        f"seed_{seed}_replicate_{replicate}.json"
                    )
                    if args.resume and probe_path.exists():
                        candidate = read_json(probe_path)
                        source_summary = read_json(
                            checkpoint_run(suite_dir, label, seed)
                            / "summary.json"
                        )
                        if (
                            candidate.get("status") == "complete"
                            and candidate.get("protocol_version")
                            == DYNAMIC_PROTOCOL_VERSION
                            and candidate.get("label") == label
                            and int(candidate.get("seed", -1)) == seed
                            and int(candidate.get("replicate", -1))
                            == replicate
                            and candidate.get(
                                "source_selected_checkpoint_sha256"
                            )
                            == source_summary.get(
                                "selected_checkpoint_sha256"
                            )
                        ):
                            probes[(label, seed, replicate)] = candidate
                            continue
                    future = executor.submit(
                        online_cache_routing_probe,
                        suite_dir,
                        label,
                        seed,
                        replicate,
                    )
                    probe_futures[future] = (
                        label,
                        seed,
                        replicate,
                    )
        for future in as_completed(probe_futures):
            key = probe_futures[future]
            probes[key] = future.result()
            label, seed, replicate = key
            directory = pilot_dir / label
            write_json(
                directory
                / f"seed_{seed}_replicate_{replicate}.json",
                probes[key],
            )
    for seed in args.seeds:
        source = checkpoint_run(
            suite_dir,
            "guided_full",
            seed,
        )
        seed_probes = {
            label: [
                probes[(label, seed, replicate)]
                for replicate in range(CALIBRATION_REPLICATES)
            ]
            for label in args.labels
        }
        for replicate in range(CALIBRATION_REPLICATES):
            if len(
                {
                    seed_probes[label][replicate]["exogenous_sha256"]
                    for label in args.labels
                }
            ) != 1:
                raise RuntimeError(
                    "Calibration pilots are not paired for "
                    f"seed {seed}, replicate {replicate}"
                )
        routing_upper_bounds = routing_upper_bounds_from_probes(
            seed_probes,
            number_of_servers=10,
        )
        bank = build_stream_bank(
            source,
            seed,
            calibration_routing_upper_bounds=routing_upper_bounds,
            calibration_metadata=seed_probes,
        )
        banks[seed] = bank
        write_json(bank_dir / f"seed_{seed}.json", bank)
        write_json(pilot_dir / f"seed_{seed}.json", seed_probes)
        initial_hashes = {
            scenario_fingerprint(
                read_json(
                    checkpoint_run(suite_dir, label, seed)
                    / "scenario_initial.json"
                )
            )
            for label in args.labels
        }
        if len(initial_hashes) != 1:
            raise RuntimeError(
                f"Static deployments are not paired for seed {seed}"
            )

    summaries = {}
    futures = {}
    # Simulator construction mutates process-global RNG state. Separate
    # processes preserve deterministic paired reconstruction across workers.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for label in args.labels:
            for seed in args.seeds:
                future = executor.submit(
                    evaluate_run,
                    suite_dir,
                    output_dir,
                    label,
                    seed,
                    banks[seed],
                    args.resume,
                    args.quiet,
                )
                futures[future] = (label, seed)
        for future in as_completed(futures):
            key = futures[future]
            summaries[key] = future.result()
    result = aggregate_results(
        output_dir,
        args.labels,
        args.seeds,
        summaries,
        banks,
        args.mode,
    )
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
