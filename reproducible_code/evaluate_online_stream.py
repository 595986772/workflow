#!/usr/bin/env python3
"""Evaluate frozen offloading models with causal online cache adaptation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from information_protocol import INFORMATION_PROTOCOL_VERSION
from capacity_protocol import (
    CAPACITY_PROTOCOL_VERSION,
    select_load_shift_servers,
)
from oracle_latency_bound import (
    scenario_capacity_aware_oracle_bounds,
)
from run_independent_experiment import (
    EPISODE_FIELDS,
    apply_deployment_state,
    apply_frozen_state,
    cache_snapshot,
    base_scenario_fingerprint,
    capture_deployment_state,
    capture_frozen_state,
    collect_episode_metrics,
    scenario_fingerprint,
    scenario_snapshot,
    SCENARIO_FINGERPRINT_VERSION,
    seed_everything,
    summarize_rows,
    verify_frozen_state,
)
from simulator import MEC_Simulator
from user import generate_task_features


ONLINE_PROTOCOL_VERSION = "causal_online_stream_v2"


ONLINE_FIELDS = EPISODE_FIELDS + [
    "stream_regime",
    "stream_segment",
    "hotspot_services",
    "shifted_server_ids",
    "load_multiplier",
    "load_vector_json",
    "pre_window_execution_latency_ema_json",
    "pre_window_server_quality_json",
    "observed_server_execution_latency_json",
    "control_payload_bytes",
    "migration_delay_applied_sec",
    "migration_adjusted_finish_time",
    "migration_adjusted_p95_finish_time",
    "dynamic_oracle_floor",
    "dynamic_oracle_p95",
    "dynamic_oracle_gap_sec",
    "dynamic_oracle_regret",
    "dynamic_oracle_capacity_ok",
    "dynamic_oracle_cache_matrix_json",
]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected checkpoints on an independent workload "
            "stream with an abrupt service-hotspot shift."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--suite-dir", type=Path)
    target.add_argument("--run-dir", type=Path)
    parser.add_argument("--labels", type=parse_str_list)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--shift-episode", type=int, default=51)
    parser.add_argument("--hotspot-share", type=float, default=0.8)
    parser.add_argument(
        "--regime",
        choices=("service_hotspot_shift", "server_load_shift"),
        default="service_hotspot_shift",
    )
    parser.add_argument("--load-multiplier", type=float, default=4.0)
    parser.add_argument("--seed-offset", type=int, default=2_000_003)
    parser.add_argument("--recovery-window", type=int, default=5)
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--cache-mode",
        choices=("adaptive", "frozen"),
        default="adaptive",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.episodes < 2:
        raise ValueError("--episodes must be at least two")
    if not 2 <= args.shift_episode <= args.episodes:
        raise ValueError(
            "--shift-episode must be between 2 and --episodes"
        )
    if not 0 < args.hotspot_share <= 1:
        raise ValueError("--hotspot-share must be in (0, 1]")
    if args.load_multiplier < 1:
        raise ValueError("--load-multiplier must be at least one")
    if args.seed_offset < 1:
        raise ValueError("--seed-offset must be positive")
    if args.recovery_window < 1:
        raise ValueError("--recovery-window must be positive")
    if args.recovery_tolerance < 1:
        raise ValueError("--recovery-tolerance must be at least one")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def hotspot_sets(number_of_services):
    width = min(3, max(1, number_of_services // 2))
    before = list(range(1, width + 1))
    after = list(
        range(number_of_services - width + 1, number_of_services + 1)
    )
    return before, after


def deterministic_uniform(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def apply_service_hotspot(
    simulator,
    scenario_seed,
    segment,
    hotspot_share,
):
    before_services, after_services = hotspot_sets(simulator.Q)
    hotspot_services = (
        before_services if segment == "pre_shift" else after_services
    )
    changed_tasks = 0
    total_tasks = 0
    service_counts = {
        service_id: 0
        for service_id in range(1, simulator.Q + 1)
    }

    for user_id, user in simulator.users.items():
        for task_id, task in user.tasks_init.items():
            total_tasks += 1
            score = deterministic_uniform(
                scenario_seed,
                segment,
                user_id,
                task_id,
                "hotspot-gate",
            )
            if score < hotspot_share:
                selector = deterministic_uniform(
                    scenario_seed,
                    segment,
                    user_id,
                    task_id,
                    "hotspot-service",
                )
                index = min(
                    int(selector * len(hotspot_services)),
                    len(hotspot_services) - 1,
                )
                new_service = hotspot_services[index]
                changed_tasks += int(task.service != new_service)
                task.service = new_service
                if task_id in user.DAG2.nodes:
                    user.DAG2.nodes[task_id]["service"] = new_service
            service_counts[task.service] += 1
        user.node_features = generate_task_features(
            user.tasks_init,
            user.numberofservices,
        )

    return {
        "segment": segment,
        "hotspot_services": hotspot_services,
        "hotspot_share": hotspot_share,
        "total_tasks": total_tasks,
        "changed_tasks": changed_tasks,
        "service_counts": service_counts,
    }


def apply_server_load_shift(
    simulator,
    segment,
    shifted_servers,
    load_multiplier,
):
    """Apply the E3 load shock without changing the workload."""
    active_multiplier = (
        float(load_multiplier)
        if segment == "post_shift"
        else 1.0
    )
    shifted_ids = sorted(shifted_servers.values())
    if segment == "post_shift":
        for server_id in shifted_ids:
            simulator.servers[server_id].load *= active_multiplier
    return {
        "segment": segment,
        "hotspot_services": [],
        "shifted_servers_by_capacity": {
            str(capacity): int(server_id)
            for capacity, server_id
            in shifted_servers.items()
        },
        "shifted_server_ids": shifted_ids,
        "load_multiplier": active_multiplier,
        "load_vector": {
            str(server_id): float(server.load)
            for server_id, server in simulator.servers.items()
        },
    }


def exogenous_stream_snapshot(simulator, stream_descriptor):
    snapshot = scenario_snapshot(simulator)
    snapshot.pop("algorithm", None)
    for server in snapshot["servers"].values():
        server.pop("cached_services", None)
    snapshot["tasks"] = {
        str(user_id): {
            str(task_id): {
                "service": int(task.service),
                "cpu_cycle": float(task.cpu_cycle),
                "input_data_length": float(task.input_data_length),
                "predecessors": [
                    str(predecessor)
                    for predecessor in task.predecessors
                ],
                "successors": [
                    str(successor)
                    for successor in task.successors
                ],
                "output_lengths": {
                    str(successor): float(length)
                    for successor, length
                    in task.outputs_length.items()
                },
            }
            for task_id, task in user.tasks_init.items()
        }
        for user_id, user in simulator.users.items()
    }
    snapshot["stream"] = stream_descriptor
    return snapshot


def estimated_control_payload_bytes(simulator, decision_calls):
    if simulator.cache_policy != "critical_path_joint":
        return 0
    scalar_bytes = 4
    service_id_bytes = 2
    if hasattr(simulator, "servers"):
        total_capacity = sum(
            int(server.capacity)
            for server in simulator.servers.values()
        )
    else:
        total_capacity = (
            simulator.S
            * int(simulator.input_dict["server capacity"])
        )
    uplink_scalars_per_server = simulator.Q + 4
    bytes_per_call = (
        simulator.S * uplink_scalars_per_server * scalar_bytes
        + total_capacity * service_id_bytes
    )
    return int(decision_calls * bytes_per_call)


def observed_server_execution_latencies(simulator):
    """Return this window's realized compute-plus-wait latency by server."""
    totals = {server_id: 0.0 for server_id in simulator.servers}
    counts = {server_id: 0 for server_id in simulator.servers}
    for user in simulator.users.values():
        for task in user.done_tasks.values():
            server_id = int(task.assigned_server)
            totals[server_id] += float(
                task.result.computing_latency
                + task.result.waiting_latency
            )
            counts[server_id] += 1
    return {
        str(server_id): (
            totals[server_id] / counts[server_id]
            if counts[server_id]
            else None
        )
        for server_id in simulator.servers
    }


def recovery_metrics(
    rows,
    shift_episode,
    recovery_window,
    metric="average_finish_time",
    tolerance=1.05,
):
    pre_rows = [
        row for row in rows
        if int(row["episode"]) < shift_episode
    ]
    post_rows = [
        row for row in rows
        if int(row["episode"]) >= shift_episode
    ]
    late_count = min(20, len(post_rows))
    steady_state = float(
        np.mean(
            [
                row[metric]
                for row in post_rows[-late_count:]
            ]
        )
    )
    threshold = float(tolerance) * steady_state
    finish_times = np.asarray(
        [row[metric] for row in post_rows],
        dtype=float,
    )
    recovery_delay = None
    for start in range(0, len(post_rows) - recovery_window + 1):
        window = finish_times[start : start + recovery_window]
        if float(window.mean()) <= threshold:
            recovery_delay = start
            break
    transient_regret = float(
        np.maximum(finish_times - steady_state, 0.0).sum()
        / steady_state
    )
    pre_mean = float(
        np.mean([row[metric] for row in pre_rows])
    )
    oracle_regret = float(
        sum(
            float(row.get("dynamic_oracle_regret", 0.0))
            for row in post_rows
        )
    )
    oracle_floor_total = float(
        sum(
            float(row.get("dynamic_oracle_floor", 0.0))
            for row in post_rows
        )
    )
    return {
        "metric": metric,
        "steady_state_window": late_count,
        "steady_state_finish_time": steady_state,
        "recovery_threshold": threshold,
        "recovery_window": recovery_window,
        "recovery_tolerance": float(tolerance),
        "adaptation_delay_windows": recovery_delay,
        "normalized_transient_regret": transient_regret,
        "cumulative_oracle_regret": oracle_regret,
        "normalized_cumulative_oracle_regret": (
            oracle_regret / oracle_floor_total
            if oracle_floor_total > 0
            else 0.0
        ),
        "pre_shift_finish_time": pre_mean,
        "steady_state_change_percent": (
            100.0 * (steady_state - pre_mean) / pre_mean
        ),
    }


def validate_dynamic_accounting(rows):
    """Audit per-window Oracle capacity and migration-delay accounting."""
    if not rows:
        raise RuntimeError("Online evaluation produced no windows")
    if any(
        not int(row["dynamic_oracle_capacity_ok"])
        for row in rows
    ):
        raise RuntimeError(
            "Dynamic Oracle violated a per-server cache capacity"
        )
    for index, row in enumerate(rows):
        expected_delay = (
            0.0
            if index == 0
            else float(
                rows[index - 1][
                    "cache_migration_critical_time_sec"
                ]
            )
        )
        if not np.isclose(
            float(row["migration_delay_applied_sec"]),
            expected_delay,
        ):
            raise RuntimeError(
                "Migration delay was not carried to the next window"
            )
        if not np.isclose(
            float(row["migration_adjusted_finish_time"]),
            float(row["average_finish_time"]) + expected_delay,
        ):
            raise RuntimeError(
                "Migration-adjusted mean accounting is inconsistent"
            )
        if not np.isclose(
            float(row["migration_adjusted_p95_finish_time"]),
            float(row["p95_finish_time"]) + expected_delay,
        ):
            raise RuntimeError(
                "Migration-adjusted P95 accounting is inconsistent"
            )
    return True


def summarize_online_rows(rows):
    summary = summarize_rows(rows)
    if summary is None:
        return None
    for metric in (
        "migration_delay_applied_sec",
        "migration_adjusted_finish_time",
        "migration_adjusted_p95_finish_time",
        "dynamic_oracle_floor",
        "dynamic_oracle_p95",
        "dynamic_oracle_gap_sec",
        "dynamic_oracle_regret",
    ):
        values = np.asarray(
            [float(row[metric]) for row in rows],
            dtype=float,
        )
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = (
            float(values.std(ddof=1))
            if len(values) > 1
            else 0.0
        )
    return summary


def segment_summary(rows, shift_episode):
    pre = [
        row for row in rows
        if int(row["episode"]) < shift_episode
    ]
    post = [
        row for row in rows
        if int(row["episode"]) >= shift_episode
    ]
    early_count = min(10, len(post))
    late_count = min(10, len(post))
    return {
        "overall": summarize_online_rows(rows),
        "pre_shift": summarize_online_rows(pre),
        "post_shift": summarize_online_rows(post),
        "post_shift_early": summarize_online_rows(
            post[:early_count]
        ),
        "post_shift_late": summarize_online_rows(
            post[-late_count:]
        ),
    }


def expected_run_summary_matches(path, args):
    if not path.exists():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    protocol = summary.get("protocol", {})
    return (
        summary.get("status") == "complete"
        and protocol.get("version") == ONLINE_PROTOCOL_VERSION
        and protocol.get("episodes") == args.episodes
        and protocol.get("shift_episode") == args.shift_episode
        and protocol.get("regime") == args.regime
        and np.isclose(
            protocol.get("hotspot_share", -1.0),
            args.hotspot_share,
        )
        and protocol.get("seed_offset") == args.seed_offset
        and protocol.get("recovery_window") == args.recovery_window
        and np.isclose(
            protocol.get("recovery_tolerance", -1.0),
            args.recovery_tolerance,
        )
        and np.isclose(
            protocol.get("load_multiplier", -1.0),
            args.load_multiplier,
        )
        and protocol.get("cache_mode") == args.cache_mode
    )


def artifact_name(args, stem, suffix):
    if args.regime == "server_load_shift":
        multiplier_suffix = (
            ""
            if np.isclose(args.load_multiplier, 4.0)
            else f"_x{args.load_multiplier:g}"
        )
        regime_suffix = f"_e3_load_shift{multiplier_suffix}"
    else:
        regime_suffix = ""
    mode_suffix = "" if args.cache_mode == "adaptive" else "_frozen"
    return f"{stem}{regime_suffix}{mode_suffix}{suffix}"


def evaluate_run(run_dir, args):
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    selected_checkpoint_path = run_dir / "selected_checkpoint.pt"
    if not summary_path.exists() or not selected_checkpoint_path.exists():
        raise RuntimeError(
            f"Missing converged run artifacts in {run_dir}"
        )
    training_summary = read_json(summary_path)
    if (
        training_summary.get("status") != "complete"
        or training_summary.get("eligible_for_comparison") is not True
        or training_summary.get("information_protocol_version")
        != INFORMATION_PROTOCOL_VERSION
        or training_summary.get("capacity_protocol_version")
        != CAPACITY_PROTOCOL_VERSION
        or training_summary.get("scenario_fingerprint_version")
        != SCENARIO_FINGERPRINT_VERSION
    ):
        raise RuntimeError(
            f"Run is not an eligible current-protocol checkpoint: {run_dir}"
        )

    config = read_json(run_dir / "config.json")
    arguments = config["arguments"]
    input_config = copy.deepcopy(config["input_config"])
    learning_config = copy.deepcopy(config["learning_config"])
    seed = int(arguments["seed"])
    label = arguments["label"]
    checkpoint = torch.load(
        selected_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    dynamic_state = checkpoint["frozen_state"]
    if "cache_runtime" not in dynamic_state:
        raise RuntimeError(
            f"Checkpoint predates cache runtime persistence: {run_dir}"
        )

    output_summary_path = run_dir / artifact_name(
        args,
        "online_stream",
        "_summary.json",
    )
    if args.resume and expected_run_summary_matches(
        output_summary_path,
        args,
    ):
        return {
            "label": label,
            "seed": seed,
            "run_dir": str(run_dir),
            "skipped": True,
        }

    online_log_path = run_dir / artifact_name(
        args,
        "online_stream",
        "_simulator.log",
    )
    rows = []
    scenarios = []
    started_total = time.perf_counter()
    with online_log_path.open("w", encoding="utf-8") as simulator_log:
        seed_everything(seed)
        base_config = copy.deepcopy(input_config)
        base_config["seed"] = seed
        base_config["save topology figure"] = False
        base_simulator = MEC_Simulator(
            outputfile=simulator_log,
            Input_dict=base_config,
            learning_arguments=learning_config,
            filename_png=str(run_dir / "figures"),
        )
        recorded_initial = read_json(run_dir / "scenario_initial.json")
        if (
            scenario_fingerprint(scenario_snapshot(base_simulator))
            != scenario_fingerprint(recorded_initial)
        ):
            raise RuntimeError(
                f"Could not reconstruct the training deployment: {run_dir}"
            )
        deployment_state = capture_deployment_state(base_simulator)
        shifted_servers = select_load_shift_servers(
            {
                int(server_id): int(state["capacity"])
                for server_id, state
                in deployment_state["servers"].items()
            },
            seed=seed,
        )
        del base_simulator
        pending_migration_delay = 0.0

        output_csv_path = run_dir / artifact_name(
            args,
            "online_stream",
            ".csv",
        )
        with output_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=ONLINE_FIELDS,
            )
            writer.writeheader()

            for episode_index in range(args.episodes):
                episode = episode_index + 1
                stream_seed = seed + args.seed_offset + episode_index
                segment = (
                    "pre_shift"
                    if episode < args.shift_episode
                    else "post_shift"
                )
                seed_everything(stream_seed)
                stream_config = copy.deepcopy(input_config)
                stream_config["seed"] = stream_seed
                stream_config["save topology figure"] = False
                simulator = MEC_Simulator(
                    outputfile=simulator_log,
                    Input_dict=stream_config,
                    learning_arguments=learning_config,
                    filename_png=str(run_dir / "figures"),
                )
                apply_deployment_state(
                    simulator,
                    deployment_state,
                )
                if args.regime == "server_load_shift":
                    descriptor = apply_server_load_shift(
                        simulator=simulator,
                        segment=segment,
                        shifted_servers=shifted_servers,
                        load_multiplier=args.load_multiplier,
                    )
                else:
                    descriptor = apply_service_hotspot(
                        simulator=simulator,
                        scenario_seed=stream_seed,
                        segment=segment,
                        hotspot_share=args.hotspot_share,
                    )
                exogenous_snapshot = exogenous_stream_snapshot(
                    simulator,
                    descriptor,
                )
                fingerprint = scenario_fingerprint(
                    exogenous_snapshot
                )
                base_fingerprint = base_scenario_fingerprint(
                    simulator
                )
                oracle_bound = (
                    scenario_capacity_aware_oracle_bounds(
                    simulator,
                    include_exogenous_waiting=(
                        args.regime == "server_load_shift"
                    ),
                    )
                )

                apply_frozen_state(simulator, dynamic_state)
                for user in simulator.users.values():
                    user.setpos0()
                simulator.set_training(
                    False,
                    update_caching=(
                        args.cache_mode == "adaptive"
                    ),
                )
                simulator.reset()
                static_state = capture_frozen_state(simulator)
                pre_window_execution_ema = dict(
                    simulator.broker
                    .cache_server_execution_latency_ema
                )
                pre_window_server_quality = (
                    simulator.broker.causal_server_quality()
                )
                before = cache_snapshot(simulator)
                run_started = time.perf_counter()
                simulator.run()
                wall_time = time.perf_counter() - run_started
                row = collect_episode_metrics(
                    simulator=simulator,
                    label=label,
                    seed=seed,
                    phase="online_eval",
                    episode=episode,
                    before=before,
                    wall_time=wall_time,
                    scenario_seed=stream_seed,
                    scenario_hash=fingerprint,
                    base_scenario_hash=base_fingerprint,
                )
                row["stream_regime"] = args.regime
                row["stream_segment"] = segment
                row["hotspot_services"] = ",".join(
                    str(service_id)
                    for service_id
                    in descriptor.get("hotspot_services", [])
                )
                row["shifted_server_ids"] = ",".join(
                    str(server_id)
                    for server_id
                    in descriptor.get("shifted_server_ids", [])
                )
                row["load_multiplier"] = float(
                    descriptor.get("load_multiplier", 1.0)
                )
                row["load_vector_json"] = json.dumps(
                    descriptor.get(
                        "load_vector",
                        {
                            str(server_id): float(server.load)
                            for server_id, server
                            in simulator.servers.items()
                        },
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                row["pre_window_execution_latency_ema_json"] = (
                    json.dumps(
                        pre_window_execution_ema,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                row["pre_window_server_quality_json"] = json.dumps(
                    pre_window_server_quality,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                row["observed_server_execution_latency_json"] = (
                    json.dumps(
                        observed_server_execution_latencies(
                            simulator
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                row["control_payload_bytes"] = (
                    estimated_control_payload_bytes(
                        simulator,
                        row["cache_decision_calls"],
                    )
                )
                row["migration_delay_applied_sec"] = (
                    pending_migration_delay
                )
                row["migration_adjusted_finish_time"] = (
                    row["average_finish_time"]
                    + pending_migration_delay
                )
                row["migration_adjusted_p95_finish_time"] = (
                    row["p95_finish_time"]
                    + pending_migration_delay
                )
                row["dynamic_oracle_floor"] = float(
                    oracle_bound["mean"]
                )
                row["dynamic_oracle_p95"] = float(
                    oracle_bound["p95"]
                )
                row["dynamic_oracle_gap_sec"] = (
                    row["migration_adjusted_finish_time"]
                    - row["dynamic_oracle_floor"]
                )
                row["dynamic_oracle_regret"] = max(
                    row["dynamic_oracle_gap_sec"],
                    0.0,
                )
                row["dynamic_oracle_capacity_ok"] = int(
                    oracle_bound["capacity_constraints_satisfied"]
                )
                row["dynamic_oracle_cache_matrix_json"] = (
                    json.dumps(
                        oracle_bound["cache_placement"],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                writer.writerow(row)
                rows.append(row)
                pending_migration_delay = float(
                    row["cache_migration_critical_time_sec"]
                )
                verify_frozen_state(
                    simulator,
                    static_state,
                    check_cache=(
                        args.cache_mode == "frozen"
                    ),
                )
                dynamic_state = capture_frozen_state(simulator)
                scenarios.append(
                    {
                        "episode": episode,
                        "seed": stream_seed,
                        "fingerprint": fingerprint,
                        "oracle_capacity_constraints_satisfied": (
                            oracle_bound[
                                "capacity_constraints_satisfied"
                            ]
                        ),
                        **descriptor,
                    }
                )
                del simulator

                if (
                    not args.quiet
                    and (
                        episode % max(1, args.episodes // 10) == 0
                        or episode == args.episodes
                    )
                ):
                    print(
                        f"{label} seed={seed} online "
                        f"{episode}/{args.episodes} "
                        f"finish={row['average_finish_time']:.6f}"
                    )

    summaries = segment_summary(rows, args.shift_episode)
    validate_dynamic_accounting(rows)
    performance_metric, performance_p95_metric = (
        performance_metric_names(args.regime)
    )
    recovery = recovery_metrics(
        rows,
        args.shift_episode,
        args.recovery_window,
        metric=performance_metric,
        tolerance=args.recovery_tolerance,
    )
    output_summary = {
        "status": "complete",
        "label": label,
        "seed": seed,
        "selected_checkpoint_episode": checkpoint["episode"],
        "information_protocol_version": INFORMATION_PROTOCOL_VERSION,
        "protocol": {
            "version": ONLINE_PROTOCOL_VERSION,
            "name": (
                "causal_online_server_load_shift_v2"
                if args.regime == "server_load_shift"
                else "causal_online_cache_hotspot_shift_v2"
            ),
            "regime": args.regime,
            "performance_metric": performance_metric,
            "load_multiplier": args.load_multiplier,
            "shifted_servers_by_capacity": {
                str(capacity): int(server_id)
                for capacity, server_id
                in shifted_servers.items()
            },
            "legacy_cache_mode_name": (
                "causal_online_cache_hotspot_shift_v1"
                if args.cache_mode == "adaptive"
                else "frozen_cache_hotspot_shift_v1"
            ),
            "cache_mode": args.cache_mode,
            "episodes": args.episodes,
            "shift_episode": args.shift_episode,
            "hotspot_share": args.hotspot_share,
            "seed_offset": args.seed_offset,
            "recovery_window": args.recovery_window,
            "recovery_tolerance": args.recovery_tolerance,
            "offloading_model_frozen": True,
            "cache_updates_enabled": (
                args.cache_mode == "adaptive"
            ),
            "deployment_fixed": True,
            "workloads_independently_generated": True,
            "oracle": (
                "clairvoyant_capacity_feasible_relaxed_assignment_v1"
            ),
            "oracle_future_workload_visible": True,
            "oracle_capacity_constraints_enforced": all(
                int(row["dynamic_oracle_capacity_ok"])
                for row in rows
            ),
            "migration_delay_carried_to_next_window": True,
        },
        "segments": summaries,
        "adaptation": recovery,
        "cache_decision_calls": int(
            sum(row["cache_decision_calls"] for row in rows)
        ),
        "cache_decision_wall_time_sec": float(
            sum(
                row["cache_decision_wall_time_sec"]
                for row in rows
            )
        ),
        "control_payload_bytes": int(
            sum(row["control_payload_bytes"] for row in rows)
        ),
        "cache_migration_events": int(
            sum(row["cache_migration_events"] for row in rows)
        ),
        "cache_migration_time_sec": float(
            sum(row["cache_migration_time_sec"] for row in rows)
        ),
        "cache_migration_critical_time_sec": float(
            sum(
                row["cache_migration_critical_time_sec"]
                for row in rows
            )
        ),
        "cumulative_oracle_regret": float(
            sum(row["dynamic_oracle_regret"] for row in rows)
        ),
        "unique_scenarios": len(
            {row["scenario_fingerprint"] for row in rows}
        ),
        "total_wall_time_sec": time.perf_counter() - started_total,
    }
    write_json(
        run_dir
        / artifact_name(
            args,
            "online_stream",
            "_scenarios.json",
        ),
        scenarios,
    )
    write_json(output_summary_path, output_summary)
    return {
        "label": label,
        "seed": seed,
        "run_dir": str(run_dir),
        "skipped": False,
    }


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(
        stats.t.ppf(0.975, values.size - 1)
        * stats.sem(values)
    )


def paired_metric(daoc, ours, lower_is_better=True):
    daoc = np.asarray(daoc, dtype=float)
    ours = np.asarray(ours, dtype=float)
    direction = 1.0 if lower_is_better else -1.0
    improvements = (
        100.0
        * direction
        * (daoc - ours)
        / np.maximum(np.abs(daoc), 1e-12)
    )
    differences = ours - daoc
    if daoc.size < 2 or np.allclose(differences, 0.0):
        wilcoxon_p = 1.0
        paired_t_p = 1.0
    else:
        wilcoxon_p = float(
            stats.wilcoxon(
                ours,
                daoc,
                alternative=(
                    "less" if lower_is_better else "greater"
                ),
            ).pvalue
        )
        paired_t_p = float(stats.ttest_rel(ours, daoc).pvalue)
    return {
        "pairs": int(daoc.size),
        "daoc_mean": float(daoc.mean()),
        "daoc_ci95_half_width": confidence_interval(daoc),
        "our_mean": float(ours.mean()),
        "our_ci95_half_width": confidence_interval(ours),
        "wins": int(
            np.sum(ours < daoc)
            if lower_is_better
            else np.sum(ours > daoc)
        ),
        "mean_paired_improvement_percent": float(
            improvements.mean()
        ),
        "paired_improvement_ci95_half_width": (
            confidence_interval(improvements)
        ),
        "wilcoxon_one_sided_p": wilcoxon_p,
        "paired_t_two_sided_p": paired_t_p,
    }


def read_online_rows(run_dir, args):
    with (
        run_dir / artifact_name(args, "online_stream", ".csv")
    ).open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        return list(csv.DictReader(input_file))


def discover_runs(suite_dir, labels, seeds):
    manifest = read_json(suite_dir / "suite_manifest.json")
    profile = manifest.get("profile_config", {})
    selected_labels = labels
    if selected_labels is None:
        available = profile.get("labels", [])
        candidates = [
            label
            for label in available
            if label in {"guided_full", "lean_our", "our"}
        ]
        selected_labels = candidates
    selected_seeds = seeds or profile.get("seeds", [])
    runs = []
    for label in selected_labels:
        for seed in selected_seeds:
            run_dir = suite_dir / "runs" / label / f"seed_{seed}"
            if (run_dir / "summary.json").exists():
                runs.append(run_dir)
    if not runs:
        raise RuntimeError(f"No matching runs found in {suite_dir}")
    return selected_labels, selected_seeds, runs


def numeric_segment_metric(summary, segment, metric):
    return float(
        summary["segments"][segment][f"mean_{metric}"]
    )


def effective_recovery_delay(summary):
    delay = summary["adaptation"]["adaptation_delay_windows"]
    if delay is not None:
        return float(delay)
    protocol = summary["protocol"]
    return float(
        protocol["episodes"] - protocol["shift_episode"] + 2
    )


def performance_metric_names(regime):
    if regime == "server_load_shift":
        return (
            "migration_adjusted_finish_time",
            "migration_adjusted_p95_finish_time",
        )
    return "average_finish_time", "p95_finish_time"


def aggregate_suite(suite_dir, labels, seeds, args):
    if "guided_full" not in labels or len(labels) != 2:
        raise RuntimeError(
            "Paired online aggregation requires guided_full and "
            "exactly one OUR label"
        )
    daoc_label = "guided_full"
    our_label = next(
        label for label in labels if label != daoc_label
    )
    performance_metric, performance_p95_metric = (
        performance_metric_names(args.regime)
    )
    per_seed = []
    trajectories = {"daoc": [], "our": []}
    for seed in seeds:
        run_dirs = {
            label: suite_dir / "runs" / label / f"seed_{seed}"
            for label in labels
        }
        summaries = {
            label: read_json(
                run_dir
                / artifact_name(
                    args,
                    "online_stream",
                    "_summary.json",
                )
            )
            for label, run_dir in run_dirs.items()
        }
        rows = {
            label: read_online_rows(run_dir, args)
            for label, run_dir in run_dirs.items()
        }
        fingerprints = {
            label: tuple(
                row["scenario_fingerprint"]
                for row in label_rows
            )
            for label, label_rows in rows.items()
        }
        if fingerprints[daoc_label] != fingerprints[our_label]:
            raise RuntimeError(
                f"Online stream fingerprints differ for seed {seed}"
            )
        if not all(
            int(row["dynamic_oracle_capacity_ok"])
            for method_rows in rows.values()
            for row in method_rows
        ):
            raise RuntimeError(
                f"Dynamic Oracle capacity audit failed for seed {seed}"
            )
        trajectories["daoc"].append(
            [
                float(row[performance_metric])
                for row in rows[daoc_label]
            ]
        )
        trajectories["our"].append(
            [
                float(row[performance_metric])
                for row in rows[our_label]
            ]
        )
        daoc = summaries[daoc_label]
        ours = summaries[our_label]
        per_seed.append(
            {
                "seed": seed,
                "scenario_fingerprints_match": 1,
                "daoc_overall_finish_time": numeric_segment_metric(
                    daoc,
                    "overall",
                    performance_metric,
                ),
                "our_overall_finish_time": numeric_segment_metric(
                    ours,
                    "overall",
                    performance_metric,
                ),
                "daoc_post_early_finish_time": numeric_segment_metric(
                    daoc,
                    "post_shift_early",
                    performance_metric,
                ),
                "our_post_early_finish_time": numeric_segment_metric(
                    ours,
                    "post_shift_early",
                    performance_metric,
                ),
                "daoc_post_early_p95": numeric_segment_metric(
                    daoc,
                    "post_shift_early",
                    performance_p95_metric,
                ),
                "our_post_early_p95": numeric_segment_metric(
                    ours,
                    "post_shift_early",
                    performance_p95_metric,
                ),
                "daoc_post_late_finish_time": numeric_segment_metric(
                    daoc,
                    "post_shift_late",
                    performance_metric,
                ),
                "our_post_late_finish_time": numeric_segment_metric(
                    ours,
                    "post_shift_late",
                    performance_metric,
                ),
                "daoc_transient_regret": daoc["adaptation"][
                    "normalized_transient_regret"
                ],
                "our_transient_regret": ours["adaptation"][
                    "normalized_transient_regret"
                ],
                "daoc_oracle_regret": daoc["adaptation"][
                    "cumulative_oracle_regret"
                ],
                "our_oracle_regret": ours["adaptation"][
                    "cumulative_oracle_regret"
                ],
                "daoc_adaptation_delay": (
                    effective_recovery_delay(daoc)
                ),
                "our_adaptation_delay": (
                    effective_recovery_delay(ours)
                ),
                "daoc_cache_hit_rate": numeric_segment_metric(
                    daoc,
                    "overall",
                    "cache_hit_rate",
                ),
                "our_cache_hit_rate": numeric_segment_metric(
                    ours,
                    "overall",
                    "cache_hit_rate",
                ),
                "daoc_cache_replacements": numeric_segment_metric(
                    daoc,
                    "overall",
                    "cache_replacements",
                ),
                "our_cache_replacements": numeric_segment_metric(
                    ours,
                    "overall",
                    "cache_replacements",
                ),
                "daoc_migration_time_sec": daoc[
                    "cache_migration_time_sec"
                ],
                "our_migration_time_sec": ours[
                    "cache_migration_time_sec"
                ],
                "daoc_cache_decision_wall_time_sec": daoc[
                    "cache_decision_wall_time_sec"
                ],
                "our_cache_decision_wall_time_sec": ours[
                    "cache_decision_wall_time_sec"
                ],
                "daoc_control_payload_bytes": daoc[
                    "control_payload_bytes"
                ],
                "our_control_payload_bytes": ours[
                    "control_payload_bytes"
                ],
            }
        )

    output_csv = suite_dir / artifact_name(
        args,
        "online_stream",
        "_per_seed.csv",
    )
    with output_csv.open(
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

    comparisons = {}
    for name, daoc_key, our_key in (
        (
            "overall_finish_time",
            "daoc_overall_finish_time",
            "our_overall_finish_time",
        ),
        (
            "post_shift_early_finish_time",
            "daoc_post_early_finish_time",
            "our_post_early_finish_time",
        ),
        (
            "post_shift_early_p95",
            "daoc_post_early_p95",
            "our_post_early_p95",
        ),
        (
            "post_shift_late_finish_time",
            "daoc_post_late_finish_time",
            "our_post_late_finish_time",
        ),
        (
            "transient_regret",
            "daoc_transient_regret",
            "our_transient_regret",
        ),
        (
            "cumulative_oracle_regret",
            "daoc_oracle_regret",
            "our_oracle_regret",
        ),
        (
            "adaptation_delay",
            "daoc_adaptation_delay",
            "our_adaptation_delay",
        ),
    ):
        comparisons[name] = paired_metric(
            [row[daoc_key] for row in per_seed],
            [row[our_key] for row in per_seed],
        )
    comparisons["cache_hit_rate"] = paired_metric(
        [row["daoc_cache_hit_rate"] for row in per_seed],
        [row["our_cache_hit_rate"] for row in per_seed],
        lower_is_better=False,
    )
    summary = {
        "status": "complete",
        "seeds": len(per_seed),
        "daoc_label": daoc_label,
        "our_label": our_label,
        "all_scenario_fingerprints_match": True,
        "all_oracle_capacity_constraints_satisfied": True,
        "protocol": ours["protocol"],
        "comparisons": comparisons,
        "coordination": {
            "daoc_mean_decision_wall_time_sec": float(
                np.mean(
                    [
                        row["daoc_cache_decision_wall_time_sec"]
                        for row in per_seed
                    ]
                )
            ),
            "our_mean_decision_wall_time_sec": float(
                np.mean(
                    [
                        row["our_cache_decision_wall_time_sec"]
                        for row in per_seed
                    ]
                )
            ),
            "daoc_mean_control_payload_bytes": float(
                np.mean(
                    [
                        row["daoc_control_payload_bytes"]
                        for row in per_seed
                    ]
                )
            ),
            "our_mean_control_payload_bytes": float(
                np.mean(
                    [
                        row["our_control_payload_bytes"]
                        for row in per_seed
                    ]
                )
            ),
            "payload_model": (
                "OUR: S*(Q+4)*float32 uplink plus "
                "sum(K_s)*uint16 downlink per cache decision; "
                "DAOC native local cache decisions: zero coordination bytes"
            ),
            "daoc_mean_migration_time_sec": float(
                np.mean(
                    [
                        row["daoc_migration_time_sec"]
                        for row in per_seed
                    ]
                )
            ),
            "our_mean_migration_time_sec": float(
                np.mean(
                    [
                        row["our_migration_time_sec"]
                        for row in per_seed
                    ]
                )
            ),
        },
    }
    write_json(
        suite_dir
        / artifact_name(
            args,
            "online_stream",
            "_summary.json",
        ),
        summary,
    )
    shift_episode = ours["protocol"]["shift_episode"]
    plot_online_trajectory(
        suite_dir
        / artifact_name(
            args,
            "online_stream",
            "_comparison",
        ),
        trajectories,
        per_seed,
        shift_episode,
        shift_label=(
            "Server load shift"
            if args.regime == "server_load_shift"
            else "Hotspot shift"
        ),
    )
    write_online_report(
        suite_dir
        / artifact_name(
            args,
            "ONLINE_STREAM",
            "_REPORT.md",
        ),
        summary,
    )
    return summary


def plot_online_trajectory(
    path,
    trajectories,
    per_seed,
    shift_episode,
    shift_label="Hotspot shift",
):
    daoc = np.asarray(trajectories["daoc"], dtype=float)
    ours = np.asarray(trajectories["our"], dtype=float)
    episodes = np.arange(1, daoc.shape[1] + 1)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
    )
    for values, label, color in (
        (daoc, "DAOC", "#4B5563"),
        (ours, "OUR", "#DC2626"),
    ):
        mean = values.mean(axis=0)
        half_width = (
            stats.t.ppf(0.975, values.shape[0] - 1)
            * stats.sem(values, axis=0)
            if values.shape[0] > 1
            else np.zeros_like(mean)
        )
        axes[0].plot(episodes, mean, label=label, color=color)
        axes[0].fill_between(
            episodes,
            mean - half_width,
            mean + half_width,
            color=color,
            alpha=0.16,
        )
    axes[0].axvline(
        shift_episode,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=shift_label,
    )
    axes[0].set_xlabel("Online workload window")
    axes[0].set_ylabel("Mean application finish time (s)")
    axes[0].set_title("Independent online stream", loc="left")
    axes[0].legend(frameon=False)

    x = np.arange(len(per_seed))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [row["daoc_transient_regret"] for row in per_seed],
        width,
        label="DAOC",
        color="#4B5563",
    )
    axes[1].bar(
        x + width / 2,
        [row["our_transient_regret"] for row in per_seed],
        width,
        label="OUR",
        color="#DC2626",
    )
    axes[1].set_xticks(
        x,
        [str(row["seed"]) for row in per_seed],
    )
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Normalized transient regret")
    axes[1].set_title("Post-shift adaptation cost", loc="left")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_online_report(path, summary):
    comparisons = summary["comparisons"]
    overall = comparisons["overall_finish_time"]
    early = comparisons["post_shift_early_finish_time"]
    early_p95 = comparisons["post_shift_early_p95"]
    late = comparisons["post_shift_late_finish_time"]
    regret = comparisons["transient_regret"]
    oracle_regret = comparisons["cumulative_oracle_regret"]
    coordination = summary["coordination"]
    load_shift = (
        summary["protocol"]["regime"] == "server_load_shift"
    )
    post_shift_windows = (
        summary["protocol"]["episodes"]
        - summary["protocol"]["shift_episode"]
        + 1
    )
    edge_window = min(10, post_shift_windows)
    artifact_stem = (
        "online_stream"
        if summary["protocol"]["cache_updates_enabled"]
        else "online_stream_frozen"
    )
    text = "\n".join(
        [
            (
                "# E3 Server-Load-Shift Evaluation"
                if load_shift
                else "# Online Hotspot-Shift Evaluation"
            ),
            "",
            "## Integrity",
            "",
            f"- {summary['seeds']} paired seeds.",
            (
                "- Offloading weights are frozen; causal cache state updates."
                if summary["protocol"]["cache_updates_enabled"]
                else "- Offloading weights and cache state are both frozen."
            ),
            "- Edge deployment is fixed and every workload is independently generated.",
            "- DAOC and OUR scenario fingerprints match exactly for every window.",
            "",
            "## Performance",
            "",
            "- Overall finish time: "
            f"{overall['daoc_mean']:.6f} s (DAOC) vs "
            f"{overall['our_mean']:.6f} s (OUR), paired improvement "
            f"{overall['mean_paired_improvement_percent']:.2f}%.",
            f"- First {edge_window} post-shift windows: "
            f"{early['daoc_mean']:.6f} s vs {early['our_mean']:.6f} s, "
            f"paired improvement {early['mean_paired_improvement_percent']:.2f}%.",
            f"- First {edge_window} post-shift-window P95: "
            f"{early_p95['daoc_mean']:.6f} s vs "
            f"{early_p95['our_mean']:.6f} s, paired improvement "
            f"{early_p95['mean_paired_improvement_percent']:.2f}%.",
            f"- Last {edge_window} post-shift windows: "
            f"{late['daoc_mean']:.6f} s vs {late['our_mean']:.6f} s, "
            f"paired improvement {late['mean_paired_improvement_percent']:.2f}%.",
            "- Normalized transient regret: "
            f"{regret['daoc_mean']:.4f} vs {regret['our_mean']:.4f}, "
            f"paired improvement {regret['mean_paired_improvement_percent']:.2f}%.",
            "- Cumulative clairvoyant-reference regret: "
            f"{oracle_regret['daoc_mean']:.6f} vs "
            f"{oracle_regret['our_mean']:.6f}, paired improvement "
            f"{oracle_regret['mean_paired_improvement_percent']:.2f}%.",
            "",
            "## Coordination",
            "",
            "- Mean measured cache-decision wall time per run: "
            f"{coordination['daoc_mean_decision_wall_time_sec']:.6f} s "
            f"(DAOC) vs {coordination['our_mean_decision_wall_time_sec']:.6f} s "
            "(OUR).",
            "- Mean estimated control payload per run: "
            f"{coordination['daoc_mean_control_payload_bytes']:.0f} bytes "
            f"(DAOC) vs {coordination['our_mean_control_payload_bytes']:.0f} bytes "
            "(OUR).",
            "- Payload accounting uses sufficient statistics, not raw tasks.",
            "- Mean estimated service-migration time per run: "
            f"{coordination['daoc_mean_migration_time_sec']:.6f} s "
            f"(DAOC) vs {coordination['our_mean_migration_time_sec']:.6f} s "
            "(OUR).",
            "",
            "## Artifacts",
            "",
            f"- `{artifact_stem}_per_seed.csv`",
            f"- `{artifact_stem}_summary.json`",
            f"- `{artifact_stem}_comparison.png` and `.pdf`",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def run_suite(args):
    suite_dir = args.suite_dir.resolve()
    labels, seeds, run_dirs = discover_runs(
        suite_dir,
        args.labels,
        args.seeds,
    )
    manifest_path = suite_dir / artifact_name(
        args,
        "online_stream",
        "_manifest.json",
    )
    manifest = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "labels": labels,
        "seeds": seeds,
        "episodes": args.episodes,
        "shift_episode": args.shift_episode,
        "regime": args.regime,
        "load_multiplier": args.load_multiplier,
        "hotspot_share": args.hotspot_share,
        "seed_offset": args.seed_offset,
        "recovery_window": args.recovery_window,
        "recovery_tolerance": args.recovery_tolerance,
        "cache_mode": args.cache_mode,
        "workers": args.workers,
        "completed_runs": 0,
        "failed_runs": [],
    }
    write_json(manifest_path, manifest)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    matplotlib_dir = suite_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(matplotlib_dir)

    def execute(run_dir):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(run_dir),
            "--episodes",
            str(args.episodes),
            "--shift-episode",
            str(args.shift_episode),
            "--regime",
            args.regime,
            "--load-multiplier",
            str(args.load_multiplier),
            "--hotspot-share",
            str(args.hotspot_share),
            "--seed-offset",
            str(args.seed_offset),
            "--recovery-window",
            str(args.recovery_window),
            "--recovery-tolerance",
            str(args.recovery_tolerance),
            "--cache-mode",
            args.cache_mode,
            "--quiet",
        ]
        if args.resume:
            command.append("--resume")
        with (
            run_dir
            / artifact_name(
                args,
                "online_stream",
                "_run.log",
            )
        ).open(
            "w",
            encoding="utf-8",
        ) as output_file:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent,
                env=env,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return run_dir, result.returncode

    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(execute, run_dir)
            for run_dir in run_dirs
        ]
        for future in as_completed(futures):
            run_dir, returncode = future.result()
            if returncode:
                failures.append(
                    {
                        "run_dir": str(run_dir),
                        "returncode": returncode,
                    }
                )
                print(
                    f"[online failed] {run_dir}; "
                    "see the per-run online stream log"
                )
            else:
                completed += 1
                print(
                    f"[online {completed}/{len(run_dirs)}] "
                    f"{run_dir.parent.name} {run_dir.name}"
                )
            manifest["completed_runs"] = completed
            manifest["failed_runs"] = failures
            write_json(manifest_path, manifest)

    if failures:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        write_json(manifest_path, manifest)
        raise RuntimeError(
            f"{len(failures)} online stream runs failed"
        )

    aggregate_suite(suite_dir, labels, seeds, args)
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    write_json(manifest_path, manifest)
    print(f"Online stream results: {suite_dir}")


def main():
    args = parse_args()
    if args.run_dir is not None:
        result = evaluate_run(args.run_dir, args)
        print(json.dumps(result, indent=2))
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
