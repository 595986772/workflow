#!/usr/bin/env python3
"""Evaluate H3 checkpoints on static H0-H4 capacities without retraining."""

import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_strict_environment_suite import paired_statistics
from run_independent_experiment import (
    EPISODE_FIELDS,
    apply_frozen_state,
    base_scenario_fingerprint,
    capture_deployment_state,
    capture_frozen_state,
    run_scenario_bank_evaluation,
    scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
    summarize_rows,
)
from simulator import MEC_Simulator
from static_heterogeneity_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    MAIN_PROFILE,
    STATIC_HETEROGENEITY_PROTOCOL_VERSION,
)


DEFAULT_LABELS = ("guided_full", "lean_our")


def parse_list(value):
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def parse_int_list(value):
    return [int(item) for item in parse_list(value)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project frozen H3 checkpoints onto static capacity "
            "profiles and evaluate without retraining."
        )
    )
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=parse_list,
        default=list(DEFAULT_LABELS),
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        required=True,
    )
    parser.add_argument(
        "--profiles",
        type=parse_list,
        default=list(CAPACITY_PROFILES),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    unknown = set(args.profiles) - set(CAPACITY_PROFILES)
    if unknown:
        raise ValueError(f"Unknown capacity profiles: {sorted(unknown)}")
    if args.episodes < 1 or args.workers < 1:
        raise ValueError("episodes and workers must be positive")
    return args


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def set_services(simulator, decisions):
    simulator.server_service_info.fill(0.0)
    for server_id, server in simulator.servers.items():
        services = [
            int(service)
            for service in decisions[server_id]
            if int(service) > 0
        ]
        if len(services) > server.capacity:
            raise RuntimeError(
                f"Projected cache exceeds server {server_id} capacity"
            )
        server.services = [0] + services
        for service in services:
            simulator.server_service_info[
                server_id,
                service - 1,
            ] = 1.0


def project_frozen_state(simulator, source_state):
    """Project source history onto target capacities without workload data."""
    actual_capacities = {
        server_id: int(server.capacity)
        for server_id, server in simulator.servers.items()
    }
    source_capacities = {
        int(server_id): int(capacity)
        for server_id, capacity
        in source_state["capacities"].items()
    }
    if actual_capacities == source_capacities:
        return copy.deepcopy(source_state), {
            "mode": "identity",
            "used_future_target_workload": False,
        }

    projected = copy.deepcopy(source_state)
    projected["capacities"] = actual_capacities
    projected["services"] = {
        server_id: (0,)
        for server_id in simulator.servers
    }
    projected["replay_sizes"] = {
        server_id: len(
            server.agent.agent.TrainNet.experience["s"]
        )
        for server_id, server in simulator.servers.items()
    }
    apply_frozen_state(simulator, projected)

    broker = simulator.broker
    minimum_residence = (
        simulator.cache_update_interval
        * simulator.cache_min_residence_updates
    )
    broker.cache_round = max(
        int(broker.cache_round),
        int(minimum_residence),
    )
    broker.last_cache_change_round = {
        server_id: 0
        for server_id in simulator.servers
    }
    if (
        simulator.cache_policy
        in {"critical_path_coordinated", "critical_path_joint"}
        and broker.cache_observations > 0
    ):
        decisions = broker.coordinated_caching_decisions()
        decision_mode = "native_coordinated_history_projection"
    else:
        decisions = {
            server_id: broker.caching_decisions(server_id)
            for server_id in simulator.servers
        }
        decision_mode = "native_independent_history_projection"
    set_services(simulator, decisions)
    frozen = capture_frozen_state(simulator)
    return frozen, {
        "mode": decision_mode,
        "used_checkpoint_weights": True,
        "used_checkpoint_cache_estimates": True,
        "used_checkpoint_history_ema": True,
        "used_future_target_workload": False,
        "target_capacities": actual_capacities,
        "projected_services": {
            server_id: [
                service
                for service
                in simulator.servers[server_id].services
                if service > 0
            ]
            for server_id in simulator.servers
        },
    }


def expected_complete(path, profile, episodes):
    if not path.exists():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "complete"
        and summary.get("profile") == profile
        and summary.get("eval_episodes") == episodes
        and summary.get("protocol_version")
        == STATIC_HETEROGENEITY_PROTOCOL_VERSION
    )


def evaluate_one(spec):
    (
        source_suite,
        output_root,
        label,
        seed,
        profile,
        episodes,
        resume,
    ) = spec
    source_run = (
        source_suite / "runs" / label / f"seed_{seed}"
    )
    target_run = (
        output_root / profile / "runs" / label / f"seed_{seed}"
    )
    target_run.mkdir(parents=True, exist_ok=True)
    summary_path = target_run / "summary.json"
    if resume and expected_complete(
        summary_path,
        profile,
        episodes,
    ):
        return {
            "label": label,
            "seed": seed,
            "profile": profile,
            "skipped": True,
        }

    source_summary = read_json(source_run / "summary.json")
    if (
        source_summary.get("status") != "complete"
        or source_summary.get("eligible_for_comparison") is not True
    ):
        raise RuntimeError(
            f"Source checkpoint is not eligible: {source_run}"
        )
    config = read_json(source_run / "config.json")
    source_arguments = config["arguments"]
    input_config = copy.deepcopy(config["input_config"])
    input_config.update(
        {
            "server capacity": 1,
            "server capacity multiset": (
                CAPACITY_PROFILES[profile]
            ),
            "capacity assignment namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
            "baseline server capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "seed": seed,
            "save topology figure": False,
        }
    )
    learning_config = copy.deepcopy(config["learning_config"])
    checkpoint = torch.load(
        source_run / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    source_state = checkpoint["frozen_state"]

    simulator_log_path = target_run / "simulator.log"
    with simulator_log_path.open("w", encoding="utf-8") as log:
        seed_everything(seed)
        target_simulator = MEC_Simulator(
            outputfile=log,
            Input_dict=input_config,
            learning_arguments=learning_config,
            filename_png=str(target_run),
        )
        source_initial = read_json(
            source_run / "scenario_initial.json"
        )
        if (
            base_scenario_fingerprint(target_simulator)
            != base_scenario_fingerprint_from_snapshot(
                source_initial
            )
        ):
            raise RuntimeError(
                "Target capacity changed the physical deployment: "
                f"{label} seed={seed} profile={profile}"
            )
        deployment_state = capture_deployment_state(
            target_simulator
        )
        projected_state, projection = project_frozen_state(
            target_simulator,
            source_state,
        )

        args = SimpleNamespace(
            eval_episodes=episodes,
            eval_seed_offset=int(
                source_arguments["eval_seed_offset"]
            ),
            seed=seed,
            eval_bank_scope="workload",
            label=label,
            quiet=True,
        )
        episodes_path = target_run / "episodes.csv"
        with episodes_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output:
            writer = csv.DictWriter(
                output,
                fieldnames=EPISODE_FIELDS,
            )
            writer.writeheader()
            rows, scenarios = run_scenario_bank_evaluation(
                writer=writer,
                csv_file=output,
                args=args,
                input_config=input_config,
                learning_config=learning_config,
                frozen_state=projected_state,
                deployment_state=deployment_state,
                simulator_log=log,
                figures_dir=target_run,
                episodes=episodes,
                phase="eval",
            )

    write_json(
        target_run / "evaluation_scenarios.json",
        scenarios,
    )
    write_json(
        target_run / "projection.json",
        projection,
    )
    summary = {
        "status": "complete",
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "source_profile": MAIN_PROFILE,
        "profile": profile,
        "label": label,
        "seed": seed,
        "source_run": str(source_run.resolve()),
        "source_checkpoint_episode": checkpoint["episode"],
        "eval_episodes": episodes,
        "server_capacities": {
            str(server_id): int(capacity)
            for server_id, capacity
            in projected_state["capacities"].items()
        },
        "projection": projection,
        "eval": summarize_rows(rows),
    }
    write_json(summary_path, summary)
    return {
        "label": label,
        "seed": seed,
        "profile": profile,
        "skipped": False,
    }


def base_scenario_fingerprint_from_snapshot(snapshot):
    clean = copy.deepcopy(snapshot)
    clean.pop("algorithm", None)
    clean.pop("environment_stress", None)
    for user in clean["users"].values():
        user.pop("dag_stress", None)
    for server in clean["servers"].values():
        server.pop("cached_services", None)
        server.pop("cache_capacity", None)
    return scenario_fingerprint(clean)


def evaluation_rows(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def aggregate_output(output_root, profiles, labels, seeds):
    profile_results = {}
    all_paired = True
    base_by_profile_seed = {}
    for profile in profiles:
        per_seed = []
        for seed in seeds:
            rows = {}
            for label in labels:
                run = (
                    output_root
                    / profile
                    / "runs"
                    / label
                    / f"seed_{seed}"
                )
                rows[label] = evaluation_rows(
                    run / "episodes.csv"
                )
            fingerprints = {
                label: [
                    row["scenario_fingerprint"]
                    for row in label_rows
                ]
                for label, label_rows in rows.items()
            }
            all_paired = all_paired and (
                len({tuple(value) for value in fingerprints.values()})
                == 1
            )
            base_by_profile_seed[(profile, seed)] = [
                row["base_scenario_fingerprint"]
                for row in rows[labels[0]]
            ]
            entry = {"seed": seed}
            for label in labels:
                entry[label] = float(
                    np.mean(
                        [
                            float(row["average_finish_time"])
                            for row in rows[label]
                        ]
                    )
                )
            per_seed.append(entry)
        profile_results[profile] = {
            "per_seed": per_seed,
            "comparison": paired_statistics(
                [row[labels[0]] for row in per_seed],
                [row[labels[1]] for row in per_seed],
                lower_is_better=True,
            ),
        }
    cross_profile_paired = all(
        base_by_profile_seed[(profile, seed)]
        == base_by_profile_seed[(MAIN_PROFILE, seed)]
        for profile in profiles
        for seed in seeds
    )
    summary = {
        "status": "complete",
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "source_profile": MAIN_PROFILE,
        "profiles": profile_results,
        "labels": labels,
        "seeds": seeds,
        "all_methods_scenario_paired": all_paired,
        "all_profiles_base_scenario_paired": (
            cross_profile_paired
        ),
        "no_target_retraining": True,
        "no_future_target_workload_in_projection": True,
    }
    write_json(
        output_root / "cross_capacity_summary.json",
        summary,
    )
    plot_cross_capacity(
        output_root / "cross_capacity_generalization",
        profile_results,
        labels,
    )
    return summary


def plot_cross_capacity(path, profile_results, labels):
    ordered = [
        profile
        for profile in CAPACITY_PROFILES
        if profile in profile_results
    ]
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(
        figsize=(7.8, 4.5),
        constrained_layout=True,
    )
    for label, marker, color in (
        (labels[0], "o", "#566573"),
        (labels[1], "s", "#D95F45"),
    ):
        values = [
            float(
                np.mean(
                    [
                        row[label]
                        for row in profile_results[profile][
                            "per_seed"
                        ]
                    ]
                )
            )
            for profile in ordered
        ]
        axis.plot(
            x,
            values,
            marker=marker,
            label=(
                "DAOC" if label == "guided_full" else "OUR"
            ),
            color=color,
        )
    axis.set_xticks(x, ordered)
    axis.set_xlabel("Target cache-capacity profile")
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    args = parse_args()
    torch.set_num_threads(1)
    source_suite = args.source_suite_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            source_suite,
            output_root,
            label,
            seed,
            profile,
            args.episodes,
            args.resume,
        )
        for profile in args.profiles
        for label in args.labels
        for seed in args.seeds
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(evaluate_one, spec)
            for spec in specs
        ]
        for future in as_completed(futures):
            result = future.result()
            action = "skip" if result["skipped"] else "complete"
            print(
                f"[{action}] {result['profile']} "
                f"{result['label']} seed={result['seed']}",
                flush=True,
            )
    summary = aggregate_output(
        output_root,
        args.profiles,
        args.labels,
        args.seeds,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
