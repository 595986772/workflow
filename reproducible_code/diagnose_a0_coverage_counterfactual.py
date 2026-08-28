#!/usr/bin/env python3
"""Evaluate a coverage-first cache counterfactual from frozen OUR states."""

import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from a0_fixed_budget_heterogeneity_protocol import (
    DEVELOPMENT_SEEDS,
    PROFILE_ORDER,
)
from critical_path_cache import service_fetch_savings
from run_independent_experiment import (
    EPISODE_FIELDS,
    apply_frozen_state,
    capture_deployment_state,
    capture_frozen_state,
    run_scenario_bank_evaluation,
    seed_everything,
    summarize_rows,
)
from simulator import MEC_Simulator


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SUITE_ROOT = (
    ROOT_DIR
    / "results"
    / "a0_fixed_budget_heterogeneity"
    / "h8v0"
    / "experiment"
)
EPSILON = 1e-12


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in parse_list(value)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profiles", type=parse_list, default=list(PROFILE_ORDER))
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=list(DEVELOPMENT_SEEDS),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("repair", "rebuild"),
        default="repair",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.suite_root / f"coverage_{args.mode}_counterfactual"
    if set(args.profiles) - set(PROFILE_ORDER):
        raise ValueError("Unknown capacity profile")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def workload_scenario_view(bank):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def coverage_first_cache_decision(
    demand,
    capacities,
    service_sizes,
    cloud_costs,
    between_server_costs,
    server_quality,
):
    """Fill scarce slots with distinct demanded services before replicas."""
    server_ids = sorted(capacities)
    service_ids = sorted(service_sizes)
    proposal = {server_id: [] for server_id in server_ids}
    demanded_services = {
        service_id
        for service_id in service_ids
        if sum(
            max(float(server_demand.get(service_id, 0.0)), 0.0)
            for server_demand in demand.values()
        )
        > EPSILON
    }
    target_slots = sum(int(capacities[server_id]) for server_id in server_ids)
    coverage_target = min(target_slots, len(demanded_services))

    while sum(len(value) for value in proposal.values()) < target_slots:
        covered = {
            service_id
            for services in proposal.values()
            for service_id in services
        }
        coverage_phase = len(covered) < coverage_target
        eligible_services = (
            demanded_services - covered
            if coverage_phase
            else set(service_ids)
        )
        base_value = service_fetch_savings(
            assignments=proposal,
            demand=demand,
            service_sizes=service_sizes,
            cloud_costs=cloud_costs,
            between_server_costs=between_server_costs,
        )
        candidates = []
        for server_id in server_ids:
            if len(proposal[server_id]) >= int(capacities[server_id]):
                continue
            for service_id in sorted(eligible_services):
                if service_id in proposal[server_id]:
                    continue
                candidate = {
                    key: list(value)
                    for key, value in proposal.items()
                }
                candidate[server_id].append(service_id)
                gain = service_fetch_savings(
                    assignments=candidate,
                    demand=demand,
                    service_sizes=service_sizes,
                    cloud_costs=cloud_costs,
                    between_server_costs=between_server_costs,
                ) - base_value
                replica_count = sum(
                    service_id in services
                    for services in proposal.values()
                )
                score = (
                    gain
                    * max(float(server_quality[server_id]), 0.0)
                    / (1.0 + replica_count)
                )
                candidates.append(
                    (
                        score,
                        -server_id,
                        -service_id,
                        server_id,
                        service_id,
                    )
                )
        if not candidates:
            break
        *_, server_id, service_id = max(candidates)
        proposal[server_id].append(service_id)
    return proposal


def coverage_repair_cache_decision(
    current_services,
    demand,
    capacities,
    service_sizes,
    cloud_costs,
    between_server_costs,
    server_quality,
):
    """Replace only redundant replicas needed to reach maximum coverage."""
    proposal = {
        server_id: [
            int(service_id)
            for service_id in current_services[server_id]
            if int(service_id) > 0
        ][: int(capacities[server_id])]
        for server_id in sorted(capacities)
    }
    demanded_services = {
        service_id
        for service_id in sorted(service_sizes)
        if sum(
            max(float(server_demand.get(service_id, 0.0)), 0.0)
            for server_demand in demand.values()
        )
        > EPSILON
    }
    target_slots = sum(int(value) for value in capacities.values())
    coverage_target = min(target_slots, len(demanded_services))

    while True:
        replica_counts = {
            service_id: sum(
                service_id in services
                for services in proposal.values()
            )
            for service_id in service_sizes
        }
        covered = {
            service_id
            for service_id, count in replica_counts.items()
            if count > 0
        }
        if len(covered) >= coverage_target:
            break
        uncovered = demanded_services - covered
        base_value = service_fetch_savings(
            assignments=proposal,
            demand=demand,
            service_sizes=service_sizes,
            cloud_costs=cloud_costs,
            between_server_costs=between_server_costs,
        )
        candidates = []
        for server_id, services in proposal.items():
            for old_service in services:
                if replica_counts[old_service] <= 1:
                    continue
                for new_service in sorted(uncovered):
                    candidate = {
                        key: list(value)
                        for key, value in proposal.items()
                    }
                    candidate[server_id].remove(old_service)
                    candidate[server_id].append(new_service)
                    value = service_fetch_savings(
                        assignments=candidate,
                        demand=demand,
                        service_sizes=service_sizes,
                        cloud_costs=cloud_costs,
                        between_server_costs=between_server_costs,
                    )
                    adjusted_delta = (
                        (value - base_value)
                        * max(float(server_quality[server_id]), 0.0)
                    )
                    candidates.append(
                        (
                            adjusted_delta,
                            -server_id,
                            -old_service,
                            -new_service,
                            server_id,
                            old_service,
                            new_service,
                        )
                    )
        if not candidates:
            break
        *_, server_id, old_service, new_service = max(candidates)
        proposal[server_id].remove(old_service)
        proposal[server_id].append(new_service)
    return proposal


def set_services(simulator, decisions):
    simulator.server_service_info.fill(0.0)
    for server_id, server in simulator.servers.items():
        services = [int(value) for value in decisions[server_id]]
        if len(services) > int(server.capacity):
            raise RuntimeError("Counterfactual cache exceeds capacity")
        server.services = [0] + services
        for service_id in services:
            simulator.server_service_info[server_id, service_id - 1] = 1.0


def build_counterfactual_state(simulator, source_state, mode):
    apply_frozen_state(simulator, source_state)
    broker = simulator.broker
    broker.coordinated_caching_decisions()
    context = broker.last_cache_decision_context
    capacities = {
        server_id: int(server.capacity)
        for server_id, server in simulator.servers.items()
    }
    arguments = {
        "demand": broker.H,
        "capacities": capacities,
        "service_sizes": {
            service_id: simulator.service_data_length[service_id]
            for service_id in range(1, broker.numberofservices + 1)
        },
        "cloud_costs": {
            server_id: 1.0 / server.rate_to_cloud
            for server_id, server in simulator.servers.items()
        },
        "between_server_costs": simulator.between_server_costs,
        "server_quality": context["server_quality"],
    }
    if mode == "repair":
        decisions = coverage_repair_cache_decision(
            current_services=source_state["services"],
            **arguments,
        )
    else:
        decisions = coverage_first_cache_decision(**arguments)
    set_services(simulator, decisions)
    return capture_frozen_state(simulator), decisions, context


def expected_complete(path, profile, seed, episodes, mode):
    if not path.exists():
        return False
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "complete"
        and value.get("diagnostic_only") is True
        and value.get("profile") == profile
        and value.get("seed") == seed
        and value.get("mode") == mode
        and value.get("eval", {}).get("episodes") == episodes
    )


def evaluate_one(
    suite_root,
    output_root,
    profile,
    seed,
    episodes,
    mode,
    resume,
):
    source_run = suite_root / profile / "runs" / "lean_our" / f"seed_{seed}"
    output_run = output_root / profile / f"seed_{seed}"
    output_run.mkdir(parents=True, exist_ok=True)
    summary_path = output_run / "summary.json"
    if resume and expected_complete(
        summary_path,
        profile,
        seed,
        episodes,
        mode,
    ):
        return read_json(summary_path)

    source_summary = read_json(source_run / "summary.json")
    config = read_json(source_run / "config.json")
    input_config = copy.deepcopy(config["input_config"])
    input_config["save topology figure"] = False
    learning_config = copy.deepcopy(config["learning_config"])
    checkpoint = torch.load(
        source_run / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )

    simulator_log_path = output_run / "simulator.log"
    with simulator_log_path.open("w", encoding="utf-8") as simulator_log:
        seed_everything(seed)
        simulator = MEC_Simulator(
            outputfile=simulator_log,
            Input_dict=input_config,
            learning_arguments=learning_config,
            filename_png=str(output_run),
        )
        deployment_state = capture_deployment_state(simulator)
        frozen_state, decisions, context = build_counterfactual_state(
            simulator,
            checkpoint["frozen_state"],
            mode,
        )
        eval_args = SimpleNamespace(
            eval_episodes=episodes,
            eval_seed_offset=int(config["arguments"]["eval_seed_offset"]),
            seed=seed,
            eval_bank_scope="workload",
            label="lean_our_coverage_counterfactual",
            quiet=True,
        )
        with (output_run / "episodes.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output:
            writer = csv.DictWriter(output, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            rows, scenarios = run_scenario_bank_evaluation(
                writer=writer,
                csv_file=output,
                args=eval_args,
                input_config=input_config,
                learning_config=learning_config,
                frozen_state=frozen_state,
                deployment_state=deployment_state,
                simulator_log=simulator_log,
                figures_dir=output_run,
                episodes=episodes,
                phase="eval",
            )

    source_bank = read_json(source_run / "evaluation_scenarios.json")
    paired = workload_scenario_view(scenarios) == workload_scenario_view(source_bank)
    original_services = {
        str(server_id): [
            int(service_id)
            for service_id in services
            if int(service_id) > 0
        ]
        for server_id, services in checkpoint["frozen_state"]["services"].items()
    }
    counterfactual_services = {
        str(server_id): [int(service_id) for service_id in services]
        for server_id, services in decisions.items()
    }
    summary = {
        "status": "complete",
        "diagnostic_only": True,
        "mode": mode,
        "profile": profile,
        "seed": seed,
        "source_run": str(source_run.resolve()),
        "source_checkpoint_episode": checkpoint["episode"],
        "network_weights_unchanged": True,
        "history_only": True,
        "future_workload_used": False,
        "scenario_bank_paired": paired,
        "server_capacities": source_summary["server_capacities"],
        "original_services": original_services,
        "counterfactual_services": counterfactual_services,
        "original_coverage": len(
            {
                service_id
                for services in original_services.values()
                for service_id in services
            }
        ),
        "counterfactual_coverage": len(
            {
                service_id
                for services in counterfactual_services.values()
                for service_id in services
            }
        ),
        "server_quality": context["server_quality"],
        "eval": summarize_rows(rows),
    }
    write_json(output_run / "evaluation_scenarios.json", scenarios)
    write_json(summary_path, summary)
    if not paired:
        raise RuntimeError("Counterfactual scenario bank is not paired")
    return summary


def mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def aggregate(suite_root, output_root, profiles, seeds, mode):
    profile_results = {}
    for profile in profiles:
        per_seed = []
        for seed in seeds:
            counterfactual = read_json(
                output_root / profile / f"seed_{seed}" / "summary.json"
            )
            original = read_json(
                suite_root / profile / "runs" / "lean_our" / f"seed_{seed}" / "summary.json"
            )
            central = read_json(
                suite_root
                / profile
                / "runs"
                / "centralized_greedy_daoc"
                / f"seed_{seed}"
                / "summary.json"
            )
            per_seed.append(
                {
                    "seed": seed,
                    "coverage_original": counterfactual["original_coverage"],
                    "coverage_counterfactual": counterfactual[
                        "counterfactual_coverage"
                    ],
                    "original_mean": original["eval"][
                        "mean_average_finish_time"
                    ],
                    "counterfactual_mean": counterfactual["eval"][
                        "mean_average_finish_time"
                    ],
                    "central_mean": central["eval"]["mean_average_finish_time"],
                    "original_p95": original["eval"]["mean_p95_finish_time"],
                    "counterfactual_p95": counterfactual["eval"][
                        "mean_p95_finish_time"
                    ],
                    "central_p95": central["eval"]["mean_p95_finish_time"],
                    "scenario_bank_paired": counterfactual[
                        "scenario_bank_paired"
                    ],
                }
            )
        profile_results[profile] = {
            "per_seed": per_seed,
            "counterfactual_vs_original_mean_percent": 100.0
            * (
                mean([row["original_mean"] for row in per_seed])
                - mean([row["counterfactual_mean"] for row in per_seed])
            )
            / mean([row["original_mean"] for row in per_seed]),
            "counterfactual_vs_original_p95_percent": 100.0
            * (
                mean([row["original_p95"] for row in per_seed])
                - mean([row["counterfactual_p95"] for row in per_seed])
            )
            / mean([row["original_p95"] for row in per_seed]),
            "counterfactual_vs_central_mean_percent": 100.0
            * (
                mean([row["central_mean"] for row in per_seed])
                - mean([row["counterfactual_mean"] for row in per_seed])
            )
            / mean([row["central_mean"] for row in per_seed]),
            "counterfactual_vs_central_p95_percent": 100.0
            * (
                mean([row["central_p95"] for row in per_seed])
                - mean([row["counterfactual_p95"] for row in per_seed])
            )
            / mean([row["central_p95"] for row in per_seed]),
        }
    result = {
        "status": "complete",
        "diagnostic_only": True,
        "mode": mode,
        "network_weights_unchanged": True,
        "history_only": True,
        "future_workload_used": False,
        "profiles": profile_results,
    }
    write_json(
        output_root / "coverage_counterfactual_summary.json",
        result,
    )
    lines = [
        f"# 覆盖优先缓存反事实诊断（{mode}）",
        "",
        "> 仅复用冻结网络和历史统计，不重训，不作为正式算法结果。",
        "",
        "| 环境 | CF vs 原OUR均值 | CF vs 原OUR P95 | CF vs Central均值 | CF vs Central P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        value = profile_results[profile]
        lines.append(
            f"| {profile} | "
            f"{value['counterfactual_vs_original_mean_percent']:.3f}% | "
            f"{value['counterfactual_vs_original_p95_percent']:.3f}% | "
            f"{value['counterfactual_vs_central_mean_percent']:.3f}% | "
            f"{value['counterfactual_vs_central_p95_percent']:.3f}% |"
        )
    (output_root / "COVERAGE_COUNTERFACTUAL_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for profile in args.profiles:
        for seed in args.seeds:
            evaluate_one(
                suite_root=args.suite_root,
                output_root=args.output_dir,
                profile=profile,
                seed=seed,
                episodes=args.episodes,
                mode=args.mode,
                resume=args.resume,
            )
    aggregate(
        args.suite_root,
        args.output_dir,
        args.profiles,
        args.seeds,
        args.mode,
    )
    print(f"Coverage counterfactual: {args.output_dir}")


if __name__ == "__main__":
    main()
