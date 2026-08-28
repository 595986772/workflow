#!/usr/bin/env python3
"""Evaluate frozen Pegasus-B8 policies on Alibaba-CP100 DAGs.

Only the DAG source changes. Infrastructure, cache budget, checkpoints, and
all observable-information rules remain those of the frozen Pegasus-B8 main
experiment, so the resulting point can be placed beside the five Pegasus
workflow families without confounding cache budget with dataset identity.
"""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_independent_experiment import (
    EPISODE_FIELDS,
    run_scenario_bank_evaluation,
    summarize_rows,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    ROOT
    / "results"
    / "pegasus_pscale"
    / "p10_alibaba_cp100_cross_dataset"
)
DATASET_PATH = (
    ROOT / "datasets" / "alibaba_cp100" / "dag_alibaba_cp100.json"
)
EXPECTED_DATASET_SHA256 = (
    "2903ff2478f5c55fe445bd2a7b6fbe595aecf6ea6383913b85ea5efd92ee2d89"
)
PROTOCOL_VERSION = "pegasus_b8_frozen_alibaba_cp100_v1"
SEEDS = tuple(range(51, 61))
METHODS = (
    "random",
    "nearest",
    "greedy",
    "dqn_wdsa_std_cache",
    "discrete_sac_std_cache",
    "daoc_paper",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.episodes < 1:
        raise ValueError("workers and episodes must be positive")
    return args


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_run(method: str, seed: int) -> Path:
    if method in {"random", "nearest", "greedy", "dqn_wdsa_std_cache"}:
        stage = "heuristics" if method in {"random", "nearest", "greedy"} else "learning"
        return (
            ROOT
            / "results/pegasus_pscale/p6_baselines_ablation"
            / stage
            / "runs"
            / method
            / f"seed_{seed}"
        )
    if method == "discrete_sac_std_cache":
        return (
            ROOT
            / "results/pegasus_pscale/p7_std_cache_discrete_sac/final/runs"
            / method
            / f"seed_{seed}"
        )
    if method == "daoc_our_coord_cache":
        return (
            ROOT
            / "results/pegasus_pscale/p8_daoc_our_coord_cache/final/runs"
            / method
            / f"seed_{seed}"
        )
    if method == "coord_cache_discrete_sac":
        return (
            ROOT
            / "results/pegasus_pscale/p5_baseline_extension/sac_final/runs"
            / method
            / f"seed_{seed}"
        )
    return (
        ROOT
        / "results/pegasus_pscale/p3_paper_closure/final/runs"
        / method
        / f"seed_{seed}"
    )


def deployment_from_snapshot(snapshot, capacities):
    return {
        "servers": {
            int(server_id): {
                "position": tuple(server["position"]),
                "frequency": float(server["frequency"]),
                "load": float(server["load"]),
                "rate_to_cloud": float(server["rate_to_cloud"]),
                "capacity": int(capacities[int(server_id)]),
            }
            for server_id, server in snapshot["servers"].items()
        },
        "users": {
            int(user_id): {
                "position": tuple(user["initial_position"]),
                "direction": user["direction"],
            }
            for user_id, user in snapshot["users"].items()
        },
        "service_data_length": {
            int(service_id): float(length)
            for service_id, length in snapshot["service_data_length"].items()
        },
        "between_server_costs": np.asarray(
            snapshot["between_server_costs"], dtype=float
        ),
    }


def is_complete(path: Path, episodes: int, source_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "complete"
        and summary.get("protocol_version") == PROTOCOL_VERSION
        and summary.get("evaluation_scenario_count") == episodes
        and summary.get("source_checkpoint_sha256") == source_hash
        and summary.get("dataset_sha256") == EXPECTED_DATASET_SHA256
    )


def evaluate_one(spec):
    method, seed, episodes, resume = spec
    torch.set_num_threads(1)
    source = source_run(method, seed)
    checkpoint_path = source / "selected_checkpoint.pt"
    source_hash = sha256_file(checkpoint_path)
    output = OUTPUT_ROOT / "runs" / method / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if resume and is_complete(summary_path, episodes, source_hash):
        return method, seed, True

    config = read_json(source / "config.json")
    input_config = copy.deepcopy(config["input_config"])
    input_config.update(
        {
            "dag dataset path": str(DATASET_PATH.resolve()),
            "dag dataset sha256": EXPECTED_DATASET_SHA256,
            "save topology figure": False,
        }
    )
    input_config.pop("application graph family", None)
    learning_config = copy.deepcopy(config["learning_config"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    frozen_state = checkpoint["frozen_state"]
    capacities = {
        int(server_id): int(capacity)
        for server_id, capacity in frozen_state["capacities"].items()
    }
    deployment_state = deployment_from_snapshot(
        read_json(source / "scenario_initial.json"), capacities
    )
    source_arguments = config["arguments"]
    run_args = SimpleNamespace(
        eval_episodes=episodes,
        eval_seed_offset=int(source_arguments["eval_seed_offset"]),
        eval_dag_families=None,
        eval_bank_scope="infrastructure",
        seed=seed,
        label=method,
        quiet=True,
    )

    with (output / "simulator.log").open("w", encoding="utf-8") as log:
        with (output / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            rows, scenarios = run_scenario_bank_evaluation(
                writer=writer,
                csv_file=csv_file,
                args=run_args,
                input_config=input_config,
                learning_config=learning_config,
                frozen_state=frozen_state,
                deployment_state=deployment_state,
                simulator_log=log,
                figures_dir=output,
                episodes=episodes,
                phase="eval",
            )

    write_json(output / "evaluation_scenarios.json", scenarios)
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "seed": seed,
        "source_run": str(source.resolve()),
        "source_checkpoint_episode": int(checkpoint["episode"]),
        "source_checkpoint_sha256": source_hash,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dataset_role": "selected_cross_dataset_stress_set_not_unbiased_holdout",
        "evaluation_mode": "frozen_pegasus_b8_checkpoint_zero_shot",
        "evaluation_scenario_count": episodes,
        "server_capacities": capacities,
        "eval": summarize_rows(rows),
    }
    write_json(summary_path, summary)
    return method, seed, False


def aggregate(episodes: int) -> None:
    paired = True
    rows = []
    for seed in SEEDS:
        reference = None
        for method in METHODS:
            run = OUTPUT_ROOT / "runs" / method / f"seed_{seed}"
            summary = read_json(run / "summary.json")
            scenarios = read_json(run / "evaluation_scenarios.json")
            signature = [
                (
                    row["seed"],
                    row["base_fingerprint"],
                    row["user_graph_keys"],
                )
                for row in scenarios
            ]
            if reference is None:
                reference = signature
            else:
                paired &= signature == reference
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "mean_completion_time": summary["eval"][
                        "mean_average_finish_time"
                    ],
                    "mean_p95_completion_time": summary["eval"][
                        "mean_p95_finish_time"
                    ],
                }
            )
    if not paired:
        raise RuntimeError("Alibaba-CP100 scenario banks are not paired")

    with (OUTPUT_ROOT / "per_seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        OUTPUT_ROOT / "protocol_summary.json",
        {
            "status": "complete",
            "protocol_version": PROTOCOL_VERSION,
            "dataset_path": str(DATASET_PATH.resolve()),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "dataset_role": "selected_cross_dataset_stress_set_not_unbiased_holdout",
            "source_environment": "Pegasus-B8",
            "only_changed_factor": "DAG dataset",
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "evaluation_scenarios_per_seed": episodes,
            "paired_scenarios_across_methods": paired,
            "cache_budget": 8,
            "capacity_multiset": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
            "checkpoint_training_dataset": "Pegasus five-family P-Scale",
            "checkpoint_weights_frozen": True,
            "cache_and_history_frozen": True,
        },
    )


def main() -> None:
    args = parse_args()
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Alibaba-CP100 dataset hash mismatch")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [
        (method, seed, args.episodes, args.resume)
        for method in METHODS
        for seed in SEEDS
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_one, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            method, seed, skipped = future.result()
            state = "reused" if skipped else "complete"
            print(f"[{index:02d}/{len(jobs)}] {method} seed={seed}: {state}")
    aggregate(args.episodes)
    print(OUTPUT_ROOT / "protocol_summary.json")


if __name__ == "__main__":
    main()
