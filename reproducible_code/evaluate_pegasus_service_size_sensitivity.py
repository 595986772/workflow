#!/usr/bin/env python3
"""Evaluate frozen Q=10 policies under scaled service image sizes."""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from pegasus_service_sensitivity_protocol import (
    DAOC,
    EVALUATION_EPISODES,
    FAMILIES,
    METHODS,
    PROTOCOL_VERSION as PARENT_PROTOCOL_VERSION,
    SEEDS,
    SERVICE_SIZE_MULTIPLIERS,
    q10_source_run,
)
from run_independent_experiment import (
    EPISODE_FIELDS,
    capture_deployment_state,
    run_scenario_bank_evaluation,
    seed_everything,
    summarize_rows,
)
from simulator import MEC_Simulator


PROTOCOL_VERSION = f"{PARENT_PROTOCOL_VERSION}_service_size_v1"


def parse_csv(value, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        default=",".join(METHODS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
    )
    parser.add_argument(
        "--multipliers",
        default=",".join(str(value) for value in SERVICE_SIZE_MULTIPLIERS),
    )
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.labels = tuple(parse_csv(args.labels, str))
    args.seeds = tuple(parse_csv(args.seeds, int))
    args.multipliers = tuple(parse_csv(args.multipliers, float))
    if not set(args.labels).issubset(METHODS):
        raise ValueError("Unknown method label")
    if args.episodes < 1 or args.workers < 1:
        raise ValueError("Episodes and workers must be positive")
    if any(value <= 0 for value in args.multipliers):
        raise ValueError("Service size multipliers must be positive")
    return args


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def multiplier_token(multiplier: float) -> str:
    return f"x{multiplier:g}".replace(".", "p")


def target_run(output_dir: Path, multiplier: float, label: str, seed: int) -> Path:
    return (
        Path(output_dir)
        / multiplier_token(multiplier)
        / "runs"
        / label
        / f"seed_{seed}"
    )


def validate_source(label: str, seed: int) -> tuple[Path, dict, dict]:
    run = q10_source_run(label, seed)
    summary = read_json(run / "summary.json")
    config = read_json(run / "config.json")
    if not (
        summary.get("status") == "complete"
        and summary.get("evaluation_state_frozen") is True
        and (run / "selected_checkpoint.pt").is_file()
    ):
        raise RuntimeError(f"Invalid frozen source run: {run}")
    return run, summary, config


def canonical_deployment(seed: int, multiplier: float, output_dir: Path) -> dict:
    source, _, config = validate_source(DAOC, seed)
    input_config = copy.deepcopy(config["input_config"])
    input_config.update({"seed": seed, "save topology figure": False})
    seed_everything(seed)
    simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=input_config,
        learning_arguments=copy.deepcopy(config["learning_config"]),
        filename_png=str(output_dir),
    )
    deployment = capture_deployment_state(simulator)
    deployment["service_data_length"] = {
        service_id: (
            0.0 if int(service_id) == 0 else float(length) * multiplier
        )
        for service_id, length in deployment["service_data_length"].items()
    }
    return deployment


def is_complete(path: Path, multiplier: float, episodes: int) -> bool:
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "complete"
        and summary.get("protocol_version") == PROTOCOL_VERSION
        and summary.get("service_size_multiplier") == multiplier
        and summary.get("eval_episodes") == episodes
        and summary.get("model_frozen") is True
        and summary.get("native_cache_updates") is True
        and summary.get("cross_scenario_state_reset") is True
    )


def evaluate_one(spec):
    output_dir, label, seed, multiplier, episodes, resume = spec
    torch.set_num_threads(1)
    target = target_run(output_dir, multiplier, label, seed)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    if resume and is_complete(summary_path, multiplier, episodes):
        return {
            "label": label,
            "seed": seed,
            "multiplier": multiplier,
            "skipped": True,
        }

    source, source_summary, config = validate_source(label, seed)
    reference_config = read_json(q10_source_run(DAOC, seed) / "config.json")
    source_offset = int(config["arguments"]["eval_seed_offset"])
    reference_offset = int(reference_config["arguments"]["eval_seed_offset"])
    if source_offset != reference_offset:
        raise RuntimeError(f"Evaluation seed offset mismatch: {source}")

    checkpoint_path = source / "selected_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    input_config = copy.deepcopy(config["input_config"])
    input_config.update({"seed": seed, "save topology figure": False})
    learning_config = copy.deepcopy(config["learning_config"])
    deployment = canonical_deployment(seed, multiplier, target)
    eval_args = SimpleNamespace(
        eval_episodes=episodes,
        eval_seed_offset=reference_offset,
        seed=seed,
        eval_bank_scope="infrastructure",
        eval_dag_families=list(FAMILIES),
        eval_update_caching=True,
        label=label,
        quiet=True,
    )

    episodes_path = target / "episodes.csv"
    with (target / "simulator.log").open("w", encoding="utf-8") as log:
        with episodes_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            rows, scenarios = run_scenario_bank_evaluation(
                writer=writer,
                csv_file=output,
                args=eval_args,
                input_config=input_config,
                learning_config=learning_config,
                frozen_state=checkpoint["frozen_state"],
                deployment_state=deployment,
                simulator_log=log,
                figures_dir=target,
                episodes=episodes,
                phase="service_size_eval",
            )

    write_json(target / "evaluation_scenarios.json", scenarios)
    exact_once = all(
        int(row["real_task_count"]) == int(row["completed_task_count"])
        and int(row["all_tasks_executed_once"]) == 1
        for row in rows
    )
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "label": label,
        "seed": seed,
        "service_size_multiplier": multiplier,
        "eval_episodes": episodes,
        "source_run": str(source.resolve()),
        "source_selected_checkpoint_sha256": source_summary.get(
            "selected_checkpoint_sha256"
        ),
        "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
        "source_checkpoint_episode": int(checkpoint["episode"]),
        "model_frozen": True,
        "native_cache_updates": True,
        "cross_scenario_state_reset": True,
        "target_retraining": False,
        "future_workload_visible": False,
        "tasks_exact_once": exact_once,
        "eval": summarize_rows(rows),
    }
    if not exact_once:
        raise RuntimeError(f"Task exact-once audit failed: {target}")
    write_json(summary_path, summary)
    return {
        "label": label,
        "seed": seed,
        "multiplier": multiplier,
        "skipped": False,
    }


def comparable_scenarios(path: Path):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "workflow_family": row["workflow_family"],
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in read_json(path)
    ]


def audit(output_dir: Path, labels, seeds, multipliers, episodes) -> dict:
    report = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "all_runs_complete": True,
        "all_tasks_exactly_once": True,
        "all_scenarios_paired": True,
        "all_networks_frozen": True,
        "all_native_cache_updates_enabled": True,
        "records": [],
    }
    for multiplier in multipliers:
        for seed in seeds:
            reference = None
            for label in labels:
                run = target_run(output_dir, multiplier, label, seed)
                summary = read_json(run / "summary.json")
                bank = comparable_scenarios(run / "evaluation_scenarios.json")
                checks = {
                    "complete": (
                        summary.get("status") == "complete"
                        and summary.get("eval_episodes") == episodes
                    ),
                    "exact_once": summary.get("tasks_exact_once") is True,
                    "model_frozen": summary.get("model_frozen") is True,
                    "cache_updates": summary.get("native_cache_updates") is True,
                    "paired": reference is None or bank == reference,
                }
                reference = bank if reference is None else reference
                report["all_runs_complete"] &= checks["complete"]
                report["all_tasks_exactly_once"] &= checks["exact_once"]
                report["all_scenarios_paired"] &= checks["paired"]
                report["all_networks_frozen"] &= checks["model_frozen"]
                report["all_native_cache_updates_enabled"] &= checks[
                    "cache_updates"
                ]
                report["records"].append(
                    {
                        "multiplier": multiplier,
                        "seed": seed,
                        "label": label,
                        **checks,
                    }
                )
    required = (
        "all_runs_complete",
        "all_tasks_exactly_once",
        "all_scenarios_paired",
        "all_networks_frozen",
        "all_native_cache_updates_enabled",
    )
    if not all(report[key] for key in required):
        report["status"] = "failed"
    write_json(Path(output_dir) / "AUDIT.json", report)
    if report["status"] != "complete":
        raise RuntimeError("Service-size sensitivity audit failed")
    return report


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            args.output_dir,
            label,
            seed,
            multiplier,
            args.episodes,
            args.resume,
        )
        for multiplier in args.multipliers
        for seed in args.seeds
        for label in args.labels
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(specs))) as pool:
        futures = [pool.submit(evaluate_one, spec) for spec in specs]
        for future in as_completed(futures):
            result = future.result()
            state = "skip" if result["skipped"] else "complete"
            print(
                f"size={result['multiplier']:g} {result['label']} "
                f"seed={result['seed']}: {state}",
                flush=True,
            )
    audit(
        args.output_dir,
        args.labels,
        args.seeds,
        args.multipliers,
        args.episodes,
    )


if __name__ == "__main__":
    main()

