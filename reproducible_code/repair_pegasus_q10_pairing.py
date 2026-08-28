#!/usr/bin/env python3
"""Repair Q=10 scenario ordering without retraining any policy.

The original Q=10 runs contain the same balanced Pegasus families as the
Q=4/6/8 runs, but their family order differs.  This script freezes the
already-audited Q=10 checkpoints and evaluates them on the exact scenario
ordering used by Q=4/6/8.  Algorithm and training artifacts are untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from evaluate_pegasus_service_size_sensitivity import (
    canonical_deployment,
    validate_source,
)
from pegasus_service_sensitivity_protocol import (
    ACTIVE_SERVICE_COUNTS,
    BANDWIDTH_HZ,
    CAPACITY_MULTISET,
    DAOC,
    EVALUATION_EPISODES,
    FAMILIES,
    LEARNING_METHODS,
    METHODS,
    PROTOCOL_VERSION as PARENT_PROTOCOL_VERSION,
    RESULT_ROOT,
    SEEDS,
    SERVICE_STATE_DIMENSION,
    active_service_run,
    q10_source_run,
)
from run_independent_experiment import (
    EPISODE_FIELDS,
    run_scenario_bank_evaluation,
    summarize_rows,
)


PROTOCOL_VERSION = f"{PARENT_PROTOCOL_VERSION}_q10_pairfix_v1"
PAIR_ROOT = RESULT_ROOT / "active_services/q10_paired"
AUDIT_PATH = RESULT_ROOT / "ACTIVE_SERVICE_AUDIT.json"
PRE_FIX_AUDIT_PATH = RESULT_ROOT / "ACTIVE_SERVICE_AUDIT_PRE_PAIRFIX.json"
DIAGNOSTIC_PATH = RESULT_ROOT / "Q10_PAIRING_DIAGNOSTIC.json"


def parse_csv(value, converter):
    return tuple(
        converter(item.strip()) for item in value.split(",") if item.strip()
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("evaluate", "audit", "all"), default="all"
    )
    parser.add_argument("--labels", default=",".join(METHODS))
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in SEEDS)
    )
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.labels = parse_csv(args.labels, str)
    args.seeds = parse_csv(args.seeds, int)
    if not set(args.labels).issubset(METHODS):
        raise ValueError("Unknown method label")
    if not set(args.seeds).issubset(SEEDS):
        raise ValueError("Unknown seed")
    if args.episodes < 1 or args.workers < 1:
        raise ValueError("Episodes and workers must be positive")
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


def paired_q10_run(method: str, seed: int) -> Path:
    return PAIR_ROOT / "runs" / method / f"seed_{seed}"


def result_run(active_services: int, method: str, seed: int) -> Path:
    if active_services == SERVICE_STATE_DIMENSION:
        return paired_q10_run(method, seed)
    return active_service_run(active_services, method, seed)


def source_run(active_services: int, method: str, seed: int) -> Path:
    if active_services == SERVICE_STATE_DIMENSION:
        return q10_source_run(method, seed)
    return active_service_run(active_services, method, seed)


def is_complete(path: Path, episodes: int) -> bool:
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "complete"
        and summary.get("protocol_version") == PROTOCOL_VERSION
        and summary.get("evaluation_scenario_count") == episodes
        and summary.get("evaluation_state_frozen") is True
        and summary.get("native_cache_updates") is False
        and summary.get("target_retraining") is False
    )


def evaluate_one(spec):
    method, seed, episodes, resume = spec
    torch.set_num_threads(1)
    target = paired_q10_run(method, seed)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    if resume and is_complete(summary_path, episodes):
        return {"method": method, "seed": seed, "skipped": True}

    source, source_summary, config = validate_source(method, seed)
    reference_config = read_json(q10_source_run(DAOC, seed) / "config.json")
    source_offset = int(config["arguments"]["eval_seed_offset"])
    reference_offset = int(reference_config["arguments"]["eval_seed_offset"])
    if source_offset != reference_offset:
        raise RuntimeError(f"Evaluation seed offset mismatch: {source}")

    checkpoint_path = source / "selected_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    input_config = copy.deepcopy(config["input_config"])
    input_config.update({"seed": seed, "save topology figure": False})
    learning_config = copy.deepcopy(config["learning_config"])
    deployment = canonical_deployment(seed, 1.0, target)
    eval_args = SimpleNamespace(
        eval_episodes=episodes,
        eval_seed_offset=reference_offset,
        seed=seed,
        eval_bank_scope="infrastructure",
        eval_dag_families=list(FAMILIES),
        eval_update_caching=False,
        label=method,
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
                phase="eval",
            )

    exact_once = all(
        int(row["real_task_count"]) == int(row["completed_task_count"])
        and int(row["all_tasks_executed_once"]) == 1
        for row in rows
    )
    if not exact_once:
        raise RuntimeError(f"Task exact-once audit failed: {target}")
    write_json(target / "evaluation_scenarios.json", scenarios)
    paired_config = copy.deepcopy(config)
    paired_config["q10_pairfix"] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_run": str(source.resolve()),
        "target_retraining": False,
        "native_cache_updates": False,
    }
    write_json(target / "config.json", paired_config)
    write_json(
        summary_path,
        {
            "status": "complete",
            "protocol_version": PROTOCOL_VERSION,
            "label": method,
            "seed": seed,
            "evaluation_scenario_count": episodes,
            "source_run": str(source.resolve()),
            "source_selected_checkpoint_sha256": source_summary.get(
                "selected_checkpoint_sha256"
            ),
            "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
            "selected_checkpoint_episode": int(checkpoint["episode"]),
            "evaluation_state_frozen": True,
            "native_cache_updates": False,
            "cross_scenario_state_reset": True,
            "target_retraining": False,
            "future_workload_visible": False,
            "tasks_exact_once": True,
            "eligible_for_comparison": source_summary.get(
                "eligible_for_comparison"
            ),
            "convergence": source_summary.get("convergence", {}),
            "total_server_capacity": source_summary.get(
                "total_server_capacity"
            ),
            "eval": summarize_rows(rows),
        },
    )
    return {"method": method, "seed": seed, "skipped": False}


def comparable_scenarios(path: Path):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "workflow_family": row["workflow_family"],
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in read_json(path)
    ]


def evaluation_rows(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["phase"] == "eval"]


def audit(episodes: int) -> dict:
    original_audit_sha256 = (
        sha256_file(AUDIT_PATH) if AUDIT_PATH.is_file() else None
    )
    report = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocol_version": PARENT_PROTOCOL_VERSION,
        "q10_pairfix_applied": True,
        "q10_retrained": False,
        "q10_native_cache_updates": False,
        "repair_reason": (
            "Legacy Q=10 runs used a different balanced workflow-family "
            "ordering; frozen checkpoints were re-evaluated on the exact "
            "Q=4/6/8 scenario ordering."
        ),
        "pre_fix_audit_sha256": original_audit_sha256,
        "all_runs_complete": True,
        "all_learning_methods_converged": True,
        "all_tasks_exactly_once": True,
        "all_capacity_constraints_valid": True,
        "all_methods_scenario_paired": True,
        "all_service_counts_cross_paired": True,
        "records": [],
    }
    cross_count_banks = {}
    for active_services in ACTIVE_SERVICE_COUNTS:
        for seed in SEEDS:
            method_reference = None
            for method in METHODS:
                run = result_run(active_services, method, seed)
                source = source_run(active_services, method, seed)
                summary = read_json(run / "summary.json")
                source_summary = read_json(source / "summary.json")
                config = read_json(source / "config.json")["arguments"]
                scenarios = comparable_scenarios(
                    run / "evaluation_scenarios.json"
                )
                scenario_count = summary.get(
                    "evaluation_scenario_count",
                    summary.get("eval_episodes"),
                )
                complete = (
                    summary.get("status") == "complete"
                    and scenario_count == episodes
                )
                converged = method not in LEARNING_METHODS or bool(
                    source_summary.get("eligible_for_comparison")
                    and source_summary.get("convergence", {}).get("reached")
                )
                exact_once = all(
                    row.get("all_tasks_executed_once") == "1"
                    for row in evaluation_rows(run / "episodes.csv")
                )
                capacity_valid = (
                    sorted(config.get("server_capacity_multiset", []))
                    == sorted(CAPACITY_MULTISET)
                    and source_summary.get("total_server_capacity")
                    == sum(CAPACITY_MULTISET)
                    and config.get("num_services")
                    == SERVICE_STATE_DIMENSION
                    and float(config.get("bandwidth")) == BANDWIDTH_HZ
                )
                paired = (
                    method_reference is None or scenarios == method_reference
                )
                if method_reference is None:
                    method_reference = scenarios
                cross_paired = (
                    seed not in cross_count_banks
                    or scenarios == cross_count_banks[seed]
                )
                cross_count_banks.setdefault(seed, scenarios)
                report["all_runs_complete"] &= complete
                report["all_learning_methods_converged"] &= converged
                report["all_tasks_exactly_once"] &= exact_once
                report["all_capacity_constraints_valid"] &= capacity_valid
                report["all_methods_scenario_paired"] &= paired
                report["all_service_counts_cross_paired"] &= cross_paired
                report["records"].append(
                    {
                        "active_services": active_services,
                        "seed": seed,
                        "method": method,
                        "result_run": str(run.resolve()),
                        "source_run": str(source.resolve()),
                        "complete": complete,
                        "converged": converged,
                        "exact_once": exact_once,
                        "capacity_valid": capacity_valid,
                        "method_paired": paired,
                        "cross_count_paired": cross_paired,
                        "selected_checkpoint_episode": source_summary.get(
                            "selected_checkpoint_episode"
                        ),
                    }
                )
    required = (
        "all_runs_complete",
        "all_learning_methods_converged",
        "all_tasks_exactly_once",
        "all_capacity_constraints_valid",
        "all_methods_scenario_paired",
        "all_service_counts_cross_paired",
    )
    if not all(report[key] for key in required):
        report["status"] = "failed"
    pairfix_audit = RESULT_ROOT / "ACTIVE_SERVICE_AUDIT_PAIRFIX.json"
    write_json(pairfix_audit, report)
    if report["status"] != "complete":
        raise RuntimeError("Q=10 pair-fix audit failed")
    if AUDIT_PATH.is_file() and not PRE_FIX_AUDIT_PATH.exists():
        shutil.copy2(AUDIT_PATH, PRE_FIX_AUDIT_PATH)
    write_json(AUDIT_PATH, report)
    write_json(
        DIAGNOSTIC_PATH,
        {
            "status": "resolved",
            "protocol_version": PROTOCOL_VERSION,
            "algorithm_changed": False,
            "training_changed": False,
            "checkpoint_changed": False,
            "evaluation_only": True,
            "original_audit": str(PRE_FIX_AUDIT_PATH.resolve()),
            "repaired_audit": str(pairfix_audit.resolve()),
            "reason": report["repair_reason"],
        },
    )
    return report


def main():
    args = parse_args()
    if args.stage in ("evaluate", "all"):
        specs = [
            (method, seed, args.episodes, args.resume)
            for seed in args.seeds
            for method in args.labels
        ]
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(specs))
        ) as pool:
            futures = [pool.submit(evaluate_one, spec) for spec in specs]
            for future in as_completed(futures):
                result = future.result()
                state = "skip" if result["skipped"] else "complete"
                print(
                    f"Q=10 paired {result['method']} "
                    f"seed={result['seed']}: {state}",
                    flush=True,
                )
    if args.stage in ("audit", "all"):
        audit(args.episodes)


if __name__ == "__main__":
    main()
