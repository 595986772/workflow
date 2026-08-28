#!/usr/bin/env python3
"""Run the governed Pegasus P-Scale three-seed experiment."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pegasus_pscale_protocol import (
    CACHE_CALIBRATION_EPISODES,
    CAPACITY_PROFILES,
    DATASET_PATH,
    DEVELOPMENT_SEEDS,
    EVALUATION_BANK_SCOPE,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    MANIFEST_PATH,
    PROTOCOL_VERSION,
    TASK_LIMIT_INCLUDING_DUMMY,
    VALIDATION_SCENARIOS,
    validate_protocol,
)
from run_a0_fixed_budget_heterogeneity import (
    ALGORITHM_SOURCE_FILES,
    source_hash,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results/pegasus_pscale/p2"
LOCK_PATH = RESULT_ROOT / "PSCALE_LOCK.json"
PARENT_LOCK = ROOT / "results/a0_fixed_budget_heterogeneity/h8v1/FINAL_LOCK.json"
P1_INVALIDATION = ROOT / "results/pegasus_pscale/p1/INVALIDATED_DIAGNOSTIC.json"
METHODS = ("guided_full", "centralized_greedy_daoc", "lean_our")
CAPACITY_MULTISET = CAPACITY_PROFILES["B8"]
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
SMOKE_SEEDS = (41,)
PROTOCOL_SOURCE_FILES = (
    "analyze_pegasus_pscale.py",
    "build_pegasus_pscale_dataset.py",
    "evaluate_oracle_latency_bound.py",
    "oracle_latency_bound.py",
    "pegasus_pscale_protocol.py",
    "run_independent_experiment.py",
    "run_reproduction_suite.py",
    "test_build_pegasus_pscale_dataset.py",
    "test_dag_completion_semantics.py",
    "test_pegasus_pscale_protocol.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "converged", "oracle", "analysis", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_source_hash():
    digest = hashlib.sha256()
    for filename in sorted(PROTOCOL_SOURCE_FILES):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT / filename).read_bytes())
    return digest.hexdigest()


def protocol_spec():
    parent = read_json(PARENT_LOCK)
    return {
        "protocol_id": PROTOCOL_VERSION,
        "parent_algorithm_revision": "h8v1",
        "parent_algorithm_source_sha256": parent[
            "algorithm_source_sha256"
        ],
        "current_algorithm_source_sha256": source_hash(
            ALGORITHM_SOURCE_FILES
        ),
        "algorithm_retuned": False,
        "changed_environment_modules": [
            "all-real-task termination",
            "max-exit application completion time",
            "complete Pegasus workflows",
            "task limit 31",
            "paired user-ingress resampling",
            "fixed 5000-episode cache calibration horizon",
        ],
        "dag_completion_protocol_version": (
            DAG_COMPLETION_PROTOCOL_VERSION
        ),
        "dataset": {
            "path": str(DATASET_PATH.resolve()),
            "sha256": EXPECTED_DATASET_SHA256,
            "manifest": str(MANIFEST_PATH.resolve()),
            "families": list(FAMILIES),
            "real_tasks": [25, 30, 24, 30, 29],
            "program_types": 39,
            "services": 10,
            "unbiased_mec_holdout": False,
        },
        "environment": {
            "users": 20,
            "servers": 10,
            "services": 10,
            "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
            "bandwidth_hz": 15000,
            "capacity_multiset": list(CAPACITY_MULTISET),
            "total_cache_budget": 8,
            "capacity_assignment_namespace": CAPACITY_NAMESPACE,
        },
        "methods": list(METHODS),
        "training": {
            "from_scratch": True,
            "seeds": list(DEVELOPMENT_SEEDS),
            "convergence_profile": "pegasus_pscale_p2_converged",
            "validation_scenarios": VALIDATION_SCENARIOS,
            "cache_calibration_episodes": CACHE_CALIBRATION_EPISODES,
            "no_checkpoint_reuse": True,
        },
        "evaluation": {
            "bank_scope": EVALUATION_BANK_SCOPE,
            "paired_scenarios_per_seed": EVALUATION_EPISODES,
            "family_order": list(FAMILIES),
            "scenarios_per_family": 20,
            "frozen_networks": True,
            "frozen_caches_after_calibration": True,
            "uniqueness_key": "base_scenario_fingerprint",
        },
        "gate": {
            "integrity": "all_checks_true",
            "our_vs_daoc": "positive_mean_and_at_least_2_of_3_wins",
            "our_vs_central": "positive_mean_and_at_least_2_of_3_wins",
            "p95_vs_central": "positive_mean_and_at_least_2_of_3_wins",
        },
        "next_stage": (
            "Run B5/B8/B10 sensitivity only if the three-seed gate passes; "
            "then decide whether to freeze a ten-seed confirmation."
        ),
    }


def initialize_lock():
    validate_protocol()
    specification = protocol_spec()
    lock = {
        "status": "locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "protocol_source_sha256": protocol_source_hash(),
        "parent_lock_path": str(PARENT_LOCK.resolve()),
        "parent_lock_sha256": sha256_file(PARENT_LOCK),
        "invalidated_p1_path": str(P1_INVALIDATION.resolve()),
        "invalidated_p1_sha256": sha256_file(P1_INVALIDATION),
        "algorithm_retuned": False,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        if existing.get("specification_sha256") != lock[
            "specification_sha256"
        ]:
            raise RuntimeError("Existing P-Scale lock specification mismatch")
        if existing.get("protocol_source_sha256") != lock[
            "protocol_source_sha256"
        ]:
            raise RuntimeError("P-Scale protocol source changed after lock")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(value) for value in command), flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def reproduction_command(profile, suite_dir, seeds, workers, resume):
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        profile,
        "--suite-dir",
        str(suite_dir),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--labels",
        ",".join(METHODS),
        "--dag-dataset-path",
        str(DATASET_PATH),
        "--dag-dataset-sha256",
        EXPECTED_DATASET_SHA256,
        "--num-tasks",
        str(TASK_LIMIT_INCLUDING_DUMMY),
        "--eval-dag-families",
        ",".join(FAMILIES),
        "--server-capacity",
        "1",
        "--server-capacity-multiset",
        ",".join(str(value) for value in CAPACITY_MULTISET),
        "--baseline-server-capacity",
        "3",
        "--capacity-assignment-namespace",
        CAPACITY_NAMESPACE,
        "--workers",
        str(workers),
        "--revision-id",
        PROTOCOL_VERSION,
        "--revision-parent",
        "h8v1",
        "--revision-reason",
        "correct_multi_exit_semantics_and_test_full_pegasus_scale",
        "--revision-changed-module",
        "environment_completion_semantics_and_workflow_size",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_our_not_better_than_both_learning_baselines",
        "--seed-partition",
        "generalization",
    ]
    if resume:
        command.append("--resume")
    return command


def evaluation_csv_rows(run_dir):
    with (run_dir / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        return [
            row for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def check_runs(suite_dir, seeds, expected_episodes, require_convergence):
    for seed in seeds:
        reference_bank = None
        reference_capacities = None
        for label in METHODS:
            directory = suite_dir / "runs" / label / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {label} seed={seed}")
            if summary.get("dag_completion_protocol_version") != (
                DAG_COMPLETION_PROTOCOL_VERSION
            ):
                raise RuntimeError("Wrong DAG completion protocol")
            if require_convergence and not (
                summary.get("eligible_for_comparison")
                and summary.get("convergence", {}).get("reached")
            ):
                raise RuntimeError(
                    f"Unconverged run: {label} seed={seed}"
                )
            if summary.get("dag_dataset", {}).get("sha256") != (
                EXPECTED_DATASET_SHA256
            ):
                raise RuntimeError("P-Scale dataset mismatch")
            if (
                summary.get("evaluation_scenario_count") != expected_episodes
                or summary.get("evaluation_unique_base_scenarios")
                != expected_episodes
            ):
                raise RuntimeError("P-Scale effective scenario count mismatch")
            capacities = {
                int(server_id): int(value)
                for server_id, value
                in summary["server_capacities"].items()
            }
            if sorted(capacities.values()) != sorted(CAPACITY_MULTISET):
                raise RuntimeError("P-Scale capacity mismatch")
            if reference_capacities is None:
                reference_capacities = capacities
            elif capacities != reference_capacities:
                raise RuntimeError("P-Scale capacity assignments are not paired")
            rows = evaluation_csv_rows(directory)
            if len(rows) != expected_episodes or not all(
                int(row["real_task_count"])
                == int(row["completed_task_count"])
                and int(row["all_tasks_executed_once"]) == 1
                for row in rows
            ):
                raise RuntimeError("P-Scale task execution audit failed")
            bank = read_json(directory / "evaluation_scenarios.json")
            if len(bank) != expected_episodes:
                raise RuntimeError("P-Scale scenario bank length mismatch")
            family_counts = Counter(
                row.get("workflow_family") for row in bank
            )
            expected_per_family = expected_episodes // len(FAMILIES)
            if family_counts != Counter(
                {family: expected_per_family for family in FAMILIES}
            ):
                raise RuntimeError("P-Scale family balance mismatch")
            comparable_bank = [
                {
                    "episode": row["episode"],
                    "seed": row["seed"],
                    "base_fingerprint": row["base_fingerprint"],
                    "workflow_family": row.get("workflow_family"),
                    "user_initial_positions": row["user_initial_positions"],
                    "user_graph_keys": row["user_graph_keys"],
                }
                for row in bank
            ]
            if reference_bank is None:
                reference_bank = comparable_bank
            elif comparable_bank != reference_bank:
                raise RuntimeError("P-Scale scenario banks are not paired")


def run_tests():
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "test_dag_completion_semantics.py",
        "test_build_pegasus_pscale_dataset.py",
        "test_pegasus_pscale_protocol.py",
        "test_oracle_latency_bound.py",
        "test_capacity_protocol.py",
        "test_a0_fixed_budget_heterogeneity_final.py",
    ]
    run_logged(command, RESULT_ROOT / "tests.log")


def run_oracle(suite_dir):
    command = [
        sys.executable,
        str(ROOT / "evaluate_oracle_latency_bound.py"),
        "--our-suite-dir",
        str(suite_dir),
        "--our-label",
        "lean_our",
        "--daoc-suite-dir",
        str(suite_dir),
        "--daoc-label",
        "guided_full",
        "--output-dir",
        str(suite_dir / "oracle"),
        "--seeds",
        ",".join(str(seed) for seed in DEVELOPMENT_SEEDS),
        "--episodes",
        str(EVALUATION_EPISODES),
        "--exact-check-scenarios",
        "0",
    ]
    run_logged(command, suite_dir / "oracle.log")


def run_analysis(suite_dir):
    command = [
        sys.executable,
        str(ROOT / "analyze_pegasus_pscale.py"),
        "--suite-dir",
        str(suite_dir),
        "--output-dir",
        str(suite_dir / "analysis"),
    ]
    run_logged(command, suite_dir / "analysis.log")


def main():
    args = parse_args()
    if args.stage in ("tests", "all"):
        run_tests()
        if args.stage == "tests":
            print(f"P-Scale artifacts: {RESULT_ROOT}")
            return

    initialize_lock()
    smoke_dir = RESULT_ROOT / "smoke"
    converged_dir = RESULT_ROOT / "converged"
    stages = (
        (
            "smoke",
            lambda: (
                run_logged(
                    reproduction_command(
                        "pegasus_pscale_p2_smoke",
                        smoke_dir,
                        SMOKE_SEEDS,
                        args.workers,
                        args.resume,
                    ),
                    smoke_dir / "runner.log",
                ),
                check_runs(smoke_dir, SMOKE_SEEDS, 20, False),
            ),
        ),
        (
            "converged",
            lambda: (
                run_logged(
                    reproduction_command(
                        "pegasus_pscale_p2_converged",
                        converged_dir,
                        DEVELOPMENT_SEEDS,
                        args.workers,
                        args.resume,
                    ),
                    converged_dir / "runner.log",
                ),
                check_runs(
                    converged_dir,
                    DEVELOPMENT_SEEDS,
                    EVALUATION_EPISODES,
                    True,
                ),
            ),
        ),
        ("oracle", lambda: run_oracle(converged_dir)),
        ("analysis", lambda: run_analysis(converged_dir)),
    )
    for stage_name, function in stages:
        if args.stage not in ("all", stage_name):
            continue
        function()

    if args.stage in ("all", "analysis"):
        summary = read_json(
            converged_dir / "analysis/pegasus_pscale_summary.json"
        )
        lock = read_json(LOCK_PATH)
        lock["status"] = "main_complete"
        lock["completed_at"] = datetime.now(timezone.utc).isoformat()
        lock["three_seed_gate"] = summary["gate"]
        lock["algorithm_retuned"] = False
        lock["next_stage_allowed"] = bool(summary["gate"]["passed"])
        write_json(LOCK_PATH, lock)
    print(f"P-Scale artifacts: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
