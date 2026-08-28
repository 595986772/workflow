#!/usr/bin/env python3
"""Run the one-time frozen h8v1 A0 confirmation experiment."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from a0_coordination_protocol import DATASET_PATH, EXPECTED_DATASET_SHA256
from a0_fixed_budget_heterogeneity_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    METHOD_LABELS,
    PROFILE_ORDER,
    TOTAL_CACHE_BUDGET,
    capacity_text,
    validate_protocol,
)
from run_a0_fixed_budget_heterogeneity import (
    ALGORITHM_SOURCE_FILES,
    ROOT_DIR,
    canonical_hash,
    file_sha256,
    read_json,
    source_hash,
    write_json,
)


REVISION_ID = "h8v1"
FINAL_SEEDS = tuple(range(11, 21))
REVISION_ROOT = (
    ROOT_DIR / "results" / "a0_fixed_budget_heterogeneity" / REVISION_ID
)
FREEZE_PATH = REVISION_ROOT / "FROZEN_ALGORITHM.json"
LOCK_PATH = REVISION_ROOT / "FINAL_LOCK.json"
FINAL_ROOT = REVISION_ROOT / "final"
FINAL_PROTOCOL_FILES = ALGORITHM_SOURCE_FILES + (
    "a0_coordination_protocol.py",
    "a0_fixed_budget_heterogeneity_protocol.py",
    "analyze_a0_coordination.py",
    "analyze_a0_fixed_budget_heterogeneity_final.py",
    "evaluate_oracle_latency_bound.py",
    "oracle_latency_bound.py",
    "run_a0_fixed_budget_heterogeneity_final.py",
    "run_independent_experiment.py",
    "run_reproduction_suite.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("lock", "train", "oracle", "analysis", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def final_specification():
    return {
        "revision_id": REVISION_ID,
        "algorithm_selection_complete": True,
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": EXPECTED_DATASET_SHA256,
            "unbiased_alibaba_holdout": False,
        },
        "environment": {
            "profiles": CAPACITY_PROFILES,
            "total_cache_budget": TOTAL_CACHE_BUDGET,
            "capacity_assignment_namespace": CAPACITY_ASSIGNMENT_NAMESPACE,
            "bandwidth_hz": 15000,
            "users": 20,
            "servers": 10,
            "services": 10,
            "tasks_per_dag_max": 10,
        },
        "methods": list(METHOD_LABELS),
        "training": {
            "seeds": list(FINAL_SEEDS),
            "from_scratch_per_profile_and_seed": True,
            "convergence_profile": "e2_converged",
            "evaluation_scenarios_per_seed": 100,
            "neural_network_frozen_during_evaluation": True,
        },
        "formal_gate": {
            "primary_metric": "mean_DAG_completion_time",
            "comparisons": [
                "lean_our_vs_guided_full",
                "lean_our_vs_centralized_greedy_daoc",
            ],
            "ci95_lower_sec": ">0",
            "wilcoxon_one_sided_p": "<0.05",
            "wins": ">=7/10",
            "p95": "positive_mean_improvement_and_at_least_5/10_wins",
        },
        "governance": {
            "one_time_final_confirmation": True,
            "no_retuning_from_final_results": True,
            "resume_requires_identical_lock": True,
        },
    }


def lock_payload():
    freeze = read_json(FREEZE_PATH)
    if not (
        freeze.get("status") == "frozen"
        and freeze.get("revision_id") == REVISION_ID
        and freeze.get("development_gate", {}).get("passed") is True
        and freeze.get("formal_final_seeds_run") is False
    ):
        raise RuntimeError("h8v1 is not eligible for final confirmation")
    algorithm_hash = source_hash(ALGORITHM_SOURCE_FILES)
    if algorithm_hash != freeze.get("algorithm_source_sha256"):
        raise RuntimeError("Algorithm source drifted after the h8v1 freeze")
    specification = final_specification()
    return {
        "revision_id": REVISION_ID,
        "algorithm_source_sha256": algorithm_hash,
        "development_freeze_sha256": file_sha256(FREEZE_PATH),
        "final_protocol_source_sha256": source_hash(FINAL_PROTOCOL_FILES),
        "final_configuration_sha256": canonical_hash(specification),
        "specification": specification,
    }


def prepare_lock(resume):
    expected = lock_payload()
    if LOCK_PATH.exists():
        current = read_json(LOCK_PATH)
        for key, value in expected.items():
            if current.get(key) != value:
                raise RuntimeError(f"FINAL_LOCK mismatch: {key}")
        if not resume:
            raise RuntimeError(
                "FINAL_LOCK already exists; use --resume without changing code"
            )
        return current
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "locked",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **expected,
        "formal_results_seen_before_lock": False,
    }
    write_json(LOCK_PATH, lock)
    return lock


def run_logged(command, log_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(item) for item in command), flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def reproduction_command(directory, profile, args):
    command = [
        sys.executable,
        str(ROOT_DIR / "run_reproduction_suite.py"),
        "--profile",
        "e2_converged",
        "--suite-dir",
        str(directory),
        "--seeds",
        ",".join(str(seed) for seed in FINAL_SEEDS),
        "--labels",
        ",".join(METHOD_LABELS),
        "--dag-dataset-path",
        str(DATASET_PATH),
        "--dag-dataset-sha256",
        EXPECTED_DATASET_SHA256,
        "--server-capacity",
        "1",
        "--server-capacity-multiset",
        capacity_text(profile),
        "--baseline-server-capacity",
        str(BASELINE_RANDOM_DRAW_CAPACITY),
        "--capacity-assignment-namespace",
        CAPACITY_ASSIGNMENT_NAMESPACE,
        "--workers",
        str(args.workers),
        "--revision-id",
        REVISION_ID,
        "--revision-parent",
        "h8v0",
        "--revision-reason",
        "frozen_algorithm_final_confirmation",
        "--revision-changed-module",
        "scarcity_aware_service_coverage_constraint",
        "--revision-expected-metric",
        "formal_superiority_across_capacity_profiles",
        "--revision-rejection-condition",
        "formal_gate_not_met_without_retuning",
        "--seed-partition",
        "final",
    ]
    if args.resume:
        command.append("--resume")
    return command


def oracle_command(directory):
    return [
        sys.executable,
        str(ROOT_DIR / "evaluate_oracle_latency_bound.py"),
        "--our-suite-dir",
        str(directory),
        "--our-label",
        "lean_our",
        "--daoc-suite-dir",
        str(directory),
        "--daoc-label",
        "guided_full",
        "--output-dir",
        str(directory / "oracle"),
        "--seeds",
        ",".join(str(seed) for seed in FINAL_SEEDS),
        "--episodes",
        "100",
        "--exact-check-scenarios",
        "0",
    ]


def analysis_command():
    return [
        sys.executable,
        str(ROOT_DIR / "analyze_a0_fixed_budget_heterogeneity_final.py"),
        "--suite-root",
        str(FINAL_ROOT),
        "--output-dir",
        str(FINAL_ROOT / "analysis"),
        "--seeds",
        ",".join(str(seed) for seed in FINAL_SEEDS),
    ]


def run_training(args):
    for profile in PROFILE_ORDER:
        directory = FINAL_ROOT / profile
        directory.mkdir(parents=True, exist_ok=True)
        run_logged(
            reproduction_command(directory, profile, args),
            directory / "runner.log",
        )


def run_oracles():
    for profile in PROFILE_ORDER:
        directory = FINAL_ROOT / profile
        run_logged(oracle_command(directory), directory / "oracle.log")


def run_analysis():
    output = FINAL_ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    run_logged(analysis_command(), output / "analysis.log")
    return read_json(output / "final_heterogeneity_summary.json")


def main():
    args = parse_args()
    validate_protocol()
    lock = prepare_lock(args.resume)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "lock":
        print(f"Final lock: {LOCK_PATH}", flush=True)
        return
    stages = (
        ("train", "oracle", "analysis")
        if args.stage == "all"
        else (args.stage,)
    )
    manifest_path = FINAL_ROOT / "RUN_MANIFEST.json"
    started = time.perf_counter()
    manifest = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "lock_sha256": file_sha256(LOCK_PATH),
        "stages": list(stages),
    }
    write_json(manifest_path, manifest)
    try:
        result = None
        for stage in stages:
            if stage == "train":
                run_training(args)
            elif stage == "oracle":
                run_oracles()
            else:
                result = run_analysis()
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "error": repr(error),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json(manifest_path, manifest)
        raise

    current = lock_payload()
    for key, value in current.items():
        if lock.get(key) != value:
            raise RuntimeError(f"Source changed during final run: {key}")
    manifest.update(
        {
            "status": "complete",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "wall_time_sec": time.perf_counter() - started,
            "formal_gate": result.get("gate") if result else None,
        }
    )
    write_json(manifest_path, manifest)
    if result is not None:
        lock.update(
            {
                "status": "complete",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "result_summary_sha256": file_sha256(
                    FINAL_ROOT
                    / "analysis"
                    / "final_heterogeneity_summary.json"
                ),
                "formal_gate": result["gate"],
                "algorithm_retuned_after_final": False,
            }
        )
        write_json(LOCK_PATH, lock)
    print(f"Final confirmation artifacts: {FINAL_ROOT}", flush=True)


if __name__ == "__main__":
    main()
