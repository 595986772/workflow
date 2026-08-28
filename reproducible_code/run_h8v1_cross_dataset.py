#!/usr/bin/env python3
"""Run the governed three-seed Pegasus cross-topology validation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from run_a0_fixed_budget_heterogeneity import (
    ALGORITHM_SOURCE_FILES,
    source_hash,
)


ROOT = Path(__file__).resolve().parent
REVISION_ROOT = ROOT / "results/h8v1_cross_topology/xv1"
DATASET_DIR = ROOT / "datasets/pegasus_cross_topology"
DATASET_PATH = DATASET_DIR / "dag_pegasus5_sub10.json"
DATASET_MANIFEST = DATASET_DIR / "manifest.json"
H8V1_LOCK = ROOT / "results/a0_fixed_budget_heterogeneity/h8v1/FINAL_LOCK.json"
LOCK_PATH = REVISION_ROOT / "CROSS_DATASET_LOCK.json"
METHODS = ("guided_full", "centralized_greedy_daoc", "lean_our")
CAPACITY_MULTISET = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2)
CAPACITY_NAMESPACE = "h8v1_pegasus_cross_topology_v1"
SMOKE_SEEDS = (31,)
DEVELOPMENT_SEEDS = (31, 32, 33)


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


def protocol_spec(dataset_sha256, locked_algorithm_hash):
    return {
        "protocol_id": "h8v1_pegasus_cross_topology_v1",
        "analysis_status": "posthoc_external_topology_validation",
        "algorithm_revision": "h8v1_frozen",
        "algorithm_source_sha256": locked_algorithm_hash,
        "dataset": {
            "path": str(DATASET_PATH.resolve()),
            "sha256": dataset_sha256,
            "source_manifest": str(DATASET_MANIFEST.resolve()),
            "unbiased_mec_holdout": False,
        },
        "environment": {
            "users": 20,
            "servers": 10,
            "services": 10,
            "tasks_including_dummy_max": 10,
            "bandwidth_hz": 15000,
            "capacity_multiset": list(CAPACITY_MULTISET),
            "total_cache_budget": 8,
            "capacity_assignment_namespace": CAPACITY_NAMESPACE,
        },
        "methods": list(METHODS),
        "smoke": {"seeds": list(SMOKE_SEEDS), "train_episodes": 200, "eval_episodes": 20},
        "development": {
            "seeds": list(DEVELOPMENT_SEEDS),
            "from_scratch": True,
            "convergence_profile": "e2_converged",
            "eval_episodes": 100,
        },
        "gate": {
            "our_beats_daoc": "positive_mean_and_at_least_2_of_3_wins",
            "our_beats_central": "positive_mean_and_at_least_2_of_3_wins",
            "p95_beats_central": "positive_mean_and_at_least_2_of_3_wins",
        },
        "governance": {
            "no_algorithm_retuning": True,
            "three_seed_diagnostic_not_formal_significance": True,
            "no_automatic_ten_seed_expansion": True,
        },
    }


def initialize_lock():
    if not DATASET_MANIFEST.exists() or not DATASET_PATH.exists():
        raise RuntimeError("Pegasus dataset has not been built")
    dataset_manifest = read_json(DATASET_MANIFEST)
    expected_dataset_hash = dataset_manifest["dataset"]["sha256"]
    if sha256_file(DATASET_PATH) != expected_dataset_hash:
        raise RuntimeError("Pegasus dataset hash mismatch")
    h8v1_lock = read_json(H8V1_LOCK)
    locked_algorithm_hash = h8v1_lock["algorithm_source_sha256"]
    current_algorithm_hash = source_hash(ALGORITHM_SOURCE_FILES)
    if current_algorithm_hash != locked_algorithm_hash:
        raise RuntimeError("Current algorithm files no longer match frozen h8v1")
    spec = protocol_spec(expected_dataset_hash, locked_algorithm_hash)
    lock = {
        "status": "locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": spec,
        "specification_sha256": canonical_hash(spec),
        "h8v1_final_lock_path": str(H8V1_LOCK.resolve()),
        "h8v1_final_lock_sha256": sha256_file(H8V1_LOCK),
        "algorithm_retuned": False,
    }
    REVISION_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        if existing.get("specification_sha256") != lock["specification_sha256"]:
            raise RuntimeError("Existing cross-dataset lock does not match this protocol")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def run_logged(command, log_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(value) for value in command), flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def reproduction_command(profile, suite_dir, seeds, workers, resume, dataset_hash):
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
        dataset_hash,
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
        "h8v1-xdag-v1",
        "--revision-parent",
        "h8v1",
        "--revision-reason",
        "posthoc_external_cross_topology_validation",
        "--revision-changed-module",
        "environment_dag_dataset_only",
        "--revision-expected-metric",
        "directional_generalization_against_daoc_and_central",
        "--revision-rejection-condition",
        "nonpositive_mean_or_fewer_than_two_seed_wins",
        "--seed-partition",
        "generalization",
    ]
    if resume:
        command.append("--resume")
    return command


def check_smoke(suite_dir, dataset_hash):
    for label in METHODS:
        summary_path = suite_dir / "runs" / label / "seed_31" / "summary.json"
        summary = read_json(summary_path)
        if summary.get("status") != "complete":
            raise RuntimeError(f"Smoke incomplete: {label}")
        if summary.get("dag_dataset", {}).get("sha256") != dataset_hash:
            raise RuntimeError(f"Smoke dataset mismatch: {label}")
        if summary.get("evaluation_unique_scenarios") != 20:
            raise RuntimeError(f"Smoke scenario count mismatch: {label}")
        if label == "lean_our" and "scarcity_aware_service_coverage_constraint" not in summary.get(
            "method_modules", {}
        ).get("active", []):
            raise RuntimeError("Smoke OUR is not the frozen h8v1 method")


def run_tests():
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "test_build_pegasus_cross_dataset.py",
        "test_analyze_h8v1_posthoc.py",
        "test_capacity_protocol.py",
        "test_a0_fixed_budget_heterogeneity_final.py",
    ]
    run_logged(command, REVISION_ROOT / "tests.log")


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
        "100",
        "--exact-check-scenarios",
        "0",
    ]
    run_logged(command, suite_dir / "oracle.log")


def run_analysis(suite_dir, dataset_hash):
    command = [
        sys.executable,
        str(ROOT / "analyze_h8v1_cross_dataset.py"),
        "--suite-dir",
        str(suite_dir),
        "--dataset-path",
        str(DATASET_PATH),
        "--dataset-sha256",
        dataset_hash,
        "--output-dir",
        str(suite_dir / "analysis"),
    ]
    run_logged(command, suite_dir / "analysis.log")


def main():
    args = parse_args()
    lock = initialize_lock()
    dataset_hash = lock["specification"]["dataset"]["sha256"]
    smoke_dir = REVISION_ROOT / "smoke"
    converged_dir = REVISION_ROOT / "converged"
    stages = (
        ("tests", run_tests),
        (
            "smoke",
            lambda: (
                run_logged(
                    reproduction_command(
                        "e2_smoke",
                        smoke_dir,
                        SMOKE_SEEDS,
                        args.workers,
                        args.resume,
                        dataset_hash,
                    ),
                    smoke_dir / "runner.log",
                ),
                check_smoke(smoke_dir, dataset_hash),
            ),
        ),
        (
            "converged",
            lambda: run_logged(
                reproduction_command(
                    "e2_converged",
                    converged_dir,
                    DEVELOPMENT_SEEDS,
                    args.workers,
                    args.resume,
                    dataset_hash,
                ),
                converged_dir / "runner.log",
            ),
        ),
        ("oracle", lambda: run_oracle(converged_dir)),
        ("analysis", lambda: run_analysis(converged_dir, dataset_hash)),
    )
    for stage_name, function in stages:
        if args.stage not in ("all", stage_name):
            continue
        function()

    if args.stage in ("all", "analysis"):
        summary = read_json(converged_dir / "analysis/cross_topology_summary.json")
        lock = read_json(LOCK_PATH)
        lock["status"] = "complete"
        lock["completed_at"] = datetime.now(timezone.utc).isoformat()
        lock["development_gate"] = summary["gate"]
        lock["algorithm_retuned"] = False
        write_json(LOCK_PATH, lock)
    print(f"Cross-dataset artifacts: {REVISION_ROOT}")


if __name__ == "__main__":
    main()
