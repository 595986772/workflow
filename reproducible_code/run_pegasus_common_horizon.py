#!/usr/bin/env python3
"""Run and audit the Pegasus-B8 26k common-horizon experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from capacity_protocol import deterministic_capacity_assignment
from pegasus_common_horizon_protocol import (
    ANALYSIS_DIR,
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DATASET_PATH,
    DEFAULT_WORKERS,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_DIR,
    FINAL_SEEDS,
    FIXED_TRAIN_EPISODES,
    PROFILE_NAME,
    PROTOCOL_VERSION,
    RERUN_METHODS,
    RESULT_ROOT,
    TASK_LIMIT_INCLUDING_DUMMY,
    sha256_file,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_LOCK = RESULT_ROOT / "PROTOCOL_LOCK.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_LOCK.json"
RUNNER_LOG = RESULT_ROOT / "runner.log"
SUMMARY_PATH = ANALYSIS_DIR / "common_horizon_summary.json"

ALGORITHM_SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_reward.py",
    "critical_path_rl.py",
    "discrete_sac.py",
    "dqn.py",
    "server.py",
    "simulator.py",
    "task.py",
    "user.py",
)
PROTOCOL_SOURCE_FILES = ALGORITHM_SOURCE_FILES + (
    "analyze_pegasus_common_horizon.py",
    "pegasus_common_horizon_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_common_horizon.py",
    "run_pegasus_common_horizon_suite.py",
    "run_reproduction_suite.py",
    "test_pegasus_common_horizon.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "final", "analysis", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash(files):
    digest = hashlib.sha256()
    for filename in sorted(set(files)):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT / filename).read_bytes())
    return digest.hexdigest()


def initialize_lock():
    specification = validate_protocol()
    lock = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "protocol_source_sha256": source_hash(PROTOCOL_SOURCE_FILES),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_LOCK.exists():
        existing = read_json(PROTOCOL_LOCK)
        for key in (
            "specification_sha256",
            "algorithm_source_sha256",
            "protocol_source_sha256",
        ):
            if existing.get(key) != lock[key]:
                raise RuntimeError(
                    f"Common-horizon protocol changed after launch: {key}"
                )
        return existing
    write_json(PROTOCOL_LOCK, lock)
    return lock


def verify_sources(lock):
    if lock["algorithm_source_sha256"] != source_hash(
        ALGORITHM_SOURCE_FILES
    ):
        raise RuntimeError("Algorithm source changed during the rerun")
    if lock["protocol_source_sha256"] != source_hash(
        PROTOCOL_SOURCE_FILES
    ):
        raise RuntimeError("Protocol source changed during the rerun")


def run_logged(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(RESULT_ROOT / ".matplotlib"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(value) for value in command), flush=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + " ".join(str(value) for value in command)
            + "\n"
        )
        output.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def comparable_bank(path):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in read_json(path)
    ]


def run_tests():
    tests = (
        "test_pegasus_common_horizon.py",
        "test_discrete_sac.py",
        "test_information_protocol.py",
        "test_capacity_protocol.py",
        "test_dag_completion_semantics.py",
        "test_pegasus_pscale_protocol.py",
    )
    run_logged(
        [sys.executable, "-m", "unittest", "-v", *tests],
        RESULT_ROOT / "tests.log",
    )


def run_final(workers, resume):
    command = [
        sys.executable,
        str(ROOT / "run_pegasus_common_horizon_suite.py"),
        "--profile",
        PROFILE_NAME,
        "--suite-dir",
        str(FINAL_DIR),
        "--seeds",
        ",".join(str(seed) for seed in FINAL_SEEDS),
        "--labels",
        ",".join(RERUN_METHODS),
        "--train-episodes",
        str(FIXED_TRAIN_EPISODES),
        "--eval-episodes",
        str(EVALUATION_EPISODES),
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
        "pegasus_p8_daoc_our_coord_cache_v1",
        "--revision-reason",
        "align_fast_learning_methods_to_common_26000_episode_horizon",
        "--revision-changed-module",
        "training_horizon_only",
        "--revision-expected-metric",
        "online_tail_episodes_25001_to_26000",
        "--revision-rejection-condition",
        "integrity_failure_or_training_instability",
        "--seed-partition",
        "confirmation",
        "--keep-going",
    ]
    if resume:
        command.append("--resume")
    run_logged(command, RUNNER_LOG)


def assert_final_complete():
    manifest = read_json(FINAL_DIR / "suite_manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError("Common-horizon suite is incomplete")
    if manifest.get("failed_runs"):
        raise RuntimeError(
            f"Common-horizon failed runs: {manifest['failed_runs']}"
        )

    for seed in FINAL_SEEDS:
        reference_bank = None
        expected_capacity = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        for label in RERUN_METHODS:
            directory = FINAL_DIR / "runs" / label / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            arguments = read_json(directory / "config.json")["arguments"]
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {directory}")
            if summary.get("train", {}).get("episodes") != FIXED_TRAIN_EPISODES:
                raise RuntimeError(f"Wrong training horizon: {directory}")
            if summary.get("selected_checkpoint_episode") != FIXED_TRAIN_EPISODES:
                raise RuntimeError(f"Wrong final checkpoint: {directory}")
            if summary.get("checkpoint_strategy") != "fixed_budget_final":
                raise RuntimeError(f"Historical checkpoint selected: {directory}")
            if summary.get("evaluation_scenario_count") != EVALUATION_EPISODES:
                raise RuntimeError(f"Wrong evaluation count: {directory}")
            if not summary.get("evaluation_state_frozen"):
                raise RuntimeError(f"Evaluation was not frozen: {directory}")
            if arguments.get("convergence_mode"):
                raise RuntimeError(f"Early stopping remained enabled: {directory}")
            if arguments.get("checkpoint_every") != 0:
                raise RuntimeError(f"Checkpoint selection remained enabled: {directory}")
            capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            if capacities != expected_capacity:
                raise RuntimeError(f"Capacity mismatch: {directory}")
            bank = comparable_bank(directory / "evaluation_scenarios.json")
            if reference_bank is None:
                reference_bank = bank
            elif bank != reference_bank:
                raise RuntimeError(
                    f"Unpaired new scenario bank for seed {seed}"
                )


def run_analysis():
    run_logged(
        [sys.executable, str(ROOT / "analyze_pegasus_common_horizon.py")],
        RESULT_ROOT / "analysis.log",
    )
    if not SUMMARY_PATH.exists():
        raise RuntimeError("Common-horizon analysis summary is missing")
    summary = read_json(SUMMARY_PATH)
    if not all(summary.get("integrity", {}).values()):
        raise RuntimeError("Common-horizon analysis integrity failed")


def finalize_lock(protocol_lock, workers):
    final = {
        **protocol_lock,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "summary_path": str(SUMMARY_PATH.resolve()),
        "summary_sha256": sha256_file(SUMMARY_PATH),
    }
    write_json(FINAL_LOCK, final)


def main():
    args = parse_args()
    protocol_lock = initialize_lock()
    verify_sources(protocol_lock)
    if args.stage in ("tests", "all"):
        run_tests()
    if args.stage in ("final", "all"):
        run_final(args.workers, args.resume)
        assert_final_complete()
    if args.stage in ("analysis", "all"):
        assert_final_complete()
        run_analysis()
    verify_sources(protocol_lock)
    if args.stage == "all":
        finalize_lock(protocol_lock, args.workers)
    print(f"Common-horizon stage complete: {args.stage}")


if __name__ == "__main__":
    main()
