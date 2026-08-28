#!/usr/bin/env python3
"""Run the governed DAOC plus OUR coordinated-cache control."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from capacity_protocol import deterministic_capacity_assignment
from pegasus_daoc_coord_extension_protocol import (
    ANALYSIS_DIR,
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DAOC_COORD_LABEL,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_DIR,
    FINAL_SEEDS,
    P3_FINAL_DIR,
    PROFILE_CONVERGED,
    PROFILE_SMOKE,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    SMOKE_DIR,
    SMOKE_SEEDS,
    TASK_LIMIT_INCLUDING_DUMMY,
    sha256_file,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_LOCK = RESULT_ROOT / "PROTOCOL_LOCK.json"
FROZEN_BASELINE = RESULT_ROOT / "FROZEN_BASELINE.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_LOCK.json"

ALGORITHM_SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_reward.py",
    "dqn.py",
    "server.py",
    "simulator.py",
    "task.py",
    "user.py",
)
EXTENSION_SOURCE_FILES = ALGORITHM_SOURCE_FILES + (
    "analyze_pegasus_daoc_coord_extension.py",
    "pegasus_daoc_coord_extension_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_daoc_coord_extension.py",
    "run_pegasus_daoc_coord_suite.py",
    "run_reproduction_suite.py",
    "test_pegasus_daoc_coord_extension.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "freeze", "final", "analysis", "all"),
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def canonical_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hash(files):
    digest = hashlib.sha256()
    for filename in sorted(set(files)):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT / filename).read_bytes())
    return digest.hexdigest()


def initialize_protocol_lock():
    specification = validate_protocol(require_p7=True)
    lock = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "extension_source_sha256": source_hash(EXTENSION_SOURCE_FILES),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_LOCK.exists():
        existing = read_json(PROTOCOL_LOCK)
        for key in (
            "specification_sha256",
            "algorithm_source_sha256",
            "extension_source_sha256",
        ):
            if existing.get(key) != lock[key]:
                raise RuntimeError(f"P8 protocol changed after freezing: {key}")
        return existing
    write_json(PROTOCOL_LOCK, lock)
    return lock


def verify_frozen_sources():
    lock = read_json(PROTOCOL_LOCK)
    if lock["algorithm_source_sha256"] != source_hash(ALGORITHM_SOURCE_FILES):
        raise RuntimeError("Algorithm source changed during P8 execution")
    if lock["extension_source_sha256"] != source_hash(EXTENSION_SOURCE_FILES):
        raise RuntimeError("Extension source changed during P8 execution")


def run_logged(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(RESULT_ROOT / ".matplotlib")
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


def suite_command(profile, suite_dir, seeds, workers, partition, resume):
    command = [
        sys.executable,
        str(ROOT / "run_pegasus_daoc_coord_suite.py"),
        "--profile",
        profile,
        "--suite-dir",
        str(suite_dir),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--labels",
        DAOC_COORD_LABEL,
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
        "pegasus_p6_baselines_ablation_v1",
        "--revision-reason",
        "isolate_our_coordinated_cache_under_daoc_scheduler",
        "--revision-changed-module",
        "cache_policy_only_daoc_to_our_coord_cache",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_unconverged_control",
        "--seed-partition",
        partition,
    ]
    if resume:
        command.append("--resume")
    return command


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


def assert_suite_complete(suite_dir, seeds, evaluations, convergence):
    manifest = read_json(Path(suite_dir) / "suite_manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete suite: {suite_dir}")
    for seed in seeds:
        run_dir = Path(suite_dir) / "runs" / DAOC_COORD_LABEL / f"seed_{seed}"
        summary = read_json(run_dir / "summary.json")
        arguments = read_json(run_dir / "config.json")["arguments"]
        if summary.get("status") != "complete":
            raise RuntimeError(f"Incomplete run: {run_dir}")
        if convergence and not (
            summary.get("eligible_for_comparison", False)
            and summary.get("convergence", {}).get("reached", False)
        ):
            raise RuntimeError(f"Unconverged run: {run_dir}")
        if summary.get("evaluation_scenario_count") != evaluations:
            raise RuntimeError(f"Wrong evaluation count: {run_dir}")
        if not summary.get("evaluation_state_frozen", False):
            raise RuntimeError(f"Evaluation state was not frozen: {run_dir}")
        if arguments.get("algorithm") != "prev_servers_plus_service_per_serverDQN":
            raise RuntimeError(f"DAOC scheduler changed: {run_dir}")
        if arguments.get("reward_mode") != "terminal_binary":
            raise RuntimeError(f"DAOC reward changed: {run_dir}")
        if arguments.get("cache_policy") != "critical_path_joint":
            raise RuntimeError(f"OUR cache is missing: {run_dir}")
        if not arguments.get("cache_coverage_constraint"):
            raise RuntimeError(f"OUR coverage constraint is missing: {run_dir}")
        capacities = {
            int(key): int(value)
            for key, value in summary["server_capacities"].items()
        }
        expected = deterministic_capacity_assignment(
            CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        if capacities != expected:
            raise RuntimeError(f"Capacity assignment mismatch: {run_dir}")


def assert_final_pairing():
    for seed in FINAL_SEEDS:
        new_dir = FINAL_DIR / "runs" / DAOC_COORD_LABEL / f"seed_{seed}"
        daoc_dir = P3_FINAL_DIR / "runs/daoc_paper" / f"seed_{seed}"
        our_dir = P3_FINAL_DIR / "runs/lean_our" / f"seed_{seed}"
        banks = [
            comparable_bank(new_dir / "evaluation_scenarios.json"),
            comparable_bank(daoc_dir / "evaluation_scenarios.json"),
            comparable_bank(our_dir / "evaluation_scenarios.json"),
        ]
        if not all(bank == banks[0] for bank in banks[1:]):
            raise RuntimeError(f"Unpaired final scenario bank for seed {seed}")


def run_tests():
    tests = (
        "test_information_protocol.py",
        "test_capacity_protocol.py",
        "test_dag_completion_semantics.py",
        "test_pegasus_daoc_coord_extension.py",
    )
    run_logged(
        [sys.executable, "-m", "unittest", "-v", *tests],
        RESULT_ROOT / "tests.log",
    )


def run_smoke(workers, resume):
    run_logged(
        suite_command(
            PROFILE_SMOKE,
            SMOKE_DIR,
            SMOKE_SEEDS,
            workers,
            "smoke",
            resume,
        ),
        SMOKE_DIR / "runner.log",
    )
    assert_suite_complete(
        SMOKE_DIR,
        SMOKE_SEEDS,
        evaluations=20,
        convergence=False,
    )


def freeze_baseline():
    validate_protocol(require_p7=True)
    assert_suite_complete(
        SMOKE_DIR,
        SMOKE_SEEDS,
        evaluations=20,
        convergence=False,
    )
    verify_frozen_sources()
    payload = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "smoke_summary_sha256": sha256_file(
            SMOKE_DIR
            / "runs"
            / DAOC_COORD_LABEL
            / f"seed_{SMOKE_SEEDS[0]}"
            / "summary.json"
        ),
        "statement": (
            "Only the DAOC+OUR-CoordCache control is frozen. Existing DAOC "
            "and OUR checkpoints remain immutable."
        ),
    }
    payload["freeze_sha256"] = canonical_hash(payload)
    if FROZEN_BASELINE.exists():
        existing = read_json(FROZEN_BASELINE)
        if existing.get("protocol_lock_sha256") != payload[
            "protocol_lock_sha256"
        ]:
            raise RuntimeError("Frozen P8 protocol mismatch")
        return
    write_json(FROZEN_BASELINE, payload)


def initialize_final_lock(resume):
    if not FROZEN_BASELINE.exists():
        raise RuntimeError("Run the freeze stage before final training")
    frozen = read_json(FROZEN_BASELINE)
    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "freeze_sha256": frozen["freeze_sha256"],
        "method": DAOC_COORD_LABEL,
        "seeds": list(FINAL_SEEDS),
        "scenarios_per_seed": EVALUATION_EPISODES,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "capacity_multiset": list(CAPACITY_MULTISET),
    }
    lock = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
    }
    if FINAL_LOCK.exists():
        existing = read_json(FINAL_LOCK)
        if not resume:
            raise RuntimeError("P8 final lock exists; use --resume")
        if existing.get("specification_sha256") != lock[
            "specification_sha256"
        ]:
            raise RuntimeError("P8 final lock specification mismatch")
        return existing
    write_json(FINAL_LOCK, lock)
    return lock


def run_final(workers, resume):
    initialize_final_lock(resume)
    verify_frozen_sources()
    run_logged(
        suite_command(
            PROFILE_CONVERGED,
            FINAL_DIR,
            FINAL_SEEDS,
            workers,
            "baseline_extension",
            resume,
        ),
        FINAL_DIR / "runner.log",
    )
    assert_suite_complete(
        FINAL_DIR,
        FINAL_SEEDS,
        evaluations=EVALUATION_EPISODES,
        convergence=True,
    )
    assert_final_pairing()


def run_analysis():
    verify_frozen_sources()
    assert_suite_complete(
        FINAL_DIR,
        FINAL_SEEDS,
        evaluations=EVALUATION_EPISODES,
        convergence=True,
    )
    assert_final_pairing()
    run_logged(
        [
            sys.executable,
            str(ROOT / "analyze_pegasus_daoc_coord_extension.py"),
            "--output-dir",
            str(ANALYSIS_DIR),
        ],
        ANALYSIS_DIR / "analysis.log",
    )
    summary_path = ANALYSIS_DIR / "daoc_coord_cache_extension_summary.json"
    summary = read_json(summary_path)
    lock = read_json(FINAL_LOCK)
    lock.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary_sha256": sha256_file(summary_path),
            "integrity": summary["integrity"],
            "evidence": summary["evidence"],
        }
    )
    write_json(FINAL_LOCK, lock)


def main():
    args = parse_args()
    initialize_protocol_lock()
    stages = (
        ("tests", run_tests),
        ("smoke", lambda: run_smoke(args.workers, args.resume)),
        ("freeze", freeze_baseline),
        ("final", lambda: run_final(args.workers, args.resume)),
        ("analysis", run_analysis),
    )
    for name, operation in stages:
        if args.stage in {name, "all"}:
            operation()
    print(f"P8 DAOC coordinated-cache artifacts: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
