#!/usr/bin/env python3
"""Run active-service and service-image-size sensitivity experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pegasus_service_sensitivity_protocol import (
    ACTIVE_SERVICE_COUNTS,
    BANDWIDTH_HZ,
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    EVALUATION_EPISODES,
    FAMILIES,
    LEARNING_METHODS,
    METHODS,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    SEEDS,
    SERVICE_STATE_DIMENSION,
    SMOKE_PROFILES,
    TASK_LIMIT_INCLUDING_DUMMY,
    TRAINING_PROFILES,
    active_service_run,
    canonical_hash,
    projected_dataset_path,
    sha256_file,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
LOCK_PATH = RESULT_ROOT / "PROTOCOL_LOCK.json"
ACTIVE_AUDIT_PATH = RESULT_ROOT / "ACTIVE_SERVICE_AUDIT.json"
SOURCE_FILES = (
    "agent.py",
    "analyze_pegasus_service_sensitivity.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_reward.py",
    "critical_path_rl.py",
    "discrete_sac.py",
    "dqn.py",
    "evaluate_pegasus_service_size_sensitivity.py",
    "pegasus_service_sensitivity_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_service_sensitivity.py",
    "run_reproduction_suite.py",
    "server.py",
    "simulator.py",
    "task.py",
    "test_pegasus_service_sensitivity.py",
    "user.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "datasets",
            "smoke",
            "service_count",
            "service_size",
            "analysis",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
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


def source_hash() -> str:
    digest = hashlib.sha256()
    for filename in sorted(SOURCE_FILES):
        path = ROOT / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def initialize_lock() -> dict:
    specification = validate_protocol()
    lock = {
        "status": "locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "source_sha256": source_hash(),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        for key in ("specification_sha256", "source_sha256"):
            if existing.get(key) != lock[key]:
                raise RuntimeError(f"P14 protocol changed after lock: {key}")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def suite_dir(stage: str, active_services: int, method: str) -> Path:
    return RESULT_ROOT / stage / f"q{active_services}" / method


def suite_command(
    stage: str,
    active_services: int,
    method: str,
    resume: bool,
) -> list[str]:
    smoke = stage == "smoke"
    dataset = projected_dataset_path(active_services)
    profile = (
        SMOKE_PROFILES[method] if smoke else TRAINING_PROFILES[method]
    )
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        profile,
        "--suite-dir",
        str(suite_dir(stage, active_services, method)),
        "--seeds",
        "51" if smoke else ",".join(str(seed) for seed in SEEDS),
        "--labels",
        method,
        "--dag-dataset-path",
        str(dataset),
        "--dag-dataset-sha256",
        sha256_file(dataset),
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
        "1",
        "--revision-id",
        PROTOCOL_VERSION,
        "--revision-parent",
        "pegasus_p6_paper_baselines_v1",
        "--revision-reason",
        "active_service_count_cache_pressure_sensitivity",
        "--revision-changed-module",
        "workload_service_vocabulary_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_unconverged_learning_method",
        "--seed-partition",
        "generalization",
    ]
    if smoke:
        command.extend(
            ["--train-episodes", "200", "--eval-episodes", "20"]
        )
    if resume:
        command.append("--resume")
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + " ".join(command)
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


def run_training_jobs(stage: str, workers: int, resume: bool) -> None:
    counts = (4,) if stage == "smoke" else ACTIVE_SERVICE_COUNTS[:-1]
    jobs = [
        (active_services, method)
        for active_services in counts
        for method in METHODS
    ]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {
            pool.submit(
                run_logged,
                suite_command(stage, active_services, method, resume),
                suite_dir(stage, active_services, method) / "runner.log",
            ): (active_services, method)
            for active_services, method in jobs
        }
        for future in as_completed(futures):
            active_services, method = futures[future]
            future.result()
            print(
                f"Q_active={active_services} {method} {stage}: complete",
                flush=True,
            )


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


def audit_active_services() -> dict:
    report = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
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
                run = active_service_run(active_services, method, seed)
                summary = read_json(run / "summary.json")
                config = read_json(run / "config.json")["arguments"]
                scenarios = comparable_scenarios(
                    run / "evaluation_scenarios.json"
                )
                complete = (
                    summary.get("status") == "complete"
                    and summary.get("evaluation_scenario_count")
                    == EVALUATION_EPISODES
                )
                converged = method not in LEARNING_METHODS or bool(
                    summary.get("eligible_for_comparison")
                    and summary.get("convergence", {}).get("reached")
                )
                exact_once = all(
                    row.get("all_tasks_executed_once") == "1"
                    for row in _evaluation_rows(run / "episodes.csv")
                )
                capacity_valid = (
                    sorted(config.get("server_capacity_multiset", []))
                    == sorted(CAPACITY_MULTISET)
                    and summary.get("total_server_capacity")
                    == sum(CAPACITY_MULTISET)
                    and config.get("num_services")
                    == SERVICE_STATE_DIMENSION
                    and float(config.get("bandwidth")) == BANDWIDTH_HZ
                )
                paired = method_reference is None or scenarios == method_reference
                method_reference = (
                    scenarios if method_reference is None else method_reference
                )
                cross_key = seed
                cross_paired = (
                    cross_key not in cross_count_banks
                    or scenarios == cross_count_banks[cross_key]
                )
                cross_count_banks.setdefault(cross_key, scenarios)
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
                        "complete": complete,
                        "converged": converged,
                        "exact_once": exact_once,
                        "capacity_valid": capacity_valid,
                        "method_paired": paired,
                        "cross_count_paired": cross_paired,
                        "selected_checkpoint_episode": summary.get(
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
    write_json(ACTIVE_AUDIT_PATH, report)
    if report["status"] != "complete":
        raise RuntimeError("Active-service sensitivity audit failed")
    return report


def _evaluation_rows(path: Path):
    import csv

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["phase"] == "eval"]


def run_tests() -> None:
    run_logged(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "test_pegasus_service_sensitivity.py",
            "test_dag_completion_semantics.py",
            "test_capacity_protocol.py",
            "test_information_protocol.py",
        ],
        RESULT_ROOT / "tests.log",
    )


def run_service_size(workers: int, resume: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "evaluate_pegasus_service_size_sensitivity.py"),
        "--output-dir",
        str(RESULT_ROOT / "service_size"),
        "--workers",
        str(workers),
    ]
    if resume:
        command.append("--resume")
    run_logged(command, RESULT_ROOT / "service_size/runner.log")


def run_analysis() -> None:
    run_logged(
        [
            sys.executable,
            str(ROOT / "analyze_pegasus_service_sensitivity.py"),
        ],
        RESULT_ROOT / "analysis/runner.log",
    )


def main():
    args = parse_args()
    if args.stage in ("tests", "all"):
        run_tests()
    if args.stage in ("datasets", "all"):
        validate_protocol()
    if args.stage in ("smoke", "all"):
        run_training_jobs("smoke", args.workers, args.resume)
    if args.stage in ("service_count", "service_size", "analysis", "all"):
        initialize_lock()
    if args.stage in ("service_count", "all"):
        run_training_jobs("active_services", args.workers, args.resume)
        audit_active_services()
    if args.stage in ("service_size", "all"):
        run_service_size(args.workers, args.resume)
    if args.stage in ("analysis", "all"):
        audit_active_services()
        run_analysis()


if __name__ == "__main__":
    main()
