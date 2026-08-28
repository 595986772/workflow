#!/usr/bin/env python3
"""Run the frozen Alibaba-CP100 budget-20 mechanism experiment."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from alibaba_cp100_protocol import (
    ALIBABA_CP100_PROTOCOL_VERSION,
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_MULTISET,
    COMPARISON_SEEDS,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    SMOKE_SEEDS,
    validate_protocol,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = (
    ROOT_DIR / "results" / "alibaba_cp100_budget20" / "a1"
)
SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_reward.py",
    "critical_path_rl.py",
    "dqn.py",
    "server.py",
    "simulator.py",
    "task.py",
    "user.py",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alibaba-CP100 DAOC versus OUR."
    )
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "converged", "all"),
        default="all",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def source_hash():
    digest = hashlib.sha256()
    for filename in SOURCE_FILES:
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT_DIR / filename).read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def merge_completed_stages(state_path, completed_stages):
    """Preserve stage history across resumable, separate invocations."""
    stage_order = ("tests", "smoke", "converged")
    existing_stages = []
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        existing_stages = existing.get("completed_stages", [])
    completed = set(existing_stages) | set(completed_stages)
    return [stage for stage in stage_order if stage in completed]


def run_logged(command, log_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed with exit {result.returncode}; see {log_path}"
        )


def reproduction_command(
    suite_dir,
    profile,
    seeds,
    workers,
    resume,
):
    command = [
        sys.executable,
        str(ROOT_DIR / "run_reproduction_suite.py"),
        "--profile",
        profile,
        "--suite-dir",
        str(suite_dir),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--labels",
        "guided_full,lean_our",
        "--dag-dataset-path",
        str(DATASET_PATH),
        "--dag-dataset-sha256",
        EXPECTED_DATASET_SHA256,
        "--server-capacity",
        "2",
        "--server-capacity-multiset",
        ",".join(str(value) for value in CAPACITY_MULTISET),
        "--capacity-assignment-namespace",
        CAPACITY_ASSIGNMENT_NAMESPACE,
        "--baseline-server-capacity",
        str(BASELINE_RANDOM_DRAW_CAPACITY),
        "--workers",
        str(workers),
        "--revision-id",
        "a1",
        "--revision-reason",
        "alibaba_cp100_budget20_mechanism_evaluation",
        "--revision-changed-module",
        "dataset_loader_only",
        "--revision-expected-metric",
        "paired_finish_time",
        "--revision-rejection-condition",
        "report_without_dataset_specific_tuning",
        "--seed-partition",
        "mechanism",
    ]
    if resume:
        command.append("--resume")
    return command


def analysis_command(suite_dir, mode, seeds):
    return [
        sys.executable,
        str(ROOT_DIR / "analyze_alibaba_cp100.py"),
        "--suite-dir",
        str(suite_dir),
        "--mode",
        mode,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
    ]


def main():
    args = parse_args()
    protocol = validate_protocol()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    lock_path = result_dir / "PROTOCOL_LOCK.json"
    lock = {
        "status": "locked",
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": protocol,
        "algorithm_source_sha256": source_hash(),
        "dataset_specific_algorithm_tuning_allowed": False,
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            existing["protocol"]["protocol_version"]
            != ALIBABA_CP100_PROTOCOL_VERSION
            or existing["algorithm_source_sha256"]
            != lock["algorithm_source_sha256"]
        ):
            raise RuntimeError(
                "Existing protocol lock does not match current source"
            )
    else:
        write_json(lock_path, lock)

    stages = (
        ("tests", "smoke", "converged")
        if args.stage == "all"
        else (args.stage,)
    )
    if "tests" in stages:
        run_logged(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-p",
                "test_*.py",
            ],
            result_dir / "tests.log",
        )

    smoke_dir = result_dir / "smoke"
    if "smoke" in stages:
        smoke_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            reproduction_command(
                smoke_dir,
                "e2_smoke",
                SMOKE_SEEDS,
                args.workers,
                args.resume,
            ),
            smoke_dir / "runner.log",
        )
        run_logged(
            analysis_command(smoke_dir, "smoke", SMOKE_SEEDS),
            smoke_dir / "analysis.log",
        )
        smoke = json.loads(
            (smoke_dir / "alibaba_cp100_analysis.json").read_text(
                encoding="utf-8"
            )
        )
        if not smoke["gate"]["passed"]:
            raise RuntimeError("Alibaba-CP100 smoke integrity gate failed")

    converged_dir = result_dir / "converged"
    if "converged" in stages:
        converged_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            reproduction_command(
                converged_dir,
                "e2_converged",
                COMPARISON_SEEDS,
                args.workers,
                args.resume,
            ),
            converged_dir / "runner.log",
        )
        run_logged(
            analysis_command(
                converged_dir,
                "converged",
                COMPARISON_SEEDS,
            ),
            converged_dir / "analysis.log",
        )

    state_path = result_dir / "EXPERIMENT_STATE.json"
    state = {
        "status": "complete",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "completed_stages": merge_completed_stages(
            state_path,
            stages,
        ),
        "result_dir": str(result_dir),
    }
    if (converged_dir / "alibaba_cp100_analysis.json").exists():
        analysis = json.loads(
            (
                converged_dir / "alibaba_cp100_analysis.json"
            ).read_text(encoding="utf-8")
        )
        state["comparison_gate_passed"] = analysis["gate"]["passed"]
    write_json(state_path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
