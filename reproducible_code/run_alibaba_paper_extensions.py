#!/usr/bin/env python3
"""Run three Alibaba-CP100 paper-extension experiments."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from alibaba_cp100_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_MULTISET,
    COMPARISON_SEEDS,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_SUITE = (
    ROOT_DIR / "results" / "alibaba_cp100_budget20" / "a1"
    / "converged"
)
DEFAULT_RESULT_DIR = (
    ROOT_DIR / "results" / "alibaba_cp100_budget20"
    / "a2_paper_extensions"
)
SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "critical_path_cache.py",
    "critical_path_rl.py",
    "run_independent_experiment.py",
    "run_reproduction_suite.py",
    "server.py",
    "simulator.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "train", "scale", "analyze", "all"),
        default="all",
    )
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        default=DEFAULT_SOURCE_SUITE,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scale-workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.scale_workers < 1
        or args.episodes < 1
    ):
        raise ValueError("workers and episodes must be positive")
    return args


def source_hash():
    digest = hashlib.sha256()
    for filename in SOURCE_FILES:
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT_DIR / filename).read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def merge_completed_stages(state_path, stages):
    order = ("tests", "smoke", "train", "scale", "analyze")
    existing = []
    if state_path.exists():
        existing = read_state = json.loads(
            state_path.read_text(encoding="utf-8")
        )
        existing = read_state.get("completed_stages", [])
    completed = set(existing) | set(stages)
    return [stage for stage in order if stage in completed]


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


def reproduction_command(suite_dir, profile, seeds, workers, resume):
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
        "our_dqn,centralized_greedy_daoc",
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
        "a2",
        "--revision-parent",
        "a1",
        "--revision-reason",
        "paper_extension_ablation_and_architecture_baseline",
        "--revision-changed-module",
        "q_architecture_or_cache_policy_only",
        "--revision-expected-metric",
        "paired_finish_time",
        "--revision-rejection-condition",
        "mechanism_contribution_not_supported",
        "--seed-partition",
        "mechanism",
    ]
    if resume:
        command.append("--resume")
    return command


def main():
    args = parse_args()
    source_suite = args.source_suite_dir.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    if not source_suite.is_dir():
        raise RuntimeError(f"Missing source suite: {source_suite}")
    lock_path = result_dir / "EXTENSION_PROTOCOL_LOCK.json"
    lock = {
        "status": "locked",
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "source_suite": str(source_suite),
        "seeds": list(COMPARISON_SEEDS),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "capacity_multiset": CAPACITY_MULTISET,
        "algorithm_source_sha256": source_hash(),
        "experiments": [
            "our_dqn_ablation",
            "centralized_greedy_daoc_baseline",
            "frozen_user_scaling_10_20_40_60",
        ],
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            existing["algorithm_source_sha256"]
            != lock["algorithm_source_sha256"]
        ):
            raise RuntimeError(
                "Existing extension lock does not match current source"
            )
    else:
        write_json(lock_path, lock)

    stages = (
        ("tests", "smoke", "train", "scale", "analyze")
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
                [COMPARISON_SEEDS[0]],
                args.workers,
                args.resume,
            ),
            smoke_dir / "runner.log",
        )

    train_dir = result_dir / "converged"
    if "train" in stages:
        train_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            reproduction_command(
                train_dir,
                "e2_converged",
                COMPARISON_SEEDS,
                args.workers,
                args.resume,
            ),
            train_dir / "runner.log",
        )

    scale_dir = result_dir / "scaling"
    if "scale" in stages:
        scale_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT_DIR / "evaluate_alibaba_user_scaling.py"),
            "--source-suite-dir",
            str(source_suite),
            "--output-dir",
            str(scale_dir),
            "--seeds",
            ",".join(str(seed) for seed in COMPARISON_SEEDS),
            "--user-counts",
            "10,20,40,60",
            "--episodes",
            str(args.episodes),
            "--workers",
            str(args.scale_workers),
        ]
        if args.resume:
            command.append("--resume")
        run_logged(command, scale_dir / "runner.log")

    analysis_dir = result_dir / "analysis"
    if "analyze" in stages:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                sys.executable,
                str(
                    ROOT_DIR
                    / "analyze_alibaba_paper_extensions.py"
                ),
                "--source-suite-dir",
                str(source_suite),
                "--extension-suite-dir",
                str(train_dir),
                "--output-dir",
                str(analysis_dir),
                "--seeds",
                ",".join(str(seed) for seed in COMPARISON_SEEDS),
            ],
            analysis_dir / "runner.log",
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
    write_json(state_path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
