#!/usr/bin/env python3
"""Run paired nested DAOC environment stress experiments."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DAOC_LABEL = "guided_full"
OUR_LABEL = "lean_our"


ENVIRONMENTS = {
    "e0_original": {
        "dag_depth_increment": 0,
        "dependency_data_scale": 1.0,
        "server_capacity": 2,
    },
    "e_dag": {
        "dag_depth_increment": 2,
        "dependency_data_scale": 1.0,
        "server_capacity": 2,
    },
    "e_comm": {
        "dag_depth_increment": 0,
        "dependency_data_scale": 2.0,
        "server_capacity": 2,
    },
    "e_cache": {
        "dag_depth_increment": 0,
        "dependency_data_scale": 1.0,
        "server_capacity": 1,
    },
    "e_dag_cache": {
        "dag_depth_increment": 2,
        "dependency_data_scale": 1.0,
        "server_capacity": 1,
    },
    "e_joint": {
        "dag_depth_increment": 2,
        "dependency_data_scale": 2.0,
        "server_capacity": 1,
    },
}

STAGES = {
    "smoke": {
        "profile": "strict_stress_smoke",
        "train_episodes": 200,
        "eval_episodes": 20,
        "seeds": [1],
    },
    "screen": {
        "profile": "strict_stress_smoke",
        "train_episodes": 2000,
        "eval_episodes": 50,
        "seeds": [1, 2, 3],
    },
    "converged": {
        "profile": "strict_stress_converged_3seed",
        "train_episodes": 50000,
        "eval_episodes": 100,
        "seeds": [1, 2, 3],
    },
}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run strictly paired DAOC environment stresses."
    )
    parser.add_argument("--stage", choices=STAGES, default="smoke")
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument(
        "--environments",
        type=parse_csv,
        default=list(ENVIRONMENTS),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--daoc-label", default=DAOC_LABEL)
    parser.add_argument("--our-label", default=OUR_LABEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    unknown = set(args.environments) - set(ENVIRONMENTS)
    if unknown:
        raise ValueError(f"Unknown environments: {sorted(unknown)}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def run_command(command, root_dir, env):
    print(" ".join(str(item) for item in command), flush=True)
    subprocess.run(
        command,
        cwd=root_dir,
        env=env,
        check=True,
    )


def write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    stage = STAGES[args.stage]
    suite_dir = (
        args.suite_dir
        if args.suite_dir is not None
        else root_dir / "results" / f"strict_environment_{args.stage}"
    ).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = suite_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(matplotlib_dir)
    manifest_path = suite_dir / "strict_suite_manifest.json"
    manifest = {
        "status": "running",
        "stage": args.stage,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "stage_config": stage,
        "daoc_label": args.daoc_label,
        "our_label": args.our_label,
        "environments": {
            name: ENVIRONMENTS[name]
            for name in args.environments
        },
        "completed_environments": [],
    }
    write_manifest(manifest_path, manifest)
    started = time.perf_counter()
    seed_list = ",".join(str(seed) for seed in stage["seeds"])

    for environment_name in args.environments:
        stress = ENVIRONMENTS[environment_name]
        environment_dir = suite_dir / environment_name
        reproduction_command = [
            sys.executable,
            str(root_dir / "run_reproduction_suite.py"),
            "--profile",
            stage["profile"],
            "--suite-dir",
            str(environment_dir),
            "--seeds",
            seed_list,
            "--train-episodes",
            str(stage["train_episodes"]),
            "--eval-episodes",
            str(stage["eval_episodes"]),
            "--workers",
            str(args.workers),
            "--labels",
            f"{args.daoc_label},{args.our_label}",
            "--dag-depth-increment",
            str(stress["dag_depth_increment"]),
            "--dependency-data-scale",
            str(stress["dependency_data_scale"]),
            "--server-capacity",
            str(stress["server_capacity"]),
        ]
        if args.resume:
            reproduction_command.append("--resume")
        run_command(reproduction_command, root_dir, env)

        oracle_command = [
            sys.executable,
            str(root_dir / "evaluate_oracle_latency_bound.py"),
            "--our-suite-dir",
            str(environment_dir),
            "--our-label",
            args.our_label,
            "--daoc-suite-dir",
            str(environment_dir),
            "--daoc-label",
            args.daoc_label,
            "--output-dir",
            str(environment_dir / "oracle"),
            "--seeds",
            seed_list,
            "--episodes",
            str(stage["eval_episodes"]),
            "--exact-check-scenarios",
            (
                "1"
                if environment_name in {"e_dag_cache", "e_joint"}
                else "0"
            ),
        ]
        run_command(oracle_command, root_dir, env)
        manifest["completed_environments"].append(environment_name)
        write_manifest(manifest_path, manifest)

    analysis_command = [
        sys.executable,
        str(root_dir / "analyze_strict_environment_suite.py"),
        "--suite-dir",
        str(suite_dir),
        "--environments",
        ",".join(args.environments),
        "--seeds",
        seed_list,
        "--daoc-label",
        args.daoc_label,
        "--our-label",
        args.our_label,
    ]
    run_command(analysis_command, root_dir, env)
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    manifest["total_wall_time_sec"] = time.perf_counter() - started
    write_manifest(manifest_path, manifest)
    print(f"Strict environment suite: {suite_dir}")


if __name__ == "__main__":
    main()
