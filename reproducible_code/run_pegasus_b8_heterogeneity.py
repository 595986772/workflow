#!/usr/bin/env python3
"""Run the frozen three-seed Pegasus-B8 heterogeneity experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import run_reproduction_suite as reproduction
from pegasus_b8_heterogeneity_protocol import (
    CAPACITY_NAMESPACE,
    CAPACITY_PROFILES,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    METHODS,
    NEW_PROFILES,
    PROFILE_NAME,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    RUN_ROOT,
    SEEDS,
    TASK_LIMIT_INCLUDING_DUMMY,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
STATE_PATH = RESULT_ROOT / "RUN_STATE.json"
LOCK_PATH = RESULT_ROOT / "PROTOCOL_LOCK.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--profiles",
        default=",".join(NEW_PROFILES),
        help="Comma-separated subset of H0,H1,H3.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def comparable_bank(run_dir):
    rows = json.loads(
        (Path(run_dir) / "evaluation_scenarios.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        (
            row["episode"],
            row["seed"],
            row["base_fingerprint"],
            row.get("workflow_family"),
        )
        for row in rows
    ]


def verify_suite(profile_name):
    suite_dir = RUN_ROOT / profile_name
    reference_banks = {}
    reference_capacities = {}
    run_audit = {}
    for method in METHODS:
        run_audit[method] = {}
        for seed in SEEDS:
            run_dir = suite_dir / "runs" / method / f"seed_{seed}"
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            config = json.loads(
                (run_dir / "config.json").read_text(encoding="utf-8")
            )
            arguments = config["arguments"]
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete {profile_name}/{method}/{seed}")
            if not summary.get("eligible_for_comparison"):
                raise RuntimeError(f"Ineligible {profile_name}/{method}/{seed}")
            if not summary.get("convergence", {}).get("reached"):
                raise RuntimeError(f"Unconverged {profile_name}/{method}/{seed}")
            if arguments.get("dag_dataset_sha256") != EXPECTED_DATASET_SHA256:
                raise RuntimeError("Dataset mismatch")
            if arguments.get("capacity_assignment_namespace") != (
                CAPACITY_NAMESPACE
            ):
                raise RuntimeError("Capacity namespace mismatch")
            expected = tuple(sorted(CAPACITY_PROFILES[profile_name]))
            actual = tuple(sorted(summary["server_capacities"].values()))
            if actual != expected:
                raise RuntimeError("Capacity profile mismatch")
            capacities = summary["server_capacities"]
            if seed not in reference_capacities:
                reference_capacities[seed] = capacities
            elif capacities != reference_capacities[seed]:
                raise RuntimeError("Methods received different capacities")
            bank = comparable_bank(run_dir)
            if seed not in reference_banks:
                reference_banks[seed] = bank
            elif bank != reference_banks[seed]:
                raise RuntimeError("Methods received different scenarios")
            run_audit[method][str(seed)] = {
                "config_sha256": summary["experiment_config_sha256"],
                "checkpoint_sha256": summary["selected_checkpoint_sha256"],
                "convergence_episode": summary["convergence"]["episode"],
                "total_wall_time_sec": summary["total_wall_time_sec"],
            }
    return run_audit


def run_profile(profile_name, workers, resume, keep_going):
    suite_dir = RUN_ROOT / profile_name
    argv = [
        "run_reproduction_suite.py",
        "--profile",
        PROFILE_NAME,
        "--suite-dir",
        str(suite_dir),
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
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
        "--eval-episodes",
        str(EVALUATION_EPISODES),
        "--server-capacity",
        "1",
        "--server-capacity-multiset",
        ",".join(str(value) for value in CAPACITY_PROFILES[profile_name]),
        "--baseline-server-capacity",
        "3",
        "--capacity-assignment-namespace",
        CAPACITY_NAMESPACE,
        "--workers",
        str(workers),
        "--revision-id",
        PROTOCOL_VERSION,
        "--revision-parent",
        "pegasus_pscale_p2",
        "--revision-reason",
        "fixed_budget_cache_heterogeneity_sensitivity",
        "--revision-changed-module",
        "environment_capacity_profile_only",
        "--revision-expected-metric",
        "robust_paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "dcc_or_pairwise_gain_not_robust_across_profiles",
        "--seed-partition",
        "heterogeneity",
    ]
    if resume:
        argv.append("--resume")
    if keep_going:
        argv.append("--keep-going")
    previous = sys.argv
    try:
        sys.argv = argv
        reproduction.main()
    finally:
        sys.argv = previous
    return verify_suite(profile_name)


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    profiles = tuple(
        value.strip() for value in args.profiles.split(",") if value.strip()
    )
    unknown = set(profiles) - set(NEW_PROFILES)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")

    protocol = validate_protocol(require_h2=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (
                ROOT / "pegasus_b8_heterogeneity_protocol.py",
                ROOT / "run_pegasus_b8_heterogeneity.py",
                ROOT / "run_independent_experiment.py",
                ROOT / "run_reproduction_suite.py",
                ROOT / "critical_path_cache.py",
                ROOT / "critical_path_rl.py",
            )
        },
    }
    if LOCK_PATH.exists():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if existing.get("protocol") != lock["protocol"]:
            raise RuntimeError("Frozen protocol changed; use a new result root")
    else:
        write_json(LOCK_PATH, lock)

    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "profiles": list(profiles),
        "completed_profiles": [],
        "run_audit": {},
    }
    write_json(STATE_PATH, state)
    for profile_name in profiles:
        state["run_audit"][profile_name] = run_profile(
            profile_name,
            workers=args.workers,
            resume=args.resume,
            keep_going=args.keep_going,
        )
        state["completed_profiles"].append(profile_name)
        write_json(STATE_PATH, state)
    state["status"] = "complete"
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(STATE_PATH, state)
    print(f"Completed: {', '.join(profiles)}")


if __name__ == "__main__":
    main()
