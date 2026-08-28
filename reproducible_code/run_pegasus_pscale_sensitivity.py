#!/usr/bin/env python3
"""Run the locked B5/B10 sensitivity around the completed P-Scale B8 study."""

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

from capacity_protocol import deterministic_capacity_assignment
from pegasus_pscale_protocol import (
    CAPACITY_PROFILES,
    DATASET_PATH,
    DEVELOPMENT_SEEDS,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
    validate_protocol,
)
from run_a0_fixed_budget_heterogeneity import (
    ALGORITHM_SOURCE_FILES,
    source_hash,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parent
MAIN_ROOT = ROOT / "results/pegasus_pscale/p2/converged"
MAIN_LOCK = ROOT / "results/pegasus_pscale/p2/PSCALE_LOCK.json"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p2/sensitivity"
LOCK_PATH = RESULT_ROOT / "SENSITIVITY_LOCK.json"
METHODS = ("guided_full", "centralized_greedy_daoc", "lean_our")
SENSITIVITY_PROFILES = ("B5", "B10")
ALL_PROFILES = ("B5", "B8", "B10")
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
PROTOCOL_VERSION = "pegasus_pscale_p2_budget_sensitivity_v1"
SOURCE_FILES = ALGORITHM_SOURCE_FILES + (
    "analyze_pegasus_pscale_sensitivity.py",
    "evaluate_oracle_latency_bound.py",
    "oracle_latency_bound.py",
    "pegasus_pscale_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_pscale_sensitivity.py",
    "run_reproduction_suite.py",
    "test_pegasus_pscale_sensitivity.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "train", "oracle", "analysis", "all"),
        default="all",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(SENSITIVITY_PROFILES),
        help="Comma-separated subset of B5,B10",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.budgets = tuple(
        item.strip().upper() for item in args.budgets.split(",")
        if item.strip()
    )
    if not args.budgets or any(
        profile not in SENSITIVITY_PROFILES for profile in args.budgets
    ):
        raise ValueError("--budgets must be a subset of B5,B10")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("--budgets must not contain duplicates")
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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sensitivity_source_hash():
    digest = hashlib.sha256()
    for filename in sorted(SOURCE_FILES):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT / filename).read_bytes())
    return digest.hexdigest()


def protocol_spec():
    main_summary = read_json(
        MAIN_ROOT / "analysis/pegasus_pscale_summary.json"
    )
    if not main_summary.get("gate", {}).get("passed"):
        raise RuntimeError("B8 main gate did not pass")
    main_lock = read_json(MAIN_LOCK)
    return {
        "protocol_id": PROTOCOL_VERSION,
        "parent_protocol": "pegasus_pscale_p2",
        "parent_lock_sha256": sha256_file(MAIN_LOCK),
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "algorithm_retuned": False,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dag_completion_protocol_version": DAG_COMPLETION_PROTOCOL_VERSION,
        "seeds": list(DEVELOPMENT_SEEDS),
        "methods": list(METHODS),
        "profiles": {
            name: list(CAPACITY_PROFILES[name])
            for name in ALL_PROFILES
        },
        "new_training_profiles": list(SENSITIVITY_PROFILES),
        "reused_profile": "B8",
        "from_scratch": True,
        "paired_scenarios_per_seed": EVALUATION_EPISODES,
        "capacity_assignment_namespace": CAPACITY_NAMESPACE,
        "main_gate": main_summary["gate"],
        "parent_algorithm_source_sha256": main_lock["specification"][
            "current_algorithm_source_sha256"
        ],
        "ten_seed_recommendation_rule": {
            "integrity": "all_checks_true",
            "our_vs_daoc": "pass_at_B5_B8_B10",
            "our_vs_central": "pass_at_B5_B8_B10",
            "p95_vs_central": "pass_at_B5_B8_B10",
        },
    }


def initialize_lock():
    validate_protocol()
    specification = protocol_spec()
    lock = {
        "status": "locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "source_sha256": sensitivity_source_hash(),
        "algorithm_retuned": False,
    }
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        if existing.get("specification_sha256") != lock[
            "specification_sha256"
        ]:
            raise RuntimeError("Sensitivity specification changed after lock")
        if existing.get("source_sha256") != lock["source_sha256"]:
            raise RuntimeError("Sensitivity source changed after lock")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def run_logged(command, log_path):
    log_path = Path(log_path)
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


def suite_dir(profile):
    return RESULT_ROOT / profile


def reproduction_command(profile, workers, resume):
    capacities = CAPACITY_PROFILES[profile]
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        "pegasus_pscale_p2_converged",
        "--suite-dir",
        str(suite_dir(profile)),
        "--seeds",
        ",".join(str(seed) for seed in DEVELOPMENT_SEEDS),
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
        ",".join(str(value) for value in capacities),
        "--baseline-server-capacity",
        "3",
        "--capacity-assignment-namespace",
        CAPACITY_NAMESPACE,
        "--workers",
        str(workers),
        "--revision-id",
        f"{PROTOCOL_VERSION}_{profile.lower()}",
        "--revision-parent",
        "pegasus_pscale_p2",
        "--revision-reason",
        f"cache_budget_sensitivity_{profile.lower()}",
        "--revision-changed-module",
        "environment_cache_budget_only",
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


def evaluation_rows(run_dir):
    with (run_dir / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        return [
            row for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def paired_bank_view(bank):
    return [
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


def check_profile(profile):
    capacities = CAPACITY_PROFILES[profile]
    for seed in DEVELOPMENT_SEEDS:
        expected = deterministic_capacity_assignment(
            capacities,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=CAPACITY_NAMESPACE,
        )
        reference_bank = None
        for label in METHODS:
            directory = suite_dir(profile) / "runs" / label / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")["arguments"]
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete {profile} {label} seed={seed}")
            if not (
                summary.get("eligible_for_comparison")
                and summary.get("convergence", {}).get("reached")
            ):
                raise RuntimeError(f"Unconverged {profile} {label} seed={seed}")
            if summary.get("dag_completion_protocol_version") != (
                DAG_COMPLETION_PROTOCOL_VERSION
            ):
                raise RuntimeError("DAG completion protocol mismatch")
            if summary.get("dag_dataset", {}).get("sha256") != (
                EXPECTED_DATASET_SHA256
            ):
                raise RuntimeError("Dataset mismatch")
            observed = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            if observed != expected or sum(observed.values()) != int(
                profile[1:]
            ):
                raise RuntimeError(f"Capacity mismatch in {profile}")
            if config.get("bandwidth") != 15000 or config.get(
                "num_tasks"
            ) != TASK_LIMIT_INCLUDING_DUMMY:
                raise RuntimeError("Environment protocol mismatch")
            if not summary.get("evaluation_state_frozen"):
                raise RuntimeError("Evaluation state was not frozen")
            if (
                summary.get("evaluation_scenario_count")
                != EVALUATION_EPISODES
                or summary.get("evaluation_unique_base_scenarios")
                != EVALUATION_EPISODES
            ):
                raise RuntimeError("Effective evaluation count mismatch")
            rows = evaluation_rows(directory)
            if len(rows) != EVALUATION_EPISODES or not all(
                int(row["real_task_count"])
                == int(row["completed_task_count"])
                and int(row["all_tasks_executed_once"]) == 1
                for row in rows
            ):
                raise RuntimeError("Task execution audit failed")
            bank = read_json(directory / "evaluation_scenarios.json")
            counts = Counter(row.get("workflow_family") for row in bank)
            if counts != Counter({family: 20 for family in FAMILIES}):
                raise RuntimeError("Workflow family balance mismatch")
            current_bank = paired_bank_view(bank)
            if reference_bank is None:
                reference_bank = current_bank
            elif current_bank != reference_bank:
                raise RuntimeError("Methods do not share paired scenarios")


def run_tests():
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "test_pegasus_pscale_sensitivity.py",
    ]
    run_logged(command, RESULT_ROOT / "tests.log")


def run_training(profile, workers, resume):
    directory = suite_dir(profile)
    run_logged(
        reproduction_command(profile, workers, resume),
        directory / "runner.log",
    )
    check_profile(profile)


def run_oracle(profile):
    directory = suite_dir(profile)
    command = [
        sys.executable,
        str(ROOT / "evaluate_oracle_latency_bound.py"),
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
        ",".join(str(seed) for seed in DEVELOPMENT_SEEDS),
        "--episodes",
        str(EVALUATION_EPISODES),
        "--exact-check-scenarios",
        "0",
    ]
    run_logged(command, directory / "oracle.log")


def run_analysis():
    command = [
        sys.executable,
        str(ROOT / "analyze_pegasus_pscale_sensitivity.py"),
        "--sensitivity-dir",
        str(RESULT_ROOT),
        "--main-suite-dir",
        str(MAIN_ROOT),
        "--output-dir",
        str(RESULT_ROOT / "analysis"),
    ]
    run_logged(command, RESULT_ROOT / "analysis.log")


def main():
    args = parse_args()
    if args.stage in ("tests", "all"):
        run_tests()
        if args.stage == "tests":
            print(f"P-Scale sensitivity artifacts: {RESULT_ROOT}")
            return

    initialize_lock()
    if args.stage in ("train", "all"):
        for profile in args.budgets:
            run_training(profile, args.workers, args.resume)
    if args.stage in ("oracle", "all"):
        for profile in args.budgets:
            check_profile(profile)
            run_oracle(profile)
    if args.stage in ("analysis", "all"):
        for profile in SENSITIVITY_PROFILES:
            check_profile(profile)
        run_analysis()
        summary = read_json(
            RESULT_ROOT / "analysis/pegasus_pscale_sensitivity_summary.json"
        )
        lock = read_json(LOCK_PATH)
        lock["status"] = "complete"
        lock["completed_at"] = datetime.now(timezone.utc).isoformat()
        lock["sensitivity_gate"] = summary["gate"]
        lock["ten_seed_recommended"] = bool(summary["gate"]["passed"])
        write_json(LOCK_PATH, lock)
    print(f"P-Scale sensitivity artifacts: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
