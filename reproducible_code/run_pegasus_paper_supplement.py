#!/usr/bin/env python3
"""Run the frozen post-lock experiments required for the Pegasus paper."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from capacity_protocol import deterministic_capacity_assignment
from pegasus_paper_closure_protocol import (
    CAPACITY_MULTISET as P3_CAPACITY_MULTISET,
    CAPACITY_NAMESPACE as P3_CAPACITY_NAMESPACE,
)
from pegasus_paper_supplement_protocol import (
    ABLATION_METHODS,
    ABLATION_REFERENCE_METHOD,
    ABLATION_SEEDS,
    CACHE_BENCHMARK_REPEATS,
    CAPACITY_PROFILES,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    HETEROGENEITY_CAPACITY_NAMESPACE,
    HETEROGENEITY_METHODS,
    HETEROGENEITY_SEEDS,
    P3_FINAL_DIR,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    SCALING_EPISODES,
    SCALING_METHODS,
    SCALING_SEEDS,
    SCALING_USER_COUNTS,
    TASK_LIMIT_INCLUDING_DUMMY,
    canonical_hash,
    specification,
    supplement_source_hash,
    validate_protocol,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parent
LOCK_PATH = RESULT_ROOT / "SUPPLEMENT_LOCK.json"
HETEROGENEITY_DIR = RESULT_ROOT / "heterogeneity"
ABLATION_DIR = RESULT_ROOT / "ablation"
SCALING_DIR = RESULT_ROOT / "scaling"
ANALYSIS_DIR = RESULT_ROOT / "analysis"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "heterogeneity",
            "ablation",
            "scaling",
            "analysis",
            "all",
        ),
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


def run_logged(command, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(exist_ok=True)
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


def initialize_lock():
    spec = validate_protocol()
    lock = {
        "status": "locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "specification": spec,
        "specification_sha256": spec["specification_sha256"],
        "supplement_source_sha256": supplement_source_hash(),
        "completed_stages": [],
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        for key in (
            "specification_sha256",
            "supplement_source_sha256",
        ):
            if existing.get(key) != lock[key]:
                raise RuntimeError(f"Supplement lock mismatch: {key}")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def mark_stage(stage, details=None):
    lock = read_json(LOCK_PATH)
    stages = lock.setdefault("completed_stages", [])
    if stage not in stages:
        stages.append(stage)
    lock.setdefault("stage_records", {})[stage] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **(details or {}),
    }
    if stage == "analysis":
        lock["status"] = "complete"
    write_json(LOCK_PATH, lock)


def reproduction_command(
    suite_dir,
    seeds,
    labels,
    capacities,
    capacity_namespace,
    workers,
    seed_partition,
    revision_reason,
    resume,
):
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        "pegasus_paper_closure_converged",
        "--suite-dir",
        str(suite_dir),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--labels",
        ",".join(labels),
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
        capacity_namespace,
        "--workers",
        str(workers),
        "--revision-id",
        PROTOCOL_VERSION,
        "--revision-parent",
        "pegasus_paper_closure_v1",
        "--revision-reason",
        revision_reason,
        "--revision-changed-module",
        "none_frozen_evaluation_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_or_predeclared_gate_failure",
        "--seed-partition",
        seed_partition,
    ]
    if resume:
        command.append("--resume")
    return command


def evaluation_rows(run_dir):
    with (Path(run_dir) / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        return [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def workload_view(bank):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def infrastructure_view(snapshot):
    return {
        "servers": {
            key: {
                field: server[field]
                for field in ("position", "frequency", "load", "rate_to_cloud")
            }
            for key, server in snapshot["servers"].items()
        },
        "between_server_costs": snapshot["between_server_costs"],
        "service_data_length": snapshot["service_data_length"],
    }


def check_run(run_dir, expected_capacity, expected_episodes):
    summary = read_json(Path(run_dir) / "summary.json")
    config = read_json(Path(run_dir) / "config.json")["arguments"]
    rows = evaluation_rows(run_dir)
    bank = read_json(Path(run_dir) / "evaluation_scenarios.json")
    observed_capacity = {
        int(key): int(value)
        for key, value in summary["server_capacities"].items()
    }
    family_count = expected_episodes // len(FAMILIES)
    checks = {
        "complete_and_converged": bool(
            summary.get("status") == "complete"
            and summary.get("eligible_for_comparison")
            and summary.get("convergence", {}).get("reached")
        ),
        "dataset_exact": (
            summary.get("dag_dataset", {}).get("sha256")
            == EXPECTED_DATASET_SHA256
        ),
        "completion_protocol_exact": (
            summary.get("dag_completion_protocol_version")
            == DAG_COMPLETION_PROTOCOL_VERSION
        ),
        "capacity_exact": observed_capacity == expected_capacity,
        "environment_exact": bool(
            config.get("num_users") == 20
            and config.get("num_servers") == 10
            and config.get("num_services") == 10
            and config.get("num_tasks") == TASK_LIMIT_INCLUDING_DUMMY
            and config.get("bandwidth") == 15000
        ),
        "evaluation_frozen": bool(summary.get("evaluation_state_frozen")),
        "tasks_exact_once": bool(
            len(rows) == expected_episodes
            and all(
                int(row["real_task_count"])
                == int(row["completed_task_count"])
                and int(row["all_tasks_executed_once"]) == 1
                for row in rows
            )
        ),
        "families_balanced": (
            Counter(row.get("workflow_family") for row in bank)
            == Counter({family: family_count for family in FAMILIES})
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Run audit failed for {run_dir}: {checks}")
    return {
        "checks": checks,
        "workload": workload_view(bank),
        "infrastructure": infrastructure_view(
            read_json(Path(run_dir) / "scenario_initial.json")
        ),
    }


def check_heterogeneity():
    profile_records = {}
    cross_profile = {seed: {} for seed in HETEROGENEITY_SEEDS}
    for profile, capacities in CAPACITY_PROFILES.items():
        expected_budget = sum(capacities)
        profile_records[profile] = {}
        for seed in HETEROGENEITY_SEEDS:
            expected_capacity = deterministic_capacity_assignment(
                capacities,
                number_of_servers=10,
                number_of_services=10,
                seed=seed,
                assignment_namespace=HETEROGENEITY_CAPACITY_NAMESPACE,
            )
            references = []
            for label in HETEROGENEITY_METHODS:
                run_dir = (
                    HETEROGENEITY_DIR
                    / profile
                    / "runs"
                    / label
                    / f"seed_{seed}"
                )
                record = check_run(
                    run_dir,
                    expected_capacity,
                    EVALUATION_EPISODES,
                )
                references.append(record)
            if not all(
                record["workload"] == references[0]["workload"]
                and record["infrastructure"]
                == references[0]["infrastructure"]
                for record in references[1:]
            ):
                raise RuntimeError(
                    f"Unpaired heterogeneity methods: {profile} seed={seed}"
                )
            if sum(expected_capacity.values()) != expected_budget:
                raise RuntimeError("Capacity budget audit failed")
            profile_records[profile][seed] = {
                "capacity": expected_capacity,
                "workload": references[0]["workload"],
                "infrastructure": references[0]["infrastructure"],
            }
            cross_profile[seed][profile] = profile_records[profile][seed]
    for seed, profiles in cross_profile.items():
        values = list(profiles.values())
        if not all(
            value["workload"] == values[0]["workload"]
            and value["infrastructure"] == values[0]["infrastructure"]
            for value in values[1:]
        ):
            raise RuntimeError(
                f"Uniform/heterogeneous workloads are not paired: seed={seed}"
            )
    return {
        "all_runs_valid": True,
        "methods_paired": True,
        "profiles_paired_except_capacity": True,
        "equal_total_budget": True,
    }


def check_ablation():
    for seed in ABLATION_SEEDS:
        expected_capacity = deterministic_capacity_assignment(
            P3_CAPACITY_MULTISET,
            number_of_servers=10,
            number_of_services=10,
            seed=seed,
            assignment_namespace=P3_CAPACITY_NAMESPACE,
        )
        reference_dir = (
            P3_FINAL_DIR
            / "runs"
            / ABLATION_REFERENCE_METHOD
            / f"seed_{seed}"
        )
        reference = check_run(
            reference_dir,
            expected_capacity,
            EVALUATION_EPISODES,
        )
        for label in ABLATION_METHODS:
            run_dir = ABLATION_DIR / "runs" / label / f"seed_{seed}"
            record = check_run(
                run_dir,
                expected_capacity,
                EVALUATION_EPISODES,
            )
            if (
                record["workload"] != reference["workload"]
                or record["infrastructure"] != reference["infrastructure"]
            ):
                raise RuntimeError(
                    f"Ablation is not paired with frozen OUR: {label} seed={seed}"
                )
    return {
        "all_runs_valid": True,
        "paired_with_frozen_our": True,
        "ten_seed_complete": True,
    }


def run_tests():
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "test_pegasus_paper_supplement.py",
        "test_pegasus_paper_closure.py",
        "test_dag_completion_semantics.py",
        "test_capacity_protocol.py",
        "test_information_protocol.py",
    ]
    run_logged(command, RESULT_ROOT / "tests.log")


def run_heterogeneity(workers, resume):
    for profile, capacities in CAPACITY_PROFILES.items():
        suite_dir = HETEROGENEITY_DIR / profile
        run_logged(
            reproduction_command(
                suite_dir=suite_dir,
                seeds=HETEROGENEITY_SEEDS,
                labels=HETEROGENEITY_METHODS,
                capacities=capacities,
                capacity_namespace=HETEROGENEITY_CAPACITY_NAMESPACE,
                workers=workers,
                seed_partition="heterogeneity",
                revision_reason=f"fixed_budget_capacity_control_{profile}",
                resume=resume,
            ),
            suite_dir / "runner.log",
        )
    audit = check_heterogeneity()
    write_json(HETEROGENEITY_DIR / "integrity.json", audit)
    mark_stage("heterogeneity", audit)


def run_ablation(workers, resume):
    run_logged(
        reproduction_command(
            suite_dir=ABLATION_DIR,
            seeds=ABLATION_SEEDS,
            labels=ABLATION_METHODS,
            capacities=P3_CAPACITY_MULTISET,
            capacity_namespace=P3_CAPACITY_NAMESPACE,
            workers=workers,
            seed_partition="mechanism",
            revision_reason="ten_seed_primary_module_confirmation",
            resume=resume,
        ),
        ABLATION_DIR / "runner.log",
    )
    audit = check_ablation()
    write_json(ABLATION_DIR / "integrity.json", audit)
    mark_stage("ablation", audit)


def run_scaling(workers, resume):
    command = [
        sys.executable,
        str(ROOT / "evaluate_pegasus_user_scaling.py"),
        "--source-suite-dir",
        str(P3_FINAL_DIR),
        "--output-dir",
        str(SCALING_DIR),
        "--labels",
        ",".join(SCALING_METHODS),
        "--seeds",
        ",".join(str(seed) for seed in SCALING_SEEDS),
        "--user-counts",
        ",".join(str(value) for value in SCALING_USER_COUNTS),
        "--episodes",
        str(SCALING_EPISODES),
        "--workers",
        str(workers),
        "--cache-benchmark-repeats",
        str(CACHE_BENCHMARK_REPEATS),
    ]
    if resume:
        command.append("--resume")
    run_logged(command, SCALING_DIR / "runner.log")
    summary = read_json(SCALING_DIR / "user_scaling_summary.json")
    if not (
        summary.get("status") == "complete"
        and summary.get("all_methods_scenario_paired")
        and summary.get("reference_reproduction_audit", {}).get("all_exact")
    ):
        raise RuntimeError("Scaling integrity gate failed")
    mark_stage(
        "scaling",
        {
            "all_methods_scenario_paired": True,
            "twenty_user_reference_exact": True,
        },
    )


def run_analysis():
    command = [
        sys.executable,
        str(ROOT / "analyze_pegasus_paper_supplement.py"),
        "--result-root",
        str(RESULT_ROOT),
        "--output-dir",
        str(ANALYSIS_DIR),
    ]
    run_logged(command, ANALYSIS_DIR / "analysis.log")
    summary = read_json(ANALYSIS_DIR / "paper_supplement_summary.json")
    if summary.get("status") != "complete":
        raise RuntimeError("Supplement analysis is incomplete")
    mark_stage("analysis", {"gate": summary.get("gate")})


def main():
    args = parse_args()
    initialize_lock()
    if args.stage in ("tests", "all"):
        run_tests()
        mark_stage("tests", {"passed": True})
        if args.stage == "tests":
            print(f"Supplement artifacts: {RESULT_ROOT}")
            return
    if args.stage in ("heterogeneity", "all"):
        run_heterogeneity(args.workers, args.resume)
    if args.stage in ("ablation", "all"):
        run_ablation(args.workers, args.resume)
    if args.stage in ("scaling", "all"):
        run_scaling(args.workers, args.resume)
    if args.stage in ("analysis", "all"):
        run_analysis()
    print(f"Supplement artifacts: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
