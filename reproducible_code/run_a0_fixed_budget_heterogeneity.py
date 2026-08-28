#!/usr/bin/env python3
"""Run the A0 fixed-budget cache-heterogeneity development study."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from a0_coordination_protocol import (
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
)
from a0_fixed_budget_heterogeneity_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    DEVELOPMENT_SEEDS,
    METHOD_LABELS,
    PROFILE_ORDER,
    PROTOCOL_VERSION,
    TOTAL_CACHE_BUDGET,
    capacity_text,
    frozen_protocol_spec,
    validate_protocol,
)
from capacity_protocol import deterministic_capacity_assignment


ROOT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = ROOT_DIR / "results" / "a0_fixed_budget_heterogeneity"
A0_FREEZE_PATH = (
    ROOT_DIR
    / "results"
    / "a0_cache_coordination"
    / "a0r2"
    / "FROZEN_ALGORITHM.json"
)
H8V0_MANIFEST_PATH = RESULT_ROOT / "h8v0" / "RUN_MANIFEST.json"
H8V0_DIAGNOSTIC_PATH = (
    RESULT_ROOT
    / "h8v0"
    / "experiment"
    / "coverage_repair_counterfactual"
    / "coverage_counterfactual_summary.json"
)
ALGORITHM_SOURCE_FILES = (
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
PROTOCOL_SOURCE_FILES = ALGORITHM_SOURCE_FILES + (
    "a0_coordination_protocol.py",
    "a0_fixed_budget_heterogeneity_protocol.py",
    "analyze_a0_fixed_budget_heterogeneity.py",
    "diagnose_a0_coverage_counterfactual.py",
    "evaluate_oracle_latency_bound.py",
    "oracle_latency_bound.py",
    "run_a0_fixed_budget_heterogeneity.py",
    "run_independent_experiment.py",
    "run_reproduction_suite.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "experiment", "all"),
        default="all",
    )
    parser.add_argument("--revision-id", default="h8v0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def source_hash(files):
    digest = hashlib.sha256()
    for filename in sorted(set(files)):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT_DIR / filename).read_bytes())
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def workload_scenario_view(bank):
    """Remove the expected capacity-dependent scenario fingerprint."""
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in bank
    ]


def run_logged(command, log_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(item) for item in command), flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def verify_revision_parent(revision_id):
    if not A0_FREEZE_PATH.exists():
        raise RuntimeError("Missing the frozen a0r2 algorithm record")
    freeze = read_json(A0_FREEZE_PATH)
    current = source_hash(ALGORITHM_SOURCE_FILES)
    if revision_id == "h8v0":
        if freeze.get("algorithm_source_sha256") != current:
            raise RuntimeError("Algorithm source changed after the a0r2 freeze")
        return freeze, None
    if revision_id != "h8v1":
        raise ValueError("Only governed revisions h8v0 and h8v1 are supported")
    if not H8V0_MANIFEST_PATH.exists():
        raise RuntimeError("Missing completed h8v0 parent manifest")
    parent = read_json(H8V0_MANIFEST_PATH)
    if parent.get("status") != "complete":
        raise RuntimeError("h8v0 parent is not complete")
    if parent.get("algorithm_source_sha256") != freeze.get(
        "algorithm_source_sha256"
    ):
        raise RuntimeError("h8v0 does not match the frozen a0r2 parent")
    if not H8V0_DIAGNOSTIC_PATH.exists():
        raise RuntimeError("Missing coverage-repair counterfactual diagnosis")
    diagnostic = read_json(H8V0_DIAGNOSTIC_PATH)
    if diagnostic.get("status") != "complete":
        raise RuntimeError("Coverage-repair diagnosis is incomplete")
    if current == parent.get("algorithm_source_sha256"):
        raise RuntimeError("h8v1 must contain the registered cache revision")
    return freeze, parent


def revision_spec(revision_id):
    spec = copy.deepcopy(frozen_protocol_spec())
    if revision_id == "h8v1":
        spec["training"]["source_algorithm_freeze"] = "h8v0"
        spec["revision"] = {
            "parent": "h8v0",
            "changed_module": (
                "scarcity_aware_service_coverage_constraint"
            ),
            "change": (
                "repair redundant replicas only when observed-service "
                "coverage is below the feasible maximum"
            ),
            "future_information_used": False,
            "counterfactual_diagnostic_only": True,
        }
        spec["training"]["reused_parent_baselines"] = [
            "guided_full",
            "centralized_greedy_daoc",
        ]
    return spec


def revision_metadata(revision_id):
    if revision_id == "h8v1":
        return {
            "parent": "h8v0",
            "reason": "repair_duplicate_replica_tail_latency",
            "changed_module": (
                "scarcity_aware_service_coverage_constraint"
            ),
            "expected_metric": (
                "our_beats_central_mean_and_p95_across_capacity_profiles"
            ),
            "rejection_condition": (
                "strong_profile_p95_regression_or_nonpositive_central_gain"
            ),
        }
    return {
        "parent": "a0r2",
        "reason": "fixed_budget_heterogeneity_mechanism_test",
        "changed_module": "environment_capacity_distribution_only",
        "expected_metric": (
            "our_central_advantage_increases_with_capacity_variance"
        ),
        "rejection_condition": (
            "nonpositive_advantage_slope_or_strong_profile_p95_regression"
        ),
    }


def reproduction_command(
    directory,
    profile,
    args,
    training_profile="e2_converged",
    seeds=DEVELOPMENT_SEEDS,
    partition="mechanism",
    labels=METHOD_LABELS,
):
    revision = revision_metadata(args.revision_id)
    command = [
        sys.executable,
        str(ROOT_DIR / "run_reproduction_suite.py"),
        "--profile",
        training_profile,
        "--suite-dir",
        str(directory),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--labels",
        ",".join(labels),
        "--dag-dataset-path",
        str(DATASET_PATH),
        "--dag-dataset-sha256",
        EXPECTED_DATASET_SHA256,
        "--server-capacity",
        "1",
        "--server-capacity-multiset",
        capacity_text(profile),
        "--baseline-server-capacity",
        str(BASELINE_RANDOM_DRAW_CAPACITY),
        "--capacity-assignment-namespace",
        CAPACITY_ASSIGNMENT_NAMESPACE,
        "--workers",
        str(args.workers),
        "--revision-id",
        args.revision_id,
        "--revision-parent",
        revision["parent"],
        "--revision-reason",
        revision["reason"],
        "--revision-changed-module",
        revision["changed_module"],
        "--revision-expected-metric",
        revision["expected_metric"],
        "--revision-rejection-condition",
        revision["rejection_condition"],
        "--seed-partition",
        partition,
    ]
    if args.resume:
        command.append("--resume")
    return command


def reuse_parent_baselines(directory, profile):
    labels = ("guided_full", "centralized_greedy_daoc")
    source_profile = RESULT_ROOT / "h8v0" / "experiment" / profile
    runs_directory = directory / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for label in labels:
        source = source_profile / "runs" / label
        if not source.exists():
            raise RuntimeError(f"Missing h8v0 baseline: {source}")
        for seed in DEVELOPMENT_SEEDS:
            summary = read_json(source / f"seed_{seed}" / "summary.json")
            config = read_json(source / f"seed_{seed}" / "config.json")
            if not (
                summary.get("status") == "complete"
                and summary.get("convergence", {}).get("reached") is True
                and summary.get("eval", {}).get("episodes") == 100
                and config.get("arguments", {}).get(
                    "cache_coverage_constraint",
                    False,
                )
                is False
            ):
                raise RuntimeError(
                    f"Ineligible h8v0 baseline: {label} seed={seed}"
                )
        target = runs_directory / label
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Wrong baseline link: {target}")
        elif target.exists():
            shutil.rmtree(target)
            target.symlink_to(source.resolve(), target_is_directory=True)
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)
        records.append(
            {
                "label": label,
                "source": str(source.resolve()),
                "target": str(target),
                "seeds": list(DEVELOPMENT_SEEDS),
                "reason": "algorithm_and_environment_path_unchanged",
            }
        )
    write_json(
        directory / "REUSED_PARENT_BASELINES.json",
        {
            "status": "complete",
            "parent_revision": "h8v0",
            "records": records,
        },
    )


def oracle_command(directory):
    return [
        sys.executable,
        str(ROOT_DIR / "evaluate_oracle_latency_bound.py"),
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
        "100",
        "--exact-check-scenarios",
        "0",
    ]


def run_tests(directory):
    run_logged(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-p",
            "test_*.py",
            "-v",
        ],
        directory / "tests.log",
    )
    write_json(directory / "tests_summary.json", {"status": "complete"})


def run_smoke(revision_root, args):
    smoke_root = revision_root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    checks = []
    scenario_banks = []
    for profile in PROFILE_ORDER:
        directory = smoke_root / profile
        directory.mkdir(exist_ok=True)
        run_logged(
            reproduction_command(
                directory,
                profile,
                args,
                training_profile="e2_smoke",
                seeds=(1,),
                partition="smoke",
            ),
            directory / "runner.log",
        )
        expected = deterministic_capacity_assignment(
            CAPACITY_PROFILES[profile],
            number_of_servers=10,
            number_of_services=10,
            seed=1,
            assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
        )
        for label in METHOD_LABELS:
            run = directory / "runs" / label / "seed_1"
            summary = read_json(run / "summary.json")
            bank = read_json(run / "evaluation_scenarios.json")
            capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            scenario_banks.append(workload_scenario_view(bank))
            checks.append(
                {
                    "profile": profile,
                    "label": label,
                    "complete": summary.get("status") == "complete",
                    "dataset_hash": summary.get("dag_dataset", {}).get(
                        "sha256"
                    )
                    == EXPECTED_DATASET_SHA256,
                    "capacity_assignment_exact": capacities == expected,
                    "total_budget_exact": (
                        sum(capacities.values()) == TOTAL_CACHE_BUDGET
                    ),
                    "evaluation_scenarios": len(bank) == 20,
                }
            )
    paired = all(bank == scenario_banks[0] for bank in scenario_banks[1:])
    passed = bool(
        paired
        and all(
            all(
                value
                for key, value in row.items()
                if key not in {"profile", "label"}
            )
            for row in checks
        )
    )
    result = {
        "status": "complete",
        "checks": checks,
        "all_profiles_base_scenario_paired": paired,
        "passed": passed,
    }
    write_json(smoke_root / "SMOKE_AUDIT.json", result)
    if not passed:
        raise RuntimeError("Fixed-budget heterogeneity smoke failed")
    return result


def run_experiment(revision_root, args):
    experiment = revision_root / "experiment"
    experiment.mkdir(parents=True, exist_ok=True)
    for profile in PROFILE_ORDER:
        directory = experiment / profile
        directory.mkdir(exist_ok=True)
        labels = METHOD_LABELS
        if args.revision_id == "h8v1":
            reuse_parent_baselines(directory, profile)
            labels = ("lean_our",)
        run_logged(
            reproduction_command(
                directory,
                profile,
                args,
                labels=labels,
            ),
            directory / "runner.log",
        )
        run_logged(oracle_command(directory), directory / "oracle.log")
    output = experiment / "analysis"
    output.mkdir(exist_ok=True)
    run_logged(
        [
            sys.executable,
            str(ROOT_DIR / "analyze_a0_fixed_budget_heterogeneity.py"),
            "--suite-root",
            str(experiment),
            "--output-dir",
            str(output),
        ],
        output / "analysis.log",
    )
    return read_json(output / "heterogeneity_summary.json")


def freeze_revision(revision_root, args, hashes, summary):
    if not (
        args.revision_id == "h8v1"
        and summary is not None
        and summary.get("gate", {}).get("passed") is True
    ):
        return None
    summary_path = (
        revision_root
        / "experiment"
        / "analysis"
        / "heterogeneity_summary.json"
    )
    freeze = {
        "status": "frozen",
        "revision_id": "h8v1",
        "parent_revision": "h8v0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **hashes,
        "result_summary_sha256": file_sha256(summary_path),
        "changed_module": (
            "scarcity_aware_service_coverage_constraint"
        ),
        "development_gate": summary["gate"],
        "trained_our_runs": 9,
        "reused_parent_baselines": [
            "guided_full",
            "centralized_greedy_daoc",
        ],
        "claim_scope": "A0_controlled_mechanism_development_only",
        "formal_final_seeds_run": False,
        "algorithm_source_files": list(ALGORITHM_SOURCE_FILES),
    }
    path = revision_root / "FROZEN_ALGORITHM.json"
    write_json(path, freeze)
    return path


def main():
    args = parse_args()
    validate_protocol()
    freeze, parent = verify_revision_parent(args.revision_id)
    specification = revision_spec(args.revision_id)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    revision_root = RESULT_ROOT / args.revision_id
    revision_root.mkdir(parents=True, exist_ok=True)
    hashes = {
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "protocol_source_sha256": source_hash(PROTOCOL_SOURCE_FILES),
        "configuration_sha256": canonical_hash(specification),
    }
    write_json(
        revision_root / "PROTOCOL.json",
        {
            "status": "active",
            "protocol_version": PROTOCOL_VERSION,
            "source_a0_freeze": str(A0_FREEZE_PATH),
            "source_a0_algorithm_sha256": freeze[
                "algorithm_source_sha256"
            ],
            "source_parent_manifest": (
                str(H8V0_MANIFEST_PATH) if parent is not None else None
            ),
            **hashes,
            "specification": specification,
        },
    )
    stages = (
        ("tests", "smoke", "experiment")
        if args.stage == "all"
        else (args.stage,)
    )
    started = time.perf_counter()
    manifest_path = revision_root / "RUN_MANIFEST.json"
    write_json(
        manifest_path,
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            **hashes,
        },
    )
    try:
        for stage in stages:
            if stage == "tests":
                directory = revision_root / "tests"
                directory.mkdir(exist_ok=True)
                if not (
                    args.resume and (directory / "tests_summary.json").exists()
                ):
                    run_tests(directory)
            elif stage == "smoke":
                smoke_audit = revision_root / "smoke" / "SMOKE_AUDIT.json"
                if not (
                    args.resume
                    and smoke_audit.exists()
                    and read_json(smoke_audit).get("passed") is True
                ):
                    run_smoke(revision_root, args)
            else:
                run_experiment(revision_root, args)
        summary_path = (
            revision_root
            / "experiment"
            / "analysis"
            / "heterogeneity_summary.json"
        )
        summary = read_json(summary_path) if summary_path.exists() else None
    except Exception as error:
        manifest = read_json(manifest_path)
        manifest.update(
            {
                "status": "failed",
                "error": repr(error),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json(manifest_path, manifest)
        raise
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "complete",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "wall_time_sec": time.perf_counter() - started,
            "development_gate": summary.get("gate") if summary else None,
        }
    )
    write_json(manifest_path, manifest)
    freeze_revision(revision_root, args, hashes, summary)
    print(f"Fixed-budget artifacts: {revision_root}", flush=True)


if __name__ == "__main__":
    main()
