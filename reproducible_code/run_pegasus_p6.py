#!/usr/bin/env python3
"""Run Pegasus-B8 standard baselines and mechanism-level ablations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pegasus_p6_protocol import (
    ANALYSIS_DIR,
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_SEEDS,
    HEURISTIC_DIR,
    HEURISTIC_METHODS,
    LEARNING_DIR,
    LEARNING_METHODS,
    P3_FINAL_DIR,
    P4_ABLATION_DIR,
    P5_SAC_DIR,
    PROTOCOL_VERSION,
    REFERENCE_METHODS,
    RESULT_ROOT,
    SMOKE_DIR,
    SMOKE_SEEDS,
    TASK_LIMIT_INCLUDING_DUMMY,
    sha256_file,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_LOCK = RESULT_ROOT / "PROTOCOL_LOCK.json"
REUSE_AUDIT = RESULT_ROOT / "REFERENCE_REUSE_AUDIT.json"
FINAL_LOCK = RESULT_ROOT / "FINAL_LOCK.json"
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
    "analyze_pegasus_p6.py",
    "pegasus_p6_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_p6.py",
    "run_reproduction_suite.py",
    "test_pegasus_p6.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "smoke",
            "heuristics",
            "learning",
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


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
                raise RuntimeError(f"P6 protocol changed after initialization: {key}")
        return existing
    write_json(PROTOCOL_LOCK, lock)
    return lock


def run_logged(command, log_path):
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


def suite_command(
    profile,
    suite_dir,
    seeds,
    labels,
    workers,
    seed_partition,
    resume,
):
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        profile,
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
        "pegasus_paper_closure_v1",
        "--revision-reason",
        "standard_cache_baselines_and_isolated_mechanism_ablations",
        "--revision-changed-module",
        "new_baselines_and_ablation_switches_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_unconverged_learning_method",
        "--seed-partition",
        seed_partition,
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


def assert_suite_complete(suite_dir, methods, seeds, evaluations, convergence):
    manifest = read_json(suite_dir / "suite_manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete suite: {suite_dir}")
    for seed in seeds:
        reference_bank = None
        for label in methods:
            directory = suite_dir / "runs" / label / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {directory}")
            if convergence and not (
                summary.get("eligible_for_comparison", False)
                and summary.get("convergence", {}).get("reached", False)
            ):
                raise RuntimeError(f"Unconverged run: {directory}")
            if summary.get("evaluation_scenario_count") != evaluations:
                raise RuntimeError(f"Wrong evaluation count: {directory}")
            capacities = sorted(
                int(value)
                for value in summary["server_capacities"].values()
            )
            if capacities != sorted(CAPACITY_MULTISET):
                raise RuntimeError(f"Capacity mismatch: {directory}")
            bank = comparable_bank(directory / "evaluation_scenarios.json")
            if reference_bank is None:
                reference_bank = bank
            elif bank != reference_bank:
                raise RuntimeError(f"Unpaired scenario bank: {directory}")


def run_tests():
    tests = (
        "test_critical_path_rl.py",
        "test_critical_path_cache.py",
        "test_information_protocol.py",
        "test_capacity_protocol.py",
        "test_dag_completion_semantics.py",
        "test_pegasus_pscale_protocol.py",
        "test_pegasus_p6.py",
    )
    run_logged(
        [sys.executable, "-m", "unittest", "-v", *tests],
        RESULT_ROOT / "tests.log",
    )


def run_smoke(workers, resume):
    methods = HEURISTIC_METHODS + LEARNING_METHODS
    run_logged(
        suite_command(
            "pegasus_p6_smoke",
            SMOKE_DIR,
            SMOKE_SEEDS,
            methods,
            workers,
            "smoke",
            resume,
        ),
        SMOKE_DIR / "runner.log",
    )
    assert_suite_complete(
        SMOKE_DIR,
        methods,
        SMOKE_SEEDS,
        evaluations=20,
        convergence=False,
    )


def run_heuristics(workers, resume):
    run_logged(
        suite_command(
            "pegasus_p6_heuristics",
            HEURISTIC_DIR,
            FINAL_SEEDS,
            HEURISTIC_METHODS,
            workers,
            "baseline_extension",
            resume,
        ),
        HEURISTIC_DIR / "runner.log",
    )
    assert_suite_complete(
        HEURISTIC_DIR,
        HEURISTIC_METHODS,
        FINAL_SEEDS,
        evaluations=EVALUATION_EPISODES,
        convergence=False,
    )


def run_learning(workers, resume):
    run_logged(
        suite_command(
            "pegasus_p6_learning_converged",
            LEARNING_DIR,
            FINAL_SEEDS,
            LEARNING_METHODS,
            workers,
            "confirmation",
            resume,
        ),
        LEARNING_DIR / "runner.log",
    )
    assert_suite_complete(
        LEARNING_DIR,
        LEARNING_METHODS,
        FINAL_SEEDS,
        evaluations=EVALUATION_EPISODES,
        convergence=True,
    )


def reference_directory(label, seed):
    if label in {"daoc_paper", "centralized_greedy_daoc", "lean_our"}:
        suite = P3_FINAL_DIR
    elif label == "our_no_coord_cache":
        suite = P4_ABLATION_DIR
    elif label == "coord_cache_discrete_sac":
        suite = P5_SAC_DIR
    else:
        raise KeyError(label)
    return suite / "runs" / label / f"seed_{seed}"


def audit_reference_reuse():
    records = []
    all_exact = True
    for seed in FINAL_SEEDS:
        reference_bank = None
        for label in REFERENCE_METHODS:
            directory = reference_directory(label, seed)
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            bank = comparable_bank(directory / "evaluation_scenarios.json")
            arguments = config["arguments"]
            exact = (
                summary.get("status") == "complete"
                and summary.get("eligible_for_comparison", False)
                and summary.get("evaluation_scenario_count")
                == EVALUATION_EPISODES
                and arguments.get("dag_dataset_sha256")
                == EXPECTED_DATASET_SHA256
                and arguments.get("bandwidth") == 15000
                and arguments.get("num_users") == 20
                and arguments.get("num_servers") == 10
                and arguments.get("num_services") == 10
                and arguments.get("num_tasks")
                == TASK_LIMIT_INCLUDING_DUMMY
                and sorted(arguments.get("server_capacity_multiset", []))
                == sorted(CAPACITY_MULTISET)
                and arguments.get("eval_bank_scope") == "infrastructure"
                and tuple(arguments.get("eval_dag_families") or ())
                == tuple(FAMILIES)
            )
            if reference_bank is None:
                reference_bank = bank
            elif bank != reference_bank:
                exact = False
            all_exact = all_exact and exact
            records.append(
                {
                    "seed": seed,
                    "method": label,
                    "exact": exact,
                    "config_sha256": sha256_file(directory / "config.json"),
                    "summary_sha256": sha256_file(directory / "summary.json"),
                    "scenario_bank_sha256": sha256_file(
                        directory / "evaluation_scenarios.json"
                    ),
                }
            )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "all_reference_artifacts_exact": all_exact,
        "behavioral_default_note": (
            "P6 mechanism switches default to enabled, preserving the "
            "historical P3/P4/P5 execution path."
        ),
        "records": records,
    }
    write_json(REUSE_AUDIT, report)
    if not all_exact:
        raise RuntimeError("Reference reuse audit failed")
    return report


def run_analysis():
    audit_reference_reuse()
    run_logged(
        [
            sys.executable,
            str(ROOT / "analyze_pegasus_p6.py"),
            "--heuristic-dir",
            str(HEURISTIC_DIR),
            "--learning-dir",
            str(LEARNING_DIR),
            "--output-dir",
            str(ANALYSIS_DIR),
        ],
        ANALYSIS_DIR / "analysis.log",
    )
    summary_path = ANALYSIS_DIR / "pegasus_p6_summary.json"
    summary = read_json(summary_path)
    lock = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "reference_reuse_audit_sha256": sha256_file(REUSE_AUDIT),
        "summary_sha256": sha256_file(summary_path),
        "integrity": summary["integrity"],
        "evidence": summary["evidence"],
    }
    write_json(FINAL_LOCK, lock)


def main():
    args = parse_args()
    initialize_lock()
    stages = (
        ("tests", run_tests),
        ("smoke", lambda: run_smoke(args.workers, args.resume)),
        ("heuristics", lambda: run_heuristics(args.workers, args.resume)),
        ("learning", lambda: run_learning(args.workers, args.resume)),
        ("analysis", run_analysis),
    )
    for name, operation in stages:
        if args.stage in {name, "all"}:
            operation()


if __name__ == "__main__":
    main()
