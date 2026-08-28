#!/usr/bin/env python3
"""Run governed post-lock Pegasus-B8 baseline experiments."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from pegasus_baseline_extension_protocol import (
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DATASET_PATH,
    DEVELOPMENT_SEEDS,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_SEEDS,
    HEURISTIC_METHODS,
    P3_FINAL_DIR,
    PROTOCOL_VERSION,
    REFERENCE_METHODS,
    RESULT_ROOT,
    SAC_CONFIG,
    SAC_METHOD,
    SMOKE_SEEDS,
    TASK_LIMIT_INCLUDING_DUMMY,
    sha256_file,
    validate_protocol,
)


HEURISTIC_DIR = RESULT_ROOT / "heuristics_final"
SAC_SMOKE_DIR = RESULT_ROOT / "sac_smoke"
SAC_SCREEN_DIR = RESULT_ROOT / "sac_screen"
SAC_DEVELOPMENT_DIR = RESULT_ROOT / "sac_development"
SAC_FINAL_DIR = RESULT_ROOT / "sac_final"
ANALYSIS_DIR = RESULT_ROOT / "analysis"
FROZEN_PATH = RESULT_ROOT / "FROZEN_BASELINE.json"
FINAL_LOCK_PATH = RESULT_ROOT / "FINAL_LOCK.json"
DEVELOPMENT_REPORT_PATH = RESULT_ROOT / "SAC_DEVELOPMENT_REPORT.json"
P2_DEVELOPMENT_DIR = (
    Path(__file__).resolve().parent
    / "results/pegasus_pscale/p2/converged"
)
P3_DEVELOPMENT_DIR = (
    Path(__file__).resolve().parent
    / "results/pegasus_pscale/p3_paper_closure/development"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "heuristics",
            "sac_smoke",
            "sac_screen",
            "sac_development",
            "freeze",
            "sac_final",
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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(RESULT_ROOT / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + " ".join(str(value) for value in command)
            + "\n"
        )
        output.flush()
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
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
        str(Path(__file__).resolve().parent / "run_reproduction_suite.py"),
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
        "post_lock_paper_grade_baseline_extension",
        "--revision-changed-module",
        "new_baselines_only_existing_our_frozen",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_discrete_sac_not_converged",
        "--seed-partition",
        seed_partition,
    ]
    if resume:
        command.append("--resume")
    return command


def run_tests():
    validate_protocol()
    tests = (
        "test_discrete_sac.py",
        "test_critical_path_rl.py",
        "test_information_protocol.py",
        "test_capacity_protocol.py",
        "test_dag_completion_semantics.py",
        "test_pegasus_pscale_protocol.py",
        "test_pegasus_paper_closure.py",
        "test_pegasus_baseline_extension.py",
    )
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        *tests,
    ]
    run_logged(command, RESULT_ROOT / "tests.log")


def assert_suite_complete(suite_dir, labels, seeds, convergence):
    manifest = read_json(suite_dir / "suite_manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Suite is incomplete: {suite_dir}")
    for label in labels:
        for seed in seeds:
            run_dir = suite_dir / "runs" / label / f"seed_{seed}"
            summary = read_json(run_dir / "summary.json")
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {run_dir}")
            if convergence and not summary.get(
                "eligible_for_comparison", False
            ):
                raise RuntimeError(f"Run did not converge: {run_dir}")
            if summary.get("evaluation_scenario_count") != 100:
                raise RuntimeError(
                    f"Evaluation bank is incomplete: {run_dir}"
                )


def run_heuristics(workers, resume):
    run_logged(
        suite_command(
            "pegasus_baseline_heuristics",
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
        convergence=False,
    )


def run_sac_stage(profile, suite_dir, seeds, workers, resume, convergence):
    run_logged(
        suite_command(
            profile,
            suite_dir,
            seeds,
            (SAC_METHOD,),
            workers,
            "baseline_extension",
            resume,
        ),
        suite_dir / "runner.log",
    )
    expected_episodes = 100 if convergence else None
    manifest = read_json(suite_dir / "suite_manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"SAC suite is incomplete: {suite_dir}")
    for seed in seeds:
        summary = read_json(
            suite_dir
            / "runs"
            / SAC_METHOD
            / f"seed_{seed}"
            / "summary.json"
        )
        if summary.get("status") != "complete":
            raise RuntimeError(f"SAC run failed for seed {seed}")
        if convergence and not summary.get(
            "eligible_for_comparison", False
        ):
            raise RuntimeError(f"SAC did not converge for seed {seed}")
        if (
            expected_episodes is not None
            and summary.get("evaluation_scenario_count")
            != expected_episodes
        ):
            raise RuntimeError("SAC evaluation bank is incomplete")


def evaluation_mean(run_dir):
    summary = read_json(run_dir / "summary.json")
    return float(summary["eval"]["mean_average_finish_time"])


def checkpoint_alpha(run_dir):
    checkpoint = torch.load(
        run_dir / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    values = []
    for weights in checkpoint["frozen_state"]["weights"].values():
        if "log_alpha" in weights:
            values.append(float(torch.exp(weights["log_alpha"]).item()))
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Invalid Discrete SAC temperature checkpoint")
    return float(np.mean(values))


def development_audit():
    records = []
    for seed in DEVELOPMENT_SEEDS:
        sac_dir = (
            SAC_DEVELOPMENT_DIR
            / "runs"
            / SAC_METHOD
            / f"seed_{seed}"
        )
        summary = read_json(sac_dir / "summary.json")
        config = read_json(sac_dir / "config.json")["arguments"]
        if config["algorithm"] != SAC_CONFIG["algorithm"]:
            raise RuntimeError("SAC algorithm mismatch")
        if config["cache_policy"] != SAC_CONFIG["cache_policy"]:
            raise RuntimeError("SAC cache policy mismatch")
        if config["reward_mode"] != SAC_CONFIG["reward_mode"]:
            raise RuntimeError("SAC reward mismatch")
        reference_dirs = {
            "daoc_paper": (
                P3_DEVELOPMENT_DIR
                / "runs/daoc_paper"
                / f"seed_{seed}"
            ),
            "centralized_greedy_daoc": (
                P2_DEVELOPMENT_DIR
                / "runs/centralized_greedy_daoc"
                / f"seed_{seed}"
            ),
            "lean_our": (
                P2_DEVELOPMENT_DIR
                / "runs/lean_our"
                / f"seed_{seed}"
            ),
        }
        records.append(
            {
                "seed": seed,
                "converged": summary["convergence"]["reached"],
                "episode": summary["selected_checkpoint_episode"],
                "sac_mean": evaluation_mean(sac_dir),
                "alpha": checkpoint_alpha(sac_dir),
                "references": {
                    label: evaluation_mean(directory)
                    for label, directory in reference_dirs.items()
                },
            }
        )
    integrity = (
        all(record["converged"] for record in records)
        and all(
            math.isfinite(record["sac_mean"])
            and 0 < record["alpha"] <= math.exp(2)
            for record in records
        )
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "records": records,
        "integrity": integrity,
        "mean_sac": float(np.mean([row["sac_mean"] for row in records])),
        "mean_alpha": float(np.mean([row["alpha"] for row in records])),
        "wins_vs_daoc": sum(
            row["sac_mean"] < row["references"]["daoc_paper"]
            for row in records
        ),
        "wins_vs_centralized_greedy": sum(
            row["sac_mean"]
            < row["references"]["centralized_greedy_daoc"]
            for row in records
        ),
        "wins_vs_our": sum(
            row["sac_mean"] < row["references"]["lean_our"]
            for row in records
        ),
    }
    write_json(DEVELOPMENT_REPORT_PATH, report)
    if not integrity:
        raise RuntimeError("Discrete SAC development integrity failed")
    return report


def source_hashes():
    root = Path(__file__).resolve().parent
    files = (
        "discrete_sac.py",
        "agent.py",
        "critical_path_rl.py",
        "run_independent_experiment.py",
        "run_reproduction_suite.py",
        "pegasus_baseline_extension_protocol.py",
        "run_pegasus_baseline_extension.py",
    )
    return {name: sha256_file(root / name) for name in files}


def freeze_baseline():
    protocol = validate_protocol()
    development = development_audit()
    payload = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "development": development,
        "source_hashes": source_hashes(),
        "sac_config": dict(SAC_CONFIG),
        "statement": (
            "Only the new baseline is frozen; P3 OUR checkpoints and "
            "reported results remain immutable."
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["freeze_sha256"] = hashlib.sha256(encoded).hexdigest()
    write_json(FROZEN_PATH, payload)


def initialize_final_lock(resume):
    if not FROZEN_PATH.exists():
        raise RuntimeError("Discrete SAC baseline is not frozen")
    if FINAL_LOCK_PATH.exists():
        lock = read_json(FINAL_LOCK_PATH)
        if resume and lock.get("status") in {"running", "complete"}:
            return lock
        raise RuntimeError(
            "Final baseline lock already exists; use --resume only to "
            "continue the same immutable specification"
        )
    frozen = read_json(FROZEN_PATH)
    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "freeze_sha256": frozen["freeze_sha256"],
        "methods": list(HEURISTIC_METHODS) + [SAC_METHOD],
        "reference_methods": list(REFERENCE_METHODS),
        "seeds": list(FINAL_SEEDS),
        "scenarios_per_seed": EVALUATION_EPISODES,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "post_lock_extension": True,
    }
    encoded = json.dumps(
        specification,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    lock = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    write_json(FINAL_LOCK_PATH, lock)
    return lock


def run_analysis():
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parent
            / "analyze_pegasus_baseline_extension.py"
        ),
        "--heuristic-dir",
        str(HEURISTIC_DIR),
        "--sac-dir",
        str(SAC_FINAL_DIR),
        "--reference-dir",
        str(P3_FINAL_DIR),
        "--output-dir",
        str(ANALYSIS_DIR),
    ]
    run_logged(command, ANALYSIS_DIR / "analysis.log")
    summary = read_json(
        ANALYSIS_DIR / "baseline_extension_summary.json"
    )
    lock = read_json(FINAL_LOCK_PATH)
    lock.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "gate": summary["gate"],
            "summary_sha256": sha256_file(
                ANALYSIS_DIR / "baseline_extension_summary.json"
            ),
        }
    )
    write_json(FINAL_LOCK_PATH, lock)


def main():
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    validate_protocol()

    if args.stage in ("tests", "all"):
        run_tests()
        if args.stage == "tests":
            return
    if args.stage in ("heuristics", "all"):
        run_heuristics(args.workers, args.resume)
    if args.stage in ("sac_smoke", "all"):
        run_sac_stage(
            "pegasus_baseline_sac_smoke",
            SAC_SMOKE_DIR,
            SMOKE_SEEDS,
            args.workers,
            args.resume,
            convergence=False,
        )
    if args.stage in ("sac_screen", "all"):
        run_sac_stage(
            "pegasus_baseline_sac_screen",
            SAC_SCREEN_DIR,
            DEVELOPMENT_SEEDS,
            args.workers,
            args.resume,
            convergence=False,
        )
    if args.stage in ("sac_development", "all"):
        run_sac_stage(
            "pegasus_baseline_sac_converged",
            SAC_DEVELOPMENT_DIR,
            DEVELOPMENT_SEEDS,
            args.workers,
            args.resume,
            convergence=True,
        )
        development_audit()
    if args.stage in ("freeze", "all"):
        freeze_baseline()
    if args.stage in ("sac_final", "all"):
        initialize_final_lock(args.resume)
        run_sac_stage(
            "pegasus_baseline_sac_converged",
            SAC_FINAL_DIR,
            FINAL_SEEDS,
            args.workers,
            args.resume,
            convergence=True,
        )
    if args.stage in ("analysis", "all"):
        if not FINAL_LOCK_PATH.exists():
            raise RuntimeError("Final lock is missing")
        assert_suite_complete(
            HEURISTIC_DIR,
            HEURISTIC_METHODS,
            FINAL_SEEDS,
            convergence=False,
        )
        assert_suite_complete(
            SAC_FINAL_DIR,
            (SAC_METHOD,),
            FINAL_SEEDS,
            convergence=True,
        )
        run_analysis()
    print(f"Baseline extension artifacts: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
