#!/usr/bin/env python3
"""Run the governed Pegasus DAOC-paper and ablation closure plan."""

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

from pegasus_paper_closure_protocol import (
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DATASET_PATH,
    DEVELOPMENT_METHODS,
    DEVELOPMENT_SEEDS,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_METHODS,
    FINAL_SEEDS,
    P2_DEVELOPMENT_SUITE,
    P2_LOCK_PATH,
    P2_ROOT,
    PROTOCOL_VERSION,
    SMOKE_SEEDS,
    TASK_LIMIT_INCLUDING_DUMMY,
    validate_protocol,
)
from run_a0_fixed_budget_heterogeneity import (
    ALGORITHM_SOURCE_FILES,
    source_hash,
)
from user import DAG_COMPLETION_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results/pegasus_pscale/p3_paper_closure"
LOCK_PATH = RESULT_ROOT / "CLOSURE_LOCK.json"
FREEZE_PATH = RESULT_ROOT / "FROZEN_ALGORITHM.json"
FINAL_LOCK_PATH = RESULT_ROOT / "FINAL_LOCK.json"
SMOKE_DIR = RESULT_ROOT / "smoke"
DEVELOPMENT_DIR = RESULT_ROOT / "development"
FINAL_DIR = RESULT_ROOT / "final"
PROTOCOL_SOURCE_FILES = ALGORITHM_SOURCE_FILES + (
    "analyze_pegasus_paper_closure.py",
    "information_protocol.py",
    "pegasus_paper_closure_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_paper_closure.py",
    "run_reproduction_suite.py",
    "test_pegasus_paper_closure.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "tests",
            "smoke",
            "development",
            "analysis",
            "freeze",
            "final",
            "final_analysis",
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
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_source_hash():
    return source_hash(PROTOCOL_SOURCE_FILES)


def protocol_specification():
    frozen = validate_protocol()
    return {
        **frozen,
        "parent_p2_lock_sha256": sha256_file(P2_LOCK_PATH),
        "daoc_variants": {
            "guided_full": (
                "released-code fixed guidance and pure popularity EMA"
            ),
            "daoc_paper": (
                "paper guidance decay and Eq.14 popularity-cost EMA"
            ),
        },
        "development_methods": list(DEVELOPMENT_METHODS),
        "final_methods": list(FINAL_METHODS),
        "development_gate": {
            "our_beats_daoc_paper": "positive mean and at least 2/3 wins",
            "our_beats_centralized_greedy": (
                "positive mean and at least 2/3 wins"
            ),
            "ablation": "at least one proposed module is supported",
        },
        "final_gate": {
            "comparisons": [
                "OUR vs DAOC-paper",
                "OUR vs Centralized-Greedy-DQN",
            ],
            "ci": "paired seed-level 95% lower bound greater than zero",
            "test": "one-sided Wilcoxon p < 0.05",
            "wins": "at least 7/10 seeds",
        },
    }


def initialize_lock():
    specification = protocol_specification()
    lock = {
        "status": "development_locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "protocol_source_sha256": protocol_source_hash(),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        if existing.get("specification_sha256") != lock[
            "specification_sha256"
        ]:
            raise RuntimeError("Paper-closure specification changed")
        if FREEZE_PATH.exists():
            for key in (
                "algorithm_source_sha256",
                "protocol_source_sha256",
            ):
                if existing.get(key) != lock[key]:
                    raise RuntimeError(
                        f"Frozen paper-closure source changed: {key}"
                    )
        elif any(
            existing.get(key) != lock[key]
            for key in (
                "algorithm_source_sha256",
                "protocol_source_sha256",
            )
        ):
            revisions = existing.setdefault(
                "development_source_revisions",
                [],
            )
            revisions.append(
                {
                    "changed_at": datetime.now(timezone.utc).isoformat(),
                    "algorithm_source_sha256": lock[
                        "algorithm_source_sha256"
                    ],
                    "protocol_source_sha256": lock[
                        "protocol_source_sha256"
                    ],
                }
            )
            existing["algorithm_source_sha256"] = lock[
                "algorithm_source_sha256"
            ]
            existing["protocol_source_sha256"] = lock[
                "protocol_source_sha256"
            ]
            write_json(LOCK_PATH, existing)
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
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


def reproduction_command(
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
        "pegasus_pscale_p2",
        "--revision-reason",
        "close_daoc_paper_protocol_and_algorithmic_ablations",
        "--revision-changed-module",
        "daoc_paper_cache_protocol_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "our_fails_primary_or_no_algorithmic_ablation_is_supported",
        "--seed-partition",
        seed_partition,
    ]
    if resume:
        command.append("--resume")
    return command


def evaluation_rows(run_dir):
    with (run_dir / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        return [
            row
            for row in csv.DictReader(input_file)
            if row["phase"] == "eval"
        ]


def comparable_bank(bank):
    return [
        {
            "episode": record["episode"],
            "seed": record["seed"],
            "base_fingerprint": record["base_fingerprint"],
            "workflow_family": record.get("workflow_family"),
            "user_initial_positions": record["user_initial_positions"],
            "user_graph_keys": record["user_graph_keys"],
        }
        for record in bank
    ]


def reference_bank_for(seed, expected_episodes):
    phase = "smoke" if expected_episodes == 20 else "converged"
    path = (
        P2_ROOT
        / phase
        / "runs/lean_our"
        / f"seed_{seed}"
        / "evaluation_scenarios.json"
    )
    if not path.exists():
        return None
    return comparable_bank(read_json(path))


def check_runs(
    suite_dir,
    labels,
    seeds,
    expected_episodes,
    require_convergence,
    compare_to_p2,
):
    for seed in seeds:
        reference_bank = (
            reference_bank_for(seed, expected_episodes)
            if compare_to_p2
            else None
        )
        reference_capacities = None
        for label in labels:
            directory = suite_dir / "runs" / label / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {label} seed={seed}")
            if require_convergence and not (
                summary.get("eligible_for_comparison")
                and summary.get("convergence", {}).get("reached")
            ):
                raise RuntimeError(f"Unconverged run: {label} seed={seed}")
            if summary.get("dag_completion_protocol_version") != (
                DAG_COMPLETION_PROTOCOL_VERSION
            ):
                raise RuntimeError("Wrong DAG completion protocol")
            if summary.get("dag_dataset", {}).get("sha256") != (
                EXPECTED_DATASET_SHA256
            ):
                raise RuntimeError("Pegasus dataset mismatch")
            capacities = {
                int(key): int(value)
                for key, value in summary["server_capacities"].items()
            }
            if sorted(capacities.values()) != sorted(CAPACITY_MULTISET):
                raise RuntimeError("B8 capacity assignment mismatch")
            if reference_capacities is None:
                reference_capacities = capacities
            elif capacities != reference_capacities:
                raise RuntimeError("Capacity assignments are not paired")
            rows = evaluation_rows(directory)
            if len(rows) != expected_episodes or not all(
                int(row["real_task_count"])
                == int(row["completed_task_count"])
                and int(row["all_tasks_executed_once"]) == 1
                for row in rows
            ):
                raise RuntimeError("Task completion audit failed")
            bank = read_json(directory / "evaluation_scenarios.json")
            if Counter(
                record.get("workflow_family") for record in bank
            ) != Counter(
                {
                    family: expected_episodes // len(FAMILIES)
                    for family in FAMILIES
                }
            ):
                raise RuntimeError("Workflow-family balance mismatch")
            bank_view = comparable_bank(bank)
            if reference_bank is None:
                reference_bank = bank_view
            elif bank_view != reference_bank:
                raise RuntimeError("Evaluation scenario banks are not paired")


def run_tests():
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "test_pegasus_paper_closure.py",
        "test_dag_completion_semantics.py",
        "test_pegasus_pscale_protocol.py",
        "test_capacity_protocol.py",
        "test_information_protocol.py",
    ]
    run_logged(command, RESULT_ROOT / "tests.log")


def run_analysis(mode, suite_dir):
    output_dir = suite_dir / "analysis"
    command = [
        sys.executable,
        str(ROOT / "analyze_pegasus_paper_closure.py"),
        "--suite-dir",
        str(suite_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
    ]
    run_logged(command, suite_dir / "analysis.log")


def freeze_algorithm():
    summary_path = (
        DEVELOPMENT_DIR
        / "analysis/pegasus_paper_closure_summary.json"
    )
    summary = read_json(summary_path)
    if not summary.get("gate", {}).get("passed"):
        raise RuntimeError("Development gate failed; final run is forbidden")
    primary = [
        name
        for name, tier in summary["module_claim_tiers"].items()
        if tier == "primary"
    ]
    secondary = [
        name
        for name, tier in summary["module_claim_tiers"].items()
        if tier == "secondary"
    ]
    unsupported = [
        name
        for name, tier in summary["module_claim_tiers"].items()
        if tier == "unsupported"
    ]
    freeze = {
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "protocol_source_sha256": protocol_source_hash(),
        "development_summary_sha256": sha256_file(summary_path),
        "primary_claim_modules": primary,
        "secondary_mechanism_modules": secondary,
        "unsupported_primary_claim_modules": unsupported,
        "formal_methods": list(FINAL_METHODS),
        "formal_seeds": list(FINAL_SEEDS),
        "statistics": {
            "unit": "seed-level paired mean DAG completion time",
            "ci": "two-sided Student-t 95% CI on paired differences",
            "test": "one-sided Wilcoxon signed-rank",
            "alpha": 0.05,
            "minimum_wins": 7,
        },
    }
    if FREEZE_PATH.exists():
        existing = read_json(FREEZE_PATH)
        for key in (
            "algorithm_source_sha256",
            "protocol_source_sha256",
            "development_summary_sha256",
        ):
            if existing.get(key) != freeze[key]:
                raise RuntimeError(f"Frozen algorithm mismatch: {key}")
        return existing
    write_json(FREEZE_PATH, freeze)
    claim_lines = [
        "# Pegasus B8 创新表述冻结\n",
        "以三 seed 消融的配对结果为依据，十 seed 开始后不再修改算法或表述口径。\n",
        "## 保留的主要贡献\n",
    ]
    claim_lines.extend(f"- `{name}`\n" for name in primary)
    claim_lines.append("\n## 仅作为次要辅助机制\n")
    claim_lines.extend(f"- `{name}`\n" for name in secondary)
    claim_lines.append("\n## 删除主要创新表述的模块\n")
    claim_lines.extend(f"- `{name}`\n" for name in unsupported)
    (RESULT_ROOT / "FROZEN_CLAIMS_ZH.md").write_text(
        "".join(claim_lines),
        encoding="utf-8",
    )
    return freeze


def initialize_final_lock(resume):
    freeze = freeze_algorithm()
    if freeze["algorithm_source_sha256"] != source_hash(
        ALGORITHM_SOURCE_FILES
    ):
        raise RuntimeError("Algorithm source changed after freeze")
    if freeze["protocol_source_sha256"] != protocol_source_hash():
        raise RuntimeError("Protocol source changed after freeze")
    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "methods": list(FINAL_METHODS),
        "seeds": list(FINAL_SEEDS),
        "scenarios_per_seed": EVALUATION_EPISODES,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "freeze_sha256": sha256_file(FREEZE_PATH),
    }
    final_lock = {
        "status": "opened_once",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "algorithm_source_sha256": freeze["algorithm_source_sha256"],
        "protocol_source_sha256": freeze["protocol_source_sha256"],
    }
    if FINAL_LOCK_PATH.exists():
        existing = read_json(FINAL_LOCK_PATH)
        if existing.get("specification_sha256") != final_lock[
            "specification_sha256"
        ]:
            raise RuntimeError("Final lock specification mismatch")
        return existing
    if (FINAL_DIR / "suite_manifest.json").exists() and not resume:
        raise RuntimeError(
            "Final suite already exists; use --resume only for interruption"
        )
    write_json(FINAL_LOCK_PATH, final_lock)
    return final_lock


def run_final_oracle():
    command = [
        sys.executable,
        str(ROOT / "evaluate_oracle_latency_bound.py"),
        "--our-suite-dir",
        str(FINAL_DIR),
        "--our-label",
        "lean_our",
        "--daoc-suite-dir",
        str(FINAL_DIR),
        "--daoc-label",
        "daoc_paper",
        "--output-dir",
        str(FINAL_DIR / "oracle"),
        "--seeds",
        ",".join(str(seed) for seed in FINAL_SEEDS),
        "--episodes",
        str(EVALUATION_EPISODES),
        "--exact-check-scenarios",
        "0",
    ]
    run_logged(command, FINAL_DIR / "oracle.log")


def main():
    args = parse_args()
    if args.stage in ("tests", "all"):
        run_tests()
        if args.stage == "tests":
            print(f"Paper-closure artifacts: {RESULT_ROOT}")
            return

    initialize_lock()
    if args.stage in ("smoke", "all"):
        run_logged(
            reproduction_command(
                "pegasus_paper_closure_smoke",
                SMOKE_DIR,
                SMOKE_SEEDS,
                DEVELOPMENT_METHODS,
                args.workers,
                "smoke",
                args.resume,
            ),
            SMOKE_DIR / "runner.log",
        )
        check_runs(
            SMOKE_DIR,
            DEVELOPMENT_METHODS,
            SMOKE_SEEDS,
            20,
            False,
            True,
        )

    if args.stage in ("development", "all"):
        run_logged(
            reproduction_command(
                "pegasus_paper_closure_converged",
                DEVELOPMENT_DIR,
                DEVELOPMENT_SEEDS,
                DEVELOPMENT_METHODS,
                args.workers,
                "ablation",
                args.resume,
            ),
            DEVELOPMENT_DIR / "runner.log",
        )
        check_runs(
            DEVELOPMENT_DIR,
            DEVELOPMENT_METHODS,
            DEVELOPMENT_SEEDS,
            EVALUATION_EPISODES,
            True,
            True,
        )

    if args.stage in ("analysis", "all"):
        run_analysis("development", DEVELOPMENT_DIR)

    if args.stage in ("freeze", "all"):
        freeze_algorithm()

    if args.stage in ("final", "all"):
        initialize_final_lock(args.resume)
        run_logged(
            reproduction_command(
                "pegasus_paper_closure_converged",
                FINAL_DIR,
                FINAL_SEEDS,
                FINAL_METHODS,
                args.workers,
                "final",
                args.resume,
            ),
            FINAL_DIR / "runner.log",
        )
        check_runs(
            FINAL_DIR,
            FINAL_METHODS,
            FINAL_SEEDS,
            EVALUATION_EPISODES,
            True,
            False,
        )
        run_final_oracle()

    if args.stage in ("final_analysis", "all"):
        run_analysis("final", FINAL_DIR)
        summary_path = (
            FINAL_DIR / "analysis/pegasus_paper_closure_summary.json"
        )
        final_lock = read_json(FINAL_LOCK_PATH)
        final_lock["status"] = "complete"
        final_lock["completed_at"] = datetime.now(timezone.utc).isoformat()
        final_lock["summary_sha256"] = sha256_file(summary_path)
        final_lock["formal_gate"] = read_json(summary_path)["gate"]
        write_json(FINAL_LOCK_PATH, final_lock)

    print(f"Paper-closure artifacts: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
