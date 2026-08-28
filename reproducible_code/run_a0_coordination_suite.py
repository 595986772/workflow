#!/usr/bin/env python3
"""Govern the full static and dynamic Alibaba-CP100-A0 experiment."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a0_coordination_protocol import (
    A0_PROTOCOL_VERSION,
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    DATASET_PATH,
    DEVELOPMENT_SEEDS,
    EXPECTED_DATASET_SHA256,
    FINAL_SEEDS,
    MAIN_BUDGET,
    METHOD_LABELS,
    frozen_protocol_spec,
    validate_protocol,
)


ROOT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = ROOT_DIR / "results" / "a0_cache_coordination"
STAGE_ORDER = ("tests", "smoke", "development", "freeze", "final")
REVISION_METADATA = {
    "a0r1": {
        "reason": (
            "B8 seed 1 exposed incomplete service coverage and a P95 "
            "regression despite a high aggregate hit rate"
        ),
        "changed_module": "coordinated_cache_demand_estimation",
        "expected_metric": (
            "service_coverage_and_p95_without_mean_completion_regression"
        ),
        "rejection_condition": (
            "mean_or_p95_fails_to_improve_against_a0r0_or_any_primary_"
            "metric_regresses_over_one_percent"
        ),
    },
    "a0r2": {
        "reason": (
            "a0r1 increased duplicate concentration and regressed B10 "
            "mean completion time by more than one percent"
        ),
        "changed_module": "coordinated_cache_replica_marginal_value",
        "expected_metric": (
            "service_coverage_and_p95_with_no_cross_budget_mean_regression"
        ),
        "rejection_condition": (
            "B8_not_superior_to_both_baselines_or_any_budget_mean_or_"
            "p95_regresses_over_one_percent_against_a0r0"
        ),
    },
}
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
    "analyze_a0_coordination.py",
    "evaluate_a0_nhpp_stream.py",
    "evaluate_oracle_latency_bound.py",
    "oracle_latency_bound.py",
    "run_a0_coordination_suite.py",
    "run_independent_experiment.py",
    "run_reproduction_suite.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER + ("all",),
        default="all",
    )
    parser.add_argument("--revision-id", default="a0r0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash(files):
    digest = hashlib.sha256()
    for filename in sorted(set(files)):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT_DIR / filename).read_bytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


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


def reproduction_command(
    directory,
    profile,
    budget,
    seeds,
    workers,
    revision_id,
    partition,
    resume,
):
    revision = REVISION_METADATA.get(
        revision_id,
        {
            "reason": "a0_heterogeneous_coordinated_cache_study",
            "changed_module": "frozen_algorithm_bundle",
            "expected_metric": "paired_dag_completion_time",
            "rejection_condition": (
                "our_not_superior_to_both_learning_baselines"
            ),
        },
    )
    command = [
        sys.executable,
        str(ROOT_DIR / "run_reproduction_suite.py"),
        "--profile",
        profile,
        "--suite-dir",
        str(directory),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--labels",
        ",".join(METHOD_LABELS),
        "--dag-dataset-path",
        str(DATASET_PATH),
        "--dag-dataset-sha256",
        EXPECTED_DATASET_SHA256,
        "--server-capacity",
        "1",
        "--server-capacity-multiset",
        ",".join(str(value) for value in CAPACITY_PROFILES[budget]),
        "--baseline-server-capacity",
        str(BASELINE_RANDOM_DRAW_CAPACITY),
        "--capacity-assignment-namespace",
        CAPACITY_ASSIGNMENT_NAMESPACE,
        "--workers",
        str(workers),
        "--revision-id",
        revision_id,
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
    if resume:
        command.append("--resume")
    return command


def oracle_command(directory, seeds, exact_checks):
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
        ",".join(str(seed) for seed in seeds),
        "--episodes",
        "100",
        "--exact-check-scenarios",
        str(exact_checks),
    ]


def analysis_command(directory, output, budget, seeds, mode):
    return [
        sys.executable,
        str(ROOT_DIR / "analyze_a0_coordination.py"),
        "--suite-dir",
        str(directory),
        "--output-dir",
        str(output),
        "--budget",
        budget,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--mode",
        mode,
    ]


def dynamic_command(suite, output, seeds, mode, workers, resume):
    command = [
        sys.executable,
        str(ROOT_DIR / "evaluate_a0_nhpp_stream.py"),
        "--suite-dir",
        str(suite),
        "--output-dir",
        str(output),
        "--labels",
        ",".join(METHOD_LABELS),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--mode",
        mode,
        "--workers",
        str(workers),
    ]
    if resume:
        command.append("--resume")
    return command


def smoke_audit(directory):
    checks = []
    for label in METHOD_LABELS:
        for seed in DEVELOPMENT_SEEDS:
            run = directory / "runs" / label / f"seed_{seed}"
            summary = read_json(run / "summary.json")
            scenarios = read_json(run / "evaluation_scenarios.json")
            checks.append(
                {
                    "label": label,
                    "seed": seed,
                    "complete": summary.get("status") == "complete",
                    "dataset_hash": summary.get("dag_dataset", {}).get(
                        "sha256"
                    )
                    == EXPECTED_DATASET_SHA256,
                    "budget": summary.get("total_server_capacity") == 8,
                    "evaluation_scenarios": len(scenarios) == 20,
                }
            )
    fingerprints_paired = True
    for seed in DEVELOPMENT_SEEDS:
        banks = [
            read_json(
                directory
                / "runs"
                / label
                / f"seed_{seed}"
                / "evaluation_scenarios.json"
            )
            for label in METHOD_LABELS
        ]
        fingerprints_paired &= all(bank == banks[0] for bank in banks[1:])
    result = {
        "status": "complete",
        "checks": checks,
        "scenario_banks_paired": fingerprints_paired,
        "passed": bool(
            fingerprints_paired
            and all(
                all(value for key, value in row.items() if key not in {"label", "seed"})
                for row in checks
            )
        ),
    }
    write_json(directory / "smoke_audit.json", result)
    return result


def hashes():
    return {
        "algorithm_source_sha256": source_hash(ALGORITHM_SOURCE_FILES),
        "protocol_source_sha256": source_hash(PROTOCOL_SOURCE_FILES),
        "frozen_configuration_sha256": canonical_hash(
            frozen_protocol_spec()
        ),
    }


def verify_freeze(revision_root, expected_hashes):
    path = revision_root / "FROZEN_ALGORITHM.json"
    if not path.exists():
        raise RuntimeError("Development has not frozen the algorithm")
    record = read_json(path)
    for key, expected in expected_hashes.items():
        if record.get(key) != expected:
            raise RuntimeError("Source changed after A0 algorithm freeze")
    return record


def freeze(revision_root, current_hashes):
    development = revision_root / "development"
    static_results = {
        budget: read_json(
            development / budget / "analysis" / "static_summary.json"
        )
        for budget in CAPACITY_PROFILES
    }
    dynamic = read_json(development / "dynamic" / "dynamic_summary.json")
    gates = {
        **{
            f"static_{budget}": result["gate"]["passed"]
            for budget, result in static_results.items()
        },
        "dynamic_B8": dynamic["gate"]["passed"],
    }
    if not all(gates.values()):
        raise RuntimeError(
            "Development gate failed; diagnose with seeds 1-3 before freeze: "
            f"{gates}"
        )
    path = revision_root / "FROZEN_ALGORITHM.json"
    record = {
        "status": "frozen",
        "revision_id": revision_root.name,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        **current_hashes,
        "development_gates": gates,
        "policy": "no_changes_after_development_before_single_use_final",
    }
    if path.exists():
        existing = read_json(path)
        for key, value in current_hashes.items():
            if existing.get(key) != value:
                raise RuntimeError("Existing freeze belongs to other source")
        return existing
    write_json(path, record)
    return record


def open_final_lock(revision_root, current_hashes, resume):
    path = revision_root / "FINAL_LOCK.json"
    experiment_hash = canonical_hash(
        {
            "protocol": frozen_protocol_spec(),
            "hashes": current_hashes,
            "seeds": FINAL_SEEDS,
        }
    )
    if path.exists():
        lock = read_json(path)
        if lock.get("experiment_sha256") != experiment_hash:
            raise RuntimeError("Final lock belongs to another configuration")
        if lock.get("status") == "complete":
            raise RuntimeError("Single-use A0 final is already complete")
        if not resume:
            raise RuntimeError("Use --resume for an interrupted final run")
        return path
    write_json(
        path,
        {
            "status": "running",
            "opened_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_sha256": experiment_hash,
            "seeds": FINAL_SEEDS,
            "policy": "single_use_no_retuning_after_results",
        },
    )
    return path


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
    write_json(
        directory / "tests_summary.json",
        {"status": "complete", "passed": True},
    )


def run_static_budget(
    directory,
    budget,
    seeds,
    mode,
    args,
):
    directory.mkdir(parents=True, exist_ok=True)
    run_logged(
        reproduction_command(
            directory,
            "e2_converged",
            budget,
            seeds,
            args.workers,
            args.revision_id,
            "development" if mode == "development" else "final",
            args.resume,
        ),
        directory / "runner.log",
    )
    run_logged(
        oracle_command(
            directory,
            seeds,
            exact_checks=0 if mode == "development" else 1,
        ),
        directory / "oracle.log",
    )
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    run_logged(
        analysis_command(directory, output, budget, seeds, mode),
        output / "analysis.log",
    )
    return read_json(output / "static_summary.json")


def run_development(revision_root, args):
    directory = revision_root / "development"
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for budget in CAPACITY_PROFILES:
        results[budget] = run_static_budget(
            directory / budget,
            budget,
            DEVELOPMENT_SEEDS,
            "development",
            args,
        )
    dynamic_dir = directory / "dynamic"
    dynamic_dir.mkdir(exist_ok=True)
    run_logged(
        dynamic_command(
            directory / MAIN_BUDGET,
            dynamic_dir,
            DEVELOPMENT_SEEDS,
            "development",
            args.workers,
            args.resume,
        ),
        dynamic_dir / "runner.log",
    )
    results["dynamic"] = read_json(dynamic_dir / "dynamic_summary.json")
    write_json(
        directory / "development_summary.json",
        {
            "status": "complete",
            "gates": {
                key: value["gate"]
                for key, value in results.items()
            },
        },
    )
    return results


def write_final_reports(final_dir, results, dynamic):
    sensitivity = {
        budget: {
            label: result["method_aggregates"][label]["mean_finish_time"]
            for label in METHOD_LABELS
        }
        for budget, result in results.items()
    }
    budgets = [5, 8, 10]
    budget_names = [f"B{value}" for value in budgets]
    figure, axis = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    for label, color in (
        ("guided_full", "#59636E"),
        ("centralized_greedy_daoc", "#E09F3E"),
        ("lean_our", "#277DA1"),
    ):
        axis.plot(
            budgets,
            [sensitivity[name][label] for name in budget_names],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axis.set_xlabel("Total cache budget")
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.set_xticks(budgets)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(final_dir / "budget_sensitivity.png", dpi=220)
    figure.savefig(final_dir / "budget_sensitivity.pdf")
    plt.close(figure)
    overall_gate = bool(
        results[MAIN_BUDGET]["gate"]["passed"]
        and dynamic["gate"]["passed"]
    )
    summary = {
        "status": "complete",
        "claim_scope": "A0_controlled_mechanism_only",
        "static": {
            budget: result["gate"] for budget, result in results.items()
        },
        "dynamic": dynamic["gate"],
        "budget_sensitivity": sensitivity,
        "formal_main_claim_passed": overall_gate,
    }
    write_json(final_dir / "FINAL_SUMMARY.json", summary)
    zh = [
        "# Alibaba-CP100-A0完整实验报告",
        "",
        "> A0是受控机制数据集，不是无偏Alibaba holdout。",
        "",
        f"- B8静态正式门槛：`{results[MAIN_BUDGET]['gate']['passed']}`。",
        f"- NHPP动态正式门槛：`{dynamic['gate']['passed']}`。",
        f"- 总体正式结论通过：`{overall_gate}`。",
        "",
        "详细数值、CI、Wilcoxon检验、Oracle gap和开销见各预算`analysis/STATIC_REPORT_ZH.md`与`dynamic/DYNAMIC_REPORT_ZH.md`。",
    ]
    (final_dir / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(zh) + "\n",
        encoding="utf-8",
    )
    en = [
        "# Alibaba-CP100-A0 Complete Experiment Report",
        "",
        "> A0 is a controlled mechanism dataset, not an unbiased Alibaba holdout.",
        "",
        f"- Formal B8 static gate: `{results[MAIN_BUDGET]['gate']['passed']}`.",
        f"- Formal NHPP dynamic gate: `{dynamic['gate']['passed']}`.",
        f"- Overall formal claim passed: `{overall_gate}`.",
        "",
        "See each budget's analysis report and the dynamic report for metrics, paired confidence intervals, Wilcoxon tests, Oracle gaps, and overheads.",
    ]
    (final_dir / "FINAL_REPORT_EN.md").write_text(
        "\n".join(en) + "\n",
        encoding="utf-8",
    )
    return summary


def run_final(revision_root, args, current_hashes):
    verify_freeze(revision_root, current_hashes)
    lock_path = open_final_lock(
        revision_root,
        current_hashes,
        args.resume,
    )
    final_dir = revision_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        results = {}
        for budget in CAPACITY_PROFILES:
            results[budget] = run_static_budget(
                final_dir / budget,
                budget,
                FINAL_SEEDS,
                "final",
                args,
            )
        dynamic_dir = final_dir / "dynamic"
        dynamic_dir.mkdir(exist_ok=True)
        run_logged(
            dynamic_command(
                final_dir / MAIN_BUDGET,
                dynamic_dir,
                FINAL_SEEDS,
                "final",
                args.workers,
                args.resume,
            ),
            dynamic_dir / "runner.log",
        )
        dynamic = read_json(dynamic_dir / "dynamic_summary.json")
        final_summary = write_final_reports(final_dir, results, dynamic)
    except Exception as error:
        lock = read_json(lock_path)
        lock["status"] = "interrupted"
        lock["error"] = repr(error)
        lock["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(lock_path, lock)
        raise
    lock = read_json(lock_path)
    lock.update(
        {
            "status": "complete",
            "closed_at": datetime.now().isoformat(timespec="seconds"),
            "wall_time_sec": time.perf_counter() - started,
            "formal_main_claim_passed": final_summary[
                "formal_main_claim_passed"
            ],
            "retuning_permitted": False,
        }
    )
    write_json(lock_path, lock)
    return final_summary


def stage_complete(revision_root, stage):
    paths = {
        "tests": revision_root / "tests" / "tests_summary.json",
        "smoke": revision_root / "smoke" / "smoke_audit.json",
        "development": (
            revision_root / "development" / "development_summary.json"
        ),
        "freeze": revision_root / "FROZEN_ALGORITHM.json",
        "final": revision_root / "final" / "FINAL_SUMMARY.json",
    }
    path = paths[stage]
    return path.exists() and read_json(path).get("status") in {
        "complete",
        "frozen",
    }


def main():
    args = parse_args()
    validate_protocol()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    revision_root = RESULT_ROOT / args.revision_id
    revision_root.mkdir(parents=True, exist_ok=True)
    current_hashes = hashes()
    write_json(
        revision_root / "PROTOCOL.json",
        {
            "status": "active",
            "protocol_version": A0_PROTOCOL_VERSION,
            **current_hashes,
            "specification": frozen_protocol_spec(),
        },
    )
    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        if args.resume and stage_complete(revision_root, stage):
            print(f"[resume] {stage} already complete", flush=True)
            continue
        print(f"\n=== A0 stage: {stage} ===", flush=True)
        if stage == "tests":
            directory = revision_root / "tests"
            directory.mkdir(exist_ok=True)
            run_tests(directory)
        elif stage == "smoke":
            directory = revision_root / "smoke"
            directory.mkdir(exist_ok=True)
            run_logged(
                reproduction_command(
                    directory,
                    "e2_smoke",
                    MAIN_BUDGET,
                    DEVELOPMENT_SEEDS,
                    args.workers,
                    args.revision_id,
                    "smoke",
                    args.resume,
                ),
                directory / "runner.log",
            )
            audit = smoke_audit(directory)
            if not audit["passed"]:
                raise RuntimeError("A0 smoke integrity gate failed")
        elif stage == "development":
            run_development(revision_root, args)
        elif stage == "freeze":
            freeze(revision_root, current_hashes)
        elif stage == "final":
            run_final(revision_root, args, current_hashes)
    print(f"A0 protocol artifacts: {revision_root}", flush=True)


if __name__ == "__main__":
    main()
