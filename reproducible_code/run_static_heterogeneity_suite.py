#!/usr/bin/env python3
"""Run the governed H0-H4 static cache-heterogeneity protocol."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_reproduction_suite import (
    ALGORITHMS,
    PROFILES,
    effective_method_profile,
)
from static_heterogeneity_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    MAIN_PROFILE,
    STATIC_HETEROGENEITY_PROTOCOL_VERSION,
    frozen_environment_spec,
)


DAOC_LABEL = "guided_full"
OUR_LABEL = "lean_our"
STAGE_ORDER = (
    "tests",
    "smoke",
    "screen",
    "converged",
    "ablation",
    "final",
    "generalization",
)
STAGES = {
    "tests": {
        "seeds": [],
        "partition": "development",
    },
    "smoke": {
        "profile": "e2_smoke",
        "capacity_profile": MAIN_PROFILE,
        "seeds": [1],
        "labels": [DAOC_LABEL, OUR_LABEL],
        "eval_episodes": 20,
        "partition": "smoke",
    },
    "screen": {
        "profile": "e2_development",
        "capacity_profile": MAIN_PROFILE,
        "seeds": [1, 2, 3],
        "labels": [DAOC_LABEL, OUR_LABEL],
        "eval_episodes": 50,
        "partition": "development",
    },
    "converged": {
        "profile": "e2_converged",
        "capacity_profile": MAIN_PROFILE,
        "seeds": [1, 2, 3],
        "labels": [DAOC_LABEL, OUR_LABEL],
        "eval_episodes": 100,
        "partition": "development",
    },
    "sweep": {
        "profile": "e2_converged",
        "capacity_profiles": list(CAPACITY_PROFILES),
        "seeds": [4, 5, 6, 7, 8],
        "labels": [DAOC_LABEL, OUR_LABEL],
        "eval_episodes": 100,
        "partition": "heterogeneity",
    },
    "ablation": {
        "profile": "e2_converged",
        "capacity_profile": MAIN_PROFILE,
        "seeds": [1, 2, 3],
        "labels": [
            OUR_LABEL,
            "our_no_telemetry",
            "our_no_coord_cache",
        ],
        "eval_episodes": 100,
        "partition": "ablation",
    },
    "final": {
        "profile": "e2_converged",
        "capacity_profile": MAIN_PROFILE,
        "seeds": list(range(11, 21)),
        "labels": [
            "random",
            "nearest",
            "greedy",
            DAOC_LABEL,
            OUR_LABEL,
        ],
        "eval_episodes": 100,
        "partition": "final",
    },
    "generalization": {
        "capacity_profiles": list(CAPACITY_PROFILES),
        "seeds": list(range(11, 21)),
        "labels": [DAOC_LABEL, OUR_LABEL],
        "eval_episodes": 100,
        "partition": "generalization",
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
PROTOCOL_SOURCE_FILES = tuple(
    sorted(
        set(ALGORITHM_SOURCE_FILES)
        | {
            "aggregate_reproduction_results.py",
            "analyze_e2_e3_results.py",
            "analyze_static_heterogeneity.py",
            "analyze_strict_environment_suite.py",
            "evaluate_oracle_latency_bound.py",
            "evaluate_static_cross_capacity.py",
            "diagnose_static_cache_oscillation.py",
            "information_protocol.py",
            "oracle_latency_bound.py",
            "run_independent_experiment.py",
            "run_reproduction_suite.py",
            "run_static_heterogeneity_suite.py",
            "static_heterogeneity_protocol.py",
        }
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run governed static cache-heterogeneity experiments."
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER + ("all",),
        default="all",
    )
    parser.add_argument("--revision-id", default="hsr0")
    parser.add_argument(
        "--reuse-compatible-h3-from",
        help=(
            "Reuse behaviorally identical tests/smoke/screen/converged "
            "artifacts from a prior revision after a protocol-only change."
        ),
    )
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


def source_hash(root, filenames):
    digest = hashlib.sha256()
    for filename in filenames:
        path = root / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_protocol_spec():
    spec = frozen_environment_spec()
    algorithms = {
        item["label"]: item
        for item in ALGORITHMS
        if item["label"] in {
            DAOC_LABEL,
            OUR_LABEL,
            "our_no_telemetry",
            "our_no_coord_cache",
        }
    }
    profiles = {}
    for label in algorithms:
        profile = effective_method_profile(
            PROFILES["e2_converged"],
            label,
        )
        profile.pop("seeds", None)
        profile.pop("labels", None)
        profile["server_capacity_multiset"] = (
            CAPACITY_PROFILES[MAIN_PROFILE]
        )
        profile["baseline_server_capacity"] = (
            BASELINE_RANDOM_DRAW_CAPACITY
        )
        profile["capacity_assignment_namespace"] = (
            CAPACITY_ASSIGNMENT_NAMESPACE
        )
        profiles[label] = profile
    spec["methods"] = algorithms
    spec["training_and_evaluation"] = profiles
    spec["stage_order"] = list(STAGE_ORDER)
    spec["development_seeds"] = [1, 2, 3]
    spec["trained_sweep"] = {
        "status": "diagnostic_only",
        "pilot_seeds": [4, 5, 6, 7, 8],
        "reason": "native_daoc_cache_nonconvergence_in_h0",
        "used_for_claims": False,
    }
    spec["final_untouched_seeds"] = list(range(11, 21))
    spec["dynamic_environment_enabled"] = False
    spec["cross_capacity_evaluation"] = {
        "source": "untouched_h3_final_checkpoints",
        "target_profiles": list(CAPACITY_PROFILES),
        "retraining": False,
        "projection_information": "checkpoint_history_only",
    }
    return spec


def stage_dir(governance_root, revision_id, stage):
    return governance_root / revision_id / stage


def analysis_path(directory, stage):
    if stage in {"smoke", "screen", "converged", "final"}:
        return directory / "static_analysis.json"
    if stage == "sweep":
        return directory / "heterogeneity_analysis.json"
    if stage == "ablation":
        return directory / "static_ablation_analysis.json"
    if stage == "generalization":
        return directory / "cross_capacity_summary.json"
    return directory / "tests_summary.json"


def completed_stage(governance_root, revision_id, stage):
    directory = stage_dir(governance_root, revision_id, stage)
    manifest_path = directory / "static_stage_manifest.json"
    result_path = analysis_path(directory, stage)
    if not manifest_path.exists() or not result_path.exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        return None
    return {
        "dir": directory,
        "manifest": manifest,
        "analysis": read_json(result_path),
    }


def result_gate(stage, analysis):
    if stage == "tests":
        return analysis.get("passed") is True
    if stage == "generalization":
        return (
            analysis.get("all_methods_scenario_paired") is True
            and analysis.get(
                "all_profiles_base_scenario_paired"
            ) is True
        )
    return analysis.get("gate", {}).get("passed") is True


def verify_prerequisite(
    root,
    revision_id,
    stage,
    algorithm_hash,
    protocol_hash,
    config_hash,
):
    previous = {
        "smoke": "tests",
        "screen": "smoke",
        "converged": "screen",
        "sweep": "converged",
        "ablation": "converged",
        "final": "ablation",
        "generalization": "final",
    }.get(stage)
    if previous is None:
        return
    completed = completed_stage(root, revision_id, previous)
    if completed is None:
        raise RuntimeError(
            f"{stage} requires completed stage {previous}"
        )
    for key, expected in (
        ("algorithm_source_sha256", algorithm_hash),
        ("protocol_source_sha256", protocol_hash),
        ("frozen_configuration_sha256", config_hash),
    ):
        if completed["manifest"].get(key) != expected:
            raise RuntimeError(
                f"{previous} artifacts do not match current source/protocol"
            )
    if not result_gate(previous, completed["analysis"]):
        raise RuntimeError(
            f"{stage} stopped because {previous} did not pass"
        )


def run_command(command, root, env, log_path=None):
    print(" ".join(str(item) for item in command), flush=True)
    if log_path is None:
        subprocess.run(
            command,
            cwd=root,
            env=env,
            check=True,
        )
        return
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def reproduction_command(
    root,
    directory,
    stage,
    capacity_profile,
    workers,
    resume,
    revision_id,
):
    config = STAGES[stage]
    command = [
        sys.executable,
        str(root / "run_reproduction_suite.py"),
        "--profile",
        config["profile"],
        "--suite-dir",
        str(directory),
        "--seeds",
        ",".join(str(seed) for seed in config["seeds"]),
        "--labels",
        ",".join(config["labels"]),
        "--eval-episodes",
        str(config["eval_episodes"]),
        "--server-capacity-multiset",
        ",".join(
            str(value)
            for value in CAPACITY_PROFILES[capacity_profile]
        ),
        "--baseline-server-capacity",
        str(BASELINE_RANDOM_DRAW_CAPACITY),
        "--capacity-assignment-namespace",
        CAPACITY_ASSIGNMENT_NAMESPACE,
        "--workers",
        str(workers),
        "--revision-id",
        revision_id,
        "--revision-reason",
        "static_cache_heterogeneity_protocol",
        "--revision-changed-module",
        "initial_version",
        "--revision-expected-metric",
        "paired_finish_time",
        "--revision-rejection-condition",
        "frozen_static_protocol_gate",
        "--seed-partition",
        config["partition"],
    ]
    if resume:
        command.append("--resume")
    return command


def run_oracle(
    root,
    directory,
    stage,
    seeds,
    episodes,
    env,
):
    run_command(
        [
            sys.executable,
            str(root / "evaluate_oracle_latency_bound.py"),
            "--our-suite-dir",
            str(directory),
            "--our-label",
            OUR_LABEL,
            "--daoc-suite-dir",
            str(directory),
            "--daoc-label",
            DAOC_LABEL,
            "--output-dir",
            str(directory / "oracle"),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--episodes",
            str(episodes),
            "--exact-check-scenarios",
            "1" if stage in {"converged", "final"} else "0",
        ],
        root,
        env,
        log_path=directory / "oracle.log",
    )


def run_analysis(root, directory, stage, env):
    config = STAGES[stage]
    command = [
        sys.executable,
        str(root / "analyze_static_heterogeneity.py"),
        "--suite-dir",
        str(directory),
        "--mode",
        stage,
        "--seeds",
        ",".join(str(seed) for seed in config["seeds"]),
    ]
    if "capacity_profile" in config:
        command.extend(
            ["--profile", config["capacity_profile"]]
        )
    run_command(
        command,
        root,
        env,
        log_path=directory / "analysis.log",
    )


def symlink_run(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise RuntimeError(
            f"Refusing to replace existing ablation run: {target}"
        )
    target.symlink_to(source.resolve(), target_is_directory=True)


def expected_h3_method_profile(stage, label):
    stage_config = STAGES[stage]
    profile = dict(PROFILES[stage_config["profile"]])
    profile["seeds"] = stage_config["seeds"]
    profile["eval_episodes"] = stage_config["eval_episodes"]
    profile["server_capacity"] = 1
    profile["server_capacity_multiset"] = (
        CAPACITY_PROFILES[MAIN_PROFILE]
    )
    profile["baseline_server_capacity"] = (
        BASELINE_RANDOM_DRAW_CAPACITY
    )
    profile["capacity_assignment_namespace"] = (
        CAPACITY_ASSIGNMENT_NAMESPACE
    )
    return effective_method_profile(profile, label)


def audit_reused_h3_run(run, stage, label, seed):
    config = read_json(run / "config.json")
    summary = read_json(run / "summary.json")
    arguments = config["arguments"]
    method_profile = expected_h3_method_profile(stage, label)
    algorithm = next(
        item for item in ALGORITHMS if item["label"] == label
    )
    expected_arguments = {
        "label": label,
        "seed": seed,
        "algorithm": algorithm["algorithm"],
        "train_episodes": method_profile["train_episodes"],
        "eval_episodes": method_profile["eval_episodes"],
        "num_users": method_profile["num_users"],
        "num_servers": method_profile["num_servers"],
        "num_services": method_profile["num_services"],
        "num_tasks": method_profile["num_tasks"],
        "bandwidth": method_profile["bandwidth"],
        "server_capacity_multiset": (
            CAPACITY_PROFILES[MAIN_PROFILE]
        ),
        "baseline_server_capacity": (
            BASELINE_RANDOM_DRAW_CAPACITY
        ),
        "capacity_assignment_namespace": (
            CAPACITY_ASSIGNMENT_NAMESPACE
        ),
        "reward_mode": algorithm.get(
            "reward_mode",
            "terminal_binary",
        ),
        "cache_policy": algorithm.get(
            "cache_policy",
            "popularity_ema",
        ),
    }
    mismatches = {
        key: {
            "expected": expected,
            "actual": arguments.get(key),
        }
        for key, expected in expected_arguments.items()
        if arguments.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Behavioral H3 reuse mismatch in {run}: {mismatches}"
        )
    if (
        summary.get("status") != "complete"
        or (
            stage == "converged"
            and summary.get("eligible_for_comparison") is not True
        )
    ):
        raise RuntimeError(
            f"Prior H3 run is not eligible for reuse: {run}"
        )
    return {
        "label": label,
        "seed": seed,
        "source_run": str(run.resolve()),
        "selected_checkpoint_episode": (
            summary.get("selected_checkpoint_episode")
        ),
        "behavior_fields_match": True,
    }


def reuse_compatible_h3_stage(
    governance_root,
    source_revision,
    target_revision,
    stage,
    target_dir,
    algorithm_hash,
):
    if stage not in {"smoke", "screen", "converged"}:
        raise ValueError("Only H3 development stages can be reused")
    source_dir = stage_dir(
        governance_root,
        source_revision,
        stage,
    )
    source_manifest = read_json(
        source_dir / "static_stage_manifest.json"
    )
    if (
        source_manifest.get("status") != "complete"
        or source_manifest.get("algorithm_source_sha256")
        != algorithm_hash
    ):
        raise RuntimeError(
            "Prior revision is not algorithm-identical and complete"
        )
    linked = []
    for name in ("runs", "suite_manifest.json"):
        source = source_dir / name
        target = target_dir / name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(
                    f"Reuse link points elsewhere: {target}"
                )
        elif target.exists():
            raise RuntimeError(
                f"Refusing to replace existing reuse target: {target}"
            )
        else:
            target.symlink_to(
                source.resolve(),
                target_is_directory=source.is_dir(),
            )
        linked.append(
            {
                "target": str(target),
                "source": str(source.resolve()),
            }
        )
    if (source_dir / "oracle").exists():
        source = source_dir / "oracle"
        target = target_dir / "oracle"
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(
                    f"Reuse link points elsewhere: {target}"
                )
        elif target.exists():
            raise RuntimeError(
                f"Refusing to replace existing reuse target: {target}"
            )
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)
        linked.append(
            {
                "target": str(target),
                "source": str(source.resolve()),
            }
        )
    runs = []
    for label in STAGES[stage]["labels"]:
        for seed in STAGES[stage]["seeds"]:
            runs.append(
                audit_reused_h3_run(
                    source_dir
                    / "runs"
                    / label
                    / f"seed_{seed}",
                    stage,
                    label,
                    seed,
                )
            )
    audit = {
        "status": "passed",
        "source_revision": source_revision,
        "target_revision": target_revision,
        "stage": stage,
        "algorithm_source_sha256_match": True,
        "capacity_assignment_namespace_preserved": (
            CAPACITY_ASSIGNMENT_NAMESPACE
        ),
        "behavioral_configuration_match": True,
        "reused_runs": runs,
        "links": linked,
        "reason": (
            "protocol-only revision after diagnostic trained-sweep "
            "failure; H3 algorithm, environment, scenarios, training, "
            "convergence, and evaluation are unchanged"
        ),
    }
    write_json(target_dir / "H3_REUSE_AUDIT.json", audit)
    return audit


def prepare_ablation_reuse(governance_root, revision_id, directory):
    for seed in (1, 2, 3):
        source = (
            stage_dir(
                governance_root,
                revision_id,
                "converged",
            )
            / "runs"
            / OUR_LABEL
            / f"seed_{seed}"
        )
        target = (
            directory
            / "runs"
            / OUR_LABEL
            / f"seed_{seed}"
        )
        symlink_run(source, target)


def freeze_algorithm(
    governance_root,
    revision_id,
    hashes,
    analysis,
):
    path = (
        governance_root
        / revision_id
        / "FROZEN_STATIC_ALGORITHM.json"
    )
    record = {
        "status": "frozen",
        "revision_id": revision_id,
        **hashes,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "development_gate": analysis["gate"],
        "policy": (
            "no algorithm or hyperparameter changes after H3 "
            "development convergence"
        ),
    }
    if path.exists():
        existing = read_json(path)
        comparable = {
            key: existing.get(key)
            for key in (
                "revision_id",
                "algorithm_source_sha256",
                "protocol_source_sha256",
                "frozen_configuration_sha256",
            )
        }
        expected = {
            key: record[key]
            for key in comparable
        }
        if comparable != expected:
            raise RuntimeError(
                "Frozen static algorithm no longer matches source"
            )
        return
    write_json(path, record)


def verify_frozen(governance_root, revision_id, hashes):
    path = (
        governance_root
        / revision_id
        / "FROZEN_STATIC_ALGORITHM.json"
    )
    if not path.exists():
        raise RuntimeError("Static algorithm has not been frozen")
    frozen = read_json(path)
    for key, expected in (
        ("revision_id", revision_id),
        *hashes.items(),
    ):
        if frozen.get(key) != expected:
            raise RuntimeError(
                "Source/protocol changed after algorithm freeze"
            )


def initialize_final_lock(
    governance_root,
    revision_id,
    experiment_hash,
    resume,
):
    path = (
        governance_root
        / revision_id
        / "FINAL_STATIC_LOCK.json"
    )
    if path.exists():
        lock = read_json(path)
        if (
            lock.get("revision_id") != revision_id
            or lock.get("experiment_sha256") != experiment_hash
            or not resume
        ):
            raise RuntimeError(
                "Untouched final seeds were already opened under "
                "another configuration"
            )
        if lock.get("status") == "complete":
            raise RuntimeError(
                "The untouched final experiment is single-use and complete"
            )
        return path
    write_json(
        path,
        {
            "status": "running",
            "revision_id": revision_id,
            "experiment_sha256": experiment_hash,
            "opened_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": list(range(11, 21)),
            "policy": (
                "single_use_untouched_static_final_no_retuning"
            ),
        },
    )
    return path


def run_tests(root, directory, env):
    log_path = directory / "tests.log"
    started = time.perf_counter()
    try:
        run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-p",
                "test_*.py",
                "-v",
            ],
            root,
            env,
            log_path=log_path,
        )
    except subprocess.CalledProcessError:
        write_json(
            directory / "tests_summary.json",
            {
                "status": "failed",
                "passed": False,
                "log": str(log_path),
                "wall_time_sec": time.perf_counter() - started,
            },
        )
        raise
    write_json(
        directory / "tests_summary.json",
        {
            "status": "complete",
            "passed": True,
            "log": str(log_path),
            "wall_time_sec": time.perf_counter() - started,
        },
    )


def execute_stage(
    stage,
    args,
    root,
    governance_root,
    hashes,
):
    existing = completed_stage(
        governance_root,
        args.revision_id,
        stage,
    )
    if args.resume and existing is not None:
        print(
            f"[resume] stage {stage} already complete: "
            f"{existing['dir']}",
            flush=True,
        )
        return existing["analysis"]

    verify_prerequisite(
        governance_root,
        args.revision_id,
        stage,
        hashes["algorithm_source_sha256"],
        hashes["protocol_source_sha256"],
        hashes["frozen_configuration_sha256"],
    )
    if stage in {"sweep", "ablation", "final", "generalization"}:
        verify_frozen(
            governance_root,
            args.revision_id,
            hashes,
        )

    directory = stage_dir(
        governance_root,
        args.revision_id,
        stage,
    )
    directory.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = directory / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(matplotlib_dir)
    record = {
        "status": "running",
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "revision_id": args.revision_id,
        "stage": stage,
        "stage_config": STAGES[stage],
        **hashes,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_steps": [],
    }
    experiment_spec = {
        key: value
        for key, value in record.items()
        if key not in {
            "status",
            "started_at",
            "completed_steps",
        }
    }
    record["experiment_sha256"] = canonical_hash(
        experiment_spec
    )
    manifest_path = directory / "static_stage_manifest.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if (
            not args.resume
            or prior.get("experiment_sha256")
            != record["experiment_sha256"]
        ):
            raise RuntimeError(
                f"{directory} already belongs to another run"
            )
    write_json(manifest_path, record)
    final_lock = None
    started = time.perf_counter()
    try:
        if stage == "tests":
            run_tests(root, directory, env)
            record["completed_steps"].append(
                "capacity_fairness_and_full_unit_tests"
            )
        elif stage in {"smoke", "screen", "converged"}:
            if args.reuse_compatible_h3_from:
                reuse_compatible_h3_stage(
                    governance_root,
                    args.reuse_compatible_h3_from,
                    args.revision_id,
                    stage,
                    directory,
                    hashes["algorithm_source_sha256"],
                )
                record["completed_steps"].append(
                    "behaviorally_audited_h3_artifact_reuse"
                )
            else:
                run_command(
                    reproduction_command(
                        root,
                        directory,
                        stage,
                        MAIN_PROFILE,
                        args.workers,
                        args.resume,
                        args.revision_id,
                    ),
                    root,
                    env,
                )
                record["completed_steps"].append(
                    "training_and_paired_evaluation"
                )
                if stage in {"smoke", "converged"}:
                    run_oracle(
                        root,
                        directory,
                        stage,
                        STAGES[stage]["seeds"],
                        STAGES[stage]["eval_episodes"],
                        env,
                    )
                    record["completed_steps"].append(
                        "capacity_aware_oracle"
                    )
            run_analysis(root, directory, stage, env)
            record["completed_steps"].append("analysis")
        elif stage == "sweep":
            for profile in CAPACITY_PROFILES:
                profile_dir = directory / profile
                profile_dir.mkdir(exist_ok=True)
                run_command(
                    reproduction_command(
                        root,
                        profile_dir,
                        stage,
                        profile,
                        args.workers,
                        args.resume,
                        args.revision_id,
                    ),
                    root,
                    env,
                )
                record["completed_steps"].append(
                    f"{profile}_converged_training_and_evaluation"
                )
                write_json(manifest_path, record)
            run_analysis(root, directory, stage, env)
            record["completed_steps"].append(
                "heterogeneity_curve_analysis"
            )
        elif stage == "ablation":
            prepare_ablation_reuse(
                governance_root,
                args.revision_id,
                directory,
            )
            run_command(
                reproduction_command(
                    root,
                    directory,
                    stage,
                    MAIN_PROFILE,
                    args.workers,
                    True,
                    args.revision_id,
                ),
                root,
                env,
            )
            record["completed_steps"].append(
                "full_our_reused_and_two_ablations_trained"
            )
            run_analysis(root, directory, stage, env)
            record["completed_steps"].append(
                "static_ablation_analysis"
            )
        elif stage == "final":
            final_lock = initialize_final_lock(
                governance_root,
                args.revision_id,
                record["experiment_sha256"],
                args.resume,
            )
            run_command(
                reproduction_command(
                    root,
                    directory,
                    stage,
                    MAIN_PROFILE,
                    args.workers,
                    args.resume,
                    args.revision_id,
                ),
                root,
                env,
            )
            record["completed_steps"].append(
                "untouched_ten_seed_training_and_evaluation"
            )
            run_oracle(
                root,
                directory,
                stage,
                STAGES[stage]["seeds"],
                STAGES[stage]["eval_episodes"],
                env,
            )
            record["completed_steps"].append(
                "final_capacity_aware_oracle"
            )
            run_analysis(root, directory, stage, env)
            record["completed_steps"].append(
                "formal_static_superiority_statistics"
            )
        elif stage == "generalization":
            final_dir = stage_dir(
                governance_root,
                args.revision_id,
                "final",
            )
            run_command(
                [
                    sys.executable,
                    str(root / "evaluate_static_cross_capacity.py"),
                    "--source-suite-dir",
                    str(final_dir),
                    "--output-dir",
                    str(directory),
                    "--labels",
                    ",".join(STAGES[stage]["labels"]),
                    "--seeds",
                    ",".join(
                        str(seed)
                        for seed in STAGES[stage]["seeds"]
                    ),
                    "--profiles",
                    ",".join(CAPACITY_PROFILES),
                    "--episodes",
                    str(STAGES[stage]["eval_episodes"]),
                    "--workers",
                    str(args.workers),
                    *(["--resume"] if args.resume else []),
                ],
                root,
                env,
                log_path=directory / "generalization.log",
            )
            record["completed_steps"].append(
                "h3_checkpoint_zero_shot_h0_h4"
            )
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        record["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        record["total_wall_time_sec"] = (
            time.perf_counter() - started
        )
        write_json(manifest_path, record)
        raise

    result = read_json(analysis_path(directory, stage))
    record["status"] = "complete"
    record["gate_passed"] = result_gate(stage, result)
    record["finished_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    record["total_wall_time_sec"] = time.perf_counter() - started
    write_json(manifest_path, record)
    if stage == "converged" and record["gate_passed"]:
        freeze_algorithm(
            governance_root,
            args.revision_id,
            hashes,
            result,
        )
    if final_lock is not None:
        lock = read_json(final_lock)
        lock["status"] = "complete"
        lock["gate_passed"] = record["gate_passed"]
        lock["closed_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        write_json(final_lock, lock)
    return result


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    governance_root = root / "results" / "static_heterogeneity"
    governance_root.mkdir(parents=True, exist_ok=True)
    hashes = {
        "algorithm_source_sha256": source_hash(
            root,
            ALGORITHM_SOURCE_FILES,
        ),
        "protocol_source_sha256": source_hash(
            root,
            PROTOCOL_SOURCE_FILES,
        ),
        "frozen_configuration_sha256": canonical_hash(
            frozen_protocol_spec()
        ),
    }
    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n=== static stage: {stage} ===", flush=True)
        result = execute_stage(
            stage,
            args,
            root,
            governance_root,
            hashes,
        )
        if not result_gate(stage, result):
            raise RuntimeError(
                f"Static protocol stopped: stage {stage} did not pass"
            )
    print(
        "Static heterogeneity protocol complete: "
        f"{governance_root / args.revision_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
