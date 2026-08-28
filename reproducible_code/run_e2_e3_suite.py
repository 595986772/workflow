#!/usr/bin/env python3
"""Run the governed E2/E3 experiment sequence."""

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


DAOC_LABEL = "guided_full"
OUR_LABEL = "lean_our"
E2_CAPACITIES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
REVISION_MODULES = {
    "initial_version",
    "optimizer_schedule",
    "actor_server_features",
    "cache_value_function",
    "replica_marginal_gain",
    "cache_stability",
    "telemetry_estimator",
    "actor_telemetry_features",
    "cache_update_interval",
    "coordination_frequency",
    "diagnosis_selected_module",
}

STAGES = {
    "smoke": {
        "profile": "e2_smoke",
        "seeds": [1],
        "eval_episodes": 20,
        "labels": [DAOC_LABEL, OUR_LABEL],
        "run_e2": True,
        "run_e3": False,
        "partition": "smoke",
    },
    "screen": {
        "profile": "e2_development",
        "seeds": [1, 2, 3],
        "eval_episodes": 50,
        "labels": [DAOC_LABEL, OUR_LABEL],
        "run_e2": True,
        "run_e3": False,
        "partition": "development",
    },
    "converged": {
        "profile": "e2_converged",
        "seeds": [1, 2, 3],
        "eval_episodes": 100,
        "labels": [DAOC_LABEL, OUR_LABEL],
        "run_e2": True,
        "run_e3": False,
        "partition": "development",
    },
    "dynamic": {
        "profile": None,
        "seeds": [1, 2, 3],
        "eval_episodes": 100,
        "labels": [DAOC_LABEL, OUR_LABEL],
        "run_e2": False,
        "run_e3": True,
        "partition": "development",
        "reuse_stage": "converged",
    },
    "ablation": {
        "profile": "e2_converged",
        "seeds": [1, 2, 3],
        "eval_episodes": 100,
        "labels": [
            DAOC_LABEL,
            OUR_LABEL,
            "our_no_telemetry",
            "our_no_coord_cache",
        ],
        "run_e2": True,
        "run_e3": True,
        "partition": "ablation",
        "reuse_stage": "converged",
        "reuse_labels": [DAOC_LABEL, OUR_LABEL],
    },
    "final": {
        "profile": "e2_converged",
        "seeds": list(range(1, 11)),
        "eval_episodes": 100,
        "labels": [
            "random",
            "nearest",
            "greedy",
            DAOC_LABEL,
            OUR_LABEL,
        ],
        "run_e2": True,
        "run_e3": True,
        "partition": "final_confirmation",
        "reuse_stage": "converged",
        "reuse_labels": [DAOC_LABEL, OUR_LABEL],
        "reuse_seeds": [1, 2, 3],
    },
    "e0_audit": {
        "profile": None,
        "seeds": [1, 2, 3],
        "eval_episodes": 100,
        "labels": [DAOC_LABEL, OUR_LABEL],
        "run_e2": False,
        "run_e3": False,
        "partition": "compatibility",
    },
}

ALGORITHM_SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_rl.py",
    "dqn.py",
    "server.py",
    "simulator.py",
    "user.py",
)
PROTOCOL_SOURCE_FILES = tuple(
    sorted(
        set(ALGORITHM_SOURCE_FILES)
        | {
            "aggregate_reproduction_results.py",
            "audit_e3_gate_feasibility.py",
            "audit_e0_compatibility.py",
            "analyze_e2_e3_ablation.py",
            "analyze_e2_e3_results.py",
            "analyze_strict_environment_suite.py",
            "compare_pd3qn_methods.py",
            "convergence_monitor.py",
            "evaluate_online_stream.py",
            "evaluate_oracle_latency_bound.py",
            "information_protocol.py",
            "oracle_latency_bound.py",
            "run_e2_e3_suite.py",
            "run_independent_experiment.py",
            "run_reproduction_suite.py",
        }
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run governed heterogeneous-cache E2/E3 experiments."
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--revision-id", default="e2r0")
    parser.add_argument("--revision-number", type=int, default=0)
    parser.add_argument("--revision-parent")
    parser.add_argument("--revision-reason", default="initial_version")
    parser.add_argument(
        "--changed-module",
        choices=sorted(REVISION_MODULES),
        default="initial_version",
    )
    parser.add_argument(
        "--expected-metric",
        default="paired_finish_time",
    )
    parser.add_argument(
        "--rejection-condition",
        default="primary_metric_not_improved_or_other_metric_worse_1pct",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.revision_number <= 3:
        raise ValueError("At most three algorithm revisions are allowed")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.revision_number == 0:
        if args.revision_parent is not None:
            raise ValueError("Revision zero cannot have a parent")
        if args.changed_module != "initial_version":
            raise ValueError(
                "Revision zero must use --changed-module initial_version"
            )
    else:
        if not args.revision_parent:
            raise ValueError(
                "Algorithm revisions require --revision-parent"
            )
        if args.revision_reason == "initial_version":
            raise ValueError(
                "Algorithm revisions require a recorded diagnosis/reason"
            )
        if args.changed_module == "initial_version":
            raise ValueError(
                "Algorithm revisions require one diagnosed changed module"
            )
    return args


def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash(root_dir, filenames):
    digest = hashlib.sha256()
    for name in filenames:
        path = root_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def algorithm_source_hash(root_dir):
    return source_hash(root_dir, ALGORITHM_SOURCE_FILES)


def protocol_source_hash(root_dir):
    return source_hash(root_dir, PROTOCOL_SOURCE_FILES)


def frozen_protocol_spec():
    algorithm_configs = {
        config["label"]: config
        for config in ALGORITHMS
        if config["label"] in {DAOC_LABEL, OUR_LABEL}
    }
    method_profiles = {}
    for label in (DAOC_LABEL, OUR_LABEL):
        profile = effective_method_profile(
            PROFILES["e2_converged"],
            label,
        )
        for field in ("seeds", "labels"):
            profile.pop(field, None)
        method_profiles[label] = profile
    return {
        "environment": {
            "num_users": 20,
            "num_servers": 10,
            "num_services": 10,
            "num_tasks_per_user": 10,
            "bandwidth_hz": 15000,
            "capacity_multiset": E2_CAPACITIES,
            "total_capacity": 10,
            "capacity_assignment": (
                "seeded_independent_rng_unrelated_to_deployment"
            ),
        },
        "methods": algorithm_configs,
        "training_and_evaluation": method_profiles,
        "e3": {
            "episodes": 100,
            "shift_episode": 51,
            "load_multiplier": 4.0,
            "shifted_servers": "one_per_capacity_class",
            "recovery_window": 5,
            "recovery_tolerance": 1.05,
            "network_weights_frozen": True,
            "native_online_state_updates": True,
        },
        "statistics": {
            "paired_ci": 0.95,
            "wilcoxon_alternative": "our_better",
            "wilcoxon_alpha": 0.05,
            "screen_seed_wins": "at_least_2_of_3",
            "converged_seed_wins": "3_of_3",
            "converged_mean_improvement_percent": 5.0,
            "converged_p95": "strictly_better",
            "dynamic_seed_wins": "at_least_2_of_3_for_both_metrics",
            "final_seed_wins": "at_least_7_of_10",
            "final_seed_unit": "per_seed_paired_mean",
        },
        "stage_protocol": {
            "stages": list(STAGES),
            "final_seeds": list(range(1, 11)),
            "development_seeds": [1, 2, 3],
            "final_is_independent_holdout": False,
            "maximum_single_module_revisions": 3,
            "e1_rerun": False,
        },
    }


def completed_stage(governance_root, revision_id, stage):
    stage_dir = governance_root / revision_id / stage
    manifest_path = stage_dir / "e2_e3_manifest.json"
    analysis_path = stage_dir / "e2_e3_analysis.json"
    if not manifest_path.exists() or not analysis_path.exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        return None
    return {
        "dir": stage_dir,
        "manifest": manifest,
        "analysis": read_json(analysis_path),
    }


def latest_diagnostic_stage(governance_root, revision_id):
    for stage in ("dynamic", "converged", "screen"):
        completed = completed_stage(
            governance_root,
            revision_id,
            stage,
        )
        if completed is not None:
            return stage, completed
    return None, None


def validate_revision_parent(args, governance_root):
    if args.revision_number == 0:
        return None
    parent_stage, parent = latest_diagnostic_stage(
        governance_root,
        args.revision_parent,
    )
    if parent is None:
        raise RuntimeError(
            "A revised algorithm requires its parent's completed "
            "screen, converged, or dynamic diagnosis"
        )
    parent_record = read_json(
        parent["dir"] / "REVISION_RECORD.json"
    )
    if (
        int(parent_record["revision_number"]) + 1
        != args.revision_number
    ):
        raise RuntimeError(
            "Revision number must increment its parent by exactly one"
        )
    diagnosed_module = parent["analysis"]["diagnosis"].get(
        "module_id"
    )
    if diagnosed_module in {None, "none"}:
        raise RuntimeError(
            "The parent diagnosis does not permit another algorithm module"
        )
    if args.changed_module != diagnosed_module:
        raise RuntimeError(
            "The revision may change only the module selected by the "
            f"parent diagnosis: {diagnosed_module}"
        )
    diagnosed_metric = parent["analysis"]["diagnosis"].get(
        "expected_metric"
    )
    if (
        diagnosed_metric is not None
        and args.expected_metric != diagnosed_metric
    ):
        raise RuntimeError(
            "The revision expected metric must match the parent "
            f"diagnosis: {diagnosed_metric}"
        )
    parent["stage"] = parent_stage
    return parent


def verify_stage_prerequisites(
    stage,
    governance_root,
    revision_id,
    algorithm_hash,
    protocol_hash,
    frozen_config_hash,
):
    previous_stage = {
        "screen": "smoke",
        "converged": "screen",
        "dynamic": "converged",
        "ablation": "dynamic",
        "final": "ablation",
        "e0_audit": "final",
    }.get(stage)
    if previous_stage is None:
        return
    previous = completed_stage(
        governance_root,
        revision_id,
        previous_stage,
    )
    if previous is None:
        raise RuntimeError(
            f"{stage} requires a completed {previous_stage} stage"
        )
    manifest = previous["manifest"]
    for key, expected in (
        ("algorithm_source_sha256", algorithm_hash),
        ("protocol_source_sha256", protocol_hash),
        ("frozen_configuration_sha256", frozen_config_hash),
    ):
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{previous_stage} no longer matches the current "
                "algorithm/protocol configuration"
            )
    analysis = previous["analysis"]
    revision = analysis.get("revision_comparison")
    if revision is not None and not revision["retained"]:
        raise RuntimeError(
            f"{stage} requires a retained {previous_stage} revision"
        )
    if stage in {"screen", "converged", "dynamic"}:
        if not analysis["e2"]["gate_passed"]:
            raise RuntimeError(
                f"{stage} requires {previous_stage} to pass its gate"
            )
    if stage == "ablation":
        if (
            not analysis["e2"]["gate_passed"]
            or analysis["e3"] is None
            or not analysis["e3"]["gate_passed"]
        ):
            raise RuntimeError(
                "Ablation requires the development version to pass E2 "
                "convergence and E3 adaptation gates"
            )
    if stage == "e0_audit":
        formal_final = analysis.get("formal_final")
        if (
            formal_final is None
            or not formal_final["superiority_claim_passed"]
            or analysis["e3"] is None
            or not analysis["e3"]["gate_passed"]
        ):
            raise RuntimeError(
                "E0 compatibility evidence is run only after final E2/E3 "
                "success"
            )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command, root_dir, env, log_path=None):
    print(" ".join(str(item) for item in command), flush=True)
    if log_path is None:
        subprocess.run(
            command,
            cwd=root_dir,
            env=env,
            check=True,
        )
        return
    with log_path.open("w", encoding="utf-8") as output_file:
        subprocess.run(
            command,
            cwd=root_dir,
            env=env,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def verify_frozen_algorithm(
    stage,
    governance_root,
    algorithm_hash,
    protocol_hash,
    frozen_config_hash,
    revision_id,
):
    if stage not in {"ablation", "final", "e0_audit"}:
        return
    freeze_path = governance_root / "FROZEN_ALGORITHM.json"
    if not freeze_path.exists():
        raise RuntimeError(
            f"{stage} requires passed E2/E3 development and a frozen algorithm"
        )
    frozen = read_json(freeze_path)
    if (
        frozen["algorithm_source_sha256"] != algorithm_hash
        or frozen["protocol_source_sha256"] != protocol_hash
        or frozen["frozen_configuration_sha256"]
        != frozen_config_hash
        or frozen["revision_id"] != revision_id
    ):
        raise RuntimeError(
            "Algorithm, protocol, statistics, environment, hyperparameters, "
            "or revision changed after E2/E3 development was frozen"
        )


def initialize_final_lock(
    stage,
    governance_root,
    experiment_hash,
    resume,
):
    if stage != "final":
        return None
    path = governance_root / "FINAL_LOCK.json"
    if path.exists():
        lock = read_json(path)
        if (
            not resume
            or lock["experiment_sha256"] != experiment_hash
            or lock["status"] == "complete"
        ):
            raise RuntimeError(
                "The ten-seed final confirmation has already been opened; "
                "it cannot be retuned or rerun"
            )
        return path
    write_json(
        path,
        {
            "status": "running",
            "experiment_sha256": experiment_hash,
            "opened_at": datetime.now().isoformat(timespec="seconds"),
            "policy": "single_use_final_confirmation_seeds_1_to_10",
            "development_seeds_reused": [1, 2, 3],
            "independent_holdout": False,
        },
    )
    return path


def run_static_oracle(
    root_dir,
    suite_dir,
    stage,
    seeds,
    episodes,
    env,
):
    command = [
        sys.executable,
        str(root_dir / "evaluate_oracle_latency_bound.py"),
        "--our-suite-dir",
        str(suite_dir),
        "--our-label",
        OUR_LABEL,
        "--daoc-suite-dir",
        str(suite_dir),
        "--daoc-label",
        DAOC_LABEL,
        "--output-dir",
        str(suite_dir / "oracle"),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--episodes",
        str(episodes),
        "--exact-check-scenarios",
        "1" if stage == "converged" else "0",
    ]
    run_command(command, root_dir, env)


def run_e3(
    root_dir,
    suite_dir,
    stage,
    seeds,
    workers,
    env,
    resume,
):
    multipliers = [2.0, 4.0, 6.0] if stage == "final" else [4.0]
    for multiplier in multipliers:
        command = [
            sys.executable,
            str(root_dir / "evaluate_online_stream.py"),
            "--suite-dir",
            str(suite_dir),
            "--labels",
            f"{DAOC_LABEL},{OUR_LABEL}",
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--episodes",
            "100",
            "--shift-episode",
            "51",
            "--regime",
            "server_load_shift",
            "--load-multiplier",
            str(multiplier),
            "--recovery-window",
            "5",
            "--recovery-tolerance",
            "1.05",
            "--workers",
            str(workers),
        ]
        if resume:
            command.append("--resume")
        run_command(command, root_dir, env)


def run_ablation_e3(
    root_dir,
    suite_dir,
    seeds,
    env,
    resume,
):
    for label in (
        "our_no_telemetry",
        "our_no_coord_cache",
    ):
        for seed in seeds:
            command = [
                sys.executable,
                str(root_dir / "evaluate_online_stream.py"),
                "--run-dir",
                str(run_dir(suite_dir, label, seed)),
                "--episodes",
                "100",
                "--shift-episode",
                "51",
                "--regime",
                "server_load_shift",
                "--load-multiplier",
                "4",
                "--recovery-window",
                "5",
                "--recovery-tolerance",
                "1.05",
                "--quiet",
            ]
            if resume:
                command.append("--resume")
            run_command(command, root_dir, env)


def run_dir(suite_dir, label, seed):
    return suite_dir / "runs" / label / f"seed_{seed}"


def link_reused_runs(
    source_suite_dir,
    target_suite_dir,
    labels,
    seeds,
):
    linked = []
    for label in labels:
        for seed in seeds:
            source = run_dir(source_suite_dir, label, seed).resolve()
            target = run_dir(target_suite_dir, label, seed)
            if not source.exists():
                raise RuntimeError(
                    f"Cannot reuse missing run: {source}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source:
                    raise RuntimeError(
                        f"Existing reuse link points elsewhere: {target}"
                    )
            elif target.exists():
                raise RuntimeError(
                    f"Refusing to replace existing run with reuse link: {target}"
                )
            else:
                relative_source = os.path.relpath(
                    source,
                    start=target.parent.resolve(),
                )
                target.symlink_to(
                    relative_source,
                    target_is_directory=True,
                )
            linked.append(
                {
                    "label": label,
                    "seed": seed,
                    "source": str(source),
                    "target": str(target),
                }
            )
    return linked


def link_reused_artifact(source, target):
    source = Path(source).resolve()
    target = Path(target)
    if not source.exists():
        return None
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(
                f"Existing reuse link points elsewhere: {target}"
            )
    elif target.exists():
        raise RuntimeError(
            f"Refusing to replace existing artifact: {target}"
        )
    else:
        relative_source = os.path.relpath(
            source,
            start=target.parent.resolve(),
        )
        target.symlink_to(
            relative_source,
            target_is_directory=source.is_dir(),
        )
    return {
        "source": str(source),
        "target": str(target),
    }


def link_dynamic_metadata(source_suite_dir, target_suite_dir):
    target_suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = link_reused_artifact(
        source_suite_dir / "suite_manifest.json",
        target_suite_dir / "suite_manifest.json",
    )
    if manifest is None:
        raise RuntimeError(
            f"Cannot reuse missing suite manifest: {source_suite_dir}"
        )
    oracle = link_reused_artifact(
        source_suite_dir / "oracle",
        target_suite_dir / "oracle",
    )
    return {
        "suite_manifest": manifest,
        "oracle": oracle,
    }


def run_e3_gate_feasibility_audit(root_dir, suite_dir, env):
    run_command(
        [
            sys.executable,
            str(root_dir / "audit_e3_gate_feasibility.py"),
            "--suite-dir",
            str(suite_dir),
        ],
        root_dir,
        env,
    )
    return read_json(suite_dir / "E3_GATE_FEASIBILITY.json")


def parent_stage_dir(governance_root, revision_parent, stage):
    if revision_parent is None:
        return None
    candidate = governance_root / revision_parent / stage
    if (
        (candidate / "e2_e3_manifest.json").exists()
        and (candidate / "e2_e3_analysis.json").exists()
    ):
        return candidate
    return None


def run_e0_compatibility_stage(
    root_dir,
    suite_dir,
    workers,
    env,
    resume,
):
    audit_command = [
        sys.executable,
        str(root_dir / "audit_e0_compatibility.py"),
        "--output-dir",
        str(suite_dir),
        "--historical-root",
        str(root_dir / "results"),
    ]
    run_command(audit_command, root_dir, env)
    audit = read_json(suite_dir / "E0_COMPATIBILITY_AUDIT.json")
    if audit["behavior_equivalent"]:
        return {
            "audit": audit,
            "e0_reproduction_run": False,
        }

    e0_dir = suite_dir / "e0_reproduction"
    command = [
        sys.executable,
        str(root_dir / "run_strict_environment_suite.py"),
        "--stage",
        "converged",
        "--environments",
        "e0_original",
        "--suite-dir",
        str(e0_dir),
        "--workers",
        str(workers),
        "--daoc-label",
        DAOC_LABEL,
        "--our-label",
        OUR_LABEL,
    ]
    if resume:
        command.append("--resume")
    run_command(command, root_dir, env)
    return {
        "audit": audit,
        "e0_reproduction_run": True,
        "e0_reproduction_dir": str(e0_dir),
    }


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    stage = STAGES[args.stage]
    governance_root = root_dir / "results" / "e2_e3"
    governance_root.mkdir(parents=True, exist_ok=True)
    algorithm_hash = algorithm_source_hash(root_dir)
    protocol_hash = protocol_source_hash(root_dir)
    protocol_spec = frozen_protocol_spec()
    frozen_config_hash = canonical_hash(protocol_spec)
    parent_diagnosis = validate_revision_parent(
        args,
        governance_root,
    )
    verify_stage_prerequisites(
        args.stage,
        governance_root,
        args.revision_id,
        algorithm_hash,
        protocol_hash,
        frozen_config_hash,
    )
    suite_dir = (
        args.suite_dir
        if args.suite_dir is not None
        else governance_root / args.revision_id / args.stage
    ).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    revision = {
        "revision_id": args.revision_id,
        "revision_number": args.revision_number,
        "parent": args.revision_parent,
        "reason": args.revision_reason,
        "changed_module": args.changed_module,
        "expected_metric": args.expected_metric,
        "rejection_condition": args.rejection_condition,
        "stage": args.stage,
        "seed_partition": stage["partition"],
        "algorithm_source_sha256": algorithm_hash,
        "protocol_source_sha256": protocol_hash,
        "frozen_configuration_sha256": frozen_config_hash,
        "stage_config": stage,
        "capacity_multiset": E2_CAPACITIES,
        "parent_diagnostic_stage": (
            parent_diagnosis["stage"]
            if parent_diagnosis is not None
            else None
        ),
    }
    experiment_hash = canonical_hash(revision)
    revision["experiment_sha256"] = experiment_hash
    revision_path = suite_dir / "REVISION_RECORD.json"
    if revision_path.exists():
        existing = read_json(revision_path)
        if (
            not args.resume
            or existing["experiment_sha256"] != experiment_hash
        ):
            raise RuntimeError(
                "Result directory already belongs to another revision/run"
            )
    else:
        write_json(revision_path, revision)

    verify_frozen_algorithm(
        args.stage,
        governance_root,
        algorithm_hash,
        protocol_hash,
        frozen_config_hash,
        args.revision_id,
    )
    final_lock = initialize_final_lock(
        args.stage,
        governance_root,
        experiment_hash,
        args.resume,
    )

    matplotlib_dir = suite_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(matplotlib_dir)
    manifest_path = suite_dir / "e2_e3_manifest.json"
    manifest = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        **revision,
        "completed_steps": [],
        "e0_e1_policy": (
            "not prerequisites; E0 only after successful E2/E3 if needed; "
            "E1 is not rerun"
        ),
        "reused_runs": [],
    }
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    seeds = stage["seeds"]
    seed_text = ",".join(str(seed) for seed in seeds)

    reuse_stage = stage.get("reuse_stage")
    if reuse_stage is not None:
        source_suite = (
            governance_root / args.revision_id / reuse_stage
        )
        reuse_labels = stage.get(
            "reuse_labels",
            stage["labels"],
        )
        reuse_seeds = stage.get("reuse_seeds", seeds)
        manifest["reused_runs"] = link_reused_runs(
            source_suite,
            suite_dir,
            reuse_labels,
            reuse_seeds,
        )
        if args.stage == "dynamic":
            dynamic_metadata = link_dynamic_metadata(
                source_suite,
                suite_dir,
            )
            manifest["reused_suite_manifest"] = dynamic_metadata[
                "suite_manifest"
            ]
            if dynamic_metadata["oracle"] is not None:
                manifest["reused_oracle"] = dynamic_metadata["oracle"]
        manifest["completed_steps"].append("checkpoint_runs_reused")
        write_json(manifest_path, manifest)

    if args.stage == "e0_audit":
        result = run_e0_compatibility_stage(
            root_dir,
            suite_dir,
            args.workers,
            env,
            args.resume,
        )
        manifest["e0_compatibility"] = result
        manifest["completed_steps"].append(
            "e0_compatibility_audit"
        )
        if result["e0_reproduction_run"]:
            manifest["completed_steps"].append(
                "e0_daoc_our_reproduction"
            )
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        manifest["total_wall_time_sec"] = (
            time.perf_counter() - started
        )
        write_json(manifest_path, manifest)
        print(f"E0 compatibility suite: {suite_dir}")
        return

    if stage["run_e2"]:
        reproduction_command = [
            sys.executable,
            str(root_dir / "run_reproduction_suite.py"),
            "--profile",
            stage["profile"],
            "--suite-dir",
            str(suite_dir),
            "--seeds",
            seed_text,
            "--labels",
            ",".join(stage["labels"]),
            "--workers",
            str(args.workers),
            "--revision-id",
            args.revision_id,
            "--revision-reason",
            args.revision_reason,
            "--revision-changed-module",
            args.changed_module,
            "--revision-expected-metric",
            args.expected_metric,
            "--revision-rejection-condition",
            args.rejection_condition,
            "--seed-partition",
            stage["partition"],
        ]
        if args.revision_parent:
            reproduction_command.extend(
                ["--revision-parent", args.revision_parent]
            )
        if args.resume or reuse_stage is not None:
            reproduction_command.append("--resume")
        run_command(reproduction_command, root_dir, env)
        manifest["completed_steps"].append(
            "e2_training_and_evaluation"
        )
        write_json(manifest_path, manifest)

        if args.stage != "ablation":
            run_static_oracle(
                root_dir,
                suite_dir,
                args.stage,
                seeds,
                stage["eval_episodes"],
                env,
            )
            manifest["completed_steps"].append(
                "capacity_aware_oracle"
            )
            write_json(manifest_path, manifest)

    if stage["run_e3"] and args.stage not in {"ablation"}:
        run_e3(
            root_dir,
            suite_dir,
            args.stage,
            seeds,
            args.workers,
            env,
            args.resume or args.stage == "final",
        )
        manifest["completed_steps"].append("e3_load_shift")
        write_json(manifest_path, manifest)
    elif args.stage == "ablation":
        run_e3(
            root_dir,
            suite_dir,
            "dynamic",
            seeds,
            args.workers,
            env,
            True,
        )
        run_ablation_e3(
            root_dir,
            suite_dir,
            seeds,
            env,
            args.resume,
        )
        manifest["completed_steps"].append("e3_ablation_streams")
        write_json(manifest_path, manifest)

    analysis_command = [
        sys.executable,
        str(root_dir / "analyze_e2_e3_results.py"),
        "--suite-dir",
        str(suite_dir),
        "--seeds",
        seed_text,
        "--stage",
        args.stage,
        "--daoc-label",
        DAOC_LABEL,
        "--our-label",
        OUR_LABEL,
        "--expected-metric",
        args.expected_metric,
    ]
    parent_suite = parent_stage_dir(
        governance_root,
        args.revision_parent,
        args.stage,
    )
    if (
        args.expected_metric.startswith("e3_")
        and args.stage != "dynamic"
    ):
        parent_suite = None
    if parent_suite is not None:
        analysis_command.extend(
            [
                "--parent-suite-dir",
                str(parent_suite),
            ]
        )
    run_command(analysis_command, root_dir, env)
    manifest["completed_steps"].append("analysis_and_diagnosis")
    if args.stage == "ablation":
        run_command(
            [
                sys.executable,
                str(root_dir / "analyze_e2_e3_ablation.py"),
                "--suite-dir",
                str(suite_dir),
                "--seeds",
                seed_text,
            ],
            root_dir,
            env,
        )
        manifest["completed_steps"].append(
            "minimal_ablation_table_and_figure"
        )
    analysis = read_json(suite_dir / "e2_e3_analysis.json")
    e3_gate_feasibility = None
    if (
        args.stage == "dynamic"
        and analysis["e3"] is not None
        and not analysis["e3"]["gate_passed"]
    ):
        e3_gate_feasibility = run_e3_gate_feasibility_audit(
            root_dir,
            suite_dir,
            env,
        )
        manifest["e3_gate_feasibility"] = e3_gate_feasibility
        manifest["completed_steps"].append(
            "e3_strict_recovery_gate_feasibility_audit"
        )

    if args.stage in {"screen", "converged", "dynamic"}:
        revision_comparison = analysis.get("revision_comparison")
        e3_passed = (
            True
            if args.stage != "dynamic"
            else (
                analysis["e3"] is not None
                and analysis["e3"]["gate_passed"]
            )
        )
        accepted = (
            analysis["e2"]["gate_passed"]
            and e3_passed
            and (
                revision_comparison is None
                or revision_comparison["retained"]
            )
        )
        manifest["revision_decision"] = (
            (
                "accepted_and_frozen"
                if args.stage == "dynamic"
                else "accepted_for_next_stage"
            )
            if accepted
            else (
                "protocol_gate_unattainable_stop"
                if (
                    e3_gate_feasibility is not None
                    and not e3_gate_feasibility[
                        "strict_recovery_gate_attainable"
                    ]
                )
                else (
                    "revision_budget_exhausted"
                    if args.revision_number == 3
                    else "diagnose_then_revise_one_module"
                )
            )
        )

    if args.stage == "dynamic":
        revision_comparison = analysis.get("revision_comparison")
        if (
            analysis["e2"]["gate_passed"]
            and analysis["e3"] is not None
            and analysis["e3"]["gate_passed"]
            and (
                revision_comparison is None
                or revision_comparison["retained"]
            )
        ):
            write_json(
                governance_root / "FROZEN_ALGORITHM.json",
                {
                    "status": "frozen",
                    "revision_id": args.revision_id,
                    "algorithm_source_sha256": algorithm_hash,
                    "protocol_source_sha256": protocol_hash,
                    "frozen_configuration_sha256": (
                        frozen_config_hash
                    ),
                    "frozen_configuration": protocol_spec,
                    "experiment_sha256": experiment_hash,
                    "development_seeds": seeds,
                    "independent_holdout": False,
                    "frozen_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                },
            )
            manifest["completed_steps"].append("algorithm_frozen")
        else:
            manifest["freeze_decision"] = "rejected"

    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    manifest["total_wall_time_sec"] = time.perf_counter() - started
    write_json(manifest_path, manifest)
    if final_lock is not None:
        lock = read_json(final_lock)
        lock["status"] = "complete"
        lock["closed_at"] = manifest["finished_at"]
        lock["formal_final"] = analysis["formal_final"]
        lock["e3_gate_passed"] = (
            analysis["e3"] is not None
            and analysis["e3"]["gate_passed"]
        )
        write_json(final_lock, lock)
    print(f"E2/E3 suite: {suite_dir}")


if __name__ == "__main__":
    main()
