"""Frozen protocol for DAOC with the OUR coordinated-cache subsystem."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import run_reproduction_suite as reproduction
from pegasus_p6_protocol import (
    CAPACITY_MULTISET,
    CAPACITY_NAMESPACE,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_SEEDS,
    P3_FINAL_DIR,
    RESULT_ROOT as P6_RESULT_ROOT,
    TASK_LIMIT_INCLUDING_DUMMY,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_p8_daoc_our_coord_cache_v1"
DAOC_LABEL = reproduction.DAOC_PAPER_LABEL
DAOC_COORD_LABEL = "daoc_our_coord_cache"

RESULT_ROOT = ROOT / "results/pegasus_pscale/p8_daoc_our_coord_cache"
SMOKE_DIR = RESULT_ROOT / "smoke"
FINAL_DIR = RESULT_ROOT / "final"
ANALYSIS_DIR = RESULT_ROOT / "analysis"
SMOKE_SEEDS = (42,)

P6_FINAL_LOCK = P6_RESULT_ROOT / "FINAL_LOCK.json"
P6_SUMMARY = P6_RESULT_ROOT / "analysis/pegasus_p6_summary.json"
P7_ROOT = ROOT / "results/pegasus_pscale/p7_std_cache_discrete_sac"
P7_FINAL_LOCK = P7_ROOT / "FINAL_LOCK.json"
P7_SUMMARY = P7_ROOT / "analysis/sac_std_cache_extension_summary.json"

PROFILE_SMOKE = "pegasus_daoc_coord_cache_smoke"
PROFILE_CONVERGED = "pegasus_daoc_coord_cache_converged"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algorithm_config(label):
    return next(
        config
        for config in reproduction.ALGORITHMS
        if config["label"] == label
    )


def daoc_coord_cache_config():
    """Keep DAOC intact and replace only its independent cache subsystem."""
    config = deepcopy(algorithm_config(DAOC_LABEL))
    config.update(
        {
            "label": DAOC_COORD_LABEL,
            "display_name": "DAOC+OUR-CoordCache",
            "cache_policy": "critical_path_joint",
            "cache_server_quality": True,
            "cache_coverage_constraint": True,
            "cache_dependency_awareness": True,
            "active_modules": [
                "distributed_dqn",
                "paper_decaying_action_guidance",
                "causal_dependency_aware_joint_cache",
                "history_based_cache_server_quality",
                "scarcity_aware_service_coverage_constraint",
            ],
            "excluded_modules": [
                "daoc_paper_popularity_cost_cache",
            ],
        }
    )
    return config


def register_suite_extension():
    existing = [
        config
        for config in reproduction.ALGORITHMS
        if config["label"] == DAOC_COORD_LABEL
    ]
    expected = daoc_coord_cache_config()
    if existing:
        if existing[0] != expected:
            raise RuntimeError("Conflicting DAOC coordinated-cache configuration")
    else:
        reproduction.ALGORITHMS.append(expected)

    smoke = deepcopy(
        reproduction.PROFILES["pegasus_paper_closure_smoke"]
    )
    smoke["labels"] = [DAOC_COORD_LABEL]
    smoke["method_overrides"] = {}
    reproduction.PROFILES[PROFILE_SMOKE] = smoke

    converged = deepcopy(
        reproduction.PROFILES["pegasus_paper_closure_converged"]
    )
    converged["labels"] = [DAOC_COORD_LABEL]
    # DAOC itself has no method-specific override in this profile.
    converged["method_overrides"] = {}
    reproduction.PROFILES[PROFILE_CONVERGED] = converged


def _effective_without_labels(profile_name, label):
    effective = reproduction.effective_method_profile(
        reproduction.PROFILES[profile_name],
        label,
    )
    effective.pop("labels", None)
    return effective


def validate_protocol(require_p7=True):
    register_suite_extension()
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus-B8 dataset hash mismatch")
    if len(CAPACITY_MULTISET) != 10 or sum(CAPACITY_MULTISET) != 8:
        raise RuntimeError("P8 requires the frozen B8 capacity profile")
    if set(SMOKE_SEEDS) & set(FINAL_SEEDS):
        raise RuntimeError("Smoke and final seeds must be disjoint")

    p6_lock = read_json(P6_FINAL_LOCK)
    if p6_lock.get("status") != "complete":
        raise RuntimeError("P6 lock is incomplete")
    if p6_lock.get("summary_sha256") != sha256_file(P6_SUMMARY):
        raise RuntimeError("P6 summary no longer matches its final lock")
    if require_p7:
        p7_lock = read_json(P7_FINAL_LOCK)
        if p7_lock.get("status") != "complete":
            raise RuntimeError("P7 must complete before P8 starts")
        if p7_lock.get("summary_sha256") != sha256_file(P7_SUMMARY):
            raise RuntimeError("P7 summary no longer matches its final lock")

    daoc = algorithm_config(DAOC_LABEL)
    coordinated = algorithm_config(DAOC_COORD_LABEL)
    shared_keys = (
        "algorithm",
        "family",
        "beta",
        "beta_min",
        "beta_decay",
        "reward_mode",
        "training_objective",
    )
    for key in shared_keys:
        if daoc.get(key) != coordinated.get(key):
            raise RuntimeError(f"DAOC+CoordCache unexpectedly changes {key}")
    if coordinated["cache_policy"] != "critical_path_joint":
        raise RuntimeError("DAOC+CoordCache must use the OUR cache policy")
    if coordinated.get("cache_coverage_constraint") is not True:
        raise RuntimeError("DAOC+CoordCache must retain OUR service coverage")
    if coordinated.get("cache_dependency_awareness") is not True:
        raise RuntimeError("DAOC+CoordCache must retain dependency cache scoring")
    if coordinated.get("cache_server_quality") is not True:
        raise RuntimeError("DAOC+CoordCache must retain causal cache quality")

    daoc_profile = _effective_without_labels(
        "pegasus_paper_closure_converged",
        DAOC_LABEL,
    )
    coordinated_profile = _effective_without_labels(
        PROFILE_CONVERGED,
        DAOC_COORD_LABEL,
    )
    if daoc_profile != coordinated_profile:
        raise RuntimeError("DAOC training profiles are not exactly matched")

    parent_locks = {"p6": sha256_file(P6_FINAL_LOCK)}
    if require_p7:
        parent_locks["p7"] = sha256_file(P7_FINAL_LOCK)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "new_method": DAOC_COORD_LABEL,
        "matched_method": DAOC_LABEL,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "families": list(FAMILIES),
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "capacity_namespace": CAPACITY_NAMESPACE,
        "smoke_seeds": list(SMOKE_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "evaluation_episodes": EVALUATION_EPISODES,
        "controlled_difference": {
            "daoc_cache": "paper_popularity_cost_ema",
            "our_cache": "critical_path_joint",
            "all_daoc_state_scheduler_guidance_reward_and_training_settings_matched": True,
        },
        "parent_locks": parent_locks,
        "governance": {
            "daoc_and_our_are_not_retrained": True,
            "new_control_trains_from_scratch": True,
            "seeds_51_to_60_are_paired_confirmation_not_holdout": True,
            "no_post_result_hyperparameter_tuning": True,
        },
    }


__all__ = [
    "ANALYSIS_DIR",
    "CAPACITY_MULTISET",
    "CAPACITY_NAMESPACE",
    "DAOC_COORD_LABEL",
    "DAOC_LABEL",
    "DATASET_PATH",
    "EVALUATION_EPISODES",
    "EXPECTED_DATASET_SHA256",
    "FAMILIES",
    "FINAL_DIR",
    "FINAL_SEEDS",
    "P3_FINAL_DIR",
    "P6_SUMMARY",
    "P7_SUMMARY",
    "PROFILE_CONVERGED",
    "PROFILE_SMOKE",
    "PROTOCOL_VERSION",
    "RESULT_ROOT",
    "SMOKE_DIR",
    "SMOKE_SEEDS",
    "TASK_LIMIT_INCLUDING_DUMMY",
    "algorithm_config",
    "daoc_coord_cache_config",
    "register_suite_extension",
    "sha256_file",
    "validate_protocol",
]
