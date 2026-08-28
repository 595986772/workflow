"""Frozen protocol for the Pegasus-B8 standard-cache Discrete SAC extension."""

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
    P5_SAC_DIR,
    RESULT_ROOT as P6_RESULT_ROOT,
    TASK_LIMIT_INCLUDING_DUMMY,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_p7_std_cache_discrete_sac_v1"
STD_SAC_LABEL = "discrete_sac_std_cache"
COORD_SAC_LABEL = reproduction.COORD_DISCRETE_SAC_LABEL

RESULT_ROOT = ROOT / "results/pegasus_pscale/p7_std_cache_discrete_sac"
SMOKE_DIR = RESULT_ROOT / "smoke"
FINAL_DIR = RESULT_ROOT / "final"
ANALYSIS_DIR = RESULT_ROOT / "analysis"
SMOKE_SEEDS = (41,)

P5_FINAL_LOCK = ROOT / "results/pegasus_pscale/p5_baseline_extension/FINAL_LOCK.json"
P6_FINAL_LOCK = P6_RESULT_ROOT / "FINAL_LOCK.json"
P6_SUMMARY = P6_RESULT_ROOT / "analysis/pegasus_p6_summary.json"

PROFILE_SMOKE = "pegasus_std_cache_sac_smoke"
PROFILE_CONVERGED = "pegasus_std_cache_sac_converged"


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


def standard_cache_sac_config():
    """Change only the cache subsystem of the frozen coordinated SAC."""
    config = deepcopy(algorithm_config(COORD_SAC_LABEL))
    config.update(
        {
            "label": STD_SAC_LABEL,
            "display_name": "DiscreteSAC-StdCache",
            "cache_policy": "popularity_ema",
            "cache_server_quality": False,
            "cache_coverage_constraint": False,
            "cache_dependency_awareness": False,
            "active_modules": [
                "pairwise_categorical_discrete_sac",
                "automatic_entropy_temperature",
                "causal_history_telemetry",
                "independent_popularity_ema_cache",
            ],
            "excluded_modules": [
                "causal_dependency_aware_joint_cache",
                "scarcity_aware_service_coverage_constraint",
                "coordinated_replica_diversity",
                "history_based_cache_server_quality",
            ],
        }
    )
    return config


def register_suite_extension():
    """Register the extension without editing the locked P5/P6 suite source."""
    existing = [
        config
        for config in reproduction.ALGORITHMS
        if config["label"] == STD_SAC_LABEL
    ]
    expected = standard_cache_sac_config()
    if existing:
        if existing[0] != expected:
            raise RuntimeError("Conflicting standard-cache SAC configuration")
    else:
        reproduction.ALGORITHMS.append(expected)

    smoke = deepcopy(
        reproduction.PROFILES["pegasus_baseline_sac_smoke"]
    )
    smoke["labels"] = [STD_SAC_LABEL]
    reproduction.PROFILES[PROFILE_SMOKE] = smoke

    converged = deepcopy(
        reproduction.PROFILES["pegasus_baseline_sac_converged"]
    )
    converged["labels"] = [STD_SAC_LABEL]
    coordinated_override = converged.get("method_overrides", {}).get(
        COORD_SAC_LABEL,
        {},
    )
    converged["method_overrides"] = {
        STD_SAC_LABEL: deepcopy(coordinated_override)
    }
    reproduction.PROFILES[PROFILE_CONVERGED] = converged


def validate_protocol():
    register_suite_extension()
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus-B8 dataset hash mismatch")
    if len(CAPACITY_MULTISET) != 10 or sum(CAPACITY_MULTISET) != 8:
        raise RuntimeError("The extension requires the frozen B8 capacity profile")
    if set(SMOKE_SEEDS) & set(FINAL_SEEDS):
        raise RuntimeError("Smoke and final seeds must be disjoint")

    p5_lock = read_json(P5_FINAL_LOCK)
    p6_lock = read_json(P6_FINAL_LOCK)
    if p5_lock.get("status") != "complete":
        raise RuntimeError("P5 coordinated-SAC lock is incomplete")
    if p6_lock.get("status") != "complete":
        raise RuntimeError("P6 result lock is incomplete")
    if p6_lock.get("summary_sha256") != sha256_file(P6_SUMMARY):
        raise RuntimeError("P6 summary no longer matches its final lock")

    coordinated = algorithm_config(COORD_SAC_LABEL)
    standard = algorithm_config(STD_SAC_LABEL)
    shared_keys = (
        "algorithm",
        "family",
        "beta",
        "beta_min",
        "beta_decay",
        "reward_mode",
        "potential_reward_weight",
        "gamma",
        "n_step",
        "entropy_coefficient",
        "sac_target_entropy_ratio",
        "sac_target_tau",
        "historical_feedback_guidance",
        "adaptive_guidance_gate",
        "training_objective",
    )
    for key in shared_keys:
        if standard.get(key) != coordinated.get(key):
            raise RuntimeError(f"Standard SAC unexpectedly changes {key}")
    if standard["cache_policy"] != "popularity_ema":
        raise RuntimeError("Standard SAC must use independent popularity EMA")
    if standard.get("cache_coverage_constraint", True):
        raise RuntimeError("Standard SAC must not use coordinated coverage")
    if standard.get("cache_dependency_awareness", True):
        raise RuntimeError("Standard SAC must not use dependency cache scoring")
    if standard.get("cache_server_quality", True):
        raise RuntimeError("Standard SAC must not use coordinated quality scoring")

    coordinated_profile = reproduction.PROFILES[
        "pegasus_baseline_sac_converged"
    ]
    standard_profile = reproduction.PROFILES[PROFILE_CONVERGED]
    coordinated_effective = reproduction.effective_method_profile(
        coordinated_profile,
        COORD_SAC_LABEL,
    )
    standard_effective = reproduction.effective_method_profile(
        standard_profile,
        STD_SAC_LABEL,
    )
    coordinated_effective.pop("labels", None)
    standard_effective.pop("labels", None)
    if coordinated_effective != standard_effective:
        raise RuntimeError("SAC training profiles are not exactly matched")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "new_method": STD_SAC_LABEL,
        "matched_method": COORD_SAC_LABEL,
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
            "coordinated_cache": "critical_path_joint",
            "standard_cache": "popularity_ema",
            "all_sac_state_reward_action_and_training_settings_matched": True,
        },
        "parent_locks": {
            "p5": sha256_file(P5_FINAL_LOCK),
            "p6": sha256_file(P6_FINAL_LOCK),
        },
        "governance": {
            "our_is_not_retrained": True,
            "new_baseline_trains_from_scratch": True,
            "seeds_51_to_60_are_paired_confirmation_not_holdout": True,
            "no_post_result_hyperparameter_tuning": True,
        },
    }


__all__ = [
    "ANALYSIS_DIR",
    "CAPACITY_MULTISET",
    "CAPACITY_NAMESPACE",
    "COORD_SAC_LABEL",
    "DATASET_PATH",
    "EVALUATION_EPISODES",
    "EXPECTED_DATASET_SHA256",
    "FAMILIES",
    "FINAL_DIR",
    "FINAL_SEEDS",
    "P3_FINAL_DIR",
    "P5_SAC_DIR",
    "P6_SUMMARY",
    "PROFILE_CONVERGED",
    "PROFILE_SMOKE",
    "PROTOCOL_VERSION",
    "RESULT_ROOT",
    "SMOKE_DIR",
    "SMOKE_SEEDS",
    "STD_SAC_LABEL",
    "TASK_LIMIT_INCLUDING_DUMMY",
    "register_suite_extension",
    "sha256_file",
    "standard_cache_sac_config",
    "validate_protocol",
]
