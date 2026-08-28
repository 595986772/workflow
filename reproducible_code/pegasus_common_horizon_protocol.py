"""Frozen protocol for the Pegasus-B8 26k common-horizon rerun."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import run_reproduction_suite as reproduction
from pegasus_p6_protocol import FINAL_SEEDS
from pegasus_pscale_protocol import (
    CAPACITY_PROFILES,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
)
from pegasus_sac_std_extension_protocol import (
    PROFILE_CONVERGED as STD_SAC_SOURCE_PROFILE,
    STD_SAC_LABEL,
    register_suite_extension,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_common_horizon_26k_v1"
PROFILE_NAME = "pegasus_common_horizon_26k"

RESULT_ROOT = ROOT / "results/pegasus_pscale/p9_common_horizon_26k"
FINAL_DIR = RESULT_ROOT / "final"
ANALYSIS_DIR = RESULT_ROOT / "analysis"

FIXED_TRAIN_EPISODES = 26_000
TAIL_EPISODES = 1_000
TAIL_START_EPISODE = FIXED_TRAIN_EPISODES - TAIL_EPISODES + 1
EVALUATION_EPISODES = 100
DEFAULT_WORKERS = 4

OUR_LABEL = "lean_our"
COORD_SAC_LABEL = reproduction.COORD_DISCRETE_SAC_LABEL
RERUN_METHODS = (OUR_LABEL, COORD_SAC_LABEL, STD_SAC_LABEL)
REFERENCE_METHODS = (
    "daoc_paper",
    reproduction.DQN_WDSA_STD_LABEL,
    "daoc_our_coord_cache",
)
ALL_MAIN_METHODS = REFERENCE_METHODS + RERUN_METHODS

CAPACITY_MULTISET = CAPACITY_PROFILES["B8"]
CAPACITY_NAMESPACE = "pegasus_pscale_p2"

P3_FINAL_DIR = ROOT / "results/pegasus_pscale/p3_paper_closure/final"
P5_SAC_DIR = ROOT / "results/pegasus_pscale/p5_baseline_extension/sac_final"
P6_LEARNING_DIR = ROOT / "results/pegasus_pscale/p6_baselines_ablation/learning"
P7_FINAL_DIR = ROOT / "results/pegasus_pscale/p7_std_cache_discrete_sac/final"
P8_FINAL_DIR = ROOT / "results/pegasus_pscale/p8_daoc_our_coord_cache/final"

PARENT_LOCKS = (
    ROOT / "results/pegasus_pscale/p3_paper_closure/FINAL_LOCK.json",
    ROOT / "results/pegasus_pscale/p5_baseline_extension/FINAL_LOCK.json",
    ROOT / "results/pegasus_pscale/p6_baselines_ablation/FINAL_LOCK.json",
    ROOT / "results/pegasus_pscale/p7_std_cache_discrete_sac/FINAL_LOCK.json",
    ROOT / "results/pegasus_pscale/p8_daoc_our_coord_cache/FINAL_LOCK.json",
)

SOURCE_PROFILES = {
    OUR_LABEL: "pegasus_pscale_p2_converged",
    COORD_SAC_LABEL: "pegasus_baseline_sac_converged",
    STD_SAC_LABEL: STD_SAC_SOURCE_PROFILE,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algorithm_config(label):
    return next(
        config for config in reproduction.ALGORITHMS
        if config["label"] == label
    )


def _fixed_method_profile(source_profile, label):
    effective = deepcopy(
        reproduction.effective_method_profile(
            reproduction.PROFILES[source_profile],
            label,
        )
    )
    effective.update(
        {
            "train_episodes": FIXED_TRAIN_EPISODES,
            "eval_episodes": EVALUATION_EPISODES,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "convergence_mode": False,
            "cache_freeze_episode": 5_000,
            "eval_scenario_bank": True,
            "eval_bank_scope": "infrastructure",
            "eval_dag_families": list(FAMILIES),
        }
    )
    effective.pop("method_overrides", None)
    return effective


def register_profile():
    """Register both the standard-cache SAC method and fixed profile."""
    register_suite_extension()
    method_overrides = {
        label: _fixed_method_profile(source, label)
        for label, source in SOURCE_PROFILES.items()
    }
    profile = deepcopy(
        reproduction.PROFILES["pegasus_pscale_p2_converged"]
    )
    profile.update(
        {
            "labels": list(RERUN_METHODS),
            "seeds": list(FINAL_SEEDS),
            "train_episodes": FIXED_TRAIN_EPISODES,
            "eval_episodes": EVALUATION_EPISODES,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "convergence_mode": False,
            "dag_dataset_path": str(DATASET_PATH.resolve()),
            "dag_dataset_sha256": EXPECTED_DATASET_SHA256,
            "num_tasks": TASK_LIMIT_INCLUDING_DUMMY,
            "eval_dag_families": list(FAMILIES),
            "server_capacity": 1,
            "server_capacity_multiset": list(CAPACITY_MULTISET),
            "baseline_server_capacity": 3,
            "capacity_assignment_namespace": CAPACITY_NAMESPACE,
            "cache_freeze_episode": 5_000,
            "eval_scenario_bank": True,
            "eval_bank_scope": "infrastructure",
            "method_overrides": method_overrides,
        }
    )
    reproduction.PROFILES[PROFILE_NAME] = profile
    return profile


def validate_protocol():
    profile = register_profile()
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus-B8 dataset hash mismatch")
    if len(CAPACITY_MULTISET) != 10 or sum(CAPACITY_MULTISET) != 8:
        raise RuntimeError("Common-horizon rerun requires Pegasus-B8")
    if TAIL_START_EPISODE != 25_001:
        raise RuntimeError("The common tail must be episodes 25001-26000")

    parent_hashes = {}
    for path in PARENT_LOCKS:
        lock = read_json(path)
        if lock.get("status") != "complete":
            raise RuntimeError(f"Incomplete parent lock: {path}")
        parent_hashes[str(path.relative_to(ROOT))] = sha256_file(path)

    for label in RERUN_METHODS:
        effective = reproduction.effective_method_profile(profile, label)
        expected = {
            "train_episodes": FIXED_TRAIN_EPISODES,
            "eval_episodes": EVALUATION_EPISODES,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "convergence_mode": False,
            "cache_freeze_episode": 5_000,
            "eval_scenario_bank": True,
            "eval_bank_scope": "infrastructure",
            "num_tasks": TASK_LIMIT_INCLUDING_DUMMY,
            "bandwidth": 15_000,
        }
        for key, value in expected.items():
            if effective.get(key) != value:
                raise RuntimeError(
                    f"{label} has invalid fixed-horizon {key}: "
                    f"{effective.get(key)!r}"
                )

    ours = algorithm_config(OUR_LABEL)
    coord_sac = algorithm_config(COORD_SAC_LABEL)
    std_sac = algorithm_config(STD_SAC_LABEL)
    if ours["cache_policy"] != "critical_path_joint":
        raise RuntimeError("OUR coordinated cache configuration changed")
    if coord_sac["cache_policy"] != "critical_path_joint":
        raise RuntimeError("Coordinated SAC cache configuration changed")
    if std_sac["cache_policy"] != "popularity_ema":
        raise RuntimeError("Standard SAC cache configuration changed")
    if coord_sac["algorithm"] != std_sac["algorithm"]:
        raise RuntimeError("The two SAC methods must share the same actor")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "profile": PROFILE_NAME,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "capacity_namespace": CAPACITY_NAMESPACE,
        "families": list(FAMILIES),
        "seeds": list(FINAL_SEEDS),
        "rerun_methods": list(RERUN_METHODS),
        "reference_methods": list(REFERENCE_METHODS),
        "fixed_train_episodes": FIXED_TRAIN_EPISODES,
        "tail_start_episode": TAIL_START_EPISODE,
        "tail_end_episode": FIXED_TRAIN_EPISODES,
        "tail_episodes": TAIL_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "default_workers": DEFAULT_WORKERS,
        "checkpoint_rule": "fixed_budget_final",
        "parent_locks": parent_hashes,
        "governance": {
            "old_results_are_not_overwritten": True,
            "all_three_methods_train_from_scratch": True,
            "no_early_stopping": True,
            "no_historical_best_checkpoint_selection": True,
            "seeds_51_to_60_are_confirmation_not_holdout": True,
        },
    }


__all__ = [
    "ALL_MAIN_METHODS",
    "ANALYSIS_DIR",
    "CAPACITY_MULTISET",
    "CAPACITY_NAMESPACE",
    "COORD_SAC_LABEL",
    "DATASET_PATH",
    "DEFAULT_WORKERS",
    "EVALUATION_EPISODES",
    "EXPECTED_DATASET_SHA256",
    "FAMILIES",
    "FINAL_DIR",
    "FINAL_SEEDS",
    "FIXED_TRAIN_EPISODES",
    "OUR_LABEL",
    "P3_FINAL_DIR",
    "P5_SAC_DIR",
    "P6_LEARNING_DIR",
    "P7_FINAL_DIR",
    "P8_FINAL_DIR",
    "PROFILE_NAME",
    "PROTOCOL_VERSION",
    "REFERENCE_METHODS",
    "RERUN_METHODS",
    "RESULT_ROOT",
    "STD_SAC_LABEL",
    "TAIL_EPISODES",
    "TAIL_START_EPISODE",
    "TASK_LIMIT_INCLUDING_DUMMY",
    "register_profile",
    "sha256_file",
    "validate_protocol",
]
