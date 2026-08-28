"""Frozen protocol for standard-cache baselines and mechanism ablations."""

import hashlib
import json
from pathlib import Path

from pegasus_pscale_protocol import (
    CAPACITY_PROFILES,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
)
from run_reproduction_suite import (
    ALGORITHMS,
    BASE_DDQN_STD_LABEL,
    DQN_WDSA_STD_LABEL,
    OUR_FLAT_DDQN_LABEL,
    OUR_NO_DEPENDENCY_CACHE_LABEL,
    OUR_NO_TASK_DEPENDENCY_LABEL,
    OUR_TERMINAL_REWARD_LABEL,
    PEGASUS_P6_HEURISTIC_LABELS,
    PEGASUS_P6_LEARNING_LABELS,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_p6_baselines_ablation_v1"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p6_baselines_ablation"
SMOKE_DIR = RESULT_ROOT / "smoke"
HEURISTIC_DIR = RESULT_ROOT / "heuristics"
LEARNING_DIR = RESULT_ROOT / "learning"
ANALYSIS_DIR = RESULT_ROOT / "analysis"

P3_ROOT = ROOT / "results/pegasus_pscale/p3_paper_closure"
P3_FINAL_DIR = P3_ROOT / "final"
P3_FINAL_LOCK = P3_ROOT / "FINAL_LOCK.json"
P4_ROOT = ROOT / "results/pegasus_pscale/p4_paper_supplement"
P4_ABLATION_DIR = P4_ROOT / "ablation"
P4_LOCK = P4_ROOT / "SUPPLEMENT_LOCK.json"
P5_ROOT = ROOT / "results/pegasus_pscale/p5_baseline_extension"
P5_SAC_DIR = P5_ROOT / "sac_final"
P5_FINAL_LOCK = P5_ROOT / "FINAL_LOCK.json"

SMOKE_SEEDS = (41,)
FINAL_SEEDS = tuple(range(51, 61))
EVALUATION_EPISODES = 100
CAPACITY_MULTISET = CAPACITY_PROFILES["B8"]
CAPACITY_NAMESPACE = "pegasus_pscale_p2"

HEURISTIC_METHODS = tuple(PEGASUS_P6_HEURISTIC_LABELS)
LEARNING_METHODS = tuple(PEGASUS_P6_LEARNING_LABELS)
NEW_METHODS = HEURISTIC_METHODS + LEARNING_METHODS
REFERENCE_METHODS = (
    "daoc_paper",
    "centralized_greedy_daoc",
    "coord_cache_discrete_sac",
    "lean_our",
    "our_no_coord_cache",
)
MAIN_COMPARISON_METHODS = (
    "random",
    "nearest",
    "greedy",
    DQN_WDSA_STD_LABEL,
    "daoc_paper",
    "centralized_greedy_daoc",
    "coord_cache_discrete_sac",
    "lean_our",
)
FACTORIAL_METHODS = (
    BASE_DDQN_STD_LABEL,
    OUR_FLAT_DDQN_LABEL,
    "our_no_coord_cache",
    "lean_our",
)
MECHANISM_ABLATIONS = (
    OUR_NO_TASK_DEPENDENCY_LABEL,
    OUR_NO_DEPENDENCY_CACHE_LABEL,
    OUR_TERMINAL_REWARD_LABEL,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algorithm_config(label):
    return next(item for item in ALGORITHMS if item["label"] == label)


def validate_protocol():
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus dataset hash mismatch")
    if len(CAPACITY_MULTISET) != 10 or sum(CAPACITY_MULTISET) != 8:
        raise RuntimeError("P6 must use the frozen Pegasus-B8 capacity profile")
    if set(SMOKE_SEEDS) & set(FINAL_SEEDS):
        raise RuntimeError("Smoke and final seeds must be disjoint")

    p3_lock = read_json(P3_FINAL_LOCK)
    p4_lock = read_json(P4_LOCK)
    p5_lock = read_json(P5_FINAL_LOCK)
    if p3_lock.get("status") != "complete":
        raise RuntimeError("P3 final lock is incomplete")
    if p4_lock.get("status") != "complete":
        raise RuntimeError("P4 supplement lock is incomplete")
    if p5_lock.get("status") != "complete":
        raise RuntimeError("P5 baseline lock is incomplete")

    ours = algorithm_config("lean_our")
    flat = algorithm_config(OUR_FLAT_DDQN_LABEL)
    base = algorithm_config(BASE_DDQN_STD_LABEL)
    no_task = algorithm_config(OUR_NO_TASK_DEPENDENCY_LABEL)
    no_cache_dependency = algorithm_config(
        OUR_NO_DEPENDENCY_CACHE_LABEL
    )
    terminal = algorithm_config(OUR_TERMINAL_REWARD_LABEL)

    if flat.get("cache_coverage_constraint") is not True:
        raise RuntimeError("OUR-FlatDDQN must retain the OUR coverage constraint")
    if base["cache_policy"] != "popularity_ema":
        raise RuntimeError("Base-DDQN must use the independent standard cache")
    if no_task.get("task_dependency_features", True):
        raise RuntimeError("Task-dependency ablation is not isolated")
    if no_cache_dependency.get("cache_dependency_awareness", True):
        raise RuntimeError("Cache-dependency ablation is not isolated")
    if terminal["reward_mode"] != "terminal_binary":
        raise RuntimeError("Terminal-reward ablation is not isolated")

    shared_keys = (
        "beta",
        "beta_min",
        "beta_decay",
        "cache_policy",
        "cache_coverage_constraint",
        "gamma",
        "n_step",
        "num_quantiles",
        "risk_tail_fraction",
        "entropy_coefficient",
        "priority_alpha",
        "priority_beta_start",
        "priority_beta_anneal_steps",
        "criticality_boost",
    )
    for ablation, changed_key in (
        (no_task, "task_dependency_features"),
        (no_cache_dependency, "cache_dependency_awareness"),
        (terminal, "reward_mode"),
    ):
        for key in shared_keys:
            if key == changed_key:
                continue
            if ablation.get(key, False) != ours.get(key, False):
                raise RuntimeError(
                    f"{ablation['label']} unexpectedly changes {key}"
                )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "families": list(FAMILIES),
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "capacity_namespace": CAPACITY_NAMESPACE,
        "smoke_seeds": list(SMOKE_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "evaluation_episodes": EVALUATION_EPISODES,
        "new_methods": list(NEW_METHODS),
        "reference_methods": list(REFERENCE_METHODS),
        "main_comparison_methods": list(MAIN_COMPARISON_METHODS),
        "factorial_methods": list(FACTORIAL_METHODS),
        "mechanism_ablations": list(MECHANISM_ABLATIONS),
        "reuse_policy": {
            "p3_daoc_central_our": True,
            "p4_pairwise_standard_cache": True,
            "p5_coord_cache_discrete_sac": True,
            "requires_exact_scenario_bank_and_environment": True,
        },
        "governance": {
            "seeds_51_to_60_are_paired_confirmation_not_holdout": True,
            "new_learning_methods_train_from_scratch": True,
            "heuristics_receive_cache_calibration_only": True,
            "stable_final_checkpoint_required": True,
        },
        "parent_locks": {
            "p3": sha256_file(P3_FINAL_LOCK),
            "p4": sha256_file(P4_LOCK),
            "p5": sha256_file(P5_FINAL_LOCK),
        },
    }


__all__ = [
    "ANALYSIS_DIR",
    "CAPACITY_MULTISET",
    "CAPACITY_NAMESPACE",
    "DATASET_PATH",
    "EVALUATION_EPISODES",
    "EXPECTED_DATASET_SHA256",
    "FACTORIAL_METHODS",
    "FAMILIES",
    "FINAL_SEEDS",
    "HEURISTIC_DIR",
    "HEURISTIC_METHODS",
    "LEARNING_DIR",
    "LEARNING_METHODS",
    "MAIN_COMPARISON_METHODS",
    "MECHANISM_ABLATIONS",
    "NEW_METHODS",
    "P3_FINAL_DIR",
    "P4_ABLATION_DIR",
    "P5_SAC_DIR",
    "PROTOCOL_VERSION",
    "REFERENCE_METHODS",
    "RESULT_ROOT",
    "SMOKE_DIR",
    "SMOKE_SEEDS",
    "TASK_LIMIT_INCLUDING_DUMMY",
    "algorithm_config",
    "sha256_file",
    "validate_protocol",
]
