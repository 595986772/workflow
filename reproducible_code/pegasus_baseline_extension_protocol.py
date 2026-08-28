"""Frozen protocol for post-lock Pegasus-B8 baseline extensions."""

import hashlib
import json
from pathlib import Path

from pegasus_pscale_protocol import (
    CAPACITY_PROFILES,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
    validate_protocol as validate_pscale_protocol,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_baseline_extension_v1"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p5_baseline_extension"
P3_ROOT = ROOT / "results/pegasus_pscale/p3_paper_closure"
P3_FINAL_DIR = P3_ROOT / "final"
P3_FINAL_LOCK = P3_ROOT / "FINAL_LOCK.json"
P4_LOCK = (
    ROOT
    / "results/pegasus_pscale/p4_paper_supplement/SUPPLEMENT_LOCK.json"
)

CAPACITY_MULTISET = CAPACITY_PROFILES["B8"]
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
DEVELOPMENT_SEEDS = (41, 42, 43)
FINAL_SEEDS = tuple(range(51, 61))
SMOKE_SEEDS = (41,)
EVALUATION_EPISODES = 100
HEURISTIC_CACHE_CALIBRATION_EPISODES = 5000

HEURISTIC_METHODS = (
    "coord_cache_random",
    "coord_cache_nearest",
    "coord_cache_nearest_service",
)
SAC_METHOD = "coord_cache_discrete_sac"
REFERENCE_METHODS = (
    "daoc_paper",
    "centralized_greedy_daoc",
    "lean_our",
)
SAC_CONFIG = {
    "algorithm": "causal_telemetryDiscreteSAC",
    "reward_mode": "causal_makespan_increment",
    "cache_policy": "critical_path_joint",
    "cache_coverage_constraint": True,
    "initial_alpha": 0.02,
    "target_entropy_ratio": 0.98,
    "target_tau": 0.005,
    "gamma": 1.0,
    "n_step": 3,
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_protocol():
    pscale = validate_pscale_protocol()
    p3_lock = read_json(P3_FINAL_LOCK)
    p4_lock = read_json(P4_LOCK)
    if p3_lock.get("status") != "complete":
        raise RuntimeError("Pegasus P3 final lock is not complete")
    if not p3_lock.get("formal_gate", {}).get("passed"):
        raise RuntimeError("Pegasus P3 formal gate did not pass")
    if p4_lock.get("status") != "complete":
        raise RuntimeError("Pegasus P4 supplement is not complete")
    if set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS):
        raise RuntimeError("Development and final seeds must be disjoint")
    if len(CAPACITY_MULTISET) != 10 or sum(CAPACITY_MULTISET) != 8:
        raise RuntimeError("Baseline extension must use Pegasus B8")
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus dataset hash mismatch")
    locked_spec = p3_lock["specification"]
    if locked_spec["seeds"] != list(FINAL_SEEDS):
        raise RuntimeError("P3 final seeds do not match")
    if locked_spec["capacity_multiset"] != list(CAPACITY_MULTISET):
        raise RuntimeError("P3 capacity profile does not match")
    if locked_spec["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("P3 dataset does not match")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocol_version": pscale["protocol_version"],
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "families": list(FAMILIES),
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "capacity_namespace": CAPACITY_NAMESPACE,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "evaluation_episodes": EVALUATION_EPISODES,
        "heuristic_cache_calibration_episodes": (
            HEURISTIC_CACHE_CALIBRATION_EPISODES
        ),
        "heuristic_methods": list(HEURISTIC_METHODS),
        "sac_method": SAC_METHOD,
        "reference_methods": list(REFERENCE_METHODS),
        "sac_config": dict(SAC_CONFIG),
        "governance": {
            "our_checkpoint_and_results_immutable": True,
            "development_only_on_seeds_41_to_43": True,
            "final_baseline_run_once_on_seeds_51_to_60": True,
            "post_lock_extension_not_new_holdout": True,
        },
        "p3_final_summary_sha256": p3_lock.get("summary_sha256"),
        "p4_specification_sha256": p4_lock.get(
            "specification_sha256"
        ),
    }
