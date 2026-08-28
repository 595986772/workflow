"""Frozen constants for the Pegasus paper-closure experiments."""

import json
from pathlib import Path

from pegasus_pscale_protocol import (
    CACHE_CALIBRATION_EPISODES,
    CAPACITY_PROFILES,
    DATASET_PATH,
    EVALUATION_BANK_SCOPE,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
    VALIDATION_SCENARIOS,
    validate_protocol as validate_pscale_protocol,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_paper_closure_v1"
P2_ROOT = ROOT / "results/pegasus_pscale/p2"
P2_LOCK_PATH = P2_ROOT / "PSCALE_LOCK.json"
P2_DEVELOPMENT_SUITE = P2_ROOT / "converged"

DEVELOPMENT_SEEDS = (41, 42, 43)
FINAL_SEEDS = tuple(range(51, 61))
SMOKE_SEEDS = (41,)
CAPACITY_MULTISET = CAPACITY_PROFILES["B8"]
CAPACITY_NAMESPACE = "pegasus_pscale_p2"

P2_REFERENCE_METHODS = (
    "guided_full",
    "centralized_greedy_daoc",
    "lean_our",
)
DEVELOPMENT_METHODS = (
    "daoc_paper",
    "our_dqn",
    "our_no_telemetry",
    "our_no_coord_cache",
)
ABLATION_METHODS = (
    "our_dqn",
    "our_no_telemetry",
    "our_no_coord_cache",
)
FINAL_METHODS = (
    "daoc_paper",
    "centralized_greedy_daoc",
    "lean_our",
)


def validate_protocol():
    pscale = validate_pscale_protocol()
    if not P2_LOCK_PATH.exists():
        raise RuntimeError("Pegasus P2 lock is missing")
    p2_lock = json.loads(P2_LOCK_PATH.read_text(encoding="utf-8"))
    if p2_lock.get("status") != "main_complete":
        raise RuntimeError("Pegasus P2 main experiment is not complete")
    if not p2_lock.get("three_seed_gate", {}).get("passed"):
        raise RuntimeError("Pegasus P2 three-seed gate did not pass")
    if len(CAPACITY_MULTISET) != 10 or sum(CAPACITY_MULTISET) != 8:
        raise RuntimeError("Paper closure must use the frozen B8 profile")
    if set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS):
        raise RuntimeError("Development and final seeds must be disjoint")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocol_version": pscale["protocol_version"],
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "capacity_multiset": list(CAPACITY_MULTISET),
        "capacity_namespace": CAPACITY_NAMESPACE,
        "families": list(FAMILIES),
        "evaluation_episodes": EVALUATION_EPISODES,
        "validation_scenarios": VALIDATION_SCENARIOS,
        "cache_calibration_episodes": CACHE_CALIBRATION_EPISODES,
        "evaluation_bank_scope": EVALUATION_BANK_SCOPE,
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "p2_lock": str(P2_LOCK_PATH.resolve()),
    }
