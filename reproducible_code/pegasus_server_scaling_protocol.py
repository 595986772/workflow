"""Frozen protocol for the Pegasus server-count sensitivity experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pegasus_paper_closure_protocol import (
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_server_scaling_p11_v1"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p11_server_scaling"
REFERENCE_ROOT = ROOT / "results/pegasus_pscale/p3_paper_closure/final"
SERVER_COUNTS = (5, 10, 15, 20)
TRAINED_SERVER_COUNTS = (5, 15, 20)
SEEDS = (51, 52, 53)
METHODS = ("daoc_paper", "lean_our")
USERS = 20
SERVICES = 10
BANDWIDTH_HZ = 15_000.0
EVALUATION_EPISODES = 100
SMOKE_EPISODES = 20
CAPACITY_NAMESPACE = "pegasus_server_scaling_p11"

CAPACITY_PROFILES = {
    5: (0, 0, 1, 1, 2),
    10: (0, 0, 0, 0, 1, 1, 1, 1, 2, 2),
    15: (
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ),
    20: (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol() -> dict:
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus dataset hash mismatch")
    if set(CAPACITY_PROFILES) != set(SERVER_COUNTS):
        raise RuntimeError("Every server count needs one capacity profile")
    for servers, capacities in CAPACITY_PROFILES.items():
        if len(capacities) != servers:
            raise RuntimeError(f"Capacity length mismatch for S={servers}")
        if capacities.count(0) * 5 != servers * 2:
            raise RuntimeError(f"K=0 fraction is not 40% for S={servers}")
        if capacities.count(1) * 5 != servers * 2:
            raise RuntimeError(f"K=1 fraction is not 40% for S={servers}")
        if capacities.count(2) * 5 != servers:
            raise RuntimeError(f"K=2 fraction is not 20% for S={servers}")
        if sum(capacities) * 5 != servers * 4:
            raise RuntimeError(f"Mean capacity is not 0.8 for S={servers}")
    if set(TRAINED_SERVER_COUNTS) | {10} != set(SERVER_COUNTS):
        raise RuntimeError("Server-count reuse partition is incomplete")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "workflow_families": list(FAMILIES),
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "users": USERS,
        "services": SERVICES,
        "bandwidth_hz": BANDWIDTH_HZ,
        "server_counts": list(SERVER_COUNTS),
        "trained_server_counts": list(TRAINED_SERVER_COUNTS),
        "reused_server_count": 10,
        "capacity_profiles": {
            str(servers): list(capacities)
            for servers, capacities in CAPACITY_PROFILES.items()
        },
        "capacity_budget_rule": "0.8 slots per server",
        "capacity_fraction_rule": "40% K0, 40% K1, 20% K2",
        "capacity_assignment_namespace": CAPACITY_NAMESPACE,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "claim_scope": "three_seed_server_count_sensitivity",
        "training_rule": (
            "from scratch to convergence for S=5,15,20; exact P3 reuse for S=10"
        ),
    }


validate_protocol()
