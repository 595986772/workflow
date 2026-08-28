"""Frozen protocol for the Alibaba-CP100 cache-budget experiment."""

import hashlib
import json
from pathlib import Path


ALIBABA_CP100_PROTOCOL_VERSION = "alibaba_cp100_budget20_v1"
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    ROOT_DIR
    / "datasets"
    / "alibaba_cp100"
    / "dag_alibaba_cp100.json"
)
SELECTION_SUMMARY_PATH = DATASET_PATH.with_name(
    "selection_summary.json"
)
EXPECTED_DATASET_SHA256 = (
    "2903ff2478f5c55fe445bd2a7b6fbe595aecf6ea6383913b85ea5efd92ee2d89"
)
EXPECTED_GRAPH_COUNT = 100
CAPACITY_MULTISET = [0, 1, 1, 2, 2, 2, 2, 3, 3, 4]
TOTAL_CACHE_BUDGET = 20
BASELINE_RANDOM_DRAW_CAPACITY = 4
CAPACITY_ASSIGNMENT_NAMESPACE = ALIBABA_CP100_PROTOCOL_VERSION
SERVICES = 10
USERS = 20
SERVERS = 10
TASK_LIMIT = 10
BANDWIDTH_HZ = 15000
SMOKE_SEEDS = [21]
COMPARISON_SEEDS = [21, 22, 23]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol():
    """Fail early if the frozen dataset or environment changed."""
    if len(CAPACITY_MULTISET) != SERVERS:
        raise ValueError("Capacity multiset must match the server count")
    if any(
        not 0 <= capacity <= SERVICES
        for capacity in CAPACITY_MULTISET
    ):
        raise ValueError("Capacity multiset contains an invalid value")
    if sum(CAPACITY_MULTISET) != TOTAL_CACHE_BUDGET:
        raise ValueError("Capacity multiset must preserve budget 20")
    if max(CAPACITY_MULTISET) > BASELINE_RANDOM_DRAW_CAPACITY:
        raise ValueError(
            "Shared initialization draw must cover the largest capacity"
        )
    actual_sha256 = file_sha256(DATASET_PATH)
    if actual_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "Alibaba-CP100 checksum changed: "
            f"expected {EXPECTED_DATASET_SHA256}, got {actual_sha256}"
        )
    graphs = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if len(graphs) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} DAGs, got {len(graphs)}"
        )
    selection_summary = json.loads(
        SELECTION_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    if selection_summary.get("formal_unbiased_holdout") is not False:
        raise ValueError(
            "Alibaba-CP100 must remain marked as a biased stress set"
        )
    return {
        "protocol_version": ALIBABA_CP100_PROTOCOL_VERSION,
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": actual_sha256,
            "graph_count": len(graphs),
            "role": selection_summary.get("dataset_role"),
            "formal_unbiased_holdout": False,
        },
        "environment": {
            "users": USERS,
            "servers": SERVERS,
            "services": SERVICES,
            "task_limit": TASK_LIMIT,
            "bandwidth_hz": BANDWIDTH_HZ,
            "capacity_multiset": list(CAPACITY_MULTISET),
            "total_cache_budget": TOTAL_CACHE_BUDGET,
            "baseline_random_draw_capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "capacity_assignment_namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
        },
        "experiment": {
            "smoke_seeds": list(SMOKE_SEEDS),
            "comparison_seeds": list(COMPARISON_SEEDS),
            "evaluation_scenarios_per_seed": 100,
            "claim_scope": "mechanism_stress_test_only",
            "algorithm_tuning_on_dataset": False,
        },
    }


validate_protocol()
