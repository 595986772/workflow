"""Frozen protocol for the Alibaba-CP100-A0 coordination study."""

import hashlib
import json
from pathlib import Path


A0_PROTOCOL_VERSION = "alibaba_cp100_a0_coordination_v1"
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    ROOT_DIR
    / "datasets"
    / "alibaba_cp100"
    / "dag_alibaba_cp100_a0.json"
)
MANIFEST_PATH = DATASET_PATH.with_name(
    "service_alignment_a0_manifest.json"
)
EXPECTED_DATASET_SHA256 = (
    "6a66ccf46eb9a033827467953f294463f4d666f5da046b7a94a4b690ceefd859"
)
EXPECTED_GRAPH_COUNT = 100

USERS = 20
SERVERS = 10
SERVICES = 10
TASK_LIMIT = 10
BANDWIDTH_HZ = 15_000
BASELINE_RANDOM_DRAW_CAPACITY = 2
CAPACITY_ASSIGNMENT_NAMESPACE = A0_PROTOCOL_VERSION

MAIN_BUDGET = "B8"
CAPACITY_PROFILES = {
    "B8": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
    "B10": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
    "B5": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
}
TOTAL_BUDGETS = {
    name: sum(capacities)
    for name, capacities in CAPACITY_PROFILES.items()
}

DEVELOPMENT_SEEDS = [1, 2, 3]
FINAL_SEEDS = list(range(11, 21))
METHOD_LABELS = [
    "guided_full",
    "centralized_greedy_daoc",
    "lean_our",
]

DYNAMIC_WINDOWS = 100
BASELINE_END_WINDOW = 40
BURST_END_WINDOW = 60
BASELINE_TARGET_UTILIZATION = 0.45
BASELINE_EXPECTED_ARRIVALS_PER_WINDOW = USERS
BURST_RATE_MULTIPLIER = 3.0
INGRESS_HOTSPOT_SHARE = 0.8
RECOVERY_ROLLING_WINDOWS = 5
RECOVERY_TOLERANCE = 1.05


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol():
    if file_sha256(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise ValueError("Alibaba-CP100-A0 checksum changed")
    graphs = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if len(graphs) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} A0 DAGs, got {len(graphs)}"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("formal_unbiased_holdout") is not False:
        raise ValueError("A0 must remain a controlled, biased mechanism set")
    output = manifest.get("output_dataset", {})
    recorded_hash = output.get("sha256") or output.get("checksum_sha256")
    if recorded_hash is not None and recorded_hash != EXPECTED_DATASET_SHA256:
        raise ValueError("A0 manifest checksum does not match the dataset")
    for name, capacities in CAPACITY_PROFILES.items():
        if len(capacities) != SERVERS:
            raise ValueError(f"{name} must define {SERVERS} capacities")
        if any(not 0 <= value <= SERVICES for value in capacities):
            raise ValueError(f"{name} contains an invalid capacity")
        if max(capacities) > BASELINE_RANDOM_DRAW_CAPACITY:
            raise ValueError(f"{name} exceeds the shared initialization draw")
        if sum(capacities) != int(name[1:]):
            raise ValueError(f"{name} does not preserve its stated budget")
    if not (
        0 < BASELINE_TARGET_UTILIZATION < 1
        and BURST_RATE_MULTIPLIER > 1
        and 0 < INGRESS_HOTSPOT_SHARE <= 1
    ):
        raise ValueError("Invalid dynamic stream parameters")
    return frozen_protocol_spec()


def frozen_protocol_spec():
    return {
        "protocol_version": A0_PROTOCOL_VERSION,
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": EXPECTED_DATASET_SHA256,
            "graph_count": EXPECTED_GRAPH_COUNT,
            "role": "controlled_service_alignment_variant",
            "formal_unbiased_holdout": False,
        },
        "environment": {
            "users": USERS,
            "servers": SERVERS,
            "services": SERVICES,
            "task_limit": TASK_LIMIT,
            "bandwidth_hz": BANDWIDTH_HZ,
            "capacity_profiles": CAPACITY_PROFILES,
            "main_budget": MAIN_BUDGET,
            "baseline_random_draw_capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "capacity_assignment_namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
        },
        "dynamic_stream": {
            "process": "piecewise_constant_nhpp",
            "windows": DYNAMIC_WINDOWS,
            "baseline_windows": [1, BASELINE_END_WINDOW],
            "burst_windows": [
                BASELINE_END_WINDOW + 1,
                BURST_END_WINDOW,
            ],
            "recovery_windows": [
                BURST_END_WINDOW + 1,
                DYNAMIC_WINDOWS,
            ],
            "baseline_target_utilization": (
                BASELINE_TARGET_UTILIZATION
            ),
            "expected_baseline_arrivals_per_window": (
                BASELINE_EXPECTED_ARRIVALS_PER_WINDOW
            ),
            "burst_rate_multiplier": BURST_RATE_MULTIPLIER,
            "burst_change": "ingress_hotspot",
            "ingress_hotspot_share": INGRESS_HOTSPOT_SHARE,
            "lambda_and_future_requests_visible_to_methods": False,
        },
        "governance": {
            "development_seeds": DEVELOPMENT_SEEDS,
            "single_use_final_seeds": FINAL_SEEDS,
            "methods": METHOD_LABELS,
            "evaluation_scenarios_per_static_seed": 100,
            "formal_seed_wins": 7,
            "paired_ci": 0.95,
            "wilcoxon_alternative": "our_better",
            "wilcoxon_alpha": 0.05,
        },
    }


validate_protocol()
