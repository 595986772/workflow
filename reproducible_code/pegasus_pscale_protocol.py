"""Frozen protocol constants and validation for Pegasus P-Scale."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import networkx as nx

from user import DAG_COMPLETION_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_pscale_p2"
DATASET_PATH = ROOT / "datasets/pegasus_pscale/dag_pegasus5_full31.json"
MANIFEST_PATH = ROOT / "datasets/pegasus_pscale/manifest.json"
INVALIDATION_PATH = (
    ROOT
    / "results/h8v1_cross_topology/xv1/INVALIDATED_DIAGNOSTIC.json"
)
P1_INVALIDATION_PATH = (
    ROOT
    / "results/pegasus_pscale/p1/INVALIDATED_DIAGNOSTIC.json"
)
EXPECTED_DATASET_SHA256 = (
    "0671d8ea1ecdd8165062e19733e4859edb8e5ce87ecdd054bcef290abc49d5a5"
)
FAMILIES = (
    "Montage",
    "CyberShake",
    "Epigenomics",
    "Inspiral",
    "Sipht",
)
EXPECTED_REAL_TASKS = {
    "Montage": 25,
    "CyberShake": 30,
    "Epigenomics": 24,
    "Inspiral": 30,
    "Sipht": 29,
}
CAPACITY_PROFILES = {
    "B5": (0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
    "B8": (0, 0, 0, 0, 1, 1, 1, 1, 2, 2),
    "B10": (0, 0, 0, 1, 1, 1, 1, 2, 2, 2),
}
MAIN_PROFILE = "B8"
DEVELOPMENT_SEEDS = (41, 42, 43)
EVALUATION_EPISODES = 100
VALIDATION_SCENARIOS = 50
CACHE_CALIBRATION_EPISODES = 5000
EVALUATION_BANK_SCOPE = "infrastructure"
TASK_LIMIT_INCLUDING_DUMMY = 31


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_family(episode_index):
    if not 0 <= episode_index < EVALUATION_EPISODES:
        raise ValueError("episode_index is outside the P-Scale bank")
    return FAMILIES[episode_index % len(FAMILIES)]


def expected_family_counts(episodes=EVALUATION_EPISODES):
    if episodes < 1 or episodes % len(FAMILIES):
        raise ValueError(
            "P-Scale scenarios must be positive and divisible by five"
        )
    return dict(
        Counter(FAMILIES[index % len(FAMILIES)] for index in range(episodes))
    )


def validate_protocol():
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("P-Scale dataset SHA-256 mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["dataset"]["sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("P-Scale manifest hash mismatch")
    if manifest["generation"]["type_mapping"]["program_type_count"] != 39:
        raise RuntimeError("P-Scale must map exactly 39 program types")

    raw_dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    observed_tasks = {}
    observed_families = set()
    for graph_data in raw_dataset.values():
        graph = nx.node_link_graph(graph_data)
        family = str(graph.graph["source_family"])
        observed_families.add(family)
        observed_tasks[family] = graph.number_of_nodes() - 1
        if graph.number_of_nodes() > TASK_LIMIT_INCLUDING_DUMMY:
            raise RuntimeError("P-Scale graph exceeds the task limit")
    if observed_families != set(FAMILIES):
        raise RuntimeError("P-Scale family set mismatch")
    if observed_tasks != EXPECTED_REAL_TASKS:
        raise RuntimeError("P-Scale real-task counts mismatch")
    if expected_family_counts() != {
        family: 20 for family in FAMILIES
    }:
        raise RuntimeError("P-Scale evaluation bank is not family-balanced")
    for profile, capacities in CAPACITY_PROFILES.items():
        if len(capacities) != 10 or sum(capacities) != int(profile[1:]):
            raise RuntimeError(f"Invalid capacity profile {profile}")
    invalidation = json.loads(
        INVALIDATION_PATH.read_text(encoding="utf-8")
    )
    if invalidation.get("status") != "invalidated_diagnostic_only":
        raise RuntimeError("Legacy Pegasus diagnostic is not invalidated")
    p1_invalidation = json.loads(
        P1_INVALIDATION_PATH.read_text(encoding="utf-8")
    )
    if p1_invalidation.get("status") != "invalidated_diagnostic_only":
        raise RuntimeError("P-Scale p1 diagnostic is not invalidated")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dag_completion_protocol_version": (
            DAG_COMPLETION_PROTOCOL_VERSION
        ),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "families": list(FAMILIES),
        "real_tasks": observed_tasks,
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "capacity_profiles": {
            key: list(value)
            for key, value in CAPACITY_PROFILES.items()
        },
        "main_profile": MAIN_PROFILE,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "evaluation_family_counts": expected_family_counts(),
        "validation_scenarios": VALIDATION_SCENARIOS,
        "cache_calibration_episodes": CACHE_CALIBRATION_EPISODES,
        "evaluation_bank_scope": EVALUATION_BANK_SCOPE,
    }
