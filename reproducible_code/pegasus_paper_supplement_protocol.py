"""Frozen protocol for the post-lock Pegasus paper supplements."""

import hashlib
import json
from pathlib import Path

from pegasus_paper_closure_protocol import (
    CAPACITY_NAMESPACE,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    FINAL_SEEDS,
    TASK_LIMIT_INCLUDING_DUMMY,
)
from run_a0_fixed_budget_heterogeneity import (
    ALGORITHM_SOURCE_FILES,
    source_hash,
)
from run_pegasus_paper_closure import protocol_source_hash


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_paper_supplement_v1"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p4_paper_supplement"
P3_ROOT = ROOT / "results/pegasus_pscale/p3_paper_closure"
P3_FINAL_DIR = P3_ROOT / "final"
P3_FREEZE_PATH = P3_ROOT / "FROZEN_ALGORITHM.json"
P3_FINAL_LOCK_PATH = P3_ROOT / "FINAL_LOCK.json"

HETEROGENEITY_SEEDS = (71, 72, 73)
HETEROGENEITY_METHODS = (
    "daoc_paper",
    "centralized_greedy_daoc",
    "lean_our",
)
CAPACITY_PROFILES = {
    "uniform_b10": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    "heterogeneous_b10": (0, 0, 0, 1, 1, 1, 1, 2, 2, 2),
}
HETEROGENEITY_CAPACITY_NAMESPACE = "pegasus_p4_b10_control"

ABLATION_SEEDS = FINAL_SEEDS
ABLATION_METHODS = ("our_dqn", "our_no_coord_cache")
ABLATION_REFERENCE_METHOD = "lean_our"

SCALING_SEEDS = FINAL_SEEDS
SCALING_METHODS = HETEROGENEITY_METHODS
SCALING_USER_COUNTS = (10, 20, 40, 60)
SCALING_EPISODES = 50
CACHE_BENCHMARK_REPEATS = 200

SUPPLEMENT_SOURCE_FILES = (
    "analyze_pegasus_paper_supplement.py",
    "evaluate_pegasus_user_scaling.py",
    "pegasus_paper_supplement_protocol.py",
    "run_pegasus_paper_supplement.py",
    "test_pegasus_paper_supplement.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def supplement_source_hash():
    return source_hash(SUPPLEMENT_SOURCE_FILES)


def validate_parent_freeze():
    freeze = read_json(P3_FREEZE_PATH)
    final_lock = read_json(P3_FINAL_LOCK_PATH)
    current_algorithm_hash = source_hash(ALGORITHM_SOURCE_FILES)
    current_protocol_hash = protocol_source_hash()
    if freeze.get("status") != "frozen":
        raise RuntimeError("Pegasus P3 algorithm is not frozen")
    if final_lock.get("status") != "complete":
        raise RuntimeError("Pegasus P3 final confirmation is incomplete")
    if not final_lock.get("formal_gate", {}).get("passed"):
        raise RuntimeError("Pegasus P3 final gate did not pass")
    if current_algorithm_hash != freeze.get("algorithm_source_sha256"):
        raise RuntimeError("Frozen algorithm source hash changed")
    if current_protocol_hash != freeze.get("protocol_source_sha256"):
        raise RuntimeError("Frozen P3 protocol source hash changed")
    return {
        "algorithm_source_sha256": current_algorithm_hash,
        "p3_protocol_source_sha256": current_protocol_hash,
        "p3_final_summary_sha256": final_lock.get("summary_sha256"),
    }


def specification():
    parent = validate_parent_freeze()
    value = {
        "protocol_version": PROTOCOL_VERSION,
        "parent": parent,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "families": list(FAMILIES),
        "evaluation_episodes": EVALUATION_EPISODES,
        "heterogeneity": {
            "seeds": list(HETEROGENEITY_SEEDS),
            "methods": list(HETEROGENEITY_METHODS),
            "capacity_profiles": {
                key: list(value)
                for key, value in CAPACITY_PROFILES.items()
            },
            "capacity_namespace": HETEROGENEITY_CAPACITY_NAMESPACE,
            "same_total_budget": 10,
        },
        "ablation": {
            "seeds": list(ABLATION_SEEDS),
            "methods": list(ABLATION_METHODS),
            "reference_method": ABLATION_REFERENCE_METHOD,
            "capacity_namespace": CAPACITY_NAMESPACE,
            "formal_gate": {
                "ci95_lower": "greater_than_zero",
                "wilcoxon_one_sided": "p_less_than_0.05",
                "wins": "at_least_7_of_10",
            },
        },
        "scaling": {
            "seeds": list(SCALING_SEEDS),
            "methods": list(SCALING_METHODS),
            "user_counts": list(SCALING_USER_COUNTS),
            "episodes_per_seed": SCALING_EPISODES,
            "model_and_cache_frozen": True,
            "cache_benchmark_repeats": CACHE_BENCHMARK_REPEATS,
        },
        "governance": {
            "algorithm_changes_forbidden": True,
            "p3_results_immutable": True,
            "supplement_is_post_lock": True,
        },
    }
    value["specification_sha256"] = canonical_hash(value)
    return value


def validate_protocol():
    spec = specification()
    if any(sum(values) != 10 for values in CAPACITY_PROFILES.values()):
        raise RuntimeError("B10 control profiles must have equal budget 10")
    if len({len(values) for values in CAPACITY_PROFILES.values()}) != 1:
        raise RuntimeError("Capacity profiles must have equal server count")
    if len(CAPACITY_PROFILES["uniform_b10"]) != 10:
        raise RuntimeError("Capacity profiles must contain ten servers")
    if len(HETEROGENEITY_SEEDS) != 3:
        raise RuntimeError("Heterogeneity control requires three seeds")
    if tuple(ABLATION_SEEDS) != tuple(FINAL_SEEDS):
        raise RuntimeError("Ablations must pair with frozen final seeds")
    if 20 not in SCALING_USER_COUNTS:
        raise RuntimeError("Scaling must include the source 20-user point")
    return spec
