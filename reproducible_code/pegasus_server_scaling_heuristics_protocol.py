"""Frozen protocol for heuristic baselines in the server-count sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pegasus_server_scaling_protocol import (
    BANDWIDTH_HZ,
    CAPACITY_NAMESPACE,
    CAPACITY_PROFILES,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    SEEDS,
    SERVER_COUNTS,
    SERVICES,
    SMOKE_EPISODES,
    TASK_LIMIT_INCLUDING_DUMMY,
    TRAINED_SERVER_COUNTS,
    USERS,
    sha256_file,
)
from run_reproduction_suite import ALGORITHMS, PROFILES, effective_method_profile


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_server_scaling_p13_heuristics_v1"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p13_server_scaling_heuristics"
P11_ROOT = ROOT / "results/pegasus_pscale/p11_server_scaling"
P12_ROOT = ROOT / "results/pegasus_pscale/p12_server_scaling_strong_baselines"
P12_AUDIT = P12_ROOT / "CONVERGED_AUDIT.json"
P6_HEURISTIC_ROOT = (
    ROOT
    / "results/pegasus_pscale/p6_baselines_ablation/heuristics/runs"
)

RANDOM = "random"
NEAREST = "nearest"
NEAREST_SERVICE = "greedy"
METHODS = (RANDOM, NEAREST, NEAREST_SERVICE)
DISPLAY_NAMES = {
    RANDOM: "Random",
    NEAREST: "Nearest",
    NEAREST_SERVICE: "Nearest+Service",
}
FULL_PROFILE = "pegasus_p6_heuristics"
SMOKE_PROFILE = "pegasus_p6_smoke"
CALIBRATION_EPISODES = 5_000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def algorithm_config(label: str) -> dict:
    return next(item for item in ALGORITHMS if item["label"] == label)


def validate_protocol() -> dict:
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus dataset hash mismatch")

    p12 = read_json(P12_AUDIT)
    integrity_keys = (
        "all_runs_complete",
        "all_learning_runs_converged",
        "all_tasks_exactly_once",
        "all_capacity_constraints_valid",
        "all_methods_scenario_paired_with_p11",
        "all_method_identities_valid",
    )
    if not all(p12.get(key) for key in integrity_keys):
        raise RuntimeError("P12 strong-baseline sweep did not pass audit")

    expected_algorithms = {
        RANDOM: "random",
        NEAREST: "nearest_server",
        NEAREST_SERVICE: "nearest_with_service",
    }
    for method in METHODS:
        config = algorithm_config(method)
        if config["algorithm"] != expected_algorithms[method]:
            raise RuntimeError(f"Heuristic identity mismatch: {method}")
        if config.get("cache_policy", "popularity_ema") != "popularity_ema":
            raise RuntimeError(f"{method} must use the standard cache")
        if config.get("cache_coverage_constraint", False):
            raise RuntimeError(f"{method} must not use OUR cache coordination")

        profile = effective_method_profile(PROFILES[FULL_PROFILE], method)
        expected = {
            "train_episodes": CALIBRATION_EPISODES,
            "eval_episodes": EVALUATION_EPISODES,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "cache_freeze_episode": CALIBRATION_EPISODES,
            "convergence_mode": False,
        }
        for key, value in expected.items():
            if profile.get(key) != value:
                raise RuntimeError(f"{method} profile mismatch: {key}")

    for method in METHODS:
        for seed in SEEDS:
            directory = P6_HEURISTIC_ROOT / method / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            arguments = read_json(directory / "config.json")["arguments"]
            if not (
                summary.get("status") == "complete"
                and summary.get("eligible_for_comparison")
                and summary.get("evaluation_scenario_count")
                == EVALUATION_EPISODES
                and summary.get("selected_checkpoint_episode")
                == CALIBRATION_EPISODES
                and arguments.get("num_servers") == 10
                and arguments.get("num_users") == USERS
                and arguments.get("num_services") == SERVICES
                and float(arguments.get("bandwidth")) == BANDWIDTH_HZ
                and sorted(arguments.get("server_capacity_multiset", []))
                == sorted(CAPACITY_PROFILES[10])
                and arguments.get("capacity_assignment_namespace")
                == "pegasus_pscale_p2"
            ):
                raise RuntimeError(
                    f"Invalid S=10 heuristic reference: {method}, seed {seed}"
                )

    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocols": [
            "pegasus_server_scaling_p11_v1",
            "pegasus_server_scaling_p12_strong_baselines_v1",
            "pegasus_p6_baselines_ablation_v1",
        ],
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
            str(servers): list(values)
            for servers, values in CAPACITY_PROFILES.items()
        },
        "capacity_assignment_namespace": CAPACITY_NAMESPACE,
        "methods": list(METHODS),
        "display_names": dict(DISPLAY_NAMES),
        "seeds": list(SEEDS),
        "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "smoke_episodes": SMOKE_EPISODES,
        "calibration_episodes": CALIBRATION_EPISODES,
        "workers": 6,
        "full_profile": FULL_PROFILE,
        "claim_scope": "three_seed_server_count_sensitivity",
        "heuristic_identity": (
            "standard independent popularity-EMA cache; no OUR coordination"
        ),
    }
    specification["specification_sha256"] = canonical_hash(specification)
    return specification


__all__ = [
    "CALIBRATION_EPISODES",
    "DISPLAY_NAMES",
    "FULL_PROFILE",
    "METHODS",
    "NEAREST",
    "NEAREST_SERVICE",
    "P11_ROOT",
    "P12_ROOT",
    "P6_HEURISTIC_ROOT",
    "PROTOCOL_VERSION",
    "RANDOM",
    "RESULT_ROOT",
    "SMOKE_PROFILE",
    "algorithm_config",
    "canonical_hash",
    "validate_protocol",
]
