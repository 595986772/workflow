"""Frozen protocol for strong baselines in the server-count sweep."""

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
from run_reproduction_suite import (
    ALGORITHMS,
    effective_method_profile,
    PROFILES,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_server_scaling_p12_strong_baselines_v1"
RESULT_ROOT = (
    ROOT / "results/pegasus_pscale/p12_server_scaling_strong_baselines"
)
P11_ROOT = ROOT / "results/pegasus_pscale/p11_server_scaling"
P11_AUDIT = P11_ROOT / "CONVERGED_AUDIT.json"

OUR_DQN = "our_flat_ddqn"
COORD_SAC = "coord_cache_discrete_sac"
METHODS = (OUR_DQN, COORD_SAC)
DISPLAY_NAMES = {
    OUR_DQN: "OUR-DQN",
    COORD_SAC: "CoordCache-DiscreteSAC",
}
TRAINING_PROFILES = {
    OUR_DQN: "pegasus_p6_learning_converged",
    COORD_SAC: "pegasus_baseline_sac_converged",
}
SMOKE_PROFILES = {
    OUR_DQN: "pegasus_p6_smoke",
    COORD_SAC: "pegasus_baseline_sac_smoke",
}
REFERENCE_ROOTS = {
    OUR_DQN: (
        ROOT
        / "results/pegasus_pscale/p6_baselines_ablation/learning/runs"
        / OUR_DQN
    ),
    COORD_SAC: (
        ROOT
        / "results/pegasus_pscale/p5_baseline_extension/sac_final/runs"
        / COORD_SAC
    ),
}


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
    p11 = read_json(P11_AUDIT)
    integrity_keys = (
        "all_runs_complete",
        "all_learning_runs_converged",
        "all_tasks_exactly_once",
        "all_capacity_constraints_valid",
        "all_methods_scenario_paired",
    )
    if not all(p11.get(key) for key in integrity_keys):
        raise RuntimeError("P11 DAOC/OUR server sweep did not pass audit")

    ours = algorithm_config("lean_our")
    dqn = algorithm_config(OUR_DQN)
    sac = algorithm_config(COORD_SAC)
    shared = (
        "reward_mode",
        "cache_policy",
        "cache_coverage_constraint",
        "gamma",
        "n_step",
        "historical_feedback_guidance",
        "adaptive_guidance_gate",
    )
    for candidate in (dqn, sac):
        for key in shared:
            if candidate.get(key) != ours.get(key):
                raise RuntimeError(
                    f"{candidate['label']} unexpectedly changes {key}"
                )
    if dqn["algorithm"] != "causal_telemetryDDQN":
        raise RuntimeError("OUR-DQN must use the flat Double DQN scheduler")
    if sac["algorithm"] != "causal_telemetryDiscreteSAC":
        raise RuntimeError("CoordCache-DiscreteSAC algorithm mismatch")

    for method, profile_name in TRAINING_PROFILES.items():
        effective = effective_method_profile(PROFILES[profile_name], method)
        expected = {
            "train_episodes": 40_000,
            "checkpoint_every": 500,
            "validation_scenarios": 50,
            "convergence_min_episodes": 5_000,
            "cache_freeze_episode": 5_000,
            "convergence_mode": True,
        }
        for key, value in expected.items():
            if effective.get(key) != value:
                raise RuntimeError(
                    f"{method} convergence setting mismatch: {key}"
                )

    for method, root in REFERENCE_ROOTS.items():
        for seed in SEEDS:
            directory = root / f"seed_{seed}"
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")["arguments"]
            if not (
                summary.get("status") == "complete"
                and summary.get("eligible_for_comparison")
                and summary.get("evaluation_scenario_count")
                == EVALUATION_EPISODES
                and config.get("num_servers") == 10
                and sorted(config.get("server_capacity_multiset", []))
                == sorted(CAPACITY_PROFILES[10])
            ):
                raise RuntimeError(
                    f"Invalid S=10 reference for {method}, seed {seed}"
                )

    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocol": "pegasus_server_scaling_p11_v1",
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
        "workers": 6,
        "training_profiles": dict(TRAINING_PROFILES),
        "claim_scope": "three_seed_server_count_sensitivity",
        "method_identity": {
            OUR_DQN: (
                "OUR state, reward, telemetry and coordinated cache; "
                "Pairwise PD3QN replaced by flat Double DQN"
            ),
            COORD_SAC: (
                "OUR state, reward, telemetry and coordinated cache; "
                "scheduler replaced by Discrete SAC"
            ),
        },
    }
    specification["specification_sha256"] = canonical_hash(specification)
    return specification


validate_protocol()

