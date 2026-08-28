"""Frozen three-seed Pegasus-B8 cache-heterogeneity protocol."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import run_reproduction_suite as reproduction
from pegasus_daoc_coord_extension_protocol import (
    DAOC_COORD_LABEL,
    register_suite_extension,
)
from pegasus_pscale_protocol import (
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    TASK_LIMIT_INCLUDING_DUMMY,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_b8_heterogeneity_v1"
PROFILE_NAME = "pegasus_b8_heterogeneity_converged"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p15_b8_heterogeneity"
RUN_ROOT = RESULT_ROOT / "converged"
ANALYSIS_DIR = RESULT_ROOT / "analysis"

SEEDS = (51, 52, 53)
EVALUATION_EPISODES = 100
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
CAPACITY_PROFILES = {
    "H0": (0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
    "H1": (0, 0, 0, 1, 1, 1, 1, 1, 1, 2),
    "H2": (0, 0, 0, 0, 1, 1, 1, 1, 2, 2),
    "H3": (0, 0, 0, 0, 0, 1, 1, 1, 2, 3),
}
NEW_PROFILES = ("H0", "H1", "H3")
METHODS = (
    reproduction.DAOC_PAPER_LABEL,
    DAOC_COORD_LABEL,
    reproduction.OUR_FLAT_DDQN_LABEL,
    reproduction.LEAN_OUR_LABEL,
)
DISPLAY_NAMES = {
    reproduction.DAOC_PAPER_LABEL: "DAOC",
    DAOC_COORD_LABEL: "DAOC + DCC",
    reproduction.OUR_FLAT_DDQN_LABEL: "DDQN + DCC",
    reproduction.LEAN_OUR_LABEL: "OUR",
}

H2_RUNS = {
    reproduction.DAOC_PAPER_LABEL: (
        ROOT / "results/pegasus_pscale/p3_paper_closure/final/runs/daoc_paper"
    ),
    DAOC_COORD_LABEL: (
        ROOT
        / "results/pegasus_pscale/p8_daoc_our_coord_cache/final/runs"
        / DAOC_COORD_LABEL
    ),
    reproduction.OUR_FLAT_DDQN_LABEL: (
        ROOT
        / "results/pegasus_pscale/p6_baselines_ablation/learning/runs"
        / reproduction.OUR_FLAT_DDQN_LABEL
    ),
    reproduction.LEAN_OUR_LABEL: (
        ROOT / "results/pegasus_pscale/p3_paper_closure/final/runs/lean_our"
    ),
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def population_variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def register_profile():
    """Register the controlled DAOC+DCC method and matched train profile."""
    register_suite_extension()
    base = deepcopy(
        reproduction.PROFILES["pegasus_paper_closure_converged"]
    )
    base["labels"] = list(METHODS)
    base["eval_episodes"] = EVALUATION_EPISODES
    base["server_capacity_multiset"] = list(CAPACITY_PROFILES["H2"])
    base["capacity_assignment_namespace"] = CAPACITY_NAMESPACE
    base["method_overrides"] = deepcopy(base.get("method_overrides", {}))
    lean_override = deepcopy(
        reproduction.PROFILES["pegasus_pscale_p2_converged"]
        ["method_overrides"][reproduction.LEAN_OUR_LABEL]
    )
    base["method_overrides"][reproduction.OUR_FLAT_DDQN_LABEL] = (
        lean_override
    )
    reproduction.PROFILES[PROFILE_NAME] = base
    return base


def _summary_arguments(run_dir):
    return read_json(Path(run_dir) / "config.json")["arguments"]


def validate_h2_reuse():
    """Require the reused H2 runs to match the frozen environment."""
    reference_banks = {}
    audit = {}
    for method, method_root in H2_RUNS.items():
        audit[method] = {}
        for seed in SEEDS:
            run_dir = method_root / f"seed_{seed}"
            summary = read_json(run_dir / "summary.json")
            arguments = _summary_arguments(run_dir)
            if summary.get("status") != "complete":
                raise RuntimeError(f"Incomplete H2 run: {method} seed={seed}")
            if not summary.get("eligible_for_comparison"):
                raise RuntimeError(f"Ineligible H2 run: {method} seed={seed}")
            if not summary.get("convergence", {}).get("reached"):
                raise RuntimeError(f"Unconverged H2 run: {method} seed={seed}")
            if tuple(sorted(summary["server_capacities"].values())) != (
                tuple(sorted(CAPACITY_PROFILES["H2"]))
            ):
                raise RuntimeError(f"Wrong H2 capacities: {method} seed={seed}")
            if arguments.get("capacity_assignment_namespace") != (
                CAPACITY_NAMESPACE
            ):
                raise RuntimeError("H2 capacity namespace mismatch")
            if arguments.get("dag_dataset_sha256") != EXPECTED_DATASET_SHA256:
                raise RuntimeError("H2 dataset mismatch")
            if arguments.get("eval_episodes") != EVALUATION_EPISODES:
                raise RuntimeError("H2 evaluation count mismatch")
            bank = read_json(run_dir / "evaluation_scenarios.json")
            comparable = [
                (
                    row["episode"],
                    row["seed"],
                    row["base_fingerprint"],
                    row.get("workflow_family"),
                )
                for row in bank
            ]
            if seed not in reference_banks:
                reference_banks[seed] = comparable
            elif comparable != reference_banks[seed]:
                raise RuntimeError(f"Unpaired H2 scenarios for seed {seed}")
            audit[method][str(seed)] = {
                "run_dir": str(run_dir.resolve()),
                "config_sha256": summary["experiment_config_sha256"],
                "checkpoint_sha256": summary["selected_checkpoint_sha256"],
                "convergence_episode": summary["convergence"]["episode"],
            }
    return audit


def validate_protocol(require_h2=True):
    profile = register_profile()
    if sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Pegasus dataset hash mismatch")
    variances = []
    for name, capacities in CAPACITY_PROFILES.items():
        if len(capacities) != 10 or sum(capacities) != 8:
            raise RuntimeError(f"{name} must contain 10 slots totalling 8")
        if min(capacities) < 0 or max(capacities) > 10:
            raise RuntimeError(f"Invalid capacity in {name}")
        variances.append(population_variance(capacities))
    if variances != sorted(variances) or len(set(variances)) != len(variances):
        raise RuntimeError("Capacity heterogeneity must increase H0 to H3")

    for method in METHODS:
        effective = reproduction.effective_method_profile(profile, method)
        if not effective.get("convergence_mode"):
            raise RuntimeError(f"Convergence disabled for {method}")
        if effective.get("eval_episodes") != EVALUATION_EPISODES:
            raise RuntimeError(f"Wrong evaluation count for {method}")
        if effective.get("validation_scenarios") != 50:
            raise RuntimeError(f"Validation protocol mismatch for {method}")

    h2_audit = validate_h2_reuse() if require_h2 else None
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "families": list(FAMILIES),
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "seeds": list(SEEDS),
        "evaluation_episodes": EVALUATION_EPISODES,
        "capacity_namespace": CAPACITY_NAMESPACE,
        "capacity_profiles": {
            name: {
                "multiset": list(values),
                "total": sum(values),
                "population_variance": population_variance(values),
            }
            for name, values in CAPACITY_PROFILES.items()
        },
        "new_profiles": list(NEW_PROFILES),
        "methods": list(METHODS),
        "display_names": DISPLAY_NAMES,
        "convergence": {
            "validation_scenarios": 50,
            "window": 5,
            "patience": 3,
            "relative_mean_change": 0.05,
            "relative_slope": 0.01,
            "stable_final_checkpoint": True,
            "historical_best_checkpoint": False,
        },
        "h2_reuse_audit": h2_audit,
    }


__all__ = [
    "ANALYSIS_DIR",
    "CAPACITY_NAMESPACE",
    "CAPACITY_PROFILES",
    "DATASET_PATH",
    "DISPLAY_NAMES",
    "EVALUATION_EPISODES",
    "EXPECTED_DATASET_SHA256",
    "FAMILIES",
    "H2_RUNS",
    "METHODS",
    "NEW_PROFILES",
    "PROFILE_NAME",
    "PROTOCOL_VERSION",
    "RESULT_ROOT",
    "RUN_ROOT",
    "SEEDS",
    "TASK_LIMIT_INCLUDING_DUMMY",
    "population_variance",
    "register_profile",
    "validate_h2_reuse",
    "validate_protocol",
]
