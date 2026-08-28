"""Protocol for the A0 fixed-budget cache-heterogeneity study."""

from statistics import pvariance

from a0_coordination_protocol import (
    BANDWIDTH_HZ,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    SERVERS,
    SERVICES,
    TASK_LIMIT,
    USERS,
    file_sha256,
)


PROTOCOL_VERSION = "a0_fixed_budget_heterogeneity_v1"
CAPACITY_ASSIGNMENT_NAMESPACE = PROTOCOL_VERSION
TOTAL_CACHE_BUDGET = 8
BASELINE_RANDOM_DRAW_CAPACITY = 3
PROFILE_ORDER = ("U8", "M8", "S8")
CAPACITY_PROFILES = {
    "U8": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    "M8": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
    "S8": [0, 0, 0, 0, 0, 1, 1, 1, 2, 3],
}
PROFILE_NAMES = {
    "U8": "balanced",
    "M8": "moderate",
    "S8": "strong",
}
CAPACITY_VARIANCES = {
    profile: float(pvariance(capacities))
    for profile, capacities in CAPACITY_PROFILES.items()
}
METHOD_LABELS = (
    "guided_full",
    "centralized_greedy_daoc",
    "lean_our",
)
DEVELOPMENT_SEEDS = (1, 2, 3)


def validate_protocol():
    """Validate the dataset and all fixed-budget capacity profiles."""
    if file_sha256(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise ValueError("Alibaba-CP100-A0 checksum changed")
    previous_variance = -1.0
    for profile in PROFILE_ORDER:
        capacities = CAPACITY_PROFILES[profile]
        if len(capacities) != SERVERS:
            raise ValueError(f"{profile} must define {SERVERS} capacities")
        if capacities != sorted(capacities):
            raise ValueError(f"{profile} must use a sorted multiset")
        if sum(capacities) != TOTAL_CACHE_BUDGET:
            raise ValueError(f"{profile} must preserve budget 8")
        if any(not 0 <= value <= SERVICES for value in capacities):
            raise ValueError(f"{profile} contains an invalid capacity")
        if max(capacities) > BASELINE_RANDOM_DRAW_CAPACITY:
            raise ValueError(f"{profile} exceeds the shared draw capacity")
        variance = CAPACITY_VARIANCES[profile]
        if variance <= previous_variance:
            raise ValueError("Profiles must have increasing heterogeneity")
        previous_variance = variance
    return frozen_protocol_spec()


def capacity_text(profile):
    return ",".join(str(value) for value in CAPACITY_PROFILES[profile])


def frozen_protocol_spec():
    """Return the preregistered development experiment specification."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": "A0_controlled_mechanism_only",
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": EXPECTED_DATASET_SHA256,
            "formal_unbiased_holdout": False,
        },
        "environment": {
            "users": USERS,
            "servers": SERVERS,
            "services": SERVICES,
            "task_limit": TASK_LIMIT,
            "bandwidth_hz": BANDWIDTH_HZ,
            "total_cache_budget": TOTAL_CACHE_BUDGET,
            "capacity_profiles": CAPACITY_PROFILES,
            "capacity_variances": CAPACITY_VARIANCES,
            "capacity_assignment_namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
            "baseline_random_draw_capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "workload": "stationary",
            "only_capacity_distribution_changes": True,
        },
        "methods": list(METHOD_LABELS),
        "training": {
            "from_scratch_per_profile": True,
            "convergence_profile": "e2_converged",
            "seeds": list(DEVELOPMENT_SEEDS),
            "frozen_evaluation_scenarios_per_seed": 100,
            "source_algorithm_freeze": "a0r2",
        },
        "development_gate": {
            "our_beats_daoc_in_each_profile": "mean_and_2_of_3_wins",
            "our_beats_central_in_strong_profile": (
                "positive_mean_and_2_of_3_wins"
            ),
            "central_advantage_slope_vs_variance": "positive",
            "strong_profile_p95": "our_not_worse_than_central",
        },
        "governance": {
            "development_only": True,
            "forbidden_final_seeds": list(range(11, 21)),
            "no_algorithm_retuning_from_a0_final_results": True,
        },
    }


validate_protocol()
