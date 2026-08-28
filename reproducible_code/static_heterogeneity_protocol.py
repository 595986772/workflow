"""Frozen protocol for the static cache-heterogeneity experiments."""

from statistics import pvariance


STATIC_HETEROGENEITY_PROTOCOL_VERSION = (
    "static_cache_heterogeneity_v2"
)
# Keep the original capacity permutation so the H3 development
# artifacts remain behaviorally identical across the transparent
# protocol-only hsr0 -> hsr1 revision.
CAPACITY_ASSIGNMENT_NAMESPACE = "static_cache_heterogeneity_v1"
TOTAL_CACHE_BUDGET = 10
BASELINE_RANDOM_DRAW_CAPACITY = 3
MAIN_PROFILE = "H3"

CAPACITY_PROFILES = {
    "H0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "H1": [0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    "H2": [0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
    "H3": [0, 0, 0, 1, 1, 1, 1, 1, 2, 3],
    "H4": [0, 0, 0, 0, 0, 1, 1, 2, 3, 3],
}

CAPACITY_VARIANCES = {
    name: float(pvariance(capacities))
    for name, capacities in CAPACITY_PROFILES.items()
}


def validate_static_capacity_profiles(
    number_of_servers=10,
    number_of_services=10,
):
    """Validate every frozen profile and return immutable copies."""
    validated = {}
    for name, capacities in CAPACITY_PROFILES.items():
        if len(capacities) != int(number_of_servers):
            raise ValueError(
                f"{name} must contain one capacity per server"
            )
        if any(
            not 0 <= capacity <= int(number_of_services)
            for capacity in capacities
        ):
            raise ValueError(
                f"{name} contains an invalid capacity"
            )
        if sum(capacities) != TOTAL_CACHE_BUDGET:
            raise ValueError(
                f"{name} must preserve the total cache budget"
            )
        if max(capacities) > BASELINE_RANDOM_DRAW_CAPACITY:
            raise ValueError(
                f"{name} exceeds the shared initialization draw"
            )
        validated[name] = tuple(capacities)
    return validated


def capacity_text(profile):
    """Return a CLI-compatible capacity multiset."""
    capacities = validate_static_capacity_profiles()[profile]
    return ",".join(str(capacity) for capacity in capacities)


def frozen_environment_spec():
    """Return the environment/statistics fields locked by the protocol."""
    validate_static_capacity_profiles()
    return {
        "protocol_version": (
            STATIC_HETEROGENEITY_PROTOCOL_VERSION
        ),
        "environment": {
            "num_users": 20,
            "num_servers": 10,
            "num_services": 10,
            "num_tasks_per_user": 10,
            "bandwidth_hz": 15000,
            "capacity_profiles": CAPACITY_PROFILES,
            "capacity_variances": CAPACITY_VARIANCES,
            "main_profile": MAIN_PROFILE,
            "total_cache_budget": TOTAL_CACHE_BUDGET,
            "baseline_random_draw_capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "capacity_assignment_namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
            "workload": "stationary",
            "server_load": "static",
        },
        "statistics": {
            "screen_seed_wins": "at_least_2_of_3",
            "converged_seed_wins": "3_of_3",
            "converged_mean_improvement_percent": 5.0,
            "converged_p95": "strictly_better",
            "trained_sweep_seeds": [4, 5, 6, 7, 8],
            "trained_sweep_role": (
                "diagnostic_only_after_daoc_cache_nonconvergence"
            ),
            "final_seeds": list(range(11, 21)),
            "final_seed_wins": "at_least_7_of_10",
            "paired_ci": 0.95,
            "wilcoxon_alternative": "our_better",
            "wilcoxon_alpha": 0.05,
            "cross_capacity_generalization": (
                "zero_shot_from_untouched_h3_checkpoints"
            ),
        },
    }


validate_static_capacity_profiles()
