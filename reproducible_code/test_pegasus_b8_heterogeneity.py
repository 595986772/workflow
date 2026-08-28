import unittest

import run_reproduction_suite as reproduction
from capacity_protocol import deterministic_capacity_assignment
from pegasus_b8_heterogeneity_protocol import (
    CAPACITY_NAMESPACE,
    CAPACITY_PROFILES,
    METHODS,
    NEW_PROFILES,
    PROFILE_NAME,
    SEEDS,
    population_variance,
    validate_protocol,
)


class PegasusB8HeterogeneityProtocolTests(unittest.TestCase):
    def test_profiles_hold_budget_and_increase_variance(self):
        variances = []
        for capacities in CAPACITY_PROFILES.values():
            self.assertEqual(len(capacities), 10)
            self.assertEqual(sum(capacities), 8)
            variances.append(population_variance(capacities))
        self.assertEqual(variances, sorted(variances))
        self.assertEqual(len(variances), len(set(variances)))

    def test_shared_namespace_is_deterministic(self):
        for seed in SEEDS:
            for profile in CAPACITY_PROFILES.values():
                first = deterministic_capacity_assignment(
                    profile, 10, 10, seed, CAPACITY_NAMESPACE
                )
                second = deterministic_capacity_assignment(
                    profile, 10, 10, seed, CAPACITY_NAMESPACE
                )
                self.assertEqual(first, second)

    def test_all_new_profiles_are_explicit(self):
        self.assertEqual(NEW_PROFILES, ("H0", "H1", "H3"))

    def test_registered_methods_share_convergence_protocol(self):
        validate_protocol(require_h2=False)
        profile = reproduction.PROFILES[PROFILE_NAME]
        self.assertEqual(tuple(profile["labels"]), METHODS)
        for method in METHODS:
            effective = reproduction.effective_method_profile(profile, method)
            self.assertTrue(effective["convergence_mode"])
            self.assertEqual(effective["validation_scenarios"], 50)
            self.assertEqual(effective["eval_episodes"], 100)


if __name__ == "__main__":
    unittest.main()
