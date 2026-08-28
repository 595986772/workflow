import unittest

from pegasus_server_scaling_protocol import (
    CAPACITY_PROFILES,
    SERVER_COUNTS,
    TRAINED_SERVER_COUNTS,
    validate_protocol,
)
from run_reproduction_suite import PROFILES, effective_method_profile


class PegasusServerScalingProtocolTests(unittest.TestCase):
    def test_capacity_profiles_preserve_per_server_resources(self):
        specification = validate_protocol()
        self.assertEqual(tuple(specification["server_counts"]), SERVER_COUNTS)
        for servers, capacities in CAPACITY_PROFILES.items():
            self.assertEqual(len(capacities), servers)
            self.assertEqual(sum(capacities), 4 * servers // 5)
            self.assertEqual(capacities.count(0), 2 * servers // 5)
            self.assertEqual(capacities.count(1), 2 * servers // 5)
            self.assertEqual(capacities.count(2), servers // 5)

    def test_only_nonreference_scales_are_retrained(self):
        self.assertEqual(TRAINED_SERVER_COUNTS, (5, 15, 20))
        self.assertNotIn(10, TRAINED_SERVER_COUNTS)

    def test_training_profile_keeps_method_specific_convergence_rules(self):
        profile = PROFILES["pegasus_paper_closure_converged"]
        daoc = effective_method_profile(profile, "daoc_paper")
        ours = effective_method_profile(profile, "lean_our")
        self.assertEqual(daoc["train_episodes"], 50_000)
        self.assertEqual(daoc["convergence_min_episodes"], 15_000)
        self.assertEqual(ours["train_episodes"], 40_000)
        self.assertEqual(ours["convergence_min_episodes"], 5_000)
        self.assertEqual(daoc["eval_episodes"], 100)
        self.assertEqual(ours["eval_episodes"], 100)


if __name__ == "__main__":
    unittest.main()
