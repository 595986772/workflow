import unittest

from pegasus_server_scaling_strong_baselines_protocol import (
    COORD_SAC,
    METHODS,
    OUR_DQN,
    SEEDS,
    TRAINED_SERVER_COUNTS,
    TRAINING_PROFILES,
    algorithm_config,
    validate_protocol,
)
from run_reproduction_suite import PROFILES, effective_method_profile


class StrongBaselineServerScalingProtocolTests(unittest.TestCase):
    def test_only_strict_scheduler_replacements_are_selected(self):
        self.assertEqual(METHODS, (OUR_DQN, COORD_SAC))
        self.assertNotIn("our_dqn", METHODS)
        self.assertTrue(algorithm_config(OUR_DQN)["cache_coverage_constraint"])
        self.assertTrue(algorithm_config(COORD_SAC)["cache_coverage_constraint"])

    def test_both_methods_keep_the_our_cache_and_reward(self):
        ours = algorithm_config("lean_our")
        for method in METHODS:
            candidate = algorithm_config(method)
            self.assertEqual(candidate["cache_policy"], ours["cache_policy"])
            self.assertEqual(candidate["reward_mode"], ours["reward_mode"])
            self.assertEqual(
                candidate["cache_coverage_constraint"],
                ours["cache_coverage_constraint"],
            )

    def test_convergence_rules_match_existing_s10_runs(self):
        for method, profile_name in TRAINING_PROFILES.items():
            profile = effective_method_profile(PROFILES[profile_name], method)
            self.assertEqual(profile["train_episodes"], 40_000)
            self.assertEqual(profile["checkpoint_every"], 500)
            self.assertEqual(profile["convergence_min_episodes"], 5_000)
            self.assertTrue(profile["convergence_mode"])

    def test_new_training_run_count_is_eighteen(self):
        self.assertEqual(len(TRAINED_SERVER_COUNTS) * len(METHODS) * len(SEEDS), 18)

    def test_full_protocol_and_s10_references_validate(self):
        specification = validate_protocol()
        self.assertEqual(specification["reused_server_count"], 10)
        self.assertEqual(specification["workers"], 6)


if __name__ == "__main__":
    unittest.main()

