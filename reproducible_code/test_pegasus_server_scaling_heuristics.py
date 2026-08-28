import unittest

from pegasus_server_scaling_heuristics_protocol import (
    CALIBRATION_EPISODES,
    FULL_PROFILE,
    METHODS,
    SEEDS,
    TRAINED_SERVER_COUNTS,
    algorithm_config,
    validate_protocol,
)
from run_reproduction_suite import PROFILES, effective_method_profile


class HeuristicServerScalingProtocolTests(unittest.TestCase):
    def test_methods_are_standard_cache_heuristics(self):
        expected_algorithms = {
            "random": "random",
            "nearest": "nearest_server",
            "greedy": "nearest_with_service",
        }
        self.assertEqual(tuple(expected_algorithms), METHODS)
        for method, algorithm in expected_algorithms.items():
            config = algorithm_config(method)
            self.assertEqual(config["algorithm"], algorithm)
            self.assertEqual(
                config.get("cache_policy", "popularity_ema"),
                "popularity_ema",
            )
            self.assertFalse(config.get("cache_coverage_constraint", False))

    def test_heuristics_use_fixed_cache_calibration(self):
        for method in METHODS:
            profile = effective_method_profile(PROFILES[FULL_PROFILE], method)
            self.assertEqual(profile["train_episodes"], CALIBRATION_EPISODES)
            self.assertEqual(profile["cache_freeze_episode"], CALIBRATION_EPISODES)
            self.assertEqual(profile["eval_episodes"], 100)
            self.assertFalse(profile["convergence_mode"])

    def test_new_run_count_is_twenty_seven(self):
        self.assertEqual(
            len(TRAINED_SERVER_COUNTS) * len(METHODS) * len(SEEDS),
            27,
        )

    def test_protocol_and_s10_references_validate(self):
        specification = validate_protocol()
        self.assertEqual(specification["reused_server_count"], 10)
        self.assertEqual(specification["workers"], 6)
        self.assertIn("no OUR coordination", specification["heuristic_identity"])


if __name__ == "__main__":
    unittest.main()
