import unittest

from pegasus_baseline_extension_protocol import (
    CAPACITY_MULTISET,
    DEVELOPMENT_SEEDS,
    FINAL_SEEDS,
    HEURISTIC_METHODS,
    SAC_CONFIG,
    SAC_METHOD,
    validate_protocol,
)
from run_reproduction_suite import ALGORITHMS, PROFILES


class PegasusBaselineExtensionProtocolTest(unittest.TestCase):
    def test_protocol_is_paired_post_lock_b8(self):
        protocol = validate_protocol()

        self.assertEqual(sum(CAPACITY_MULTISET), 8)
        self.assertEqual(len(CAPACITY_MULTISET), 10)
        self.assertFalse(set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS))
        self.assertTrue(
            protocol["governance"][
                "our_checkpoint_and_results_immutable"
            ]
        )
        self.assertTrue(
            protocol["governance"][
                "post_lock_extension_not_new_holdout"
            ]
        )

    def test_new_method_definitions_use_same_coordinated_cache(self):
        methods = {entry["label"]: entry for entry in ALGORITHMS}
        for label in HEURISTIC_METHODS:
            method = methods[label]
            self.assertEqual(method["family"], "heuristic")
            self.assertEqual(
                method["cache_policy"],
                "critical_path_joint",
            )
            self.assertTrue(method["cache_coverage_constraint"])

        sac = methods[SAC_METHOD]
        self.assertEqual(sac["algorithm"], SAC_CONFIG["algorithm"])
        self.assertEqual(sac["reward_mode"], SAC_CONFIG["reward_mode"])
        self.assertEqual(sac["cache_policy"], SAC_CONFIG["cache_policy"])
        self.assertTrue(sac["cache_coverage_constraint"])
        self.assertEqual(sac["gamma"], 1.0)
        self.assertEqual(sac["n_step"], 3)
        self.assertFalse(sac["historical_feedback_guidance"])
        self.assertFalse(sac["adaptive_guidance_gate"])

    def test_heuristics_calibrate_cache_without_neural_training(self):
        profile = PROFILES["pegasus_baseline_heuristics"]
        self.assertEqual(tuple(profile["labels"]), HEURISTIC_METHODS)
        for label in HEURISTIC_METHODS:
            override = profile["method_overrides"][label]
            self.assertEqual(override["train_episodes"], 5000)
            self.assertEqual(override["cache_freeze_episode"], 5000)
            self.assertEqual(override["checkpoint_every"], 0)
            self.assertFalse(override["convergence_mode"])

    def test_sac_convergence_profile_matches_our_budget(self):
        profile = PROFILES["pegasus_baseline_sac_converged"]
        override = profile["method_overrides"][SAC_METHOD]

        self.assertEqual(profile["num_users"], 20)
        self.assertEqual(profile["num_servers"], 10)
        self.assertEqual(profile["num_services"], 10)
        self.assertEqual(profile["num_tasks"], 31)
        self.assertEqual(profile["bandwidth"], 15000)
        self.assertEqual(
            profile["server_capacity_multiset"],
            list(CAPACITY_MULTISET),
        )
        self.assertEqual(override["train_episodes"], 40000)
        self.assertEqual(override["cache_freeze_episode"], 5000)
        self.assertEqual(override["validation_scenarios"], 50)
        self.assertTrue(override["convergence_mode"])


if __name__ == "__main__":
    unittest.main()
