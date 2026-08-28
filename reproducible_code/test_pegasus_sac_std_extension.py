import unittest

import run_reproduction_suite as reproduction
from pegasus_sac_std_extension_protocol import (
    COORD_SAC_LABEL,
    PROFILE_CONVERGED,
    STD_SAC_LABEL,
    algorithm_config,
    register_suite_extension,
    validate_protocol,
)


class StandardCacheDiscreteSacProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_suite_extension()

    def test_protocol_validates(self):
        specification = validate_protocol()
        self.assertEqual(specification["new_method"], STD_SAC_LABEL)
        self.assertTrue(
            specification["controlled_difference"][
                "all_sac_state_reward_action_and_training_settings_matched"
            ]
        )

    def test_only_cache_subsystem_differs(self):
        coordinated = algorithm_config(COORD_SAC_LABEL)
        standard = algorithm_config(STD_SAC_LABEL)
        for key in (
            "algorithm",
            "family",
            "beta",
            "beta_min",
            "beta_decay",
            "reward_mode",
            "potential_reward_weight",
            "gamma",
            "n_step",
            "entropy_coefficient",
            "sac_target_entropy_ratio",
            "sac_target_tau",
            "historical_feedback_guidance",
            "adaptive_guidance_gate",
            "training_objective",
        ):
            self.assertEqual(standard.get(key), coordinated.get(key), key)
        self.assertEqual(standard["cache_policy"], "popularity_ema")
        self.assertFalse(standard["cache_server_quality"])
        self.assertFalse(standard["cache_coverage_constraint"])
        self.assertFalse(standard["cache_dependency_awareness"])

    def test_training_profile_matches_coordinated_sac(self):
        coordinated = reproduction.effective_method_profile(
            reproduction.PROFILES["pegasus_baseline_sac_converged"],
            COORD_SAC_LABEL,
        )
        standard = reproduction.effective_method_profile(
            reproduction.PROFILES[PROFILE_CONVERGED],
            STD_SAC_LABEL,
        )
        coordinated.pop("labels", None)
        standard.pop("labels", None)
        self.assertEqual(standard, coordinated)


if __name__ == "__main__":
    unittest.main()
