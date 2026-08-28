import unittest

import run_reproduction_suite as reproduction
from pegasus_common_horizon_protocol import (
    CAPACITY_MULTISET,
    EVALUATION_EPISODES,
    FIXED_TRAIN_EPISODES,
    PROFILE_NAME,
    RERUN_METHODS,
    TAIL_EPISODES,
    TAIL_START_EPISODE,
    register_profile,
    validate_protocol,
)


class PegasusCommonHorizonProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = register_profile()

    def test_fixed_budget_and_tail_are_exact(self):
        self.assertEqual(FIXED_TRAIN_EPISODES, 26_000)
        self.assertEqual(TAIL_START_EPISODE, 25_001)
        self.assertEqual(TAIL_EPISODES, 1_000)

    def test_all_rerun_methods_use_the_same_horizon(self):
        for label in RERUN_METHODS:
            effective = reproduction.effective_method_profile(
                self.profile,
                label,
            )
            self.assertEqual(effective["train_episodes"], 26_000)
            self.assertEqual(
                effective["eval_episodes"],
                EVALUATION_EPISODES,
            )
            self.assertEqual(effective["checkpoint_every"], 0)
            self.assertEqual(effective["validation_scenarios"], 0)
            self.assertFalse(effective["convergence_mode"])
            self.assertTrue(effective["eval_scenario_bank"])
            self.assertEqual(
                effective["eval_bank_scope"],
                "infrastructure",
            )

    def test_environment_remains_pegasus_b8(self):
        self.assertEqual(PROFILE_NAME, "pegasus_common_horizon_26k")
        self.assertEqual(tuple(CAPACITY_MULTISET), (0, 0, 0, 0, 1, 1, 1, 1, 2, 2))
        self.assertEqual(self.profile["num_tasks"], 31)
        self.assertEqual(self.profile["bandwidth"], 15_000)

    def test_complete_protocol_validation(self):
        specification = validate_protocol()
        self.assertEqual(
            specification["rerun_methods"],
            list(RERUN_METHODS),
        )
        self.assertEqual(
            specification["checkpoint_rule"],
            "fixed_budget_final",
        )


if __name__ == "__main__":
    unittest.main()
