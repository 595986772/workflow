import unittest
from types import SimpleNamespace

from a0_fixed_budget_heterogeneity_protocol import PROFILE_ORDER
from analyze_a0_fixed_budget_heterogeneity_final import (
    FINAL_SEEDS,
    build_formal_gate,
)
from run_a0_fixed_budget_heterogeneity_final import (
    final_specification,
    reproduction_command,
)


class A0FixedBudgetHeterogeneityFinalTest(unittest.TestCase):
    def test_final_protocol_uses_only_unseen_seeds(self):
        specification = final_specification()
        self.assertEqual(tuple(specification["training"]["seeds"]), FINAL_SEEDS)
        self.assertEqual(FINAL_SEEDS, tuple(range(11, 21)))
        self.assertTrue(
            specification["governance"]["one_time_final_confirmation"]
        )
        self.assertTrue(
            specification["governance"]["no_retuning_from_final_results"]
        )

    def test_final_command_retrains_all_learning_methods(self):
        command = reproduction_command(
            "/tmp/a0-h8v1-final",
            "M8",
            SimpleNamespace(workers=2, resume=True),
        )
        self.assertEqual(
            command[command.index("--seeds") + 1],
            ",".join(str(seed) for seed in FINAL_SEEDS),
        )
        self.assertEqual(
            command[command.index("--labels") + 1],
            "guided_full,centralized_greedy_daoc,lean_our",
        )
        self.assertEqual(
            command[command.index("--seed-partition") + 1], "final"
        )
        self.assertIn("--resume", command)

    def test_formal_gate_requires_all_primary_comparisons(self):
        passing_profile = {
            "integrity": {"paired": True},
            "paired_superiority": {
                "our_vs_guided_full": {"passed": True},
                "our_vs_centralized_greedy_daoc": {"passed": True},
            },
            "p95_our_vs_central": {
                "mean_improvement_sec": 0.01,
                "wins": 7,
            },
            "method_aggregates": {
                "lean_our": {"mean_cache_service_coverage": 0.8}
            },
        }
        profiles = {name: passing_profile for name in PROFILE_ORDER}
        self.assertTrue(build_formal_gate(profiles, True)["passed"])
        failing = {
            name: {
                **passing_profile,
                "paired_superiority": dict(
                    passing_profile["paired_superiority"]
                ),
            }
            for name in PROFILE_ORDER
        }
        failing["M8"]["paired_superiority"][
            "our_vs_centralized_greedy_daoc"
        ] = {"passed": False}
        self.assertFalse(build_formal_gate(failing, True)["passed"])


if __name__ == "__main__":
    unittest.main()
