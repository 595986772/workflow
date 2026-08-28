import unittest
from types import SimpleNamespace

from a0_fixed_budget_heterogeneity_protocol import (
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_PROFILES,
    CAPACITY_VARIANCES,
    DEVELOPMENT_SEEDS,
    PROFILE_ORDER,
    TOTAL_CACHE_BUDGET,
    frozen_protocol_spec,
    validate_protocol,
)
from capacity_protocol import deterministic_capacity_assignment
from analyze_a0_fixed_budget_heterogeneity import build_gate
from diagnose_a0_coverage_counterfactual import (
    coverage_first_cache_decision,
    coverage_repair_cache_decision,
)
from run_a0_fixed_budget_heterogeneity import (
    reproduction_command,
    revision_spec,
    workload_scenario_view,
)


class A0FixedBudgetHeterogeneityTest(unittest.TestCase):
    def test_profiles_preserve_budget_and_increase_variance(self):
        validate_protocol()
        self.assertTrue(
            all(
                sum(CAPACITY_PROFILES[name]) == TOTAL_CACHE_BUDGET
                for name in PROFILE_ORDER
            )
        )
        self.assertEqual(
            [CAPACITY_VARIANCES[name] for name in PROFILE_ORDER],
            [0.16, 0.56, 0.96],
        )

    def test_profiles_share_one_capacity_rank_per_seed(self):
        marker = deterministic_capacity_assignment(
            list(range(10)),
            number_of_servers=10,
            number_of_services=10,
            seed=3,
            assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
        )
        for profile in PROFILE_ORDER:
            capacities = sorted(CAPACITY_PROFILES[profile])
            expected = {
                server_id: capacities[rank]
                for server_id, rank in marker.items()
            }
            actual = deterministic_capacity_assignment(
                CAPACITY_PROFILES[profile],
                number_of_servers=10,
                number_of_services=10,
                seed=3,
                assignment_namespace=CAPACITY_ASSIGNMENT_NAMESPACE,
            )
            self.assertEqual(actual, expected)

    def test_protocol_is_development_only(self):
        spec = frozen_protocol_spec()
        self.assertEqual(spec["training"]["seeds"], list(DEVELOPMENT_SEEDS))
        self.assertTrue(spec["training"]["from_scratch_per_profile"])
        self.assertTrue(spec["governance"]["development_only"])
        self.assertEqual(
            spec["governance"]["forbidden_final_seeds"],
            list(range(11, 21)),
        )
        self.assertFalse(spec["dataset"]["formal_unbiased_holdout"])

    def test_h8v1_changes_only_our_and_reuses_parent_baselines(self):
        spec = revision_spec("h8v1")
        self.assertEqual(spec["revision"]["parent"], "h8v0")
        self.assertEqual(
            spec["revision"]["changed_module"],
            "scarcity_aware_service_coverage_constraint",
        )
        self.assertEqual(
            spec["training"]["reused_parent_baselines"],
            ["guided_full", "centralized_greedy_daoc"],
        )
        command = reproduction_command(
            directory="/tmp/h8v1-test",
            profile="S8",
            args=SimpleNamespace(
                workers=2,
                revision_id="h8v1",
                resume=False,
            ),
            labels=("lean_our",),
        )
        self.assertEqual(command[command.index("--labels") + 1], "lean_our")

    def test_h8v1_gate_does_not_reuse_h8v0_slope_requirement(self):
        profile = {
            "integrity": {"paired": True},
            "paired_superiority": {
                "our_vs_guided_full": {"passed": True},
                "our_vs_centralized_greedy_daoc": {"passed": True},
            },
            "p95_our_vs_central": {"passed": True},
            "method_aggregates": {
                "lean_our": {
                    "mean_cache_service_coverage": 0.8,
                    "mean_p95_finish_time": 1.0,
                },
                "centralized_greedy_daoc": {
                    "mean_p95_finish_time": 1.1,
                },
            },
        }
        profiles = {name: profile for name in PROFILE_ORDER}
        revised = build_gate(
            profiles,
            cross_profile_paired=True,
            slope=-0.01,
            revision_id="h8v1",
        )
        original = build_gate(
            profiles,
            cross_profile_paired=True,
            slope=-0.01,
            revision_id="h8v0",
        )
        self.assertTrue(revised["passed"])
        self.assertFalse(original["passed"])

    def test_cross_profile_view_ignores_only_capacity_fingerprint(self):
        base = {
            "episode": 1,
            "seed": 100,
            "base_fingerprint": "same-physical-workload",
            "user_graph_keys": {"0": "dag-a"},
        }
        left = [{**base, "fingerprint": "capacity-u8"}]
        right = [{**base, "fingerprint": "capacity-s8"}]
        self.assertEqual(
            workload_scenario_view(left),
            workload_scenario_view(right),
        )

    def test_coverage_counterfactual_fills_distinct_services_first(self):
        decision = coverage_first_cache_decision(
            demand={
                0: {1: 100.0, 2: 5.0, 3: 1.0},
                1: {1: 100.0, 2: 5.0, 3: 1.0},
            },
            capacities={0: 2, 1: 1},
            service_sizes={1: 1.0, 2: 1.0, 3: 1.0},
            cloud_costs={0: 1.0, 1: 1.0},
            between_server_costs={(0, 0): 0.0, (0, 1): 0.5,
                                  (1, 0): 0.5, (1, 1): 0.0},
            server_quality={0: 1.0, 1: 1.0},
        )
        self.assertEqual(sum(map(len, decision.values())), 3)
        self.assertEqual(
            {service for services in decision.values() for service in services},
            {1, 2, 3},
        )
        self.assertLessEqual(len(decision[0]), 2)
        self.assertLessEqual(len(decision[1]), 1)

    def test_coverage_repair_changes_only_a_redundant_slot(self):
        arguments = {
            "demand": {
                0: {1: 100.0, 2: 5.0, 3: 1.0},
                1: {1: 100.0, 2: 5.0, 3: 1.0},
            },
            "capacities": {0: 2, 1: 1},
            "service_sizes": {1: 1.0, 2: 1.0, 3: 1.0},
            "cloud_costs": {0: 1.0, 1: 1.0},
            "between_server_costs": {
                (0, 0): 0.0,
                (0, 1): 0.5,
                (1, 0): 0.5,
                (1, 1): 0.0,
            },
            "server_quality": {0: 1.0, 1: 1.0},
        }
        original = {0: [1, 2], 1: [1]}
        repaired = coverage_repair_cache_decision(
            current_services=original,
            **arguments,
        )
        self.assertEqual(sum(map(len, repaired.values())), 3)
        self.assertEqual(
            {service for services in repaired.values() for service in services},
            {1, 2, 3},
        )
        already_diverse = {0: [1, 2], 1: [3]}
        self.assertEqual(
            coverage_repair_cache_decision(
                current_services=already_diverse,
                **arguments,
            ),
            already_diverse,
        )


if __name__ == "__main__":
    unittest.main()
