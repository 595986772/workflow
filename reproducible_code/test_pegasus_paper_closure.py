from types import SimpleNamespace
import unittest

from broker import Broker, DAOC_PAPER_CACHE_POLICY
from pegasus_paper_closure_protocol import (
    CAPACITY_MULTISET,
    DEVELOPMENT_SEEDS,
    FINAL_SEEDS,
    validate_protocol,
)
from run_reproduction_suite import (
    ALGORITHMS,
    DAOC_PAPER_LABEL,
    PEGASUS_PAPER_CLOSURE_LABELS,
    PROFILES,
    effective_method_profile,
)
from analyze_pegasus_paper_closure import development_claim_tier


class PegasusPaperClosureTest(unittest.TestCase):
    @staticmethod
    def _broker(cache_policy, capacity=1):
        broker = Broker.__new__(Broker)
        broker.numberofservices = 2
        broker.numberofservers = 1
        broker.H = {0: {1: 0.0, 2: 0.0}}
        broker.cache_observations = 0
        broker.simulator = SimpleNamespace(
            cache_policy=cache_policy,
            cache_score_alpha=0.1,
            service_data_length={0: 0.0, 1: 100.0, 2: 400.0},
            servers={
                0: SimpleNamespace(
                    rate_to_cloud=100.0,
                    capacity=capacity,
                    services=[0],
                )
            },
        )
        return broker

    def test_paper_cache_ranks_popularity_times_loading_time(self):
        broker = self._broker(DAOC_PAPER_CACHE_POLICY)
        broker.caching_decisions_update(0, 1, cost=0.0)
        broker.caching_decisions_update(0, 2, cost=999.0)
        self.assertAlmostEqual(broker.H[0][1], 0.09)
        self.assertAlmostEqual(broker.H[0][2], 0.4)
        self.assertEqual(broker.caching_decisions(0), [2])

    def test_paper_cache_does_not_use_realized_fetch_cost(self):
        first = self._broker(DAOC_PAPER_CACHE_POLICY)
        second = self._broker(DAOC_PAPER_CACHE_POLICY)
        first.caching_decisions_update(0, 2, cost=0.0)
        second.caching_decisions_update(0, 2, cost=1e9)
        self.assertEqual(first.H, second.H)

    def test_released_code_popularity_behavior_is_unchanged(self):
        broker = self._broker("popularity_ema")
        broker.caching_decisions_update(0, 1, cost=0.0)
        broker.caching_decisions_update(0, 2, cost=999.0)
        self.assertAlmostEqual(broker.H[0][1], 0.09)
        self.assertAlmostEqual(broker.H[0][2], 0.1)

    def test_zero_capacity_server_still_has_empty_cache_decision(self):
        broker = self._broker(DAOC_PAPER_CACHE_POLICY, capacity=0)
        broker.caching_decisions_update(0, 2, cost=1.0)
        self.assertEqual(broker.caching_decisions(0), [])

    def test_daoc_paper_configuration_matches_equation_14_protocol(self):
        config = next(
            row for row in ALGORITHMS
            if row["label"] == DAOC_PAPER_LABEL
        )
        self.assertEqual(config["beta"], 0.9)
        self.assertEqual(config["beta_min"], 0.1)
        self.assertEqual(config["beta_decay"], 0.995)
        self.assertEqual(config["reward_mode"], "terminal_binary")
        self.assertEqual(
            config["cache_policy"],
            DAOC_PAPER_CACHE_POLICY,
        )

    def test_all_ablations_use_the_same_cache_calibration_horizon(self):
        profile = PROFILES["pegasus_paper_closure_converged"]
        for label in PEGASUS_PAPER_CLOSURE_LABELS:
            method = effective_method_profile(profile, label)
            self.assertEqual(method["cache_freeze_episode"], 5000)
            self.assertEqual(method["validation_scenarios"], 50)
        for label in (
            "our_dqn",
            "our_no_telemetry",
            "our_no_coord_cache",
        ):
            method = effective_method_profile(profile, label)
            self.assertEqual(method["convergence_min_episodes"], 5000)

    def test_protocol_keeps_development_and_final_seeds_disjoint(self):
        protocol = validate_protocol()
        self.assertFalse(set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS))
        self.assertEqual(sum(CAPACITY_MULTISET), 8)
        self.assertEqual(protocol["capacity_namespace"], "pegasus_pscale_p2")

    def test_claim_tiers_require_three_seed_consistency_for_primary(self):
        comparison = {
            "wins": 3,
            "mean_improvement_percent": 26.0,
            "passed": True,
        }
        self.assertEqual(development_claim_tier(comparison), "primary")
        comparison["wins"] = 2
        self.assertEqual(development_claim_tier(comparison), "secondary")
        comparison["passed"] = False
        self.assertEqual(development_claim_tier(comparison), "unsupported")


if __name__ == "__main__":
    unittest.main()
