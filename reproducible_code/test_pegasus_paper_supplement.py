import unittest
from types import SimpleNamespace

from analyze_pegasus_paper_supplement import paired_delta_summary
from evaluate_pegasus_user_scaling import coordination_payload
from pegasus_paper_closure_protocol import FINAL_SEEDS
from pegasus_paper_supplement_protocol import (
    ABLATION_METHODS,
    ABLATION_SEEDS,
    ALGORITHM_SOURCE_FILES,
    CAPACITY_PROFILES,
    SCALING_USER_COUNTS,
    SUPPLEMENT_SOURCE_FILES,
    supplement_source_hash,
    validate_parent_freeze,
    validate_protocol,
)


class PegasusPaperSupplementTest(unittest.TestCase):
    def test_parent_algorithm_and_final_gate_remain_frozen(self):
        parent = validate_parent_freeze()
        self.assertEqual(len(parent["algorithm_source_sha256"]), 64)
        self.assertEqual(len(parent["p3_protocol_source_sha256"]), 64)

    def test_capacity_control_has_equal_budget_and_distinct_variance(self):
        uniform = CAPACITY_PROFILES["uniform_b10"]
        heterogeneous = CAPACITY_PROFILES["heterogeneous_b10"]
        self.assertEqual(sum(uniform), 10)
        self.assertEqual(sum(heterogeneous), 10)
        self.assertEqual(len(uniform), len(heterogeneous))
        self.assertEqual(len(set(uniform)), 1)
        self.assertGreater(len(set(heterogeneous)), 1)

    def test_primary_ablations_pair_with_final_seeds(self):
        self.assertEqual(tuple(ABLATION_SEEDS), tuple(FINAL_SEEDS))
        self.assertEqual(
            set(ABLATION_METHODS),
            {"our_dqn", "our_no_coord_cache"},
        )
        self.assertNotIn("our_no_telemetry", ABLATION_METHODS)

    def test_scaling_includes_reference_and_stress_counts(self):
        self.assertEqual(tuple(SCALING_USER_COUNTS), (10, 20, 40, 60))

    def test_supplement_does_not_modify_algorithm_file_set(self):
        self.assertTrue(
            set(SUPPLEMENT_SOURCE_FILES).isdisjoint(ALGORITHM_SOURCE_FILES)
        )
        self.assertEqual(len(supplement_source_hash()), 64)

    def test_protocol_is_self_consistent(self):
        spec = validate_protocol()
        self.assertEqual(len(spec["specification_sha256"]), 64)
        self.assertTrue(spec["governance"]["algorithm_changes_forbidden"])

    def test_coordination_payload_distinguishes_local_and_central(self):
        servers = {
            0: SimpleNamespace(services=[0, 1], capacity=1),
            1: SimpleNamespace(services=[0, 2], capacity=1),
        }
        broker = SimpleNamespace(
            H={0: {1: 0.4}, 1: {2: 0.6}},
            last_cache_decision_context={"expected_requests": {1: 2, 2: 3}},
            cache_server_compute_per_mcycle_ema={0: 0.1, 1: 0.2},
            cache_server_waiting_latency_ema={0: 0.01, 1: 0.02},
            cache_server_sample_counts={0: 10, 1: 11},
            cache_server_last_observed_window={0: 5, 1: 5},
        )
        local = SimpleNamespace(
            cache_policy="paper_popularity_cost_ema",
            servers=servers,
            broker=broker,
        )
        central = SimpleNamespace(
            cache_policy="critical_path_joint",
            servers=servers,
            broker=broker,
        )
        decisions = {0: [1], 1: [2]}
        self.assertEqual(coordination_payload(local, decisions)["total_bytes"], 0)
        self.assertGreater(
            coordination_payload(central, decisions)["total_bytes"],
            0,
        )

    def test_difference_in_advantage_summary(self):
        summary = paired_delta_summary([0.1, 0.2, 0.3])
        self.assertAlmostEqual(summary["mean_delta_sec"], 0.2)
        self.assertEqual(summary["positive_seeds"], 3)


if __name__ == "__main__":
    unittest.main()
