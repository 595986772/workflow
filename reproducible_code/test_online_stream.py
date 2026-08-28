import unittest
from types import SimpleNamespace

from evaluate_online_stream import (
    apply_server_load_shift,
    deterministic_uniform,
    estimated_control_payload_bytes,
    hotspot_sets,
    observed_server_execution_latencies,
    performance_metric_names,
    recovery_metrics,
    validate_dynamic_accounting,
)


class OnlineStreamTest(unittest.TestCase):
    def test_load_shift_aggregation_uses_migration_adjusted_p95(self):
        self.assertEqual(
            performance_metric_names("server_load_shift"),
            (
                "migration_adjusted_finish_time",
                "migration_adjusted_p95_finish_time",
            ),
        )
        self.assertEqual(
            performance_metric_names("service_hotspot_shift"),
            ("average_finish_time", "p95_finish_time"),
        )

    def test_hotspot_sets_are_disjoint_at_paper_scale(self):
        before, after = hotspot_sets(10)
        self.assertEqual(before, [1, 2, 3])
        self.assertEqual(after, [8, 9, 10])
        self.assertFalse(set(before) & set(after))

    def test_deterministic_uniform_is_stable_and_bounded(self):
        first = deterministic_uniform(7, "pre", 1, "task")
        second = deterministic_uniform(7, "pre", 1, "task")
        different = deterministic_uniform(8, "pre", 1, "task")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

    def test_control_payload_uses_sufficient_statistics(self):
        simulator = SimpleNamespace(
            cache_policy="critical_path_joint",
            S=10,
            Q=10,
            input_dict={"server capacity": 2},
        )
        self.assertEqual(
            estimated_control_payload_bytes(simulator, 3),
            1800,
        )
        simulator.cache_policy = "popularity_ema"
        self.assertEqual(
            estimated_control_payload_bytes(simulator, 3),
            0,
        )

    def test_recovery_is_measured_against_post_shift_steady_state(self):
        values = [1.0] * 4 + [2.0, 1.5, 1.1, 1.0, 1.0, 1.0]
        rows = [
            {
                "episode": index + 1,
                "average_finish_time": value,
            }
            for index, value in enumerate(values)
        ]
        metrics = recovery_metrics(
            rows,
            shift_episode=5,
            recovery_window=2,
        )
        self.assertGreater(
            metrics["normalized_transient_regret"],
            0.0,
        )
        self.assertIsNotNone(
            metrics["adaptation_delay_windows"]
        )
        self.assertAlmostEqual(
            metrics["recovery_tolerance"],
            1.05,
        )

    def test_load_shift_changes_only_selected_post_shift_servers(self):
        simulator = SimpleNamespace(
            servers={
                0: SimpleNamespace(load=1.0),
                1: SimpleNamespace(load=2.0),
                2: SimpleNamespace(load=3.0),
            }
        )
        descriptor = apply_server_load_shift(
            simulator,
            segment="post_shift",
            shifted_servers={0: 0, 1: 2},
            load_multiplier=4.0,
        )
        self.assertEqual(simulator.servers[0].load, 4.0)
        self.assertEqual(simulator.servers[1].load, 2.0)
        self.assertEqual(simulator.servers[2].load, 12.0)
        self.assertEqual(descriptor["shifted_server_ids"], [0, 2])

    def test_dynamic_accounting_carries_migration_to_next_window(self):
        rows = [
            {
                "average_finish_time": 1.0,
                "p95_finish_time": 1.5,
                "cache_migration_critical_time_sec": 0.2,
                "migration_delay_applied_sec": 0.0,
                "migration_adjusted_finish_time": 1.0,
                "migration_adjusted_p95_finish_time": 1.5,
                "dynamic_oracle_capacity_ok": 1,
            },
            {
                "average_finish_time": 1.1,
                "p95_finish_time": 1.6,
                "cache_migration_critical_time_sec": 0.0,
                "migration_delay_applied_sec": 0.2,
                "migration_adjusted_finish_time": 1.3,
                "migration_adjusted_p95_finish_time": 1.8,
                "dynamic_oracle_capacity_ok": 1,
            },
        ]
        self.assertTrue(validate_dynamic_accounting(rows))
        rows[1]["migration_delay_applied_sec"] = 0.0
        with self.assertRaises(RuntimeError):
            validate_dynamic_accounting(rows)

    def test_observed_server_latency_uses_only_completed_tasks(self):
        simulator = SimpleNamespace(
            servers={0: object(), 1: object()},
            users={
                0: SimpleNamespace(
                    done_tasks={
                        "a": SimpleNamespace(
                            assigned_server=0,
                            result=SimpleNamespace(
                                computing_latency=0.2,
                                waiting_latency=0.3,
                            ),
                        )
                    }
                )
            },
        )
        self.assertEqual(
            observed_server_execution_latencies(simulator),
            {"0": 0.5, "1": None},
        )


if __name__ == "__main__":
    unittest.main()
