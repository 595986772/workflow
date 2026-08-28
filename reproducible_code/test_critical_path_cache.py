import unittest
from types import SimpleNamespace

import numpy as np

from broker import Broker
from critical_path_cache import (
    alternative_service_fetch_delays,
    coordinated_cache_decision,
    dependency_locality_bonuses,
    exponential_moving_average,
    history_only_server_quality,
    hysteretic_cache_decision,
    service_fetch_savings,
    workload_normalized_server_telemetry,
)


class PrivilegedStateGuardServer:
    def __init__(self, services, rate_to_cloud):
        self.services = services
        self.rate_to_cloud = rate_to_cloud

    @property
    def load(self):
        raise AssertionError("cache decision read exact server.load")

    @property
    def frequency(self):
        raise AssertionError(
            "cache decision read exact server.frequency"
        )


class CriticalPathCacheTest(unittest.TestCase):
    def test_ema_uses_first_observation_then_smooths(self):
        first = exponential_moving_average(
            previous=None,
            observation=10.0,
            alpha=0.25,
        )
        second = exponential_moving_average(
            previous=first,
            observation=14.0,
            alpha=0.25,
        )
        self.assertEqual(first, 10.0)
        self.assertEqual(second, 11.0)

    def test_history_only_quality_uses_observed_latency_and_fallback(self):
        quality = history_only_server_quality(
            server_execution_latency_ema={
                0: 2.0,
                1: 1.0,
                2: None,
            },
            global_execution_latency_ema=1.5,
            compute_weight=1.0,
            compute_intensity=1.0,
        )
        self.assertAlmostEqual(quality[0], 0.5)
        self.assertAlmostEqual(quality[1], 1.0)
        self.assertAlmostEqual(quality[2], 2.0 / 3.0)

    def test_normalized_telemetry_separates_workload_from_compute_rate(self):
        telemetry = workload_normalized_server_telemetry(
            server_compute_per_mcycle_ema={
                0: 0.01,
                1: 0.01,
            },
            server_waiting_latency_ema={
                0: 0.2,
                1: 0.2,
            },
            global_compute_per_mcycle_ema=0.01,
            global_waiting_latency_ema=0.2,
            server_sample_counts={0: 20, 1: 20},
            server_last_observed_window={0: 4, 1: 4},
            current_window=4,
            min_samples=5,
            freshness_half_life=10,
        )

        self.assertEqual(telemetry[0], telemetry[1])
        self.assertEqual(telemetry[0][:2], (1.0, 1.0))
        self.assertAlmostEqual(telemetry[0][2], 0.8)

    def test_normalized_telemetry_downweights_stale_observations(self):
        telemetry = workload_normalized_server_telemetry(
            server_compute_per_mcycle_ema={
                0: 0.02,
                1: 0.01,
            },
            server_waiting_latency_ema={
                0: 0.4,
                1: 0.2,
            },
            global_compute_per_mcycle_ema=0.01,
            global_waiting_latency_ema=0.2,
            server_sample_counts={0: 20, 1: 20},
            server_last_observed_window={0: 0, 1: 10},
            current_window=10,
            min_samples=5,
            freshness_half_life=10,
        )

        self.assertAlmostEqual(telemetry[0][2], 0.4)
        self.assertAlmostEqual(telemetry[1][2], 0.8)
        self.assertLess(telemetry[0][0], telemetry[1][0])
        self.assertLess(telemetry[0][1], telemetry[1][1])

    def test_history_is_not_visible_until_completed_window_finalizes(self):
        broker = Broker.__new__(Broker)
        broker.numberofservers = 2
        broker.cache_history_windows = 0
        broker.cache_history_window_requests = 0
        broker.cache_history_window_cpu_cycles = 0.0
        broker.cache_history_window_server_latency = {
            0: 0.0,
            1: 0.0,
        }
        broker.cache_history_window_server_samples = {
            0: 0,
            1: 0,
        }
        broker.cache_history_window_server_compute_per_mcycle = {
            0: 0.0,
            1: 0.0,
        }
        broker.cache_history_window_server_compute_samples = {
            0: 0,
            1: 0,
        }
        broker.cache_history_window_server_waiting_latency = {
            0: 0.0,
            1: 0.0,
        }
        broker.cache_expected_requests_ema = None
        broker.cache_mean_cpu_cycles_ema = None
        broker.cache_global_execution_latency_ema = None
        broker.cache_global_compute_per_mcycle_ema = None
        broker.cache_global_waiting_latency_ema = None
        broker.cache_server_execution_latency_ema = {
            0: None,
            1: None,
        }
        broker.cache_server_compute_per_mcycle_ema = {
            0: None,
            1: None,
        }
        broker.cache_server_waiting_latency_ema = {
            0: None,
            1: None,
        }
        broker.cache_server_sample_counts = {0: 0, 1: 0}
        broker.cache_server_last_observed_window = {
            0: None,
            1: None,
        }
        broker.last_server_telemetry_context = None
        broker.simulator = SimpleNamespace(
            cache_history_alpha=0.5,
        )
        task = SimpleNamespace(
            assigned_server=0,
            cpu_cycle=100.0,
            result=SimpleNamespace(
                computing_latency=1.5,
                waiting_latency=0.5,
            ),
        )

        broker._observe_causal_cache_history(task)
        self.assertIsNone(broker.cache_expected_requests_ema)
        self.assertIsNone(broker.cache_mean_cpu_cycles_ema)
        self.assertIsNone(
            broker.cache_server_execution_latency_ema[0]
        )

        broker._finalize_causal_cache_history()
        self.assertEqual(broker.cache_expected_requests_ema, 1.0)
        self.assertEqual(broker.cache_mean_cpu_cycles_ema, 100.0)
        self.assertEqual(
            broker.cache_server_execution_latency_ema[0],
            2.0,
        )
        self.assertIsNone(
            broker.cache_server_execution_latency_ema[1]
        )
        self.assertEqual(broker.cache_history_windows, 1)

    def test_cache_runtime_state_round_trip(self):
        source = Broker.__new__(Broker)
        source.cache_replacements = 7
        source.cache_update_events = 4
        source.cache_decision_calls = 6
        source.cache_decision_wall_time_sec = 0.75
        source.last_cache_decision_wall_time_sec = 0.125
        source.cache_round = 13
        source.cache_observations = 29
        source.cache_window_observations = 3
        source.cache_window_values = {
            0: {1: 0.25, 2: 0.75},
            1: {1: 1.25, 2: 1.75},
        }
        source.last_cache_change_round = {0: 10, 1: 12}

        restored = Broker.__new__(Broker)
        restored.load_cache_runtime_state_dict(
            source.cache_runtime_state_dict()
        )

        self.assertEqual(
            restored.cache_runtime_state_dict(),
            source.cache_runtime_state_dict(),
        )

    def test_joint_decision_cannot_read_privileged_current_state(self):
        broker = Broker.__new__(Broker)
        broker.numberofservers = 2
        broker.numberofservices = 2
        broker.cache_round = 10
        broker.last_cache_change_round = {0: 0, 1: 0}
        broker.cache_information_regime = (
            "causal_history_only_v1"
        )
        broker.cache_history_windows = 3
        broker.cache_expected_requests_ema = 7.5
        broker.cache_mean_cpu_cycles_ema = 100e6
        broker.cache_global_execution_latency_ema = 1.5
        broker.cache_global_compute_per_mcycle_ema = 0.01
        broker.cache_global_waiting_latency_ema = 0.5
        broker.cache_server_execution_latency_ema = {
            0: 2.0,
            1: 1.0,
        }
        broker.cache_server_compute_per_mcycle_ema = {
            0: 0.02,
            1: 0.01,
        }
        broker.cache_server_waiting_latency_ema = {
            0: 0.5,
            1: 0.5,
        }
        broker.cache_server_sample_counts = {0: 5, 1: 5}
        broker.cache_server_last_observed_window = {0: 3, 1: 3}
        broker.last_server_telemetry_context = None
        broker.last_cache_decision_context = None
        broker.H = {
            0: {1: 1.0, 2: 0.5},
            1: {1: 0.5, 2: 1.0},
        }
        broker.simulator = SimpleNamespace(
            cache_policy="critical_path_joint",
            cache_update_interval=5,
            cache_min_residence_updates=0,
            cache_compute_weight=1.0,
            alg="causal_task_serverPD3QN",
            telemetry_min_samples=5,
            telemetry_freshness_half_life=10.0,
            cache_hysteresis_factor=0.0,
            max_cpu_cycles=200.0,
            input_dict={"server capacity": 1},
            service_data_length={1: 1.0, 2: 1.0},
            between_server_costs=np.asarray(
                [[0.0, 0.1], [0.1, 0.0]]
            ),
            servers={
                0: PrivilegedStateGuardServer(
                    services=[0, 1],
                    rate_to_cloud=2.0,
                ),
                1: PrivilegedStateGuardServer(
                    services=[0, 2],
                    rate_to_cloud=2.0,
                ),
            },
        )

        decision = broker.coordinated_caching_decisions()

        self.assertEqual(set(decision), {0, 1})
        self.assertEqual(
            broker.last_cache_decision_context[
                "information_regime"
            ],
            "causal_history_only_v1",
        )
        self.assertEqual(
            broker.last_cache_decision_context[
                "expected_requests"
            ],
            7.5,
        )
        self.assertEqual(
            broker.last_cache_decision_context[
                "mean_cpu_cycles_ema"
            ],
            100e6,
        )
        self.assertEqual(
            broker.last_cache_decision_context[
                "compute_intensity"
            ],
            0.5,
        )

    def test_locality_bonus_prefers_predecessor_server(self):
        predecessor = SimpleNamespace(
            assigned_server=1,
            outputs_length={"2": 100.0},
            result=SimpleNamespace(finish_time=0.5),
        )
        task = SimpleNamespace(
            task_number="2",
            predecessors=["1"],
        )
        bonuses = dependency_locality_bonuses(
            task=task,
            done_tasks={"1": predecessor},
            number_of_servers=2,
            between_server_costs=np.asarray(
                [[0.0, 0.01], [0.01, 0.0]]
            ),
        )
        self.assertGreater(bonuses[1], bonuses[0])

    def test_fetch_delay_excludes_the_candidate_itself(self):
        servers = {
            0: SimpleNamespace(
                services=[0, 1],
                rate_to_cloud=10.0,
            ),
            1: SimpleNamespace(
                services=[0],
                rate_to_cloud=10.0,
            ),
        }
        delays = alternative_service_fetch_delays(
            service=1,
            service_size=10.0,
            servers=servers,
            between_server_costs=np.asarray(
                [[0.0, 0.02], [0.02, 0.0]]
            ),
        )
        self.assertAlmostEqual(delays[0], 1.0)
        self.assertAlmostEqual(delays[1], 0.2)

    def test_hysteresis_rejects_unprofitable_swap(self):
        decision = hysteretic_cache_decision(
            scores={1: 0.5, 2: 0.6},
            current_services=[1],
            capacity=1,
            switching_costs={2: 2.0},
            expected_requests=10,
            hysteresis_factor=1.0,
        )
        self.assertEqual(decision, [1])

    def test_hysteresis_accepts_profitable_swap(self):
        decision = hysteretic_cache_decision(
            scores={1: 0.1, 2: 0.6},
            current_services=[1],
            capacity=1,
            switching_costs={2: 2.0},
            expected_requests=10,
            hysteresis_factor=1.0,
        )
        self.assertEqual(decision, [2])

    def test_fetch_objective_rewards_local_replica(self):
        savings = service_fetch_savings(
            assignments={0: [1], 1: []},
            demand={0: {1: 2.0}, 1: {1: 0.0}},
            service_sizes={1: 3.0},
            cloud_costs={0: 0.5, 1: 0.5},
            between_server_costs=np.asarray(
                [[0.0, 0.1], [0.1, 0.0]]
            ),
        )
        self.assertAlmostEqual(savings, 3.0)

    def test_coordinated_cache_breaks_redundant_replica(self):
        decision = coordinated_cache_decision(
            demand={
                0: {1: 1.0, 2: 0.0},
                1: {1: 0.0, 2: 1.0},
            },
            current_services={0: [1], 1: [1]},
            capacity=1,
            service_sizes={1: 1.0, 2: 1.0},
            cloud_costs={0: 10.0, 1: 10.0},
            between_server_costs=np.asarray(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            expected_requests=10,
            hysteresis_factor=1.0,
        )
        self.assertEqual(decision, {0: [1], 1: [2]})

    def test_replica_diversity_discount_preserves_first_replica(self):
        arguments = {
            "demand": {
                0: {1: 1.0, 2: 0.55},
                1: {1: 1.0, 2: 0.55},
            },
            "current_services": {0: [], 1: []},
            "capacity": {0: 1, 1: 1},
            "service_sizes": {1: 1.0, 2: 1.0},
            "cloud_costs": {0: 1.0, 1: 1.0},
            "between_server_costs": np.asarray(
                [[0.0, 0.9], [0.9, 0.0]]
            ),
            "expected_requests": 1.0,
            "hysteresis_factor": 0.0,
        }
        without_discount = coordinated_cache_decision(**arguments)
        with_discount = coordinated_cache_decision(
            **arguments,
            replica_diversity_regularization=True,
        )

        self.assertEqual(
            len(
                {
                    service
                    for services in without_discount.values()
                    for service in services
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    service
                    for services in with_discount.values()
                    for service in services
                }
            ),
            2,
        )

    def test_coverage_constraint_repairs_only_redundant_replica(self):
        arguments = {
            "demand": {
                0: {1: 100.0, 2: 5.0, 3: 1.0},
                1: {1: 100.0, 2: 5.0, 3: 1.0},
            },
            "current_services": {0: [1, 2], 1: [1]},
            "capacity": {0: 2, 1: 1},
            "service_sizes": {1: 1.0, 2: 1.0, 3: 1.0},
            "cloud_costs": {0: 1.0, 1: 1.0},
            "between_server_costs": np.asarray(
                [[0.0, 0.5], [0.5, 0.0]]
            ),
            "expected_requests": 1.0,
            "hysteresis_factor": 0.0,
            "server_quality": {0: 1.0, 1: 1.0},
            "replica_diversity_regularization": True,
        }
        original = coordinated_cache_decision(**arguments)
        repaired = coordinated_cache_decision(
            **arguments,
            coverage_constraint=True,
        )
        original_coverage = {
            service
            for services in original.values()
            for service in services
        }
        repaired_coverage = {
            service
            for services in repaired.values()
            for service in services
        }
        self.assertLess(len(original_coverage), 3)
        self.assertEqual(repaired_coverage, {1, 2, 3})
        self.assertEqual(sum(map(len, repaired.values())), 3)

    def test_coordinated_cache_respects_heterogeneous_capacities(self):
        decision = coordinated_cache_decision(
            demand={
                0: {1: 1.0, 2: 0.0, 3: 0.0},
                1: {1: 0.0, 2: 1.0, 3: 1.0},
            },
            current_services={0: [1], 1: [1, 2]},
            capacity={0: 0, 1: 2},
            service_sizes={1: 1.0, 2: 1.0, 3: 1.0},
            cloud_costs={0: 10.0, 1: 10.0},
            between_server_costs=np.asarray(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            expected_requests=10,
            hysteresis_factor=0.0,
        )
        self.assertEqual(decision[0], [])
        self.assertEqual(len(decision[1]), 2)

    def test_coordinated_hysteresis_rejects_expensive_change(self):
        decision = coordinated_cache_decision(
            demand={
                0: {1: 1.0, 2: 0.0},
                1: {1: 0.0, 2: 1.0},
            },
            current_services={0: [1], 1: [1]},
            capacity=1,
            service_sizes={1: 1.0, 2: 1.0},
            cloud_costs={0: 10.0, 1: 10.0},
            between_server_costs=np.asarray(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            expected_requests=1,
            hysteresis_factor=2.0,
        )
        self.assertEqual(decision, {0: [1], 1: [1]})

    def test_coordinated_cache_preserves_locked_server(self):
        decision = coordinated_cache_decision(
            demand={
                0: {1: 1.0, 2: 0.0},
                1: {1: 0.0, 2: 1.0},
            },
            current_services={0: [2], 1: [1]},
            capacity=1,
            service_sizes={1: 1.0, 2: 1.0},
            cloud_costs={0: 10.0, 1: 10.0},
            between_server_costs=np.asarray(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            expected_requests=10,
            hysteresis_factor=0.0,
            locked_servers={0},
        )
        self.assertEqual(decision[0], [2])

    def test_compute_quality_breaks_symmetric_placement_tie(self):
        decision = coordinated_cache_decision(
            demand={
                0: {1: 1.0, 2: 1.0},
                1: {1: 1.0, 2: 1.0},
            },
            current_services={0: [], 1: []},
            capacity=1,
            service_sizes={1: 1.0, 2: 1.0},
            cloud_costs={0: 10.0, 1: 10.0},
            between_server_costs=np.asarray(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            expected_requests=10,
            hysteresis_factor=0.0,
            server_quality={0: 0.1, 1: 1.0},
        )
        self.assertEqual(decision, {0: [2], 1: [1]})


if __name__ == "__main__":
    unittest.main()
