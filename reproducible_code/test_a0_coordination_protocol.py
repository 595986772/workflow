import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from a0_coordination_protocol import (
    CAPACITY_PROFILES,
    EXPECTED_DATASET_SHA256,
    validate_protocol,
)
from input import INPUT_DICT, learning_arg
from simulator import MEC_Simulator
from evaluate_a0_nhpp_stream import (
    CALIBRATION_REPLICATES,
    bottleneck_rate_calibration,
    build_stream_bank,
    calibration_events,
    recovery_delay,
    routing_upper_bounds_from_probes,
)


class A0CoordinationProtocolTest(unittest.TestCase):
    def test_frozen_a0_protocol(self):
        protocol = validate_protocol()
        self.assertEqual(
            protocol["dataset"]["sha256"],
            EXPECTED_DATASET_SHA256,
        )
        self.assertFalse(
            protocol["dataset"]["formal_unbiased_holdout"]
        )
        self.assertEqual(sum(CAPACITY_PROFILES["B8"]), 8)
        self.assertEqual(sum(CAPACITY_PROFILES["B10"]), 10)
        self.assertEqual(sum(CAPACITY_PROFILES["B5"]), 5)

    def test_dynamic_arrivals_create_response_times_and_queue_wait(self):
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "random",
                "Number of users": 2,
                "Number of servers": 2,
                "Number of services": 2,
                "Number of tasks for each user": 10,
                "server capacity": 1,
                "baseline server capacity": 1,
                "application arrival times": [0.0, 0.01],
                "dynamic queueing enabled": True,
                "periodic cache updates": False,
                "caching decision enabled": False,
                "save topology figure": False,
                "seed": 919,
            }
        )
        simulator = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png=".",
        )
        simulator.set_training(False, update_caching=False)
        simulator.run()
        for user in simulator.users.values():
            self.assertGreaterEqual(
                user.finish_time_of_application,
                user.arrival_time,
            )
        self.assertTrue(
            all(
                task.completion_event is not None
                for user in simulator.users.values()
                for task in user.done_tasks.values()
            )
        )
        self.assertGreaterEqual(
            sum(
                task.queue_waiting_latency
                for user in simulator.users.values()
                for task in user.done_tasks.values()
            ),
            0.0,
        )

    def test_arrival_vector_must_match_dag_instances(self):
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "random",
                "Number of users": 2,
                "application arrival times": [0.0],
                "save topology figure": False,
            }
        )
        with self.assertRaises(ValueError):
            MEC_Simulator(
                outputfile=io.StringIO(),
                Input_dict=config,
                learning_arguments=copy.deepcopy(learning_arg),
                filename_png=".",
            )

    def test_nhpp_bank_has_target_utilization_and_finite_burst(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            scenario = {
                "servers": {
                    str(server_id): {
                        "position": [float(server_id), 0.0],
                        "frequency": 10e9,
                    }
                    for server_id in range(10)
                },
                "users": {
                    str(user_id): {
                        "initial_position": [float(user_id % 10), 0.0]
                    }
                    for user_id in range(20)
                },
            }
            config = {
                "input_config": {
                    "dag dataset path": str(
                        Path(__file__).resolve().parent
                        / "datasets"
                        / "alibaba_cp100"
                        / "dag_alibaba_cp100_a0.json"
                    ),
                    "max_cpu_cycles": 200,
                }
            }
            (run_dir / "scenario_initial.json").write_text(
                json.dumps(scenario),
                encoding="utf-8",
            )
            (run_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            bank = build_stream_bank(run_dir, seed=3)
        self.assertAlmostEqual(
            bank["estimated_baseline_utilization"],
            0.45,
        )
        self.assertEqual(len(bank["window_arrival_counts"]), 100)
        self.assertGreater(
            sum(bank["window_arrival_counts"][40:60]),
            sum(bank["window_arrival_counts"][:20]),
        )
        for event in bank["events"]:
            lower = (event["window"] - 1) * bank["window_duration"]
            upper = event["window"] * bank["window_duration"]
            self.assertGreaterEqual(event["arrival_time"], lower)
            self.assertLess(event["arrival_time"], upper)

    def test_rate_calibration_uses_frozen_routing_bottleneck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = {}
            for label, concentration in (
                ("balanced", 10),
                ("concentrated", 90),
            ):
                run_dir = root / label
                run_dir.mkdir()
                run_dirs[label] = run_dir
                path = run_dir / "episodes.csv"
                with path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(
                        output,
                        fieldnames=(
                            "phase",
                            "server_action_histogram_json",
                        ),
                    )
                    writer.writeheader()
                    histogram = {
                        str(server_id): (
                            concentration
                            if server_id == 0
                            else (100 - concentration) / 9
                        )
                        for server_id in range(10)
                    }
                    for _ in range(100):
                        writer.writerow(
                            {
                                "phase": "eval",
                                "server_action_histogram_json": json.dumps(
                                    histogram
                                ),
                            }
                        )
            calibration = bottleneck_rate_calibration(
                run_dirs=run_dirs,
                frequencies=[10e9] * 10,
                mean_cycles=100e6,
            )
        self.assertAlmostEqual(
            calibration["worst_bottleneck_utilization"],
            0.45,
        )
        self.assertAlmostEqual(calibration["lambda0"], 50.0)
        self.assertLessEqual(
            calibration["method_bottleneck_utilization"]["balanced"],
            0.45,
        )

    def test_rate_calibration_prefers_cpu_work_over_action_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "weighted"
            run_dir.mkdir()
            with (run_dir / "episodes.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "phase",
                        "server_action_histogram_json",
                        "server_cpu_cycle_histogram_json",
                    ),
                )
                writer.writeheader()
                action_histogram = {
                    str(server_id): 10 for server_id in range(10)
                }
                cpu_histogram = {
                    str(server_id): (
                        90e6 if server_id == 0 else 10e6 / 9
                    )
                    for server_id in range(10)
                }
                for _ in range(100):
                    writer.writerow(
                        {
                            "phase": "eval",
                            "server_action_histogram_json": json.dumps(
                                action_histogram
                            ),
                            "server_cpu_cycle_histogram_json": json.dumps(
                                cpu_histogram
                            ),
                        }
                    )
            calibration = bottleneck_rate_calibration(
                run_dirs={"weighted": run_dir},
                frequencies=[10e9] * 10,
                mean_cycles=100e6,
            )
        self.assertAlmostEqual(calibration["lambda0"], 50.0)
        self.assertEqual(
            calibration["calibration_histogram_fields"]["weighted"],
            "server_cpu_cycle_histogram_json",
        )

    def test_rate_calibration_accepts_independent_online_profiles(self):
        balanced = [0.1] * 10
        concentrated = [0.9] + [0.1 / 9] * 9
        calibration = bottleneck_rate_calibration(
            run_dirs=None,
            frequencies=[10e9] * 10,
            mean_cycles=100e6,
            routing_profiles={
                "balanced": balanced,
                "concentrated": concentrated,
            },
        )
        self.assertAlmostEqual(calibration["lambda0"], 50.0)
        self.assertEqual(
            calibration["calibration_source"],
            "independent_online_cache_pilot_cpu_work",
        )
        self.assertAlmostEqual(
            calibration["method_bottleneck_utilization"]["concentrated"],
            0.45,
        )
        self.assertLess(
            calibration["method_bottleneck_utilization"]["balanced"],
            0.45,
        )

    def test_rate_calibration_accepts_non_normalized_upper_bounds(self):
        calibration = bottleneck_rate_calibration(
            run_dirs=None,
            frequencies=[10e9] * 10,
            mean_cycles=100e6,
            routing_upper_bounds={
                "multimodal": [0.6, 0.6] + [0.0] * 8,
            },
        )
        self.assertAlmostEqual(calibration["lambda0"], 75.0)
        self.assertTrue(calibration["routing_values_are_upper_bounds"])
        self.assertEqual(
            calibration["calibration_source"],
            (
                "independent_multi_stream_online_cache_"
                "cpu_work_upper_bound"
            ),
        )

    def test_routing_upper_bound_preserves_rare_server_modes(self):
        probes = []
        for replicate in range(CALIBRATION_REPLICATES):
            fractions = {str(server_id): 0.0 for server_id in range(10)}
            fractions[str(replicate)] = 1.0
            probes.append({"routing_fractions": fractions})
        upper = routing_upper_bounds_from_probes(
            {"method": probes},
            number_of_servers=10,
        )["method"]
        self.assertEqual(upper[:CALIBRATION_REPLICATES], [1.0] * 5)
        self.assertEqual(upper[CALIBRATION_REPLICATES:], [0.1] * 5)
        self.assertGreater(sum(upper), 1.0)

    def test_calibration_stream_is_deterministic_and_window_bounded(self):
        first = calibration_events(seed=7)
        second = calibration_events(seed=7)
        other = calibration_events(seed=8)
        other_replicate = calibration_events(seed=7, replicate=1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, other_replicate)
        self.assertEqual(len(first), 800)
        self.assertEqual({event["window"] for event in first}, set(range(1, 41)))
        for window in range(1, 41):
            events = [event for event in first if event["window"] == window]
            self.assertEqual(len(events), 20)
            self.assertTrue(
                all(
                    (window - 1) * 4000.0
                    <= event["arrival_time"]
                    < window * 4000.0
                    for event in events
                )
            )

    def test_recovery_uses_pre_burst_baseline(self):
        rows = []
        for window in range(1, 101):
            value = 1.0 if window <= 40 or window >= 66 else 2.0
            rows.append(
                {
                    "window": window,
                    "migration_adjusted_mean_finish_time": value,
                }
            )
        recovery = recovery_delay(rows)
        self.assertAlmostEqual(recovery["baseline_mean"], 1.0)
        self.assertEqual(recovery["delay_windows"], 5)


if __name__ == "__main__":
    unittest.main()
