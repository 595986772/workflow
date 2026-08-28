import copy
import io
import random
import unittest

import networkx as nx
import numpy as np
import simpy

from analyze_strict_environment_suite import paired_statistics
from input import INPUT_DICT, learning_arg
from oracle_latency_bound import scenario_relaxed_oracle_bounds
from run_independent_experiment import (
    base_scenario_fingerprint,
    scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
)
from server import Server
from simulator import MEC_Simulator
from strict_environment import (
    apply_strict_dag_stress,
    dag_task_depth,
)


def example_graph():
    graph = nx.DiGraph()
    graph.add_node("0", service=0, cpucycle=0.0)
    for task_id in ("1", "2", "3", "4"):
        graph.add_node(task_id, service=1.2, cpucycle=0.5)
    graph.add_edge("0", "1", datalength=0.5)
    graph.add_edge("0", "3", datalength=0.6)
    graph.add_edge("1", "2", datalength=0.2)
    graph.add_edge("3", "4", datalength=0.3)
    return graph


def simulator_for(
    seed,
    server_capacity=2,
    dag_depth_increment=0,
    dependency_data_scale=1.0,
):
    seed_everything(seed)
    config = copy.deepcopy(INPUT_DICT)
    config.update(
        {
            "alg": "prev_servers_plus_service_per_serverDQN",
            "Number of users": 3,
            "Number of servers": 3,
            "Number of services": 3,
            "Number of tasks for each user": 6,
            "server capacity": server_capacity,
            "baseline server capacity": 2,
            "dag depth increment": dag_depth_increment,
            "dependency data scale": dependency_data_scale,
            "Bandwidth": 15000.0,
            "save topology figure": False,
        }
    )
    return MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=config,
        learning_arguments=copy.deepcopy(learning_arg),
        filename_png="/tmp",
    )


class StrictDAGStressTests(unittest.TestCase):
    def test_default_transform_is_identical(self):
        graph = example_graph()
        stressed, metadata = apply_strict_dag_stress(graph)
        self.assertEqual(
            nx.node_link_data(stressed),
            nx.node_link_data(graph),
        )
        self.assertEqual(metadata.base_depth, 2)
        self.assertEqual(metadata.stressed_depth, 2)
        self.assertEqual(metadata.added_control_edges, ())

    def test_depth_increment_only_adds_zero_data_dependencies(self):
        graph = example_graph()
        original_edges = {
            (source, target): dict(attributes)
            for source, target, attributes
            in graph.edges(data=True)
        }
        stressed, metadata = apply_strict_dag_stress(
            graph,
            depth_increment=1,
        )
        self.assertEqual(metadata.base_depth, 2)
        self.assertEqual(metadata.target_depth, 3)
        self.assertEqual(metadata.stressed_depth, 3)
        self.assertGreater(len(metadata.added_control_edges), 0)
        self.assertTrue(nx.is_directed_acyclic_graph(stressed))
        for edge, attributes in original_edges.items():
            self.assertEqual(stressed.edges[edge], attributes)
        for edge in metadata.added_control_edges:
            self.assertEqual(stressed.edges[edge]["datalength"], 0.0)

    def test_data_scale_leaves_root_uploads_unchanged(self):
        graph = example_graph()
        stressed, metadata = apply_strict_dag_stress(
            graph,
            dependency_data_scale=4.0,
        )
        self.assertEqual(
            stressed.edges["0", "1"]["datalength"],
            graph.edges["0", "1"]["datalength"],
        )
        self.assertEqual(
            stressed.edges["1", "2"]["datalength"],
            4.0 * graph.edges["1", "2"]["datalength"],
        )
        self.assertAlmostEqual(
            metadata.stressed_dependency_data,
            4.0 * metadata.base_dependency_data,
        )

    def test_invalid_relaxation_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_strict_dag_stress(
                example_graph(),
                dependency_data_scale=0.5,
            )


class StrictScenarioPairingTests(unittest.TestCase):
    def test_cache_stress_preserves_random_stream(self):
        def construct(capacity):
            random.seed(9)
            np.random.seed(9)
            server = Server(
                simpy.Environment(),
                id=0,
                numberofservices=5,
                xlim=1000,
                ylim=1000,
                min_freq=0.2,
                max_freq=30,
                iscloud=False,
                minload=0.2,
                maxload=100,
                minratetocloud=1,
                maxratetocloud=3,
                capacity=capacity,
                random_draw_capacity=2,
            )
            return (
                server,
                random.random(),
                float(np.random.random()),
            )

        baseline, baseline_random, baseline_numpy = construct(2)
        stressed, stressed_random, stressed_numpy = construct(1)
        self.assertEqual(
            stressed.services[1:],
            baseline.services[1:2],
        )
        self.assertEqual(stressed.pos, baseline.pos)
        self.assertEqual(stressed.frequency, baseline.frequency)
        self.assertEqual(stressed.load, baseline.load)
        self.assertEqual(stressed_random, baseline_random)
        self.assertEqual(stressed_numpy, baseline_numpy)

    def test_combined_stress_has_same_base_scenario(self):
        baseline = simulator_for(seed=4, server_capacity=2)
        stressed = simulator_for(
            seed=4,
            server_capacity=1,
            dag_depth_increment=2,
            dependency_data_scale=2.0,
        )
        self.assertEqual(
            base_scenario_fingerprint(baseline),
            base_scenario_fingerprint(stressed),
        )
        self.assertNotEqual(
            scenario_fingerprint(scenario_snapshot(baseline)),
            scenario_fingerprint(scenario_snapshot(stressed)),
        )
        baseline_floor = scenario_relaxed_oracle_bounds(
            baseline
        )
        stressed_floor = scenario_relaxed_oracle_bounds(
            stressed
        )
        self.assertTrue(
            all(
                stressed_value >= baseline_value - 1e-12
                for baseline_value, stressed_value in zip(
                    baseline_floor["per_user"],
                    stressed_floor["per_user"],
                )
            )
        )

    def test_e0_fingerprint_regression(self):
        seed_everything(1)
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "prev_servers_plus_service_per_serverDQN",
                "Number of users": 20,
                "Number of servers": 10,
                "Number of services": 10,
                "Number of tasks for each user": 10,
                "Bandwidth": 15000.0,
                "save topology figure": False,
            }
        )
        simulator = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png="/tmp",
        )
        self.assertEqual(
            scenario_fingerprint(scenario_snapshot(simulator)),
            "1094a15dd2de4bad242e35ef09dc965567210da6cf8335ac814d8708abdc2d30",
        )

    def test_depth_metadata_matches_realized_graph(self):
        simulator = simulator_for(seed=5)
        metadata = simulator.user_graph_stress[0]
        self.assertEqual(
            metadata.stressed_depth,
            dag_task_depth(simulator.users[0].DAG2),
        )


class StrictPairedStatisticsTests(unittest.TestCase):
    def test_lower_metric_counts_seed_paired_wins(self):
        result = paired_statistics(
            [2.0, 4.0, 8.0],
            [1.0, 3.0, 7.0],
            lower_is_better=True,
        )
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["wins"], 3)
        self.assertGreater(
            result["mean_paired_improvement_percent"],
            0.0,
        )

    def test_higher_metric_uses_correct_direction(self):
        result = paired_statistics(
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            lower_is_better=False,
        )
        self.assertEqual(result["wins"], 3)
        self.assertGreater(
            result["mean_paired_improvement_percent"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
