import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import networkx as nx
import numpy as np
import torch

from input import INPUT_DICT, learning_arg
from simulator import MEC_Simulator


def _encoded_service(service_id, service_count=1):
    return 1.0 + (service_id - 0.5) / service_count


def _dataset(edges):
    graph = nx.DiGraph()
    graph.add_node("0", service=0, cpucycle=0)
    real_nodes = sorted(
        {node for edge in edges for node in edge if node != "0"},
        key=int,
    )
    for node_id in real_nodes:
        graph.add_node(
            node_id,
            service=_encoded_service(1),
            cpucycle=0.1 + 0.05 * int(node_id),
        )
    for source, target in edges:
        graph.add_edge(source, target, datalength=0.0)
    return {"completion_case": nx.node_link_data(graph)}


def _run_dataset(dataset, arrival_time=0.0):
    with tempfile.TemporaryDirectory() as temporary:
        dataset_path = Path(temporary) / "dag.json"
        dataset_path.write_text(
            json.dumps(dataset, sort_keys=True),
            encoding="utf-8",
        )
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "nearest_server",
                "Number of users": 1,
                "Number of servers": 1,
                "Number of services": 1,
                "Number of tasks for each user": 4,
                "server capacity": 1,
                "baseline server capacity": 1,
                "dag dataset path": str(dataset_path),
                "dag dataset sha256": None,
                "Bandwidth": 15000,
                "update deadline": False,
                "caching decision enabled": False,
                "save topology figure": False,
                "application arrival times": [float(arrival_time)],
                "seed": 707,
            }
        )
        random.seed(707)
        np.random.seed(707)
        torch.manual_seed(707)
        simulator = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png=temporary,
        )
        simulator.set_training(False, update_caching=False)
        simulator.run()
        return simulator.users[0]


class DagCompletionSemanticsTest(unittest.TestCase):
    def assert_every_task_completed_once(self, user):
        self.assertEqual(set(user.done_tasks), set(user.tasks_init))
        self.assertEqual(
            user.episode_decision_count,
            len(user.tasks_init),
        )
        self.assertEqual(
            user.task_completion_counts,
            {task_id: 1 for task_id in user.tasks_init},
        )

    def test_single_exit_keeps_legacy_finish_time(self):
        user = _run_dataset(
            _dataset((("0", "1"), ("1", "2"), ("2", "3")))
        )

        self.assert_every_task_completed_once(user)
        self.assertEqual(user.exit_task_ids, ("3",))
        self.assertEqual(
            user.finish_time_of_application,
            user.done_tasks["3"].result.finish_time,
        )

    def test_multiple_exits_waits_for_all_real_tasks(self):
        user = _run_dataset(
            _dataset((("0", "1"), ("1", "2"), ("1", "3")))
        )

        self.assert_every_task_completed_once(user)
        self.assertEqual(set(user.exit_task_ids), {"2", "3"})
        exit_finish_times = [
            user.done_tasks[task_id].result.finish_time
            for task_id in user.exit_task_ids
        ]
        self.assertEqual(
            user.finish_time_of_application,
            max(exit_finish_times),
        )
        self.assertTrue(user.complete)

    def test_nonzero_arrival_shifts_absolute_but_not_response_time(self):
        dataset = _dataset((('0', '1'), ('1', '2'), ('1', '3')))
        base_user = _run_dataset(dataset, arrival_time=0.0)
        shifted_user = _run_dataset(dataset, arrival_time=4.0)

        self.assert_every_task_completed_once(shifted_user)
        self.assertAlmostEqual(
            shifted_user.finish_time_of_application,
            base_user.finish_time_of_application + 4.0,
        )
        self.assertAlmostEqual(
            shifted_user.finish_time_of_application
            - shifted_user.arrival_time,
            base_user.finish_time_of_application,
        )


if __name__ == "__main__":
    unittest.main()
