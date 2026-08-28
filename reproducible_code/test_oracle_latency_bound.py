import unittest

from oracle_latency_bound import (
    AssignmentProblem,
    assignment_problem_from_simulator,
    exact_optimistic_assignment_oracle,
    relaxed_assignment_lower_bound,
    clairvoyant_capacity_placement,
)
from types import SimpleNamespace
import numpy as np


def chain_problem():
    task_ids = ("a", "b", "c")
    predecessors = {
        "a": (),
        "b": ("a",),
        "c": ("b",),
    }
    local_costs = {
        ("a", 0): 1.0,
        ("a", 1): 0.5,
        ("b", 0): 0.8,
        ("b", 1): 0.4,
        ("c", 0): 0.6,
        ("c", 1): 0.3,
    }
    transfer_costs = {}
    for predecessor, task_id in (("a", "b"), ("b", "c")):
        for predecessor_server in range(2):
            for server_id in range(2):
                transfer_costs[
                    (
                        predecessor,
                        task_id,
                        predecessor_server,
                        server_id,
                    )
                ] = (
                    0.0
                    if predecessor_server == server_id
                    else 0.7
                )
    return AssignmentProblem(
        task_ids=task_ids,
        predecessors=predecessors,
        local_costs=local_costs,
        transfer_costs=transfer_costs,
        sink="c",
        num_servers=2,
    )


def branching_problem():
    task_ids = ("a", "b", "c", "d")
    predecessors = {
        "a": (),
        "b": ("a",),
        "c": ("a",),
        "d": ("b", "c"),
    }
    local_costs = {
        ("a", 0): 0.1,
        ("a", 1): 0.1,
        ("b", 0): 0.1,
        ("b", 1): 10.0,
        ("c", 0): 10.0,
        ("c", 1): 0.1,
        ("d", 0): 0.1,
        ("d", 1): 100.0,
    }
    transfer_costs = {}
    for predecessor, task_id in (
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
    ):
        for predecessor_server in range(2):
            for server_id in range(2):
                transfer_costs[
                    (
                        predecessor,
                        task_id,
                        predecessor_server,
                        server_id,
                    )
                ] = (
                    0.0
                    if predecessor_server == server_id
                    else (1.0 if predecessor == "a" else 0.0)
                )
    return AssignmentProblem(
        task_ids=task_ids,
        predecessors=predecessors,
        local_costs=local_costs,
        transfer_costs=transfer_costs,
        sink="d",
        num_servers=2,
    )


class OracleLatencyBoundTest(unittest.TestCase):
    def test_relaxed_bound_is_exact_for_chain(self):
        problem = chain_problem()
        relaxed = relaxed_assignment_lower_bound(problem)
        exact = exact_optimistic_assignment_oracle(problem)
        self.assertAlmostEqual(relaxed, exact.objective, places=7)

    def test_branch_relaxation_never_exceeds_exact_oracle(self):
        problem = branching_problem()
        relaxed = relaxed_assignment_lower_bound(problem)
        exact = exact_optimistic_assignment_oracle(problem)
        self.assertLess(relaxed, exact.objective)

    def test_rejects_negative_latency(self):
        problem = chain_problem()
        problem.local_costs[("a", 0)] = -1.0
        with self.assertRaisesRegex(ValueError, "Negative local cost"):
            relaxed_assignment_lower_bound(problem)

    def test_simulator_problem_supports_multiple_sinks(self):
        tasks = {
            0: SimpleNamespace(
                predecessors=[],
                successors=[1, 2],
                input_data_length=0.0,
                cpu_cycle=1.0,
                outputs_length={1: 0.0, 2: 0.0},
                service=0,
            ),
            1: SimpleNamespace(
                predecessors=[0],
                successors=[],
                input_data_length=0.0,
                cpu_cycle=2.0,
                outputs_length={},
                service=0,
            ),
            2: SimpleNamespace(
                predecessors=[0],
                successors=[],
                input_data_length=0.0,
                cpu_cycle=4.0,
                outputs_length={},
                service=0,
            ),
        }
        simulator = SimpleNamespace(
            S=1,
            servers={
                0: SimpleNamespace(
                    frequency=1.0,
                    load=0.0,
                    rate_to_cloud=1.0,
                )
            },
            between_server_costs=np.zeros((1, 1)),
            service_data_length={},
        )
        user = SimpleNamespace(
            tasks_init=tasks,
            rate_to_gateway=1.0,
            nearest_server=0,
        )

        problem = assignment_problem_from_simulator(simulator, user)

        self.assertEqual(problem.predecessors[problem.sink], (1, 2))
        self.assertAlmostEqual(
            relaxed_assignment_lower_bound(problem),
            5.0,
        )

    def test_clairvoyant_cache_respects_mixed_capacities(self):
        simulator = SimpleNamespace(
            Q=3,
            servers={
                0: SimpleNamespace(capacity=0, rate_to_cloud=2.0),
                1: SimpleNamespace(capacity=1, rate_to_cloud=2.0),
                2: SimpleNamespace(capacity=2, rate_to_cloud=2.0),
            },
            users={
                0: SimpleNamespace(
                    nearest_server=0,
                    tasks_init={
                        0: SimpleNamespace(service=1),
                        1: SimpleNamespace(service=2),
                        2: SimpleNamespace(service=3),
                    },
                )
            },
            service_data_length={1: 1.0, 2: 1.0, 3: 1.0},
            between_server_costs=np.asarray(
                [
                    [0.0, 0.1, 0.2],
                    [0.1, 0.0, 0.1],
                    [0.2, 0.1, 0.0],
                ]
            ),
        )
        placement = clairvoyant_capacity_placement(simulator)
        self.assertEqual(placement[0], [])
        self.assertLessEqual(len(placement[1]), 1)
        self.assertLessEqual(len(placement[2]), 2)
        self.assertEqual(
            sum(len(services) for services in placement.values()),
            3,
        )


if __name__ == "__main__":
    unittest.main()
