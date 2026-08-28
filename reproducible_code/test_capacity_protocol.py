import copy
import io
import unittest

from capacity_protocol import (
    deterministic_capacity_assignment,
    normalize_capacity_mapping,
    select_load_shift_servers,
)
from input import INPUT_DICT, learning_arg
from run_independent_experiment import (
    base_scenario_fingerprint,
    scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
)
from simulator import MEC_Simulator


E2_CAPACITIES = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def simulator_for(seed, heterogeneous):
    seed_everything(seed)
    config = copy.deepcopy(INPUT_DICT)
    config.update(
        {
            "alg": "nearest_with_service",
            "Number of users": 3,
            "Number of servers": 10,
            "Number of services": 10,
            "Number of tasks for each user": 6,
            "server capacity": 1,
            "server capacity multiset": (
                E2_CAPACITIES if heterogeneous else None
            ),
            "baseline server capacity": 2,
            "save topology figure": False,
        }
    )
    return MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=config,
        learning_arguments=copy.deepcopy(learning_arg),
        filename_png="/tmp",
    )


class CapacityProtocolTest(unittest.TestCase):
    def test_assignment_is_stable_and_preserves_multiset(self):
        first = deterministic_capacity_assignment(
            E2_CAPACITIES,
            number_of_servers=10,
            number_of_services=10,
            seed=7,
        )
        second = deterministic_capacity_assignment(
            E2_CAPACITIES,
            number_of_servers=10,
            number_of_services=10,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(sorted(first.values()), E2_CAPACITIES)
        self.assertEqual(sum(first.values()), 10)

    def test_assignment_does_not_change_physical_random_stream(self):
        scalar = simulator_for(seed=8, heterogeneous=False)
        heterogeneous = simulator_for(seed=8, heterogeneous=True)
        self.assertEqual(
            base_scenario_fingerprint(scalar),
            base_scenario_fingerprint(heterogeneous),
        )
        self.assertNotEqual(
            scenario_fingerprint(scenario_snapshot(scalar)),
            scenario_fingerprint(scenario_snapshot(heterogeneous)),
        )

    def test_heterogeneous_fingerprint_covers_full_task_workload(self):
        simulator = simulator_for(seed=8, heterogeneous=True)
        original = scenario_fingerprint(
            scenario_snapshot(simulator)
        )
        base = base_scenario_fingerprint(simulator)
        task = next(
            iter(next(iter(simulator.users.values())).tasks_init.values())
        )
        task.cpu_cycle += 1.0
        self.assertNotEqual(
            scenario_fingerprint(scenario_snapshot(simulator)),
            original,
        )
        self.assertEqual(
            base_scenario_fingerprint(simulator),
            base,
        )

    def test_zero_capacity_server_computes_without_real_cache(self):
        simulator = simulator_for(seed=4, heterogeneous=True)
        zero_servers = [
            server
            for server in simulator.servers.values()
            if server.capacity == 0
        ]
        self.assertEqual(len(zero_servers), 3)
        self.assertTrue(
            all(server.services == [0] for server in zero_servers)
        )
        self.assertTrue(
            all(server.frequency > 0 for server in zero_servers)
        )

    def test_load_shift_selects_one_server_per_capacity_class(self):
        capacities = deterministic_capacity_assignment(
            E2_CAPACITIES,
            number_of_servers=10,
            number_of_services=10,
            seed=9,
        )
        selected = select_load_shift_servers(capacities, seed=9)
        self.assertEqual(set(selected), {0, 1, 2})
        self.assertEqual(len(set(selected.values())), 3)
        for capacity, server_id in selected.items():
            self.assertEqual(capacities[server_id], capacity)

    def test_capacity_mapping_supports_scalar_and_vector(self):
        self.assertEqual(
            normalize_capacity_mapping(2, [0, 1]),
            {0: 2, 1: 2},
        )
        self.assertEqual(
            normalize_capacity_mapping([0, 2], [0, 1]),
            {0: 0, 1: 2},
        )


if __name__ == "__main__":
    unittest.main()
