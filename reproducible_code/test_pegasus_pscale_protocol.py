import copy
from collections import Counter
import io
import random
import unittest

from capacity_protocol import deterministic_capacity_assignment
from input import INPUT_DICT, learning_arg
from pegasus_pscale_protocol import (
    CACHE_CALIBRATION_EPISODES,
    CAPACITY_PROFILES,
    DATASET_PATH,
    EVALUATION_BANK_SCOPE,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    VALIDATION_SCENARIOS,
    evaluation_family,
    validate_protocol,
)
from run_independent_experiment import (
    apply_deployment_state,
    base_scenario_fingerprint,
    capture_deployment_state,
)
from run_reproduction_suite import LEAN_OUR_LABEL, PROFILES
from simulator import MEC_Simulator


class PegasusPScaleProtocolTest(unittest.TestCase):
    def test_frozen_protocol(self):
        protocol = validate_protocol()
        self.assertEqual(
            protocol["dataset_sha256"],
            EXPECTED_DATASET_SHA256,
        )
        self.assertEqual(
            protocol["evaluation_family_counts"],
            {family: 20 for family in FAMILIES},
        )
        self.assertEqual(protocol["task_limit_including_dummy"], 31)
        self.assertEqual(
            protocol["evaluation_bank_scope"],
            "infrastructure",
        )
        self.assertEqual(
            protocol["cache_calibration_episodes"],
            CACHE_CALIBRATION_EPISODES,
        )
        self.assertEqual(
            protocol["validation_scenarios"],
            VALIDATION_SCENARIOS,
        )
        self.assertEqual(EVALUATION_BANK_SCOPE, "infrastructure")

    def test_family_bank_is_exactly_balanced(self):
        counts = Counter(
            evaluation_family(index)
            for index in range(EVALUATION_EPISODES)
        )
        self.assertEqual(counts, Counter({family: 20 for family in FAMILIES}))

    def test_p2_convergence_profile_freezes_cache_before_selection(self):
        profile = PROFILES["pegasus_pscale_p2_converged"]
        self.assertEqual(profile["cache_freeze_episode"], 5000)
        self.assertEqual(profile["validation_scenarios"], 50)
        self.assertEqual(profile["eval_bank_scope"], "infrastructure")
        lean = profile["method_overrides"][LEAN_OUR_LABEL]
        self.assertEqual(lean["cache_freeze_episode"], 5000)
        self.assertEqual(lean["validation_scenarios"], 50)
        self.assertEqual(lean["convergence_min_episodes"], 5000)

    def test_simulator_can_force_one_workflow_family(self):
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "nearest_server",
                "Number of users": 2,
                "Number of servers": 10,
                "Number of services": 10,
                "Number of tasks for each user": 31,
                "server capacity": 1,
                "baseline server capacity": 3,
                "server capacity multiset": list(CAPACITY_PROFILES["B8"]),
                "capacity assignment namespace": "pegasus_pscale_p2",
                "dag dataset path": str(DATASET_PATH),
                "dag dataset sha256": EXPECTED_DATASET_SHA256,
                "application graph family": "CyberShake",
                "caching decision enabled": False,
                "save topology figure": False,
                "seed": 41,
            }
        )
        simulator = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png=".",
        )
        self.assertEqual(
            set(simulator.user_graph_keys.values()),
            {"pegasus_full_cybershake"},
        )
        self.assertTrue(
            all(user.numberoftasks == 30 for user in simulator.users.values())
        )

    def test_capacity_shuffle_does_not_consume_deployment_rng(self):
        random.seed(100)
        first_draw = random.random()
        assignment = deterministic_capacity_assignment(
            CAPACITY_PROFILES["B8"],
            number_of_servers=10,
            number_of_services=10,
            seed=41,
            assignment_namespace="pegasus_pscale_p2",
        )
        second_draw = random.random()

        random.seed(100)
        self.assertEqual(first_draw, random.random())
        self.assertEqual(second_draw, random.random())
        self.assertEqual(sum(assignment.values()), 8)
        self.assertEqual(
            sorted(assignment.values()),
            sorted(CAPACITY_PROFILES["B8"]),
        )

    def test_infrastructure_bank_keeps_servers_and_resamples_users(self):
        training = self._simulator(seed=41)
        deployment = capture_deployment_state(training)
        first = self._simulator(seed=1001)
        second = self._simulator(seed=1002)

        apply_deployment_state(first, deployment, include_users=False)
        apply_deployment_state(second, deployment, include_users=False)

        expected_servers = {
            server_id: (
                tuple(server.pos),
                server.frequency,
                server.load,
                server.capacity,
            )
            for server_id, server in training.servers.items()
        }
        for simulator in (first, second):
            self.assertEqual(
                {
                    server_id: (
                        tuple(server.pos),
                        server.frequency,
                        server.load,
                        server.capacity,
                    )
                    for server_id, server in simulator.servers.items()
                },
                expected_servers,
            )

        first_users = {
            user_id: tuple(user.pos0)
            for user_id, user in first.users.items()
        }
        second_users = {
            user_id: tuple(user.pos0)
            for user_id, user in second.users.items()
        }
        self.assertNotEqual(first_users, second_users)
        self.assertNotEqual(
            base_scenario_fingerprint(first),
            base_scenario_fingerprint(second),
        )

    @staticmethod
    def _simulator(seed):
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "nearest_server",
                "Number of users": 20,
                "Number of servers": 10,
                "Number of services": 10,
                "Number of tasks for each user": 31,
                "server capacity": 1,
                "baseline server capacity": 3,
                "server capacity multiset": list(CAPACITY_PROFILES["B8"]),
                "capacity assignment namespace": "pegasus_pscale_p2",
                "dag dataset path": str(DATASET_PATH),
                "dag dataset sha256": EXPECTED_DATASET_SHA256,
                "application graph family": "Montage",
                "caching decision enabled": False,
                "save topology figure": False,
                "seed": seed,
            }
        )
        return MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png=".",
        )


if __name__ == "__main__":
    unittest.main()
