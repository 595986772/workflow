import unittest

from broker import Broker
from run_reproduction_suite import (
    ALGORITHMS,
    LEAN_OUR_LABEL,
    OUR_DQN_LABEL,
)


def algorithm(label):
    return next(
        config for config in ALGORITHMS
        if config["label"] == label
    )


class AlibabaPaperExtensionsTest(unittest.TestCase):
    def test_our_dqn_changes_only_q_architecture(self):
        ours = algorithm(LEAN_OUR_LABEL)
        dqn = algorithm(OUR_DQN_LABEL)
        self.assertEqual(dqn["algorithm"], "causal_telemetryDDQN")
        self.assertEqual(ours["algorithm"], "causal_telemetryPD3QN")
        for key in (
            "beta",
            "beta_min",
            "beta_decay",
            "reward_mode",
            "potential_reward_weight",
            "cache_policy",
            "gamma",
            "n_step",
        ):
            self.assertEqual(dqn[key], ours[key])

    def test_centralized_baseline_keeps_daoc_dqn(self):
        daoc = algorithm("guided_full")
        centralized = algorithm("centralized_greedy_daoc")
        self.assertEqual(
            centralized["algorithm"],
            daoc["algorithm"],
        )
        self.assertEqual(centralized["beta"], daoc["beta"])
        self.assertEqual(
            centralized["reward_mode"],
            "terminal_binary",
        )
        self.assertEqual(
            centralized["cache_policy"],
            "popularity_coordinated",
        )

    def test_centralized_popularity_is_unweighted_and_local(self):
        broker = Broker.__new__(Broker)
        broker.cache_window_values = {
            0: {1: 0.0, 2: 0.0},
            1: {1: 0.0, 2: 0.0},
        }
        broker.cache_window_observations = 0
        task = type(
            "TaskStub",
            (),
            {
                "assigned_server": 1,
                "service": 2,
            },
        )()

        broker._coordinated_popularity_update(task)

        self.assertEqual(
            broker.cache_window_values,
            {
                0: {1: 0.0, 2: 0.0},
                1: {1: 0.0, 2: 1.0},
            },
        )
        self.assertEqual(broker.cache_window_observations, 1)

    def test_our_cache_demand_remains_criticality_weighted(self):
        broker = Broker.__new__(Broker)
        broker.numberofservers = 2
        broker.cache_window_values = {
            0: {1: 0.0, 2: 0.0},
            1: {1: 0.0, 2: 0.0},
        }
        broker.cache_window_observations = 0
        broker.simulator = type(
            "SimulatorStub",
            (),
            {
                "cache_locality_weight": 1.0,
                "max_data_length": 100.0,
                "users": {
                    0: type(
                        "UserStub",
                        (),
                        {
                            "nominal_upward_ranks": {0: 0.25},
                            "maximum_nominal_rank": 1.0,
                            "done_tasks": {},
                        },
                    )()
                },
            },
        )()
        task = type(
            "TaskStub",
            (),
            {
                "user_id": 0,
                "task_number": 0,
                "assigned_server": 1,
                "service": 2,
                "predecessors": [],
            },
        )()

        broker._coordinated_cache_update(task)

        self.assertEqual(
            broker.cache_window_values[1][2],
            0.25,
        )
        self.assertEqual(broker.cache_window_observations, 1)


if __name__ == "__main__":
    unittest.main()
