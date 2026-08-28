import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agent import Agent
from broker import Broker
from pegasus_p6_protocol import (
    BASE_DDQN_STD_LABEL,
    FINAL_SEEDS,
    OUR_FLAT_DDQN_LABEL,
    OUR_NO_DEPENDENCY_CACHE_LABEL,
    OUR_NO_TASK_DEPENDENCY_LABEL,
    OUR_TERMINAL_REWARD_LABEL,
    P3_FINAL_DIR,
    P4_ABLATION_DIR,
    P5_SAC_DIR,
    algorithm_config,
    validate_protocol,
)
from run_reproduction_suite import PROFILES


def learner_arguments():
    return {
        "hidden_units": [8, 8],
        "gamma": 1.0,
        "max_experiences": 32,
        "min_experiences": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "epsilon": 0.01,
        "maximum_exploration": 10,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
    }


def comparable_bank(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in rows
    ]


class PegasusP6ProtocolTest(unittest.TestCase):
    def test_protocol_and_profiles_are_registered(self):
        protocol = validate_protocol()
        self.assertEqual(protocol["final_seeds"], list(FINAL_SEEDS))
        self.assertIn("pegasus_p6_smoke", PROFILES)
        self.assertIn("pegasus_p6_heuristics", PROFILES)
        self.assertIn("pegasus_p6_learning_converged", PROFILES)

    def test_flat_ddqn_factorial_is_clean(self):
        base = algorithm_config(BASE_DDQN_STD_LABEL)
        coordinated = algorithm_config(OUR_FLAT_DDQN_LABEL)
        self.assertEqual(base["algorithm"], coordinated["algorithm"])
        self.assertEqual(base["reward_mode"], coordinated["reward_mode"])
        self.assertEqual(base["gamma"], coordinated["gamma"])
        self.assertEqual(base["n_step"], coordinated["n_step"])
        self.assertEqual(base["cache_policy"], "popularity_ema")
        self.assertEqual(coordinated["cache_policy"], "critical_path_joint")
        self.assertTrue(coordinated["cache_coverage_constraint"])

    def test_mechanism_ablations_change_one_semantic_switch(self):
        ours = algorithm_config("lean_our")
        no_task = algorithm_config(OUR_NO_TASK_DEPENDENCY_LABEL)
        no_cache = algorithm_config(OUR_NO_DEPENDENCY_CACHE_LABEL)
        terminal = algorithm_config(OUR_TERMINAL_REWARD_LABEL)
        self.assertEqual(no_task["algorithm"], ours["algorithm"])
        self.assertFalse(no_task["task_dependency_features"])
        self.assertFalse(no_cache["cache_dependency_awareness"])
        self.assertEqual(terminal["reward_mode"], "terminal_binary")
        for method in (no_task, no_cache, terminal):
            self.assertEqual(method["cache_policy"], ours["cache_policy"])
            self.assertEqual(
                method["cache_coverage_constraint"],
                ours["cache_coverage_constraint"],
            )

    def test_reused_artifacts_have_identical_evaluation_banks(self):
        seed = FINAL_SEEDS[0]
        paths = (
            P3_FINAL_DIR
            / "runs/lean_our"
            / f"seed_{seed}"
            / "evaluation_scenarios.json",
            P4_ABLATION_DIR
            / "runs/our_no_coord_cache"
            / f"seed_{seed}"
            / "evaluation_scenarios.json",
            P5_SAC_DIR
            / "runs/coord_cache_discrete_sac"
            / f"seed_{seed}"
            / "evaluation_scenarios.json",
        )
        banks = [comparable_bank(path) for path in paths]
        self.assertEqual(banks[0], banks[1])
        self.assertEqual(banks[0], banks[2])


class DependencyAblationTest(unittest.TestCase):
    def test_task_dependency_switch_preserves_only_nonstructural_state(self):
        predecessor = SimpleNamespace(
            assigned_server=1,
            result=SimpleNamespace(finish_time=0.4),
        )
        tasks = {
            "1": SimpleNamespace(
                task_number="1",
                service=1,
                predecessors=["0"],
                successors=["2"],
                cpu_cycle=20e6,
                input_data_length=25.0,
            ),
            "2": SimpleNamespace(
                task_number="2",
                service=2,
                predecessors=["1"],
                successors=[],
                cpu_cycle=30e6,
                input_data_length=20.0,
            ),
        }
        kwargs = {
            "algorithm": "causal_telemetryDDQN",
            "learning_arguments": learner_arguments(),
            "numberofservers": 2,
            "numberofservices": 2,
            "max_cpu_cycles": 100,
            "max_data_length": 100,
            "filename_png": None,
        }
        full = Agent(**kwargs, task_dependency_features_enabled=True)
        ablated = Agent(**kwargs, task_dependency_features_enabled=False)
        state_args = (
            tasks["1"],
            tasks,
            {"0": predecessor},
            1.0,
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )
        full_state = full.state(
            *state_args,
            server_quality={0: 0.5, 1: 1.0},
        )
        ablated_state = ablated.state(
            *state_args,
            server_quality={0: 0.5, 1: 1.0},
        )

        self.assertTrue(torch.equal(full_state[:7], ablated_state[:7]))
        self.assertTrue(torch.equal(full_state[9:13], ablated_state[9:13]))
        self.assertGreater(float(full_state[8]), 0.0)
        self.assertTrue(torch.equal(ablated_state[7:9], torch.zeros(2)))
        self.assertTrue(torch.equal(ablated_state[13:15], torch.zeros(2)))
        self.assertEqual(float(ablated_state[-1]), 0.0)
        self.assertGreater(float(full_state[-1]), 0.0)

    def test_cache_dependency_switch_routes_to_popularity_observation(self):
        broker = Broker.__new__(Broker)
        broker.cache_observations = 0
        broker.simulator = SimpleNamespace(
            cache_policy="critical_path_joint",
            cache_dependency_awareness_enabled=False,
        )
        task = SimpleNamespace()
        with (
            patch.object(
                broker,
                "_coordinated_popularity_update",
            ) as popularity,
            patch.object(broker, "_coordinated_cache_update") as dependency,
            patch.object(broker, "_observe_causal_cache_history"),
        ):
            broker.caching_decisions_update(0, 1, 0.0, task=task)
        popularity.assert_called_once_with(task)
        dependency.assert_not_called()
        self.assertEqual(broker.cache_observations, 1)


if __name__ == "__main__":
    unittest.main()
