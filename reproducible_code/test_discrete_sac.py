import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from agent import Agent
from discrete_sac import DiscreteSACAgent, DiscreteSACLearner


def learner_arguments():
    return {
        "hidden_units": [16, 16],
        "gamma": 1.0,
        "max_experiences": 100,
        "min_experiences": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "epsilon": 0.01,
        "maximum_exploration": 100,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.02,
        "sac_target_entropy_ratio": 0.98,
        "sac_target_tau": 0.005,
    }


def task(
    task_id,
    service,
    predecessors=(),
    successors=(),
    cpu_cycle=10e6,
    input_data_length=10.0,
):
    return SimpleNamespace(
        task_number=task_id,
        service=service,
        predecessors=list(predecessors),
        successors=list(successors),
        cpu_cycle=cpu_cycle,
        input_data_length=input_data_length,
    )


class DiscreteSACStateProtocolTest(unittest.TestCase):
    def test_state_is_identical_to_our_causal_telemetry_state(self):
        tasks = {
            "1": task("1", 1, successors=("2",)),
            "2": task(
                "2",
                2,
                predecessors=("1",),
                successors=("3",),
            ),
            "3": task("3", 1, predecessors=("2",)),
        }
        done_tasks = {
            "1": SimpleNamespace(
                assigned_server=1,
                result=SimpleNamespace(finish_time=0.25),
            )
        }
        services = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=float,
        )
        quality = {0: 0.4, 1: 1.0, 2: 0.7}
        common = {
            "learning_arguments": learner_arguments(),
            "numberofservers": 3,
            "numberofservices": 2,
            "max_cpu_cycles": 100,
            "max_data_length": 100,
            "filename_png": None,
        }
        our = Agent(
            algorithm="causal_telemetryPD3QN",
            **common,
        )
        sac = Agent(
            algorithm="causal_telemetryDiscreteSAC",
            **common,
        )

        our_state = our.state(
            tasks["2"],
            tasks,
            done_tasks,
            1.0,
            services,
            server_quality=quality,
        )
        sac_state = sac.state(
            tasks["2"],
            tasks,
            done_tasks,
            1.0,
            services,
            server_quality=quality,
        )

        self.assertEqual(our.state_size, sac.state_size)
        self.assertTrue(torch.equal(our_state, sac_state))

    def test_state_does_not_read_future_workload_magnitudes(self):
        tasks = {
            "1": task("1", 1, successors=("2",)),
            "2": task("2", 2, predecessors=("1",)),
        }
        agent = Agent(
            algorithm="causal_telemetryDiscreteSAC",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        state_before = agent.state(
            tasks["1"],
            tasks,
            {},
            1.0,
            np.zeros((3, 2), dtype=float),
            server_quality={0: 1.0, 1: 1.0, 2: 1.0},
        )
        tasks["2"].cpu_cycle = 1e18
        tasks["2"].input_data_length = 1e18
        state_after = agent.state(
            tasks["1"],
            tasks,
            {},
            1.0,
            np.zeros((3, 2), dtype=float),
            server_quality={0: 1.0, 1: 1.0, 2: 1.0},
        )
        self.assertTrue(torch.equal(state_before, state_after))


class DiscreteSACLearnerTest(unittest.TestCase):
    def make_learner(self):
        num_servers = 3
        num_services = 2
        state_size = (
            3
            + num_servers * num_services
            + num_servers
            + num_servers
            + 2 * num_services
            + 1
        )
        return DiscreteSACLearner(
            num_states=state_size,
            num_actions=num_servers,
            num_services=num_services,
            hidden_units=[16, 16],
            gamma=1.0,
            max_experiences=50,
            min_experiences=2,
            batch_size=2,
            learning_rate=1e-3,
            initial_alpha=0.02,
            target_entropy_ratio=0.98,
            target_tau=0.005,
        )

    def populate(self, learner):
        state_size = learner.num_states
        for index in range(6):
            state = torch.zeros(state_size)
            next_state = torch.zeros(state_size)
            state[index % state_size] = 0.1 * (index + 1)
            next_state[(index + 1) % state_size] = 0.1
            learner.add_experience(
                {
                    "s": state,
                    "a": index % learner.num_actions,
                    "r": -0.05 * (index + 1),
                    "s2": next_state,
                    "done": index == 5,
                    "discount": 1.0,
                }
            )

    def test_categorical_actions_and_training_update(self):
        torch.manual_seed(7)
        learner = self.make_learner()
        self.populate(learner)
        state = learner.experience["s"][0]
        logits = learner.model.actor(state)
        before_critic = [
            parameter.detach().clone()
            for parameter in learner.model.critic1.parameters()
        ]
        before_target = [
            parameter.detach().clone()
            for parameter in learner.model.target_critic1.parameters()
        ]

        loss = learner.train()
        action = learner.sample_action(state)

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertIn(action, range(3))
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isfinite(learner.alpha))
        self.assertGreater(learner.alpha, 0.0)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    before_critic,
                    learner.model.critic1.parameters(),
                )
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    before_target,
                    learner.model.target_critic1.parameters(),
                )
            )
        )

    def test_checkpoint_round_trip_preserves_policy(self):
        torch.manual_seed(11)
        source = self.make_learner()
        target = self.make_learner()
        observation = torch.randn(source.num_states)
        target.model.load_state_dict(source.model.state_dict())

        self.assertEqual(
            source.get_action(observation),
            target.get_action(observation),
        )
        self.assertAlmostEqual(source.alpha, target.alpha)

    def test_agent_uses_n_step_replay_and_zero_epsilon(self):
        state_size = 20
        agent = DiscreteSACAgent(
            num_states=state_size,
            num_actions=3,
            num_services=2,
            hidden_units=[16, 16],
            gamma=1.0,
            max_experiences=50,
            min_experiences=2,
            batch_size=2,
            learning_rate=1e-3,
            epsilon=0.01,
            maximum_exploration=100,
            n_step=3,
            entropy_coefficient=0.02,
        )
        trajectory = [
            {
                "s": torch.zeros(state_size),
                "a": index % 3,
                "r": -0.1,
                "s2": torch.ones(state_size) * (index + 1),
                "done": index == 2,
            }
            for index in range(3)
        ]
        agent.add_trajectory(trajectory)

        self.assertEqual(agent.epsilon, 0.0)
        self.assertTrue(agent.stochastic_policy)
        self.assertEqual(len(agent.TrainNet.experience["s"]), 3)


if __name__ == "__main__":
    unittest.main()
