import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agent import Agent
from critical_path_rl import (
    CAPQLearner,
    CPQACLearner,
    CorrectDoubleDQNLearner,
    PD3QNLearner,
    PairwiseDuelingQModel,
    TaskServerQuantileModel,
    make_n_step_transitions,
    structural_task_criticality,
)


def make_task(
    task_id,
    service,
    predecessors=None,
    successors=None,
    cpu_cycle=10e6,
    input_data_length=10.0,
):
    return SimpleNamespace(
        task_number=task_id,
        service=service,
        predecessors=list(predecessors or []),
        successors=list(successors or []),
        cpu_cycle=cpu_cycle,
        input_data_length=input_data_length,
    )


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
        "num_quantiles": 8,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
    }


class StructuralCriticalityTest(unittest.TestCase):
    def make_chain(self):
        return {
            "1": make_task("1", 1, successors=["2"]),
            "2": make_task(
                "2",
                2,
                predecessors=["1"],
                successors=["3"],
            ),
            "3": make_task("3", 1, predecessors=["2"]),
        }

    def test_topology_rank_ignores_future_workload_values(self):
        tasks = self.make_chain()
        before = structural_task_criticality("1", tasks)
        tasks["2"].cpu_cycle = 1e18
        tasks["2"].input_data_length = 1e18
        after = structural_task_criticality("1", tasks)

        self.assertAlmostEqual(before, after)
        self.assertGreater(
            before,
            structural_task_criticality("3", tasks),
        )

    def test_agent_state_ignores_unexecuted_cpu_and_input(self):
        tasks = {
            "1": make_task("1", 1, successors=["2"]),
            "2": make_task("2", 2, predecessors=["1"]),
        }
        agent = Agent(
            algorithm="causal_task_serverDDQN",
            learning_arguments=learner_arguments(),
            numberofservers=2,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        services = np.zeros((2, 2), dtype=float)
        before = agent.state(
            tasks["1"],
            tasks,
            {},
            1.0,
            services,
        )
        tasks["2"].cpu_cycle = 1e18
        tasks["2"].input_data_length = 1e18
        after = agent.state(
            tasks["1"],
            tasks,
            {},
            1.0,
            services,
        )
        self.assertTrue(torch.equal(before, after))

    def test_telemetry_state_contains_causal_server_quality(self):
        tasks = {
            "1": make_task("1", 1, successors=["2"]),
            "2": make_task("2", 2, predecessors=["1"]),
        }
        agent = Agent(
            algorithm="causal_telemetryDDQN",
            learning_arguments=learner_arguments(),
            numberofservers=2,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        state = agent.state(
            tasks["1"],
            tasks,
            {},
            1.0,
            np.zeros((2, 2), dtype=float),
            server_quality={0: 0.25, 1: 1.0},
        )
        quality_start = 3 + 2 * 2 + 2
        self.assertTrue(
            torch.equal(
                state[quality_start:quality_start + 2],
                torch.tensor([0.25, 1.0]),
            )
        )


class NStepTransitionTest(unittest.TestCase):
    def test_n_step_rewards_and_terminal_flags(self):
        trajectory = [
            {
                "s": torch.tensor([float(index)]),
                "a": 0,
                "r": float(index + 1),
                "s2": torch.tensor([float(index + 1)]),
                "done": index == 3,
            }
            for index in range(4)
        ]
        transitions = make_n_step_transitions(
            trajectory,
            n_step=3,
            gamma=0.5,
        )

        self.assertEqual(len(transitions), 4)
        self.assertAlmostEqual(transitions[0]["r"], 2.75)
        self.assertFalse(transitions[0]["done"])
        self.assertAlmostEqual(transitions[0]["discount"], 0.125)
        self.assertAlmostEqual(transitions[1]["r"], 4.5)
        self.assertTrue(transitions[1]["done"])
        self.assertAlmostEqual(transitions[2]["r"], 5.0)
        self.assertAlmostEqual(transitions[3]["r"], 4.0)

    def test_n_step_preserves_hindsight_metadata(self):
        trajectory = [
            {
                "s": torch.tensor([float(index)]),
                "a": 0,
                "r": -1.0,
                "s2": torch.tensor([float(index + 1)]),
                "done": index == 1,
                "task_id": str(index),
                "posterior_criticality": float(index),
            }
            for index in range(2)
        ]

        transitions = make_n_step_transitions(
            trajectory,
            n_step=2,
            gamma=1.0,
        )

        self.assertEqual(transitions[0]["task_id"], "0")
        self.assertEqual(transitions[0]["posterior_criticality"], 0.0)
        self.assertEqual(transitions[1]["task_id"], "1")
        self.assertEqual(transitions[1]["posterior_criticality"], 1.0)


class LearnerSmokeTest(unittest.TestCase):
    def test_correct_double_dqn_updates_parameters(self):
        learner = CorrectDoubleDQNLearner(
            num_states=4,
            num_actions=2,
            hidden_units=[8],
            gamma=1.0,
            max_experiences=20,
            min_experiences=2,
            batch_size=2,
            learning_rate=1e-3,
        )
        target = CorrectDoubleDQNLearner(
            num_states=4,
            num_actions=2,
            hidden_units=[8],
            gamma=1.0,
            max_experiences=20,
            min_experiences=2,
            batch_size=2,
            learning_rate=1e-3,
        )
        target.copy_weights(learner)
        for index in range(4):
            learner.add_experience(
                {
                    "s": torch.full((4,), float(index)),
                    "a": index % 2,
                    "r": -0.1 * (index + 1),
                    "s2": torch.full((4,), float(index + 1)),
                    "done": index == 3,
                    "discount": 1.0,
                }
            )
        before = [
            parameter.detach().clone()
            for parameter in learner.model.parameters()
        ]
        loss = learner.train(target)
        changed = any(
            not torch.equal(old, new)
            for old, new in zip(before, learner.model.parameters())
        )

        self.assertTrue(math.isfinite(loss))
        self.assertTrue(changed)

    def test_cpqac_shapes_and_update(self):
        num_servers = 3
        num_services = 2
        state_size = (
            3
            + num_servers * num_services
            + num_servers
            + 2 * num_services
            + 1
        )
        model = TaskServerQuantileModel(
            input_dim=state_size,
            num_servers=num_servers,
            num_services=num_services,
            embedding_dim=8,
            num_quantiles=8,
            risk_tail_fraction=0.25,
        )
        states = torch.zeros((4, state_size))
        states[:, -1] = torch.linspace(0.0, 1.0, 4)
        logits, quantiles1, quantiles2, _ = model(states)
        self.assertEqual(tuple(logits.shape), (4, num_servers))
        self.assertEqual(
            tuple(quantiles1.shape),
            (4, num_servers, 8),
        )
        self.assertEqual(quantiles1.shape, quantiles2.shape)

        telemetry_model = TaskServerQuantileModel(
            input_dim=state_size + num_servers,
            num_servers=num_servers,
            num_services=num_services,
            embedding_dim=8,
            num_quantiles=8,
            risk_tail_fraction=0.25,
        )
        telemetry_states = torch.zeros(
            (4, state_size + num_servers)
        )
        telemetry_logits, _, _, _ = telemetry_model(
            telemetry_states
        )
        self.assertEqual(
            tuple(telemetry_logits.shape),
            (4, num_servers),
        )

        learner = CPQACLearner(
            num_states=state_size,
            num_actions=num_servers,
            num_services=num_services,
            hidden_units=[8, 8],
            gamma=1.0,
            max_experiences=20,
            min_experiences=2,
            batch_size=2,
            learning_rate=1e-3,
            num_quantiles=8,
            risk_tail_fraction=0.25,
            entropy_coefficient=0.02,
        )
        target = CPQACLearner(
            num_states=state_size,
            num_actions=num_servers,
            num_services=num_services,
            hidden_units=[8, 8],
            gamma=1.0,
            max_experiences=20,
            min_experiences=2,
            batch_size=2,
            learning_rate=1e-3,
            num_quantiles=8,
            risk_tail_fraction=0.25,
            entropy_coefficient=0.02,
        )
        target.copy_weights(learner)
        for index in range(4):
            state = torch.zeros(state_size)
            state[-1] = index / 3
            learner.add_experience(
                {
                    "s": state,
                    "a": index % num_servers,
                    "r": -0.1 * (index + 1),
                    "s2": state.clone(),
                    "done": index == 3,
                    "discount": 1.0,
                }
            )
        loss = learner.train(target)
        self.assertTrue(math.isfinite(loss))

    def test_capq_updates_distributional_critics(self):
        num_servers = 3
        num_services = 2
        state_size = (
            3
            + num_servers * num_services
            + num_servers
            + 2 * num_services
            + 1
        )
        learner_args = {
            "num_states": state_size,
            "num_actions": num_servers,
            "num_services": num_services,
            "hidden_units": [8, 8],
            "gamma": 1.0,
            "max_experiences": 20,
            "min_experiences": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "num_quantiles": 8,
            "risk_tail_fraction": 0.25,
        }
        learner = CAPQLearner(**learner_args)
        target = CAPQLearner(**learner_args)
        target.copy_weights(learner)
        for index in range(4):
            state = torch.zeros(state_size)
            state[-1] = index / 3
            learner.add_experience(
                {
                    "s": state,
                    "a": index % num_servers,
                    "r": -0.1 * (index + 1),
                    "s2": state.clone(),
                    "done": index == 3,
                    "discount": 1.0,
                }
            )
        before = [
            parameter.detach().clone()
            for parameter in learner.model.parameters()
        ]
        loss = learner.train(target)
        changed = any(
            not torch.equal(old, new)
            for old, new in zip(before, learner.model.parameters())
        )
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(changed)

    def test_pairwise_dueling_double_dqn_updates(self):
        num_servers = 3
        num_services = 2
        state_size = (
            3
            + num_servers * num_services
            + num_servers
            + 2 * num_services
            + 1
        )
        model = PairwiseDuelingQModel(
            input_dim=state_size,
            num_servers=num_servers,
            num_services=num_services,
            embedding_dim=8,
        )
        self.assertEqual(
            tuple(model(torch.zeros((4, state_size))).shape),
            (4, num_servers),
        )
        learner_args = {
            "num_states": state_size,
            "num_actions": num_servers,
            "num_services": num_services,
            "hidden_units": [8, 8],
            "gamma": 1.0,
            "max_experiences": 20,
            "min_experiences": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
        }
        learner = PD3QNLearner(**learner_args)
        target = PD3QNLearner(**learner_args)
        target.copy_weights(learner)
        for index in range(4):
            state = torch.zeros(state_size)
            state[-1] = index / 3
            learner.add_experience(
                {
                    "s": state,
                    "a": index % num_servers,
                    "r": -0.1 * (index + 1),
                    "s2": state.clone(),
                    "done": index == 3,
                    "discount": 1.0,
                }
            )
        loss = learner.train(target)
        self.assertTrue(math.isfinite(loss))

    def test_pairwise_model_uses_per_server_causal_telemetry(self):
        num_servers = 3
        num_services = 2
        base_state_size = (
            3
            + num_servers * num_services
            + num_servers
            + 2 * num_services
            + 1
        )
        model = PairwiseDuelingQModel(
            input_dim=base_state_size + num_servers,
            num_servers=num_servers,
            num_services=num_services,
            embedding_dim=8,
        )
        state = torch.zeros(base_state_size + num_servers)
        quality_start = (
            3
            + num_servers * num_services
            + num_servers
        )
        state[quality_start:quality_start + num_servers] = (
            torch.tensor([0.2, 0.8, 0.5])
        )
        state[-1] = 0.5
        swapped = state.clone()
        swapped[quality_start] = state[quality_start + 1]
        swapped[quality_start + 1] = state[quality_start]

        with torch.no_grad():
            values = model(state)
            swapped_values = model(swapped)

        self.assertTrue(model.has_server_quality)
        self.assertTrue(
            torch.allclose(
                values[[1, 0, 2]],
                swapped_values,
                atol=1e-6,
            )
        )

    def test_hcpr_priorities_and_importance_weights_are_active(self):
        num_servers = 2
        num_services = 2
        state_size = (
            3
            + num_servers * num_services
            + num_servers
            + num_servers
            + 2 * num_services
            + 1
        )
        learner_args = {
            "num_states": state_size,
            "num_actions": num_servers,
            "num_services": num_services,
            "hidden_units": [8, 8],
            "gamma": 1.0,
            "max_experiences": 3,
            "min_experiences": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "prioritized_replay": True,
            "priority_alpha": 0.6,
            "priority_beta_start": 0.4,
            "priority_beta_anneal_steps": 10,
            "criticality_boost": 2.0,
        }
        learner = PD3QNLearner(**learner_args)
        target = PD3QNLearner(**learner_args)
        target.copy_weights(learner)
        for index, criticality in enumerate((0.0, 1.0)):
            state = torch.zeros(state_size)
            state[-1] = criticality
            learner.add_experience(
                {
                    "s": state,
                    "a": index,
                    "r": -float(index + 1),
                    "s2": state.clone(),
                    "done": index == 1,
                    "discount": 1.0,
                    "posterior_criticality": criticality,
                }
            )

        self.assertEqual(learner.priorities, [1.0, 3.0])
        learner.add_experience(
            {
                "s": torch.ones(state_size),
                "a": 0,
                "r": -1.0,
                "s2": torch.ones(state_size),
                "done": True,
                "discount": 1.0,
                "posterior_criticality": 0.0,
            }
        )
        self.assertEqual(learner.priorities, [1.0, 3.0, 1.0])
        beta_before = learner.priority_beta
        with patch(
            "critical_path_rl.np.random.choice",
            return_value=np.asarray([0, 1]),
        ):
            loss = learner.train(target)
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(learner.priority_beta, beta_before)
        self.assertGreater(learner.last_sampled_mean_criticality, 0.0)
        self.assertGreater(learner.last_importance_weight_mean, 0.0)
        self.assertLessEqual(learner.last_importance_weight_mean, 1.0)

        learner.add_experience(
            {
                "s": torch.ones(state_size),
                "a": 0,
                "r": -1.0,
                "s2": torch.ones(state_size),
                "done": True,
                "discount": 1.0,
                "posterior_criticality": 0.5,
            }
        )
        self.assertEqual(len(learner.experience["s"]), 3)
        self.assertEqual(len(learner.td_priorities), 3)
        self.assertEqual(len(learner.priorities), 3)
        self.assertEqual(len(learner.posterior_criticalities), 3)

    def test_factorial_agents_isolate_hcpr_and_telemetry(self):
        base = Agent(
            algorithm="causal_task_serverPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        telemetry_only = Agent(
            algorithm="causal_telemetryPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        hcpr_only = Agent(
            algorithm="causal_task_serverHCPRPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        full = Agent(
            algorithm="causal_telemetryHCPRPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )

        self.assertFalse(
            base.agent.uses_hindsight_critical_path_replay
        )
        self.assertFalse(base.agent.TrainNet.prioritized_replay)
        self.assertFalse(base.agent.TrainNet.model.has_server_quality)

        self.assertFalse(
            telemetry_only.agent.uses_hindsight_critical_path_replay
        )
        self.assertFalse(
            telemetry_only.agent.TrainNet.prioritized_replay
        )
        self.assertTrue(
            telemetry_only.agent.TrainNet.model.has_server_quality
        )

        self.assertTrue(
            hcpr_only.agent.uses_hindsight_critical_path_replay
        )
        self.assertTrue(hcpr_only.agent.TrainNet.prioritized_replay)
        self.assertFalse(
            hcpr_only.agent.TrainNet.model.has_server_quality
        )

        self.assertTrue(
            full.agent.uses_hindsight_critical_path_replay
        )
        self.assertTrue(full.agent.TrainNet.prioritized_replay)
        self.assertTrue(full.agent.TrainNet.model.has_server_quality)

        self.assertEqual(hcpr_only.state_size, base.state_size)
        self.assertEqual(
            telemetry_only.state_size,
            base.state_size + 3,
        )
        self.assertEqual(full.state_size, telemetry_only.state_size)

    def test_redesigned_agents_isolate_bcr_and_normalized_telemetry(self):
        base = Agent(
            algorithm="causal_task_serverPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        normalized = Agent(
            algorithm="causal_normalizedTelemetryPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        bcr = Agent(
            algorithm="causal_task_serverBCRPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )
        full = Agent(
            algorithm="causal_normalizedTelemetryBCRPD3QN",
            learning_arguments=learner_arguments(),
            numberofservers=3,
            numberofservices=2,
            max_cpu_cycles=100,
            max_data_length=100,
            filename_png=None,
        )

        self.assertEqual(normalized.state_size, base.state_size + 9)
        self.assertEqual(full.state_size, normalized.state_size)
        self.assertEqual(bcr.state_size, base.state_size)
        self.assertEqual(
            normalized.agent.TrainNet.model.telemetry_channels,
            3,
        )
        self.assertFalse(
            normalized.agent.uses_hindsight_critical_path_replay
        )
        self.assertTrue(
            bcr.agent.uses_hindsight_critical_path_replay
        )
        self.assertEqual(
            bcr.agent.posterior_replay_mode,
            "bottleneck_contribution",
        )
        self.assertEqual(
            full.agent.posterior_replay_mode,
            "bottleneck_contribution",
        )
        self.assertTrue(full.agent.TrainNet.prioritized_replay)


if __name__ == "__main__":
    unittest.main()
