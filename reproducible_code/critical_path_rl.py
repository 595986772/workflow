"""Causal n-step RL agents for dependency-aware edge offloading."""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_CP_RL_ALGORITHMS = {
    "causal_task_serverDDQN",
    "causal_task_serverCPQAC",
    "causal_task_serverCAPQ",
    "causal_task_serverPD3QN",
    "causal_task_serverHCPRPD3QN",
    "causal_task_serverBCRPD3QN",
}
RAW_TELEMETRY_CP_RL_ALGORITHMS = {
    "causal_telemetryDDQN",
    "causal_telemetryCPQAC",
    "causal_telemetryDiscreteSAC",
    "causal_telemetryPD3QN",
    "causal_telemetryHCPRPD3QN",
}
NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS = {
    "causal_normalizedTelemetryPD3QN",
    "causal_normalizedTelemetryBCRPD3QN",
}
TELEMETRY_CP_RL_ALGORITHMS = (
    RAW_TELEMETRY_CP_RL_ALGORITHMS
    | NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS
)
HCPR_CP_RL_ALGORITHMS = {
    "causal_task_serverHCPRPD3QN",
    "causal_telemetryHCPRPD3QN",
}
BCR_CP_RL_ALGORITHMS = {
    "causal_task_serverBCRPD3QN",
    "causal_normalizedTelemetryBCRPD3QN",
}
POSTERIOR_REPLAY_CP_RL_ALGORITHMS = (
    HCPR_CP_RL_ALGORITHMS
    | BCR_CP_RL_ALGORITHMS
)
CP_RL_ALGORITHMS = (
    BASE_CP_RL_ALGORITHMS
    | TELEMETRY_CP_RL_ALGORITHMS
)


def telemetry_channels_for_algorithm(algorithm):
    if algorithm in NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS:
        return 3
    if algorithm in RAW_TELEMETRY_CP_RL_ALGORITHMS:
        return 1
    return 0


def structural_task_criticality(task_id, tasks):
    """Return a topology-only upward rank in [0, 1]."""
    ranks = {}
    visiting = set()

    def rank(node_id):
        if node_id in ranks:
            return ranks[node_id]
        if node_id in visiting:
            raise ValueError("The application graph must be acyclic")
        visiting.add(node_id)
        successors = [
            successor_id
            for successor_id in tasks[node_id].successors
            if successor_id in tasks
        ]
        ranks[node_id] = 1.0 + max(
            (rank(successor_id) for successor_id in successors),
            default=0.0,
        )
        visiting.remove(node_id)
        return ranks[node_id]

    for node_id in tasks:
        rank(node_id)
    maximum = max(ranks.values(), default=1.0)
    return float(ranks[task_id] / maximum)


def make_n_step_transitions(trajectory, n_step, gamma):
    """Convert one completed trajectory into causal n-step transitions."""
    if n_step < 1:
        raise ValueError("n_step must be positive")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")

    transitions = []
    for start in range(len(trajectory)):
        cumulative_reward = 0.0
        reward_discount = 1.0
        final_index = start
        for offset in range(n_step):
            index = start + offset
            if index >= len(trajectory):
                break
            transition = trajectory[index]
            cumulative_reward += (
                reward_discount * float(transition["r"])
            )
            final_index = index
            if transition["done"]:
                break
            reward_discount *= gamma

        steps = final_index - start + 1
        final_transition = trajectory[final_index]
        output = copy.copy(trajectory[start])
        output["r"] = cumulative_reward
        output["s2"] = final_transition["s2"]
        output["done"] = bool(final_transition["done"])
        output["discount"] = float(gamma**steps)
        transitions.append(output)
    return transitions


class ReplayLearner:
    """Small compatibility layer for the repository's legacy agent API."""

    replay_fields = ("s", "a", "r", "s2", "done", "discount")

    def __init__(self, max_experiences, min_experiences, batch_size):
        self.max_experiences = int(max_experiences)
        self.min_experiences = int(min_experiences)
        self.batch_size = int(batch_size)
        self.experience = {
            field: []
            for field in self.replay_fields
        }
        self.device = torch.device("cpu")

    def add_experience(self, transition):
        if len(self.experience["s"]) >= self.max_experiences:
            for values in self.experience.values():
                values.pop(0)
        values = {
            "s": torch.as_tensor(
                transition["s"],
                dtype=torch.float32,
            ).detach().clone(),
            "a": int(transition["a"]),
            "r": float(transition["r"]),
            "s2": torch.as_tensor(
                transition["s2"],
                dtype=torch.float32,
            ).detach().clone(),
            "done": bool(transition["done"]),
            "discount": float(transition.get("discount", 1.0)),
        }
        for field, value in values.items():
            self.experience[field].append(value)

    def sample_batch(self):
        if len(self.experience["s"]) < self.min_experiences:
            return None
        indices = np.random.randint(
            low=0,
            high=len(self.experience["s"]),
            size=self.batch_size,
        )
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices):
        states = torch.stack(
            [self.experience["s"][index] for index in indices]
        ).to(self.device)
        next_states = torch.stack(
            [self.experience["s2"][index] for index in indices]
        ).to(self.device)
        actions = torch.as_tensor(
            [self.experience["a"][index] for index in indices],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [self.experience["r"][index] for index in indices],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [self.experience["done"][index] for index in indices],
            dtype=torch.float32,
            device=self.device,
        )
        discounts = torch.as_tensor(
            [
                self.experience["discount"][index]
                for index in indices
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            discounts,
        )


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, num_actions):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_units:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, states):
        return self.network(states)


class CorrectDoubleDQNLearner(ReplayLearner):
    def __init__(
        self,
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
    ):
        super().__init__(
            max_experiences=max_experiences,
            min_experiences=min_experiences,
            batch_size=batch_size,
        )
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.model = MLPQNetwork(
            num_states,
            hidden_units,
            num_actions,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def predict(self, inputs):
        tensor = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(tensor)

    def get_action(self, states):
        with torch.no_grad():
            values = self.predict(states)
            return int(torch.argmax(values, dim=-1).item())

    def train(self, target_network):
        batch = self.sample_batch()
        if batch is None:
            return 0.0
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            discounts,
        ) = batch
        predicted = self.model(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_values = target_network.model(next_states).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            targets = (
                rewards
                + discounts * (1.0 - dones) * next_values
            )
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def copy_weights(self, train_network):
        self.model.load_state_dict(train_network.model.state_dict())


class CorrectDDQNAgent:
    stochastic_policy = False
    uses_n_step_transitions = True

    def __init__(
        self,
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        epsilon,
        maximum_exploration,
        n_step,
        **_,
    ):
        learner_args = {
            "num_states": num_states,
            "num_actions": num_actions,
            "hidden_units": hidden_units,
            "gamma": gamma,
            "max_experiences": max_experiences,
            "min_experiences": min_experiences,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        self.TrainNet = CorrectDoubleDQNLearner(**learner_args)
        self.TargetNet = CorrectDoubleDQNLearner(**learner_args)
        self.TargetNet.copy_weights(self.TrainNet)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.initial_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.min_epsilon = float(epsilon)
        self.decay = (
            self.min_epsilon / self.initial_epsilon
        ) ** (1.0 / maximum_exploration)

    def add_trajectory(self, trajectory):
        for transition in make_n_step_transitions(
            trajectory,
            n_step=self.n_step,
            gamma=self.gamma,
        ):
            self.TrainNet.add_experience(transition)

    def replay(self, *args, **kwargs):
        return self.TrainNet.train(self.TargetNet)

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay,
        )

    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)


class TaskServerQuantileModel(nn.Module):
    """Pairwise task-server encoder with categorical actor and twin critics."""

    def __init__(
        self,
        input_dim,
        num_servers,
        num_services,
        embedding_dim,
        num_quantiles,
        risk_tail_fraction,
    ):
        super().__init__()
        self.num_servers = int(num_servers)
        self.num_services = int(num_services)
        self.num_quantiles = int(num_quantiles)
        self.risk_tail_fraction = float(risk_tail_fraction)
        base_dim = (
            3
            + self.num_servers * self.num_services
            + self.num_servers
            + 2 * self.num_services
            + 1
        )
        telemetry_dim = base_dim + self.num_servers
        if input_dim not in {base_dim, telemetry_dim}:
            raise ValueError(
                f"CPQAC state has {input_dim} values; "
                f"expected {base_dim} or {telemetry_dim}"
            )
        self.has_server_quality = input_dim == telemetry_dim

        task_dim = 4 + 2 * self.num_services
        server_dim = (
            self.num_services
            + 2
            + int(self.has_server_quality)
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.server_encoder = nn.Sequential(
            nn.Linear(server_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.pair_projection = nn.Linear(
            2 * embedding_dim,
            embedding_dim,
        )
        self.pair_norm = nn.LayerNorm(embedding_dim)
        self.actor_head = nn.Linear(embedding_dim, 1)
        self.critic1_head = nn.Linear(
            embedding_dim,
            self.num_quantiles,
        )
        self.critic2_head = nn.Linear(
            embedding_dim,
            self.num_quantiles,
        )

    def unpack_state(self, states):
        single = states.ndim == 1
        if single:
            states = states.unsqueeze(0)
        index = 3
        cache = states[
            :,
            index : index
            + self.num_servers * self.num_services,
        ].reshape(-1, self.num_servers, self.num_services)
        index += self.num_servers * self.num_services
        predecessor_servers = states[
            :,
            index : index + self.num_servers,
        ]
        index += self.num_servers
        server_quality = None
        if self.has_server_quality:
            server_quality = states[
                :,
                index : index + self.num_servers,
            ]
            index += self.num_servers
        current_service = states[
            :,
            index : index + self.num_services,
        ]
        index += self.num_services
        successor_services = states[
            :,
            index : index + self.num_services,
        ]
        criticality = states[:, -1:].clamp(0.0, 1.0)

        task_features = torch.cat(
            [
                states[:, :3],
                current_service,
                successor_services,
                criticality,
            ],
            dim=1,
        )
        current_cache_hit = (
            cache * current_service.unsqueeze(1)
        ).sum(dim=2, keepdim=True)
        server_feature_parts = [
            cache,
            predecessor_servers.unsqueeze(2),
        ]
        if server_quality is not None:
            server_feature_parts.append(
                server_quality.unsqueeze(2)
            )
        server_feature_parts.append(current_cache_hit)
        server_features = torch.cat(server_feature_parts, dim=2)
        return task_features, server_features, criticality, single

    def pair_embeddings(self, states):
        (
            task_features,
            server_features,
            criticality,
            single,
        ) = self.unpack_state(states)
        task_embedding = self.task_encoder(task_features)
        server_embedding = self.server_encoder(server_features)
        attention_logits = (
            self.query(task_embedding).unsqueeze(1)
            * self.key(server_embedding)
        ).sum(dim=2) / math.sqrt(task_embedding.shape[-1])
        attention = torch.softmax(attention_logits, dim=1)
        context = (
            attention.unsqueeze(2)
            * self.value(server_embedding)
        ).sum(dim=1)
        task_context = torch.cat(
            [task_embedding, context],
            dim=1,
        )
        pair_task = self.pair_projection(task_context).unsqueeze(1)
        pairs = self.pair_norm(pair_task + server_embedding)
        return F.silu(pairs), criticality, single

    def forward(self, states):
        pairs, criticality, single = self.pair_embeddings(states)
        logits = self.actor_head(pairs).squeeze(2)
        quantiles1 = self.critic1_head(pairs)
        quantiles2 = self.critic2_head(pairs)
        if single:
            return (
                logits.squeeze(0),
                quantiles1.squeeze(0),
                quantiles2.squeeze(0),
                criticality.squeeze(0),
            )
        return logits, quantiles1, quantiles2, criticality

    def risk_values(self, states):
        _, quantiles1, quantiles2, criticality = self(states)
        single = quantiles1.ndim == 2
        if single:
            quantiles1 = quantiles1.unsqueeze(0)
            quantiles2 = quantiles2.unsqueeze(0)
            criticality = criticality.reshape(1, 1)
        conservative = torch.minimum(quantiles1, quantiles2)
        ordered = torch.sort(conservative, dim=2).values
        mean_return = conservative.mean(dim=2)
        tail_count = max(
            1,
            int(round(
                self.num_quantiles * self.risk_tail_fraction
            )),
        )
        lower_tail = ordered[:, :, :tail_count].mean(dim=2)
        values = (
            (1.0 - criticality) * mean_return
            + criticality * lower_tail
        )
        return values.squeeze(0) if single else values


def quantile_huber_loss(predicted, target, taus, kappa=1.0):
    delta = target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute = delta.abs()
    huber = torch.where(
        absolute <= kappa,
        0.5 * delta.square(),
        kappa * (absolute - 0.5 * kappa),
    )
    weights = torch.abs(
        taus.view(1, -1, 1)
        - (delta.detach() < 0.0).float()
    )
    return (weights * huber / kappa).mean()


class CPQACLearner(ReplayLearner):
    def __init__(
        self,
        num_states,
        num_actions,
        num_services,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        num_quantiles,
        risk_tail_fraction,
        entropy_coefficient,
    ):
        super().__init__(
            max_experiences=max_experiences,
            min_experiences=min_experiences,
            batch_size=batch_size,
        )
        self.num_actions = int(num_actions)
        self.num_quantiles = int(num_quantiles)
        self.gamma = float(gamma)
        self.entropy_coefficient = float(entropy_coefficient)
        embedding_dim = int(hidden_units[0])
        self.model = TaskServerQuantileModel(
            input_dim=num_states,
            num_servers=num_actions,
            num_services=num_services,
            embedding_dim=embedding_dim,
            num_quantiles=num_quantiles,
            risk_tail_fraction=risk_tail_fraction,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )
        self.taus = (
            torch.arange(
                num_quantiles,
                dtype=torch.float32,
                device=self.device,
            )
            + 0.5
        ) / num_quantiles

    def predict(self, inputs):
        states = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=self.device,
        )
        return self.model.risk_values(states)

    def get_action(self, states):
        with torch.no_grad():
            tensor = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )
            logits, _, _, _ = self.model(tensor)
            return int(torch.argmax(logits, dim=-1).item())

    def sample_action(self, states):
        with torch.no_grad():
            tensor = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )
            logits, _, _, _ = self.model(tensor)
            return int(
                torch.distributions.Categorical(
                    logits=logits,
                ).sample().item()
            )

    def train(self, target_network):
        batch = self.sample_batch()
        if batch is None:
            return 0.0
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            discounts,
        ) = batch
        logits, quantiles1, quantiles2, _ = self.model(states)
        action_index = actions[:, None, None].expand(
            -1,
            1,
            self.num_quantiles,
        )
        selected1 = quantiles1.gather(
            1,
            action_index,
        ).squeeze(1)
        selected2 = quantiles2.gather(
            1,
            action_index,
        ).squeeze(1)

        with torch.no_grad():
            next_logits, _, _, _ = self.model(next_states)
            next_probabilities = torch.softmax(
                next_logits,
                dim=1,
            )
            next_actions = torch.multinomial(
                next_probabilities,
                num_samples=1,
            )
            next_log_probabilities = torch.log_softmax(
                next_logits,
                dim=1,
            ).gather(1, next_actions).squeeze(1)
            _, target1, target2, _ = target_network.model(
                next_states
            )
            target_index = next_actions.unsqueeze(2).expand(
                -1,
                1,
                self.num_quantiles,
            )
            target_quantiles = torch.minimum(
                target1.gather(1, target_index).squeeze(1),
                target2.gather(1, target_index).squeeze(1),
            )
            target_quantiles = (
                rewards.unsqueeze(1)
                + discounts.unsqueeze(1)
                * (1.0 - dones).unsqueeze(1)
                * (
                    target_quantiles
                    - self.entropy_coefficient
                    * next_log_probabilities.unsqueeze(1)
                )
            )

        critic_loss = (
            quantile_huber_loss(
                selected1,
                target_quantiles,
                self.taus,
            )
            + quantile_huber_loss(
                selected2,
                target_quantiles,
                self.taus,
            )
        )
        probabilities = torch.softmax(logits, dim=1)
        log_probabilities = torch.log_softmax(logits, dim=1)
        with torch.no_grad():
            risk_values = self.model.risk_values(states)
        actor_loss = (
            probabilities
            * (
                self.entropy_coefficient * log_probabilities
                - risk_values
            )
        ).sum(dim=1).mean()
        loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def copy_weights(self, train_network):
        self.model.load_state_dict(train_network.model.state_dict())


class CPQACAgent:
    stochastic_policy = True
    uses_n_step_transitions = True

    def __init__(
        self,
        num_states,
        num_actions,
        num_services,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        epsilon,
        maximum_exploration,
        n_step,
        num_quantiles,
        risk_tail_fraction,
        entropy_coefficient,
        **_,
    ):
        learner_args = {
            "num_states": num_states,
            "num_actions": num_actions,
            "num_services": num_services,
            "hidden_units": hidden_units,
            "gamma": gamma,
            "max_experiences": max_experiences,
            "min_experiences": min_experiences,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_quantiles": num_quantiles,
            "risk_tail_fraction": risk_tail_fraction,
            "entropy_coefficient": entropy_coefficient,
        }
        self.TrainNet = CPQACLearner(**learner_args)
        self.TargetNet = CPQACLearner(**learner_args)
        self.TargetNet.copy_weights(self.TrainNet)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.initial_epsilon = 0.0
        self.epsilon = 0.0
        self.min_epsilon = 0.0
        self.decay = 1.0

    def add_trajectory(self, trajectory):
        for transition in make_n_step_transitions(
            trajectory,
            n_step=self.n_step,
            gamma=self.gamma,
        ):
            self.TrainNet.add_experience(transition)

    def replay(self, *args, **kwargs):
        return self.TrainNet.train(self.TargetNet)

    def decay_epsilon(self):
        return None

    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)

    def sample_action(self, states):
        return self.TrainNet.sample_action(states)


class CAPQLearner(ReplayLearner):
    """Criticality-adaptive pairwise quantile Double-Q learner."""

    def __init__(
        self,
        num_states,
        num_actions,
        num_services,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        num_quantiles,
        risk_tail_fraction,
    ):
        super().__init__(
            max_experiences=max_experiences,
            min_experiences=min_experiences,
            batch_size=batch_size,
        )
        self.num_actions = int(num_actions)
        self.num_quantiles = int(num_quantiles)
        self.gamma = float(gamma)
        self.model = TaskServerQuantileModel(
            input_dim=num_states,
            num_servers=num_actions,
            num_services=num_services,
            embedding_dim=int(hidden_units[0]),
            num_quantiles=num_quantiles,
            risk_tail_fraction=risk_tail_fraction,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )
        self.taus = (
            torch.arange(
                num_quantiles,
                dtype=torch.float32,
                device=self.device,
            )
            + 0.5
        ) / num_quantiles

    def predict(self, inputs):
        states = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=self.device,
        )
        return self.model.risk_values(states)

    def get_action(self, states):
        with torch.no_grad():
            return int(
                torch.argmax(
                    self.predict(states),
                    dim=-1,
                ).item()
            )

    def train(self, target_network):
        batch = self.sample_batch()
        if batch is None:
            return 0.0
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            discounts,
        ) = batch
        _, quantiles1, quantiles2, _ = self.model(states)
        action_index = actions[:, None, None].expand(
            -1,
            1,
            self.num_quantiles,
        )
        selected1 = quantiles1.gather(
            1,
            action_index,
        ).squeeze(1)
        selected2 = quantiles2.gather(
            1,
            action_index,
        ).squeeze(1)

        with torch.no_grad():
            next_actions = self.model.risk_values(
                next_states
            ).argmax(dim=1)
            _, target1, target2, _ = target_network.model(
                next_states
            )
            target_index = next_actions[:, None, None].expand(
                -1,
                1,
                self.num_quantiles,
            )
            target_quantiles = torch.minimum(
                target1.gather(
                    1,
                    target_index,
                ).squeeze(1),
                target2.gather(
                    1,
                    target_index,
                ).squeeze(1),
            )
            target_quantiles = (
                rewards.unsqueeze(1)
                + discounts.unsqueeze(1)
                * (1.0 - dones).unsqueeze(1)
                * target_quantiles
            )

        loss = (
            quantile_huber_loss(
                selected1,
                target_quantiles,
                self.taus,
            )
            + quantile_huber_loss(
                selected2,
                target_quantiles,
                self.taus,
            )
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def copy_weights(self, train_network):
        self.model.load_state_dict(
            train_network.model.state_dict()
        )


class CAPQAgent:
    stochastic_policy = False
    uses_n_step_transitions = True

    def __init__(
        self,
        num_states,
        num_actions,
        num_services,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        epsilon,
        maximum_exploration,
        n_step,
        num_quantiles,
        risk_tail_fraction,
        **_,
    ):
        learner_args = {
            "num_states": num_states,
            "num_actions": num_actions,
            "num_services": num_services,
            "hidden_units": hidden_units,
            "gamma": gamma,
            "max_experiences": max_experiences,
            "min_experiences": min_experiences,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_quantiles": num_quantiles,
            "risk_tail_fraction": risk_tail_fraction,
        }
        self.TrainNet = CAPQLearner(**learner_args)
        self.TargetNet = CAPQLearner(**learner_args)
        self.TargetNet.copy_weights(self.TrainNet)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.initial_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.min_epsilon = float(epsilon)
        self.decay = (
            self.min_epsilon / self.initial_epsilon
        ) ** (1.0 / maximum_exploration)

    def add_trajectory(self, trajectory):
        for transition in make_n_step_transitions(
            trajectory,
            n_step=self.n_step,
            gamma=self.gamma,
        ):
            self.TrainNet.add_experience(transition)

    def replay(self, *args, **kwargs):
        return self.TrainNet.train(self.TargetNet)

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay,
        )

    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)


class PairwiseDuelingQModel(nn.Module):
    """Task-server cross-attention with a dueling value decomposition."""

    def __init__(
        self,
        input_dim,
        num_servers,
        num_services,
        embedding_dim,
    ):
        super().__init__()
        self.num_servers = int(num_servers)
        self.num_services = int(num_services)
        base_dim = (
            3
            + self.num_servers * self.num_services
            + self.num_servers
            + 2 * self.num_services
            + 1
        )
        telemetry_values = input_dim - base_dim
        if (
            telemetry_values < 0
            or telemetry_values % self.num_servers != 0
        ):
            raise ValueError(
                f"PD3QN state has {input_dim} values; "
                "telemetry must contain a fixed number of values "
                "per server"
            )
        self.telemetry_channels = (
            telemetry_values // self.num_servers
        )
        if self.telemetry_channels not in {0, 1, 3}:
            raise ValueError(
                f"PD3QN state has {self.telemetry_channels} telemetry "
                "channels per server; expected 0, 1, or 3"
            )
        self.has_server_quality = self.telemetry_channels > 0

        task_dim = 4 + 2 * self.num_services
        server_dim = (
            self.num_services
            + 2
            + self.telemetry_channels
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.server_encoder = nn.Sequential(
            nn.Linear(server_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.task_context = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.pair_norm = nn.LayerNorm(embedding_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, states):
        single = states.ndim == 1
        if single:
            states = states.unsqueeze(0)
        index = 3
        cache = states[
            :,
            index:index
            + self.num_servers * self.num_services,
        ].reshape(-1, self.num_servers, self.num_services)
        index += self.num_servers * self.num_services
        predecessor_servers = states[
            :,
            index:index + self.num_servers,
        ]
        index += self.num_servers
        server_quality = None
        if self.has_server_quality:
            server_quality = states[
                :,
                index:(
                    index
                    + self.num_servers * self.telemetry_channels
                ),
            ].reshape(
                -1,
                self.num_servers,
                self.telemetry_channels,
            )
            index += self.num_servers * self.telemetry_channels
        current_service = states[
            :,
            index:index + self.num_services,
        ]
        index += self.num_services
        successor_services = states[
            :,
            index:index + self.num_services,
        ]
        criticality = states[:, -1:].clamp(0.0, 1.0)

        task_features = torch.cat(
            [
                states[:, :3],
                current_service,
                successor_services,
                criticality,
            ],
            dim=1,
        )
        cache_hit = (
            cache * current_service.unsqueeze(1)
        ).sum(dim=2, keepdim=True)
        server_feature_parts = [
            cache,
            predecessor_servers.unsqueeze(2),
        ]
        if server_quality is not None:
            server_feature_parts.append(server_quality)
        server_feature_parts.append(cache_hit)
        server_features = torch.cat(server_feature_parts, dim=2)
        task_embedding = self.task_encoder(task_features)
        server_embedding = self.server_encoder(server_features)
        attention_logits = (
            self.query(task_embedding).unsqueeze(1)
            * self.key(server_embedding)
        ).sum(dim=2) / math.sqrt(task_embedding.shape[-1])
        attention = torch.softmax(attention_logits, dim=1)
        context = (
            attention.unsqueeze(2)
            * self.value(server_embedding)
        ).sum(dim=1)
        task_context = self.task_context(
            torch.cat([task_embedding, context], dim=1)
        )
        pairs = F.silu(
            self.pair_norm(
                task_context.unsqueeze(1)
                + server_embedding
            )
        )
        value = self.value_head(task_context)
        advantages = self.advantage_head(pairs).squeeze(2)
        q_values = (
            value
            + advantages
            - advantages.mean(dim=1, keepdim=True)
        )
        return q_values.squeeze(0) if single else q_values


class PD3QNLearner(ReplayLearner):
    def __init__(
        self,
        num_states,
        num_actions,
        num_services,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        prioritized_replay=False,
        priority_alpha=0.6,
        priority_beta_start=0.4,
        priority_beta_anneal_steps=2000,
        criticality_boost=2.0,
    ):
        super().__init__(
            max_experiences=max_experiences,
            min_experiences=min_experiences,
            batch_size=batch_size,
        )
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.prioritized_replay = bool(prioritized_replay)
        self.priority_alpha = float(priority_alpha)
        self.priority_beta = float(priority_beta_start)
        if self.priority_alpha < 0:
            raise ValueError("priority_alpha must be non-negative")
        if not 0 <= self.priority_beta <= 1:
            raise ValueError(
                "priority_beta_start must be between zero and one"
            )
        if int(priority_beta_anneal_steps) < 1:
            raise ValueError(
                "priority_beta_anneal_steps must be positive"
            )
        if float(criticality_boost) < 0:
            raise ValueError("criticality_boost must be non-negative")
        self.priority_beta_increment = (
            (1.0 - self.priority_beta)
            / max(int(priority_beta_anneal_steps), 1)
        )
        self.criticality_boost = float(criticality_boost)
        self.priority_epsilon = 1e-6
        self.td_priorities = []
        self.priorities = []
        self.posterior_criticalities = []
        self.last_buffer_mean_criticality = 0.0
        self.last_sampled_mean_criticality = 0.0
        self.last_importance_weight_mean = 1.0
        self.model = PairwiseDuelingQModel(
            input_dim=num_states,
            num_servers=num_actions,
            num_services=num_services,
            embedding_dim=int(hidden_units[0]),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def predict(self, inputs):
        states = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(states)

    def get_action(self, states):
        with torch.no_grad():
            return int(
                torch.argmax(
                    self.predict(states),
                    dim=-1,
                ).item()
            )

    def add_experience(self, transition):
        if not self.prioritized_replay:
            super().add_experience(transition)
            return

        if len(self.experience["s"]) >= self.max_experiences:
            self.td_priorities.pop(0)
            self.priorities.pop(0)
            self.posterior_criticalities.pop(0)
        super().add_experience(transition)
        criticality = float(
            np.clip(
                transition.get("posterior_criticality", 0.0),
                0.0,
                1.0,
            )
        )
        self.posterior_criticalities.append(criticality)
        td_priority = max(
            max(self.td_priorities, default=1.0),
            self.priority_epsilon,
        )
        self.td_priorities.append(td_priority)
        self.priorities.append(
            td_priority
            * (1.0 + self.criticality_boost * criticality)
        )

    def sample_batch(self):
        if not self.prioritized_replay:
            return super().sample_batch()
        size = len(self.experience["s"])
        if size < self.min_experiences:
            return None

        scaled = np.power(
            np.asarray(self.priorities, dtype=float),
            self.priority_alpha,
        )
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(
            size,
            size=self.batch_size,
            replace=True,
            p=probabilities,
        )
        importance_weights = np.power(
            size * probabilities[indices],
            -self.priority_beta,
        )
        importance_weights /= max(
            float(importance_weights.max()),
            self.priority_epsilon,
        )
        criticalities = np.asarray(
            [
                self.posterior_criticalities[index]
                for index in indices
            ],
            dtype=float,
        )
        self.last_sampled_mean_criticality = float(
            criticalities.mean()
        )
        self.last_buffer_mean_criticality = float(
            np.mean(self.posterior_criticalities)
        )
        self.last_importance_weight_mean = float(
            importance_weights.mean()
        )
        self.priority_beta = min(
            1.0,
            self.priority_beta + self.priority_beta_increment,
        )
        return self._batch_from_indices(indices) + (
            indices,
            torch.as_tensor(
                importance_weights,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def train(self, target_network):
        batch = self.sample_batch()
        if batch is None:
            return 0.0
        if self.prioritized_replay:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                discounts,
                indices,
                importance_weights,
            ) = batch
        else:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                discounts,
            ) = batch
        predicted = self.model(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(
                next_states
            ).argmax(dim=1)
            next_values = target_network.model(
                next_states
            ).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            targets = (
                rewards
                + discounts
                * (1.0 - dones)
                * next_values
            )
        elementwise_loss = F.smooth_l1_loss(
            predicted,
            targets,
            reduction="none",
        )
        if self.prioritized_replay:
            loss = (importance_weights * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        if self.prioritized_replay:
            td_errors = (
                targets - predicted
            ).detach().abs().cpu().numpy()
            updated_priorities = {}
            for index, td_error in zip(indices, td_errors):
                priority = float(td_error) + self.priority_epsilon
                updated_priorities[index] = max(
                    updated_priorities.get(index, 0.0),
                    priority,
                )
            for index, priority in updated_priorities.items():
                td_priority = max(
                    priority,
                    self.priority_epsilon,
                )
                criticality = self.posterior_criticalities[index]
                self.td_priorities[index] = td_priority
                self.priorities[index] = (
                    td_priority
                    * (
                        1.0
                        + self.criticality_boost * criticality
                    )
                )
        return float(loss.item())

    def copy_weights(self, train_network):
        self.model.load_state_dict(
            train_network.model.state_dict()
        )


class PD3QNAgent:
    stochastic_policy = False
    uses_n_step_transitions = True

    def __init__(
        self,
        num_states,
        num_actions,
        num_services,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learning_rate,
        epsilon,
        maximum_exploration,
        n_step,
        priority_alpha=0.6,
        priority_beta_start=0.4,
        priority_beta_anneal_steps=2000,
        criticality_boost=2.0,
        algorithm=None,
        **_,
    ):
        if algorithm in BCR_CP_RL_ALGORITHMS:
            self.posterior_replay_mode = "bottleneck_contribution"
        elif algorithm in HCPR_CP_RL_ALGORITHMS:
            self.posterior_replay_mode = "critical_path"
        else:
            self.posterior_replay_mode = "none"
        self.uses_hindsight_critical_path_replay = (
            algorithm in POSTERIOR_REPLAY_CP_RL_ALGORITHMS
        )
        learner_args = {
            "num_states": num_states,
            "num_actions": num_actions,
            "num_services": num_services,
            "hidden_units": hidden_units,
            "gamma": gamma,
            "max_experiences": max_experiences,
            "min_experiences": min_experiences,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "prioritized_replay": (
                self.uses_hindsight_critical_path_replay
            ),
            "priority_alpha": priority_alpha,
            "priority_beta_start": priority_beta_start,
            "priority_beta_anneal_steps": (
                priority_beta_anneal_steps
            ),
            "criticality_boost": criticality_boost,
        }
        self.TrainNet = PD3QNLearner(**learner_args)
        self.TargetNet = PD3QNLearner(**learner_args)
        self.TargetNet.copy_weights(self.TrainNet)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.initial_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.min_epsilon = float(epsilon)
        self.decay = (
            self.min_epsilon / self.initial_epsilon
        ) ** (1.0 / maximum_exploration)

    def add_trajectory(self, trajectory):
        for transition in make_n_step_transitions(
            trajectory,
            n_step=self.n_step,
            gamma=self.gamma,
        ):
            self.TrainNet.add_experience(transition)

    def replay(self, *args, **kwargs):
        return self.TrainNet.train(self.TargetNet)

    def decay_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay,
        )

    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)
