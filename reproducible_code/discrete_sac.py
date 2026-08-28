"""Categorical Soft Actor-Critic for causal discrete edge offloading."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from critical_path_rl import ReplayLearner, make_n_step_transitions


class PairwiseTaskServerEncoder(nn.Module):
    """OUR-compatible task-server encoder without privileged inputs."""

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
        telemetry_values = int(input_dim) - base_dim
        if (
            telemetry_values < 0
            or telemetry_values % self.num_servers != 0
        ):
            raise ValueError(
                "Discrete SAC state must contain a fixed number of "
                "causal telemetry values per server"
            )
        self.telemetry_channels = (
            telemetry_values // self.num_servers
        )
        if self.telemetry_channels not in {0, 1, 3}:
            raise ValueError(
                "Discrete SAC supports zero, one, or three causal "
                "telemetry channels per server"
            )

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

        telemetry = None
        if self.telemetry_channels:
            telemetry = states[
                :,
                index:index
                + self.num_servers * self.telemetry_channels,
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
        server_parts = [
            cache,
            predecessor_servers.unsqueeze(2),
        ]
        if telemetry is not None:
            server_parts.append(telemetry)
        server_parts.append(cache_hit)
        server_features = torch.cat(server_parts, dim=2)

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
                task_context.unsqueeze(1) + server_embedding
            )
        )
        return pairs.squeeze(0) if single else pairs


class CategoricalActor(nn.Module):
    def __init__(
        self,
        input_dim,
        num_servers,
        num_services,
        embedding_dim,
    ):
        super().__init__()
        self.encoder = PairwiseTaskServerEncoder(
            input_dim,
            num_servers,
            num_services,
            embedding_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, states):
        return self.head(self.encoder(states)).squeeze(-1)


class PairwiseCritic(nn.Module):
    def __init__(
        self,
        input_dim,
        num_servers,
        num_services,
        embedding_dim,
    ):
        super().__init__()
        self.encoder = PairwiseTaskServerEncoder(
            input_dim,
            num_servers,
            num_services,
            embedding_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, states):
        return self.head(self.encoder(states)).squeeze(-1)


class DiscreteSACModel(nn.Module):
    """Actor, twin critics, targets, and temperature in one checkpoint."""

    def __init__(
        self,
        input_dim,
        num_servers,
        num_services,
        embedding_dim,
        initial_alpha,
    ):
        super().__init__()
        network_args = (
            input_dim,
            num_servers,
            num_services,
            embedding_dim,
        )
        self.actor = CategoricalActor(*network_args)
        self.critic1 = PairwiseCritic(*network_args)
        self.critic2 = PairwiseCritic(*network_args)
        self.target_critic1 = PairwiseCritic(*network_args)
        self.target_critic2 = PairwiseCritic(*network_args)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(float(initial_alpha)))
        )
        self.hard_update_targets()
        for parameter in self.target_parameters():
            parameter.requires_grad_(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def target_parameters(self):
        return list(self.target_critic1.parameters()) + list(
            self.target_critic2.parameters()
        )

    def hard_update_targets(self):
        self.target_critic1.load_state_dict(
            self.critic1.state_dict()
        )
        self.target_critic2.load_state_dict(
            self.critic2.state_dict()
        )

    @torch.no_grad()
    def soft_update_targets(self, tau):
        for target, online in zip(
            self.target_critic1.parameters(),
            self.critic1.parameters(),
        ):
            target.mul_(1.0 - tau).add_(online, alpha=tau)
        for target, online in zip(
            self.target_critic2.parameters(),
            self.critic2.parameters(),
        ):
            target.mul_(1.0 - tau).add_(online, alpha=tau)


class DiscreteSACLearner(ReplayLearner):
    """Expectation-form Discrete SAC with automatic entropy tuning."""

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
        initial_alpha,
        target_entropy_ratio,
        target_tau,
    ):
        super().__init__(
            max_experiences=max_experiences,
            min_experiences=min_experiences,
            batch_size=batch_size,
        )
        if initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")
        if not 0 < target_entropy_ratio <= 1:
            raise ValueError(
                "target_entropy_ratio must be in (0, 1]"
            )
        if not 0 < target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.target_entropy = float(
            target_entropy_ratio * math.log(self.num_actions)
        )
        self.target_tau = float(target_tau)
        self.model = DiscreteSACModel(
            input_dim=num_states,
            num_servers=num_actions,
            num_services=num_services,
            embedding_dim=int(hidden_units[0]),
            initial_alpha=initial_alpha,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=learning_rate,
        )
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0
        self.last_alpha_loss = 0.0
        self.last_entropy = math.log(self.num_actions)

    @property
    def alpha(self):
        return float(self.model.alpha.detach().item())

    def policy(self, states):
        logits = self.model.actor(states)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        return logits, probabilities, log_probabilities

    def predict(self, inputs):
        states = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.minimum(
            self.model.critic1(states),
            self.model.critic2(states),
        )

    def get_action(self, states):
        with torch.no_grad():
            tensor = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.model.actor(tensor)
            return int(torch.argmax(logits, dim=-1).item())

    def sample_action(self, states):
        with torch.no_grad():
            tensor = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.model.actor(tensor)
            return int(
                torch.distributions.Categorical(
                    logits=logits
                ).sample().item()
            )

    def train(self):
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

        with torch.no_grad():
            _, next_probabilities, next_log_probabilities = (
                self.policy(next_states)
            )
            target_q = torch.minimum(
                self.model.target_critic1(next_states),
                self.model.target_critic2(next_states),
            )
            soft_next_value = (
                next_probabilities
                * (
                    target_q
                    - self.model.alpha.detach()
                    * next_log_probabilities
                )
            ).sum(dim=1)
            targets = (
                rewards
                + discounts
                * (1.0 - dones)
                * soft_next_value
            )

        predicted1 = self.model.critic1(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        predicted2 = self.model.critic2(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        critic_loss = (
            F.smooth_l1_loss(predicted1, targets)
            + F.smooth_l1_loss(predicted2, targets)
        )
        self.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.model.critic1.parameters())
            + list(self.model.critic2.parameters()),
            10.0,
        )
        self.optimizer.step()

        _, probabilities, log_probabilities = self.policy(states)
        with torch.no_grad():
            policy_q = torch.minimum(
                self.model.critic1(states),
                self.model.critic2(states),
            )
        actor_loss = (
            probabilities
            * (
                self.model.alpha.detach() * log_probabilities
                - policy_q
            )
        ).sum(dim=1).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.actor.parameters(),
            10.0,
        )
        self.optimizer.step()

        with torch.no_grad():
            _, updated_probabilities, updated_log_probabilities = (
                self.policy(states)
            )
            entropy = -(
                updated_probabilities * updated_log_probabilities
            ).sum(dim=1)
        alpha_loss = (
            self.model.log_alpha
            * (entropy.detach() - self.target_entropy)
        ).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.model.log_alpha.clamp_(-10.0, 2.0)
            self.model.soft_update_targets(self.target_tau)

        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_alpha_loss = float(alpha_loss.item())
        self.last_entropy = float(entropy.mean().item())
        return float(
            self.last_critic_loss
            + self.last_actor_loss
            + self.last_alpha_loss
        )


class DiscreteSACAgent:
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
        entropy_coefficient,
        sac_target_entropy_ratio=0.98,
        sac_target_tau=0.005,
        **_,
    ):
        del epsilon, maximum_exploration
        self.TrainNet = DiscreteSACLearner(
            num_states=num_states,
            num_actions=num_actions,
            num_services=num_services,
            hidden_units=hidden_units,
            gamma=gamma,
            max_experiences=max_experiences,
            min_experiences=min_experiences,
            batch_size=batch_size,
            learning_rate=learning_rate,
            initial_alpha=entropy_coefficient,
            target_entropy_ratio=sac_target_entropy_ratio,
            target_tau=sac_target_tau,
        )
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
        return self.TrainNet.train()

    def decay_epsilon(self):
        return None

    def update_target_model(self):
        # Targets are updated by Polyak averaging after every SAC update.
        return None

    def sample_action(self, states):
        return self.TrainNet.sample_action(states)
