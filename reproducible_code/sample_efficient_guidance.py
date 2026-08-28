"""Non-oracle helpers for sample-efficient DAG offloading.

The functions in this module deliberately accept only application metadata,
cache broadcasts, predecessor placements, Q values, and completed-task
measurements. They never receive live server queues, CPU frequencies, or link
rates, which keeps the proposed guidance inside the original information
boundary.
"""

import math

import numpy as np


EPSILON = 1e-12


def compute_nominal_upward_ranks(tasks, max_cpu_cycles, max_data_length):
    """Return static HEFT-style ranks built from normalized DAG metadata."""
    cpu_scale = max(float(max_cpu_cycles) * 1e6, EPSILON)
    data_scale = max(float(max_data_length), EPSILON)
    ranks = {}
    visiting = set()

    def rank(task_id):
        if task_id in ranks:
            return ranks[task_id]
        if task_id in visiting:
            raise ValueError("The application graph must be acyclic")

        visiting.add(task_id)
        task = tasks[task_id]
        own_work = (
            float(task.cpu_cycle) / cpu_scale
            + float(task.input_data_length) / data_scale
        )
        successor_tails = []
        for successor_id in task.successors:
            if successor_id not in tasks:
                continue
            edge_work = (
                float(task.outputs_length.get(successor_id, 0.0))
                / data_scale
            )
            successor_tails.append(edge_work + rank(successor_id))
        ranks[task_id] = own_work + max(successor_tails, default=0.0)
        visiting.remove(task_id)
        return ranks[task_id]

    for task_id in tasks:
        rank(task_id)

    maximum_rank = max(ranks.values(), default=1.0)
    return ranks, max(maximum_rank, EPSILON)


def normalized_task_features(task, max_cpu_cycles, max_data_length):
    """Return observable CPU and input-data features for online regression."""
    cpu_work = float(task.cpu_cycle) / max(
        float(max_cpu_cycles) * 1e6,
        EPSILON,
    )
    input_work = float(task.input_data_length) / max(
        float(max_data_length),
        EPSILON,
    )
    return cpu_work, input_work


def normalized_remote_predecessor_data(
    task,
    done_tasks,
    candidate_server,
    max_data_length,
):
    """Use predecessor placements from the original state to price DAG traffic."""
    total = 0.0
    for predecessor_id in task.predecessors:
        predecessor = done_tasks[predecessor_id]
        if predecessor.assigned_server != candidate_server:
            total += float(
                predecessor.outputs_length.get(task.task_number, 0.0)
            )
    return total / max(float(max_data_length), EPSILON)


def q_margin_confidence(q_values):
    """Map the top-two Q-value margin to a bounded confidence score."""
    values = np.asarray(q_values, dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        return 0.0
    top_two = np.partition(values, -2)[-2:]
    best = float(np.max(top_two))
    second = float(np.min(top_two))
    margin = max(0.0, best - second)
    return float(
        np.clip(
            margin / (1.0 + abs(best) + abs(second)),
            0.0,
            1.0,
        )
    )


def adaptive_guidance_probability(
    q_confidence,
    task_criticality,
    expert_confidence,
    maximum_probability,
    handoff_factor=1.0,
):
    """Return the probability of trusting guidance for the current decision."""
    probability = (
        float(maximum_probability)
        * (0.5 + 0.5 * float(task_criticality))
        * (1.0 - float(q_confidence))
        * float(expert_confidence)
        * float(handoff_factor)
    )
    return float(np.clip(probability, 0.0, 1.0))


class HistoricalFeedbackGuide:
    """Online per-server delay models learned from completed tasks only."""

    def __init__(self, number_of_servers, alpha=0.1, min_samples=3):
        if number_of_servers < 1:
            raise ValueError("number_of_servers must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")

        self.number_of_servers = int(number_of_servers)
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.feature_dimension = 5
        self.ridge = 1e-3
        ridge_matrix = self.ridge * np.eye(self.feature_dimension)
        self.counts = np.zeros(self.number_of_servers, dtype=np.int64)
        self.gram_matrices = np.repeat(
            ridge_matrix[np.newaxis, :, :],
            self.number_of_servers,
            axis=0,
        )
        self.target_vectors = np.zeros(
            (self.number_of_servers, self.feature_dimension),
            dtype=float,
        )
        self.mean_absolute_residuals = np.zeros(
            self.number_of_servers,
            dtype=float,
        )
        self.pooled_gram = ridge_matrix.copy()
        self.pooled_target = np.zeros(self.feature_dimension, dtype=float)
        self.pooled_count = 0
        self.pooled_mean_absolute_residual = 0.0

    @staticmethod
    def _update_ewma(previous, sample, count, alpha):
        if count == 0:
            return float(sample)
        return float((1.0 - alpha) * previous + alpha * sample)

    @staticmethod
    def _feature(
        normalized_cpu,
        normalized_input,
        normalized_remote_data,
        cache_hit,
    ):
        return np.asarray(
            [
                1.0,
                max(float(normalized_cpu), 0.0),
                max(float(normalized_input), 0.0),
                max(float(normalized_remote_data), 0.0),
                float(not bool(cache_hit)),
            ],
            dtype=float,
        )

    @staticmethod
    def _solve(gram, target):
        try:
            return np.linalg.solve(gram, target)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram) @ target

    def _predict(self, server_id, feature):
        if self.counts[server_id]:
            coefficients = self._solve(
                self.gram_matrices[server_id],
                self.target_vectors[server_id],
            )
        else:
            coefficients = self._solve(
                self.pooled_gram,
                self.pooled_target,
            )
        return max(0.0, float(feature @ coefficients))

    def _update_normal_equations(self, gram, target, feature, outcome):
        decay = 1.0 - self.alpha
        gram *= decay
        gram += self.alpha * (
            np.outer(feature, feature)
            + self.ridge * np.eye(self.feature_dimension)
        )
        target *= decay
        target += self.alpha * feature * outcome

    def update(
        self,
        server_id,
        cache_hit,
        normalized_cpu,
        normalized_input,
        normalized_remote_data,
        observed_path_delay,
    ):
        """Update one target-server model after the task has completed."""
        server_id = int(server_id)
        feature = self._feature(
            normalized_cpu,
            normalized_input,
            normalized_remote_data,
            cache_hit,
        )
        outcome = max(float(observed_path_delay), 0.0)
        prediction = self._predict(server_id, feature)
        residual = abs(outcome - prediction)
        count = int(self.counts[server_id])
        self.mean_absolute_residuals[server_id] = self._update_ewma(
            self.mean_absolute_residuals[server_id],
            residual,
            count,
            self.alpha,
        )
        self.pooled_mean_absolute_residual = self._update_ewma(
            self.pooled_mean_absolute_residual,
            residual,
            self.pooled_count,
            self.alpha,
        )

        self._update_normal_equations(
            self.gram_matrices[server_id],
            self.target_vectors[server_id],
            feature,
            outcome,
        )
        self._update_normal_equations(
            self.pooled_gram,
            self.pooled_target,
            feature,
            outcome,
        )
        self.counts[server_id] += 1
        self.pooled_count += 1

    def recommend(
        self,
        cache_hits,
        normalized_cpu,
        normalized_input,
        remote_data_by_server,
    ):
        """Rank candidate servers from historical measurements and observables."""
        cache_hits = np.asarray(cache_hits, dtype=bool).reshape(-1)
        remote_data = np.asarray(remote_data_by_server, dtype=float).reshape(-1)
        if cache_hits.size != self.number_of_servers:
            raise ValueError("cache_hits has the wrong number of servers")
        if remote_data.size != self.number_of_servers:
            raise ValueError(
                "remote_data_by_server has the wrong number of servers"
            )

        relevant_total = int(self.counts.sum())
        observed_targets = int(np.count_nonzero(self.counts))
        required_targets = min(2, self.number_of_servers)
        ready = (
            relevant_total >= self.min_samples
            and observed_targets >= required_targets
        )
        if not ready:
            return {
                "ready": False,
                "action": None,
                "confidence": 0.0,
                "scores": None,
                "relevant_samples": relevant_total,
                "observed_targets": observed_targets,
            }

        scores = np.empty(self.number_of_servers, dtype=float)
        for server_id in range(self.number_of_servers):
            feature = self._feature(
                normalized_cpu,
                normalized_input,
                remote_data[server_id],
                cache_hits[server_id],
            )
            prediction = self._predict(server_id, feature)
            count = int(self.counts[server_id])
            if count:
                residual_scale = self.mean_absolute_residuals[server_id]
            else:
                residual_scale = max(
                    self.pooled_mean_absolute_residual,
                    0.25 * prediction,
                    EPSILON,
                )
            uncertainty_penalty = (
                0.25 * residual_scale / math.sqrt(count + 1.0)
            )
            scores[server_id] = prediction + uncertainty_penalty

        sample_target = self.min_samples * self.number_of_servers
        sample_confidence = min(1.0, relevant_total / sample_target)
        coverage_target = max(
            required_targets,
            math.ceil(self.number_of_servers / 2),
        )
        coverage_confidence = min(
            1.0,
            observed_targets / coverage_target,
        )
        confidence = math.sqrt(sample_confidence * coverage_confidence)
        return {
            "ready": True,
            "action": int(np.argmin(scores)),
            "confidence": float(confidence),
            "scores": scores,
            "relevant_samples": relevant_total,
            "observed_targets": observed_targets,
        }

    def state_dict(self):
        """Return a JSON-compatible snapshot for frozen-evaluation checks."""
        return {
            "counts": self.counts.tolist(),
            "gram_matrices": self.gram_matrices.tolist(),
            "target_vectors": self.target_vectors.tolist(),
            "mean_absolute_residuals": (
                self.mean_absolute_residuals.tolist()
            ),
            "pooled_gram": self.pooled_gram.tolist(),
            "pooled_target": self.pooled_target.tolist(),
            "pooled_count": int(self.pooled_count),
            "pooled_mean_absolute_residual": float(
                self.pooled_mean_absolute_residual
            ),
        }
