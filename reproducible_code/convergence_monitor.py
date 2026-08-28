from dataclasses import dataclass, field
import math


def relative_linear_slope(values):
    values = [float(value) for value in values]
    mean_value = sum(values) / len(values)
    if mean_value <= 0:
        raise ValueError("convergence values must have a positive mean")

    x_mean = (len(values) - 1) / 2.0
    numerator = sum(
        (index - x_mean) * (value - mean_value)
        for index, value in enumerate(values)
    )
    denominator = sum(
        (index - x_mean) ** 2
        for index in range(len(values))
    )
    slope = numerator / denominator if denominator else 0.0
    return slope / mean_value


@dataclass
class ConvergenceMonitor:
    min_episode: int
    window: int
    patience: int
    relative_mean_change_threshold: float
    relative_slope_threshold: float
    observations: list = field(default_factory=list)
    stable_streak: int = 0
    converged: bool = False

    def __post_init__(self):
        if self.min_episode < 1:
            raise ValueError("min_episode must be positive")
        if self.window < 2:
            raise ValueError("window must be at least two")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.relative_mean_change_threshold < 0:
            raise ValueError(
                "relative_mean_change_threshold must be non-negative"
            )
        if self.relative_slope_threshold < 0:
            raise ValueError("relative_slope_threshold must be non-negative")

    def update(self, episode, value):
        episode = int(episode)
        value = float(value)
        if self.observations and episode <= self.observations[-1][0]:
            raise ValueError("checkpoint episodes must be strictly increasing")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("convergence value must be finite and positive")

        self.observations.append((episode, value))
        diagnostics = {
            "eligible": False,
            "window_start_episode": None,
            "window_end_episode": episode,
            "window_mean": None,
            "previous_window_mean": None,
            "relative_mean_change": None,
            "relative_range": None,
            "relative_slope_per_checkpoint": None,
            "stable_window": False,
            "stable_streak": self.stable_streak,
            "converged": self.converged,
        }
        required_observations = 2 * self.window
        if len(self.observations) < required_observations:
            return diagnostics

        recent = self.observations[-required_observations:]
        if recent[0][0] < self.min_episode:
            return diagnostics
        previous_values = [
            observation[1]
            for observation in recent[: self.window]
        ]
        current_values = [
            observation[1]
            for observation in recent[self.window :]
        ]
        recent_values = previous_values + current_values
        previous_mean = sum(previous_values) / len(previous_values)
        mean_value = sum(current_values) / len(current_values)
        relative_mean_change = abs(
            mean_value - previous_mean
        ) / ((mean_value + previous_mean) / 2.0)
        relative_range = (
            max(current_values) - min(current_values)
        ) / mean_value
        relative_slope = relative_linear_slope(recent_values)
        stable_window = (
            relative_mean_change
            <= self.relative_mean_change_threshold
            and abs(relative_slope) <= self.relative_slope_threshold
        )
        self.stable_streak = self.stable_streak + 1 if stable_window else 0
        self.converged = self.stable_streak >= self.patience

        diagnostics.update(
            {
                "eligible": True,
                "window_start_episode": recent[0][0],
                "window_mean": mean_value,
                "previous_window_mean": previous_mean,
                "relative_mean_change": relative_mean_change,
                "relative_range": relative_range,
                "relative_slope_per_checkpoint": relative_slope,
                "stable_window": stable_window,
                "stable_streak": self.stable_streak,
                "converged": self.converged,
            }
        )
        return diagnostics
