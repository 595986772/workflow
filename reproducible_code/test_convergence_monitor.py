import unittest
from types import SimpleNamespace

from convergence_monitor import ConvergenceMonitor, relative_linear_slope
from run_independent_experiment import scheduled_learning_rate


class ConvergenceMonitorTest(unittest.TestCase):
    def test_relative_slope_is_normalized(self):
        self.assertAlmostEqual(
            relative_linear_slope([1.0, 1.01, 1.02]),
            0.01 / 1.01,
        )

    def test_requires_stable_patience_after_minimum_episode(self):
        monitor = ConvergenceMonitor(
            min_episode=400,
            window=2,
            patience=2,
            relative_mean_change_threshold=0.05,
            relative_slope_threshold=0.02,
        )
        diagnostics = []
        for episode, value in (
            (100, 1.0),
            (200, 1.0),
            (300, 1.0),
            (400, 1.00),
            (500, 1.01),
            (600, 1.00),
            (700, 1.01),
            (800, 1.00),
        ):
            diagnostics.append(monitor.update(episode, value))

        self.assertFalse(diagnostics[2]["eligible"])
        self.assertFalse(diagnostics[5]["eligible"])
        self.assertTrue(diagnostics[6]["stable_window"])
        self.assertEqual(diagnostics[6]["stable_streak"], 1)
        self.assertTrue(diagnostics[7]["converged"])

    def test_trend_resets_stable_streak(self):
        monitor = ConvergenceMonitor(
            min_episode=100,
            window=2,
            patience=2,
            relative_mean_change_threshold=0.20,
            relative_slope_threshold=0.01,
        )
        monitor.update(100, 1.0)
        monitor.update(200, 1.0)
        monitor.update(300, 1.0)
        first = monitor.update(400, 1.0)
        second = monitor.update(500, 0.8)

        self.assertTrue(first["stable_window"])
        self.assertFalse(second["stable_window"])
        self.assertEqual(second["stable_streak"], 0)
        self.assertFalse(second["converged"])

    def test_rejects_non_increasing_episodes(self):
        monitor = ConvergenceMonitor(
            min_episode=100,
            window=2,
            patience=1,
            relative_mean_change_threshold=0.05,
            relative_slope_threshold=0.02,
        )
        monitor.update(100, 1.0)
        with self.assertRaises(ValueError):
            monitor.update(100, 1.0)


class LearningRateScheduleTest(unittest.TestCase):
    def test_cosine_schedule_reaches_floor(self):
        args = SimpleNamespace(
            learning_rate_schedule="cosine",
            learning_rate=1e-3,
            min_learning_rate=1e-5,
            learning_rate_decay_start=100,
            learning_rate_decay_end=300,
        )
        self.assertEqual(scheduled_learning_rate(args, 100), 1e-3)
        self.assertAlmostEqual(
            scheduled_learning_rate(args, 200),
            (1e-3 + 1e-5) / 2,
        )
        self.assertEqual(scheduled_learning_rate(args, 300), 1e-5)

    def test_constant_schedule_does_not_decay(self):
        args = SimpleNamespace(
            learning_rate_schedule="constant",
            learning_rate=1e-3,
        )
        self.assertEqual(scheduled_learning_rate(args, 100000), 1e-3)


if __name__ == "__main__":
    unittest.main()
