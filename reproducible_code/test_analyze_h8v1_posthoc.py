import unittest

import numpy as np

from analyze_h8v1_posthoc import (
    confidence_interval,
    normalized_entropy,
    normalized_gini,
    normalized_hhi,
    paired_diagnostic,
)


class DistributionMetricTests(unittest.TestCase):
    def test_uniform_distribution_is_not_concentrated(self):
        values = np.ones(10)
        self.assertAlmostEqual(normalized_hhi(values), 0.0)
        self.assertAlmostEqual(normalized_gini(values), 0.0)
        self.assertAlmostEqual(normalized_entropy(values), 1.0)

    def test_single_server_distribution_is_maximally_concentrated(self):
        values = np.asarray([10.0] + [0.0] * 9)
        self.assertAlmostEqual(normalized_hhi(values), 1.0)
        self.assertAlmostEqual(normalized_gini(values), 1.0)
        self.assertAlmostEqual(normalized_entropy(values), 0.0)

    def test_empty_distribution_has_finite_metrics(self):
        values = np.zeros(10)
        self.assertEqual(normalized_hhi(values), 0.0)
        self.assertEqual(normalized_gini(values), 0.0)
        self.assertEqual(normalized_entropy(values), 0.0)


class StatisticalDiagnosticTests(unittest.TestCase):
    def test_single_value_interval_is_degenerate(self):
        self.assertEqual(confidence_interval([0.25]), (0.25, 0.25))

    def test_paired_diagnostic_tracks_wins(self):
        result = paired_diagnostic([0.2, 0.1, -0.01])
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["wins"], 2)
        self.assertTrue(result["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
