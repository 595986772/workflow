import unittest

from compare_pd3qn_methods import paired_statistics


class PairedStatisticsTests(unittest.TestCase):
    def test_ties_are_not_counted_as_wins_or_losses(self):
        result = paired_statistics(
            reference=[1.0, 1.0, 1.0],
            candidate=[1.0, 0.9, 1.1],
            lower_is_better=True,
        )

        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(
            result["wins"] + result["losses"] + result["ties"],
            result["pairs"],
        )

    def test_direction_is_reversed_for_higher_is_better(self):
        result = paired_statistics(
            reference=[0.5, 0.5, 0.5],
            candidate=[0.6, 0.4, 0.5],
            lower_is_better=False,
        )

        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)


if __name__ == "__main__":
    unittest.main()
