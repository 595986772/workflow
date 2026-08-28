import unittest

from build_pegasus_cross_dataset import (
    SERVICE_COUNT,
    encoded_service_value,
    scale_log,
    service_id,
)


class PegasusDatasetBuilderTests(unittest.TestCase):
    def test_service_mapping_is_stable_and_valid(self):
        first = service_id("Montage", "mProjectPP")
        second = service_id("Montage", "mProjectPP")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first, SERVICE_COUNT)

    def test_encoded_service_round_trip(self):
        for identifier in range(1, SERVICE_COUNT + 1):
            value = encoded_service_value(identifier)
            decoded = int(SERVICE_COUNT * (value - 1)) + 1
            self.assertEqual(decoded, identifier)

    def test_log_scaling_is_bounded(self):
        self.assertAlmostEqual(scale_log(0, 0, 10), 0.05)
        self.assertAlmostEqual(scale_log(1e9, 0, 10), 1.0)


if __name__ == "__main__":
    unittest.main()
