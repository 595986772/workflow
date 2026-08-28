"""Regression tests for the P14 service sensitivity protocol."""

import unittest

from pegasus_service_sensitivity_protocol import (
    ACTIVE_SERVICE_COUNTS,
    BASE_DATASET_PATH,
    BASE_DATASET_SHA256,
    DAOC,
    DISPLAY_NAMES,
    DQN_COORD_CACHE,
    EXPECTED_PROJECTED_SHA256,
    METHODS,
    SERVICE_STATE_DIMENSION,
    build_projected_datasets,
    decode_service,
    encode_service,
    project_service_id,
    projected_dataset_path,
    sha256_file,
    validate_method_identity,
    validate_projected_dataset,
)


class PegasusServiceSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = build_projected_datasets()

    def test_base_dataset_is_frozen(self):
        self.assertEqual(sha256_file(BASE_DATASET_PATH), BASE_DATASET_SHA256)

    def test_projected_dataset_hashes_are_frozen(self):
        self.assertEqual(self.hashes, EXPECTED_PROJECTED_SHA256)

    def test_service_encoding_round_trip(self):
        for service_id in range(SERVICE_STATE_DIMENSION + 1):
            self.assertEqual(decode_service(encode_service(service_id)), service_id)

    def test_modulo_projection_is_deterministic(self):
        self.assertEqual(
            [project_service_id(value, 4) for value in range(1, 11)],
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
        )

    def test_projected_datasets_only_change_service_identity(self):
        for active_services in ACTIVE_SERVICE_COUNTS:
            record = validate_projected_dataset(active_services)
            self.assertEqual(
                record["sha256"],
                EXPECTED_PROJECTED_SHA256[active_services],
            )
            self.assertEqual(
                record["observed_services"],
                list(range(1, active_services + 1)),
            )
            self.assertTrue(projected_dataset_path(active_services).is_file())

    def test_dqn_coord_cache_replaces_daoc_hybrid(self):
        self.assertIn(DAOC, METHODS)
        self.assertIn(DQN_COORD_CACHE, METHODS)
        self.assertEqual(DISPLAY_NAMES[DQN_COORD_CACHE], "DQN+CoordCache")
        self.assertNotIn("daoc_our_coord_cache", METHODS)

    def test_learning_baseline_identities(self):
        validate_method_identity()


if __name__ == "__main__":
    unittest.main()

