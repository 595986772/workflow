import unittest

import run_reproduction_suite as reproduction
from pegasus_daoc_coord_extension_protocol import (
    DAOC_COORD_LABEL,
    DAOC_LABEL,
    PROFILE_CONVERGED,
    algorithm_config,
    register_suite_extension,
)


class DaocOurCoordCacheProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_suite_extension()

    def test_only_cache_subsystem_differs(self):
        daoc = algorithm_config(DAOC_LABEL)
        coordinated = algorithm_config(DAOC_COORD_LABEL)
        for key in (
            "algorithm",
            "family",
            "beta",
            "beta_min",
            "beta_decay",
            "reward_mode",
            "training_objective",
        ):
            self.assertEqual(daoc.get(key), coordinated.get(key), key)
        self.assertEqual(coordinated["cache_policy"], "critical_path_joint")
        self.assertTrue(coordinated["cache_server_quality"])
        self.assertTrue(coordinated["cache_coverage_constraint"])
        self.assertTrue(coordinated["cache_dependency_awareness"])

    def test_training_profile_matches_daoc(self):
        daoc = reproduction.effective_method_profile(
            reproduction.PROFILES["pegasus_paper_closure_converged"],
            DAOC_LABEL,
        )
        coordinated = reproduction.effective_method_profile(
            reproduction.PROFILES[PROFILE_CONVERGED],
            DAOC_COORD_LABEL,
        )
        daoc.pop("labels", None)
        coordinated.pop("labels", None)
        self.assertEqual(daoc, coordinated)


if __name__ == "__main__":
    unittest.main()
