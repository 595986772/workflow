import unittest

from analyze_pegasus_pscale_sensitivity import workload_view
from pegasus_pscale_protocol import CAPACITY_PROFILES
from run_pegasus_pscale_sensitivity import (
    ALL_PROFILES,
    CAPACITY_NAMESPACE,
    SENSITIVITY_PROFILES,
    reproduction_command,
)


class PegasusPScaleSensitivityTests(unittest.TestCase):
    def test_budget_profiles_are_fixed(self):
        self.assertEqual(ALL_PROFILES, ("B5", "B8", "B10"))
        self.assertEqual(SENSITIVITY_PROFILES, ("B5", "B10"))
        for profile in ALL_PROFILES:
            capacities = CAPACITY_PROFILES[profile]
            self.assertEqual(len(capacities), 10)
            self.assertEqual(sum(capacities), int(profile[1:]))

    def test_reproduction_command_changes_capacity_only(self):
        for profile in SENSITIVITY_PROFILES:
            command = reproduction_command(profile, workers=2, resume=False)
            capacity_index = command.index("--server-capacity-multiset") + 1
            namespace_index = (
                command.index("--capacity-assignment-namespace") + 1
            )
            self.assertEqual(
                command[capacity_index],
                ",".join(str(value) for value in CAPACITY_PROFILES[profile]),
            )
            self.assertEqual(command[namespace_index], CAPACITY_NAMESPACE)
            self.assertIn("pegasus_pscale_p2_converged", command)
            self.assertNotIn("--pretrained-checkpoint", command)

    def test_cross_budget_workload_view_ignores_capacity_fingerprint(self):
        base = {
            "episode": 1,
            "seed": 101,
            "workflow_family": "Montage",
            "user_initial_positions": {"0": [1.0, 2.0]},
            "user_graph_keys": {"0": "pegasus_full_montage"},
        }
        first = {**base, "base_fingerprint": "budget-five"}
        second = {**base, "base_fingerprint": "budget-ten"}
        self.assertEqual(workload_view([first]), workload_view([second]))


if __name__ == "__main__":
    unittest.main()
