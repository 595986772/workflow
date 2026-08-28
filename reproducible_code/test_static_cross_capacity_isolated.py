import unittest
from pathlib import Path

from evaluate_static_cross_capacity_isolated import (
    build_specs,
    worker_command,
)


class StaticCrossCapacityIsolationTest(unittest.TestCase):
    def test_build_specs_has_one_unique_run_per_tuple(self):
        specs = build_specs(
            labels=["guided_full", "lean_our"],
            seeds=[11, 12],
            profiles=["H0", "H3"],
        )
        self.assertEqual(len(specs), 8)
        self.assertEqual(len(set(specs)), 8)
        self.assertEqual(
            specs[0],
            ("H0", "guided_full", 11),
        )

    def test_worker_command_contains_exactly_one_run(self):
        command = worker_command(
            script=Path("/tmp/evaluator.py"),
            source_suite=Path("/tmp/source"),
            output_dir=Path("/tmp/output"),
            label="lean_our",
            seed=14,
            profile="H2",
            episodes=100,
            resume=True,
        )
        self.assertIn("--worker", command)
        self.assertEqual(command.count("--label"), 1)
        self.assertEqual(command.count("--seed"), 1)
        self.assertEqual(command.count("--profile"), 1)
        self.assertIn("lean_our", command)
        self.assertIn("14", command)
        self.assertIn("H2", command)
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
