import csv
import json
from pathlib import Path
import tempfile
import unittest

from analyze_e2_e3_results import compare_revision
from audit_e3_gate_feasibility import build_feasibility
from audit_e0_compatibility import build_audit
from run_e2_e3_suite import (
    E2_CAPACITIES,
    STAGES,
    frozen_protocol_spec,
    link_dynamic_metadata,
    link_reused_runs,
)


class E2E3GovernanceTest(unittest.TestCase):
    def _write_run(self, root, seed, finish, fingerprint):
        run_dir = (
            Path(root)
            / "runs"
            / "lean_our"
            / f"seed_{seed}"
        )
        run_dir.mkdir(parents=True)
        with (run_dir / "episodes.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=(
                    "phase",
                    "scenario_fingerprint",
                    "average_finish_time",
                    "p95_finish_time",
                    "cache_remote_loading_rate",
                    "cache_service_coverage",
                    "cache_migration_time_sec",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "phase": "eval",
                    "scenario_fingerprint": fingerprint,
                    "average_finish_time": finish,
                    "p95_finish_time": 1.2 * finish,
                    "cache_remote_loading_rate": 0.5,
                    "cache_service_coverage": 0.8,
                    "cache_migration_time_sec": 0.1,
                }
            )
        with (
            run_dir / "online_stream_e3_load_shift.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=("scenario_fingerprint",),
            )
            writer.writeheader()
            writer.writerow(
                {"scenario_fingerprint": f"e3-{fingerprint}"}
            )
        (run_dir / "online_stream_e3_load_shift_summary.json").write_text(
            json.dumps(
                {
                    "adaptation": {
                        "cumulative_oracle_regret": finish,
                        "adaptation_delay_windows": 2,
                    },
                    "protocol": {
                        "episodes": 100,
                        "shift_episode": 51,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_frozen_protocol_captures_environment_and_budgets(self):
        protocol = frozen_protocol_spec()
        self.assertEqual(
            protocol["environment"]["capacity_multiset"],
            E2_CAPACITIES,
        )
        self.assertEqual(
            protocol["training_and_evaluation"]["guided_full"][
                "train_episodes"
            ],
            50000,
        )
        self.assertEqual(
            protocol["training_and_evaluation"]["lean_our"][
                "train_episodes"
            ],
            40000,
        )
        self.assertEqual(
            protocol["stage_protocol"]["final_seeds"],
            list(range(1, 11)),
        )
        self.assertFalse(
            protocol["stage_protocol"][
                "final_is_independent_holdout"
            ]
        )

    def test_public_stages_match_revised_protocol(self):
        self.assertEqual(
            list(STAGES),
            [
                "smoke",
                "screen",
                "converged",
                "dynamic",
                "ablation",
                "final",
                "e0_audit",
            ],
        )
        self.assertEqual(STAGES["ablation"]["seeds"], [1, 2, 3])
        self.assertEqual(
            STAGES["final"]["seeds"],
            list(range(1, 11)),
        )
        self.assertFalse(STAGES["dynamic"]["run_e2"])

    def test_reused_runs_are_links_to_frozen_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (
                root / "source" / "runs" / "lean_our" / "seed_1"
            )
            source.mkdir(parents=True)
            (source / "summary.json").write_text(
                "{}",
                encoding="utf-8",
            )
            target = root / "target"
            linked = link_reused_runs(
                root / "source",
                target,
                ["lean_our"],
                [1],
            )
            target_run = (
                target / "runs" / "lean_our" / "seed_1"
            )
            self.assertTrue(target_run.is_symlink())
            self.assertEqual(target_run.resolve(), source.resolve())
            self.assertEqual(len(linked), 1)

    def test_dynamic_metadata_reuses_manifest_and_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "suite_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (source / "oracle").mkdir()
            linked = link_dynamic_metadata(source, target)
            self.assertTrue(
                (target / "suite_manifest.json").is_symlink()
            )
            self.assertEqual(
                (target / "suite_manifest.json").resolve(),
                (source / "suite_manifest.json").resolve(),
            )
            self.assertTrue((target / "oracle").is_symlink())
            self.assertIsNotNone(linked["oracle"])

    def test_strict_recovery_gate_detects_zero_delay_ceiling(self):
        rows = [
            {
                "seed": seed,
                "daoc_adaptation_delay": daoc,
                "our_adaptation_delay": ours,
            }
            for seed, daoc, ours in (
                (1, 0, 2),
                (2, 0, 1),
                (3, 7, 0),
            )
        ]
        result = build_feasibility(rows)
        self.assertEqual(result["required_strict_wins"], 2)
        self.assertEqual(
            result["maximum_possible_strict_wins"],
            1,
        )
        self.assertFalse(
            result["strict_recovery_gate_attainable"]
        )
        self.assertEqual(
            result["protocol_decision"],
            "stop_without_algorithm_revision",
        )

    def test_e0_audit_rejects_historical_reward_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = build_audit(Path(temporary))
            self.assertFalse(audit["behavior_equivalent"])
            self.assertFalse(audit["groups"]["reward"]["equal"])
            self.assertEqual(
                audit["decision"],
                "run_new_e0_daoc_vs_lean_our",
            )

    def test_revision_gate_uses_paired_seeds_and_e2_safeguard(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            candidate = Path(temporary) / "candidate"
            for seed in (1, 2, 3):
                self._write_run(parent, seed, 1.0, f"e2-{seed}")
                self._write_run(
                    candidate,
                    seed,
                    0.95,
                    f"e2-{seed}",
                )
            result = compare_revision(
                suite_dir=candidate,
                parent_suite_dir=parent,
                seeds=[1, 2, 3],
                our_label="lean_our",
                expected_metric="paired_finish_time",
            )
            self.assertTrue(result["all_scenarios_paired"])
            self.assertEqual(
                result["comparisons"]["average_finish_time"]["wins"],
                3,
            )
            self.assertTrue(result["retained"])


if __name__ == "__main__":
    unittest.main()
