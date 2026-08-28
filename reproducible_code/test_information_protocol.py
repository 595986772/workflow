import json
import tempfile
import unittest
from pathlib import Path

from aggregate_reproduction_results import (
    comparison_pairs,
    load_runs,
)
from information_protocol import INFORMATION_PROTOCOL_VERSION
from run_reproduction_suite import (
    ALGORITHMS,
    LEAN_OUR_LABEL,
    PROFILES,
    effective_method_profile,
    is_complete,
)
from run_strict_environment_suite import STAGES


class InformationProtocolTest(unittest.TestCase):
    def test_resume_rejects_pre_fairness_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "eligible_for_comparison": True,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(is_complete(run_dir))

    def test_resume_accepts_current_protocol_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "eligible_for_comparison": True,
                        "information_protocol_version": (
                            INFORMATION_PROTOCOL_VERSION
                        ),
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                is_complete(
                    run_dir,
                    require_convergence=True,
                )
            )

    def test_aggregator_refuses_pre_fairness_run(self):
        with tempfile.TemporaryDirectory() as directory:
            suite_dir = Path(directory)
            run_dir = suite_dir / "runs" / "our" / "seed_1"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "pre-fairness run",
            ):
                load_runs(suite_dir)

    def test_strict_converged_profile_keeps_native_optimizers(self):
        profile = PROFILES["strict_stress_converged_3seed"]
        daoc = effective_method_profile(profile, "guided_full")
        ours = effective_method_profile(
            profile,
            LEAN_OUR_LABEL,
        )
        self.assertEqual(daoc["batch_size"], 1024)
        self.assertEqual(daoc["max_explore"], 20000)
        self.assertEqual(
            daoc["learning_rate_schedule"],
            "cosine",
        )
        self.assertEqual(ours["batch_size"], 64)
        self.assertEqual(ours["max_explore"], 150)
        self.assertEqual(
            ours["learning_rate_schedule"],
            "constant",
        )
        self.assertEqual(daoc["checkpoint_every"], 1000)
        self.assertEqual(daoc["convergence_min_episodes"], 15000)
        self.assertEqual(ours["checkpoint_every"], 500)
        self.assertEqual(ours["convergence_min_episodes"], 2000)
        self.assertEqual(daoc["train_episodes"], 50000)
        self.assertEqual(ours["train_episodes"], 40000)
        for key in (
            "eval_episodes",
            "validation_scenarios",
            "convergence_window",
            "convergence_patience",
            "convergence_relative_mean_change",
            "convergence_relative_slope",
        ):
            self.assertEqual(daoc[key], ours[key])

    def test_strict_converged_stage_selects_converged_profile(self):
        stage = STAGES["converged"]
        self.assertEqual(
            stage["profile"],
            "strict_stress_converged_3seed",
        )
        self.assertEqual(stage["seeds"], [1, 2, 3])
        self.assertEqual(stage["eval_episodes"], 100)

    def test_strict_our_is_paired_against_daoc(self):
        labels = ["guided_full", LEAN_OUR_LABEL]
        values = {
            "guided_full": {1: 2.0},
            LEAN_OUR_LABEL: {1: 1.5},
        }
        self.assertIn(
            (LEAN_OUR_LABEL, "guided_full"),
            comparison_pairs(labels, values),
        )

    def test_lean_our_excludes_failed_modules(self):
        config = next(
            item
            for item in ALGORITHMS
            if item["label"] == LEAN_OUR_LABEL
        )
        self.assertEqual(
            config["algorithm"],
            "causal_telemetryPD3QN",
        )
        self.assertEqual(
            config["reward_mode"],
            "causal_makespan_increment",
        )
        self.assertEqual(config["priority_alpha"], 0.0)
        self.assertEqual(config["priority_beta_start"], 0.0)
        self.assertEqual(config["criticality_boost"], 0.0)
        self.assertEqual(config["num_quantiles"], 1)
        self.assertEqual(config["entropy_coefficient"], 0.0)
        self.assertEqual(config["potential_reward_weight"], 0.0)
        self.assertEqual(
            config["active_modules"],
            [
                "pairwise_dueling_double_dqn",
                "causal_history_telemetry",
                "causal_dependency_aware_joint_cache",
                "scarcity_aware_service_coverage_constraint",
            ],
        )
        self.assertTrue(config["cache_coverage_constraint"])
        self.assertEqual(
            config["training_objective"],
            "undiscounted_makespan",
        )
        self.assertIn(
            "hindsight_critical_path_replay",
            config["excluded_modules"],
        )
        self.assertIn(
            "bottleneck_contribution_replay",
            config["excluded_modules"],
        )


if __name__ == "__main__":
    unittest.main()
