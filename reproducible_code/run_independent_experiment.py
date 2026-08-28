#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from convergence_monitor import ConvergenceMonitor
from critical_path_rl import (
    BCR_CP_RL_ALGORITHMS,
    CP_RL_ALGORITHMS,
    HCPR_CP_RL_ALGORITHMS,
    NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS,
    RAW_TELEMETRY_CP_RL_ALGORITHMS,
)
from information_protocol import INFORMATION_PROTOCOL_VERSION
from input import INPUT_DICT, learning_arg
from simulator import MEC_Simulator
from user import DAG_COMPLETION_PROTOCOL_VERSION
from capacity_protocol import (
    CAPACITY_PROTOCOL_VERSION,
    parse_capacity_multiset,
    validate_capacity_multiset,
)


SCENARIO_FINGERPRINT_VERSION = "full_dag_workload_v2"


EPISODE_FIELDS = [
    "label",
    "algorithm",
    "seed",
    "reward_mode",
    "cache_policy",
    "cache_information_regime",
    "scenario_seed",
    "scenario_fingerprint",
    "base_scenario_fingerprint",
    "phase",
    "episode",
    "real_task_count",
    "completed_task_count",
    "all_tasks_executed_once",
    "average_finish_time",
    "median_finish_time",
    "p95_finish_time",
    "min_finish_time",
    "max_finish_time",
    "std_finish_time",
    "computing_latency",
    "data_transfer_latency",
    "predecessor_latency",
    "service_latency",
    "waiting_latency",
    "computing_share",
    "data_transfer_share",
    "predecessor_share",
    "service_share",
    "waiting_share",
    "cache_hits",
    "cache_misses",
    "cache_hit_rate",
    "cache_replacements",
    "cache_update_events",
    "cache_decision_calls",
    "cache_decision_wall_time_sec",
    "policy_inference_calls",
    "policy_inference_wall_time_sec",
    "policy_inference_time_per_decision_ms",
    "cache_migration_events",
    "cache_migration_time_sec",
    "cache_migration_critical_time_sec",
    "cache_remote_loading_rate",
    "cache_zero_capacity_remote_loads",
    "cache_zero_capacity_assignment_rate",
    "cache_capacity_used",
    "cache_capacity_total",
    "cache_capacity_utilization",
    "cache_service_coverage",
    "cache_matrix_json",
    "cache_capacity_vector_json",
    "cache_replica_counts_json",
    "cache_server_quality_json",
    "cache_execution_latency_ema_json",
    "server_action_histogram_json",
    "server_cpu_cycle_histogram_json",
    "cache_history_windows",
    "cache_expected_requests_ema",
    "cache_mean_cpu_cycles_ema",
    "cache_global_execution_latency_ema",
    "cache_global_compute_per_mcycle_ema",
    "cache_global_waiting_latency_ema",
    "cache_mean_telemetry_confidence",
    "cache_updates_enabled",
    "beta",
    "mean_epsilon",
    "mean_learning_rate",
    "decision_count",
    "policy_actions",
    "cache_guidance_actions",
    "feedback_guidance_actions",
    "adaptive_cache_actions",
    "guidance_ready_count",
    "history_update_count",
    "feedback_guidance_rate",
    "adaptive_cache_rate",
    "cache_guidance_rate",
    "mean_q_confidence",
    "mean_expert_confidence",
    "mean_guidance_probability",
    "mean_episode_return",
    "max_reward_identity_error",
    "mean_potential_initial",
    "mean_potential_final",
    "hcpr_exact_path_rate",
    "hcpr_mean_posterior_criticality",
    "hcpr_selected_rate",
    "hcpr_replay_beta",
    "hcpr_buffer_criticality",
    "hcpr_sampled_criticality",
    "hcpr_sampling_criticality_lift",
    "hcpr_importance_weight_mean",
    "episode_wall_time_sec",
]


def method_module_manifest(args):
    active = []
    if args.algorithm.endswith("PD3QN"):
        active.append("pairwise_dueling_double_dqn")
    elif (
        args.algorithm in CP_RL_ALGORITHMS
        and args.algorithm.endswith("DDQN")
    ):
        active.append("flat_double_dqn")
    if args.algorithm.endswith("DiscreteSAC"):
        active.extend(
            [
                "pairwise_categorical_discrete_sac",
                "automatic_entropy_temperature",
            ]
        )
    if args.algorithm in RAW_TELEMETRY_CP_RL_ALGORITHMS:
        active.append("causal_history_telemetry")
    if args.algorithm in NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS:
        active.append("workload_normalized_telemetry")
    if args.algorithm in HCPR_CP_RL_ALGORITHMS:
        active.append("hindsight_critical_path_replay")
    if args.algorithm in BCR_CP_RL_ALGORITHMS:
        active.append("bottleneck_contribution_replay")
    if args.algorithm in CP_RL_ALGORITHMS:
        if args.task_dependency_features:
            active.append("causal_dag_state_features")
        else:
            active.append("dag_state_features_removed")
    if args.cache_policy == "critical_path_joint":
        if args.cache_dependency_awareness:
            active.append("causal_dependency_aware_joint_cache")
        else:
            active.append("coordinated_popularity_joint_cache")
    elif args.cache_policy == "popularity_coordinated":
        active.append("centralized_popularity_coverage_cache")
    elif args.cache_policy == "paper_popularity_cost_ema":
        active.append("daoc_paper_popularity_cost_cache")
    elif args.cache_policy == "popularity_ema":
        active.append("independent_popularity_ema_cache")
    if args.cache_coverage_constraint:
        active.append("scarcity_aware_service_coverage_constraint")
    objective = args.reward_mode
    if args.reward_mode == "causal_makespan_increment":
        objective = "undiscounted_makespan"
    elif args.reward_mode == "causal_critical_path":
        objective = "undiscounted_makespan_legacy_name"

    excluded = []
    if args.label == "lean_our":
        excluded = [
            "hindsight_critical_path_replay",
            "bottleneck_contribution_replay",
            "workload_normalized_telemetry",
            "quantile_risk_head",
            "entropy_regularization",
            "historical_feedback_guidance",
            "adaptive_guidance_gate",
        ]
    return {
        "active": active,
        "objective": objective,
        "excluded": excluded,
    }


def parse_hidden_units(value):
    units = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not units:
        raise argparse.ArgumentTypeError(
            "--hidden-units must contain at least one integer"
        )
    return units


def parse_server_capacity_multiset(value):
    try:
        return parse_capacity_multiset(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_string_list(value):
    values = tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected at least one comma-separated value"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Values must be unique")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one isolated training and evaluation experiment."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--num-users", type=int, default=5)
    parser.add_argument("--num-servers", type=int, default=5)
    parser.add_argument("--num-services", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument(
        "--dag-dataset-path",
        help=(
            "DAOC-compatible node-link JSON dataset. Relative paths are "
            "resolved from the repository root."
        ),
    )
    parser.add_argument(
        "--dag-dataset-sha256",
        help="Optional expected SHA-256 for the DAG dataset.",
    )
    parser.add_argument("--dag-depth-increment", type=int, default=0)
    parser.add_argument(
        "--dependency-data-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--server-capacity", type=int, default=2)
    parser.add_argument(
        "--server-capacity-multiset",
        type=parse_server_capacity_multiset,
        help=(
            "Comma-separated per-server capacities. Values are shuffled "
            "deterministically from an RNG independent of the deployment."
        ),
    )
    parser.add_argument(
        "--capacity-assignment-namespace",
        help=(
            "Optional namespace that applies one shared server-order "
            "permutation across equal-length capacity profiles."
        ),
    )
    parser.add_argument(
        "--baseline-server-capacity",
        type=int,
        default=int(INPUT_DICT.get("baseline server capacity", 2)),
        help=(
            "Number of services sampled before truncating to each "
            "server's active capacity. Keep this fixed across paired "
            "capacity profiles."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-experiences", type=int, default=128)
    parser.add_argument("--filling-steps", type=int, default=100)
    parser.add_argument("--steps-to-updates", type=int, default=20)
    parser.add_argument("--max-explore", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--num-quantiles", type=int, default=16)
    parser.add_argument(
        "--risk-tail-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--sac-target-entropy-ratio",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--sac-target-tau",
        type=float,
        default=0.005,
    )
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument(
        "--priority-beta-start",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--priority-beta-anneal-steps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--criticality-boost",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--hcpr-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--bcr-top-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument(
        "--learning-rate-decay-start",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--learning-rate-decay-end",
        type=int,
        default=20000,
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=parse_hidden_units, default=[64])
    parser.add_argument("--bandwidth", type=float, default=1_000_000)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-decay", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        choices=(
            "terminal_binary",
            "critical_path_potential",
            "terminal_plus_potential",
            "causal_critical_path",
            "causal_makespan_increment",
        ),
        default="terminal_binary",
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--potential-reward-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cache-policy",
        choices=(
            "popularity_ema",
            "paper_popularity_cost_ema",
            "popularity_coordinated",
            "critical_path_hysteresis",
            "critical_path_coordinated",
            "critical_path_joint",
        ),
        default="popularity_ema",
    )
    parser.add_argument("--cache-score-alpha", type=float, default=0.1)
    parser.add_argument(
        "--cache-history-alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--cache-locality-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cache-update-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cache-hysteresis-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cache-min-residence-updates",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cache-freeze-episode",
        type=int,
        default=0,
        help=(
            "Keep the learned cache fixed after this training episode; "
            "zero keeps online cache updates active throughout training."
        ),
    )
    parser.add_argument(
        "--cache-compute-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cache-server-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-coverage-constraint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-dependency-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expose predecessor-server, successor-service and structural "
            "criticality features to causal task-server learners."
        ),
    )
    parser.add_argument(
        "--cache-dependency-awareness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use DAG criticality and predecessor locality when updating "
            "the coordinated cache demand matrix."
        ),
    )
    parser.add_argument(
        "--telemetry-min-samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--telemetry-freshness-half-life",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--eval-scenario-bank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate the frozen checkpoint on independently generated, "
            "paired scenarios."
        ),
    )
    parser.add_argument(
        "--eval-bank-scope",
        choices=("workload", "infrastructure", "full"),
        default="workload",
        help=(
            "Keep the complete trained deployment fixed; keep only the "
            "server infrastructure fixed while resampling users; or "
            "resample the full topology as a stress test."
        ),
    )
    parser.add_argument(
        "--eval-dag-families",
        type=parse_string_list,
        help=(
            "Comma-separated workflow families assigned round-robin "
            "to frozen validation and evaluation scenarios."
        ),
    )
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=1_000_003,
    )
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--validation-scenarios", type=int, default=0)
    parser.add_argument(
        "--validation-seed-offset",
        type=int,
        default=500_003,
    )
    parser.add_argument(
        "--convergence-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop at the final checkpoint of a sustained validation "
            "plateau instead of selecting the historical best checkpoint."
        ),
    )
    parser.add_argument("--convergence-min-episodes", type=int, default=5000)
    parser.add_argument("--convergence-window", type=int, default=5)
    parser.add_argument("--convergence-patience", type=int, default=3)
    parser.add_argument(
        "--convergence-relative-mean-change",
        "--convergence-relative-range",
        dest="convergence_relative_mean_change",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--convergence-relative-slope",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--enable-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-update-caching",
        action="store_true",
        help="Keep the online cache estimator active during evaluation.",
    )
    parser.add_argument(
        "--historical-feedback-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--adaptive-guidance-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--history-feedback-alpha", type=float, default=0.1)
    parser.add_argument(
        "--history-feedback-min-samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--history-feedback-max-probability",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--history-feedback-fixed-probability",
        type=float,
        default=0.1,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--revision-id", default="r0")
    parser.add_argument("--revision-parent")
    parser.add_argument("--revision-reason", default="initial_version")
    parser.add_argument(
        "--revision-changed-module",
        default="initial_version",
    )
    parser.add_argument(
        "--revision-expected-metric",
        default="paired_finish_time",
    )
    parser.add_argument(
        "--revision-rejection-condition",
        default="primary_metric_not_improved",
    )
    parser.add_argument(
        "--seed-partition",
        choices=(
            "smoke",
            "development",
            "confirmation",
            "ablation",
            "holdout",
            "heterogeneity",
            "final",
            "generalization",
            "mechanism",
            "baseline_extension",
        ),
        default="development",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    for name in (
        "train_episodes",
        "eval_episodes",
        "num_users",
        "num_servers",
        "num_services",
        "num_tasks",
        "batch_size",
        "min_experiences",
        "filling_steps",
        "steps_to_updates",
        "max_explore",
        "n_step",
        "num_quantiles",
        "priority_beta_anneal_steps",
        "history_feedback_min_samples",
        "telemetry_min_samples",
        "torch_threads",
        "convergence_min_episodes",
        "convergence_window",
        "convergence_patience",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(unit < 1 for unit in args.hidden_units):
        raise ValueError("--hidden-units values must be positive")
    if args.dag_dataset_path is not None:
        dag_dataset_path = Path(args.dag_dataset_path).expanduser()
        if not dag_dataset_path.is_absolute():
            dag_dataset_path = (
                Path(__file__).resolve().parent / dag_dataset_path
            )
        dag_dataset_path = dag_dataset_path.resolve()
        if not dag_dataset_path.is_file():
            raise ValueError(
                f"--dag-dataset-path does not exist: {dag_dataset_path}"
            )
        args.dag_dataset_path = str(dag_dataset_path)
    if args.dag_dataset_sha256 is not None:
        args.dag_dataset_sha256 = args.dag_dataset_sha256.lower()
        if (
            len(args.dag_dataset_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.dag_dataset_sha256
            )
        ):
            raise ValueError(
                "--dag-dataset-sha256 must contain 64 hexadecimal digits"
            )
    if args.dag_depth_increment < 0:
        raise ValueError(
            "--dag-depth-increment must be non-negative"
        )
    if args.dependency_data_scale < 1.0:
        raise ValueError(
            "--dependency-data-scale must be at least one"
        )
    if not 0 <= args.server_capacity <= args.num_services:
        raise ValueError(
            "--server-capacity must be between zero and "
            "--num-services"
        )
    if args.server_capacity_multiset is not None:
        capacities = validate_capacity_multiset(
            args.server_capacity_multiset,
            args.num_servers,
            args.num_services,
        )
    else:
        capacities = (args.server_capacity,)
    if not 0 <= args.baseline_server_capacity <= args.num_services:
        raise ValueError(
            "--baseline-server-capacity must be between zero and "
            "--num-services"
        )
    if max(capacities, default=0) > args.baseline_server_capacity:
        raise ValueError(
            "active server capacity cannot exceed "
            "--baseline-server-capacity"
        )
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0 < args.risk_tail_fraction <= 1:
        raise ValueError("--risk-tail-fraction must be in (0, 1]")
    if args.entropy_coefficient < 0:
        raise ValueError("--entropy-coefficient must be non-negative")
    if not 0 < args.sac_target_entropy_ratio <= 1:
        raise ValueError(
            "--sac-target-entropy-ratio must be in (0, 1]"
        )
    if not 0 < args.sac_target_tau <= 1:
        raise ValueError("--sac-target-tau must be in (0, 1]")
    if args.priority_alpha < 0:
        raise ValueError("--priority-alpha must be non-negative")
    if not 0 <= args.priority_beta_start <= 1:
        raise ValueError(
            "--priority-beta-start must be between zero and one"
        )
    if args.criticality_boost < 0:
        raise ValueError("--criticality-boost must be non-negative")
    if args.hcpr_temperature <= 0:
        raise ValueError("--hcpr-temperature must be positive")
    if not 0 < args.bcr_top_fraction <= 1:
        raise ValueError("--bcr-top-fraction must be in (0, 1]")
    if not 0 < args.epsilon <= 1:
        raise ValueError("--epsilon must be in (0, 1]")
    if args.bandwidth <= 0:
        raise ValueError("--bandwidth must be positive")
    if args.telemetry_freshness_half_life <= 0:
        raise ValueError(
            "--telemetry-freshness-half-life must be positive"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.min_learning_rate <= 0:
        raise ValueError("--min-learning-rate must be positive")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError(
            "--min-learning-rate cannot exceed --learning-rate"
        )
    if args.learning_rate_decay_start < 0:
        raise ValueError(
            "--learning-rate-decay-start must be non-negative"
        )
    if (
        args.learning_rate_decay_end
        <= args.learning_rate_decay_start
    ):
        raise ValueError(
            "--learning-rate-decay-end must exceed "
            "--learning-rate-decay-start"
        )
    if not 0 <= args.beta <= 1:
        raise ValueError("--beta must be between 0 and 1")
    if not 0 <= args.beta_min <= args.beta:
        raise ValueError("--beta-min must be between 0 and --beta")
    if not 0 < args.beta_decay <= 1:
        raise ValueError("--beta-decay must be in (0, 1]")
    if not 0 < args.history_feedback_alpha <= 1:
        raise ValueError("--history-feedback-alpha must be in (0, 1]")
    if args.eval_seed_offset < 1:
        raise ValueError("--eval-seed-offset must be positive")
    if args.eval_dag_families is not None:
        family_count = len(args.eval_dag_families)
        if args.eval_episodes % family_count:
            raise ValueError(
                "--eval-episodes must be divisible by the number of "
                "--eval-dag-families"
            )
        if (
            args.validation_scenarios
            and args.validation_scenarios % family_count
        ):
            raise ValueError(
                "--validation-scenarios must be divisible by the "
                "number of --eval-dag-families"
            )
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.potential_reward_weight < 0:
        raise ValueError(
            "--potential-reward-weight must be non-negative"
        )
    if not 0 < args.cache_score_alpha <= 1:
        raise ValueError(
            "--cache-score-alpha must be in (0, 1]"
        )
    if not 0 < args.cache_history_alpha <= 1:
        raise ValueError(
            "--cache-history-alpha must be in (0, 1]"
        )
    if args.cache_locality_weight < 0:
        raise ValueError(
            "--cache-locality-weight must be non-negative"
        )
    if args.cache_update_interval < 1:
        raise ValueError(
            "--cache-update-interval must be positive"
        )
    if args.cache_hysteresis_factor < 0:
        raise ValueError(
            "--cache-hysteresis-factor must be non-negative"
        )
    if args.cache_min_residence_updates < 0:
        raise ValueError(
            "--cache-min-residence-updates must be non-negative"
        )
    if args.cache_freeze_episode < 0:
        raise ValueError(
            "--cache-freeze-episode must be non-negative"
        )
    if args.cache_compute_weight < 0:
        raise ValueError(
            "--cache-compute-weight must be non-negative"
        )
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.validation_scenarios < 0:
        raise ValueError(
            "--validation-scenarios must be non-negative"
        )
    if args.validation_seed_offset < 1:
        raise ValueError(
            "--validation-seed-offset must be positive"
        )
    if bool(args.checkpoint_every) != bool(
        args.validation_scenarios
    ):
        raise ValueError(
            "--checkpoint-every and --validation-scenarios "
            "must be enabled together"
        )
    if args.validation_seed_offset == args.eval_seed_offset:
        raise ValueError(
            "validation and evaluation seed offsets must differ"
        )
    if args.checkpoint_every and not args.eval_scenario_bank:
        raise ValueError(
            "checkpoint selection requires --eval-scenario-bank"
        )
    if args.convergence_mode and not args.checkpoint_every:
        raise ValueError(
            "convergence mode requires checkpoint validation"
        )
    if (
        args.convergence_mode
        and args.train_episodes < args.convergence_min_episodes
    ):
        raise ValueError(
            "--train-episodes must be at least "
            "--convergence-min-episodes"
        )
    if args.convergence_window < 2:
        raise ValueError("--convergence-window must be at least two")
    if args.convergence_relative_mean_change < 0:
        raise ValueError(
            "--convergence-relative-mean-change must be non-negative"
        )
    if args.convergence_relative_slope < 0:
        raise ValueError(
            "--convergence-relative-slope must be non-negative"
        )
    if not 0 <= args.history_feedback_max_probability <= 1:
        raise ValueError(
            "--history-feedback-max-probability must be between 0 and 1"
        )
    if not 0 <= args.history_feedback_fixed_probability <= 1:
        raise ValueError(
            "--history-feedback-fixed-probability must be between 0 and 1"
        )
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def cache_snapshot(simulator):
    hits = sum(
        server.statistic["service required counter"][0]
        for server in simulator.servers.values()
    )
    misses = sum(
        server.statistic["service required counter"][1]
        for server in simulator.servers.values()
    )
    return {
        "hits": hits,
        "misses": misses,
        "replacements": simulator.broker.cache_replacements,
        "update_events": simulator.broker.cache_update_events,
        "decision_calls": simulator.broker.cache_decision_calls,
        "decision_wall_time_sec": (
            simulator.broker.cache_decision_wall_time_sec
        ),
        "policy_inference_calls": sum(
            server.policy_inference_calls
            for server in simulator.servers.values()
        ),
        "policy_inference_wall_time_sec": sum(
            server.policy_inference_wall_time_sec
            for server in simulator.servers.values()
        ),
        "migration_events": (
            simulator.broker.cache_migration_events
        ),
        "migration_time_sec": (
            simulator.broker.cache_migration_time_sec
        ),
        "migration_critical_time_sec": (
            simulator.broker.cache_migration_critical_time_sec
        ),
        "zero_capacity_remote_loads": sum(
            server.statistic["service required counter"][1]
            for server in simulator.servers.values()
            if server.capacity == 0
        ),
        "cache_matrix": {
            server_id: tuple(
                service_id
                for service_id in server.services
                if service_id > 0
            )
            for server_id, server in simulator.servers.items()
        },
    }


def scenario_snapshot(simulator):
    snapshot = {
        "algorithm": simulator.alg,
        "dag_completion_protocol_version": (
            DAG_COMPLETION_PROTOCOL_VERSION
        ),
        "users": {
            str(user_id): {
                "initial_position": [float(value) for value in user.pos0],
                "direction": simulator.user_directions[user_id],
                "nearest_server": int(user.nearest_server),
                "dag_key": simulator.user_graph_keys[user_id],
                "dag_family": simulator.loaded_graphs[
                    simulator.user_graph_keys[user_id]
                ].graph.get("source_family"),
                "task_count": int(user.numberoftasks),
            }
            for user_id, user in simulator.users.items()
        },
        "servers": {
            str(server_id): {
                "position": [float(value) for value in server.pos],
                "frequency": float(server.frequency),
                "load": float(server.load),
                "rate_to_cloud": float(server.rate_to_cloud),
                "cached_services": [int(value) for value in server.services],
                **(
                    {"cache_capacity": int(server.capacity)}
                    if simulator.heterogeneous_capacity
                    else {}
                ),
            }
            for server_id, server in simulator.servers.items()
        },
        "service_data_length": {
            str(service_id): float(length)
            for service_id, length in simulator.service_data_length.items()
        },
        "between_server_costs": simulator.between_server_costs.tolist(),
    }
    stress_active = (
        simulator.dag_depth_increment > 0
        or simulator.dependency_data_scale > 1.0
        or simulator.server_capacity
        != simulator.baseline_server_capacity
        or simulator.heterogeneous_capacity
    )
    if stress_active:
        snapshot["environment_stress"] = {
            "dag_depth_increment": simulator.dag_depth_increment,
            "dependency_data_scale": (
                simulator.dependency_data_scale
            ),
            "server_capacity": simulator.server_capacity,
            "server_capacities": {
                str(server_id): int(capacity)
                for server_id, capacity
                in simulator.server_capacities.items()
            },
            "total_server_capacity": (
                simulator.total_server_capacity
            ),
            "capacity_protocol_version": (
                CAPACITY_PROTOCOL_VERSION
            ),
            "baseline_server_capacity": (
                simulator.baseline_server_capacity
            ),
            "task_workloads": {
                str(user_id): {
                    str(task_id): {
                        "service": int(task.service),
                        "cpu_cycle": float(task.cpu_cycle),
                        "input_data_length": float(
                            task.input_data_length
                        ),
                        "predecessors": [
                            str(predecessor)
                            for predecessor in task.predecessors
                        ],
                        "successors": [
                            str(successor)
                            for successor in task.successors
                        ],
                        "output_lengths": {
                            str(successor): float(length)
                            for successor, length
                            in task.outputs_length.items()
                        },
                    }
                    for task_id, task in user.tasks_init.items()
                }
                for user_id, user in simulator.users.items()
            },
        }
        for user_id, metadata in (
            simulator.user_graph_stress.items()
        ):
            snapshot["users"][str(user_id)]["dag_stress"] = (
                metadata.to_dict()
            )
    if not simulator.dag_dataset_is_default:
        snapshot["dag_dataset"] = {
            "sha256": simulator.dag_dataset_sha256,
            "graph_count": simulator.dag_dataset_graph_count,
            "eligible_graph_count": (
                simulator.dag_dataset_eligible_graph_count
            ),
            "selected_family": simulator.application_graph_family,
        }
    if simulator.dynamic_queueing:
        snapshot["dynamic_arrivals"] = {
            "queueing_enabled": True,
            "arrival_times": {
                str(user_id): float(user.arrival_time)
                for user_id, user in simulator.users.items()
            },
        }
    return snapshot


def scenario_fingerprint(snapshot):
    scenario = copy.deepcopy(snapshot)
    scenario.pop("algorithm", None)
    payload = json.dumps(
        scenario,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def base_scenario_snapshot(simulator):
    snapshot = scenario_snapshot(simulator)
    snapshot.pop("environment_stress", None)
    for user in snapshot["users"].values():
        user.pop("dag_stress", None)
    for server in snapshot["servers"].values():
        server.pop("cached_services", None)
        server.pop("cache_capacity", None)
    return snapshot


def base_scenario_fingerprint(simulator):
    return scenario_fingerprint(base_scenario_snapshot(simulator))


def capture_deployment_state(simulator):
    return {
        "servers": {
            server_id: {
                "position": tuple(server.pos),
                "frequency": float(server.frequency),
                "load": float(server.load),
                "rate_to_cloud": float(server.rate_to_cloud),
                "capacity": int(server.capacity),
            }
            for server_id, server in simulator.servers.items()
        },
        "users": {
            user_id: {
                "position": tuple(user.pos0),
                "direction": simulator.user_directions[user_id],
            }
            for user_id, user in simulator.users.items()
        },
        "service_data_length": copy.deepcopy(
            simulator.service_data_length
        ),
        "between_server_costs": (
            simulator.between_server_costs.copy()
        ),
    }


def apply_deployment_state(
    simulator,
    deployment_state,
    include_users=True,
):
    for server_id, state in deployment_state["servers"].items():
        server = simulator.servers[server_id]
        server.pos = tuple(state["position"])
        server.frequency = state["frequency"]
        server.load = state["load"]
        server.rate_to_cloud = state["rate_to_cloud"]
        if "capacity" in state:
            server.capacity = int(state["capacity"])
            simulator.server_capacities[server_id] = (
                server.capacity
            )
            server.services = (
                [0]
                + [
                    service_id
                    for service_id in server.services
                    if service_id > 0
                ][:server.capacity]
            )

    simulator.total_server_capacity = sum(
        server.capacity
        for server in simulator.servers.values()
    )

    simulator.service_data_length = copy.deepcopy(
        deployment_state["service_data_length"]
    )
    simulator.between_server_costs = deployment_state[
        "between_server_costs"
    ].copy()
    if include_users:
        for user_id, state in deployment_state["users"].items():
            user = simulator.users[user_id]
            user.pos0 = tuple(state["position"])
            user.pos = tuple(state["position"])
            simulator.user_directions[user_id] = state["direction"]

    # Server positions may have changed after users were constructed, so the
    # ingress server must be recomputed for both deployment scopes.
    for user in simulator.users.values():
        user.nearest_server = user.find_nearest_server(
            simulator.servers
        )


def mean_epsilon(simulator):
    if not simulator.learning_enabled:
        return None
    return float(
        np.mean(
            [
                server.agent.agent.epsilon
                for server in simulator.servers.values()
            ]
        )
    )


def mean_learning_rate(simulator):
    if not simulator.learning_enabled:
        return None
    values = [
        group["lr"]
        for server in simulator.servers.values()
        for group in server.agent.agent.TrainNet.optimizer.param_groups
    ]
    return float(np.mean(values))


def scheduled_learning_rate(args, episode):
    if args.learning_rate_schedule == "constant":
        return args.learning_rate
    if episode <= args.learning_rate_decay_start:
        return args.learning_rate
    if episode >= args.learning_rate_decay_end:
        return args.min_learning_rate

    progress = (
        episode - args.learning_rate_decay_start
    ) / (
        args.learning_rate_decay_end
        - args.learning_rate_decay_start
    )
    cosine_weight = 0.5 * (1.0 + np.cos(np.pi * progress))
    return (
        args.min_learning_rate
        + (args.learning_rate - args.min_learning_rate)
        * cosine_weight
    )


def set_learning_rate(simulator, learning_rate):
    for server in simulator.servers.values():
        optimizer = server.agent.agent.TrainNet.optimizer
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def collect_innovation_metrics(simulator):
    users = list(simulator.users.values())
    decisions = sum(user.episode_decision_count for user in users)
    q_count = sum(user.episode_q_confidence_count for user in users)
    expert_count = sum(
        user.episode_expert_confidence_count for user in users
    )
    guidance_ready = sum(
        user.episode_guidance_ready_count for user in users
    )
    feedback_actions = sum(
        user.episode_feedback_guidance_actions for user in users
    )
    adaptive_cache_actions = sum(
        user.episode_adaptive_cache_actions for user in users
    )
    cache_actions = sum(
        user.episode_cache_guidance_actions for user in users
    )
    hcpr_total_tasks = sum(
        user.episode_hcpr_total_tasks for user in users
    )
    hcpr_learners = [
        server.agent.agent.TrainNet
        for server in simulator.servers.values()
        if getattr(
            server.agent.agent.TrainNet,
            "prioritized_replay",
            False,
        )
    ]
    return {
        "decision_count": decisions,
        "policy_actions": sum(
            user.episode_policy_actions for user in users
        ),
        "cache_guidance_actions": cache_actions,
        "feedback_guidance_actions": feedback_actions,
        "adaptive_cache_actions": adaptive_cache_actions,
        "guidance_ready_count": guidance_ready,
        "history_update_count": sum(
            user.episode_history_update_count for user in users
        ),
        "feedback_guidance_rate": (
            feedback_actions / decisions if decisions else 0.0
        ),
        "adaptive_cache_rate": (
            adaptive_cache_actions / decisions if decisions else 0.0
        ),
        "cache_guidance_rate": (
            cache_actions / decisions if decisions else 0.0
        ),
        "mean_q_confidence": (
            sum(user.episode_q_confidence_sum for user in users) / q_count
            if q_count
            else 0.0
        ),
        "mean_expert_confidence": (
            sum(
                user.episode_expert_confidence_sum for user in users
            )
            / expert_count
            if expert_count
            else 0.0
        ),
        "mean_guidance_probability": (
            sum(
                user.episode_guidance_probability_sum for user in users
            )
            / guidance_ready
            if guidance_ready
            else 0.0
        ),
        "mean_episode_return": float(
            np.mean([user.episode_reward_sum for user in users])
        ),
        "max_reward_identity_error": float(
            max(
                (
                    user.episode_reward_identity_error
                    for user in users
                ),
                default=0.0,
            )
        ),
        "mean_potential_initial": float(
            np.mean(
                [user.episode_potential_initial for user in users]
            )
        ),
        "mean_potential_final": float(
            np.mean(
                [user.episode_potential_final for user in users]
            )
        ),
        "hcpr_exact_path_rate": (
            sum(
                user.episode_hcpr_exact_path_tasks
                for user in users
            )
            / hcpr_total_tasks
            if hcpr_total_tasks
            else 0.0
        ),
        "hcpr_mean_posterior_criticality": (
            sum(
                user.episode_hcpr_criticality_sum
                for user in users
            )
            / hcpr_total_tasks
            if hcpr_total_tasks
            else 0.0
        ),
        "hcpr_selected_rate": (
            sum(
                user.episode_hcpr_selected_tasks
                for user in users
            )
            / hcpr_total_tasks
            if hcpr_total_tasks
            else 0.0
        ),
        "hcpr_replay_beta": (
            float(
                np.mean(
                    [
                        learner.priority_beta
                        for learner in hcpr_learners
                    ]
                )
            )
            if hcpr_learners
            else 0.0
        ),
        "hcpr_sampled_criticality": (
            float(
                np.mean(
                    [
                        learner.last_sampled_mean_criticality
                        for learner in hcpr_learners
                    ]
                )
            )
            if hcpr_learners
            else 0.0
        ),
        "hcpr_buffer_criticality": (
            float(
                np.mean(
                    [
                        learner.last_buffer_mean_criticality
                        for learner in hcpr_learners
                    ]
                )
            )
            if hcpr_learners
            else 0.0
        ),
        "hcpr_sampling_criticality_lift": (
            float(
                np.mean(
                    [
                        learner.last_sampled_mean_criticality
                        - learner.last_buffer_mean_criticality
                        for learner in hcpr_learners
                    ]
                )
            )
            if hcpr_learners
            else 0.0
        ),
        "hcpr_importance_weight_mean": (
            float(
                np.mean(
                    [
                        learner.last_importance_weight_mean
                        for learner in hcpr_learners
                    ]
                )
            )
            if hcpr_learners
            else 0.0
        ),
    }


def collect_episode_metrics(
    simulator,
    label,
    seed,
    phase,
    episode,
    before,
    wall_time,
    scenario_seed,
    scenario_hash,
    base_scenario_hash,
):
    real_task_count = sum(
        len(user.tasks_init) for user in simulator.users.values()
    )
    completed_task_count = sum(
        len(user.done_tasks) for user in simulator.users.values()
    )
    all_tasks_executed_once = all(
        set(user.done_tasks) == set(user.tasks_init)
        and all(
            count == 1
            for count in user.task_completion_counts.values()
        )
        for user in simulator.users.values()
    )
    if (
        completed_task_count != real_task_count
        or not all_tasks_executed_once
    ):
        raise RuntimeError(
            "Episode ended before every real task executed exactly once"
        )
    finish_times = np.asarray(
        [
            (
                simulator.users[user_id].finish_time_of_application
                - simulator.users[user_id].arrival_time
            )
            for user_id in range(simulator.M)
        ],
        dtype=float,
    )
    if np.any(finish_times < -1e-9):
        raise RuntimeError("DAG completion preceded its arrival")
    after = cache_snapshot(simulator)
    cache_hits = after["hits"] - before["hits"]
    cache_misses = after["misses"] - before["misses"]
    cache_requests = cache_hits + cache_misses
    policy_inference_calls = (
        after["policy_inference_calls"]
        - before["policy_inference_calls"]
    )
    policy_inference_wall_time_sec = (
        after["policy_inference_wall_time_sec"]
        - before["policy_inference_wall_time_sec"]
    )
    cache_matrix = after["cache_matrix"]
    capacity_vector = {
        server_id: int(server.capacity)
        for server_id, server in simulator.servers.items()
    }
    cache_capacity_used = sum(
        len(services)
        for services in cache_matrix.values()
    )
    cache_capacity_total = sum(capacity_vector.values())
    replica_counts = {
        service_id: sum(
            service_id in services
            for services in cache_matrix.values()
        )
        for service_id in range(1, simulator.Q + 1)
    }
    server_quality = (
        simulator.broker.last_cache_decision_context or {}
    ).get(
        "server_quality",
        {
            server_id: 1.0
            for server_id in simulator.servers
        },
    )
    action_histogram = {
        server_id: 0
        for server_id in simulator.servers
    }
    cpu_cycle_histogram = {
        server_id: 0.0
        for server_id in simulator.servers
    }
    for user in simulator.users.values():
        for task in user.done_tasks.values():
            if (
                task.service > 0
                and task.assigned_server in action_histogram
            ):
                action_histogram[task.assigned_server] += 1
                cpu_cycle_histogram[task.assigned_server] += float(
                    task.cpu_cycle
                )

    row = {
        "label": label,
        "algorithm": simulator.alg,
        "seed": seed,
        "reward_mode": simulator.reward_mode,
        "cache_policy": simulator.cache_policy,
        "cache_information_regime": (
            simulator.broker.cache_information_regime
            if simulator.cache_policy == "critical_path_joint"
            else "policy_native"
        ),
        "scenario_seed": scenario_seed,
        "scenario_fingerprint": scenario_hash,
        "base_scenario_fingerprint": base_scenario_hash,
        "phase": phase,
        "episode": episode,
        "real_task_count": real_task_count,
        "completed_task_count": completed_task_count,
        "all_tasks_executed_once": int(all_tasks_executed_once),
        "average_finish_time": float(finish_times.mean()),
        "median_finish_time": float(np.median(finish_times)),
        "p95_finish_time": float(np.percentile(finish_times, 95)),
        "min_finish_time": float(finish_times.min()),
        "max_finish_time": float(finish_times.max()),
        "std_finish_time": float(finish_times.std()),
        "computing_latency": float(simulator.avg_computing_latency),
        "data_transfer_latency": float(simulator.avg_data_transfer_latency),
        "predecessor_latency": float(simulator.avg_pred_latency),
        "service_latency": float(simulator.avg_service_latency),
        "waiting_latency": float(simulator.avg_waiting_latency),
        "computing_share": float(simulator.computing_share),
        "data_transfer_share": float(simulator.data_transfer_share),
        "predecessor_share": float(simulator.pred_share),
        "service_share": float(simulator.service_share),
        "waiting_share": float(simulator.waiting_share),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_requests if cache_requests else 0.0,
        "cache_replacements": after["replacements"] - before["replacements"],
        "cache_update_events": after["update_events"] - before["update_events"],
        "cache_decision_calls": (
            after["decision_calls"] - before["decision_calls"]
        ),
        "cache_decision_wall_time_sec": (
            after["decision_wall_time_sec"]
            - before["decision_wall_time_sec"]
        ),
        "policy_inference_calls": policy_inference_calls,
        "policy_inference_wall_time_sec": (
            policy_inference_wall_time_sec
        ),
        "policy_inference_time_per_decision_ms": (
            1000.0
            * policy_inference_wall_time_sec
            / policy_inference_calls
            if policy_inference_calls
            else 0.0
        ),
        "cache_migration_events": (
            after["migration_events"]
            - before["migration_events"]
        ),
        "cache_migration_time_sec": (
            after["migration_time_sec"]
            - before["migration_time_sec"]
        ),
        "cache_migration_critical_time_sec": (
            after["migration_critical_time_sec"]
            - before["migration_critical_time_sec"]
        ),
        "cache_remote_loading_rate": (
            cache_misses / cache_requests
            if cache_requests
            else 0.0
        ),
        "cache_zero_capacity_remote_loads": (
            after["zero_capacity_remote_loads"]
            - before["zero_capacity_remote_loads"]
        ),
        "cache_zero_capacity_assignment_rate": (
            (
                after["zero_capacity_remote_loads"]
                - before["zero_capacity_remote_loads"]
            )
            / cache_requests
            if cache_requests
            else 0.0
        ),
        "cache_capacity_used": cache_capacity_used,
        "cache_capacity_total": cache_capacity_total,
        "cache_capacity_utilization": (
            cache_capacity_used / cache_capacity_total
            if cache_capacity_total
            else 0.0
        ),
        "cache_service_coverage": (
            sum(count > 0 for count in replica_counts.values())
            / simulator.Q
        ),
        "cache_matrix_json": json.dumps(
            {
                str(server_id): list(services)
                for server_id, services in cache_matrix.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "cache_capacity_vector_json": json.dumps(
            {
                str(server_id): capacity
                for server_id, capacity
                in capacity_vector.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "cache_replica_counts_json": json.dumps(
            {
                str(service_id): count
                for service_id, count in replica_counts.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "cache_server_quality_json": json.dumps(
            {
                str(server_id): float(value)
                for server_id, value in server_quality.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "cache_execution_latency_ema_json": json.dumps(
            {
                str(server_id): (
                    None if value is None else float(value)
                )
                for server_id, value
                in simulator.broker
                .cache_server_execution_latency_ema.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "server_action_histogram_json": json.dumps(
            {
                str(server_id): count
                for server_id, count in action_histogram.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "server_cpu_cycle_histogram_json": json.dumps(
            {
                str(server_id): cycles
                for server_id, cycles in cpu_cycle_histogram.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "cache_history_windows": (
            simulator.broker.cache_history_windows
        ),
        "cache_expected_requests_ema": (
            simulator.broker.cache_expected_requests_ema
        ),
        "cache_mean_cpu_cycles_ema": (
            simulator.broker.cache_mean_cpu_cycles_ema
        ),
        "cache_global_execution_latency_ema": (
            simulator.broker.cache_global_execution_latency_ema
        ),
        "cache_global_compute_per_mcycle_ema": (
            simulator.broker.cache_global_compute_per_mcycle_ema
        ),
        "cache_global_waiting_latency_ema": (
            simulator.broker.cache_global_waiting_latency_ema
        ),
        "cache_mean_telemetry_confidence": (
            float(
                np.mean(
                    [
                        values[2]
                        for values in (
                            simulator.broker
                            ._normalized_server_telemetry()
                            .values()
                        )
                    ]
                )
            )
            if (
                simulator.alg.startswith(
                    "causal_normalizedTelemetry"
                )
            )
            else 0.0
        ),
        "cache_updates_enabled": int(simulator.update_caching),
        "beta": float(simulator.beta),
        "mean_epsilon": mean_epsilon(simulator),
        "mean_learning_rate": mean_learning_rate(simulator),
        "episode_wall_time_sec": wall_time,
    }
    row.update(collect_innovation_metrics(simulator))
    return row


def run_phase(
    simulator,
    writer,
    csv_file,
    args,
    phase,
    episodes,
    scenario_seed=None,
    scenario_hash=None,
    base_scenario_hash=None,
    checkpoint_callback=None,
):
    rows = []
    if scenario_seed is None:
        scenario_seed = args.seed
    if scenario_hash is None:
        scenario_hash = scenario_fingerprint(
            scenario_snapshot(simulator)
        )
    if base_scenario_hash is None:
        base_scenario_hash = base_scenario_fingerprint(
            simulator
        )
    report_every = max(1, episodes // 10)
    for episode_index in range(episodes):
        if phase == "train":
            simulator.beta = max(
                args.beta_min,
                args.beta * args.beta_decay**episode_index,
            )
            if (
                args.cache_freeze_episode
                and episode_index + 1 > args.cache_freeze_episode
            ):
                simulator.update_caching = False
            set_learning_rate(
                simulator,
                scheduled_learning_rate(
                    args,
                    episode_index + 1,
                ),
            )

        simulator.reset()
        before = cache_snapshot(simulator)
        started = time.perf_counter()
        simulator.run()
        wall_time = time.perf_counter() - started
        row = collect_episode_metrics(
            simulator=simulator,
            label=args.label,
            seed=args.seed,
            phase=phase,
            episode=episode_index + 1,
            before=before,
            wall_time=wall_time,
            scenario_seed=scenario_seed,
            scenario_hash=scenario_hash,
            base_scenario_hash=base_scenario_hash,
        )
        writer.writerow(row)
        rows.append(row)
        should_stop = False
        if checkpoint_callback is not None:
            should_stop = bool(
                checkpoint_callback(
                    episode_index + 1,
                    simulator,
                )
            )

        if (
            (episode_index + 1) % report_every == 0
            or episode_index + 1 == episodes
            or should_stop
        ):
            csv_file.flush()
            if not args.quiet:
                print(
                    f"{phase}: {episode_index + 1}/{episodes} "
                    f"finish={row['average_finish_time']:.6f}"
                )
        if should_stop:
            break
    return rows


def summarize_rows(rows):
    if not rows:
        return None
    metrics = (
        "average_finish_time",
        "p95_finish_time",
        "computing_latency",
        "data_transfer_latency",
        "predecessor_latency",
        "service_latency",
        "waiting_latency",
        "cache_hit_rate",
        "cache_replacements",
        "cache_decision_calls",
        "cache_decision_wall_time_sec",
        "policy_inference_calls",
        "policy_inference_wall_time_sec",
        "policy_inference_time_per_decision_ms",
        "cache_migration_events",
        "cache_migration_time_sec",
        "cache_migration_critical_time_sec",
        "cache_remote_loading_rate",
        "cache_zero_capacity_remote_loads",
        "cache_zero_capacity_assignment_rate",
        "cache_capacity_utilization",
        "cache_service_coverage",
        "cache_mean_telemetry_confidence",
        "feedback_guidance_rate",
        "adaptive_cache_rate",
        "cache_guidance_rate",
        "mean_q_confidence",
        "mean_expert_confidence",
        "mean_guidance_probability",
        "mean_episode_return",
        "max_reward_identity_error",
        "mean_potential_initial",
        "mean_potential_final",
        "hcpr_exact_path_rate",
        "hcpr_mean_posterior_criticality",
        "hcpr_selected_rate",
        "hcpr_replay_beta",
        "hcpr_buffer_criticality",
        "hcpr_sampled_criticality",
        "hcpr_sampling_criticality_lift",
        "hcpr_importance_weight_mean",
        "episode_wall_time_sec",
    )
    summary = {"episodes": len(rows)}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def paper_style_training_metric(rows, moving_window=200, final_window=100):
    values = np.asarray(
        [row["average_finish_time"] for row in rows],
        dtype=float,
    )
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    smoothed = np.empty_like(values)
    for index in range(values.size):
        if index < moving_window:
            start = 0
            stop = index + 1
        else:
            start = index - moving_window
            stop = index
        smoothed[index] = (prefix[stop] - prefix[start]) / (stop - start)

    score_window = min(final_window, smoothed.size)
    return {
        "moving_average_window": moving_window,
        "final_average_window": score_window,
        "mean_average_finish_time": float(smoothed[-score_window:].mean()),
    }


def save_checkpoint(simulator, path, args):
    if not simulator.learning_enabled:
        return None
    checkpoint = {
        "algorithm": args.algorithm,
        "label": args.label,
        "seed": args.seed,
        "reward_mode": args.reward_mode,
        "reward_scale": args.reward_scale,
        "potential_reward_weight": args.potential_reward_weight,
        "cache_policy": args.cache_policy,
        "information_protocol_version": (
            INFORMATION_PROTOCOL_VERSION
        ),
        "scenario_fingerprint_version": (
            SCENARIO_FINGERPRINT_VERSION
            if args.server_capacity_multiset is not None
            else "legacy_deployment_v1"
        ),
        "cache_information_regime": (
            simulator.broker.cache_information_regime
            if args.cache_policy == "critical_path_joint"
            else "policy_native"
        ),
        "cache_history": simulator.broker.cache_history_state_dict(),
        "cache_runtime": simulator.broker.cache_runtime_state_dict(),
        "server_models": {
            server_id: server.agent.agent.TrainNet.model.state_dict()
            for server_id, server in simulator.servers.items()
        },
        "server_epsilons": {
            server_id: server.agent.agent.epsilon
            for server_id, server in simulator.servers.items()
        },
        "server_learning_rates": {
            server_id: [
                group["lr"]
                for group in server.agent.agent.TrainNet.optimizer.param_groups
            ]
            for server_id, server in simulator.servers.items()
        },
        "history_feedback_guides": {
            server_id: (
                server.history_feedback_guide.state_dict()
                if server.history_feedback_guide is not None
                else None
            )
            for server_id, server in simulator.servers.items()
        },
    }
    torch.save(checkpoint, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_frozen_checkpoint(path, episode, frozen_state):
    torch.save(
        {
            "episode": int(episode),
            "frozen_state": frozen_state,
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_frozen_state(simulator):
    return {
        "weights": {
            server_id: {
                name: value.detach().clone()
                for name, value in server.agent.agent.TrainNet.model.state_dict().items()
            }
            for server_id, server in simulator.servers.items()
        },
        "replay_sizes": {
            server_id: len(server.agent.agent.TrainNet.experience["s"])
            for server_id, server in simulator.servers.items()
        },
        "epsilons": {
            server_id: server.agent.agent.epsilon
            for server_id, server in simulator.servers.items()
        },
        "learning_rates": {
            server_id: [
                group["lr"]
                for group in server.agent.agent.TrainNet.optimizer.param_groups
            ]
            for server_id, server in simulator.servers.items()
        },
        "deadlines": {
            user_id: user.deadline
            for user_id, user in simulator.users.items()
        },
        "services": {
            server_id: tuple(server.services)
            for server_id, server in simulator.servers.items()
        },
        "capacities": {
            server_id: int(server.capacity)
            for server_id, server in simulator.servers.items()
        },
        "cache_estimates": copy.deepcopy(simulator.broker.H),
        "cache_history": copy.deepcopy(
            simulator.broker.cache_history_state_dict()
        ),
        "cache_runtime": copy.deepcopy(
            simulator.broker.cache_runtime_state_dict()
        ),
        "history_feedback_guides": {
            server_id: (
                copy.deepcopy(server.history_feedback_guide.state_dict())
                if server.history_feedback_guide is not None
                else None
            )
            for server_id, server in simulator.servers.items()
        },
    }


def apply_frozen_state(simulator, frozen_state):
    expected_capacities = frozen_state.get("capacities")
    if expected_capacities is not None:
        actual_capacities = {
            server_id: int(server.capacity)
            for server_id, server in simulator.servers.items()
        }
        if actual_capacities != expected_capacities:
            raise RuntimeError(
                "Frozen cache capacities do not match the deployment"
            )
    for server_id, weights in frozen_state["weights"].items():
        agent = simulator.servers[server_id].agent.agent
        agent.TrainNet.model.load_state_dict(weights)
        agent.update_target_model()
        agent.epsilon = frozen_state["epsilons"][server_id]
        for group, learning_rate in zip(
            agent.TrainNet.optimizer.param_groups,
            frozen_state["learning_rates"][server_id],
        ):
            group["lr"] = learning_rate

    simulator.broker.H = copy.deepcopy(
        frozen_state["cache_estimates"]
    )
    if "cache_history" in frozen_state:
        simulator.broker.load_cache_history_state_dict(
            copy.deepcopy(frozen_state["cache_history"])
        )
    if "cache_runtime" in frozen_state:
        simulator.broker.load_cache_runtime_state_dict(
            copy.deepcopy(frozen_state["cache_runtime"])
        )
    simulator.server_service_info.fill(0.0)
    for server_id, services in frozen_state["services"].items():
        real_services = [
            service_id
            for service_id in services
            if service_id > 0
        ]
        if len(real_services) > simulator.servers[server_id].capacity:
            raise RuntimeError(
                f"Frozen services exceed server {server_id} capacity"
            )
        simulator.servers[server_id].services = list(services)
        for service in services:
            if service > 0:
                simulator.server_service_info[
                    server_id,
                    service - 1,
                ] = 1.0

    for user_id, deadline in frozen_state["deadlines"].items():
        simulator.users[user_id].deadline = deadline


def run_scenario_bank_evaluation(
    writer,
    csv_file,
    args,
    input_config,
    learning_config,
    frozen_state,
    deployment_state,
    simulator_log,
    figures_dir,
    episodes=None,
    seed_offset=None,
    phase="eval",
):
    if episodes is None:
        episodes = args.eval_episodes
    if seed_offset is None:
        seed_offset = args.eval_seed_offset
    rows = []
    scenarios = []
    report_every = max(1, episodes // 10)
    update_caching = bool(
        getattr(args, "eval_update_caching", False)
    )

    for episode_index in range(episodes):
        eval_seed = (
            args.seed
            + seed_offset
            + episode_index
        )
        seed_everything(eval_seed)
        eval_config = copy.deepcopy(input_config)
        eval_config["seed"] = eval_seed
        eval_config["save topology figure"] = False
        workflow_family = None
        if args.eval_dag_families is not None:
            workflow_family = args.eval_dag_families[
                episode_index % len(args.eval_dag_families)
            ]
            eval_config["application graph family"] = (
                workflow_family
            )
        eval_simulator = MEC_Simulator(
            outputfile=simulator_log,
            Input_dict=eval_config,
            learning_arguments=learning_config,
            filename_png=str(figures_dir),
        )
        if args.eval_bank_scope in ("workload", "infrastructure"):
            apply_deployment_state(
                eval_simulator,
                deployment_state,
                include_users=args.eval_bank_scope == "workload",
            )
        raw_snapshot = scenario_snapshot(eval_simulator)
        scenario_hash = scenario_fingerprint(raw_snapshot)
        base_scenario_hash = base_scenario_fingerprint(
            eval_simulator
        )

        apply_frozen_state(eval_simulator, frozen_state)
        for user in eval_simulator.users.values():
            user.setpos0()
        eval_simulator.set_training(
            False,
            update_caching=update_caching,
        )
        eval_simulator.reset()
        episode_frozen_state = capture_frozen_state(
            eval_simulator
        )

        before = cache_snapshot(eval_simulator)
        started = time.perf_counter()
        eval_simulator.run()
        wall_time = time.perf_counter() - started
        row = collect_episode_metrics(
            simulator=eval_simulator,
            label=args.label,
            seed=args.seed,
            phase=phase,
            episode=episode_index + 1,
            before=before,
            wall_time=wall_time,
            scenario_seed=eval_seed,
            scenario_hash=scenario_hash,
            base_scenario_hash=base_scenario_hash,
        )
        if writer is not None:
            writer.writerow(row)
        rows.append(row)
        verify_frozen_state(
            eval_simulator,
            episode_frozen_state,
            check_cache=not update_caching,
        )
        scenarios.append(
            {
                "episode": episode_index + 1,
                "seed": eval_seed,
                "fingerprint": scenario_hash,
                "base_fingerprint": base_scenario_hash,
                "workflow_family": workflow_family,
                "user_initial_positions": {
                    user_id: user_data["initial_position"]
                    for user_id, user_data
                    in raw_snapshot["users"].items()
                },
                "user_graph_keys": {
                    user_id: user_data["dag_key"]
                    for user_id, user_data
                    in raw_snapshot["users"].items()
                },
            }
        )

        if (
            (episode_index + 1) % report_every == 0
            or episode_index + 1 == episodes
        ):
            if csv_file is not None:
                csv_file.flush()
            if not args.quiet:
                print(
                    f"{phase}-bank: {episode_index + 1}/"
                    f"{episodes} "
                    f"finish={row['average_finish_time']:.6f}"
                )

    return rows, scenarios


def verify_frozen_state(simulator, frozen_state, check_cache):
    for server_id, expected_weights in frozen_state["weights"].items():
        actual_weights = simulator.servers[
            server_id
        ].agent.agent.TrainNet.model.state_dict()
        for name, expected in expected_weights.items():
            if not torch.equal(expected, actual_weights[name]):
                raise RuntimeError(
                    f"Model changed during evaluation: server={server_id}, tensor={name}"
                )

    current_replay_sizes = {
        server_id: len(server.agent.agent.TrainNet.experience["s"])
        for server_id, server in simulator.servers.items()
    }
    current_epsilons = {
        server_id: server.agent.agent.epsilon
        for server_id, server in simulator.servers.items()
    }
    current_learning_rates = {
        server_id: [
            group["lr"]
            for group in server.agent.agent.TrainNet.optimizer.param_groups
        ]
        for server_id, server in simulator.servers.items()
    }
    current_deadlines = {
        user_id: user.deadline
        for user_id, user in simulator.users.items()
    }
    if current_replay_sizes != frozen_state["replay_sizes"]:
        raise RuntimeError("Replay buffer changed during evaluation")
    if current_epsilons != frozen_state["epsilons"]:
        raise RuntimeError("Epsilon changed during evaluation")
    if current_learning_rates != frozen_state["learning_rates"]:
        raise RuntimeError("Learning rate changed during evaluation")
    if current_deadlines != frozen_state["deadlines"]:
        raise RuntimeError("Adaptive deadlines changed during evaluation")

    current_guides = {
        server_id: (
            server.history_feedback_guide.state_dict()
            if server.history_feedback_guide is not None
            else None
        )
        for server_id, server in simulator.servers.items()
    }
    if current_guides != frozen_state["history_feedback_guides"]:
        raise RuntimeError(
            "Historical feedback changed during evaluation"
        )

    if check_cache:
        current_services = {
            server_id: tuple(server.services)
            for server_id, server in simulator.servers.items()
        }
        if current_services != frozen_state["services"]:
            raise RuntimeError("Cached services changed during frozen evaluation")
        if simulator.broker.H != frozen_state["cache_estimates"]:
            raise RuntimeError("Cache estimates changed during frozen evaluation")
        if (
            "cache_history" in frozen_state
            and simulator.broker.cache_history_state_dict()
            != frozen_state["cache_history"]
        ):
            raise RuntimeError(
                "Causal cache history changed during frozen evaluation"
            )
        if (
            "cache_runtime" in frozen_state
            and simulator.broker.cache_runtime_state_dict()
            != frozen_state["cache_runtime"]
        ):
            raise RuntimeError(
                "Cache runtime state changed during frozen evaluation"
            )


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)

    run_dir = args.result_dir.resolve()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_config = copy.deepcopy(INPUT_DICT)
    input_config.update(
        {
            "Folder": args.label,
            "alg": args.algorithm,
            "Number of users": args.num_users,
            "Number of servers": args.num_servers,
            "Number of services": args.num_services,
            "Number of tasks for each user": args.num_tasks,
            "dag dataset path": args.dag_dataset_path,
            "dag dataset sha256": args.dag_dataset_sha256,
            "dag depth increment": args.dag_depth_increment,
            "dependency data scale": (
                args.dependency_data_scale
            ),
            "server capacity": args.server_capacity,
            "server capacity multiset": (
                args.server_capacity_multiset
            ),
            "capacity assignment namespace": (
                args.capacity_assignment_namespace
            ),
            "baseline server capacity": (
                args.baseline_server_capacity
            ),
            "Number of episodes": args.train_episodes,
            "Number of runs": 1,
            "filling steps": args.filling_steps,
            "steps to updates": args.steps_to_updates,
            "seed": args.seed,
            "beta": args.beta,
            "Bandwidth": args.bandwidth,
            "reward mode": args.reward_mode,
            "reward scale": args.reward_scale,
            "potential reward weight": (
                args.potential_reward_weight
            ),
            "hcpr temperature": args.hcpr_temperature,
            "bcr top fraction": args.bcr_top_fraction,
            "cache policy": args.cache_policy,
            "cache score alpha": args.cache_score_alpha,
            "cache history alpha": args.cache_history_alpha,
            "cache locality weight": args.cache_locality_weight,
            "cache update interval": args.cache_update_interval,
            "cache hysteresis factor": (
                args.cache_hysteresis_factor
            ),
            "cache min residence updates": (
                args.cache_min_residence_updates
            ),
            "cache compute weight": args.cache_compute_weight,
            "cache server quality enabled": (
                args.cache_server_quality
            ),
            "cache coverage constraint": (
                args.cache_coverage_constraint
            ),
            "task dependency features enabled": (
                args.task_dependency_features
            ),
            "cache dependency awareness enabled": (
                args.cache_dependency_awareness
            ),
            "telemetry min samples": args.telemetry_min_samples,
            "telemetry freshness half life": (
                args.telemetry_freshness_half_life
            ),
            "caching decision enabled": args.enable_caching,
            "historical feedback guidance": (
                args.historical_feedback_guidance
            ),
            "adaptive guidance gate": args.adaptive_guidance_gate,
            "history feedback alpha": args.history_feedback_alpha,
            "history feedback min samples": (
                args.history_feedback_min_samples
            ),
            "history feedback max probability": (
                args.history_feedback_max_probability
            ),
            "history feedback fixed probability": (
                args.history_feedback_fixed_probability
            ),
        }
    )
    learning_config = copy.deepcopy(learning_arg)
    learning_config.update(
        {
            "batch_size": args.batch_size,
            "min_experiences": args.min_experiences,
            "learning_rate": args.learning_rate,
            "hidden_units": args.hidden_units,
            "epsilon": args.epsilon,
            "maximum_exploration": args.max_explore,
            "gamma": args.gamma,
            "n_step": args.n_step,
            "num_quantiles": args.num_quantiles,
            "risk_tail_fraction": args.risk_tail_fraction,
            "entropy_coefficient": args.entropy_coefficient,
            "sac_target_entropy_ratio": (
                args.sac_target_entropy_ratio
            ),
            "sac_target_tau": args.sac_target_tau,
            "priority_alpha": args.priority_alpha,
            "priority_beta_start": args.priority_beta_start,
            "priority_beta_anneal_steps": (
                args.priority_beta_anneal_steps
            ),
            "criticality_boost": args.criticality_boost,
        }
    )

    config = {
        "information_protocol_version": (
            INFORMATION_PROTOCOL_VERSION
        ),
        "method_modules": method_module_manifest(args),
        "revision": {
            "id": args.revision_id,
            "parent": args.revision_parent,
            "reason": args.revision_reason,
            "changed_module": args.revision_changed_module,
            "expected_metric": args.revision_expected_metric,
            "rejection_condition": (
                args.revision_rejection_condition
            ),
            "seed_partition": args.seed_partition,
        },
        "arguments": vars(args) | {"result_dir": str(run_dir)},
        "input_config": input_config,
        "learning_config": learning_config,
    }
    config_payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config["experiment_config_sha256"] = hashlib.sha256(
        config_payload
    ).hexdigest()
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    all_rows = []
    total_started = time.perf_counter()
    with (run_dir / "simulator.log").open("w", encoding="utf-8") as simulator_log:
        simulator = MEC_Simulator(
            outputfile=simulator_log,
            Input_dict=input_config,
            learning_arguments=learning_config,
            filename_png=str(figures_dir),
        )
        deployment_state = capture_deployment_state(simulator)
        (run_dir / "scenario_initial.json").write_text(
            json.dumps(scenario_snapshot(simulator), indent=2),
            encoding="utf-8",
        )
        for user in simulator.users.values():
            user.setpos0()

        training_checkpoint_states = {}
        checkpoint_validation = []
        convergence_reached = False
        convergence_episode = None
        final_convergence_diagnostics = None
        convergence_monitor = (
            ConvergenceMonitor(
                min_episode=args.convergence_min_episodes,
                window=args.convergence_window,
                patience=args.convergence_patience,
                relative_mean_change_threshold=(
                    args.convergence_relative_mean_change
                ),
                relative_slope_threshold=(
                    args.convergence_relative_slope
                ),
            )
            if args.convergence_mode
            else None
        )
        checkpoints_dir = run_dir / "training_checkpoints"
        if args.checkpoint_every:
            checkpoints_dir.mkdir(exist_ok=True)

        def validate_training_checkpoint(episode, checkpoint_state):
            rng_state = capture_rng_state()
            try:
                validation_rows, validation_scenarios = (
                    run_scenario_bank_evaluation(
                        writer=None,
                        csv_file=None,
                        args=args,
                        input_config=input_config,
                        learning_config=learning_config,
                        frozen_state=checkpoint_state,
                        deployment_state=deployment_state,
                        simulator_log=simulator_log,
                        figures_dir=figures_dir,
                        episodes=args.validation_scenarios,
                        seed_offset=args.validation_seed_offset,
                        phase="validation",
                    )
                )
            finally:
                restore_rng_state(rng_state)

            validation_summary = summarize_rows(validation_rows)
            return {
                "episode": episode,
                "mean_average_finish_time": validation_summary[
                    "mean_average_finish_time"
                ],
                "mean_p95_finish_time": validation_summary[
                    "mean_p95_finish_time"
                ],
                "scenario_fingerprints": [
                    scenario["fingerprint"]
                    for scenario in validation_scenarios
                ],
            }

        def capture_training_checkpoint(episode, current_simulator):
            nonlocal convergence_episode
            nonlocal convergence_reached
            nonlocal final_convergence_diagnostics
            if not args.checkpoint_every:
                return False
            if (
                episode % args.checkpoint_every != 0
                and episode != args.train_episodes
            ):
                return False
            state = capture_frozen_state(current_simulator)
            save_frozen_checkpoint(
                checkpoints_dir / f"episode_{episode:06d}.pt",
                episode,
                state,
            )
            if not args.convergence_mode:
                training_checkpoint_states[episode] = state
                return False

            record = validate_training_checkpoint(episode, state)
            diagnostics = convergence_monitor.update(
                episode,
                record["mean_average_finish_time"],
            )
            record["convergence"] = diagnostics
            checkpoint_validation.append(record)
            final_convergence_diagnostics = diagnostics
            if diagnostics["converged"]:
                convergence_reached = True
                convergence_episode = episode
            return convergence_reached

        with (run_dir / "episodes.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=EPISODE_FIELDS)
            writer.writeheader()

            simulator.set_training(
                True,
                update_caching=args.enable_caching,
            )
            training_rows = run_phase(
                simulator,
                writer,
                csv_file,
                args,
                phase="train",
                episodes=args.train_episodes,
                checkpoint_callback=capture_training_checkpoint,
            )
            all_rows.extend(training_rows)

            checkpoint_hash = save_checkpoint(
                simulator,
                run_dir / "checkpoint.pt",
                args,
            )
            (run_dir / "scenario_after_training.json").write_text(
                json.dumps(scenario_snapshot(simulator), indent=2),
                encoding="utf-8",
            )
            frozen_state = capture_frozen_state(simulator)
            actual_train_episodes = len(training_rows)
            selected_checkpoint_episode = actual_train_episodes
            selected_checkpoint_hash = checkpoint_hash

            if args.checkpoint_every and not args.convergence_mode:
                if actual_train_episodes not in training_checkpoint_states:
                    training_checkpoint_states[
                        actual_train_episodes
                    ] = frozen_state
                for episode, checkpoint_state in sorted(
                    training_checkpoint_states.items()
                ):
                    checkpoint_validation.append(
                        validate_training_checkpoint(
                            episode,
                            checkpoint_state,
                        )
                    )

                selected_record = min(
                    checkpoint_validation,
                    key=lambda record: (
                        record["mean_average_finish_time"],
                        record["episode"],
                    ),
                )
                selected_checkpoint_episode = selected_record[
                    "episode"
                ]
                frozen_state = training_checkpoint_states[
                    selected_checkpoint_episode
                ]
                selected_checkpoint_hash = save_frozen_checkpoint(
                    run_dir / "selected_checkpoint.pt",
                    selected_checkpoint_episode,
                    frozen_state,
                )
            elif args.convergence_mode:
                selected_checkpoint_hash = save_frozen_checkpoint(
                    run_dir / "selected_checkpoint.pt",
                    selected_checkpoint_episode,
                    frozen_state,
                )
            else:
                selected_checkpoint_hash = save_frozen_checkpoint(
                    run_dir / "selected_checkpoint.pt",
                    selected_checkpoint_episode,
                    frozen_state,
                )

            if args.checkpoint_every:
                (run_dir / "checkpoint_validation.json").write_text(
                    json.dumps(
                        {
                            "strategy": (
                                "convergence_final"
                                if args.convergence_mode
                                else "validation_best"
                            ),
                            "selected_episode": selected_checkpoint_episode,
                            "convergence_reached": (
                                convergence_reached
                                if args.convergence_mode
                                else None
                            ),
                            "records": checkpoint_validation,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            evaluation_scenarios = []
            evaluation_rows = []
            eligible_for_comparison = (
                not args.convergence_mode
                or convergence_reached
            )

            if eligible_for_comparison and args.eval_scenario_bank:
                evaluation_rows, evaluation_scenarios = (
                    run_scenario_bank_evaluation(
                        writer=writer,
                        csv_file=csv_file,
                        args=args,
                        input_config=input_config,
                        learning_config=learning_config,
                        frozen_state=frozen_state,
                        deployment_state=deployment_state,
                        simulator_log=simulator_log,
                        figures_dir=figures_dir,
                    )
                )
                (run_dir / "evaluation_scenarios.json").write_text(
                    json.dumps(
                        evaluation_scenarios,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            elif eligible_for_comparison:
                seed_everything(args.seed + args.eval_seed_offset)
                for user in simulator.users.values():
                    user.setpos0()
                simulator.beta = 0.0
                simulator.set_training(
                    False,
                    update_caching=(
                        args.enable_caching
                        and args.eval_update_caching
                    ),
                )
                evaluation_rows = run_phase(
                    simulator,
                    writer,
                    csv_file,
                    args,
                    phase="eval",
                    episodes=args.eval_episodes,
                    scenario_seed=args.seed,
                )
                verify_frozen_state(
                    simulator,
                    frozen_state,
                    check_cache=not args.eval_update_caching,
                )
            all_rows.extend(evaluation_rows)

    tail_size = min(100, max(1, len(training_rows) // 10))
    convergence_summary = {
        "enabled": args.convergence_mode,
        "reached": (
            convergence_reached
            if args.convergence_mode
            else None
        ),
        "stop_reason": (
            (
                "criterion_met"
                if convergence_reached
                else "max_episodes_reached"
            )
            if args.convergence_mode
            else "fixed_budget"
        ),
        "episode": convergence_episode,
        "actual_train_episodes": actual_train_episodes,
        "max_train_episodes": args.train_episodes,
        "min_train_episodes": args.convergence_min_episodes,
        "checkpoint_every": args.checkpoint_every,
        "validation_scenarios": args.validation_scenarios,
        "learning_rate_schedule": args.learning_rate_schedule,
        "initial_learning_rate": args.learning_rate,
        "minimum_learning_rate": args.min_learning_rate,
        "learning_rate_decay_start": (
            args.learning_rate_decay_start
        ),
        "learning_rate_decay_end": args.learning_rate_decay_end,
        "cache_freeze_episode": args.cache_freeze_episode,
        "window": args.convergence_window,
        "patience": args.convergence_patience,
        "relative_mean_change_threshold": (
            args.convergence_relative_mean_change
        ),
        "relative_slope_threshold": (
            args.convergence_relative_slope
        ),
        "final_diagnostics": final_convergence_diagnostics,
    }
    summary = {
        "status": "complete",
        "dag_completion_protocol_version": (
            DAG_COMPLETION_PROTOCOL_VERSION
        ),
        "label": args.label,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "method_modules": method_module_manifest(args),
        "revision": config["revision"],
        "experiment_config_sha256": (
            config["experiment_config_sha256"]
        ),
        "capacity_protocol_version": (
            CAPACITY_PROTOCOL_VERSION
            if args.server_capacity_multiset is not None
            else "legacy_scalar_capacity"
        ),
        "server_capacities": {
            str(server_id): int(server.capacity)
            for server_id, server in simulator.servers.items()
        },
        "total_server_capacity": simulator.total_server_capacity,
        "dag_dataset": {
            "path": str(simulator.dag_dataset_path),
            "sha256": simulator.dag_dataset_sha256,
            "is_default": simulator.dag_dataset_is_default,
            "graph_count": simulator.dag_dataset_graph_count,
            "eligible_graph_count": (
                simulator.dag_dataset_eligible_graph_count
            ),
        },
        "checkpoint_sha256": checkpoint_hash,
        "selected_checkpoint_sha256": (
            selected_checkpoint_hash
        ),
        "selected_checkpoint_episode": (
            selected_checkpoint_episode
        ),
        "checkpoint_selection_enabled": bool(
            args.checkpoint_every and not args.convergence_mode
        ),
        "checkpoint_strategy": (
            "convergence_final"
            if args.convergence_mode
            else (
                "validation_best"
                if args.checkpoint_every
                else "fixed_budget_final"
            )
        ),
        "checkpoint_validation": checkpoint_validation,
        "convergence": convergence_summary,
        "eligible_for_comparison": eligible_for_comparison,
        "evaluation_state_frozen": bool(evaluation_rows),
        "evaluation_mode": (
            (
                f"independent_{args.eval_bank_scope}_bank"
                if args.eval_scenario_bank
                else "repeated_training_scenario"
            )
            if evaluation_rows
            else "not_evaluated_unconverged"
        ),
        "evaluation_unique_scenarios": len(
            {
                row["scenario_fingerprint"]
                for row in evaluation_rows
            }
        ),
        "evaluation_unique_base_scenarios": len(
            {
                row["base_scenario_fingerprint"]
                for row in evaluation_rows
            }
        ),
        "evaluation_scenario_count": len(evaluation_rows),
        "reward_scale": args.reward_scale,
        "potential_reward_weight": args.potential_reward_weight,
        "information_protocol_version": (
            INFORMATION_PROTOCOL_VERSION
        ),
        "scenario_fingerprint_version": (
            SCENARIO_FINGERPRINT_VERSION
            if args.server_capacity_multiset is not None
            else "legacy_deployment_v1"
        ),
        "cache_information_regime": (
            simulator.broker.cache_information_regime
            if args.cache_policy == "critical_path_joint"
            else "policy_native"
        ),
        "cache_history": simulator.broker.cache_history_state_dict(),
        "train": summarize_rows(training_rows),
        "train_tail": summarize_rows(training_rows[-tail_size:]),
        "paper_training_metric": paper_style_training_metric(training_rows),
        "eval": summarize_rows(evaluation_rows),
        "total_wall_time_sec": time.perf_counter() - total_started,
        "episode_rows": len(all_rows),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
