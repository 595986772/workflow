#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from information_protocol import INFORMATION_PROTOCOL_VERSION


LEAN_OUR_LABEL = "lean_our"
OUR_DQN_LABEL = "our_dqn"
DAOC_PAPER_LABEL = "daoc_paper"
COORD_DISCRETE_SAC_LABEL = "coord_cache_discrete_sac"
COORD_RANDOM_LABEL = "coord_cache_random"
COORD_NEAREST_LABEL = "coord_cache_nearest"
COORD_NEAREST_SERVICE_LABEL = "coord_cache_nearest_service"
DQN_WDSA_STD_LABEL = "dqn_wdsa_std_cache"
BASE_DDQN_STD_LABEL = "base_ddqn_std_cache"
OUR_FLAT_DDQN_LABEL = "our_flat_ddqn"
OUR_NO_TASK_DEPENDENCY_LABEL = "our_no_task_dependency"
OUR_NO_DEPENDENCY_CACHE_LABEL = "our_no_dependency_cache"
OUR_TERMINAL_REWARD_LABEL = "our_terminal_reward"


ALGORITHMS = [
    {
        "label": "random",
        "display_name": "Random",
        "algorithm": "random",
        "family": "heuristic",
        "beta": 0.0,
    },
    {
        "label": "nearest",
        "display_name": "Nearest",
        "algorithm": "nearest_server",
        "family": "heuristic",
        "beta": 0.0,
    },
    {
        "label": "greedy",
        "display_name": "Nearest + Service",
        "algorithm": "nearest_with_service",
        "family": "heuristic",
        "beta": 0.0,
    },
    {
        "label": COORD_RANDOM_LABEL,
        "display_name": "Random (Coordinated Cache)",
        "algorithm": "random",
        "family": "heuristic",
        "beta": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "active_modules": [
            "random_offloading",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
    },
    {
        "label": COORD_NEAREST_LABEL,
        "display_name": "Nearest (Coordinated Cache)",
        "algorithm": "nearest_server",
        "family": "heuristic",
        "beta": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "active_modules": [
            "nearest_server_offloading",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
    },
    {
        "label": COORD_NEAREST_SERVICE_LABEL,
        "display_name": "Nearest-with-Service (Coordinated Cache)",
        "algorithm": "nearest_with_service",
        "family": "heuristic",
        "beta": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "active_modules": [
            "nearest_cached_service_offloading",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
    },
    {
        "label": "basic_dqn",
        "display_name": "Basic DQN",
        "algorithm": "simpleDQN",
        "family": "learning",
        "beta": 0.0,
    },
    {
        "label": DQN_WDSA_STD_LABEL,
        "display_name": "DQN-WDSA (Standard Cache)",
        "algorithm": "simpleDQN",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "terminal_binary",
        "cache_policy": "popularity_ema",
        "active_modules": [
            "basic_workload_state_dqn",
            "independent_popularity_ema_cache",
        ],
        "training_objective": "terminal_deadline_reward",
    },
    {
        "label": "service_dqn",
        "display_name": "Service-aware DQN",
        "algorithm": "justserviceDQN",
        "family": "learning",
        "beta": 0.0,
    },
    {
        "label": "dependency_dqn",
        "display_name": "Dependency-aware DQN",
        "algorithm": "prev_serversDQN",
        "family": "learning",
        "beta": 0.0,
    },
    {
        "label": "unguided_full",
        "display_name": "Unguided Full",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.0,
    },
    {
        "label": "guided_full",
        "display_name": "Guided Full (Fixed beta)",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
    },
    {
        "label": "guided_decay",
        "display_name": "Guided Full (Paper Decay)",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.9,
        "beta_min": 0.1,
        "beta_decay": 0.995,
    },
    {
        "label": DAOC_PAPER_LABEL,
        "display_name": "DAOC-paper",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.9,
        "beta_min": 0.1,
        "beta_decay": 0.995,
        "reward_mode": "terminal_binary",
        "cache_policy": "paper_popularity_cost_ema",
        "active_modules": [
            "distributed_dqn",
            "paper_decaying_action_guidance",
            "daoc_paper_popularity_cost_cache",
        ],
        "training_objective": "terminal_deadline_reward",
    },
    {
        "label": "gate_only",
        "display_name": "Adaptive Cache Gate Only",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "adaptive_guidance_gate": True,
    },
    {
        "label": "history_only",
        "display_name": "Historical Feedback Only",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "historical_feedback_guidance": True,
    },
    {
        "label": "proposed_full",
        "display_name": "Proposed Full",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "historical_feedback_guidance": True,
        "adaptive_guidance_gate": True,
    },
    {
        "label": "cpr_reward",
        "display_name": "CPR Potential Reward",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "critical_path_potential",
    },
    {
        "label": "cpr_cache",
        "display_name": "CPR + Critical Hysteretic Cache",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "critical_path_potential",
        "cache_policy": "critical_path_hysteresis",
    },
    {
        "label": "cpr_coord_cache",
        "display_name": "CPR + Coordinated Critical Cache",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "critical_path_potential",
        "cache_policy": "critical_path_coordinated",
    },
    {
        "label": "cpr_joint_cache",
        "display_name": "CPR + Causal Compute-Aware Cache",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "critical_path_potential",
        "cache_policy": "critical_path_joint",
    },
    {
        "label": "hybrid_reward",
        "display_name": "Hybrid Terminal + CPR Reward",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "terminal_plus_potential",
        "potential_reward_weight": 0.5,
    },
    {
        "label": "our",
        "display_name": "OUR",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "terminal_plus_potential",
        "potential_reward_weight": 0.5,
        "cache_policy": "critical_path_joint",
    },
    {
        "label": "correct_ddqn",
        "display_name": "Causal n-step Double DQN",
        "algorithm": "causal_task_serverDDQN",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
    },
    {
        "label": "cpqac",
        "display_name": "OUR (CPQAC)",
        "algorithm": "causal_task_serverCPQAC",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
    },
    {
        "label": "telemetry_ddqn",
        "display_name": "Telemetry n-step Double DQN",
        "algorithm": "causal_telemetryDDQN",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
    },
    {
        "label": "telemetry_cpqac",
        "display_name": "OUR (Telemetry CPQAC)",
        "algorithm": "causal_telemetryCPQAC",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
    },
    {
        "label": "capq",
        "display_name": "OUR (CAPQ)",
        "algorithm": "causal_task_serverCAPQ",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
    },
    {
        "label": "capq_tail50",
        "display_name": "CAPQ (50% tail)",
        "algorithm": "causal_task_serverCAPQ",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.5,
    },
    {
        "label": "capq_mean",
        "display_name": "CAPQ (mean)",
        "algorithm": "causal_task_serverCAPQ",
        "family": "learning",
        "beta": 0.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 1.0,
    },
    {
        "label": "guided_correct_ddqn",
        "display_name": "Guided causal Double DQN",
        "algorithm": "causal_task_serverDDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
    },
    {
        "label": "guided_cpqac",
        "display_name": "OUR (Guided CPQAC)",
        "algorithm": "causal_task_serverCPQAC",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
    },
    {
        "label": "pd3qn",
        "display_name": "OUR (Pairwise Dueling DDQN)",
        "algorithm": "causal_task_serverPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
    },
    {
        "label": "telemetry_pd3qn",
        "display_name": "Causal Telemetry PD3QN",
        "algorithm": "causal_telemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
    },
    {
        "label": LEAN_OUR_LABEL,
        "display_name": "OUR (Lean Telemetry PD3QN)",
        "algorithm": "causal_telemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "pairwise_dueling_double_dqn",
            "causal_history_telemetry",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
        "training_objective": "undiscounted_makespan",
        "excluded_modules": [
            "hindsight_critical_path_replay",
            "bottleneck_contribution_replay",
            "workload_normalized_telemetry",
            "quantile_risk_head",
            "entropy_regularization",
            "historical_feedback_guidance",
            "adaptive_guidance_gate",
        ],
    },
    {
        "label": BASE_DDQN_STD_LABEL,
        "display_name": "Base-DDQN (Standard Cache)",
        "algorithm": "causal_telemetryDDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "popularity_ema",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "flat_double_dqn",
            "causal_dag_state_features",
            "causal_history_telemetry",
            "independent_popularity_ema_cache",
        ],
        "training_objective": "undiscounted_makespan",
    },
    {
        "label": OUR_FLAT_DDQN_LABEL,
        "display_name": "OUR-FlatDDQN",
        "algorithm": "causal_telemetryDDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "flat_double_dqn",
            "causal_dag_state_features",
            "causal_history_telemetry",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
        "training_objective": "undiscounted_makespan",
    },
    {
        "label": OUR_NO_TASK_DEPENDENCY_LABEL,
        "display_name": "OUR-noTaskDependency",
        "algorithm": "causal_telemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "task_dependency_features": False,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "pairwise_dueling_double_dqn",
            "causal_history_telemetry",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
        "excluded_modules": ["causal_dag_state_features"],
        "training_objective": "undiscounted_makespan",
    },
    {
        "label": OUR_NO_DEPENDENCY_CACHE_LABEL,
        "display_name": "OUR-noDependencyCache",
        "algorithm": "causal_telemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "cache_dependency_awareness": False,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "pairwise_dueling_double_dqn",
            "causal_dag_state_features",
            "causal_history_telemetry",
            "coordinated_popularity_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
        "excluded_modules": [
            "cache_criticality_weighting",
            "cache_predecessor_locality_weighting",
        ],
        "training_objective": "undiscounted_makespan",
    },
    {
        "label": OUR_TERMINAL_REWARD_LABEL,
        "display_name": "OUR-TerminalReward",
        "algorithm": "causal_telemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "terminal_binary",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "pairwise_dueling_double_dqn",
            "causal_dag_state_features",
            "causal_history_telemetry",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
        "excluded_modules": ["causal_makespan_increment_reward"],
        "training_objective": "terminal_deadline_reward",
    },
    {
        "label": OUR_DQN_LABEL,
        "display_name": "OUR-DQN",
        "algorithm": "causal_telemetryDDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "flat_double_dqn",
            "causal_history_telemetry",
            "causal_dependency_aware_joint_cache",
        ],
        "training_objective": "undiscounted_makespan",
    },
    {
        "label": COORD_DISCRETE_SAC_LABEL,
        "display_name": "CoordCache-DiscreteSAC",
        "algorithm": "causal_telemetryDiscreteSAC",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_coverage_constraint": True,
        "gamma": 1.0,
        "n_step": 3,
        "entropy_coefficient": 0.02,
        "sac_target_entropy_ratio": 0.98,
        "sac_target_tau": 0.005,
        "historical_feedback_guidance": False,
        "adaptive_guidance_gate": False,
        "active_modules": [
            "pairwise_categorical_discrete_sac",
            "automatic_entropy_temperature",
            "causal_history_telemetry",
            "causal_dependency_aware_joint_cache",
            "scarcity_aware_service_coverage_constraint",
        ],
        "training_objective": "undiscounted_makespan",
    },
    {
        "label": "centralized_greedy_daoc",
        "display_name": "Centralized-GreedyCache + DAOC-DQN",
        "algorithm": "prev_servers_plus_service_per_serverDQN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "terminal_binary",
        "cache_policy": "popularity_coordinated",
    },
    {
        "label": "our_no_telemetry",
        "display_name": "OUR-noTelemetry",
        "algorithm": "causal_task_serverPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "critical_path_joint",
        "cache_server_quality": False,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
    },
    {
        "label": "our_no_coord_cache",
        "display_name": "OUR-noCoordCache",
        "algorithm": "causal_telemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_makespan_increment",
        "potential_reward_weight": 0.0,
        "cache_policy": "popularity_ema",
        "cache_server_quality": True,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 1,
        "risk_tail_fraction": 1.0,
        "entropy_coefficient": 0.0,
        "priority_alpha": 0.0,
        "priority_beta_start": 0.0,
        "priority_beta_anneal_steps": 1,
        "criticality_boost": 0.0,
    },
    {
        "label": "normalized_telemetry_pd3qn",
        "display_name": "Workload-Normalized Telemetry PD3QN",
        "algorithm": "causal_normalizedTelemetryPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
    },
    {
        "label": "hcpr_pd3qn",
        "display_name": "HCPR-only PD3QN",
        "algorithm": "causal_task_serverHCPRPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "priority_alpha": 0.6,
        "priority_beta_start": 0.4,
        "priority_beta_anneal_steps": 2000,
        "criticality_boost": 2.0,
        "hcpr_temperature": 0.05,
    },
    {
        "label": "hcpr_telemetry_pd3qn",
        "display_name": "OUR (HCPR + Causal Telemetry PD3QN)",
        "algorithm": "causal_telemetryHCPRPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "priority_alpha": 0.6,
        "priority_beta_start": 0.4,
        "priority_beta_anneal_steps": 2000,
        "criticality_boost": 2.0,
        "hcpr_temperature": 0.05,
    },
    {
        "label": "bcr_pd3qn",
        "display_name": "Bottleneck-Contribution Replay PD3QN",
        "algorithm": "causal_task_serverBCRPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "priority_alpha": 0.6,
        "priority_beta_start": 0.4,
        "priority_beta_anneal_steps": 2000,
        "criticality_boost": 2.0,
        "hcpr_temperature": 0.05,
        "bcr_top_fraction": 0.25,
    },
    {
        "label": "normalized_bcr_pd3qn",
        "display_name": "OUR (Normalized Telemetry + BCR PD3QN)",
        "algorithm": "causal_normalizedTelemetryBCRPD3QN",
        "family": "learning",
        "beta": 0.1,
        "beta_min": 0.1,
        "beta_decay": 1.0,
        "reward_mode": "causal_critical_path",
        "cache_policy": "critical_path_joint",
        "gamma": 1.0,
        "n_step": 3,
        "priority_alpha": 0.6,
        "priority_beta_start": 0.4,
        "priority_beta_anneal_steps": 2000,
        "criticality_boost": 2.0,
        "hcpr_temperature": 0.05,
        "bcr_top_fraction": 0.25,
    },
]


PROFILES = {
    "quick": {
        "train_episodes": 40,
        "eval_episodes": 10,
        "num_users": 3,
        "num_servers": 3,
        "num_services": 3,
        "num_tasks": 6,
        "batch_size": 16,
        "min_experiences": 16,
        "filling_steps": 5,
        "steps_to_updates": 2,
        "max_explore": 30,
        "seeds": [1],
    },
    "pilot": {
        "train_episodes": 3000,
        "eval_episodes": 200,
        "num_users": 5,
        "num_servers": 5,
        "num_services": 5,
        "num_tasks": 10,
        "batch_size": 128,
        "min_experiences": 128,
        "filling_steps": 100,
        "steps_to_updates": 20,
        "max_explore": 1000,
        "seeds": [1, 2, 3],
    },
    "paper_lite": {
        "train_episodes": 10000,
        "eval_episodes": 500,
        "num_users": 10,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 512,
        "min_experiences": 512,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 5000,
        "seeds": [1, 2, 3, 4, 5],
    },
    "paper_audit": {
        "train_episodes": 30000,
        "eval_episodes": 500,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 1024,
        "min_experiences": 1024,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 20000,
        "learning_rate": 0.001,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "seeds": list(range(1, 11)),
        "labels": [
            "nearest",
            "greedy",
            "unguided_full",
            "guided_full",
            "guided_decay",
        ],
    },
    "paper_dual_smoke": {
        "train_episodes": 20,
        "eval_episodes": 4,
        "checkpoint_every": 20,
        "validation_scenarios": 2,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 1024,
        "min_experiences": 1024,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 20000,
        "learning_rate": 0.001,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "seeds": [1],
        "labels": ["guided_full", "our"],
        "eval_scenario_bank": True,
        "cache_freeze_episode": 0,
    },
    "paper_dual_converged": {
        "train_episodes": 40000,
        "eval_episodes": 100,
        "checkpoint_every": 1000,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 1024,
        "min_experiences": 1024,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 20000,
        "learning_rate": 0.001,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "seeds": list(range(1, 11)),
        "labels": ["guided_full", "our"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "cosine",
        "learning_rate_decay_start": 5000,
        "learning_rate_decay_end": 20000,
        "min_learning_rate": 1e-5,
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 15000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "cp_rl_smoke": {
        "train_episodes": 200,
        "eval_episodes": 20,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
        "seeds": [1],
        "labels": ["correct_ddqn", "cpqac"],
        "eval_scenario_bank": True,
        "cache_freeze_episode": 0,
    },
    "cp_rl_calibration": {
        "train_episodes": 40000,
        "eval_episodes": 100,
        "checkpoint_every": 1000,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 1024,
        "min_experiences": 1024,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 20000,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
        "seeds": [1, 2, 3],
        "labels": ["correct_ddqn", "cpqac"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "cosine",
        "learning_rate_decay_start": 5000,
        "learning_rate_decay_end": 20000,
        "min_learning_rate": 1e-5,
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 15000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "cp_rl_converged": {
        "train_episodes": 40000,
        "eval_episodes": 100,
        "checkpoint_every": 1000,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 1024,
        "min_experiences": 1024,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 20000,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "num_quantiles": 16,
        "risk_tail_fraction": 0.25,
        "entropy_coefficient": 0.02,
        "seeds": list(range(1, 11)),
        "labels": ["correct_ddqn", "cpqac"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "cosine",
        "learning_rate_decay_start": 5000,
        "learning_rate_decay_end": 20000,
        "min_learning_rate": 1e-5,
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 15000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "pd3qn_calibration": {
        "train_episodes": 10000,
        "eval_episodes": 100,
        "checkpoint_every": 500,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": [1, 2, 3],
        "labels": ["pd3qn"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "constant",
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 2000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "pd3qn_converged": {
        "train_episodes": 10000,
        "eval_episodes": 100,
        "checkpoint_every": 500,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": list(range(1, 11)),
        "labels": ["pd3qn"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "constant",
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 2000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "hcpr_pd3qn_calibration": {
        "train_episodes": 10000,
        "eval_episodes": 100,
        "checkpoint_every": 500,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": [1, 2, 3],
        "labels": ["hcpr_telemetry_pd3qn"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "constant",
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 2000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "hcpr_pd3qn_converged": {
        "train_episodes": 10000,
        "eval_episodes": 100,
        "checkpoint_every": 500,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": list(range(1, 11)),
        "labels": ["hcpr_telemetry_pd3qn"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "constant",
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 2000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "hcpr_factorial_smoke": {
        "train_episodes": 200,
        "eval_episodes": 20,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": [1],
        "labels": ["telemetry_pd3qn", "hcpr_pd3qn"],
        "eval_scenario_bank": True,
        "cache_freeze_episode": 0,
    },
    "hcpr_factorial_ablation_3seed": {
        "train_episodes": 10000,
        "eval_episodes": 100,
        "checkpoint_every": 500,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": [1, 2, 3],
        "labels": ["telemetry_pd3qn", "hcpr_pd3qn"],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "constant",
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 2000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
    "bcr_normalized_smoke": {
        "train_episodes": 200,
        "eval_episodes": 20,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": [1],
        "labels": [
            "normalized_telemetry_pd3qn",
            "bcr_pd3qn",
            "normalized_bcr_pd3qn",
        ],
        "eval_scenario_bank": True,
        "cache_freeze_episode": 0,
        "telemetry_min_samples": 5,
        "telemetry_freshness_half_life": 10.0,
        "bcr_top_fraction": 0.25,
    },
    "bcr_normalized_ablation_3seed": {
        "train_episodes": 10000,
        "eval_episodes": 100,
        "checkpoint_every": 500,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "gamma": 1.0,
        "n_step": 3,
        "seeds": [1, 2, 3],
        "labels": [
            "normalized_telemetry_pd3qn",
            "bcr_pd3qn",
            "normalized_bcr_pd3qn",
        ],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "constant",
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 2000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
        "telemetry_min_samples": 5,
        "telemetry_freshness_half_life": 10.0,
        "bcr_top_fraction": 0.25,
    },
    "strict_stress_smoke": {
        "train_episodes": 200,
        "eval_episodes": 20,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 64,
        "min_experiences": 64,
        "filling_steps": 20,
        "steps_to_updates": 5,
        "max_explore": 150,
        "learning_rate": 0.0005,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "seeds": [1],
        "labels": [
            "guided_full",
            LEAN_OUR_LABEL,
        ],
        "eval_scenario_bank": True,
        "cache_freeze_episode": 0,
        "dag_depth_increment": 0,
        "dependency_data_scale": 1.0,
        "server_capacity": 2,
    },
    "strict_stress_converged_3seed": {
        "train_episodes": 50000,
        "eval_episodes": 100,
        "checkpoint_every": 1000,
        "validation_scenarios": 10,
        "num_users": 20,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 1024,
        "min_experiences": 1024,
        "filling_steps": 500,
        "steps_to_updates": 100,
        "max_explore": 20000,
        "learning_rate": 0.001,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "seeds": [1, 2, 3],
        "labels": [
            "guided_full",
            LEAN_OUR_LABEL,
        ],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "cosine",
        "learning_rate_decay_start": 5000,
        "learning_rate_decay_end": 20000,
        "min_learning_rate": 1e-5,
        "cache_freeze_episode": 0,
        "convergence_min_episodes": 15000,
        "convergence_window": 5,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
        "dag_depth_increment": 0,
        "dependency_data_scale": 1.0,
        "server_capacity": 2,
        "method_overrides": {
            LEAN_OUR_LABEL: {
                "train_episodes": 40000,
                "batch_size": 64,
                "min_experiences": 64,
                "filling_steps": 20,
                "steps_to_updates": 5,
                "max_explore": 150,
                "learning_rate": 0.0005,
                "learning_rate_schedule": "constant",
                "checkpoint_every": 500,
                "convergence_min_episodes": 2000,
            },
        },
    },
    "innovation_quick": {
        "train_episodes": 80,
        "eval_episodes": 20,
        "num_users": 3,
        "num_servers": 3,
        "num_services": 3,
        "num_tasks": 6,
        "batch_size": 16,
        "min_experiences": 16,
        "filling_steps": 5,
        "steps_to_updates": 2,
        "max_explore": 60,
        "seeds": [1],
        "labels": [
            "nearest",
            "greedy",
            "guided_full",
            "gate_only",
            "history_only",
            "proposed_full",
        ],
    },
    "innovation_pilot": {
        "train_episodes": 1500,
        "eval_episodes": 200,
        "num_users": 5,
        "num_servers": 5,
        "num_services": 5,
        "num_tasks": 10,
        "batch_size": 128,
        "min_experiences": 128,
        "filling_steps": 100,
        "steps_to_updates": 20,
        "max_explore": 1000,
        "seeds": [1, 2, 3],
        "labels": [
            "nearest",
            "greedy",
            "guided_full",
            "gate_only",
            "history_only",
            "proposed_full",
        ],
    },
    "innovation_main": {
        "train_episodes": 3000,
        "eval_episodes": 300,
        "num_users": 10,
        "num_servers": 10,
        "num_services": 10,
        "num_tasks": 10,
        "batch_size": 256,
        "min_experiences": 256,
        "filling_steps": 100,
        "steps_to_updates": 20,
        "max_explore": 2000,
        "learning_rate": 0.001,
        "epsilon": 0.01,
        "hidden_units": [64, 64],
        "bandwidth": 15000,
        "seeds": [1, 2, 3, 4, 5],
        "labels": [
            "nearest",
            "greedy",
            "guided_full",
            "gate_only",
            "history_only",
            "proposed_full",
        ],
    },
    "cpr_quick": {
        "train_episodes": 100,
        "eval_episodes": 10,
        "num_users": 3,
        "num_servers": 3,
        "num_services": 3,
        "num_tasks": 6,
        "batch_size": 16,
        "min_experiences": 16,
        "filling_steps": 5,
        "steps_to_updates": 2,
        "max_explore": 60,
        "seeds": [1],
        "labels": ["guided_full", "cpr_reward"],
        "eval_scenario_bank": True,
    },
    "cpr_pilot": {
        "train_episodes": 1000,
        "eval_episodes": 30,
        "checkpoint_every": 200,
        "validation_scenarios": 10,
        "num_users": 5,
        "num_servers": 5,
        "num_services": 5,
        "num_tasks": 10,
        "batch_size": 128,
        "min_experiences": 128,
        "filling_steps": 100,
        "steps_to_updates": 20,
        "max_explore": 800,
        "seeds": [1, 2, 3],
        "labels": ["guided_full", "cpr_reward"],
        "eval_scenario_bank": True,
    },
    "cpr_cache_quick": {
        "train_episodes": 120,
        "eval_episodes": 10,
        "num_users": 3,
        "num_servers": 3,
        "num_services": 3,
        "num_tasks": 6,
        "batch_size": 16,
        "min_experiences": 16,
        "filling_steps": 5,
        "steps_to_updates": 2,
        "max_explore": 80,
        "seeds": [1],
        "labels": [
            "guided_full",
            "cpr_reward",
            "cpr_cache",
            "cpr_coord_cache",
            "cpr_joint_cache",
            "hybrid_reward",
            "our",
        ],
        "eval_scenario_bank": True,
    },
    "cpr_cache_pilot": {
        "train_episodes": 1000,
        "eval_episodes": 30,
        "checkpoint_every": 200,
        "validation_scenarios": 10,
        "num_users": 5,
        "num_servers": 5,
        "num_services": 5,
        "num_tasks": 10,
        "batch_size": 128,
        "min_experiences": 128,
        "filling_steps": 100,
        "steps_to_updates": 20,
        "max_explore": 800,
        "seeds": [1, 2, 3],
        "labels": [
            "guided_full",
            "cpr_reward",
            "cpr_cache",
            "cpr_coord_cache",
            "cpr_joint_cache",
            "hybrid_reward",
            "our",
        ],
        "eval_scenario_bank": True,
    },
    "cpr_converged": {
        "train_episodes": 40000,
        "eval_episodes": 30,
        "checkpoint_every": 500,
        "validation_scenarios": 30,
        "num_users": 5,
        "num_servers": 5,
        "num_services": 5,
        "num_tasks": 10,
        "batch_size": 128,
        "min_experiences": 128,
        "filling_steps": 100,
        "steps_to_updates": 20,
        "max_explore": 800,
        "seeds": list(range(1, 11)),
        "labels": [
            "guided_full",
            "cpr_reward",
            "cpr_cache",
            "cpr_coord_cache",
            "cpr_joint_cache",
            "hybrid_reward",
            "our",
        ],
        "eval_scenario_bank": True,
        "convergence_mode": True,
        "learning_rate_schedule": "cosine",
        "learning_rate_decay_start": 5000,
        "learning_rate_decay_end": 20000,
        "min_learning_rate": 1e-5,
        "cache_freeze_episode": 20000,
        "convergence_min_episodes": 20000,
        "convergence_window": 10,
        "convergence_patience": 3,
        "convergence_relative_mean_change": 0.05,
        "convergence_relative_slope": 0.01,
    },
}

E2_CAPACITY_MULTISET = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
PROFILES["e2_smoke"] = {
    **PROFILES["strict_stress_smoke"],
    "server_capacity": 1,
    "server_capacity_multiset": E2_CAPACITY_MULTISET,
}
PROFILES["e2_development"] = {
    **PROFILES["strict_stress_smoke"],
    "train_episodes": 2000,
    "eval_episodes": 50,
    "seeds": [1, 2, 3],
    "server_capacity": 1,
    "server_capacity_multiset": E2_CAPACITY_MULTISET,
}
PROFILES["e2_converged"] = {
    **PROFILES["strict_stress_converged_3seed"],
    "server_capacity": 1,
    "server_capacity_multiset": E2_CAPACITY_MULTISET,
}
PROFILES["e2_converged"]["method_overrides"] = {
    **PROFILES["strict_stress_converged_3seed"].get(
        "method_overrides",
        {},
    ),
    **{
        label: {
            "train_episodes": 1,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "convergence_mode": False,
        }
        for label in ("random", "nearest", "greedy")
    },
}
for _ablation_label in (
    OUR_DQN_LABEL,
    "our_no_telemetry",
    "our_no_coord_cache",
):
    PROFILES["e2_converged"]["method_overrides"][
        _ablation_label
    ] = dict(
        PROFILES["e2_converged"]["method_overrides"][
            LEAN_OUR_LABEL
        ]
    )


PEGASUS_PSCALE_P2_CAPACITY_MULTISET = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
PROFILES["pegasus_pscale_p2_smoke"] = {
    **PROFILES["e2_smoke"],
    "train_episodes": 200,
    "eval_episodes": 20,
    "num_tasks": 31,
    "server_capacity_multiset": PEGASUS_PSCALE_P2_CAPACITY_MULTISET,
    "eval_bank_scope": "infrastructure",
}
PROFILES["pegasus_pscale_p2_converged"] = {
    **PROFILES["e2_converged"],
    "num_tasks": 31,
    "validation_scenarios": 50,
    "cache_freeze_episode": 5000,
    "server_capacity_multiset": PEGASUS_PSCALE_P2_CAPACITY_MULTISET,
    "eval_bank_scope": "infrastructure",
    "method_overrides": {
        label: {
            **override,
            **(
                {
                    "validation_scenarios": 50,
                    "cache_freeze_episode": 5000,
                    "convergence_min_episodes": 5000,
                }
                if label == LEAN_OUR_LABEL
                else {}
            ),
        }
        for label, override in PROFILES["e2_converged"]
        .get("method_overrides", {})
        .items()
    },
}

PEGASUS_PAPER_CLOSURE_LABELS = (
    DAOC_PAPER_LABEL,
    OUR_DQN_LABEL,
    "our_no_telemetry",
    "our_no_coord_cache",
)
PROFILES["pegasus_paper_closure_smoke"] = {
    **PROFILES["pegasus_pscale_p2_smoke"],
    "labels": list(PEGASUS_PAPER_CLOSURE_LABELS),
}
PROFILES["pegasus_paper_closure_converged"] = {
    **PROFILES["pegasus_pscale_p2_converged"],
    "labels": list(PEGASUS_PAPER_CLOSURE_LABELS),
    "method_overrides": {
        **PROFILES["pegasus_pscale_p2_converged"].get(
            "method_overrides",
            {},
        ),
        **{
            label: {
                **PROFILES["e2_converged"]["method_overrides"][
                    LEAN_OUR_LABEL
                ],
                "validation_scenarios": 50,
                "cache_freeze_episode": 5000,
                "convergence_min_episodes": 5000,
            }
            for label in (
                OUR_DQN_LABEL,
                "our_no_telemetry",
                "our_no_coord_cache",
            )
        },
    },
}


PEGASUS_BASELINE_HEURISTICS = (
    COORD_RANDOM_LABEL,
    COORD_NEAREST_LABEL,
    COORD_NEAREST_SERVICE_LABEL,
)
PROFILES["pegasus_baseline_heuristics"] = {
    **PROFILES["pegasus_pscale_p2_converged"],
    "labels": list(PEGASUS_BASELINE_HEURISTICS),
    "method_overrides": {
        label: {
            "train_episodes": 5000,
            "eval_episodes": 100,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "convergence_mode": False,
            "cache_freeze_episode": 5000,
        }
        for label in PEGASUS_BASELINE_HEURISTICS
    },
}
PROFILES["pegasus_baseline_sac_smoke"] = {
    **PROFILES["pegasus_pscale_p2_smoke"],
    "labels": [COORD_DISCRETE_SAC_LABEL],
}
PROFILES["pegasus_baseline_sac_screen"] = {
    **PROFILES["pegasus_pscale_p2_smoke"],
    "train_episodes": 2000,
    "eval_episodes": 50,
    "labels": [COORD_DISCRETE_SAC_LABEL],
}
PROFILES["pegasus_baseline_sac_converged"] = {
    **PROFILES["pegasus_pscale_p2_converged"],
    "labels": [COORD_DISCRETE_SAC_LABEL],
    "method_overrides": {
        COORD_DISCRETE_SAC_LABEL: {
            **PROFILES["e2_converged"]["method_overrides"][
                LEAN_OUR_LABEL
            ],
            "validation_scenarios": 50,
            "cache_freeze_episode": 5000,
            "convergence_min_episodes": 5000,
            "convergence_mode": True,
        },
    },
}


PEGASUS_P6_HEURISTIC_LABELS = (
    "random",
    "nearest",
    "greedy",
)
PEGASUS_P6_LEARNING_LABELS = (
    DQN_WDSA_STD_LABEL,
    BASE_DDQN_STD_LABEL,
    OUR_FLAT_DDQN_LABEL,
    OUR_NO_TASK_DEPENDENCY_LABEL,
    OUR_NO_DEPENDENCY_CACHE_LABEL,
    OUR_TERMINAL_REWARD_LABEL,
)
PROFILES["pegasus_p6_smoke"] = {
    **PROFILES["pegasus_pscale_p2_smoke"],
    "labels": list(
        PEGASUS_P6_HEURISTIC_LABELS
        + PEGASUS_P6_LEARNING_LABELS
    ),
}
PROFILES["pegasus_p6_heuristics"] = {
    **PROFILES["pegasus_pscale_p2_converged"],
    "labels": list(PEGASUS_P6_HEURISTIC_LABELS),
    "method_overrides": {
        label: {
            "train_episodes": 5000,
            "eval_episodes": 100,
            "checkpoint_every": 0,
            "validation_scenarios": 0,
            "convergence_mode": False,
            "cache_freeze_episode": 5000,
        }
        for label in PEGASUS_P6_HEURISTIC_LABELS
    },
}
PROFILES["pegasus_p6_learning_converged"] = {
    **PROFILES["pegasus_pscale_p2_converged"],
    "labels": list(PEGASUS_P6_LEARNING_LABELS),
    "method_overrides": {
        **PROFILES["pegasus_pscale_p2_converged"].get(
            "method_overrides",
            {},
        ),
        **{
            label: {
                **PROFILES["e2_converged"]["method_overrides"][
                    LEAN_OUR_LABEL
                ],
                "validation_scenarios": 50,
                "cache_freeze_episode": 5000,
                "convergence_min_episodes": 5000,
                "convergence_mode": True,
            }
            for label in PEGASUS_P6_LEARNING_LABELS
            if label != DQN_WDSA_STD_LABEL
        },
    },
}


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run isolated baseline reproduction experiments."
    )
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--seeds", type=parse_int_list)
    parser.add_argument("--labels", type=parse_str_list)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--num-servers", type=int)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--eval-dag-families", type=parse_str_list)
    parser.add_argument("--dag-dataset-path")
    parser.add_argument("--dag-dataset-sha256")
    parser.add_argument("--dag-depth-increment", type=int)
    parser.add_argument("--dependency-data-scale", type=float)
    parser.add_argument("--server-capacity", type=int)
    parser.add_argument(
        "--server-capacity-multiset",
        type=parse_int_list,
    )
    parser.add_argument("--capacity-assignment-namespace")
    parser.add_argument("--baseline-server-capacity", type=int)
    parser.add_argument("--workers", type=int, default=1)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def is_complete(run_dir, require_convergence=False):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            return False
        if (
            summary.get("information_protocol_version")
            != INFORMATION_PROTOCOL_VERSION
        ):
            return False
        return (
            summary.get("eligible_for_comparison", False)
            if require_convergence
            else True
        )
    except (OSError, json.JSONDecodeError):
        return False


def effective_method_profile(profile, label):
    effective = dict(profile)
    effective.update(
        profile.get("method_overrides", {}).get(label, {})
    )
    effective.pop("method_overrides", None)
    return effective


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    profile = dict(PROFILES[args.profile])
    if args.seeds:
        profile["seeds"] = args.seeds
    if args.train_episodes is not None:
        profile["train_episodes"] = args.train_episodes
    if args.eval_episodes is not None:
        profile["eval_episodes"] = args.eval_episodes
    if args.num_servers is not None:
        if args.num_servers < 1:
            raise ValueError("--num-servers must be positive")
        profile["num_servers"] = args.num_servers
    if args.num_tasks is not None:
        if args.num_tasks < 1:
            raise ValueError("--num-tasks must be positive")
        profile["num_tasks"] = args.num_tasks
    if args.eval_dag_families is not None:
        if len(args.eval_dag_families) != len(
            set(args.eval_dag_families)
        ):
            raise ValueError("--eval-dag-families must be unique")
        profile["eval_dag_families"] = args.eval_dag_families
    if args.dag_dataset_path is not None:
        dag_dataset_path = Path(args.dag_dataset_path).expanduser()
        if not dag_dataset_path.is_absolute():
            dag_dataset_path = root_dir / dag_dataset_path
        dag_dataset_path = dag_dataset_path.resolve()
        if not dag_dataset_path.is_file():
            raise ValueError(
                f"--dag-dataset-path does not exist: {dag_dataset_path}"
            )
        profile["dag_dataset_path"] = str(dag_dataset_path)
    if args.dag_dataset_sha256 is not None:
        dag_dataset_sha256 = args.dag_dataset_sha256.lower()
        if (
            len(dag_dataset_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in dag_dataset_sha256
            )
        ):
            raise ValueError(
                "--dag-dataset-sha256 must contain 64 hexadecimal digits"
            )
        profile["dag_dataset_sha256"] = dag_dataset_sha256
    if args.dag_depth_increment is not None:
        profile["dag_depth_increment"] = args.dag_depth_increment
    if args.dependency_data_scale is not None:
        profile["dependency_data_scale"] = (
            args.dependency_data_scale
        )
    if args.server_capacity is not None:
        profile["server_capacity"] = args.server_capacity
    if args.server_capacity_multiset is not None:
        profile["server_capacity_multiset"] = (
            args.server_capacity_multiset
        )
    if args.capacity_assignment_namespace is not None:
        profile["capacity_assignment_namespace"] = (
            args.capacity_assignment_namespace
        )
    if args.baseline_server_capacity is not None:
        profile["baseline_server_capacity"] = (
            args.baseline_server_capacity
        )

    selected_algorithms = ALGORITHMS
    requested_labels = args.labels or profile.get("labels")
    if requested_labels:
        requested = set(requested_labels)
        selected_algorithms = [
            config for config in ALGORITHMS if config["label"] in requested
        ]
        missing = requested - {
            config["label"] for config in selected_algorithms
        }
        if missing:
            raise ValueError(f"Unknown labels: {sorted(missing)}")

    suite_dir = (
        args.suite_dir
        if args.suite_dir is not None
        else root_dir / "results" / f"reproduction_{args.profile}"
    ).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = suite_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)

    manifest_path = suite_dir / "suite_manifest.json"
    manifest = {
        "status": "running",
        "profile": args.profile,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "information_protocol_version": (
            INFORMATION_PROTOCOL_VERSION
        ),
        "profile_config": profile,
        "algorithms": selected_algorithms,
        "workers": args.workers,
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
        "completed_runs": 0,
        "failed_runs": [],
    }
    write_manifest(manifest_path, manifest)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(matplotlib_dir)
    failures = []
    completed = 0
    total = len(selected_algorithms) * len(profile["seeds"])
    suite_started = time.perf_counter()

    pending_runs = []
    for algorithm_config in selected_algorithms:
        for seed in profile["seeds"]:
            label = algorithm_config["label"]
            run_dir = suite_dir / "runs" / label / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.resume and is_complete(
                run_dir,
                require_convergence=profile.get(
                    "convergence_mode",
                    False,
                ),
            ):
                completed += 1
                print(f"[{completed}/{total}] skip {label} seed={seed}")
                continue
            pending_runs.append((algorithm_config, seed, run_dir))

    def execute_run(run_spec):
        algorithm_config, seed, run_dir = run_spec
        label = algorithm_config["label"]
        method_profile = effective_method_profile(profile, label)
        command = [
            sys.executable,
            str(root_dir / "run_independent_experiment.py"),
            "--result-dir",
            str(run_dir),
            "--label",
            label,
            "--algorithm",
            algorithm_config["algorithm"],
            "--seed",
            str(seed),
            "--revision-id",
            args.revision_id,
            "--revision-reason",
            args.revision_reason,
            "--revision-changed-module",
            args.revision_changed_module,
            "--revision-expected-metric",
            args.revision_expected_metric,
            "--revision-rejection-condition",
            args.revision_rejection_condition,
            "--seed-partition",
            args.seed_partition,
            "--train-episodes",
            str(method_profile["train_episodes"]),
            "--eval-episodes",
            str(method_profile["eval_episodes"]),
            "--num-users",
            str(method_profile["num_users"]),
            "--num-servers",
            str(method_profile["num_servers"]),
            "--num-services",
            str(method_profile["num_services"]),
            "--num-tasks",
            str(method_profile["num_tasks"]),
            "--dag-depth-increment",
            str(method_profile.get("dag_depth_increment", 0)),
            "--dependency-data-scale",
            str(method_profile.get("dependency_data_scale", 1.0)),
            "--server-capacity",
            str(method_profile.get("server_capacity", 2)),
            "--baseline-server-capacity",
            str(
                method_profile.get(
                    "baseline_server_capacity",
                    2,
                )
            ),
            "--batch-size",
            str(method_profile["batch_size"]),
            "--min-experiences",
            str(method_profile["min_experiences"]),
            "--filling-steps",
            str(method_profile["filling_steps"]),
            "--steps-to-updates",
            str(method_profile["steps_to_updates"]),
            "--max-explore",
            str(method_profile["max_explore"]),
            "--gamma",
            str(
                algorithm_config.get(
                    "gamma",
                    method_profile.get("gamma", 0.9),
                )
            ),
            "--n-step",
            str(
                algorithm_config.get(
                    "n_step",
                    method_profile.get("n_step", 3),
                )
            ),
            "--num-quantiles",
            str(
                algorithm_config.get(
                    "num_quantiles",
                    method_profile.get("num_quantiles", 16),
                )
            ),
            "--risk-tail-fraction",
            str(
                algorithm_config.get(
                    "risk_tail_fraction",
                    method_profile.get("risk_tail_fraction", 0.25),
                )
            ),
            "--entropy-coefficient",
            str(
                algorithm_config.get(
                    "entropy_coefficient",
                    method_profile.get("entropy_coefficient", 0.02),
                )
            ),
            "--sac-target-entropy-ratio",
            str(
                algorithm_config.get(
                    "sac_target_entropy_ratio",
                    method_profile.get(
                        "sac_target_entropy_ratio",
                        0.98,
                    ),
                )
            ),
            "--sac-target-tau",
            str(
                algorithm_config.get(
                    "sac_target_tau",
                    method_profile.get("sac_target_tau", 0.005),
                )
            ),
            "--priority-alpha",
            str(
                algorithm_config.get(
                    "priority_alpha",
                    method_profile.get("priority_alpha", 0.6),
                )
            ),
            "--priority-beta-start",
            str(
                algorithm_config.get(
                    "priority_beta_start",
                    method_profile.get("priority_beta_start", 0.4),
                )
            ),
            "--priority-beta-anneal-steps",
            str(
                algorithm_config.get(
                    "priority_beta_anneal_steps",
                    method_profile.get(
                        "priority_beta_anneal_steps",
                        2000,
                    ),
                )
            ),
            "--criticality-boost",
            str(
                algorithm_config.get(
                    "criticality_boost",
                    method_profile.get("criticality_boost", 2.0),
                )
            ),
            "--hcpr-temperature",
            str(
                algorithm_config.get(
                    "hcpr_temperature",
                    method_profile.get("hcpr_temperature", 0.05),
                )
            ),
            "--bcr-top-fraction",
            str(
                algorithm_config.get(
                    "bcr_top_fraction",
                    method_profile.get("bcr_top_fraction", 0.25),
                )
            ),
            "--learning-rate",
            str(method_profile.get("learning_rate", 0.001)),
            "--learning-rate-schedule",
            method_profile.get(
                "learning_rate_schedule",
                "constant",
            ),
            "--learning-rate-decay-start",
            str(
                method_profile.get(
                    "learning_rate_decay_start",
                    5000,
                )
            ),
            "--learning-rate-decay-end",
            str(
                method_profile.get(
                    "learning_rate_decay_end",
                    20000,
                )
            ),
            "--min-learning-rate",
            str(method_profile.get("min_learning_rate", 1e-5)),
            "--epsilon",
            str(method_profile.get("epsilon", 0.001)),
            "--hidden-units",
            ",".join(
                str(unit)
                for unit in method_profile.get("hidden_units", [64])
            ),
            "--bandwidth",
            str(method_profile.get("bandwidth", 1_000_000)),
            "--beta",
            str(algorithm_config["beta"]),
            "--beta-min",
            str(algorithm_config.get("beta_min", algorithm_config["beta"])),
            "--beta-decay",
            str(algorithm_config.get("beta_decay", 1.0)),
            "--reward-mode",
            algorithm_config.get("reward_mode", "terminal_binary"),
            "--reward-scale",
            str(method_profile.get("reward_scale", 1.0)),
            "--potential-reward-weight",
            str(
                algorithm_config.get(
                    "potential_reward_weight",
                    method_profile.get("potential_reward_weight", 0.5),
                )
            ),
            "--cache-policy",
            algorithm_config.get("cache_policy", "popularity_ema"),
            "--cache-score-alpha",
            str(method_profile.get("cache_score_alpha", 0.1)),
            "--cache-history-alpha",
            str(method_profile.get("cache_history_alpha", 0.1)),
            "--cache-locality-weight",
            str(method_profile.get("cache_locality_weight", 1.0)),
            "--cache-update-interval",
            str(method_profile.get("cache_update_interval", 5)),
            "--cache-hysteresis-factor",
            str(method_profile.get("cache_hysteresis_factor", 1.0)),
            "--cache-min-residence-updates",
            str(method_profile.get("cache_min_residence_updates", 2)),
            "--cache-freeze-episode",
            str(method_profile.get("cache_freeze_episode", 0)),
            "--cache-compute-weight",
            str(method_profile.get("cache_compute_weight", 1.0)),
            "--telemetry-min-samples",
            str(method_profile.get("telemetry_min_samples", 5)),
            "--telemetry-freshness-half-life",
            str(
                method_profile.get(
                    "telemetry_freshness_half_life",
                    10.0,
                )
            ),
            "--checkpoint-every",
            str(method_profile.get("checkpoint_every", 0)),
            "--validation-scenarios",
            str(method_profile.get("validation_scenarios", 0)),
            "--convergence-min-episodes",
            str(
                method_profile.get(
                    "convergence_min_episodes",
                    5000,
                )
            ),
            "--convergence-window",
            str(method_profile.get("convergence_window", 5)),
            "--convergence-patience",
            str(method_profile.get("convergence_patience", 3)),
            "--convergence-relative-mean-change",
            str(
                method_profile.get(
                    "convergence_relative_mean_change",
                    0.05,
                )
            ),
            "--convergence-relative-slope",
            str(
                method_profile.get(
                    "convergence_relative_slope",
                    0.01,
                )
            ),
            "--history-feedback-alpha",
            str(method_profile.get("history_feedback_alpha", 0.1)),
            "--history-feedback-min-samples",
            str(method_profile.get("history_feedback_min_samples", 3)),
            "--history-feedback-max-probability",
            str(method_profile.get("history_feedback_max_probability", 0.9)),
            "--history-feedback-fixed-probability",
            str(
                method_profile.get(
                    "history_feedback_fixed_probability",
                    0.1,
                )
            ),
            "--quiet",
        ]
        if args.revision_parent:
            command.extend(
                ["--revision-parent", args.revision_parent]
            )
        if method_profile.get("dag_dataset_path") is not None:
            command.extend(
                [
                    "--dag-dataset-path",
                    str(method_profile["dag_dataset_path"]),
                ]
            )
        if method_profile.get("dag_dataset_sha256") is not None:
            command.extend(
                [
                    "--dag-dataset-sha256",
                    str(method_profile["dag_dataset_sha256"]),
                ]
            )
        if method_profile.get("server_capacity_multiset") is not None:
            command.extend(
                [
                    "--server-capacity-multiset",
                    ",".join(
                        str(value)
                        for value in method_profile[
                            "server_capacity_multiset"
                        ]
                    ),
                ]
            )
        if method_profile.get(
            "capacity_assignment_namespace"
        ) is not None:
            command.extend(
                [
                    "--capacity-assignment-namespace",
                    str(
                        method_profile[
                            "capacity_assignment_namespace"
                        ]
                    ),
                ]
            )
        if not algorithm_config.get(
            "cache_server_quality",
            True,
        ):
            command.append("--no-cache-server-quality")
        if algorithm_config.get(
            "cache_coverage_constraint",
            False,
        ):
            command.append("--cache-coverage-constraint")
        if not algorithm_config.get(
            "task_dependency_features",
            True,
        ):
            command.append("--no-task-dependency-features")
        if not algorithm_config.get(
            "cache_dependency_awareness",
            True,
        ):
            command.append("--no-cache-dependency-awareness")
        if algorithm_config.get("historical_feedback_guidance", False):
            command.append("--historical-feedback-guidance")
        if algorithm_config.get("adaptive_guidance_gate", False):
            command.append("--adaptive-guidance-gate")
        if method_profile.get("eval_scenario_bank", False):
            command.append("--eval-scenario-bank")
            command.extend(
                [
                    "--eval-bank-scope",
                    method_profile.get("eval_bank_scope", "workload"),
                ]
            )
        if method_profile.get("eval_dag_families") is not None:
            command.extend(
                [
                    "--eval-dag-families",
                    ",".join(method_profile["eval_dag_families"]),
                ]
            )
        if method_profile.get("convergence_mode", False):
            command.append("--convergence-mode")
        run_started = time.perf_counter()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=root_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {
            "label": label,
            "seed": seed,
            "run_dir": run_dir,
            "command": command,
            "returncode": result.returncode,
            "elapsed": time.perf_counter() - run_started,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(execute_run, run_spec)
            for run_spec in pending_runs
        ]
        for future in as_completed(futures):
            outcome = future.result()
            if outcome["returncode"] != 0:
                failure = {
                    "label": outcome["label"],
                    "seed": outcome["seed"],
                    "returncode": outcome["returncode"],
                }
                failures.append(failure)
                print(
                    f"[failed] {outcome['label']} seed={outcome['seed']} "
                    f"exit={outcome['returncode']}; "
                    f"see {outcome['run_dir'] / 'run.log'}"
                )
            else:
                completed += 1
                print(
                    f"[{completed}/{total}] {outcome['label']} "
                    f"seed={outcome['seed']} "
                    f"finished in {outcome['elapsed']:.1f}s"
                )
            manifest["completed_runs"] = completed
            manifest["failed_runs"] = failures
            write_manifest(manifest_path, manifest)

    if failures and not args.keep_going:
        manifest["status"] = "failed"
        write_manifest(manifest_path, manifest)
        first_failure = failures[0]
        raise RuntimeError(
            "Experiment failed: "
            f"{first_failure['label']} seed={first_failure['seed']}"
        )

    nonconverged = []
    if profile.get("convergence_mode", False):
        for algorithm_config in selected_algorithms:
            for seed in profile["seeds"]:
                summary_path = (
                    suite_dir
                    / "runs"
                    / algorithm_config["label"]
                    / f"seed_{seed}"
                    / "summary.json"
                )
                if not summary_path.exists():
                    nonconverged.append(
                        {
                            "label": algorithm_config["label"],
                            "seed": seed,
                            "reason": "missing_summary",
                        }
                    )
                    continue
                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                if not summary.get("eligible_for_comparison", False):
                    nonconverged.append(
                        {
                            "label": algorithm_config["label"],
                            "seed": seed,
                            "reason": summary.get("convergence", {}).get(
                                "stop_reason",
                                "not_converged",
                            ),
                        }
                    )
        manifest["converged_runs"] = total - len(nonconverged)
        manifest["nonconverged_runs"] = nonconverged

    if nonconverged:
        manifest["status"] = "incomplete_convergence"
        manifest["completed_runs"] = completed
        manifest["failed_runs"] = failures
        manifest["total_wall_time_sec"] = time.perf_counter() - suite_started
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_manifest(manifest_path, manifest)
        print(
            f"Convergence incomplete for {len(nonconverged)}/{total} runs; "
            f"see {manifest_path}"
        )
        return

    aggregate_command = [
        sys.executable,
        str(root_dir / "aggregate_reproduction_results.py"),
        "--suite-dir",
        str(suite_dir),
    ]
    subprocess.run(
        aggregate_command,
        cwd=root_dir,
        env=env,
        check=True,
    )

    manifest["status"] = "complete" if not failures else "partial"
    manifest["completed_runs"] = completed
    manifest["failed_runs"] = failures
    manifest["total_wall_time_sec"] = time.perf_counter() - suite_started
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_manifest(manifest_path, manifest)
    print(f"Suite results: {suite_dir}")


if __name__ == "__main__":
    main()
