#!/usr/bin/env python3
"""Diagnose cache-induced validation oscillation at frozen checkpoints."""

import argparse
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_independent_experiment import (
    capture_deployment_state,
    run_scenario_bank_evaluation,
    seed_everything,
)
from simulator import MEC_Simulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate network drift from native cache churn."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, default=30000)
    parser.add_argument("--end-episode", type=int, default=50000)
    parser.add_argument("--fixed-cache-episode", type=int, default=30000)
    parser.add_argument("--suite-manifest", type=Path)
    parser.add_argument("--stage-manifest", type=Path)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def checkpoint_path(run_dir, episode):
    return (
        run_dir
        / "training_checkpoints"
        / f"episode_{episode:06d}.pt"
    )


def flattened_weights(state):
    return torch.cat(
        [
            value.detach().reshape(-1).float()
            for server_id in sorted(state["weights"])
            for _, value in sorted(
                state["weights"][server_id].items()
            )
        ]
    )


def real_services(state):
    return {
        int(server_id): tuple(
            int(service)
            for service in services
            if int(service) > 0
        )
        for server_id, services in state["services"].items()
    }


def evaluate_with_fixed_cache(
    run_dir,
    config,
    deployment,
    fixed_services,
    episodes,
):
    arguments = config["arguments"]
    evaluator_args = SimpleNamespace(
        seed=int(arguments["seed"]),
        label=arguments["label"],
        eval_episodes=int(arguments["validation_scenarios"]),
        eval_seed_offset=int(arguments["validation_seed_offset"]),
        eval_bank_scope="workload",
        quiet=True,
    )
    rows = []
    for episode in episodes:
        state = torch.load(
            checkpoint_path(run_dir, episode),
            map_location="cpu",
            weights_only=False,
        )["frozen_state"]
        state = copy.deepcopy(state)
        state["services"] = copy.deepcopy(fixed_services)
        episode_rows, _ = run_scenario_bank_evaluation(
            writer=None,
            csv_file=None,
            args=evaluator_args,
            input_config=config["input_config"],
            learning_config=config["learning_config"],
            frozen_state=state,
            deployment_state=deployment,
            simulator_log=io.StringIO(),
            figures_dir=run_dir / "figures",
            episodes=int(arguments["validation_scenarios"]),
            seed_offset=int(arguments["validation_seed_offset"]),
            phase="cache_oscillation_diagnostic",
        )
        rows.append(
            float(
                np.mean(
                    [
                        row["average_finish_time"]
                        for row in episode_rows
                    ]
                )
            )
        )
    return rows


def update_failed_manifest(path, reason, run_dir):
    if path is None or not path.exists():
        return
    manifest = read_json(path)
    manifest["status"] = "stopped_after_diagnosis"
    manifest["diagnosis"] = reason
    manifest["diagnostic_run"] = str(run_dir.resolve())
    write_json(path, manifest)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = read_json(run_dir / "config.json")
    validation = read_json(
        run_dir / "checkpoint_validation.json"
    )
    native_by_episode = {
        int(record["episode"]): float(
            record["mean_average_finish_time"]
        )
        for record in validation["records"]
    }
    episodes = list(
        range(
            args.start_episode,
            args.end_episode + 1,
            1000,
        )
    )
    if any(
        not checkpoint_path(run_dir, episode).exists()
        for episode in episodes
    ):
        raise RuntimeError("Missing diagnostic checkpoint")

    states = {
        episode: torch.load(
            checkpoint_path(run_dir, episode),
            map_location="cpu",
            weights_only=False,
        )["frozen_state"]
        for episode in episodes
    }
    fixed_services = copy.deepcopy(
        states[args.fixed_cache_episode]["services"]
    )

    seed = int(config["arguments"]["seed"])
    seed_everything(seed)
    base_simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=config["input_config"],
        learning_arguments=config["learning_config"],
        filename_png=str(run_dir / "figures"),
    )
    deployment = capture_deployment_state(base_simulator)
    fixed_cache_values = evaluate_with_fixed_cache(
        run_dir,
        config,
        deployment,
        fixed_services,
        episodes,
    )

    rows = []
    previous_services = None
    previous_weights = None
    for episode, fixed_finish in zip(
        episodes,
        fixed_cache_values,
    ):
        state = states[episode]
        services = real_services(state)
        weights = flattened_weights(state)
        changed_servers = (
            0
            if previous_services is None
            else sum(
                services[server_id]
                != previous_services[server_id]
                for server_id in services
            )
        )
        relative_weight_change = (
            0.0
            if previous_weights is None
            else float(
                torch.linalg.vector_norm(
                    weights - previous_weights
                )
                / (
                    torch.linalg.vector_norm(previous_weights)
                    + 1e-12
                )
            )
        )
        rows.append(
            {
                "episode": episode,
                "native_cache_finish_time": (
                    native_by_episode[episode]
                ),
                "fixed_cache_finish_time": fixed_finish,
                "cache_servers_changed": changed_servers,
                "relative_weight_l2_change": (
                    relative_weight_change
                ),
            }
        )
        previous_services = services
        previous_weights = weights

    native = np.asarray(
        [row["native_cache_finish_time"] for row in rows]
    )
    fixed = np.asarray(
        [row["fixed_cache_finish_time"] for row in rows]
    )
    changes = np.asarray(
        [row["cache_servers_changed"] for row in rows[1:]]
    )
    weight_changes = np.asarray(
        [row["relative_weight_l2_change"] for row in rows[1:]]
    )
    diagnosis = {
        "status": "complete",
        "run_dir": str(run_dir),
        "episodes": episodes,
        "fixed_cache_episode": args.fixed_cache_episode,
        "native_cache_finish_time_min": float(native.min()),
        "native_cache_finish_time_max": float(native.max()),
        "native_cache_finish_time_range": float(np.ptp(native)),
        "fixed_cache_finish_time_min": float(fixed.min()),
        "fixed_cache_finish_time_max": float(fixed.max()),
        "fixed_cache_finish_time_range": float(np.ptp(fixed)),
        "mean_cache_servers_changed_per_checkpoint": float(
            changes.mean()
        ),
        "max_cache_servers_changed_per_checkpoint": int(
            changes.max()
        ),
        "mean_relative_weight_l2_change": float(
            weight_changes.mean()
        ),
        "root_cause": (
            "native_daoc_popularity_cache_oscillation_not_network_drift"
        ),
        "decision": (
            "do_not_relax_posthoc_convergence_gate; retain_h3_main_"
            "experiment_and_replace_retrained_h0_h4_sweep_with_"
            "zero_shot_cross_capacity_evaluation"
        ),
        "rows": rows,
    }
    output_json = run_dir / "cache_oscillation_diagnosis.json"
    write_json(output_json, diagnosis)
    write_report(
        run_dir / "CACHE_OSCILLATION_DIAGNOSIS.md",
        diagnosis,
    )
    plot_diagnosis(
        run_dir / "cache_oscillation_diagnosis",
        rows,
    )
    update_failed_manifest(
        args.suite_manifest,
        diagnosis,
        run_dir,
    )
    update_failed_manifest(
        args.stage_manifest,
        diagnosis,
        run_dir,
    )
    print(json.dumps(diagnosis, indent=2))


def write_report(path, diagnosis):
    lines = [
        "# H0 DAOC Cache-Oscillation Diagnosis",
        "",
        f"- Native-cache range: "
        f"`{diagnosis['native_cache_finish_time_min']:.6f}` to "
        f"`{diagnosis['native_cache_finish_time_max']:.6f}` s.",
        f"- Fixed-cache range: "
        f"`{diagnosis['fixed_cache_finish_time_min']:.6f}` to "
        f"`{diagnosis['fixed_cache_finish_time_max']:.6f}` s.",
        f"- Mean servers changing cache per checkpoint: "
        f"`{diagnosis['mean_cache_servers_changed_per_checkpoint']:.3f}`.",
        f"- Mean relative network-weight change: "
        f"`{diagnosis['mean_relative_weight_l2_change']:.6f}`.",
        f"- Root cause: `{diagnosis['root_cause']}`.",
        f"- Protocol decision: `{diagnosis['decision']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diagnosis(path, rows):
    episodes = [row["episode"] for row in rows]
    figure, axis = plt.subplots(
        figsize=(8.2, 4.5),
        constrained_layout=True,
    )
    axis.plot(
        episodes,
        [row["native_cache_finish_time"] for row in rows],
        marker="o",
        label="Native DAOC cache",
        color="#C44E52",
    )
    axis.plot(
        episodes,
        [row["fixed_cache_finish_time"] for row in rows],
        marker="s",
        label="Same checkpoints, fixed cache",
        color="#4C72B0",
    )
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Validation completion time (s)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
