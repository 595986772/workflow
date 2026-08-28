#!/usr/bin/env python3
"""Recompute Figure 4 latency components on realized DAG critical paths.

The legacy figure summed ``pred_latency`` over all tasks.  That field is an
absolute predecessor-ready timestamp, so it is not an additive latency
component.  This script replays the saved frozen checkpoints on the original
paired scenario banks and backtracks the predecessor that actually determines
each DAG completion time.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

import plot_topconf_paper_figures as figure_style
from run_independent_experiment import (
    apply_deployment_state,
    apply_frozen_state,
    base_scenario_fingerprint,
    capture_deployment_state,
    capture_frozen_state,
    scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
    verify_frozen_state,
)
from simulator import MEC_Simulator


ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = (
    ROOT
    / "results"
    / "pegasus_pscale"
    / "p15_latency_composition_repair"
)
CANONICAL_FIGURE_DIR = ROOT / "paper_drafts" / "figures_topconf"
STEM = "fig03b_latency_composition"
FINAL_SEEDS = tuple(range(51, 61))
METHODS = (
    "greedy",
    "daoc_paper",
    "discrete_sac_std_cache",
    "daoc_our_coord_cache",
    "coord_cache_discrete_sac",
    "lean_our",
)
COMPONENT_FIELDS = (
    "user_input_transfer_s",
    "dependency_transfer_s",
    "service_loading_s",
    "waiting_s",
    "computation_s",
)
COMPONENT_LABELS = {
    "user_input_transfer_s": "User input transfer",
    "dependency_transfer_s": "Dependency transfer",
    "service_loading_s": "Service loading",
    "waiting_s": "Waiting",
    "computation_s": "Task processing",
}
COMPONENT_COLORS = {
    "user_input_transfer_s": "#587FA0",
    "dependency_transfer_s": "#4F887F",
    "service_loading_s": "#C58A4A",
    "waiting_s": "#A86269",
    "computation_s": "#766B91",
}
COMPONENT_HATCHES = {
    "user_input_transfer_s": "//",
    "dependency_transfer_s": "\\\\",
    "service_loading_s": "xx",
    "waiting_s": "..",
    "computation_s": "",
}
TRACE_FIELDS = (
    "method_key",
    "seed",
    "episode",
    "scenario_seed",
    "scenario_fingerprint",
    "workflow_family",
    "user_id",
    "completion_time_s",
    *COMPONENT_FIELDS,
    "component_sum_s",
    "identity_residual_s",
    "critical_path_task_count",
    "critical_path_task_ids",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_sort_key(task_id) -> tuple[int, str]:
    text = str(task_id)
    try:
        return (0, f"{int(text):012d}")
    except ValueError:
        return (1, text)


def extract_critical_path_components(user, simulator) -> dict[str, object]:
    """Return non-overlapping components on the realized critical chain."""
    tasks = user.done_tasks
    if set(tasks) != set(user.tasks_init):
        raise RuntimeError(f"User {user.id}: incomplete DAG in trace extraction")
    if any(count != 1 for count in user.task_completion_counts.values()):
        raise RuntimeError(f"User {user.id}: a task was not executed exactly once")

    exit_id = max(
        user.exit_task_ids,
        key=lambda task_id: (
            float(tasks[task_id].result.finish_time),
            task_sort_key(task_id),
        ),
    )
    expected_completion = float(tasks[exit_id].result.finish_time)
    if not np.isclose(
        expected_completion,
        float(user.finish_time_of_application),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(f"User {user.id}: exit and DAG completion disagree")

    totals = {field: 0.0 for field in COMPONENT_FIELDS}
    reverse_path = []
    current_id = exit_id
    visited = set()
    while True:
        if current_id in visited:
            raise RuntimeError(f"User {user.id}: cycle found during backtracking")
        visited.add(current_id)
        reverse_path.append(current_id)
        task = tasks[current_id]
        totals["user_input_transfer_s"] += float(
            task.result.data_transfer_latency
        )
        totals["service_loading_s"] += float(task.result.service_latency)
        totals["waiting_s"] += float(task.result.waiting_latency)
        totals["computation_s"] += float(task.result.computing_latency)

        if not task.predecessors:
            source_ready = (
                float(user.arrival_time)
                if simulator.dynamic_queueing
                else 0.0
            )
            if not np.isclose(
                float(task.result.pred_latency),
                source_ready,
                rtol=1e-10,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"User {user.id}: source predecessor time is inconsistent"
                )
            break

        candidates = []
        for predecessor_id in sorted(task.predecessors, key=task_sort_key):
            predecessor = tasks[predecessor_id]
            edge_transfer = float(
                simulator.between_server_costs[
                    predecessor.assigned_server,
                    task.assigned_server,
                ]
                * predecessor.outputs_length[current_id]
            )
            ready_time = float(predecessor.result.finish_time) + edge_transfer
            candidates.append((ready_time, predecessor_id, edge_transfer))
        selected_ready, predecessor_id, edge_transfer = max(
            candidates,
            key=lambda item: (item[0], task_sort_key(item[1])),
        )
        if not np.isclose(
            selected_ready,
            float(task.result.pred_latency),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"User {user.id}: reconstructed predecessor frontier disagrees"
            )
        totals["dependency_transfer_s"] += edge_transfer
        current_id = predecessor_id

    arrival_offset = float(user.arrival_time) if simulator.dynamic_queueing else 0.0
    component_sum = float(sum(totals.values()))
    residual = expected_completion - arrival_offset - component_sum
    tolerance = 1e-9 + 1e-9 * max(1.0, abs(expected_completion))
    if abs(residual) > tolerance:
        raise RuntimeError(
            f"User {user.id}: critical-path identity residual {residual:.3e}"
        )
    path = list(reversed(reverse_path))
    return {
        "completion_time_s": expected_completion - arrival_offset,
        **totals,
        "component_sum_s": component_sum,
        "identity_residual_s": residual,
        "critical_path_task_count": len(path),
        "critical_path_task_ids": "->".join(str(item) for item in path),
    }


def completed_run_is_reusable(
    output_dir: Path,
    *,
    method: str,
    seed: int,
    episodes: int,
    checkpoint_hash: str,
) -> bool:
    validation_path = output_dir / "validation.json"
    trace_path = output_dir / "critical_path_components.csv"
    if not validation_path.exists() or not trace_path.exists():
        return False
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        validation.get("status") == "complete"
        and validation.get("method_key") == method
        and validation.get("seed") == seed
        and validation.get("episodes") == episodes
        and validation.get("selected_checkpoint_sha256") == checkpoint_hash
        and validation.get("trace_sha256") == sha256(trace_path)
    )


def replay_method_seed(
    method: str,
    seed: int,
    episodes: int,
    analysis_dir: Path,
    resume: bool,
) -> dict[str, object]:
    torch.set_num_threads(1)
    run_dir = figure_style.run_directory(method, seed)
    checkpoint_path = run_dir / "selected_checkpoint.pt"
    checkpoint_hash = sha256(checkpoint_path)
    output_dir = analysis_dir / "runs" / method / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and completed_run_is_reusable(
        output_dir,
        method=method,
        seed=seed,
        episodes=episodes,
        checkpoint_hash=checkpoint_hash,
    ):
        return {"method_key": method, "seed": seed, "status": "reused"}

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**config["arguments"])
    input_config = config["input_config"]
    learning_config = config["learning_config"]
    saved_scenarios = json.loads(
        (run_dir / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    if episodes > len(saved_scenarios):
        raise ValueError(f"{method} seed {seed}: only {len(saved_scenarios)} scenarios")
    official_rows = {
        int(row["episode"]): row
        for row in read_csv(run_dir / "episodes.csv")
        if row["phase"] == "eval"
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frozen_state = checkpoint["frozen_state"]

    seed_everything(seed)
    base_simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=copy.deepcopy(input_config),
        learning_arguments=copy.deepcopy(learning_config),
        filename_png=str(output_dir),
    )
    deployment_state = capture_deployment_state(base_simulator)

    trace_rows = []
    maximum_finish_difference = 0.0
    maximum_identity_residual = 0.0
    for episode_index in range(episodes):
        episode = episode_index + 1
        saved_scenario = saved_scenarios[episode_index]
        eval_seed = seed + int(args.eval_seed_offset) + episode_index
        if int(saved_scenario["seed"]) != eval_seed:
            raise RuntimeError(f"{method} seed {seed}: scenario seed mismatch")
        seed_everything(eval_seed)
        eval_config = copy.deepcopy(input_config)
        eval_config["seed"] = eval_seed
        eval_config["save topology figure"] = False
        workflow_family = None
        if args.eval_dag_families is not None:
            workflow_family = args.eval_dag_families[
                episode_index % len(args.eval_dag_families)
            ]
            eval_config["application graph family"] = workflow_family
        simulator = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=eval_config,
            learning_arguments=copy.deepcopy(learning_config),
            filename_png=str(output_dir),
        )
        if args.eval_bank_scope in ("workload", "infrastructure"):
            apply_deployment_state(
                simulator,
                deployment_state,
                include_users=args.eval_bank_scope == "workload",
            )
        fingerprint = scenario_fingerprint(scenario_snapshot(simulator))
        base_fingerprint = base_scenario_fingerprint(simulator)
        if fingerprint != saved_scenario["fingerprint"]:
            raise RuntimeError(f"{method} seed {seed}: scenario fingerprint mismatch")
        if base_fingerprint != saved_scenario["base_fingerprint"]:
            raise RuntimeError(f"{method} seed {seed}: base fingerprint mismatch")

        apply_frozen_state(simulator, frozen_state)
        for user in simulator.users.values():
            user.setpos0()
        update_caching = bool(getattr(args, "eval_update_caching", False))
        simulator.set_training(False, update_caching=update_caching)
        simulator.reset()
        episode_frozen_state = capture_frozen_state(simulator)
        simulator.run()

        replayed_finish = float(
            np.mean(
                [user.finish_time_of_application for user in simulator.users.values()]
            )
        )
        official_finish = float(official_rows[episode]["average_finish_time"])
        finish_difference = abs(replayed_finish - official_finish)
        maximum_finish_difference = max(maximum_finish_difference, finish_difference)
        if finish_difference > 1e-9:
            raise RuntimeError(
                f"{method} seed {seed} episode {episode}: replay differs from "
                f"official result by {finish_difference:.3e} s"
            )

        for user_id, user in simulator.users.items():
            trace = extract_critical_path_components(user, simulator)
            maximum_identity_residual = max(
                maximum_identity_residual,
                abs(float(trace["identity_residual_s"])),
            )
            trace_rows.append(
                {
                    "method_key": method,
                    "seed": seed,
                    "episode": episode,
                    "scenario_seed": eval_seed,
                    "scenario_fingerprint": fingerprint,
                    "workflow_family": workflow_family,
                    "user_id": user_id,
                    **trace,
                }
            )
        verify_frozen_state(
            simulator,
            episode_frozen_state,
            check_cache=not update_caching,
        )

    trace_path = output_dir / "critical_path_components.csv"
    write_csv(trace_path, trace_rows, TRACE_FIELDS)
    validation = {
        "status": "complete",
        "method_key": method,
        "seed": seed,
        "episodes": episodes,
        "users_per_episode": int(args.num_users),
        "trace_rows": len(trace_rows),
        "selected_checkpoint_episode": int(checkpoint["episode"]),
        "selected_checkpoint_sha256": checkpoint_hash,
        "scenario_fingerprints_verified": episodes,
        "maximum_official_finish_difference_s": maximum_finish_difference,
        "maximum_critical_path_identity_residual_s": maximum_identity_residual,
        "trace_sha256": sha256(trace_path),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    return {"method_key": method, "seed": seed, "status": "computed"}


def ci95(values) -> tuple[float, float]:
    return figure_style.ci95(np.asarray(values, dtype=float))


def aggregate_traces(
    analysis_dir: Path,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    episodes: int,
) -> tuple[list[dict], list[dict]]:
    seed_rows = []
    for method in methods:
        for seed in seeds:
            rows = read_csv(
                analysis_dir
                / "runs"
                / method
                / f"seed_{seed}"
                / "critical_path_components.csv"
            )
            aggregate = {
                field: float(np.mean([float(row[field]) for row in rows]))
                for field in ("completion_time_s", *COMPONENT_FIELDS)
            }
            component_sum = sum(aggregate[field] for field in COMPONENT_FIELDS)
            residual = aggregate["completion_time_s"] - component_sum
            if abs(residual) > 1e-9:
                raise RuntimeError(
                    f"{method} seed {seed}: aggregate identity residual {residual:.3e}"
                )
            official = float(
                np.mean(
                    [
                        float(row["average_finish_time"])
                        for row in figure_style.evaluation_rows(method, seed)[:episodes]
                    ]
                )
            )
            if abs(official - aggregate["completion_time_s"]) > 1e-9:
                raise RuntimeError(f"{method} seed {seed}: official aggregate mismatch")
            seed_rows.append(
                {
                    "method_key": method,
                    "method": figure_style.DISPLAY[method],
                    "seed": seed,
                    **aggregate,
                    "component_sum_s": component_sum,
                    "identity_residual_s": residual,
                }
            )

    method_rows = []
    for method in methods:
        selected = [row for row in seed_rows if row["method_key"] == method]
        completion_mean, completion_ci = ci95(
            [row["completion_time_s"] for row in selected]
        )
        component_means = {
            field: float(np.mean([row[field] for row in selected]))
            for field in COMPONENT_FIELDS
        }
        denominator = sum(component_means.values())
        for field in COMPONENT_FIELDS:
            component_mean, component_ci = ci95([row[field] for row in selected])
            method_rows.append(
                {
                    "method_key": method,
                    "method": figure_style.DISPLAY[method],
                    "mean_completion_time_s": completion_mean,
                    "completion_ci95_half_width_s": completion_ci,
                    "latency_component_key": field,
                    "latency_component": COMPONENT_LABELS[field],
                    "mean_component_time_s": component_mean,
                    "component_ci95_half_width_s": component_ci,
                    "mean_share_percent": 100.0 * component_mean / denominator,
                    "independent_seeds": len(selected),
                    "decomposition_scope": "realized_completion_critical_path",
                }
            )
    write_csv(analysis_dir / "critical_path_seed_summary.csv", seed_rows)
    write_csv(analysis_dir / "critical_path_method_summary.csv", method_rows)
    return seed_rows, method_rows


def render_figure(
    analysis_dir: Path,
    output_dir: Path,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Path]:
    method_rows = read_csv(analysis_dir / "critical_path_method_summary.csv")
    by_method = {
        method: [row for row in method_rows if row["method_key"] == method]
        for method in methods
    }
    figure_style.configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(figure_style.DOUBLE_COLUMN, 91 * figure_style.MM),
        gridspec_kw={"width_ratios": (0.95, 1.48)},
        sharey=True,
        layout="constrained",
    )
    performance_axis, composition_axis = axes
    positions = np.arange(len(methods), dtype=float)
    completion_means = np.asarray(
        [float(by_method[method][0]["mean_completion_time_s"]) for method in methods]
    )
    completion_errors = np.asarray(
        [
            float(by_method[method][0]["completion_ci95_half_width_s"])
            for method in methods
        ]
    )
    bars = performance_axis.barh(
        positions,
        completion_means,
        xerr=completion_errors,
        height=0.58,
        color=[figure_style.COLORS[method] for method in methods],
        edgecolor="#30363A",
        linewidth=0.7,
        error_kw={
            "ecolor": "#30363A",
            "elinewidth": 0.8,
            "capsize": 2.5,
            "capthick": 0.8,
        },
        zorder=2,
    )
    completion_limit = float(np.max(completion_means + completion_errors))
    for bar, mean, error in zip(bars, completion_means, completion_errors):
        performance_axis.text(
            mean + error + 0.03 * completion_limit,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f}",
            ha="left",
            va="center",
            fontsize=6.4,
            color="#30363A",
        )
    performance_axis.set_yticks(positions)
    performance_axis.set_yticklabels(
        [figure_style.DISPLAY[method] for method in methods]
    )
    performance_axis.invert_yaxis()
    performance_axis.set_xlim(0, completion_limit * 1.24)
    performance_axis.set_xlabel("Mean DAG completion time (s)")
    figure_style.panel_label(performance_axis, "(a)")
    figure_style.style_axis(performance_axis, grid="x")

    left = np.zeros(len(methods), dtype=float)
    for field in COMPONENT_FIELDS:
        values = np.asarray(
            [
                float(
                    next(
                        row["mean_share_percent"]
                        for row in by_method[method]
                        if row["latency_component_key"] == field
                    )
                )
                for method in methods
            ]
        )
        component_bars = composition_axis.barh(
            positions,
            values,
            left=left,
            height=0.58,
            color=COMPONENT_COLORS[field],
            edgecolor="#30363A",
            linewidth=0.58,
            label=COMPONENT_LABELS[field],
            zorder=2,
        )
        for bar in component_bars:
            bar.set_hatch(COMPONENT_HATCHES[field])
        absolute_values_ms = np.asarray(
            [
                1000.0
                * float(
                    next(
                        row["mean_component_time_s"]
                        for row in by_method[method]
                        if row["latency_component_key"] == field
                    )
                )
                for method in methods
            ]
        )
        for index, (value, absolute_ms) in enumerate(
            zip(values, absolute_values_ms)
        ):
            # Input transfer is zero and dependency transfer is below 0.3 ms
            # in this experiment. Their bars are too narrow for truthful text;
            # a compact note below reports those magnitudes instead.
            if value >= 2.25:
                composition_axis.text(
                    left[index] + value / 2.0,
                    index,
                    f"{absolute_ms:.0f}\n{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5.15,
                    linespacing=0.86,
                    color="#202427",
                )
        left += values
    if not np.allclose(left, 100.0, atol=1e-7):
        raise RuntimeError("Rendered component shares do not sum to 100%")
    composition_axis.set_xlim(0, 100)
    composition_axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    composition_axis.set_xlabel(
        "Critical-path component share (labels: ms / %)"
    )
    dependency_values_ms = [
        1000.0
        * float(
            next(
                row["mean_component_time_s"]
                for row in by_method[method]
                if row["latency_component_key"]
                == "dependency_transfer_s"
            )
        )
        for method in methods
    ]
    composition_axis.text(
        0.995,
        0.012,
        (
            "Input transfer: 0 ms; dependency transfer: "
            f"{min(dependency_values_ms):.2f}--"
            f"{max(dependency_values_ms):.2f} ms"
        ),
        transform=composition_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#4A5054",
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
        zorder=5,
    )
    composition_axis.tick_params(axis="y", left=False, labelleft=False)
    composition_axis.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
        columnspacing=0.85,
        handlelength=1.55,
    )
    figure_style.panel_label(composition_axis, "(b)")
    figure_style.style_axis(composition_axis, grid="x")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / STEM
    figure_style.export_figure(
        figure,
        output_stem,
        formats=("pdf", "png", "svg"),
        dpi=400,
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        font_mode="truetype",
        overwrite=True,
        mkdir=True,
        metadata={
            "Creator": "regenerate_critical_path_latency_figure.py",
            "Title": "DAG completion time and realized critical-path latency composition",
        },
        provenance={
            "raw_data": [
                str(
                    (analysis_dir / "critical_path_seed_summary.csv").relative_to(ROOT)
                ),
                str(
                    (analysis_dir / "critical_path_method_summary.csv").relative_to(ROOT)
                ),
            ],
            "transformations": [
                "replay saved final stable checkpoints on original paired scenarios",
                "select maximum-finish exit and backtrack realized predecessor frontier",
                "count each user-input, dependency-transfer, service, waiting, and computation term once",
                "aggregate user traces to independent seed means",
            ],
            "uncertainty": "95% Student-t CI over ten independent seed-level means",
            "validation": "component identity and official-result replay checked per DAG and episode",
            "skill": "scientific-visualization",
            "skill_commit": figure_style.SCI_VIS_COMMIT,
        },
        write_manifest=True,
    )
    plt.close(figure)
    return {
        suffix: output_stem.with_suffix(f".{suffix}")
        for suffix in ("png", "pdf", "svg", "export.json")
    }


def backup_once(source: Path, backup_dir: Path) -> None:
    if not source.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / source.name
    if not destination.exists():
        shutil.copy2(source, destination)


def replace_canonical_outputs(analysis_dir: Path, rendered_dir: Path) -> list[Path]:
    dated_backup = "superseded_20260819_share_only_labels"
    figure_targets = (
        CANONICAL_FIGURE_DIR,
        Path.home() / "Desktop" / "dag论文图片",
        ROOT
        / "deliverables"
        / "OUR_repro_writing_package_20260818"
        / "04_FINAL_FIGURES",
        ROOT
        / "deliverables"
        / "OUR_chapters_DAOC_review_package_20260818"
        / "04_KEY_FIGURES",
    )
    copied = []
    for target_dir in figure_targets:
        if not target_dir.exists():
            continue
        for suffix in ("png", "pdf", "svg", "export.json"):
            source = rendered_dir / f"{STEM}.{suffix}"
            destination = target_dir / source.name
            backup_once(destination, target_dir / dated_backup)
            shutil.copy2(source, destination)
            copied.append(destination)

    data_source = analysis_dir / "critical_path_method_summary.csv"
    data_targets = (
        CANONICAL_FIGURE_DIR / f"{STEM}_data.csv",
        ROOT
        / "deliverables"
        / "OUR_repro_writing_package_20260818"
        / "03_KEY_RESULTS"
        / "figure_data"
        / f"{STEM}_data.csv",
    )
    for destination in data_targets:
        if not destination.parent.exists():
            continue
        backup_once(destination, destination.parent / dated_backup)
        shutil.copy2(data_source, destination)
        copied.append(destination)
    return copied


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown methods: {sorted(unknown)}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--methods", type=parse_methods, default=METHODS)
    parser.add_argument("--seeds", type=parse_ints, default=FINAL_SEEDS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--replace-canonical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (method, seed)
        for method in args.methods
        for seed in args.seeds
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                replay_method_seed,
                method,
                seed,
                args.episodes,
                analysis_dir,
                not args.no_resume,
            ): (method, seed)
            for method, seed in jobs
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            result = future.result()
            results.append(result)
            print(f"[{len(results):02d}/{len(jobs):02d}] {method} seed {seed}: {result['status']}")

    aggregate_traces(analysis_dir, args.methods, args.seeds, args.episodes)
    rendered_dir = analysis_dir / "figure"
    outputs = render_figure(analysis_dir, rendered_dir, args.methods)
    copied = []
    if args.replace_canonical:
        if args.methods != METHODS or args.seeds != FINAL_SEEDS or args.episodes != 100:
            raise RuntimeError("Canonical replacement requires all methods, seeds 51-60, and 100 episodes")
        copied = replace_canonical_outputs(analysis_dir, rendered_dir)
    manifest = {
        "status": "complete",
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "episodes_per_seed": args.episodes,
        "workers": args.workers,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "canonical_replacement": bool(args.replace_canonical),
        "copied_outputs": [str(path) for path in copied],
        "source_sha256": sha256(Path(__file__)),
    }
    (analysis_dir / "REPAIR_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
