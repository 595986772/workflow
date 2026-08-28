#!/usr/bin/env python3
"""Run DAOC/OUR Pegasus sensitivity experiments over server count."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pegasus_server_scaling_protocol import (
    BANDWIDTH_HZ,
    CAPACITY_NAMESPACE,
    CAPACITY_PROFILES,
    DATASET_PATH,
    EVALUATION_EPISODES,
    EXPECTED_DATASET_SHA256,
    FAMILIES,
    METHODS,
    PROTOCOL_VERSION,
    REFERENCE_ROOT,
    RESULT_ROOT,
    SEEDS,
    SERVER_COUNTS,
    SERVICES,
    SMOKE_EPISODES,
    TASK_LIMIT_INCLUDING_DUMMY,
    TRAINED_SERVER_COUNTS,
    USERS,
    sha256_file,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
LOCK_PATH = RESULT_ROOT / "PROTOCOL_LOCK.json"
ANALYSIS_DIR = RESULT_ROOT / "analysis"
FIGURE_DIR = ROOT / "paper_drafts/figures_topconf"
STYLE_PATH = ROOT.parent / ".codex/skills/scientific-visualization/assets/publication.mplstyle"
SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_reward.py",
    "critical_path_rl.py",
    "dqn.py",
    "pegasus_server_scaling_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_server_scaling.py",
    "run_reproduction_suite.py",
    "server.py",
    "simulator.py",
    "task.py",
    "user.py",
)
DISPLAY = {"daoc_paper": "DAOC", "lean_our": "OUR"}
COLORS = {"daoc_paper": "#7B708E", "lean_our": "#1F77A8"}
MARKERS = {"daoc_paper": "o", "lean_our": "*"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "train", "analysis", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    return args


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for filename in sorted(SOURCE_FILES):
        digest.update(filename.encode("utf-8"))
        digest.update((ROOT / filename).read_bytes())
    return digest.hexdigest()


def initialize_lock() -> dict:
    specification = validate_protocol()
    lock = {
        "status": "locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "specification_sha256": canonical_hash(specification),
        "source_sha256": source_hash(),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH)
        for key in ("specification_sha256", "source_sha256"):
            if existing.get(key) != lock[key]:
                raise RuntimeError(f"Server-scaling protocol changed: {key}")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def suite_dir(stage: str, servers: int) -> Path:
    return RESULT_ROOT / stage / f"servers_{servers}"


def run_dir(stage: str, servers: int, method: str, seed: int) -> Path:
    if servers == 10 and stage == "converged":
        return REFERENCE_ROOT / "runs" / method / f"seed_{seed}"
    return suite_dir(stage, servers) / "runs" / method / f"seed_{seed}"


def suite_command(stage: str, servers: int, workers: int, resume: bool):
    smoke = stage == "smoke"
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        (
            "pegasus_paper_closure_smoke"
            if smoke
            else "pegasus_paper_closure_converged"
        ),
        "--suite-dir",
        str(suite_dir(stage, servers)),
        "--seeds",
        "51" if smoke else ",".join(str(seed) for seed in SEEDS),
        "--labels",
        ",".join(METHODS),
        "--num-servers",
        str(servers),
        "--dag-dataset-path",
        str(DATASET_PATH),
        "--dag-dataset-sha256",
        EXPECTED_DATASET_SHA256,
        "--num-tasks",
        str(TASK_LIMIT_INCLUDING_DUMMY),
        "--eval-dag-families",
        ",".join(FAMILIES),
        "--server-capacity",
        "1",
        "--server-capacity-multiset",
        ",".join(str(value) for value in CAPACITY_PROFILES[servers]),
        "--baseline-server-capacity",
        "3",
        "--capacity-assignment-namespace",
        CAPACITY_NAMESPACE,
        "--workers",
        str(workers),
        "--revision-id",
        PROTOCOL_VERSION,
        "--revision-parent",
        "pegasus_paper_closure_v1",
        "--revision-reason",
        "server_count_sensitivity_with_constant_per_server_cache_profile",
        "--revision-changed-module",
        "environment_server_count_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_unconverged_method",
        "--seed-partition",
        "generalization",
    ]
    if smoke:
        command.extend(
            [
                "--train-episodes",
                "200",
                "--eval-episodes",
                str(SMOKE_EPISODES),
            ]
        )
    if resume:
        command.append("--resume")
    return command


def run_logged(command, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(value) for value in command), flush=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + " ".join(str(value) for value in command)
            + "\n"
        )
        output.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def worker_allocation(total: int, count: int) -> list[int]:
    base, extra = divmod(total, count)
    return [max(1, base + (index < extra)) for index in range(count)]


def run_scales(stage: str, workers: int, resume: bool) -> None:
    scales = list(TRAINED_SERVER_COUNTS)
    allocations = worker_allocation(workers, len(scales))
    with ThreadPoolExecutor(max_workers=len(scales)) as executor:
        futures = {
            executor.submit(
                run_logged,
                suite_command(stage, servers, allocation, resume),
                suite_dir(stage, servers) / "runner.log",
            ): servers
            for servers, allocation in zip(scales, allocations)
        }
        for future in as_completed(futures):
            servers = futures[future]
            future.result()
            print(f"S={servers} {stage}: complete", flush=True)


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    with (path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle) if row["phase"] == "eval"
        ]


def comparable_bank(path: Path):
    return [
        {
            "episode": row["episode"],
            "seed": row["seed"],
            "base_fingerprint": row["base_fingerprint"],
            "workflow_family": row.get("workflow_family"),
            "user_initial_positions": row["user_initial_positions"],
            "user_graph_keys": row["user_graph_keys"],
        }
        for row in read_json(path)
    ]


def audit_stage(stage: str) -> dict:
    smoke = stage == "smoke"
    seeds = (51,) if smoke else SEEDS
    episodes = SMOKE_EPISODES if smoke else EVALUATION_EPISODES
    scales = TRAINED_SERVER_COUNTS if smoke else SERVER_COUNTS
    audit = {
        "status": "complete",
        "stage": stage,
        "servers": {},
        "all_runs_complete": True,
        "all_learning_runs_converged": True,
        "all_tasks_exactly_once": True,
        "all_capacity_constraints_valid": True,
        "all_methods_scenario_paired": True,
    }
    for servers in scales:
        per_server = {"seeds": {}}
        for seed in seeds:
            reference_bank = None
            per_seed = {"methods": {}}
            for method in METHODS:
                directory = run_dir(stage, servers, method, seed)
                summary = read_json(directory / "summary.json")
                config = read_json(directory / "config.json")
                rows = read_eval_rows(directory)
                capacities = tuple(
                    sorted(int(value) for value in summary["server_capacities"].values())
                )
                complete = summary.get("status") == "complete"
                converged = (
                    True
                    if smoke
                    else bool(
                        summary.get("eligible_for_comparison")
                        and summary.get("convergence", {}).get("reached")
                    )
                )
                exact_once = len(rows) == episodes and all(
                    int(row["real_task_count"])
                    == int(row["completed_task_count"])
                    and int(row["all_tasks_executed_once"]) == 1
                    for row in rows
                )
                capacity_valid = (
                    config["arguments"]["num_servers"] == servers
                    and config["arguments"]["num_users"] == USERS
                    and config["arguments"]["num_services"] == SERVICES
                    and float(config["arguments"]["bandwidth"]) == BANDWIDTH_HZ
                    and capacities == tuple(sorted(CAPACITY_PROFILES[servers]))
                    and summary.get("total_server_capacity")
                    == sum(CAPACITY_PROFILES[servers])
                )
                bank = comparable_bank(directory / "evaluation_scenarios.json")
                paired = reference_bank is None or bank == reference_bank
                if reference_bank is None:
                    reference_bank = bank
                audit["all_runs_complete"] &= complete
                audit["all_learning_runs_converged"] &= converged
                audit["all_tasks_exactly_once"] &= exact_once
                audit["all_capacity_constraints_valid"] &= capacity_valid
                audit["all_methods_scenario_paired"] &= paired
                per_seed["methods"][method] = {
                    "complete": complete,
                    "converged": converged,
                    "tasks_exactly_once": exact_once,
                    "capacity_valid": capacity_valid,
                    "paired_with_daoc": paired,
                    "selected_checkpoint_episode": summary.get(
                        "selected_checkpoint_episode"
                    ),
                }
            per_server["seeds"][str(seed)] = per_seed
        audit["servers"][str(servers)] = per_server
    if not all(
        audit[key]
        for key in (
            "all_runs_complete",
            "all_learning_runs_converged",
            "all_tasks_exactly_once",
            "all_capacity_constraints_valid",
            "all_methods_scenario_paired",
        )
    ):
        raise RuntimeError(f"{stage} integrity audit failed")
    write_json(RESULT_ROOT / f"{stage.upper()}_AUDIT.json", audit)
    return audit


def ci95(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0
    return mean, float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))


def aggregate() -> dict:
    audit = audit_stage("converged")
    rows = []
    results = {}
    for servers in SERVER_COUNTS:
        per_seed = []
        for seed in SEEDS:
            entry = {"seed": seed, "methods": {}}
            for method in METHODS:
                directory = run_dir("converged", servers, method, seed)
                summary = read_json(directory / "summary.json")
                values = {
                    "mean_completion_time_s": float(
                        summary["eval"]["mean_average_finish_time"]
                    ),
                    "p95_completion_time_s": float(
                        summary["eval"]["mean_p95_finish_time"]
                    ),
                    "cache_hit_rate": float(
                        summary["eval"]["mean_cache_hit_rate"]
                    ),
                    "remote_loading_rate": float(
                        summary["eval"]["mean_cache_remote_loading_rate"]
                    ),
                    "inference_ms_per_decision": float(
                        summary["eval"][
                            "mean_policy_inference_time_per_decision_ms"
                        ]
                    ),
                    "cache_decision_ms_per_episode": 1000.0
                    * float(
                        summary["eval"]["mean_cache_decision_wall_time_sec"]
                    ),
                    "selected_checkpoint_episode": int(
                        summary["selected_checkpoint_episode"]
                    ),
                    "total_wall_time_sec": float(summary["total_wall_time_sec"]),
                }
                entry["methods"][method] = values
                rows.append(
                    {
                        "servers": servers,
                        "seed": seed,
                        "method": DISPLAY[method],
                        "method_key": method,
                        "source": "reused_p3" if servers == 10 else "p11_retrained",
                        **values,
                    }
                )
            per_seed.append(entry)
        summaries = {}
        for method in METHODS:
            summaries[method] = {}
            for metric in (
                "mean_completion_time_s",
                "p95_completion_time_s",
                "cache_hit_rate",
                "remote_loading_rate",
                "inference_ms_per_decision",
                "cache_decision_ms_per_episode",
                "selected_checkpoint_episode",
            ):
                values = [row["methods"][method][metric] for row in per_seed]
                mean, half = ci95(values)
                summaries[method][metric] = {
                    "mean": mean,
                    "ci95_half_width": half,
                    "per_seed": values,
                }
        daoc = np.asarray(
            [
                row["methods"]["daoc_paper"]["mean_completion_time_s"]
                for row in per_seed
            ]
        )
        ours = np.asarray(
            [
                row["methods"]["lean_our"]["mean_completion_time_s"]
                for row in per_seed
            ]
        )
        reductions = 100.0 * (daoc - ours) / daoc
        reduction_mean, reduction_half = ci95(reductions)
        results[str(servers)] = {
            "per_seed": per_seed,
            "method_summary": summaries,
            "our_vs_daoc": {
                "relative_reduction_percent_mean": reduction_mean,
                "relative_reduction_percent_ci95_half_width": reduction_half,
                "relative_reduction_percent_per_seed": reductions.tolist(),
                "seed_wins": int(np.sum(ours < daoc)),
                "seed_count": len(SEEDS),
                "wilcoxon_one_sided_p": float(
                    stats.wilcoxon(
                        daoc,
                        ours,
                        alternative="greater",
                        zero_method="wilcox",
                    ).pvalue
                ),
            },
        }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with (ANALYSIS_DIR / "server_scaling_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "protocol": validate_protocol(),
        "integrity": {
            key: audit[key]
            for key in (
                "all_runs_complete",
                "all_learning_runs_converged",
                "all_tasks_exactly_once",
                "all_capacity_constraints_valid",
                "all_methods_scenario_paired",
            )
        },
        "results": results,
        "statistical_scope": (
            "descriptive three-seed sensitivity; not a ten-seed superiority test"
        ),
    }
    write_json(ANALYSIS_DIR / "server_scaling_summary.json", summary)
    write_report(summary)
    plot_results(summary)
    return summary


def write_report(summary: dict) -> None:
    lines = [
        "# Pegasus服务器数量敏感性实验",
        "",
        "- 固定20用户、10服务、15 kHz和五类Pegasus工作流。",
        "- 服务器数为5、10、15、20；每台服务器容量分布比例固定为"
        "40% K=0、40% K=1、20% K=2，因此总缓存预算为4、8、12、16。",
        "- S=5/15/20均从头训练至收敛；S=10严格复用P3同协议checkpoint。",
        "- Seeds为51--53，每seed使用100个DAOC/OUR完全配对场景。",
        "- 统计仅作为三seed敏感性证据，不作为十seed正式显著性结论。",
        "",
        "| Servers | Budget | DAOC Mean | OUR Mean | OUR降低 | DAOC P95 | OUR P95 | OUR胜出seed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for servers in SERVER_COUNTS:
        row = summary["results"][str(servers)]
        daoc = row["method_summary"]["daoc_paper"]
        ours = row["method_summary"]["lean_our"]
        comparison = row["our_vs_daoc"]
        lines.append(
            f"| {servers} | {sum(CAPACITY_PROFILES[servers])} | "
            f"{daoc['mean_completion_time_s']['mean']:.4f} | "
            f"{ours['mean_completion_time_s']['mean']:.4f} | "
            f"{comparison['relative_reduction_percent_mean']:.2f}% | "
            f"{daoc['p95_completion_time_s']['mean']:.4f} | "
            f"{ours['p95_completion_time_s']['mean']:.4f} | "
            f"{comparison['seed_wins']}/{comparison['seed_count']} |"
        )
    (ANALYSIS_DIR / "SERVER_SCALING_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_results(summary: dict) -> None:
    plt.style.use(str(STYLE_PATH))
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    mm = 1.0 / 25.4
    figure, axes = plt.subplots(
        1, 2, figsize=(180 * mm, 68 * mm), layout="constrained"
    )
    metrics = (
        ("mean_completion_time_s", "Mean completion time (s)", "(a)"),
        ("p95_completion_time_s", "P95 completion time (s)", "(b)"),
    )
    x = np.asarray(SERVER_COUNTS, dtype=float)
    for axis, (metric, ylabel, panel) in zip(axes, metrics):
        for method in METHODS:
            means = [
                summary["results"][str(servers)]["method_summary"][method][
                    metric
                ]["mean"]
                for servers in SERVER_COUNTS
            ]
            errors = [
                summary["results"][str(servers)]["method_summary"][method][
                    metric
                ]["ci95_half_width"]
                for servers in SERVER_COUNTS
            ]
            axis.errorbar(
                x,
                means,
                yerr=errors,
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor=(
                    COLORS[method] if method == "lean_our" else "white"
                ),
                markeredgecolor=COLORS[method],
                linewidth=1.55 if method == "lean_our" else 1.2,
                markersize=7 if method == "lean_our" else 5,
                capsize=2.5,
                label=DISPLAY[method],
            )
        axis.set_xticks(SERVER_COUNTS)
        axis.set_xlabel("Number of edge servers")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D7DADD", linewidth=0.6, linestyle="--")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.01,
            0.98,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    axes[0].legend(frameon=False, loc="best")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            FIGURE_DIR / f"fig10_server_count_sensitivity.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches=None,
        )
    plt.close(figure)


def run_tests() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "test_pegasus_server_scaling.py",
            "test_capacity_protocol.py",
            "test_dag_completion_semantics.py",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    initialize_lock()
    if args.stage in ("tests", "all"):
        run_tests()
    if args.stage in ("smoke", "all"):
        run_scales("smoke", args.workers, args.resume)
        audit_stage("smoke")
    if args.stage in ("train", "all"):
        run_scales("converged", args.workers, args.resume)
    if args.stage in ("analysis", "all"):
        aggregate()
    print(RESULT_ROOT)


if __name__ == "__main__":
    main()
