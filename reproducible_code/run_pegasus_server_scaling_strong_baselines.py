#!/usr/bin/env python3
"""Extend the P11 server sweep with OUR-DQN and CoordCache-DiscreteSAC."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
    SEEDS,
    SERVER_COUNTS,
    SERVICES,
    SMOKE_EPISODES,
    TASK_LIMIT_INCLUDING_DUMMY,
    TRAINED_SERVER_COUNTS,
    USERS,
)
from pegasus_server_scaling_strong_baselines_protocol import (
    COORD_SAC,
    DISPLAY_NAMES,
    METHODS,
    OUR_DQN,
    P11_ROOT,
    PROTOCOL_VERSION,
    REFERENCE_ROOTS,
    RESULT_ROOT,
    SMOKE_PROFILES,
    TRAINING_PROFILES,
    canonical_hash,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parent
LOCK_PATH = RESULT_ROOT / "PROTOCOL_LOCK.json"
ANALYSIS_DIR = RESULT_ROOT / "analysis"
FIGURE_DIR = ROOT / "paper_drafts/figures_topconf"
STYLE_PATH = (
    ROOT.parent
    / ".codex/skills/scientific-visualization/assets/publication.mplstyle"
)
SOURCE_FILES = (
    "agent.py",
    "broker.py",
    "capacity_protocol.py",
    "critical_path_cache.py",
    "critical_path_reward.py",
    "critical_path_rl.py",
    "discrete_sac.py",
    "dqn.py",
    "pegasus_server_scaling_strong_baselines_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_server_scaling_strong_baselines.py",
    "run_reproduction_suite.py",
    "server.py",
    "simulator.py",
    "task.py",
    "user.py",
)
ALL_METHODS = ("daoc_paper", OUR_DQN, COORD_SAC, "lean_our")
DISPLAY = {
    "daoc_paper": "DAOC",
    OUR_DQN: DISPLAY_NAMES[OUR_DQN],
    COORD_SAC: DISPLAY_NAMES[COORD_SAC],
    "lean_our": "OUR",
}
COLORS = {
    "daoc_paper": "#746985",
    OUR_DQN: "#4D8B75",
    COORD_SAC: "#C2764B",
    "lean_our": "#176B9A",
}
MARKERS = {
    "daoc_paper": "o",
    OUR_DQN: "s",
    COORD_SAC: "D",
    "lean_our": "*",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "train", "analysis", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=6)
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
                raise RuntimeError(f"P12 protocol changed: {key}")
        return existing
    write_json(LOCK_PATH, lock)
    return lock


def suite_dir(stage: str, servers: int, method: str) -> Path:
    return RESULT_ROOT / stage / f"servers_{servers}" / method


def p11_run_dir(stage: str, servers: int, method: str, seed: int) -> Path:
    if stage == "converged" and servers == 10:
        return (
            ROOT
            / "results/pegasus_pscale/p3_paper_closure/final/runs"
            / method
            / f"seed_{seed}"
        )
    return (
        P11_ROOT
        / stage
        / f"servers_{servers}"
        / "runs"
        / method
        / f"seed_{seed}"
    )


def new_run_dir(stage: str, servers: int, method: str, seed: int) -> Path:
    if stage == "converged" and servers == 10:
        return REFERENCE_ROOTS[method] / f"seed_{seed}"
    return suite_dir(stage, servers, method) / "runs" / method / f"seed_{seed}"


def suite_command(
    stage: str, servers: int, method: str, resume: bool
) -> list[str]:
    smoke = stage == "smoke"
    profile = SMOKE_PROFILES[method] if smoke else TRAINING_PROFILES[method]
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        profile,
        "--suite-dir",
        str(suite_dir(stage, servers, method)),
        "--seeds",
        "51" if smoke else ",".join(str(seed) for seed in SEEDS),
        "--labels",
        method,
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
        "1",
        "--revision-id",
        PROTOCOL_VERSION,
        "--revision-parent",
        "pegasus_server_scaling_p11_v1",
        "--revision-reason",
        "strong_scheduler_baselines_for_server_count_sensitivity",
        "--revision-changed-module",
        "scheduler_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_unconverged_method",
        "--seed-partition",
        "generalization",
    ]
    if smoke:
        command.extend(
            ["--train-episodes", "200", "--eval-episodes", str(SMOKE_EPISODES)]
        )
    if resume:
        command.append("--resume")
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(log_path.parent / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + " ".join(command)
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


def run_jobs(stage: str, workers: int, resume: bool) -> None:
    jobs = [
        (servers, method)
        for servers in TRAINED_SERVER_COUNTS
        for method in METHODS
    ]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                run_logged,
                suite_command(stage, servers, method, resume),
                suite_dir(stage, servers, method) / "runner.log",
            ): (servers, method)
            for servers, method in jobs
        }
        for future in as_completed(futures):
            servers, method = futures[future]
            future.result()
            print(f"S={servers} {method} {stage}: complete", flush=True)


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    with (path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["phase"] == "eval"]


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
        "all_runs_complete": True,
        "all_learning_runs_converged": True,
        "all_tasks_exactly_once": True,
        "all_capacity_constraints_valid": True,
        "all_methods_scenario_paired_with_p11": True,
        "all_method_identities_valid": True,
        "servers": {},
    }
    expected_algorithms = {
        OUR_DQN: "causal_telemetryDDQN",
        COORD_SAC: "causal_telemetryDiscreteSAC",
    }
    for servers in scales:
        per_server = {"seeds": {}}
        for seed in seeds:
            reference = p11_run_dir(stage, servers, "daoc_paper", seed)
            reference_bank = comparable_bank(reference / "evaluation_scenarios.json")
            per_seed = {"methods": {}}
            for method in METHODS:
                directory = new_run_dir(stage, servers, method, seed)
                summary = read_json(directory / "summary.json")
                arguments = read_json(directory / "config.json")["arguments"]
                rows = read_eval_rows(directory)
                capacities = tuple(
                    sorted(int(value) for value in summary["server_capacities"].values())
                )
                complete = summary.get("status") == "complete"
                converged = smoke or bool(
                    summary.get("eligible_for_comparison")
                    and summary.get("convergence", {}).get("reached")
                )
                exact_once = len(rows) == episodes and all(
                    int(row["real_task_count"])
                    == int(row["completed_task_count"])
                    and int(row["all_tasks_executed_once"]) == 1
                    for row in rows
                )
                capacity_valid = (
                    arguments.get("num_servers") == servers
                    and arguments.get("num_users") == USERS
                    and arguments.get("num_services") == SERVICES
                    and float(arguments.get("bandwidth")) == BANDWIDTH_HZ
                    and capacities == tuple(sorted(CAPACITY_PROFILES[servers]))
                    and summary.get("total_server_capacity")
                    == sum(CAPACITY_PROFILES[servers])
                )
                paired = (
                    comparable_bank(directory / "evaluation_scenarios.json")
                    == reference_bank
                )
                identity = bool(
                    arguments.get("algorithm") == expected_algorithms[method]
                    and arguments.get("reward_mode")
                    == "causal_makespan_increment"
                    and arguments.get("cache_policy") == "critical_path_joint"
                    and arguments.get("cache_coverage_constraint") is True
                )
                audit["all_runs_complete"] &= complete
                audit["all_learning_runs_converged"] &= converged
                audit["all_tasks_exactly_once"] &= exact_once
                audit["all_capacity_constraints_valid"] &= capacity_valid
                audit["all_methods_scenario_paired_with_p11"] &= paired
                audit["all_method_identities_valid"] &= identity
                per_seed["methods"][method] = {
                    "complete": complete,
                    "converged": converged,
                    "tasks_exactly_once": exact_once,
                    "capacity_valid": capacity_valid,
                    "paired_with_p11": paired,
                    "method_identity_valid": identity,
                    "selected_checkpoint_episode": summary.get(
                        "selected_checkpoint_episode"
                    ),
                }
            per_server["seeds"][str(seed)] = per_seed
        audit["servers"][str(servers)] = per_server
    required = [key for key in audit if key.startswith("all_")]
    if not all(audit[key] for key in required):
        write_json(RESULT_ROOT / f"{stage.upper()}_AUDIT.json", audit)
        raise RuntimeError(f"{stage} strong-baseline audit failed")
    write_json(RESULT_ROOT / f"{stage.upper()}_AUDIT.json", audit)
    return audit


def method_run_dir(servers: int, method: str, seed: int) -> Path:
    if method in ("daoc_paper", "lean_our"):
        return p11_run_dir("converged", servers, method, seed)
    return new_run_dir("converged", servers, method, seed)


def ci95(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))


def aggregate() -> dict:
    audit = audit_stage("converged")
    results = {}
    flat_rows = []
    metrics = (
        "mean_completion_time_s",
        "p95_completion_time_s",
        "cache_hit_rate",
        "remote_loading_rate",
        "inference_ms_per_decision",
        "selected_checkpoint_episode",
    )
    for servers in SERVER_COUNTS:
        per_seed = []
        for seed in SEEDS:
            entry = {"seed": seed, "methods": {}}
            for method in ALL_METHODS:
                summary = read_json(method_run_dir(servers, method, seed) / "summary.json")
                values = {
                    "mean_completion_time_s": float(
                        summary["eval"]["mean_average_finish_time"]
                    ),
                    "p95_completion_time_s": float(
                        summary["eval"]["mean_p95_finish_time"]
                    ),
                    "cache_hit_rate": float(summary["eval"]["mean_cache_hit_rate"]),
                    "remote_loading_rate": float(
                        summary["eval"]["mean_cache_remote_loading_rate"]
                    ),
                    "inference_ms_per_decision": float(
                        summary["eval"][
                            "mean_policy_inference_time_per_decision_ms"
                        ]
                    ),
                    "selected_checkpoint_episode": int(
                        summary["selected_checkpoint_episode"]
                    ),
                }
                entry["methods"][method] = values
                flat_rows.append(
                    {
                        "servers": servers,
                        "seed": seed,
                        "method": DISPLAY[method],
                        "method_key": method,
                        "source": "reused" if servers == 10 else "p12_retrained",
                        **values,
                    }
                )
            per_seed.append(entry)
        method_summary = {}
        for method in ALL_METHODS:
            method_summary[method] = {}
            for metric in metrics:
                values = [row["methods"][method][metric] for row in per_seed]
                mean, half = ci95(values)
                method_summary[method][metric] = {
                    "mean": mean,
                    "ci95_half_width": half,
                    "per_seed": values,
                }
        comparisons = {}
        ours = np.asarray(
            [row["methods"]["lean_our"]["mean_completion_time_s"] for row in per_seed]
        )
        for method in ("daoc_paper", OUR_DQN, COORD_SAC):
            baseline = np.asarray(
                [row["methods"][method]["mean_completion_time_s"] for row in per_seed]
            )
            reductions = 100.0 * (baseline - ours) / baseline
            mean, half = ci95(reductions)
            comparisons[f"our_vs_{method}"] = {
                "relative_reduction_percent_mean": mean,
                "relative_reduction_percent_ci95_half_width": half,
                "relative_reduction_percent_per_seed": reductions.tolist(),
                "seed_wins": int(np.sum(ours < baseline)),
                "seed_count": len(SEEDS),
                "wilcoxon_one_sided_p": float(
                    stats.wilcoxon(baseline, ours, alternative="greater").pvalue
                ),
            }
        results[str(servers)] = {
            "per_seed": per_seed,
            "method_summary": method_summary,
            "comparisons": comparisons,
        }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with (ANALYSIS_DIR / "server_scaling_four_methods_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {
        "status": "complete",
        "protocol": validate_protocol(),
        "integrity": {key: value for key, value in audit.items() if key.startswith("all_")},
        "results": results,
        "statistical_scope": (
            "descriptive three-seed sensitivity; not a ten-seed superiority test"
        ),
    }
    write_json(ANALYSIS_DIR / "server_scaling_four_methods_summary.json", summary)
    write_report(summary)
    plot_results(summary)
    return summary


def write_report(summary: dict) -> None:
    lines = [
        "# 服务器数量敏感性：强学习基线扩展",
        "",
        "- 固定20用户、10服务、15 kHz和五类Pegasus工作流。",
        "- Seeds 51--53，每seed 100个完全配对场景。",
        "- S=5/15/20从头训练；S=10复用完全相同协议的正式结果。",
        "- OUR-DQN只将OUR的Pairwise PD3QN替换为普通Double DQN。",
        "- CoordCache-DiscreteSAC保留相同协调缓存和信息协议。",
        "",
        "| Servers | DAOC | OUR-DQN | CoordCache-SAC | OUR | OUR vs DQN | OUR vs SAC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for servers in SERVER_COUNTS:
        row = summary["results"][str(servers)]
        methods = row["method_summary"]
        comparisons = row["comparisons"]
        lines.append(
            f"| {servers} | "
            f"{methods['daoc_paper']['mean_completion_time_s']['mean']:.4f} | "
            f"{methods[OUR_DQN]['mean_completion_time_s']['mean']:.4f} | "
            f"{methods[COORD_SAC]['mean_completion_time_s']['mean']:.4f} | "
            f"{methods['lean_our']['mean_completion_time_s']['mean']:.4f} | "
            f"{comparisons[f'our_vs_{OUR_DQN}']['relative_reduction_percent_mean']:.2f}% | "
            f"{comparisons[f'our_vs_{COORD_SAC}']['relative_reduction_percent_mean']:.2f}% |"
        )
    (ANALYSIS_DIR / "SERVER_SCALING_FOUR_METHODS_REPORT_ZH.md").write_text(
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
    panels = (
        ("mean_completion_time_s", "Mean completion time (s)", "(a)"),
        ("p95_completion_time_s", "P95 completion time (s)", "(b)"),
    )
    x = np.asarray(SERVER_COUNTS, dtype=float)
    for axis, (metric, ylabel, panel) in zip(axes, panels):
        for method in ALL_METHODS:
            means = [
                summary["results"][str(servers)]["method_summary"][method][metric]["mean"]
                for servers in SERVER_COUNTS
            ]
            errors = [
                summary["results"][str(servers)]["method_summary"][method][metric]["ci95_half_width"]
                for servers in SERVER_COUNTS
            ]
            axis.errorbar(
                x,
                means,
                yerr=errors,
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor=(COLORS[method] if method == "lean_our" else "white"),
                markeredgecolor=COLORS[method],
                linewidth=1.45 if method == "lean_our" else 1.15,
                markersize=6.5 if method == "lean_our" else 4.5,
                capsize=2.2,
                label=DISPLAY[method],
            )
        axis.set_xticks(SERVER_COUNTS)
        axis.set_xlabel("Number of edge servers")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D7DADD", linewidth=0.6, linestyle="--")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.01, 0.98, panel, transform=axis.transAxes,
            ha="left", va="top", fontweight="bold"
        )
    axes[0].legend(frameon=False, loc="best", fontsize=7)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "fig10_server_count_sensitivity_4methods",
        "fig10_server_count_sensitivity",
    ):
        for suffix in ("pdf", "svg", "png"):
            figure.savefig(
                FIGURE_DIR / f"{name}.{suffix}",
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
            "test_pegasus_server_scaling_strong_baselines.py",
            "test_pegasus_server_scaling.py",
            "test_capacity_protocol.py",
            "test_dag_completion_semantics.py",
            "test_discrete_sac.py",
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
        run_jobs("smoke", args.workers, args.resume)
        audit_stage("smoke")
    if args.stage in ("train", "all"):
        run_jobs("converged", args.workers, args.resume)
    if args.stage in ("analysis", "all"):
        aggregate()
    print(RESULT_ROOT)


if __name__ == "__main__":
    main()

