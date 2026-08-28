#!/usr/bin/env python3
"""Add standard-cache heuristics to the Pegasus server-count figure."""

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
from pegasus_server_scaling_heuristics_protocol import (
    CALIBRATION_EPISODES,
    DISPLAY_NAMES,
    FULL_PROFILE,
    METHODS,
    NEAREST,
    NEAREST_SERVICE,
    P11_ROOT,
    P12_ROOT,
    P6_HEURISTIC_ROOT,
    PROTOCOL_VERSION,
    RANDOM,
    RESULT_ROOT,
    SMOKE_PROFILE,
    canonical_hash,
    validate_protocol,
)
from pegasus_server_scaling_strong_baselines_protocol import (
    COORD_SAC,
    OUR_DQN,
    REFERENCE_ROOTS,
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
    "dqn.py",
    "pegasus_server_scaling_heuristics_protocol.py",
    "run_independent_experiment.py",
    "run_pegasus_server_scaling_heuristics.py",
    "run_reproduction_suite.py",
    "server.py",
    "simulator.py",
    "task.py",
    "test_pegasus_server_scaling_heuristics.py",
    "user.py",
)

ALL_METHODS = (
    RANDOM,
    NEAREST,
    NEAREST_SERVICE,
    "daoc_paper",
    OUR_DQN,
    COORD_SAC,
    "lean_our",
)
DISPLAY = {
    **DISPLAY_NAMES,
    "daoc_paper": "DAOC",
    OUR_DQN: "OUR-DQN",
    COORD_SAC: "CoordCache-DiscreteSAC",
    "lean_our": "OUR",
}
COLORS = {
    RANDOM: "#A7ADB4",
    NEAREST: "#7A8792",
    NEAREST_SERVICE: "#4F5D67",
    "daoc_paper": "#756A86",
    OUR_DQN: "#4E8A74",
    COORD_SAC: "#C1774A",
    "lean_our": "#176B9A",
}
MARKERS = {
    RANDOM: "^",
    NEAREST: "x",
    NEAREST_SERVICE: "+",
    "daoc_paper": "o",
    OUR_DQN: "s",
    COORD_SAC: "D",
    "lean_our": "*",
}
LINESTYLES = {
    RANDOM: (0, (1.2, 1.6)),
    NEAREST: (0, (4.0, 2.2)),
    NEAREST_SERVICE: (0, (6.5, 2.0, 1.2, 2.0)),
    "daoc_paper": "-",
    OUR_DQN: "-",
    COORD_SAC: "-",
    "lean_our": "-",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("tests", "smoke", "heuristics", "analysis", "all"),
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
                raise RuntimeError(f"P13 protocol changed: {key}")
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


def p12_run_dir(servers: int, method: str, seed: int) -> Path:
    if servers == 10:
        return REFERENCE_ROOTS[method] / f"seed_{seed}"
    return (
        P12_ROOT
        / "converged"
        / f"servers_{servers}"
        / method
        / "runs"
        / method
        / f"seed_{seed}"
    )


def heuristic_run_dir(
    stage: str, servers: int, method: str, seed: int
) -> Path:
    if stage == "converged" and servers == 10:
        return P6_HEURISTIC_ROOT / method / f"seed_{seed}"
    return (
        suite_dir(stage, servers, method)
        / "runs"
        / method
        / f"seed_{seed}"
    )


def suite_command(
    stage: str, servers: int, method: str, resume: bool
) -> list[str]:
    smoke = stage == "smoke"
    command = [
        sys.executable,
        str(ROOT / "run_reproduction_suite.py"),
        "--profile",
        SMOKE_PROFILE if smoke else FULL_PROFILE,
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
        "standard_cache_heuristics_for_server_count_sensitivity",
        "--revision-changed-module",
        "heuristic_scale_only",
        "--revision-expected-metric",
        "paired_full_dag_completion_time",
        "--revision-rejection-condition",
        "integrity_failure_or_scenario_mismatch",
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
    expected_checkpoint = 200 if smoke else CALIBRATION_EPISODES
    expected_algorithms = {
        RANDOM: "random",
        NEAREST: "nearest_server",
        NEAREST_SERVICE: "nearest_with_service",
    }
    audit = {
        "status": "complete",
        "stage": stage,
        "all_runs_complete": True,
        "all_cache_calibrations_complete": True,
        "all_tasks_exactly_once": True,
        "all_capacity_constraints_valid": True,
        "all_methods_scenario_paired_with_p11": True,
        "all_method_identities_valid": True,
        "servers": {},
    }
    for servers in scales:
        per_server = {"seeds": {}}
        for seed in seeds:
            reference = p11_run_dir(stage, servers, "daoc_paper", seed)
            reference_bank = comparable_bank(reference / "evaluation_scenarios.json")
            per_seed = {"methods": {}}
            for method in METHODS:
                directory = heuristic_run_dir(stage, servers, method, seed)
                summary = read_json(directory / "summary.json")
                arguments = read_json(directory / "config.json")["arguments"]
                rows = read_eval_rows(directory)
                capacities = tuple(
                    sorted(int(value) for value in summary["server_capacities"].values())
                )
                complete = bool(
                    summary.get("status") == "complete"
                    and summary.get("eligible_for_comparison")
                    and len(rows) == episodes
                )
                calibrated = bool(
                    summary.get("selected_checkpoint_episode")
                    == expected_checkpoint
                    and summary.get("convergence", {}).get("stop_reason")
                    == "fixed_budget"
                )
                exact_once = len(rows) == episodes and all(
                    int(row["real_task_count"])
                    == int(row["completed_task_count"])
                    and int(row["all_tasks_executed_once"]) == 1
                    for row in rows
                )
                capacity_valid = bool(
                    arguments.get("num_servers") == servers
                    and arguments.get("num_users") == USERS
                    and arguments.get("num_services") == SERVICES
                    and float(arguments.get("bandwidth")) == BANDWIDTH_HZ
                    and capacities == tuple(sorted(CAPACITY_PROFILES[servers]))
                    and summary.get("total_server_capacity")
                    == sum(CAPACITY_PROFILES[servers])
                )
                paired = bool(
                    comparable_bank(directory / "evaluation_scenarios.json")
                    == reference_bank
                )
                identity = bool(
                    arguments.get("algorithm") == expected_algorithms[method]
                    and arguments.get("reward_mode") == "terminal_binary"
                    and arguments.get("cache_policy") == "popularity_ema"
                    and arguments.get("cache_coverage_constraint") is False
                )
                audit["all_runs_complete"] &= complete
                audit["all_cache_calibrations_complete"] &= calibrated
                audit["all_tasks_exactly_once"] &= exact_once
                audit["all_capacity_constraints_valid"] &= capacity_valid
                audit["all_methods_scenario_paired_with_p11"] &= paired
                audit["all_method_identities_valid"] &= identity
                per_seed["methods"][method] = {
                    "complete": complete,
                    "cache_calibrated": calibrated,
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
    write_json(RESULT_ROOT / f"{stage.upper()}_AUDIT.json", audit)
    if not all(audit[key] for key in required):
        raise RuntimeError(f"{stage} heuristic audit failed")
    return audit


def method_run_dir(servers: int, method: str, seed: int) -> Path:
    if method in METHODS:
        return heuristic_run_dir("converged", servers, method, seed)
    if method in ("daoc_paper", "lean_our"):
        return p11_run_dir("converged", servers, method, seed)
    if method in (OUR_DQN, COORD_SAC):
        return p12_run_dir(servers, method, seed)
    raise KeyError(method)


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
                if method in METHODS:
                    source = "p6_reused" if servers == 10 else "p13_calibrated"
                elif method in ("daoc_paper", "lean_our"):
                    source = "p11_reused"
                else:
                    source = "p12_reused"
                flat_rows.append(
                    {
                        "servers": servers,
                        "seed": seed,
                        "method": DISPLAY[method],
                        "method_key": method,
                        "source": source,
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
        for method in ALL_METHODS[:-1]:
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
    with (ANALYSIS_DIR / "server_scaling_seven_methods_per_seed.csv").open(
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
        "plotted_metric": "mean_completion_time_s",
        "p95_retained_in_data_but_not_plotted": True,
        "statistical_scope": (
            "descriptive three-seed sensitivity; not a ten-seed superiority test"
        ),
    }
    write_json(ANALYSIS_DIR / "server_scaling_seven_methods_summary.json", summary)
    write_report(summary)
    plot_results(summary)
    return summary


def write_report(summary: dict) -> None:
    lines = [
        "# 服务器数量敏感性：启发式与强学习基线",
        "",
        "- 固定20用户、10服务、15 kHz和五类Pegasus工作流。",
        "- Seeds 51--53，每seed 100个完全配对场景。",
        "- Random、Nearest和Nearest+Service使用独立流行度EMA缓存，不使用OUR协调缓存。",
        "- S=5/15/20新运5000轮缓存校准；S=10复用完全同协议的P6结果。",
        "- 图仅绘制平均DAG完成时间；P95数值仍保留在JSON/CSV中。",
        "",
        "| Servers | Random | Nearest | Nearest+Service | DAOC | OUR-DQN | CoordCache-SAC | OUR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for servers in SERVER_COUNTS:
        methods = summary["results"][str(servers)]["method_summary"]
        values = [
            methods[method]["mean_completion_time_s"]["mean"]
            for method in ALL_METHODS
        ]
        lines.append(
            f"| {servers} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    (ANALYSIS_DIR / "SERVER_SCALING_SEVEN_METHODS_REPORT_ZH.md").write_text(
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
    figure, axis = plt.subplots(
        figsize=(155 * mm, 82 * mm), layout="constrained"
    )
    x = np.asarray(SERVER_COUNTS, dtype=float)
    upper_extent = 0.0
    for method in ALL_METHODS:
        means = np.asarray(
            [
                summary["results"][str(servers)]["method_summary"][method][
                    "mean_completion_time_s"
                ]["mean"]
                for servers in SERVER_COUNTS
            ],
            dtype=float,
        )
        errors = np.asarray(
            [
                summary["results"][str(servers)]["method_summary"][method][
                    "mean_completion_time_s"
                ]["ci95_half_width"]
                for servers in SERVER_COUNTS
            ],
            dtype=float,
        )
        upper_extent = max(upper_extent, float(np.max(means + errors)))
        is_ours = method == "lean_our"
        open_marker = method not in (NEAREST, NEAREST_SERVICE, "lean_our")
        axis.errorbar(
            x,
            means,
            yerr=errors,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markerfacecolor="white" if open_marker else COLORS[method],
            markeredgecolor=COLORS[method],
            markeredgewidth=0.8,
            linewidth=1.55 if is_ours else 1.1,
            markersize=7.0 if is_ours else 4.8,
            elinewidth=0.75,
            capsize=2.0,
            zorder=5 if is_ours else 3,
            label=DISPLAY[method],
        )
    axis.set_xticks(SERVER_COUNTS)
    axis.set_xlabel("Number of edge servers")
    axis.set_ylabel("Mean DAG completion time (s)")
    axis.set_ylim(0.0, upper_extent * 1.28)
    axis.grid(axis="y", color="#D7DADD", linewidth=0.6, linestyle="--")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(
        loc="upper right",
        ncol=2,
        fontsize=7.0,
        frameon=True,
        borderpad=0.45,
        columnspacing=0.9,
        handlelength=2.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.94)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "fig10_server_count_sensitivity_7methods",
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
            "test_pegasus_server_scaling_heuristics.py",
            "test_pegasus_server_scaling_strong_baselines.py",
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
        run_jobs("smoke", args.workers, args.resume)
        audit_stage("smoke")
    if args.stage in ("heuristics", "all"):
        run_jobs("converged", args.workers, args.resume)
    if args.stage in ("analysis", "all"):
        aggregate()
    print(RESULT_ROOT)


if __name__ == "__main__":
    main()
