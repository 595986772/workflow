#!/usr/bin/env python3
"""Evaluate frozen Alibaba-CP100 checkpoints at multiple user counts."""

import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_strict_environment_suite import paired_statistics
from run_independent_experiment import (
    EPISODE_FIELDS,
    apply_frozen_state,
    capture_deployment_state,
    run_scenario_bank_evaluation,
    seed_everything,
    summarize_rows,
)
from simulator import MEC_Simulator


PROTOCOL_VERSION = "alibaba_cp100_user_scaling_v1"
DAOC_LABEL = "guided_full"
OUR_LABEL = "lean_our"
DEFAULT_USERS = (10, 20, 40, 60)


def parse_list(value):
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def parse_int_list(value):
    return [int(item) for item in parse_list(value)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen DAOC/OUR checkpoints at 10-60 users."
        )
    )
    parser.add_argument(
        "--source-suite-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=parse_list,
        default=[DAOC_LABEL, OUR_LABEL],
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        required=True,
    )
    parser.add_argument(
        "--user-counts",
        type=parse_int_list,
        default=list(DEFAULT_USERS),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-benchmark-repeats", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.labels != [DAOC_LABEL, OUR_LABEL]:
        raise ValueError(
            "Scaling protocol requires guided_full,lean_our in that order"
        )
    if (
        not args.seeds
        or not args.user_counts
        or min(args.user_counts) < 1
        or args.episodes < 1
        or args.workers < 1
        or args.cache_benchmark_repeats < 1
    ):
        raise ValueError("Scaling arguments must be positive and non-empty")
    return args


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def source_run(source_suite, label, seed):
    return source_suite / "runs" / label / f"seed_{seed}"


def target_run(output_dir, users, label, seed):
    return (
        output_dir
        / f"users_{users}"
        / "runs"
        / label
        / f"seed_{seed}"
    )


def project_user_count_state(simulator, source_state):
    """Reuse learned state while deterministically extending user deadlines."""
    projected = copy.deepcopy(source_state)
    source_deadlines = [
        float(deadline)
        for _, deadline in sorted(source_state["deadlines"].items())
    ]
    if not source_deadlines:
        raise RuntimeError("Source checkpoint has no user deadlines")
    projected["deadlines"] = {
        user_id: source_deadlines[user_id % len(source_deadlines)]
        for user_id in simulator.users
    }
    projected["replay_sizes"] = {
        server_id: len(
            server.agent.agent.TrainNet.experience["s"]
        )
        for server_id, server in simulator.servers.items()
    }
    return projected


def canonical_deployment(source_suite, seed, users, output_dir):
    run = source_run(source_suite, DAOC_LABEL, seed)
    config = read_json(run / "config.json")
    input_config = copy.deepcopy(config["input_config"])
    input_config.update(
        {
            "Number of users": users,
            "seed": seed,
            "save topology figure": False,
        }
    )
    seed_everything(seed)
    simulator = MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=input_config,
        learning_arguments=copy.deepcopy(config["learning_config"]),
        filename_png=str(output_dir),
    )
    return capture_deployment_state(simulator)


def native_cache_decision(simulator):
    if simulator.cache_policy in {
        "popularity_coordinated",
        "critical_path_coordinated",
        "critical_path_joint",
    }:
        return simulator.broker.coordinated_caching_decisions()
    return {
        server_id: simulator.broker.caching_decisions(server_id)
        for server_id in simulator.servers
    }


def benchmark_cache_decision(simulator, repeats):
    for _ in range(5):
        native_cache_decision(simulator)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        native_cache_decision(simulator)
        timings.append(
            1000.0 * (time.perf_counter() - started)
        )
    return {
        "repeats": repeats,
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def expected_complete(path, users, episodes):
    if not path.exists():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "complete"
        and summary.get("protocol_version") == PROTOCOL_VERSION
        and summary.get("num_users") == users
        and summary.get("eval_episodes") == episodes
        and summary.get("model_frozen") is True
        and summary.get("cache_frozen") is True
    )


def evaluate_one(spec):
    (
        source_suite,
        output_dir,
        canonical,
        label,
        seed,
        users,
        episodes,
        cache_repeats,
        resume,
    ) = spec
    source = source_run(source_suite, label, seed)
    target = target_run(output_dir, users, label, seed)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    if resume and expected_complete(summary_path, users, episodes):
        return {
            "label": label,
            "seed": seed,
            "users": users,
            "skipped": True,
        }

    source_summary = read_json(source / "summary.json")
    if (
        source_summary.get("status") != "complete"
        or source_summary.get("eligible_for_comparison") is not True
    ):
        raise RuntimeError(f"Ineligible source checkpoint: {source}")
    config = read_json(source / "config.json")
    source_arguments = config["arguments"]
    input_config = copy.deepcopy(config["input_config"])
    input_config.update(
        {
            "Number of users": users,
            "seed": seed,
            "save topology figure": False,
        }
    )
    learning_config = copy.deepcopy(config["learning_config"])
    checkpoint = torch.load(
        source / "selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )

    log_path = target / "simulator.log"
    with log_path.open("w", encoding="utf-8") as log:
        seed_everything(seed)
        simulator = MEC_Simulator(
            outputfile=log,
            Input_dict=input_config,
            learning_arguments=learning_config,
            filename_png=str(target),
        )
        projected_state = project_user_count_state(
            simulator,
            checkpoint["frozen_state"],
        )
        apply_frozen_state(simulator, projected_state)
        cache_benchmark = benchmark_cache_decision(
            simulator,
            cache_repeats,
        )

        eval_args = SimpleNamespace(
            eval_episodes=episodes,
            eval_seed_offset=int(
                source_arguments["eval_seed_offset"]
            ),
            seed=seed,
            eval_bank_scope="workload",
            label=label,
            quiet=True,
        )
        episodes_path = target / "episodes.csv"
        with episodes_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output:
            writer = csv.DictWriter(
                output,
                fieldnames=EPISODE_FIELDS,
            )
            writer.writeheader()
            rows, scenarios = run_scenario_bank_evaluation(
                writer=writer,
                csv_file=output,
                args=eval_args,
                input_config=input_config,
                learning_config=learning_config,
                frozen_state=projected_state,
                deployment_state=canonical,
                simulator_log=log,
                figures_dir=target,
                episodes=episodes,
                phase="scale_eval",
            )

    write_json(target / "evaluation_scenarios.json", scenarios)
    total_inference_calls = sum(
        int(row["policy_inference_calls"]) for row in rows
    )
    total_inference_time = sum(
        float(row["policy_inference_wall_time_sec"])
        for row in rows
    )
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "label": label,
        "seed": seed,
        "num_users": users,
        "eval_episodes": episodes,
        "source_run": str(source.resolve()),
        "source_checkpoint_episode": int(checkpoint["episode"]),
        "model_frozen": True,
        "cache_frozen": True,
        "target_retraining": False,
        "deadline_projection": {
            "mode": "cyclic_checkpoint_deadlines",
            "used_future_target_workload": False,
        },
        "eval": summarize_rows(rows),
        "policy_inference": {
            "calls": total_inference_calls,
            "wall_time_sec": total_inference_time,
            "milliseconds_per_task": (
                1000.0
                * total_inference_time
                / total_inference_calls
                if total_inference_calls
                else 0.0
            ),
        },
        "native_cache_decision_benchmark": cache_benchmark,
    }
    write_json(summary_path, summary)
    return {
        "label": label,
        "seed": seed,
        "users": users,
        "skipped": False,
    }


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def reference_reproduction_audit(
    source_suite,
    output_dir,
    labels,
    seeds,
    reference_users=20,
):
    fields = (
        "scenario_fingerprint",
        "average_finish_time",
        "p95_finish_time",
        "cache_matrix_json",
        "server_action_histogram_json",
    )
    combinations = []
    for label in labels:
        for seed in seeds:
            source = [
                row
                for row in read_rows(
                    source_run(
                        source_suite,
                        label,
                        seed,
                    )
                    / "episodes.csv"
                )
                if row["phase"] == "eval"
            ]
            scaled = read_rows(
                target_run(
                    output_dir,
                    reference_users,
                    label,
                    seed,
                )
                / "episodes.csv"
            )
            mismatch_counts = {
                field: sum(
                    left[field] != right[field]
                    for left, right in zip(source, scaled)
                )
                for field in fields
            }
            combinations.append(
                {
                    "label": label,
                    "seed": seed,
                    "source_rows": len(source),
                    "scaled_rows": len(scaled),
                    "mismatch_counts": mismatch_counts,
                }
            )
    return {
        "reference_users": reference_users,
        "fields": list(fields),
        "all_exact": all(
            row["source_rows"] == row["scaled_rows"]
            and not any(row["mismatch_counts"].values())
            for row in combinations
        ),
        "combinations": combinations,
    }


def aggregate(
    source_suite,
    output_dir,
    user_counts,
    labels,
    seeds,
):
    results = {}
    all_paired = True
    for users in user_counts:
        per_seed = []
        for seed in seeds:
            rows_by_label = {}
            summaries = {}
            for label in labels:
                run = target_run(output_dir, users, label, seed)
                rows_by_label[label] = read_rows(
                    run / "episodes.csv"
                )
                summaries[label] = read_json(run / "summary.json")
            fingerprints = {
                label: [
                    row["scenario_fingerprint"] for row in rows
                ]
                for label, rows in rows_by_label.items()
            }
            paired = (
                len(
                    {
                        tuple(values)
                        for values in fingerprints.values()
                    }
                )
                == 1
            )
            all_paired = all_paired and paired
            entry = {
                "seed": seed,
                "scenario_paired": paired,
            }
            for label in labels:
                rows = rows_by_label[label]
                entry[label] = {
                    "mean_finish_time": float(
                        np.mean(
                            [
                                float(row["average_finish_time"])
                                for row in rows
                            ]
                        )
                    ),
                    "mean_p95_finish_time": float(
                        np.mean(
                            [
                                float(row["p95_finish_time"])
                                for row in rows
                            ]
                        )
                    ),
                    "inference_ms_per_task": summaries[label][
                        "policy_inference"
                    ]["milliseconds_per_task"],
                    "cache_decision_mean_ms": summaries[label][
                        "native_cache_decision_benchmark"
                    ]["mean_ms"],
                }
            per_seed.append(entry)
        daoc_mean = [
            row[DAOC_LABEL]["mean_finish_time"] for row in per_seed
        ]
        our_mean = [
            row[OUR_LABEL]["mean_finish_time"] for row in per_seed
        ]
        daoc_p95 = [
            row[DAOC_LABEL]["mean_p95_finish_time"]
            for row in per_seed
        ]
        our_p95 = [
            row[OUR_LABEL]["mean_p95_finish_time"]
            for row in per_seed
        ]
        results[str(users)] = {
            "per_seed": per_seed,
            "finish_time": paired_statistics(
                daoc_mean,
                our_mean,
                lower_is_better=True,
            ),
            "p95_finish_time": paired_statistics(
                daoc_p95,
                our_p95,
                lower_is_better=True,
            ),
            "mean_inference_ms_per_task": {
                label: float(
                    np.mean(
                        [
                            row[label]["inference_ms_per_task"]
                            for row in per_seed
                        ]
                    )
                )
                for label in labels
            },
            "mean_cache_decision_ms": {
                label: float(
                    np.mean(
                        [
                            row[label]["cache_decision_mean_ms"]
                            for row in per_seed
                        ]
                    )
                )
                for label in labels
            },
        }
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "labels": labels,
        "seeds": seeds,
        "user_counts": user_counts,
        "episodes_per_seed": len(
            read_rows(
                target_run(
                    output_dir,
                    user_counts[0],
                    labels[0],
                    seeds[0],
                )
                / "episodes.csv"
            )
        ),
        "all_methods_scenario_paired": all_paired,
        "model_frozen": True,
        "cache_frozen": True,
        "target_retraining": False,
        "reference_reproduction_audit": (
            reference_reproduction_audit(
                source_suite,
                output_dir,
                labels,
                seeds,
            )
        ),
        "results": results,
    }
    write_json(output_dir / "user_scaling_summary.json", summary)
    plot_scaling(output_dir / "user_scaling", summary)
    write_report(output_dir / "USER_SCALING_REPORT.md", summary)
    return summary


def plot_scaling(path, summary):
    users = summary["user_counts"]
    results = summary["results"]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.2),
        constrained_layout=True,
    )
    methods = (
        (DAOC_LABEL, "DAOC", "#59636E", "o"),
        (OUR_LABEL, "OUR", "#D65F45", "s"),
    )
    for label, display, color, marker in methods:
        axes[0, 0].plot(
            users,
            [
                results[str(count)]["finish_time"][
                    "reference_mean"
                    if label == DAOC_LABEL
                    else "candidate_mean"
                ]
                for count in users
            ],
            marker=marker,
            color=color,
            label=display,
        )
        axes[0, 1].plot(
            users,
            [
                results[str(count)]["p95_finish_time"][
                    "reference_mean"
                    if label == DAOC_LABEL
                    else "candidate_mean"
                ]
                for count in users
            ],
            marker=marker,
            color=color,
            label=display,
        )
        axes[1, 0].plot(
            users,
            [
                results[str(count)]["mean_inference_ms_per_task"][
                    label
                ]
                for count in users
            ],
            marker=marker,
            color=color,
            label=display,
        )
        axes[1, 1].plot(
            users,
            [
                results[str(count)]["mean_cache_decision_ms"][
                    label
                ]
                for count in users
            ],
            marker=marker,
            color=color,
            label=display,
        )
    axes[0, 0].set_ylabel("Mean completion time (s)")
    axes[0, 1].set_ylabel("Mean episode P95 (s)")
    axes[1, 0].set_ylabel("Policy inference (ms/task)")
    axes[1, 1].set_ylabel("Native cache decision (ms/call)")
    for axis in axes.flat:
        axis.set_xlabel("Concurrent users")
        axis.set_xticks(users)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path, summary):
    lines = [
        "# Alibaba-CP100 三Seed用户规模实验",
        "",
        "- 网络与缓存 checkpoint 均冻结，目标规模不重新训练。",
        f"- Seeds：`{summary['seeds']}`；每 seed "
        f"`{summary['episodes_per_seed']}` 个配对场景。",
        f"- 场景逐项配对：`{summary['all_methods_scenario_paired']}`。",
        "- 20用户点逐行复现原实验："
        f"`{summary['reference_reproduction_audit']['all_exact']}`。",
        "",
        "| Users | Mean DAOC | Mean OUR | OUR改善 | P95改善 | "
        "DAOC推理ms | OUR推理ms | OUR协调ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for users in summary["user_counts"]:
        result = summary["results"][str(users)]
        finish = result["finish_time"]
        p95 = result["p95_finish_time"]
        lines.append(
            f"| {users} | {finish['reference_mean']:.6f} | "
            f"{finish['candidate_mean']:.6f} | "
            f"{finish['mean_paired_improvement_percent']:.3f}% | "
            f"{p95['mean_paired_improvement_percent']:.3f}% | "
            f"{result['mean_inference_ms_per_task'][DAOC_LABEL]:.4f} | "
            f"{result['mean_inference_ms_per_task'][OUR_LABEL]:.4f} | "
            f"{result['mean_cache_decision_ms'][OUR_LABEL]:.4f} |"
        )
    lines.extend(
        [
            "",
            "三 seed 结果用于轻量级扩展性诊断，不单独作正式显著性结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    torch.set_num_threads(1)
    source_suite = args.source_suite_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = {
        (seed, users): canonical_deployment(
            source_suite,
            seed,
            users,
            output_dir,
        )
        for seed in args.seeds
        for users in args.user_counts
    }
    specs = [
        (
            source_suite,
            output_dir,
            canonical[(seed, users)],
            label,
            seed,
            users,
            args.episodes,
            args.cache_benchmark_repeats,
            args.resume,
        )
        for users in args.user_counts
        for label in args.labels
        for seed in args.seeds
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(evaluate_one, spec)
            for spec in specs
        ]
        for future in as_completed(futures):
            result = future.result()
            action = "skip" if result["skipped"] else "complete"
            print(
                f"[{action}] users={result['users']} "
                f"{result['label']} seed={result['seed']}",
                flush=True,
            )
    summary = aggregate(
        source_suite,
        output_dir,
        args.user_counts,
        args.labels,
        args.seeds,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
