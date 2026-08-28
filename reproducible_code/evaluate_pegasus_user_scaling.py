#!/usr/bin/env python3
"""Evaluate frozen Pegasus checkpoints across concurrent-user counts."""

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from analyze_a0_coordination import paired_superiority
from pegasus_paper_supplement_protocol import (
    FAMILIES,
    PROTOCOL_VERSION as PARENT_PROTOCOL_VERSION,
)
from run_independent_experiment import (
    EPISODE_FIELDS,
    apply_frozen_state,
    capture_deployment_state,
    run_scenario_bank_evaluation,
    seed_everything,
    summarize_rows,
)
from simulator import MEC_Simulator


PROTOCOL_VERSION = f"{PARENT_PROTOCOL_VERSION}_user_scaling_v1"
COORDINATED_POLICIES = {
    "popularity_coordinated",
    "critical_path_coordinated",
    "critical_path_joint",
}


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in parse_list(value)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=parse_list, required=True)
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    parser.add_argument("--user-counts", type=parse_int_list, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-benchmark-repeats", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        not args.labels
        or not args.seeds
        or not args.user_counts
        or min(args.user_counts) < 1
        or args.episodes < 1
        or args.workers < 1
        or args.cache_benchmark_repeats < 1
    ):
        raise ValueError("Scaling arguments must be positive and non-empty")
    if "lean_our" not in args.labels or "daoc_paper" not in args.labels:
        raise ValueError("Scaling requires daoc_paper and lean_our")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def source_run(source_suite, label, seed):
    return Path(source_suite) / "runs" / label / f"seed_{seed}"


def target_run(output_dir, users, label, seed):
    return (
        Path(output_dir)
        / f"users_{users}"
        / "runs"
        / label
        / f"seed_{seed}"
    )


def project_user_count_state(simulator, source_state):
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
        server_id: len(server.agent.agent.TrainNet.experience["s"])
        for server_id, server in simulator.servers.items()
    }
    return projected


def canonical_deployment(source_suite, reference_label, seed, users, output_dir):
    run = source_run(source_suite, reference_label, seed)
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
    if simulator.cache_policy in COORDINATED_POLICIES:
        return simulator.broker.coordinated_caching_decisions()
    return {
        server_id: simulator.broker.caching_decisions(server_id)
        for server_id in simulator.servers
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compact_json_bytes(value):
    return len(
        json.dumps(
            json_ready(value),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def coordination_payload(simulator, decisions):
    if simulator.cache_policy not in COORDINATED_POLICIES:
        return {
            "required": False,
            "uplink_bytes": 0,
            "downlink_bytes": 0,
            "total_bytes": 0,
            "schema": "independent_local_cache",
        }
    broker = simulator.broker
    current_services = {
        server_id: [
            service_id
            for service_id in server.services
            if service_id > 0
        ]
        for server_id, server in simulator.servers.items()
    }
    normalized = getattr(
        broker,
        "last_cache_decision_context",
        {},
    )
    per_server = {}
    for server_id, server in simulator.servers.items():
        demand = broker.H.get(server_id, {})
        record = {
            "server_id": server_id,
            "service_demand": demand,
            "capacity": int(server.capacity),
            "current_services": current_services[server_id],
        }
        if simulator.cache_policy == "critical_path_joint":
            record["compute_per_mcycle_ema"] = (
                broker.cache_server_compute_per_mcycle_ema.get(server_id)
            )
            record["waiting_latency_ema"] = (
                broker.cache_server_waiting_latency_ema.get(server_id)
            )
            record["sample_count"] = int(
                broker.cache_server_sample_counts.get(server_id, 0)
            )
            record["last_observed_window"] = (
                broker.cache_server_last_observed_window.get(server_id)
            )
        per_server[server_id] = record
    uplink = {
        "servers": per_server,
        "expected_requests": normalized.get("expected_requests", {}),
    }
    downlink = {
        server_id: [int(service_id) for service_id in services]
        for server_id, services in decisions.items()
    }
    uplink_bytes = compact_json_bytes(uplink)
    downlink_bytes = compact_json_bytes(downlink)
    return {
        "required": True,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes,
        "total_bytes": uplink_bytes + downlink_bytes,
        "schema": (
            "demand_capacity_cache_plus_causal_telemetry"
            if simulator.cache_policy == "critical_path_joint"
            else "demand_capacity_cache"
        ),
    }


def benchmark_cache_decision(simulator, repeats):
    decisions = native_cache_decision(simulator)
    payload = coordination_payload(simulator, decisions)
    for _ in range(5):
        native_cache_decision(simulator)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        native_cache_decision(simulator)
        timings.append(1000.0 * (time.perf_counter() - started))
    return {
        "repeats": repeats,
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "communication": payload,
    }


def model_footprint(simulator, checkpoint_path):
    per_server = {}
    for server_id, server in simulator.servers.items():
        model = server.agent.agent.TrainNet.model
        per_server[server_id] = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
    return {
        "per_server_parameters": per_server,
        "mean_parameters_per_server": float(np.mean(list(per_server.values()))),
        "total_parameters_all_servers": int(sum(per_server.values())),
        "selected_checkpoint_bytes": int(Path(checkpoint_path).stat().st_size),
    }


def expected_complete(path, users, episodes):
    if not Path(path).exists():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
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
        return {"label": label, "seed": seed, "users": users, "skipped": True}

    source_summary = read_json(source / "summary.json")
    if not (
        source_summary.get("status") == "complete"
        and source_summary.get("eligible_for_comparison") is True
        and source_summary.get("convergence", {}).get("reached")
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
    checkpoint_path = source / "selected_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path,
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
        cache_benchmark = benchmark_cache_decision(simulator, cache_repeats)
        footprint = model_footprint(simulator, checkpoint_path)

        eval_args = SimpleNamespace(
            eval_episodes=episodes,
            eval_seed_offset=int(source_arguments["eval_seed_offset"]),
            seed=seed,
            eval_bank_scope="infrastructure",
            eval_dag_families=list(FAMILIES),
            label=label,
            quiet=True,
        )
        episodes_path = target / "episodes.csv"
        with episodes_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=EPISODE_FIELDS)
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
    total_inference_calls = sum(int(row["policy_inference_calls"]) for row in rows)
    total_inference_time = sum(
        float(row["policy_inference_wall_time_sec"]) for row in rows
    )
    total_tasks = sum(int(row["real_task_count"]) for row in rows)
    total_wall_time = sum(float(row["episode_wall_time_sec"]) for row in rows)
    family_counts = {
        family: sum(
            scenario.get("workflow_family") == family
            for scenario in scenarios
        )
        for family in FAMILIES
    }
    tasks_exact_once = all(
        int(row["real_task_count"]) == int(row["completed_task_count"])
        and int(row["all_tasks_executed_once"]) == 1
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
        "tasks_exact_once": tasks_exact_once,
        "workflow_family_counts": family_counts,
        "deadline_projection": {
            "mode": "cyclic_checkpoint_deadlines",
            "used_future_target_workload": False,
        },
        "eval": summarize_rows(rows),
        "policy_inference": {
            "calls": total_inference_calls,
            "wall_time_sec": total_inference_time,
            "milliseconds_per_task": (
                1000.0 * total_inference_time / total_inference_calls
                if total_inference_calls
                else 0.0
            ),
        },
        "throughput": {
            "tasks": total_tasks,
            "evaluation_wall_time_sec": total_wall_time,
            "tasks_per_wall_second": (
                total_tasks / total_wall_time if total_wall_time else 0.0
            ),
        },
        "native_cache_decision_benchmark": cache_benchmark,
        "model_footprint": footprint,
    }
    if not tasks_exact_once:
        raise RuntimeError(f"Task exact-once audit failed: {target}")
    write_json(summary_path, summary)
    return {"label": label, "seed": seed, "users": users, "skipped": False}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def reference_reproduction_audit(
    source_suite,
    output_dir,
    labels,
    seeds,
    episodes,
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
                    source_run(source_suite, label, seed) / "episodes.csv"
                )
                if row["phase"] == "eval"
            ][:episodes]
            scaled = read_rows(
                target_run(output_dir, reference_users, label, seed)
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
            row["source_rows"] == episodes
            and row["scaled_rows"] == episodes
            and not any(row["mismatch_counts"].values())
            for row in combinations
        ),
        "combinations": combinations,
    }


def aggregate(source_suite, output_dir, user_counts, labels, seeds, episodes):
    results = {}
    all_paired = True
    for users in user_counts:
        per_seed = []
        for seed in seeds:
            rows_by_label = {}
            summaries = {}
            for label in labels:
                run = target_run(output_dir, users, label, seed)
                rows_by_label[label] = read_rows(run / "episodes.csv")
                summaries[label] = read_json(run / "summary.json")
            fingerprints = {
                label: [row["scenario_fingerprint"] for row in rows]
                for label, rows in rows_by_label.items()
            }
            paired = len({tuple(values) for values in fingerprints.values()}) == 1
            all_paired = all_paired and paired
            entry = {"seed": seed, "scenario_paired": paired, "methods": {}}
            for label in labels:
                rows = rows_by_label[label]
                summary = summaries[label]
                entry["methods"][label] = {
                    "mean_finish_time": float(
                        np.mean([float(row["average_finish_time"]) for row in rows])
                    ),
                    "mean_p95_finish_time": float(
                        np.mean([float(row["p95_finish_time"]) for row in rows])
                    ),
                    "inference_ms_per_task": summary["policy_inference"][
                        "milliseconds_per_task"
                    ],
                    "tasks_per_wall_second": summary["throughput"][
                        "tasks_per_wall_second"
                    ],
                    "cache_decision_mean_ms": summary[
                        "native_cache_decision_benchmark"
                    ]["mean_ms"],
                    "cache_decision_p95_ms": summary[
                        "native_cache_decision_benchmark"
                    ]["p95_ms"],
                    "coordination_bytes": summary[
                        "native_cache_decision_benchmark"
                    ]["communication"]["total_bytes"],
                    "parameters": summary["model_footprint"][
                        "total_parameters_all_servers"
                    ],
                    "checkpoint_bytes": summary["model_footprint"][
                        "selected_checkpoint_bytes"
                    ],
                }
            per_seed.append(entry)
        method_means = {
            label: {
                metric: float(
                    np.mean([row["methods"][label][metric] for row in per_seed])
                )
                for metric in per_seed[0]["methods"][label]
            }
            for label in labels
        }
        comparisons = {}
        for reference in labels:
            if reference == "lean_our":
                continue
            comparisons[f"our_vs_{reference}"] = paired_superiority(
                [row["methods"][reference]["mean_finish_time"] for row in per_seed],
                [row["methods"]["lean_our"]["mean_finish_time"] for row in per_seed],
                formal=True,
            )
        results[str(users)] = {
            "per_seed": per_seed,
            "method_means": method_means,
            "comparisons": comparisons,
        }
    reproduction = reference_reproduction_audit(
        source_suite,
        output_dir,
        labels,
        seeds,
        episodes,
    )
    summary = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "labels": labels,
        "seeds": seeds,
        "user_counts": user_counts,
        "episodes_per_seed": episodes,
        "all_methods_scenario_paired": all_paired,
        "model_frozen": True,
        "cache_frozen": True,
        "target_retraining": False,
        "reference_reproduction_audit": reproduction,
        "results": results,
    }
    if not all_paired or not reproduction["all_exact"]:
        raise RuntimeError("Scaling pairing or 20-user reproduction audit failed")
    write_json(Path(output_dir) / "user_scaling_summary.json", summary)
    write_scaling_report(Path(output_dir) / "USER_SCALING_REPORT_ZH.md", summary)
    return summary


def write_scaling_report(path, summary):
    display = {
        "daoc_paper": "DAOC-paper",
        "centralized_greedy_daoc": "Centralized-Greedy-DQN",
        "lean_our": "OUR",
    }
    lines = [
        "# Pegasus冻结模型用户规模与部署开销",
        "",
        "- 网络与缓存checkpoint冻结，目标用户规模不重新训练。",
        f"- Seeds：`{summary['seeds']}`；每seed每规模"
        f"`{summary['episodes_per_seed']}`个配对场景。",
        f"- 方法间场景配对：`{summary['all_methods_scenario_paired']}`。",
        "- 20用户点逐行复现冻结主实验前缀："
        f"`{summary['reference_reproduction_audit']['all_exact']}`。",
        "",
        "| Users | 方法 | Mean | P95 | 推理ms/task | tasks/s | "
        "缓存决策ms | 协调字节/次 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for users in summary["user_counts"]:
        result = summary["results"][str(users)]
        for label in summary["labels"]:
            values = result["method_means"][label]
            lines.append(
                f"| {users} | {display.get(label, label)} | "
                f"{values['mean_finish_time']:.6f} | "
                f"{values['mean_p95_finish_time']:.6f} | "
                f"{values['inference_ms_per_task']:.4f} | "
                f"{values['tasks_per_wall_second']:.2f} | "
                f"{values['cache_decision_mean_ms']:.4f} | "
                f"{values['coordination_bytes']:.1f} |"
            )
    lines.extend(
        [
            "",
            "协调字节采用紧凑JSON序列化的实际历史摘要载荷；静态拓扑、"
            "服务大小和模型参数视为预部署信息，不计入每次协调。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_label = "daoc_paper"
    canonical = {
        (users, seed): canonical_deployment(
            args.source_suite_dir,
            reference_label,
            seed,
            users,
            args.output_dir,
        )
        for users in args.user_counts
        for seed in args.seeds
    }
    specs = [
        (
            args.source_suite_dir,
            args.output_dir,
            canonical[(users, seed)],
            label,
            seed,
            users,
            args.episodes,
            args.cache_benchmark_repeats,
            args.resume,
        )
        for users in args.user_counts
        for seed in args.seeds
        for label in args.labels
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_one, spec) for spec in specs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(
                f"[{index}/{len(futures)}] users={result['users']} "
                f"{result['label']} seed={result['seed']} "
                f"skipped={result['skipped']}",
                flush=True,
            )
    aggregate(
        args.source_suite_dir,
        args.output_dir,
        args.user_counts,
        args.labels,
        args.seeds,
        args.episodes,
    )


if __name__ == "__main__":
    main()
