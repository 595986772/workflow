#!/usr/bin/env python3
"""Build a DAOC-compatible critical-path stress set from Alibaba 2018.

The source trace supplies task topology and task-level execution metadata.
DAOC-specific service labels and edge payloads are generated deterministically
and are intentionally biased toward critical-path stress. The resulting data
must therefore be treated as a mechanism/development benchmark, not as an
unbiased sample of the Alibaba trace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import math
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path


GENERATOR_VERSION = "1.0.0"
TASK_NAME_PATTERN = re.compile(r"^[A-Za-z]+([0-9]+(?:_[0-9]+)*)$")
DEFAULT_SEED = 20260730
DEFAULT_COUNT = 100
DEFAULT_SERVICES = 10


@dataclass(frozen=True)
class TaskRecord:
    source_name: str
    task_id: int
    predecessors: tuple[int, ...]
    instance_num: int
    start_time: int
    end_time: int
    plan_cpu: float
    plan_mem: float
    duration: int
    weight: float


@dataclass(frozen=True)
class Candidate:
    job_name: str
    tasks: dict[int, TaskRecord]
    predecessors: dict[int, tuple[int, ...]]
    successors: dict[int, tuple[int, ...]]
    topological_order: tuple[int, ...]
    critical_path: tuple[int, ...]
    second_path: tuple[int, ...]
    task_count: int
    edge_count: int
    structural_depth: int
    fork_count: int
    join_count: int
    critical_path_weight: float
    second_path_weight: float
    critical_path_work_fraction: float
    critical_path_node_fraction: float
    critical_path_dominance_gap: float
    max_critical_node_share: float
    selection_score: float


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[1]
    project_dir = repo_dir.parent
    dataset_dir = repo_dir / "datasets" / "alibaba_cp100"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=project_dir / "batch_task.tar.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_dir / "dag_alibaba_cp100.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=dataset_dir / "selection_manifest.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=dataset_dir / "selection_summary.json",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-services", type=int, default=DEFAULT_SERVICES)
    parser.add_argument("--progress-every", type=int, default=2_000_000)
    return parser.parse_args()


def parse_task_name(name: str) -> tuple[int, tuple[int, ...]] | None:
    match = TASK_NAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    numbers = tuple(int(value) for value in match.group(1).split("_"))
    if not numbers or numbers[0] <= 0:
        return None
    task_id = numbers[0]
    predecessors = numbers[1:]
    if task_id in predecessors or any(value <= 0 for value in predecessors):
        return None
    if len(set(predecessors)) != len(predecessors):
        return None
    return task_id, predecessors


def deterministic_topological_order(
    predecessors: dict[int, tuple[int, ...]],
    successors: dict[int, tuple[int, ...]],
) -> tuple[int, ...] | None:
    indegree = {
        task_id: len(task_predecessors)
        for task_id, task_predecessors in predecessors.items()
    }
    ready = [task_id for task_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        task_id = heapq.heappop(ready)
        order.append(task_id)
        for successor in successors[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if len(order) != len(predecessors):
        return None
    return tuple(order)


def root_to_leaf_paths(
    successors: dict[int, tuple[int, ...]],
    roots: tuple[int, ...],
    weights: dict[int, float],
) -> list[tuple[float, tuple[int, ...]]]:
    paths: list[tuple[float, tuple[int, ...]]] = []

    def visit(task_id: int, path: tuple[int, ...], score: float) -> None:
        next_path = path + (task_id,)
        next_score = score + weights[task_id]
        if not successors[task_id]:
            paths.append((next_score, next_path))
            return
        for successor in successors[task_id]:
            visit(successor, next_path, next_score)

    for root in roots:
        visit(root, (), 0.0)
    paths.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return paths


def evaluate_job(
    job_name: str,
    rows: list[list[str]],
    counters: dict[str, int],
) -> Candidate | None:
    task_count = len(rows)
    if task_count not in (7, 8, 9):
        return None
    counters["jobs_7_9_tasks"] += 1

    if any(row[4] != "Terminated" for row in rows):
        return None
    counters["jobs_7_9_all_terminated"] += 1

    tasks: dict[int, TaskRecord] = {}
    for row in rows:
        parsed_name = parse_task_name(row[0])
        if parsed_name is None:
            return None
        task_id, task_predecessors = parsed_name
        if task_id in tasks:
            return None
        try:
            instance_num = int(row[1])
            start_time = int(row[5])
            end_time = int(row[6])
            plan_cpu = float(row[7])
            plan_mem = float(row[8])
        except ValueError:
            return None
        duration = end_time - start_time
        if (
            instance_num <= 0
            or start_time < 0
            or duration <= 0
            or not math.isfinite(plan_cpu)
            or not math.isfinite(plan_mem)
            or plan_cpu <= 0
            or plan_mem < 0
        ):
            return None
        weight = math.log1p(duration * plan_cpu)
        tasks[task_id] = TaskRecord(
            source_name=row[0],
            task_id=task_id,
            predecessors=task_predecessors,
            instance_num=instance_num,
            start_time=start_time,
            end_time=end_time,
            plan_cpu=plan_cpu,
            plan_mem=plan_mem,
            duration=duration,
            weight=weight,
        )
    counters["jobs_parseable"] += 1

    task_ids = set(tasks)
    if any(
        predecessor not in task_ids
        for task in tasks.values()
        for predecessor in task.predecessors
    ):
        return None
    counters["jobs_complete_dependencies"] += 1

    predecessors = {
        task_id: tuple(sorted(task.predecessors))
        for task_id, task in tasks.items()
    }
    successor_sets = {task_id: set() for task_id in tasks}
    for task_id, task_predecessors in predecessors.items():
        for predecessor in task_predecessors:
            successor_sets[predecessor].add(task_id)
    successors = {
        task_id: tuple(sorted(values))
        for task_id, values in successor_sets.items()
    }
    topological_order = deterministic_topological_order(
        predecessors,
        successors,
    )
    if topological_order is None:
        return None
    counters["jobs_acyclic"] += 1

    roots = tuple(
        task_id for task_id in topological_order if not predecessors[task_id]
    )
    fork_count = sum(len(values) >= 2 for values in successors.values())
    join_count = sum(len(values) >= 2 for values in predecessors.values())
    edge_count = sum(len(values) for values in successors.values())
    if not roots or fork_count == 0 or join_count == 0:
        return None
    counters["jobs_fork_join"] += 1

    weights = {task_id: task.weight for task_id, task in tasks.items()}
    paths = root_to_leaf_paths(successors, roots, weights)
    if len(paths) < 2:
        return None
    critical_path_weight, critical_path = paths[0]
    second_path_weight, second_path = paths[1]
    structural_depth = max(len(path) for _, path in paths)
    critical_path_work_fraction = critical_path_weight / sum(weights.values())
    critical_path_node_fraction = len(critical_path) / task_count
    critical_path_dominance_gap = (
        critical_path_weight - second_path_weight
    ) / critical_path_weight
    max_critical_node_share = max(
        weights[task_id] for task_id in critical_path
    ) / critical_path_weight

    if (
        structural_depth < 4
        or len(critical_path) < 4
        or len(critical_path) >= task_count
        or critical_path_work_fraction < 0.50
        or critical_path_dominance_gap < 0.08
        or max_critical_node_share > 0.60
    ):
        return None
    counters["jobs_cp_qualified"] += 1
    counters[f"jobs_cp_qualified_{task_count}_tasks"] += 1

    gap_component = min(critical_path_dominance_gap / 0.50, 1.0)
    complexity_component = min((fork_count + join_count) / 4.0, 1.0)
    selection_score = (
        0.40 * critical_path_work_fraction
        + 0.30 * gap_component
        + 0.20 * critical_path_node_fraction
        + 0.10 * complexity_component
    )

    return Candidate(
        job_name=job_name,
        tasks=tasks,
        predecessors=predecessors,
        successors=successors,
        topological_order=topological_order,
        critical_path=critical_path,
        second_path=second_path,
        task_count=task_count,
        edge_count=edge_count,
        structural_depth=structural_depth,
        fork_count=fork_count,
        join_count=join_count,
        critical_path_weight=critical_path_weight,
        second_path_weight=second_path_weight,
        critical_path_work_fraction=critical_path_work_fraction,
        critical_path_node_fraction=critical_path_node_fraction,
        critical_path_dominance_gap=critical_path_dominance_gap,
        max_critical_node_share=max_critical_node_share,
        selection_score=selection_score,
    )


def job_number(job_name: str) -> int:
    try:
        return int(job_name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        digest = hashlib.sha256(job_name.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")


def source_job_number(job_name: str) -> int:
    if not job_name.startswith("j_"):
        raise ValueError(f"unexpected Alibaba job name: {job_name}")
    try:
        value = int(job_name[2:])
    except ValueError as error:
        raise ValueError(
            f"unexpected Alibaba job name: {job_name}"
        ) from error
    if value < 0:
        raise ValueError(f"negative Alibaba job id: {job_name}")
    return value


def selection_quotas(count: int) -> dict[int, int]:
    if count < 3:
        raise ValueError("count must be at least 3")
    quotient, remainder = divmod(count, 3)
    quotas = {7: quotient, 8: quotient, 9: quotient}
    for task_count in (9, 8, 7)[:remainder]:
        quotas[task_count] += 1
    return quotas


def keep_candidate(
    heaps: dict[int, list[tuple[float, int, Candidate]]],
    candidate: Candidate,
    quota: int,
) -> None:
    entry = (
        candidate.selection_score,
        -job_number(candidate.job_name),
        candidate,
    )
    heap = heaps[candidate.task_count]
    if len(heap) < quota:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def scan_archive(
    archive_path: Path,
    count: int,
    progress_every: int,
) -> tuple[list[Candidate], dict[str, int]]:
    quotas = selection_quotas(count)
    heaps: dict[int, list[tuple[float, int, Candidate]]] = {
        task_count: [] for task_count in quotas
    }
    counters = {
        "task_rows": 0,
        "jobs": 0,
        "malformed_rows": 0,
        "job_order_regressions": 0,
        "duplicate_job_groups": 0,
        "jobs_7_9_tasks": 0,
        "jobs_7_9_all_terminated": 0,
        "jobs_parseable": 0,
        "jobs_complete_dependencies": 0,
        "jobs_acyclic": 0,
        "jobs_fork_join": 0,
        "jobs_cp_qualified": 0,
        "jobs_cp_qualified_7_tasks": 0,
        "jobs_cp_qualified_8_tasks": 0,
        "jobs_cp_qualified_9_tasks": 0,
    }

    started = time.monotonic()
    current_job: str | None = None
    current_rows: list[list[str]] = []
    previous_job_number: int | None = None
    seen_job_numbers = bytearray()

    def finish_current_job() -> None:
        nonlocal previous_job_number
        if current_job is None:
            return
        counters["jobs"] += 1
        current_number = source_job_number(current_job)
        if (
            previous_job_number is not None
            and current_number < previous_job_number
        ):
            counters["job_order_regressions"] += 1
        previous_job_number = current_number
        if current_number >= len(seen_job_numbers):
            seen_job_numbers.extend(
                b"\x00" * (current_number + 1 - len(seen_job_numbers))
            )
        if seen_job_numbers[current_number]:
            counters["duplicate_job_groups"] += 1
        else:
            seen_job_numbers[current_number] = 1
        candidate = evaluate_job(current_job, current_rows, counters)
        if candidate is not None:
            keep_candidate(
                heaps,
                candidate,
                quotas[candidate.task_count],
            )

    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.getmember("batch_task.csv")
        raw_stream = archive.extractfile(member)
        if raw_stream is None:
            raise RuntimeError("batch_task.csv is missing from the archive")
        with io.TextIOWrapper(
            raw_stream,
            encoding="utf-8",
            newline="",
        ) as text_stream:
            reader = csv.reader(text_stream)
            for row in reader:
                counters["task_rows"] += 1
                if len(row) != 9:
                    counters["malformed_rows"] += 1
                    continue
                job_name = row[2]
                if current_job is None:
                    current_job = job_name
                elif job_name != current_job:
                    finish_current_job()
                    current_job = job_name
                    current_rows = []
                current_rows.append(row)
                if (
                    progress_every > 0
                    and counters["task_rows"] % progress_every == 0
                ):
                    elapsed = time.monotonic() - started
                    print(
                        f"scanned {counters['task_rows']:,} rows, "
                        f"{counters['jobs']:,} jobs in {elapsed:.1f}s",
                        flush=True,
                    )
    finish_current_job()

    if counters["malformed_rows"]:
        raise RuntimeError(
            f"found {counters['malformed_rows']} malformed CSV rows"
        )
    if counters["duplicate_job_groups"]:
        raise RuntimeError(
            "one or more Alibaba jobs occur in non-contiguous groups"
        )

    selected: list[Candidate] = []
    for task_count, quota in quotas.items():
        heap = heaps[task_count]
        if len(heap) != quota:
            raise RuntimeError(
                f"needed {quota} candidates with {task_count} tasks, "
                f"but found {len(heap)}"
            )
        selected.extend(entry[2] for entry in heap)
    selected.sort(
        key=lambda candidate: (
            candidate.task_count,
            -candidate.selection_score,
            job_number(candidate.job_name),
        )
    )
    return selected, counters


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stable_unit(seed: int, *parts: object) -> float:
    payload = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def encoded_service(service_id: int, num_services: int) -> float:
    if not 1 <= service_id <= num_services:
        raise ValueError("service_id is outside the configured range")
    return 1.0 + (service_id - 0.5) / num_services


def graph_from_candidate(
    candidate: Candidate,
    low_weight: float,
    high_weight: float,
    seed: int,
    num_services: int,
) -> tuple[dict[str, object], dict[str, object]]:
    id_map = {
        source_id: str(index + 1)
        for index, source_id in enumerate(candidate.topological_order)
    }
    critical_set = set(candidate.critical_path)
    critical_edges = set(zip(
        candidate.critical_path,
        candidate.critical_path[1:],
    ))
    hot_service_pattern = (1, 2, 1, 3, 2, 1, 3, 1, 2)
    critical_position = {
        task_id: index
        for index, task_id in enumerate(candidate.critical_path)
    }

    nodes: list[dict[str, object]] = [
        {"service": 0, "cpucycle": 0, "id": "0"}
    ]
    assigned_services: dict[int, int] = {}
    for source_id in candidate.topological_order:
        task = candidate.tasks[source_id]
        if high_weight == low_weight:
            normalized_weight = 1.0
        else:
            normalized_weight = (
                task.weight - low_weight
            ) / (high_weight - low_weight)
        normalized_weight = min(max(normalized_weight, 0.0), 1.0)
        cpucycle = 0.05 + 0.95 * normalized_weight

        if source_id in critical_set:
            service_id = hot_service_pattern[
                critical_position[source_id] % len(hot_service_pattern)
            ]
        else:
            service_id = 4 + min(
                int(
                    stable_unit(
                        seed,
                        candidate.job_name,
                        source_id,
                        "service",
                    )
                    * max(num_services - 3, 1)
                ),
                max(num_services - 4, 0),
            )
        assigned_services[source_id] = service_id
        nodes.append(
            {
                "service": round(
                    encoded_service(service_id, num_services),
                    12,
                ),
                "cpucycle": round(cpucycle, 12),
                "id": id_map[source_id],
            }
        )

    links: list[dict[str, object]] = []
    roots = [
        task_id
        for task_id in candidate.topological_order
        if not candidate.predecessors[task_id]
    ]
    for task_id in roots:
        unit = stable_unit(
            seed,
            candidate.job_name,
            task_id,
            "input",
        )
        links.append(
            {
                "datalength": round(0.35 + 0.30 * unit, 12),
                "source": "0",
                "target": id_map[task_id],
            }
        )

    for source_id in candidate.topological_order:
        for target_id in candidate.successors[source_id]:
            unit = stable_unit(
                seed,
                candidate.job_name,
                source_id,
                target_id,
                "edge",
            )
            if (source_id, target_id) in critical_edges:
                data_length = 0.75 + 0.25 * unit
            elif source_id in critical_set or target_id in critical_set:
                data_length = 0.45 + 0.25 * unit
            else:
                data_length = 0.15 + 0.35 * unit
            links.append(
                {
                    "datalength": round(data_length, 12),
                    "source": id_map[source_id],
                    "target": id_map[target_id],
                }
            )

    graph_data: dict[str, object] = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links,
    }
    manifest = {
        "dataset_key": candidate.job_name,
        "source_job": candidate.job_name,
        "source_task_names": {
            id_map[task_id]: candidate.tasks[task_id].source_name
            for task_id in candidate.topological_order
        },
        "source_to_daoc_task_id": {
            str(task_id): id_map[task_id]
            for task_id in candidate.topological_order
        },
        "task_count": candidate.task_count,
        "edge_count_without_dummy": candidate.edge_count,
        "structural_depth": candidate.structural_depth,
        "fork_count": candidate.fork_count,
        "join_count": candidate.join_count,
        "critical_path_source_ids": list(candidate.critical_path),
        "critical_path_daoc_ids": [
            id_map[task_id] for task_id in candidate.critical_path
        ],
        "second_path_source_ids": list(candidate.second_path),
        "critical_path_work_fraction": (
            candidate.critical_path_work_fraction
        ),
        "critical_path_node_fraction": (
            candidate.critical_path_node_fraction
        ),
        "critical_path_dominance_gap": (
            candidate.critical_path_dominance_gap
        ),
        "max_critical_node_share": (
            candidate.max_critical_node_share
        ),
        "selection_score": candidate.selection_score,
        "assigned_service_ids": {
            id_map[task_id]: assigned_services[task_id]
            for task_id in candidate.topological_order
        },
    }
    return graph_data, manifest


def validate_graphs(
    graphs: dict[str, dict[str, object]],
    expected_count: int,
    num_services: int,
) -> None:
    if len(graphs) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} graphs, found {len(graphs)}"
        )
    for graph_name, graph in graphs.items():
        if graph.get("directed") is not True:
            raise RuntimeError(f"{graph_name}: graph is not directed")
        if graph.get("multigraph") is not False:
            raise RuntimeError(f"{graph_name}: graph is a multigraph")
        nodes = graph.get("nodes")
        links = graph.get("links")
        if not isinstance(nodes, list) or not isinstance(links, list):
            raise RuntimeError(f"{graph_name}: invalid node-link schema")
        node_ids = {str(node["id"]) for node in nodes}
        real_nodes = [node for node in nodes if str(node["id"]) != "0"]
        if len(real_nodes) not in (7, 8, 9):
            raise RuntimeError(f"{graph_name}: invalid real task count")
        if len(nodes) > 10:
            raise RuntimeError(f"{graph_name}: exceeds DAOC I=10")
        if "0" not in node_ids or len(node_ids) != len(nodes):
            raise RuntimeError(f"{graph_name}: invalid or duplicate node ids")
        for node in real_nodes:
            service_value = float(node["service"])
            mapped_service = int(
                num_services * (service_value - 1.0)
            ) + 1
            if not 1 <= mapped_service <= num_services:
                raise RuntimeError(
                    f"{graph_name}: service mapping is out of range"
                )
            if not 0.0 < float(node["cpucycle"]) <= 1.0:
                raise RuntimeError(
                    f"{graph_name}: cpucycle is out of range"
                )
        indegree = {node_id: 0 for node_id in node_ids}
        successors = {node_id: [] for node_id in node_ids}
        for link in links:
            source = str(link["source"])
            target = str(link["target"])
            if source not in node_ids or target not in node_ids:
                raise RuntimeError(f"{graph_name}: edge endpoint is missing")
            data_length = float(link["datalength"])
            if not 0.0 < data_length <= 1.0:
                raise RuntimeError(
                    f"{graph_name}: datalength is out of range"
                )
            indegree[target] += 1
            successors[source].append(target)
        ready = [node_id for node_id, value in indegree.items() if value == 0]
        visited = 0
        while ready:
            node_id = ready.pop()
            visited += 1
            for successor in successors[node_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited != len(nodes):
            raise RuntimeError(f"{graph_name}: graph contains a cycle")


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("metric values must not be empty")
    return {
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def output_stress_metrics(
    graphs: dict[str, dict[str, object]],
    manifest: list[dict[str, object]],
    num_services: int,
) -> dict[str, object]:
    manifest_by_key = {
        str(item["dataset_key"]): item for item in manifest
    }
    path_matches = 0
    path_fractions: list[float] = []
    path_gaps: list[float] = []
    critical_edge_lengths: list[float] = []
    other_edge_lengths: list[float] = []
    service_counts = {
        service_id: 0 for service_id in range(1, num_services + 1)
    }

    for graph_name, graph in graphs.items():
        nodes = {
            str(node["id"]): node
            for node in graph["nodes"]
            if str(node["id"]) != "0"
        }
        successors = {node_id: [] for node_id in nodes}
        indegree = {node_id: 0 for node_id in nodes}
        edge_lengths: dict[tuple[str, str], float] = {}
        for link in graph["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source == "0":
                continue
            successors[source].append(target)
            indegree[target] += 1
            edge_lengths[(source, target)] = float(link["datalength"])
        roots = sorted(
            (
                node_id
                for node_id, degree in indegree.items()
                if degree == 0
            ),
            key=int,
        )
        node_weights = {
            node_id: float(node["cpucycle"])
            for node_id, node in nodes.items()
        }
        paths: list[tuple[float, tuple[str, ...]]] = []

        def visit(
            node_id: str,
            path: tuple[str, ...],
            score: float,
        ) -> None:
            next_path = path + (node_id,)
            next_score = score + node_weights[node_id]
            if not successors[node_id]:
                paths.append((next_score, next_path))
                return
            for successor in sorted(successors[node_id], key=int):
                visit(successor, next_path, next_score)

        for root in roots:
            visit(root, (), 0.0)
        paths.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_path = paths[0]
        second_score = paths[1][0]
        expected_path = tuple(
            str(value)
            for value in manifest_by_key[graph_name][
                "critical_path_daoc_ids"
            ]
        )
        path_matches += best_path == expected_path
        path_fractions.append(best_score / sum(node_weights.values()))
        path_gaps.append((best_score - second_score) / best_score)

        expected_edges = set(zip(expected_path, expected_path[1:]))
        for edge, data_length in edge_lengths.items():
            if edge in expected_edges:
                critical_edge_lengths.append(data_length)
            else:
                other_edge_lengths.append(data_length)

        for node in nodes.values():
            service_value = float(node["service"])
            service_id = int(
                num_services * (service_value - 1.0)
            ) + 1
            service_counts[service_id] += 1

    total_services = sum(service_counts.values())
    hot_services = sum(
        count
        for service_id, count in service_counts.items()
        if service_id <= 3
    )
    return {
        "manifest_path_is_output_node_weighted_longest": {
            "matched": path_matches,
            "total": len(graphs),
        },
        "output_node_weighted_critical_path_fraction": (
            metric_summary(path_fractions)
        ),
        "output_node_weighted_critical_path_gap": (
            metric_summary(path_gaps)
        ),
        "critical_edge_datalength": (
            metric_summary(critical_edge_lengths)
        ),
        "other_edge_datalength": metric_summary(other_edge_lengths),
        "mapped_service_counts": {
            str(service_id): count
            for service_id, count in service_counts.items()
        },
        "hot_service_fraction_services_1_to_3": (
            hot_services / total_services
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.num_services < 4:
        raise ValueError("num-services must be at least 4")
    archive_path = args.archive.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    selected, counters = scan_archive(
        archive_path,
        args.count,
        args.progress_every,
    )
    all_weights = [
        task.weight
        for candidate in selected
        for task in candidate.tasks.values()
    ]
    low_weight = quantile(all_weights, 0.05)
    high_weight = quantile(all_weights, 0.95)

    graphs: dict[str, dict[str, object]] = {}
    manifest: list[dict[str, object]] = []
    for candidate in selected:
        graph, graph_manifest = graph_from_candidate(
            candidate,
            low_weight,
            high_weight,
            args.seed,
            args.num_services,
        )
        graphs[candidate.job_name] = graph
        manifest.append(graph_manifest)
    validate_graphs(graphs, args.count, args.num_services)
    stress_metrics = output_stress_metrics(
        graphs,
        manifest,
        args.num_services,
    )

    summary = {
        "dataset_name": "Alibaba-CP100",
        "dataset_role": "mechanism_and_development_stress_test",
        "formal_unbiased_holdout": False,
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "source_archive": str(archive_path),
        "source_archive_sha256": sha256_file(archive_path),
        "selected_graphs": len(graphs),
        "task_count_quotas": selection_quotas(args.count),
        "scan_counters": counters,
        "selection_requirements": {
            "real_tasks": [7, 8, 9],
            "status": "all Terminated",
            "dependency_names": "letters + numeric id/dependency suffix",
            "complete_predecessors": True,
            "acyclic": True,
            "minimum_forks": 1,
            "minimum_joins": 1,
            "minimum_structural_depth": 4,
            "minimum_critical_path_nodes": 4,
            "critical_path_must_not_include_all_nodes": True,
            "minimum_critical_path_work_fraction": 0.50,
            "minimum_critical_path_dominance_gap": 0.08,
            "maximum_single_critical_node_share": 0.60,
        },
        "attribute_protocol": {
            "task_weight": "log1p((end_time-start_time)*plan_cpu)",
            "cpucycle": (
                "selected-task weight normalized by selected-set "
                "5th/95th percentiles to [0.05,1.0]"
            ),
            "services": (
                "critical-path tasks use deterministic hot services 1-3; "
                "off-path tasks use deterministic services 4-10"
            ),
            "edge_payloads": (
                "critical-path edges [0.75,1.0], incident edges "
                "[0.45,0.70], off-path edges [0.15,0.50]"
            ),
            "dummy_source": (
                "node 0 connects to every real source with input payload "
                "in [0.35,0.65]"
            ),
        },
        "cpucycle_weight_quantiles": {
            "q05": low_weight,
            "q95": high_weight,
        },
        "selected_source_metrics": {
            "critical_path_work_fraction": metric_summary(
                [
                    candidate.critical_path_work_fraction
                    for candidate in selected
                ]
            ),
            "critical_path_dominance_gap": metric_summary(
                [
                    candidate.critical_path_dominance_gap
                    for candidate in selected
                ]
            ),
            "critical_path_node_fraction": metric_summary(
                [
                    candidate.critical_path_node_fraction
                    for candidate in selected
                ]
            ),
            "selection_score": metric_summary(
                [
                    candidate.selection_score
                    for candidate in selected
                ]
            ),
        },
        "output_stress_metrics": stress_metrics,
    }

    for path in (args.output, args.manifest, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(graphs, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(graphs)} graphs to {args.output}")
    print(f"wrote manifest to {args.manifest}")
    print(f"wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
