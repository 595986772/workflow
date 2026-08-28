#!/usr/bin/env python3
"""Build the full-size five-family Pegasus P-Scale dataset."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from build_pegasus_cross_dataset import (
    LICENSE_FILES,
    RAW_BASE_URL,
    SOURCE_COMMIT,
    SOURCE_FILES,
    SOURCE_REPOSITORY,
    download,
    encoded_service_value,
    local_name,
    parse_dax,
    service_id,
    sha256_bytes,
    sha256_file,
)


SERVICE_COUNT = 10
EXPECTED_PROGRAM_TYPES = 39
MAX_REAL_TASKS = 30
TASK_LIMIT_INCLUDING_DUMMY = 31
TYPE_MAPPING_VERSION = "sha256_namespace_name_mod10_v1"
NORMALIZATION_VERSION = "pegasus_global_log_minmax_v1"


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "datasets" / "pegasus_pscale",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def scale_log(value, lower, upper, minimum=0.05):
    transformed = math.log1p(max(float(value), 0.0))
    if upper <= lower:
        return 1.0
    unit = min(max((transformed - lower) / (upper - lower), 0.0), 1.0)
    return minimum + (1.0 - minimum) * unit


def source_job_type(job):
    return f"{job['namespace']}:{job['name']}"


def collect_source_data(raw_dir):
    parsed = {}
    runtimes = []
    positive_data_sizes = []
    program_types = set()
    for family, source_path in SOURCE_FILES.items():
        graph, jobs = parse_dax(raw_dir / local_name(source_path), family)
        parsed[family] = (graph, jobs)
        runtimes.extend(job["runtime"] for job in jobs.values())
        program_types.update(source_job_type(job) for job in jobs.values())
        positive_data_sizes.extend(
            attributes["data_bytes"]
            for _, _, attributes in graph.edges(data=True)
            if attributes["data_bytes"] > 0
        )
    if len(program_types) != EXPECTED_PROGRAM_TYPES:
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_PROGRAM_TYPES} Pegasus program types, "
            f"found {len(program_types)}"
        )
    runtime_logs = np.log1p(np.asarray(runtimes, dtype=float))
    data_logs = np.log1p(np.asarray(positive_data_sizes, dtype=float))
    return parsed, {
        "runtime_log_bounds": (
            float(runtime_logs.min()),
            float(runtime_logs.max()),
        ),
        "data_log_bounds": (
            float(data_logs.min()),
            float(data_logs.max()),
        ),
        "fallback_data_bytes": float(np.median(positive_data_sizes)),
        "program_types": tuple(sorted(program_types)),
    }


def convert_full_graph(family, graph, jobs, normalization):
    topological = list(nx.lexicographical_topological_sort(graph))
    if not 24 <= len(topological) <= MAX_REAL_TASKS:
        raise RuntimeError(
            f"{family} must contain 24-{MAX_REAL_TASKS} real tasks"
        )
    node_mapping = {
        source_id: str(position + 1)
        for position, source_id in enumerate(topological)
    }
    output = nx.DiGraph(
        source="WorkflowSim/Pegasus DAX",
        source_family=family,
        source_commit=SOURCE_COMMIT,
        extraction="complete_workflow_v1",
        type_mapping=TYPE_MAPPING_VERSION,
        normalization=NORMALIZATION_VERSION,
    )
    output.add_node("0", service=0, cpucycle=0)
    runtime_lower, runtime_upper = normalization["runtime_log_bounds"]
    data_lower, data_upper = normalization["data_log_bounds"]
    fallback_data = normalization["fallback_data_bytes"]

    for source_id in topological:
        job = jobs[source_id]
        mapped_service = service_id(job["namespace"], job["name"])
        output.add_node(
            node_mapping[source_id],
            service=encoded_service_value(mapped_service),
            cpucycle=scale_log(
                job["runtime"],
                runtime_lower,
                runtime_upper,
            ),
            source_job_id=source_id,
            source_job_type=source_job_type(job),
        )

    for source, target, attributes in graph.edges(data=True):
        raw_bytes = attributes["data_bytes"]
        if raw_bytes <= 0:
            raw_bytes = fallback_data
        output.add_edge(
            node_mapping[source],
            node_mapping[target],
            datalength=scale_log(
                raw_bytes,
                data_lower,
                data_upper,
                minimum=0.01,
            ),
        )
    real_roots = [
        node_id
        for node_id in output
        if node_id != "0" and output.in_degree(node_id) == 0
    ]
    for root_id in real_roots:
        output.add_edge("0", root_id, datalength=0.0)

    if output.number_of_nodes() > TASK_LIMIT_INCLUDING_DUMMY:
        raise RuntimeError(
            f"{family} exceeds the {TASK_LIMIT_INCLUDING_DUMMY}-node limit"
        )
    if not nx.is_directed_acyclic_graph(output):
        raise RuntimeError(f"Converted {family} workflow is not a DAG")
    return output


def build_dataset(raw_dir):
    parsed, normalization = collect_source_data(raw_dir)
    dataset = {}
    graph_manifest = []
    task_service_counts = Counter()
    type_mapping = {
        program_type: service_id(*program_type.split(":", 1))
        for program_type in normalization["program_types"]
    }
    for family in SOURCE_FILES:
        graph, jobs = parsed[family]
        converted = convert_full_graph(
            family,
            graph,
            jobs,
            normalization,
        )
        key = f"pegasus_full_{family.lower()}"
        dataset[key] = nx.node_link_data(converted)
        services = [
            int(SERVICE_COUNT * (float(attributes["service"]) - 1)) + 1
            for node_id, attributes in converted.nodes(data=True)
            if node_id != "0"
        ]
        task_service_counts.update(services)
        graph_manifest.append(
            {
                "key": key,
                "family": family,
                "real_tasks": converted.number_of_nodes() - 1,
                "nodes_including_dummy": converted.number_of_nodes(),
                "edges_including_dummy": converted.number_of_edges(),
                "depth": nx.dag_longest_path_length(converted),
                "root_count": sum(
                    1
                    for node_id in converted
                    if node_id != "0"
                    and converted.in_degree(node_id) == 1
                    and converted.has_edge("0", node_id)
                ),
                "exit_count": sum(
                    1
                    for node_id in converted
                    if node_id != "0"
                    and converted.out_degree(node_id) == 0
                ),
                "service_ids": sorted(set(services)),
            }
        )
    if len(dataset) != len(SOURCE_FILES):
        raise RuntimeError("Expected one complete graph per Pegasus family")
    if set(task_service_counts) != set(range(1, SERVICE_COUNT + 1)):
        raise RuntimeError("P-Scale dataset must cover all ten services")
    metadata = {
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "runtime_log_bounds": normalization["runtime_log_bounds"],
            "data_log_bounds": normalization["data_log_bounds"],
            "fallback_data_bytes": normalization["fallback_data_bytes"],
        },
        "type_mapping": {
            "version": TYPE_MAPPING_VERSION,
            "program_type_count": len(type_mapping),
            "program_type_to_service": dict(sorted(type_mapping.items())),
        },
        "task_service_counts": dict(sorted(task_service_counts.items())),
        "graphs": graph_manifest,
    }
    return dataset, metadata


def render_readme(dataset_sha256, metadata):
    task_counts = [row["real_tasks"] for row in metadata["graphs"]]
    return f"""# Pegasus P-Scale DAG Dataset

This controlled benchmark keeps each complete WorkflowSim/Pegasus workflow
and tests larger DAGs on the fixed DAOC MEC infrastructure. It is not an
unbiased MEC trace.

- Source: {SOURCE_REPOSITORY}
- Pinned source commit: `{SOURCE_COMMIT}`
- Families: Montage, CyberShake, Epigenomics, Inspiral, SIPHT
- Graphs: one complete workflow per family
- Real tasks: `{task_counts}`; one dummy source is added to each graph
- Task limit including dummy source: `{TASK_LIMIT_INCLUDING_DUMMY}`
- Program types: `{EXPECTED_PROGRAM_TYPES}` mapped deterministically to 10 services
- Type mapping: `{TYPE_MAPPING_VERSION}`
- CPU and edge data: `{NORMALIZATION_VERSION}`
- Dataset SHA-256: `{dataset_sha256}`

Regenerate with:

```bash
/opt/anaconda3/envs/dl/bin/python build_pegasus_pscale_dataset.py
```
"""


def main():
    args = parse_args()
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_records = {}
    for family, source_path in SOURCE_FILES.items():
        target = raw_dir / local_name(source_path)
        download(RAW_BASE_URL + source_path, target, refresh=args.refresh)
        source_records[family] = {
            "upstream_path": source_path,
            "local_path": str(target.resolve()),
            "sha256": sha256_file(target),
        }
    for source_path in LICENSE_FILES:
        target = raw_dir / local_name(source_path)
        download(RAW_BASE_URL + source_path, target, refresh=args.refresh)

    dataset, generation = build_dataset(raw_dir)
    dataset_path = args.output_dir / "dag_pegasus5_full31.json"
    payload = json.dumps(
        dataset,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ).encode("utf-8")
    dataset_path.write_bytes(payload)
    dataset_sha256 = sha256_bytes(payload)
    manifest = {
        "status": "complete",
        "claim_scope": "controlled_pegasus_pscale_workflow_benchmark",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_files": source_records,
        "generator": {
            "families": list(SOURCE_FILES),
            "complete_workflows": True,
            "real_task_range": [24, MAX_REAL_TASKS],
            "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
            "service_count": SERVICE_COUNT,
            "type_mapping": TYPE_MAPPING_VERSION,
            "normalization": NORMALIZATION_VERSION,
        },
        "generation": generation,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_sha256,
            "graph_count": len(dataset),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        render_readme(dataset_sha256, generation),
        encoding="utf-8",
    )
    print(dataset_path)
    print(dataset_sha256)


if __name__ == "__main__":
    main()
