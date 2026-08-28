#!/usr/bin/env python3
"""Build a DAOC-compatible cross-topology benchmark from Pegasus DAX files."""

import argparse
import hashlib
import json
import math
import random
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


SOURCE_REPOSITORY = "https://github.com/WorkflowSim/WorkflowSim-1.0"
SOURCE_COMMIT = "d3ea21afd8ce6479bd292d3bd7469045d7a36089"
SOURCE_FILES = {
    "Montage": "config/dax/Montage_25.xml",
    "CyberShake": "config/dax/CyberShake_30.xml",
    "Epigenomics": "config/dax/Epigenomics_24.xml",
    "Inspiral": "config/dax/Inspiral_30.xml",
    "Sipht": "config/dax/Sipht_30.xml",
}
LICENSE_FILES = (
    "license-workflowsim.txt",
    "license.txt",
)
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/WorkflowSim/WorkflowSim-1.0/"
    f"{SOURCE_COMMIT}/"
)
GRAPHS_PER_FAMILY = 20
REAL_TASKS_PER_GRAPH = 9
SERVICE_COUNT = 10
GENERATOR_SEED = 20260803


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "datasets/pegasus_cross_topology",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def download(url, path, refresh=False):
    path = Path(path)
    if path.exists() and not refresh:
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DAOC-h8v1-cross-dataset-builder"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def local_name(path):
    return Path(path).name


def parse_dax(path, family):
    root = ET.parse(path).getroot()
    jobs = {}
    graph = nx.DiGraph(family=family, source_file=Path(path).name)
    for element in root:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag != "job":
            continue
        job_id = element.attrib["id"]
        inputs = {}
        outputs = {}
        for use in element:
            if use.tag.rsplit("}", 1)[-1] != "uses":
                continue
            filename = use.attrib.get("file", "")
            try:
                size = max(float(use.attrib.get("size", 0)), 0.0)
            except ValueError:
                size = 0.0
            if use.attrib.get("link") == "output":
                outputs[filename] = size
            elif use.attrib.get("link") == "input":
                inputs[filename] = size
        runtime = max(float(element.attrib.get("runtime", 1.0)), 1e-9)
        jobs[job_id] = {
            "runtime": runtime,
            "namespace": element.attrib.get("namespace", family),
            "name": element.attrib.get("name", "unknown"),
            "inputs": inputs,
            "outputs": outputs,
        }
        graph.add_node(job_id)

    for child in root:
        if child.tag.rsplit("}", 1)[-1] != "child":
            continue
        child_id = child.attrib["ref"]
        for parent in child:
            if parent.tag.rsplit("}", 1)[-1] == "parent":
                parent_id = parent.attrib["ref"]
                shared_files = set(jobs[parent_id]["outputs"]) & set(
                    jobs[child_id]["inputs"]
                )
                data_bytes = sum(
                    max(
                        jobs[parent_id]["outputs"][filename],
                        jobs[child_id]["inputs"][filename],
                    )
                    for filename in shared_files
                )
                graph.add_edge(parent_id, child_id, data_bytes=data_bytes)

    if set(graph) != set(jobs):
        raise RuntimeError(f"DAX job mismatch: {path}")
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError(f"DAX is not acyclic: {path}")
    if not nx.is_weakly_connected(graph):
        raise RuntimeError(f"DAX is not weakly connected: {path}")
    return graph, jobs


def select_connected_subgraphs(graph, family):
    undirected = graph.to_undirected()
    node_ids = sorted(graph.nodes)
    selected_signatures = set()
    subgraphs = []
    attempts = 0
    while len(subgraphs) < GRAPHS_PER_FAMILY:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError(f"Could not sample enough {family} subgraphs")
        rng = random.Random(stable_seed(GENERATOR_SEED, family, attempts))
        selected = {rng.choice(node_ids)}
        while len(selected) < REAL_TASKS_PER_GRAPH:
            frontier = sorted(
                {
                    neighbor
                    for node_id in selected
                    for neighbor in undirected.neighbors(node_id)
                    if neighbor not in selected
                }
            )
            if not frontier:
                break
            selected.add(rng.choice(frontier))
        if len(selected) != REAL_TASKS_PER_GRAPH:
            continue
        signature = tuple(sorted(selected))
        if signature in selected_signatures:
            continue
        subgraph = graph.subgraph(selected).copy()
        if (
            not nx.is_weakly_connected(subgraph)
            or subgraph.number_of_edges() < REAL_TASKS_PER_GRAPH - 2
            or nx.dag_longest_path_length(subgraph) < 2
        ):
            continue
        selected_signatures.add(signature)
        subgraphs.append(subgraph)
    return subgraphs


def scale_log(value, lower, upper, minimum=0.05):
    transformed = math.log1p(max(float(value), 0.0))
    if upper <= lower:
        return 1.0
    unit = min(max((transformed - lower) / (upper - lower), 0.0), 1.0)
    return minimum + (1.0 - minimum) * unit


def service_id(namespace, name):
    payload = f"{namespace}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % SERVICE_COUNT + 1


def encoded_service_value(identifier):
    return 1.0 + (int(identifier) - 0.5) / SERVICE_COUNT


def convert_subgraph(
    family,
    index,
    subgraph,
    jobs,
    runtime_bounds,
    data_bounds,
):
    topological = list(nx.lexicographical_topological_sort(subgraph))
    node_mapping = {
        source_id: str(position + 1)
        for position, source_id in enumerate(topological)
    }
    output = nx.DiGraph(
        source="WorkflowSim/Pegasus DAX",
        source_family=family,
        source_commit=SOURCE_COMMIT,
        extraction="deterministic_connected_induced_subgraph_v1",
    )
    output.add_node("0", service=0, cpucycle=0)
    for source_id in topological:
        job = jobs[source_id]
        identifier = service_id(job["namespace"], job["name"])
        output.add_node(
            node_mapping[source_id],
            service=encoded_service_value(identifier),
            cpucycle=scale_log(
                job["runtime"],
                runtime_bounds[0],
                runtime_bounds[1],
            ),
            source_job_type=f"{job['namespace']}:{job['name']}",
        )
    for source, target, attributes in subgraph.edges(data=True):
        raw_bytes = attributes["data_bytes"]
        if raw_bytes <= 0:
            raw_bytes = data_bounds[2]
        output.add_edge(
            node_mapping[source],
            node_mapping[target],
            datalength=scale_log(
                raw_bytes,
                data_bounds[0],
                data_bounds[1],
                minimum=0.01,
            ),
        )
    real_roots = [node for node in output.nodes if node != "0" and output.in_degree(node) == 0]
    for root_id in real_roots:
        output.add_edge("0", root_id, datalength=0.0)
    if output.number_of_nodes() != REAL_TASKS_PER_GRAPH + 1:
        raise RuntimeError("Converted graph violates the ten-task limit")
    if not nx.is_directed_acyclic_graph(output):
        raise RuntimeError("Converted graph is not a DAG")
    return f"pegasus_{family.lower()}_{index:02d}", nx.node_link_data(output)


def build_dataset(raw_dir):
    parsed = {}
    runtimes = []
    positive_data_sizes = []
    for family, source_path in SOURCE_FILES.items():
        graph, jobs = parse_dax(raw_dir / local_name(source_path), family)
        parsed[family] = (graph, jobs)
        runtimes.extend(job["runtime"] for job in jobs.values())
        positive_data_sizes.extend(
            attributes["data_bytes"]
            for _, _, attributes in graph.edges(data=True)
            if attributes["data_bytes"] > 0
        )
    runtime_logs = np.log1p(np.asarray(runtimes, dtype=float))
    data_logs = np.log1p(np.asarray(positive_data_sizes, dtype=float))
    runtime_bounds = (float(runtime_logs.min()), float(runtime_logs.max()))
    fallback_data_size = float(np.median(positive_data_sizes))
    data_bounds = (
        float(data_logs.min()),
        float(data_logs.max()),
        fallback_data_size,
    )

    dataset = {}
    graph_manifest = []
    service_counter = Counter()
    type_mapping = {}
    for family in SOURCE_FILES:
        graph, jobs = parsed[family]
        subgraphs = select_connected_subgraphs(graph, family)
        for index, subgraph in enumerate(subgraphs):
            key, converted = convert_subgraph(
                family,
                index,
                subgraph,
                jobs,
                runtime_bounds,
                data_bounds,
            )
            dataset[key] = converted
            graph_object = nx.node_link_graph(converted)
            services = [
                int(SERVICE_COUNT * (float(value["service"]) - 1)) + 1
                for node_id, value in graph_object.nodes(data=True)
                if node_id != "0"
            ]
            service_counter.update(services)
            for source_id in subgraph.nodes:
                job = jobs[source_id]
                job_type = f"{job['namespace']}:{job['name']}"
                type_mapping[job_type] = service_id(job["namespace"], job["name"])
            graph_manifest.append(
                {
                    "key": key,
                    "family": family,
                    "nodes_including_dummy": graph_object.number_of_nodes(),
                    "edges_including_dummy": graph_object.number_of_edges(),
                    "depth": nx.dag_longest_path_length(graph_object),
                    "service_ids": sorted(set(services)),
                }
            )
    if len(dataset) != len(SOURCE_FILES) * GRAPHS_PER_FAMILY:
        raise RuntimeError("Unexpected generated graph count")
    if set(service_counter) != set(range(1, SERVICE_COUNT + 1)):
        raise RuntimeError("Generated benchmark does not cover all services")
    metadata = {
        "runtime_log_bounds": runtime_bounds,
        "data_log_bounds": data_bounds[:2],
        "fallback_data_bytes": fallback_data_size,
        "service_counts": dict(sorted(service_counter.items())),
        "job_type_to_service": dict(sorted(type_mapping.items())),
        "graphs": graph_manifest,
    }
    return dataset, metadata


def render_readme(dataset_sha256):
    return f"""# Pegasus Cross-Topology DAG Benchmark

This dataset is a deterministic, post-hoc cross-topology benchmark for h8v1.
It is not an unbiased MEC trace or an independent preregistered holdout.

- Source: {SOURCE_REPOSITORY}
- Pinned source commit: `{SOURCE_COMMIT}`
- Families: Montage, CyberShake, Epigenomics, Inspiral, SIPHT
- Output: 20 connected sub-DAGs per family, 100 graphs total
- Size: 9 real tasks plus one dummy source per graph
- Services: stable hash of the original Pegasus job type into 10 service IDs
- CPU and edge data: global log scaling into the existing DAOC normalized ranges
- Dataset SHA-256: `{dataset_sha256}`

The five source DAX files and upstream license notices are stored under `raw/`.
Regenerate with:

```bash
python build_pegasus_cross_dataset.py
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
    dataset_path = args.output_dir / "dag_pegasus5_sub10.json"
    dataset_payload = json.dumps(
        dataset,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ).encode("utf-8")
    dataset_path.write_bytes(dataset_payload)
    dataset_sha256 = sha256_bytes(dataset_payload)
    manifest = {
        "status": "complete",
        "claim_scope": "posthoc_external_cross_topology_only",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_files": source_records,
        "generator": {
            "seed": GENERATOR_SEED,
            "families": list(SOURCE_FILES),
            "graphs_per_family": GRAPHS_PER_FAMILY,
            "real_tasks_per_graph": REAL_TASKS_PER_GRAPH,
            "dummy_tasks_per_graph": 1,
            "service_count": SERVICE_COUNT,
            "subgraph_rule": "deterministic_connected_induced_subgraph_v1",
            "normalization": "global_log_minmax_v1",
        },
        "generation": generation,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_sha256,
            "graph_count": len(dataset),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        render_readme(dataset_sha256),
        encoding="utf-8",
    )
    print(dataset_path)
    print(dataset_sha256)


if __name__ == "__main__":
    main()
