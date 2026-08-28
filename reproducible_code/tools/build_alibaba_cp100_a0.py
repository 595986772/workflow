#!/usr/bin/env python3
"""Build the Alibaba-CP100 A0 service-alignment control variant."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260801
NUM_SERVICES = 10
HOT_SERVICE_MAX = 3

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "alibaba_cp100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically permute service labels within each CP100 DAG "
            "while preserving every per-DAG service multiset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_DIR / "dag_alibaba_cp100.json",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DATASET_DIR / "selection_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "dag_alibaba_cp100_a0.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DATASET_DIR / "service_alignment_a0_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-abs-correlation",
        type=float,
        default=0.10,
        help=(
            "Maximum accepted absolute task-level correlation between "
            "service popularity and critical-path membership."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output only when explicitly requested.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_output(path: Path, payload: bytes, force: bool) -> None:
    if path.exists():
        current = path.read_bytes()
        if current == payload:
            return
        if not force:
            raise FileExistsError(
                f"refusing to replace different output without --force: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def stable_digest(seed: int, *parts: object) -> bytes:
    text = "|".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(text.encode("utf-8")).digest()


def node_sort_key(node_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(node_id))
    except ValueError:
        return (1, node_id)


def decode_service(value: object) -> int:
    service_id = int(NUM_SERVICES * (float(value) - 1.0)) + 1
    if not 1 <= service_id <= NUM_SERVICES:
        raise ValueError(f"encoded service is out of range: {value}")
    return service_id


def encode_service(service_id: int) -> float:
    if not 1 <= service_id <= NUM_SERVICES:
        raise ValueError(f"service id is out of range: {service_id}")
    return round(1.0 + (service_id - 0.5) / NUM_SERVICES, 12)


def pearson(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("correlation inputs must have equal non-zero length")
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0.0:
        return 0.0
    return sum(
        value_a * value_b
        for value_a, value_b in zip(centered_a, centered_b)
    ) / denominator


def graph_nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["id"]): node
        for node in graph["nodes"]
        if str(node["id"]) != "0"
    }


def service_counts(graphs: dict[str, dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for graph in graphs.values():
        for node in graph_nodes(graph).values():
            counts[decode_service(node["service"])] += 1
    return counts


def association_metrics(
    graphs: dict[str, dict[str, Any]],
    selection_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = service_counts(graphs)
    critical_by_service: Counter[int] = Counter()
    popularity_values: list[float] = []
    critical_flags: list[float] = []
    contingency = {
        "critical_hot": 0,
        "critical_cold": 0,
        "off_path_hot": 0,
        "off_path_cold": 0,
    }

    for graph_key, graph in graphs.items():
        critical_nodes = {
            str(value)
            for value in selection_by_key[graph_key][
                "critical_path_daoc_ids"
            ]
        }
        for node_id, node in graph_nodes(graph).items():
            service_id = decode_service(node["service"])
            is_critical = node_id in critical_nodes
            is_hot = service_id <= HOT_SERVICE_MAX
            critical_by_service[service_id] += int(is_critical)
            popularity_values.append(float(counts[service_id]))
            critical_flags.append(float(is_critical))
            if is_critical and is_hot:
                contingency["critical_hot"] += 1
            elif is_critical:
                contingency["critical_cold"] += 1
            elif is_hot:
                contingency["off_path_hot"] += 1
            else:
                contingency["off_path_cold"] += 1

    a = contingency["critical_hot"]
    b = contingency["critical_cold"]
    c = contingency["off_path_hot"]
    d = contingency["off_path_cold"]
    phi_denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / phi_denominator if phi_denominator else 0.0
    critical_total = a + b
    off_path_total = c + d

    return {
        "definition": (
            "Criticality is membership in the trace-derived weighted "
            "critical path. Popularity is the global request count of the "
            "assigned service. Services 1-3 form the original hot group."
        ),
        "task_level_popularity_criticality_pearson": pearson(
            popularity_values,
            critical_flags,
        ),
        "critical_path_hot_service_phi": phi,
        "critical_path_hot_fraction": (
            a / critical_total if critical_total else 0.0
        ),
        "off_path_hot_fraction": (
            c / off_path_total if off_path_total else 0.0
        ),
        "contingency": contingency,
        "per_service": {
            str(service_id): {
                "request_count": counts[service_id],
                "critical_path_count": critical_by_service[service_id],
                "critical_path_fraction": (
                    critical_by_service[service_id] / counts[service_id]
                    if counts[service_id]
                    else 0.0
                ),
            }
            for service_id in range(1, NUM_SERVICES + 1)
        },
    }


def validate_base_assignments(
    graphs: dict[str, dict[str, Any]],
    selection_by_key: dict[str, dict[str, Any]],
) -> None:
    for graph_key, graph in graphs.items():
        expected = {
            str(node_id): int(service_id)
            for node_id, service_id in selection_by_key[graph_key][
                "assigned_service_ids"
            ].items()
        }
        observed = {
            node_id: decode_service(node["service"])
            for node_id, node in graph_nodes(graph).items()
        }
        if observed != expected:
            raise RuntimeError(
                f"{graph_key}: base graph does not match selection manifest"
            )


def validate_non_service_content(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> None:
    if list(before) != list(after):
        raise RuntimeError("graph key order changed")
    for graph_key in before:
        before_graph = copy.deepcopy(before[graph_key])
        after_graph = copy.deepcopy(after[graph_key])
        before_nodes = {
            str(node["id"]): node for node in before_graph["nodes"]
        }
        for node in after_graph["nodes"]:
            node["service"] = before_nodes[str(node["id"])]["service"]
        if before_graph != after_graph:
            raise RuntimeError(
                f"{graph_key}: non-service graph content changed"
            )


def transform_graphs(
    graphs: dict[str, dict[str, Any]],
    selection_by_key: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int, int]:
    transformed = copy.deepcopy(graphs)
    per_graph: list[dict[str, Any]] = []
    total_tasks = 0
    changed_tasks = 0

    for graph_key in graphs:
        before_nodes = graph_nodes(graphs[graph_key])
        after_nodes = graph_nodes(transformed[graph_key])
        node_ids = sorted(before_nodes, key=node_sort_key)
        original_services = [
            decode_service(before_nodes[node_id]["service"])
            for node_id in node_ids
        ]
        target_ids = sorted(
            node_ids,
            key=lambda node_id: stable_digest(
                seed,
                graph_key,
                node_id,
                "a0-target",
            ),
        )
        new_services = dict(zip(target_ids, original_services))
        before_services = {
            node_id: decode_service(before_nodes[node_id]["service"])
            for node_id in node_ids
        }
        mapping: dict[str, dict[str, Any]] = {}

        for node_id in node_ids:
            old_service = before_services[node_id]
            new_service = new_services[node_id]
            after_nodes[node_id]["service"] = encode_service(new_service)
            changed = old_service != new_service
            total_tasks += 1
            changed_tasks += int(changed)
            mapping[node_id] = {
                "before": old_service,
                "after": new_service,
                "critical_path": (
                    node_id
                    in {
                        str(value)
                        for value in selection_by_key[graph_key][
                            "critical_path_daoc_ids"
                        ]
                    }
                ),
            }

        before_multiset = Counter(before_services.values())
        after_multiset = Counter(new_services.values())
        if before_multiset != after_multiset:
            raise RuntimeError(
                f"{graph_key}: per-DAG service multiset changed"
            )
        per_graph.append(
            {
                "dataset_key": graph_key,
                "task_count": len(node_ids),
                "changed_tasks": sum(
                    value["before"] != value["after"]
                    for value in mapping.values()
                ),
                "service_counts": {
                    str(service_id): before_multiset[service_id]
                    for service_id in range(1, NUM_SERVICES + 1)
                    if before_multiset[service_id]
                },
                "task_service_map": mapping,
            }
        )

    return transformed, per_graph, total_tasks, changed_tasks


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    selection_path = args.selection_manifest.resolve()
    output_path = args.output.resolve()
    output_manifest_path = args.output_manifest.resolve()
    if output_path == input_path:
        raise ValueError("output must not overwrite the base CP100 dataset")
    if not 0.0 <= args.max_abs_correlation < 1.0:
        raise ValueError("max-abs-correlation must be in [0, 1)")

    graphs = read_json(input_path)
    selection_manifest = read_json(selection_path)
    if not isinstance(graphs, dict) or not isinstance(selection_manifest, list):
        raise TypeError("unexpected CP100 dataset or manifest schema")
    selection_by_key = {
        str(item["dataset_key"]): item for item in selection_manifest
    }
    if set(graphs) != set(selection_by_key):
        raise RuntimeError("dataset and selection manifest graph keys differ")
    validate_base_assignments(graphs, selection_by_key)

    before_counts = service_counts(graphs)
    before_association = association_metrics(graphs, selection_by_key)
    transformed, per_graph, total_tasks, changed_tasks = transform_graphs(
        graphs,
        selection_by_key,
        args.seed,
    )
    validate_non_service_content(graphs, transformed)
    after_counts = service_counts(transformed)
    if before_counts != after_counts:
        raise RuntimeError("global service counts changed")
    after_association = association_metrics(transformed, selection_by_key)
    correlation = after_association[
        "task_level_popularity_criticality_pearson"
    ]
    phi = after_association["critical_path_hot_service_phi"]
    if abs(correlation) > args.max_abs_correlation:
        raise RuntimeError(
            "A0 popularity-criticality correlation exceeds threshold: "
            f"{correlation:.6f}"
        )
    if abs(phi) > args.max_abs_correlation:
        raise RuntimeError(
            f"A0 critical/hot phi exceeds threshold: {phi:.6f}"
        )

    dataset_payload = json_bytes(transformed)
    dataset_sha256 = sha256_bytes(dataset_payload)
    write_output(output_path, dataset_payload, args.force)

    manifest = {
        "dataset_name": "Alibaba-CP100-A0",
        "dataset_role": "controlled_service_alignment_variant",
        "formal_unbiased_holdout": False,
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "base_dataset": {
            "path": input_path.name,
            "sha256": sha256_file(input_path),
        },
        "selection_manifest": {
            "path": selection_path.name,
            "sha256": sha256_file(selection_path),
        },
        "output_dataset": {
            "path": output_path.name,
            "sha256": dataset_sha256,
        },
        "transformation": {
            "name": "within_dag_independent_service_permutation",
            "scope": "each DAG independently",
            "algorithm": (
                "sort target task IDs by SHA-256(seed, graph key, task "
                "ID, a0-target), then assign the original node-order "
                "service sequence"
            ),
            "dummy_source": "node 0 is unchanged",
            "preserved_exactly": [
                "graph key order",
                "DAG topology",
                "node IDs",
                "CPU-cycle attributes",
                "edge payloads",
                "per-DAG service multisets",
                "global service counts",
            ],
        },
        "integrity": {
            "graph_count": len(graphs),
            "real_task_count": total_tasks,
            "changed_task_count": changed_tasks,
            "changed_task_fraction": changed_tasks / total_tasks,
            "graph_key_order_unchanged": list(graphs) == list(transformed),
            "non_service_content_unchanged": True,
            "per_dag_service_multisets_unchanged": True,
            "global_service_counts_unchanged": True,
            "a0_max_abs_correlation": args.max_abs_correlation,
            "a0_acceptance_passed": True,
        },
        "service_counts": {
            "before": {
                str(service_id): before_counts[service_id]
                for service_id in range(1, NUM_SERVICES + 1)
            },
            "after": {
                str(service_id): after_counts[service_id]
                for service_id in range(1, NUM_SERVICES + 1)
            },
        },
        "association": {
            "before": before_association,
            "after": after_association,
        },
        "per_graph": per_graph,
    }
    manifest_payload = json_bytes(manifest)
    write_output(output_manifest_path, manifest_payload, args.force)

    print(f"Wrote {output_path}")
    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"Wrote {output_manifest_path}")
    print(
        "Popularity-criticality correlation: "
        f"{before_association['task_level_popularity_criticality_pearson']:.6f} "
        f"-> {correlation:.6f}"
    )
    print(
        "Critical-path/hot-service phi: "
        f"{before_association['critical_path_hot_service_phi']:.6f} "
        f"-> {phi:.6f}"
    )
    print(f"Changed task labels: {changed_tasks}/{total_tasks}")


if __name__ == "__main__":
    main()
