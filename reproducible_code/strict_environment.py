"""Deterministic, nested stress transforms for DAOC DAG instances."""

from dataclasses import asdict, dataclass
import copy

import networkx as nx


DUMMY_ROOT = "0"


@dataclass(frozen=True)
class DAGStressMetadata:
    base_depth: int
    stressed_depth: int
    target_depth: int
    dependency_data_scale: float
    base_dependency_data: float
    stressed_dependency_data: float
    added_control_edges: tuple

    def to_dict(self):
        result = asdict(self)
        result["added_control_edges"] = [
            [source, target]
            for source, target in self.added_control_edges
        ]
        return result


def task_subgraph(graph):
    task_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if node != DUMMY_ROOT and attributes.get("service", 0) > 0
    ]
    return graph.subgraph(task_nodes).copy()


def dag_task_depth(graph):
    tasks = task_subgraph(graph)
    if not tasks:
        return 0
    if not nx.is_directed_acyclic_graph(tasks):
        raise ValueError("Application graph must be a DAG")
    return nx.dag_longest_path_length(tasks) + 1


def dependency_data_total(graph):
    return float(
        sum(
            float(attributes.get("datalength", 0.0))
            for source, _, attributes in graph.edges(data=True)
            if source != DUMMY_ROOT
        )
    )


def _candidate_depth(graph, source, target):
    graph.add_edge(
        source,
        target,
        datalength=0.0,
        stress_control_dependency=True,
    )
    depth = dag_task_depth(graph)
    graph.remove_edge(source, target)
    return depth


def augment_dag_depth(graph, depth_increment):
    if isinstance(depth_increment, bool) or not isinstance(
        depth_increment,
        int,
    ):
        raise TypeError("depth_increment must be an integer")
    if depth_increment < 0:
        raise ValueError("depth_increment must be non-negative")

    stressed = copy.deepcopy(nx.DiGraph(graph))
    if not nx.is_directed_acyclic_graph(stressed):
        raise ValueError("Application graph must be a DAG")

    base_depth = dag_task_depth(stressed)
    number_of_tasks = task_subgraph(stressed).number_of_nodes()
    target_depth = min(base_depth + depth_increment, number_of_tasks)
    added_edges = []

    while dag_task_depth(stressed) < target_depth:
        tasks = task_subgraph(stressed)
        order = list(nx.topological_sort(tasks))
        current_depth = dag_task_depth(stressed)
        candidates = []
        for source_index, source in enumerate(order):
            for target_index in range(source_index + 1, len(order)):
                target = order[target_index]
                if stressed.has_edge(source, target):
                    continue
                candidate_depth = _candidate_depth(
                    stressed,
                    source,
                    target,
                )
                if candidate_depth <= current_depth:
                    continue
                candidates.append(
                    (
                        abs(target_depth - candidate_depth),
                        candidate_depth,
                        source_index,
                        target_index,
                        source,
                        target,
                    )
                )

        if not candidates:
            break
        *_, source, target = min(candidates)
        stressed.add_edge(
            source,
            target,
            datalength=0.0,
            stress_control_dependency=True,
        )
        added_edges.append((source, target))

    return stressed, tuple(added_edges), target_depth


def apply_strict_dag_stress(
    graph,
    depth_increment=0,
    dependency_data_scale=1.0,
):
    dependency_data_scale = float(dependency_data_scale)
    if dependency_data_scale < 1.0:
        raise ValueError(
            "dependency_data_scale must be at least one"
        )

    base_depth = dag_task_depth(graph)
    base_dependency_data = dependency_data_total(graph)
    stressed, added_edges, target_depth = augment_dag_depth(
        graph,
        depth_increment,
    )
    for source, _, attributes in stressed.edges(data=True):
        if source == DUMMY_ROOT:
            continue
        attributes["datalength"] = (
            float(attributes.get("datalength", 0.0))
            * dependency_data_scale
        )

    metadata = DAGStressMetadata(
        base_depth=base_depth,
        stressed_depth=dag_task_depth(stressed),
        target_depth=target_depth,
        dependency_data_scale=dependency_data_scale,
        base_dependency_data=base_dependency_data,
        stressed_dependency_data=dependency_data_total(stressed),
        added_control_edges=added_edges,
    )
    return stressed, metadata
