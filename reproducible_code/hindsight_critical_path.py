"""Posterior critical-path labels computed from completed DAG outcomes."""

import math


EPSILON = 1e-12


def _edge_latency(task, successor_id, successor, between_server_costs):
    return float(
        between_server_costs[
            int(task.assigned_server),
            int(successor.assigned_server),
        ]
        * task.outputs_length.get(successor_id, 0.0)
    )


def posterior_critical_path(tasks, between_server_costs, temperature=0.05):
    """Return exact-path and soft-slack labels after a DAG has completed.

    The function deliberately requires completed outcomes. It cannot be used
    for action selection because every task must already have a finite finish
    time and an assigned server.
    """
    if not tasks:
        raise ValueError("tasks must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    task_ids = set(tasks)
    indegree = {}
    for task_id, task in tasks.items():
        finish_time = float(task.result.finish_time)
        if not task.done or not math.isfinite(finish_time):
            raise ValueError(
                "posterior critical paths require completed tasks"
            )
        if int(task.assigned_server) < 0:
            raise ValueError(
                "posterior critical paths require assigned servers"
            )
        indegree[task_id] = sum(
            predecessor_id in task_ids
            for predecessor_id in task.predecessors
        )

    ready = sorted(
        (
            task_id
            for task_id, degree in indegree.items()
            if degree == 0
        ),
        key=str,
    )
    order = []
    while ready:
        task_id = ready.pop(0)
        order.append(task_id)
        for successor_id in tasks[task_id].successors:
            if successor_id not in indegree:
                continue
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)
                ready.sort(key=str)
    if len(order) != len(tasks):
        raise ValueError("the application graph must be acyclic")

    predecessor_arrival = {}
    local_latency = {}
    critical_predecessor = {}
    for task_id in order:
        task = tasks[task_id]
        candidates = []
        for predecessor_id in task.predecessors:
            if predecessor_id not in tasks:
                continue
            predecessor = tasks[predecessor_id]
            arrival = (
                float(predecessor.result.finish_time)
                + _edge_latency(
                    predecessor,
                    task_id,
                    task,
                    between_server_costs,
                )
            )
            candidates.append((arrival, str(predecessor_id), predecessor_id))
        if candidates:
            _, _, parent = max(candidates)
            arrival = max(value[0] for value in candidates)
            critical_predecessor[task_id] = parent
        else:
            arrival = 0.0
            critical_predecessor[task_id] = None
        predecessor_arrival[task_id] = arrival
        local_latency[task_id] = max(
            float(task.result.finish_time) - arrival,
            0.0,
        )

    sinks = [
        task_id
        for task_id, task in tasks.items()
        if not any(
            successor_id in tasks
            for successor_id in task.successors
        )
    ]
    if not sinks:
        raise ValueError("the application graph must contain a sink")
    sink = max(
        sinks,
        key=lambda task_id: (
            float(tasks[task_id].result.finish_time),
            str(task_id),
        ),
    )
    makespan = max(
        float(tasks[task_id].result.finish_time)
        for task_id in sinks
    )

    exact_path = []
    cursor = sink
    while cursor is not None:
        exact_path.append(cursor)
        cursor = critical_predecessor[cursor]
    exact_path.reverse()
    exact_path_set = set(exact_path)

    tail_latency = {}
    for task_id in reversed(order):
        task = tasks[task_id]
        tails = []
        for successor_id in task.successors:
            if successor_id not in tasks:
                continue
            successor = tasks[successor_id]
            tails.append(
                _edge_latency(
                    task,
                    successor_id,
                    successor,
                    between_server_costs,
                )
                + local_latency[successor_id]
                + tail_latency[successor_id]
            )
        tail_latency[task_id] = max(tails, default=0.0)

    scale = max(float(temperature) * makespan, EPSILON)
    slack = {}
    scores = {}
    for task_id in order:
        path_through = (
            float(tasks[task_id].result.finish_time)
            + tail_latency[task_id]
        )
        task_slack = max(makespan - path_through, 0.0)
        slack[task_id] = task_slack
        scores[task_id] = float(
            math.exp(-task_slack / scale)
        )
    for task_id in exact_path_set:
        scores[task_id] = 1.0

    return {
        "path": tuple(exact_path),
        "scores": scores,
        "slack": slack,
        "makespan": makespan,
        "local_latency": local_latency,
        "tail_latency": tail_latency,
    }


def bottleneck_contribution_scores(posterior, top_fraction=0.25):
    """Select a sparse set of high-contribution posterior bottlenecks.

    The score combines near-critical-path membership with the task's realized
    local contribution to application makespan. Only the highest-scoring
    fraction is retained, so replay does not amplify most tasks in a nearly
    chain-shaped DAG.
    """
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    path_scores = posterior["scores"]
    local_latency = posterior["local_latency"]
    if set(path_scores) != set(local_latency):
        raise ValueError(
            "posterior scores and local latencies must share task ids"
        )
    if not path_scores:
        raise ValueError("posterior must contain at least one task")

    makespan = float(posterior["makespan"])
    if not math.isfinite(makespan) or makespan < 0.0:
        raise ValueError("posterior makespan must be finite and non-negative")

    denominator = max(makespan, EPSILON)
    raw_scores = {
        task_id: (
            max(float(path_scores[task_id]), 0.0)
            * max(float(local_latency[task_id]), 0.0)
            / denominator
        )
        for task_id in path_scores
    }
    selected_count = max(
        1,
        int(math.ceil(float(top_fraction) * len(raw_scores))),
    )
    ranked = sorted(
        raw_scores,
        key=lambda task_id: (
            -raw_scores[task_id],
            str(task_id),
        ),
    )
    selected = tuple(
        task_id
        for task_id in ranked[:selected_count]
        if raw_scores[task_id] > 0.0
    )
    maximum = max(
        (raw_scores[task_id] for task_id in selected),
        default=0.0,
    )
    scores = {
        task_id: (
            raw_scores[task_id] / maximum
            if task_id in selected and maximum > 0.0
            else 0.0
        )
        for task_id in raw_scores
    }
    return {
        "scores": scores,
        "raw_scores": raw_scores,
        "selected": selected,
        "selected_fraction": len(selected) / len(raw_scores),
    }
