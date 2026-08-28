"""Clairvoyant optimistic latency bounds for the DAOC environment."""

from dataclasses import dataclass
import time

import networkx as nx
import numpy as np

from critical_path_cache import coordinated_cache_decision

try:
    import pulp
except ImportError:  # pragma: no cover - exercised by runtime validation
    pulp = None


@dataclass(frozen=True)
class AssignmentProblem:
    task_ids: tuple
    predecessors: dict
    local_costs: dict
    transfer_costs: dict
    sink: object
    num_servers: int


@dataclass(frozen=True)
class ExactOracleResult:
    objective: float
    status: str
    wall_time_sec: float
    assignment: dict


def _topological_order(problem):
    graph = nx.DiGraph()
    graph.add_nodes_from(problem.task_ids)
    for task_id in problem.task_ids:
        for predecessor in problem.predecessors[task_id]:
            graph.add_edge(predecessor, task_id)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Oracle assignment problem must be a DAG")
    if problem.sink not in graph:
        raise ValueError("Oracle sink is not part of the DAG")
    return tuple(nx.topological_sort(graph))


def validate_assignment_problem(problem):
    if problem.num_servers < 1:
        raise ValueError("Oracle requires at least one server")
    if not problem.task_ids:
        raise ValueError("Oracle requires at least one task")
    task_set = set(problem.task_ids)
    if len(task_set) != len(problem.task_ids):
        raise ValueError("Oracle task ids must be unique")

    for task_id in problem.task_ids:
        if task_id not in problem.predecessors:
            raise ValueError(f"Missing predecessors for task {task_id}")
        if any(
            predecessor not in task_set
            for predecessor in problem.predecessors[task_id]
        ):
            raise ValueError(
                f"Unknown predecessor for task {task_id}"
            )
        for server_id in range(problem.num_servers):
            key = (task_id, server_id)
            if key not in problem.local_costs:
                raise ValueError(f"Missing local cost {key}")
            if problem.local_costs[key] < 0:
                raise ValueError(f"Negative local cost {key}")
        for predecessor in problem.predecessors[task_id]:
            for predecessor_server in range(problem.num_servers):
                for server_id in range(problem.num_servers):
                    key = (
                        predecessor,
                        task_id,
                        predecessor_server,
                        server_id,
                    )
                    if key not in problem.transfer_costs:
                        raise ValueError(f"Missing transfer cost {key}")
                    if problem.transfer_costs[key] < 0:
                        raise ValueError(
                            f"Negative transfer cost {key}"
                        )
    _topological_order(problem)


def relaxed_assignment_lower_bound(problem):
    """Return a certified DAG latency lower bound.

    The recurrence permits a predecessor to use a different placement for
    each outgoing dependency. This relaxation can only reduce latency.
    """

    validate_assignment_problem(problem)
    completion = {}
    for task_id in _topological_order(problem):
        predecessors = problem.predecessors[task_id]
        for server_id in range(problem.num_servers):
            local_cost = problem.local_costs[(task_id, server_id)]
            if not predecessors:
                completion[(task_id, server_id)] = local_cost
                continue

            predecessor_floor = max(
                min(
                    completion[(predecessor, predecessor_server)]
                    + problem.transfer_costs[
                        (
                            predecessor,
                            task_id,
                            predecessor_server,
                            server_id,
                        )
                    ]
                    for predecessor_server in range(
                        problem.num_servers
                    )
                )
                for predecessor in predecessors
            )
            completion[(task_id, server_id)] = (
                predecessor_floor + local_cost
            )

    return min(
        completion[(problem.sink, server_id)]
        for server_id in range(problem.num_servers)
    )


def exact_optimistic_assignment_oracle(
    problem,
    time_limit_sec=30.0,
    msg=False,
):
    """Solve the optimistic placement problem exactly with CBC."""

    validate_assignment_problem(problem)
    if pulp is None:
        raise RuntimeError("Exact oracle validation requires PuLP")
    if time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive")

    model = pulp.LpProblem(
        "ExactOptimisticDAGPlacement",
        pulp.LpMinimize,
    )
    task_indices = {
        task_id: index
        for index, task_id in enumerate(problem.task_ids)
    }
    x = {
        (task_id, server_id): pulp.LpVariable(
            f"x_{task_indices[task_id]}_{server_id}",
            cat=pulp.LpBinary,
        )
        for task_id in problem.task_ids
        for server_id in range(problem.num_servers)
    }
    completion = {
        task_id: pulp.LpVariable(
            f"finish_{task_indices[task_id]}",
            lowBound=0.0,
        )
        for task_id in problem.task_ids
    }
    edge_assignments = {}

    for task_id in problem.task_ids:
        model += (
            pulp.lpSum(
                x[(task_id, server_id)]
                for server_id in range(problem.num_servers)
            )
            == 1
        )

    for task_id in problem.task_ids:
        selected_local_cost = pulp.lpSum(
            problem.local_costs[(task_id, server_id)]
            * x[(task_id, server_id)]
            for server_id in range(problem.num_servers)
        )
        predecessors = problem.predecessors[task_id]
        if not predecessors:
            model += completion[task_id] >= selected_local_cost
            continue

        for predecessor in predecessors:
            edge_cost = []
            for predecessor_server in range(problem.num_servers):
                for server_id in range(problem.num_servers):
                    key = (
                        predecessor,
                        task_id,
                        predecessor_server,
                        server_id,
                    )
                    variable = pulp.LpVariable(
                        "y_"
                        f"{task_indices[predecessor]}_"
                        f"{task_indices[task_id]}_"
                        f"{predecessor_server}_{server_id}",
                        cat=pulp.LpBinary,
                    )
                    edge_assignments[key] = variable
                    model += (
                        variable
                        <= x[(predecessor, predecessor_server)]
                    )
                    model += variable <= x[(task_id, server_id)]
                    model += (
                        variable
                        >= x[(predecessor, predecessor_server)]
                        + x[(task_id, server_id)]
                        - 1
                    )
                    edge_cost.append(
                        problem.transfer_costs[key] * variable
                    )
            model += (
                completion[task_id]
                >= completion[predecessor]
                + selected_local_cost
                + pulp.lpSum(edge_cost)
            )

    model += completion[problem.sink]
    started = time.perf_counter()
    model.solve(
        pulp.PULP_CBC_CMD(
            msg=msg,
            timeLimit=float(time_limit_sec),
            threads=1,
        )
    )
    wall_time = time.perf_counter() - started
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(
            f"Exact optimistic oracle did not converge: {status}"
        )

    assignment = {
        task_id: max(
            range(problem.num_servers),
            key=lambda server_id: pulp.value(
                x[(task_id, server_id)]
            ),
        )
        for task_id in problem.task_ids
    }
    return ExactOracleResult(
        objective=float(pulp.value(model.objective)),
        status=status,
        wall_time_sec=wall_time,
        assignment=assignment,
    )


def assignment_problem_from_simulator(
    simulator,
    user,
    include_exogenous_waiting=False,
    service_placement=None,
):
    task_ids = tuple(user.tasks_init)
    task_set = set(task_ids)
    predecessors = {
        task_id: tuple(
            predecessor
            for predecessor in user.tasks_init[
                task_id
            ].predecessors
            if predecessor in task_set
        )
        for task_id in task_ids
    }
    sinks = [
        task_id
        for task_id in task_ids
        if not user.tasks_init[task_id].successors
    ]
    if not sinks:
        raise ValueError("Expected at least one application sink")
    sink = sinks[0]

    local_costs = {}
    transfer_costs = {}
    for task_id, task in user.tasks_init.items():
        for server_id, server in simulator.servers.items():
            data_latency = 0.0
            if task.input_data_length > 0:
                data_latency = task.input_data_length * (
                    1.0 / user.rate_to_gateway
                    + simulator.between_server_costs[
                        user.nearest_server,
                        server_id,
                    ]
                )
            waiting_latency = (
                server.load * 1e6 / server.frequency
                if include_exogenous_waiting
                else 0.0
            )
            service_latency = 0.0
            if (
                service_placement is not None
                and task.service > 0
                and task.service
                not in service_placement[server_id]
            ):
                source_costs = [1.0 / server.rate_to_cloud]
                source_costs.extend(
                    simulator.between_server_costs[
                        source_id,
                        server_id,
                    ]
                    for source_id, services
                    in service_placement.items()
                    if task.service in services
                )
                service_latency = (
                    simulator.service_data_length[task.service]
                    * min(source_costs)
                )
            local_costs[(task_id, server_id)] = (
                data_latency
                + task.cpu_cycle / server.frequency
                + waiting_latency
                + service_latency
            )

        for predecessor in predecessors[task_id]:
            output_length = user.tasks_init[
                predecessor
            ].outputs_length[task_id]
            for predecessor_server in range(simulator.S):
                for server_id in range(simulator.S):
                    transfer_costs[
                        (
                            predecessor,
                            task_id,
                            predecessor_server,
                            server_id,
                        )
                    ] = (
                        output_length
                        * simulator.between_server_costs[
                            predecessor_server,
                            server_id,
                        ]
                    )

    if len(sinks) > 1:
        # A zero-cost super-sink turns multi-exit workflow completion into
        # the maximum completion time over all real sinks.
        sink = "__oracle_super_sink__"
        suffix = 1
        while sink in task_set:
            sink = f"__oracle_super_sink__{suffix}"
            suffix += 1
        task_ids = (*task_ids, sink)
        predecessors[sink] = tuple(sinks)
        for server_id in range(simulator.S):
            local_costs[(sink, server_id)] = 0.0
        for predecessor in sinks:
            for predecessor_server in range(simulator.S):
                for server_id in range(simulator.S):
                    transfer_costs[
                        (
                            predecessor,
                            sink,
                            predecessor_server,
                            server_id,
                        )
                    ] = 0.0

    return AssignmentProblem(
        task_ids=task_ids,
        predecessors=predecessors,
        local_costs=local_costs,
        transfer_costs=transfer_costs,
        sink=sink,
        num_servers=simulator.S,
    )


def scenario_relaxed_oracle_bounds(
    simulator,
    include_exogenous_waiting=False,
):
    values = np.asarray(
        [
            relaxed_assignment_lower_bound(
                assignment_problem_from_simulator(
                    simulator,
                    user,
                    include_exogenous_waiting=(
                        include_exogenous_waiting
                    ),
                )
            )
            for user in simulator.users.values()
        ],
        dtype=float,
    )
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "per_user": values.tolist(),
    }


def clairvoyant_capacity_placement(simulator):
    """Place services using the complete current workload and K_s."""
    demand = {
        server_id: {
            service_id: 0.0
            for service_id in range(1, simulator.Q + 1)
        }
        for server_id in simulator.servers
    }
    for user in simulator.users.values():
        destination = int(user.nearest_server)
        for task in user.tasks_init.values():
            if task.service > 0:
                demand[destination][task.service] += 1.0

    capacities = {
        server_id: int(server.capacity)
        for server_id, server in simulator.servers.items()
    }
    placement = coordinated_cache_decision(
        demand=demand,
        current_services={
            server_id: []
            for server_id in simulator.servers
        },
        capacity=capacities,
        service_sizes={
            service_id: simulator.service_data_length[service_id]
            for service_id in range(1, simulator.Q + 1)
        },
        cloud_costs={
            server_id: 1.0 / server.rate_to_cloud
            for server_id, server in simulator.servers.items()
        },
        between_server_costs=simulator.between_server_costs,
        expected_requests=sum(
            len(user.tasks_init)
            for user in simulator.users.values()
        ),
        hysteresis_factor=0.0,
        server_quality={
            server_id: 1.0
            for server_id in simulator.servers
        },
    )
    if any(
        len(placement[server_id]) > capacities[server_id]
        for server_id in capacities
    ):
        raise RuntimeError(
            "Clairvoyant placement violated server capacity"
        )
    return placement


def scenario_capacity_aware_oracle_bounds(
    simulator,
    include_exogenous_waiting=False,
):
    """Return a clairvoyant, capacity-feasible optimistic reference."""
    placement = clairvoyant_capacity_placement(simulator)
    values = np.asarray(
        [
            relaxed_assignment_lower_bound(
                assignment_problem_from_simulator(
                    simulator,
                    user,
                    include_exogenous_waiting=(
                        include_exogenous_waiting
                    ),
                    service_placement=placement,
                )
            )
            for user in simulator.users.values()
        ],
        dtype=float,
    )
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "per_user": values.tolist(),
        "cache_placement": {
            str(server_id): list(services)
            for server_id, services in placement.items()
        },
        "server_capacities": {
            str(server_id): int(server.capacity)
            for server_id, server in simulator.servers.items()
        },
        "capacity_constraints_satisfied": all(
            len(placement[server_id]) <= server.capacity
            for server_id, server in simulator.servers.items()
        ),
        "future_workload_visible": True,
        "queue_coupling_relaxed": True,
    }


def scenario_exact_optimistic_bounds(
    simulator,
    time_limit_sec=30.0,
    include_exogenous_waiting=False,
):
    results = [
        exact_optimistic_assignment_oracle(
            assignment_problem_from_simulator(
                simulator,
                user,
                include_exogenous_waiting=(
                    include_exogenous_waiting
                ),
            ),
            time_limit_sec=time_limit_sec,
        )
        for user in simulator.users.values()
    ]
    values = np.asarray(
        [result.objective for result in results],
        dtype=float,
    )
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "wall_time_sec": float(
            sum(result.wall_time_sec for result in results)
        ),
        "per_user": values.tolist(),
    }
