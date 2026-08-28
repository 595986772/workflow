"""Pure helpers for critical-path-aware hysteretic service caching."""

import numpy as np

from capacity_protocol import normalize_capacity_mapping


EPSILON = 1e-12


def exponential_moving_average(previous, observation, alpha):
    """Update an EMA, initializing it from the first causal observation."""
    alpha = float(alpha)
    observation = float(observation)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not np.isfinite(observation):
        raise ValueError("observation must be finite")
    if previous is None:
        return observation
    previous = float(previous)
    if not np.isfinite(previous):
        raise ValueError("previous EMA must be finite")
    return previous + alpha * (observation - previous)


def history_only_server_quality(
    server_execution_latency_ema,
    global_execution_latency_ema,
    compute_weight,
    compute_intensity=1.0,
):
    """Convert causal latency EMAs into relative cache-placement quality."""
    server_ids = sorted(server_execution_latency_ema)
    if not server_ids:
        return {}

    observed_latencies = [
        float(latency)
        for latency in server_execution_latency_ema.values()
        if (
            latency is not None
            and np.isfinite(float(latency))
            and float(latency) > 0.0
        )
    ]
    if not observed_latencies:
        return {server_id: 1.0 for server_id in server_ids}

    if (
        global_execution_latency_ema is not None
        and np.isfinite(float(global_execution_latency_ema))
        and float(global_execution_latency_ema) > 0.0
    ):
        fallback_latency = float(global_execution_latency_ema)
    else:
        fallback_latency = float(np.mean(observed_latencies))

    costs = {
        server_id: (
            float(latency)
            if (
                latency is not None
                and np.isfinite(float(latency))
                and float(latency) > 0.0
            )
            else fallback_latency
        )
        for server_id, latency
        in server_execution_latency_ema.items()
    }
    exponent = (
        max(float(compute_weight), 0.0)
        * min(max(float(compute_intensity), 0.0), 1.0)
    )
    if exponent == 0.0:
        return {server_id: 1.0 for server_id in server_ids}

    best_cost = min(costs.values())
    return {
        server_id: (
            best_cost / max(cost, EPSILON)
        ) ** exponent
        for server_id, cost in costs.items()
    }


def workload_normalized_server_telemetry(
    server_compute_per_mcycle_ema,
    server_waiting_latency_ema,
    global_compute_per_mcycle_ema,
    global_waiting_latency_ema,
    server_sample_counts,
    server_last_observed_window,
    current_window,
    min_samples=5,
    freshness_half_life=10.0,
):
    """Return causal compute, waiting, and confidence features per server."""
    server_ids = sorted(server_compute_per_mcycle_ema)
    if set(server_waiting_latency_ema) != set(server_ids):
        raise ValueError("telemetry dictionaries must share server ids")
    if int(min_samples) < 1:
        raise ValueError("min_samples must be positive")
    if float(freshness_half_life) <= 0.0:
        raise ValueError("freshness_half_life must be positive")

    def valid_cost(value):
        return (
            value is not None
            and np.isfinite(float(value))
            and float(value) >= 0.0
        )

    global_compute = (
        float(global_compute_per_mcycle_ema)
        if valid_cost(global_compute_per_mcycle_ema)
        else None
    )
    global_waiting = (
        float(global_waiting_latency_ema)
        if valid_cost(global_waiting_latency_ema)
        else None
    )
    telemetry = {}
    for server_id in server_ids:
        samples = max(
            int(server_sample_counts.get(server_id, 0)),
            0,
        )
        last_window = server_last_observed_window.get(server_id)
        if last_window is None:
            age = max(int(current_window), 0)
        else:
            age = max(
                int(current_window) - int(last_window),
                0,
            )
        sample_confidence = samples / (samples + int(min_samples))
        freshness = 0.5 ** (
            age / float(freshness_half_life)
        )
        confidence = float(
            np.clip(sample_confidence * freshness, 0.0, 1.0)
        )

        server_compute = server_compute_per_mcycle_ema[server_id]
        if not valid_cost(server_compute):
            server_compute = global_compute
            confidence = 0.0
        server_waiting = server_waiting_latency_ema[server_id]
        if not valid_cost(server_waiting):
            server_waiting = global_waiting
            confidence = 0.0

        def relative_quality(server_cost, global_cost):
            if global_cost is None or server_cost is None:
                return 1.0
            blended_cost = (
                confidence * float(server_cost)
                + (1.0 - confidence) * float(global_cost)
            )
            if float(global_cost) <= EPSILON:
                return 1.0 if blended_cost <= EPSILON else 0.0
            return float(
                np.clip(
                    float(global_cost)
                    / max(float(global_cost), blended_cost),
                    0.0,
                    1.0,
                )
            )

        telemetry[server_id] = (
            relative_quality(server_compute, global_compute),
            relative_quality(server_waiting, global_waiting),
            confidence,
        )
    return telemetry


def dependency_locality_bonuses(
    task,
    done_tasks,
    number_of_servers,
    between_server_costs,
):
    """Return candidate-specific savings in predecessor transfer delay."""
    if not task.predecessors:
        return np.zeros(number_of_servers, dtype=float)

    latest_predecessor_finish = max(
        done_tasks[predecessor_id].result.finish_time
        for predecessor_id in task.predecessors
    )
    communication_costs = np.zeros(number_of_servers, dtype=float)
    for server_id in range(number_of_servers):
        candidate_finish = max(
            (
                done_tasks[predecessor_id].result.finish_time
                + between_server_costs[
                    done_tasks[predecessor_id].assigned_server,
                    server_id,
                ]
                * done_tasks[predecessor_id].outputs_length.get(
                    task.task_number,
                    0.0,
                )
            )
            for predecessor_id in task.predecessors
        )
        communication_costs[server_id] = max(
            0.0,
            candidate_finish - latest_predecessor_finish,
        )

    return communication_costs.max() - communication_costs


def alternative_service_fetch_delays(
    service,
    service_size,
    servers,
    between_server_costs,
):
    """Estimate the loading delay avoided by caching at each server."""
    delays = np.zeros(len(servers), dtype=float)
    for server_id, server in servers.items():
        source_costs = [1.0 / server.rate_to_cloud]
        source_costs.extend(
            between_server_costs[other_id, server_id]
            for other_id, other_server in servers.items()
            if (
                other_id != server_id
                and service in other_server.services
            )
        )
        delays[server_id] = float(service_size) * min(source_costs)
    return delays


def critical_service_values(
    criticality,
    fetch_delays,
    locality_bonuses,
    locality_weight,
):
    """Combine loading and dependency-locality savings in seconds."""
    return max(float(criticality), 0.0) * (
        np.asarray(fetch_delays, dtype=float)
        + max(float(locality_weight), 0.0)
        * np.asarray(locality_bonuses, dtype=float)
    )


def service_fetch_savings(
    assignments,
    demand,
    service_sizes,
    cloud_costs,
    between_server_costs,
):
    """Return criticality-weighted service-fetch delay saved by a placement."""
    service_locations = {}
    for server_id, services in assignments.items():
        for service_id in services:
            service_locations.setdefault(service_id, []).append(server_id)

    savings = 0.0
    for destination, service_demand in demand.items():
        cloud_cost = float(cloud_costs[destination])
        for service_id, weight in service_demand.items():
            if weight <= 0:
                continue
            locations = service_locations.get(service_id, ())
            fetch_cost = min(
                [cloud_cost]
                + [
                    float(
                        between_server_costs[
                            source,
                            destination,
                        ]
                    )
                    for source in locations
                ]
            )
            savings += (
                float(weight)
                * float(service_sizes[service_id])
                * max(0.0, cloud_cost - fetch_cost)
            )
    return savings


def repair_redundant_replicas_for_coverage(
    assignments,
    demand,
    service_sizes,
    cloud_costs,
    between_server_costs,
    server_quality,
    locked_servers=(),
):
    """Reach maximal observed-service coverage with minimal slot changes."""
    proposal = {
        server_id: list(services)
        for server_id, services in assignments.items()
    }
    locked_servers = set(locked_servers)
    demanded_services = {
        service_id
        for service_id in service_sizes
        if sum(
            max(float(server_demand.get(service_id, 0.0)), 0.0)
            for server_demand in demand.values()
        )
        > EPSILON
    }
    target_slots = sum(len(services) for services in proposal.values())
    coverage_target = min(target_slots, len(demanded_services))

    while True:
        replica_counts = {
            service_id: sum(
                service_id in services
                for services in proposal.values()
            )
            for service_id in service_sizes
        }
        covered = {
            service_id
            for service_id, count in replica_counts.items()
            if count > 0 and service_id in demanded_services
        }
        if len(covered) >= coverage_target:
            break
        uncovered = demanded_services - covered
        base_value = service_fetch_savings(
            assignments=proposal,
            demand=demand,
            service_sizes=service_sizes,
            cloud_costs=cloud_costs,
            between_server_costs=between_server_costs,
        )
        candidates = []
        for server_id, services in proposal.items():
            if server_id in locked_servers:
                continue
            for old_service in services:
                removable = (
                    replica_counts[old_service] > 1
                    or old_service not in demanded_services
                )
                if not removable:
                    continue
                for new_service in sorted(uncovered):
                    candidate = {
                        key: list(value)
                        for key, value in proposal.items()
                    }
                    candidate[server_id].remove(old_service)
                    candidate[server_id].append(new_service)
                    value = service_fetch_savings(
                        assignments=candidate,
                        demand=demand,
                        service_sizes=service_sizes,
                        cloud_costs=cloud_costs,
                        between_server_costs=between_server_costs,
                    )
                    candidates.append(
                        (
                            (value - base_value)
                            * max(float(server_quality[server_id]), 0.0),
                            -server_id,
                            -old_service,
                            -new_service,
                            server_id,
                            old_service,
                            new_service,
                        )
                    )
        if not candidates:
            break
        *_, server_id, old_service, new_service = max(candidates)
        proposal[server_id].remove(old_service)
        proposal[server_id].append(new_service)
    return proposal


def coordinated_cache_decision(
    demand,
    current_services,
    capacity,
    service_sizes,
    cloud_costs,
    between_server_costs,
    expected_requests,
    hysteresis_factor,
    locked_servers=(),
    server_quality=None,
    replica_diversity_regularization=False,
    coverage_constraint=False,
):
    """Greedily place replicas by global marginal gain with a switch gate."""
    server_ids = sorted(current_services)
    service_ids = sorted(service_sizes)
    capacities = normalize_capacity_mapping(
        capacity,
        server_ids,
    )
    locked_servers = set(locked_servers)
    if server_quality is None:
        server_quality = {
            server_id: 1.0 for server_id in server_ids
        }
    current = {
        server_id: [
            service_id
            for service_id in current_services[server_id]
            if service_id in service_sizes
        ][:capacities[server_id]]
        for server_id in server_ids
    }
    proposal = {
        server_id: (
            list(current[server_id])
            if server_id in locked_servers
            else []
        )
        for server_id in server_ids
    }

    target_slots = sum(capacities.values())
    while sum(len(services) for services in proposal.values()) < target_slots:
        base_value = service_fetch_savings(
            assignments=proposal,
            demand=demand,
            service_sizes=service_sizes,
            cloud_costs=cloud_costs,
            between_server_costs=between_server_costs,
        )
        candidates = []
        for server_id in server_ids:
            if len(proposal[server_id]) >= capacities[server_id]:
                continue
            for service_id in service_ids:
                if service_id in proposal[server_id]:
                    continue
                candidate = {
                    key: list(value)
                    for key, value in proposal.items()
                }
                candidate[server_id].append(service_id)
                gain = service_fetch_savings(
                    assignments=candidate,
                    demand=demand,
                    service_sizes=service_sizes,
                    cloud_costs=cloud_costs,
                    between_server_costs=between_server_costs,
                ) - base_value
                if replica_diversity_regularization:
                    replica_count = sum(
                        service_id in services
                        for services in proposal.values()
                    )
                    gain /= 1.0 + replica_count
                candidates.append(
                    (
                        gain
                        * max(
                            float(server_quality[server_id]),
                            0.0,
                        ),
                        int(service_id in current[server_id]),
                        -server_id,
                        -service_id,
                        server_id,
                        service_id,
                    )
                )
        if not candidates:
            break
        *_, server_id, service_id = max(candidates)
        proposal[server_id].append(service_id)

    if coverage_constraint:
        proposal = repair_redundant_replicas_for_coverage(
            assignments=proposal,
            demand=demand,
            service_sizes=service_sizes,
            cloud_costs=cloud_costs,
            between_server_costs=between_server_costs,
            server_quality=server_quality,
            locked_servers=locked_servers,
        )

    current_value = service_fetch_savings(
        assignments=current,
        demand=demand,
        service_sizes=service_sizes,
        cloud_costs=cloud_costs,
        between_server_costs=between_server_costs,
    )
    proposed_value = service_fetch_savings(
        assignments=proposal,
        demand=demand,
        service_sizes=service_sizes,
        cloud_costs=cloud_costs,
        between_server_costs=between_server_costs,
    )
    switching_cost = 0.0
    for server_id in server_ids:
        current_locations = {
            service_id: [
                source_id
                for source_id, services in current.items()
                if service_id in services
            ]
            for service_id in service_ids
        }
        for service_id in set(proposal[server_id]) - set(current[server_id]):
            source_costs = [float(cloud_costs[server_id])]
            source_costs.extend(
                float(between_server_costs[source_id, server_id])
                for source_id in current_locations[service_id]
            )
            switching_cost += (
                float(service_sizes[service_id])
                * min(source_costs)
            )

    projected_gain = max(float(expected_requests), 1.0) * (
        proposed_value - current_value
    )
    required_gain = (
        max(float(hysteresis_factor), 0.0)
        * switching_cost
    )
    demanded_services = {
        service_id
        for service_id in service_ids
        if sum(
            max(float(server_demand.get(service_id, 0.0)), 0.0)
            for server_demand in demand.values()
        )
        > EPSILON
    }
    current_coverage = len(
        demanded_services
        & {
            service_id
            for services in current.values()
            for service_id in services
        }
    )
    proposed_coverage = len(
        demanded_services
        & {
            service_id
            for services in proposal.values()
            for service_id in services
        }
    )
    if coverage_constraint and proposed_coverage > current_coverage:
        return proposal
    if projected_gain > required_gain + EPSILON:
        return proposal
    return current


def hysteretic_cache_decision(
    scores,
    current_services,
    capacity,
    switching_costs,
    expected_requests,
    hysteresis_factor,
):
    """Return a top-k cache only when projected gain pays switching cost."""
    if capacity < 1:
        return []

    service_ids = sorted(
        scores,
        key=lambda service_id: (
            -float(scores[service_id]),
            service_id,
        ),
    )
    proposed = service_ids[:capacity]
    current = [
        service_id
        for service_id in current_services
        if service_id in scores
    ][:capacity]
    if len(current) < capacity:
        current.extend(
            service_id
            for service_id in proposed
            if service_id not in current
        )
        current = current[:capacity]
    if set(proposed) == set(current):
        return current

    score_gain = sum(scores[q] for q in proposed) - sum(
        scores[q] for q in current
    )
    incoming = set(proposed) - set(current)
    switching_cost = sum(
        float(switching_costs.get(service_id, 0.0))
        for service_id in incoming
    )
    projected_gain = max(float(expected_requests), 1.0) * score_gain
    required_gain = (
        max(float(hysteresis_factor), 0.0) * switching_cost
    )
    if projected_gain > required_gain + EPSILON:
        return proposed
    return current
