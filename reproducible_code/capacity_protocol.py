"""Deterministic heterogeneous cache-capacity protocol for E2/E3."""

import hashlib
import random
from collections.abc import Mapping, Sequence


CAPACITY_PROTOCOL_VERSION = "heterogeneous_cache_capacity_v1"


def parse_capacity_multiset(value):
    """Parse a comma-separated non-negative integer multiset."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError("capacity multiset must be a string or sequence")
    if not items:
        raise ValueError("capacity multiset cannot be empty")
    try:
        capacities = [int(item) for item in items]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "capacity multiset must contain integers"
        ) from error
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("cache capacities must be non-negative")
    return capacities


def validate_capacity_multiset(
    capacities,
    number_of_servers,
    number_of_services,
):
    """Validate one capacity value per server."""
    capacities = parse_capacity_multiset(capacities)
    if len(capacities) != int(number_of_servers):
        raise ValueError(
            "capacity multiset length must equal the number of servers"
        )
    if any(
        capacity > int(number_of_services)
        for capacity in capacities
    ):
        raise ValueError(
            "cache capacity cannot exceed the number of services"
        )
    return tuple(capacities)


def _stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    )


def deterministic_capacity_assignment(
    capacities,
    number_of_servers,
    number_of_services,
    seed,
    assignment_namespace=None,
):
    """Shuffle capacities without consuming the simulator RNG stream."""
    assigned = list(
        validate_capacity_multiset(
            capacities,
            number_of_servers,
            number_of_services,
        )
    )
    if assignment_namespace is None:
        shuffle_seed = _stable_seed(
            CAPACITY_PROTOCOL_VERSION,
            int(seed),
            ",".join(
                str(value) for value in sorted(assigned)
            ),
        )
    else:
        # A shared namespace applies the same permutation to every
        # equal-length capacity profile. This permits paired
        # heterogeneity sweeps without coupling capacity to the
        # simulator's physical-deployment RNG.
        assigned.sort()
        shuffle_seed = _stable_seed(
            CAPACITY_PROTOCOL_VERSION,
            "shared-assignment",
            str(assignment_namespace),
            int(seed),
            int(number_of_servers),
        )
    rng = random.Random(shuffle_seed)
    rng.shuffle(assigned)
    return {
        server_id: assigned[server_id]
        for server_id in range(int(number_of_servers))
    }


def resolve_server_capacities(
    input_config,
    number_of_servers,
    number_of_services,
):
    """Resolve heterogeneous or legacy scalar server capacities."""
    multiset = input_config.get("server capacity multiset")
    if multiset is not None:
        return deterministic_capacity_assignment(
            capacities=multiset,
            number_of_servers=number_of_servers,
            number_of_services=number_of_services,
            seed=int(input_config.get("seed", 0)),
            assignment_namespace=input_config.get(
                "capacity assignment namespace"
            ),
        )

    scalar = int(input_config.get("server capacity", 2))
    if not 0 <= scalar <= int(number_of_services):
        raise ValueError(
            "server capacity must be between zero and "
            "the number of services"
        )
    return {
        server_id: scalar
        for server_id in range(int(number_of_servers))
    }


def normalize_capacity_mapping(capacity, server_ids):
    """Normalize a scalar, sequence, or mapping for cache optimizers."""
    server_ids = tuple(sorted(int(server_id) for server_id in server_ids))
    if isinstance(capacity, Mapping):
        normalized_input = {
            int(server_id): value
            for server_id, value in capacity.items()
        }
        if set(normalized_input) != set(server_ids):
            raise ValueError(
                "capacity mapping must contain every server exactly once"
            )
        values = {
            server_id: int(normalized_input[server_id])
            for server_id in server_ids
        }
    elif isinstance(capacity, Sequence) and not isinstance(
        capacity,
        (str, bytes),
    ):
        if len(capacity) != len(server_ids):
            raise ValueError(
                "capacity sequence length must match server count"
            )
        values = {
            server_id: int(value)
            for server_id, value in zip(server_ids, capacity)
        }
    else:
        values = {
            server_id: int(capacity)
            for server_id in server_ids
        }
    if any(value < 0 for value in values.values()):
        raise ValueError("cache capacities must be non-negative")
    return values


def select_load_shift_servers(capacities, seed):
    """Select one server from each capacity class deterministically."""
    capacity_map = normalize_capacity_mapping(
        capacities,
        capacities.keys(),
    )
    selected = {}
    for capacity in sorted(set(capacity_map.values())):
        candidates = sorted(
            server_id
            for server_id, value in capacity_map.items()
            if value == capacity
        )
        rng = random.Random(
            _stable_seed(
                CAPACITY_PROTOCOL_VERSION,
                "load-shift",
                int(seed),
                capacity,
            )
        )
        selected[capacity] = candidates[
            rng.randrange(len(candidates))
        ]
    return selected
