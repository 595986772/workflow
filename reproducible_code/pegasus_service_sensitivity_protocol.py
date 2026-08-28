"""Frozen three-seed protocol for cache-pressure sensitivity experiments."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from run_reproduction_suite import ALGORITHMS


ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "pegasus_p14_service_sensitivity_v1"
RESULT_ROOT = ROOT / "results/pegasus_pscale/p14_service_sensitivity"
DATASET_DIR = ROOT / "datasets/pegasus_service_sensitivity"
BASE_DATASET_PATH = (
    ROOT / "datasets/pegasus_pscale/dag_pegasus5_full31.json"
)
BASE_DATASET_SHA256 = (
    "0671d8ea1ecdd8165062e19733e4859edb8e5ce87ecdd054bcef290abc49d5a5"
)

SEEDS = (51, 52, 53)
ACTIVE_SERVICE_COUNTS = (4, 6, 8, 10)
SERVICE_SIZE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
CAPACITY_MULTISET = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2)
CAPACITY_NAMESPACE = "pegasus_pscale_p2"
FAMILIES = (
    "CyberShake",
    "Epigenomics",
    "Inspiral",
    "Montage",
    "Sipht",
)
USERS = 20
SERVERS = 10
SERVICE_STATE_DIMENSION = 10
TASK_LIMIT_INCLUDING_DUMMY = 31
BANDWIDTH_HZ = 15_000
EVALUATION_EPISODES = 100

GREEDY = "greedy"
DAOC = "daoc_paper"
DQN_COORD_CACHE = "our_flat_ddqn"
COORD_SAC = "coord_cache_discrete_sac"
OUR = "lean_our"
METHODS = (GREEDY, DAOC, DQN_COORD_CACHE, COORD_SAC, OUR)
LEARNING_METHODS = (DAOC, DQN_COORD_CACHE, COORD_SAC, OUR)
DISPLAY_NAMES = {
    GREEDY: "Nearest+Service",
    DAOC: "DAOC",
    DQN_COORD_CACHE: "DQN+CoordCache",
    COORD_SAC: "CoordCache-DiscreteSAC",
    OUR: "OUR",
}
TRAINING_PROFILES = {
    GREEDY: "pegasus_p6_heuristics",
    DAOC: "pegasus_paper_closure_converged",
    DQN_COORD_CACHE: "pegasus_p6_learning_converged",
    COORD_SAC: "pegasus_baseline_sac_converged",
    OUR: "pegasus_paper_closure_converged",
}
SMOKE_PROFILES = {
    GREEDY: "pegasus_p6_smoke",
    DAOC: "pegasus_paper_closure_smoke",
    DQN_COORD_CACHE: "pegasus_p6_smoke",
    COORD_SAC: "pegasus_baseline_sac_smoke",
    OUR: "pegasus_pscale_p2_smoke",
}
Q10_SOURCE_ROOTS = {
    GREEDY: (
        ROOT
        / "results/pegasus_pscale/p6_baselines_ablation/heuristics/runs/greedy"
    ),
    DAOC: (
        ROOT
        / "results/pegasus_pscale/p3_paper_closure/final/runs/daoc_paper"
    ),
    DQN_COORD_CACHE: (
        ROOT
        / "results/pegasus_pscale/p6_baselines_ablation/learning/runs/our_flat_ddqn"
    ),
    COORD_SAC: (
        ROOT
        / "results/pegasus_pscale/p5_baseline_extension/sac_final/runs/coord_cache_discrete_sac"
    ),
    OUR: (
        ROOT
        / "results/pegasus_pscale/p3_paper_closure/final/runs/lean_our"
    ),
}

# Filled from deterministic projections of BASE_DATASET_PATH. Keeping these
# hashes in the protocol prevents an edited projection from entering a rerun.
EXPECTED_PROJECTED_SHA256 = {
    4: "30bf4786d86c2f76389f5e7c99c72d7739e2f4aaa3d43780b95747e24a31a583",
    6: "f21e58331568478f0cdfeec442dc675e6400384fdaeef27882e3381a78375a21",
    8: "924670197cf7dfc81f4d3b05146dc309987e62765200b25efc0e09d9dc10bde1",
    10: BASE_DATASET_SHA256,
}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decode_service(value: float) -> int:
    value = float(value)
    if value == 0.0:
        return 0
    service_id = int(round((value - 1.0) * 10.0 + 0.5))
    if not 1 <= service_id <= SERVICE_STATE_DIMENSION:
        raise ValueError(f"Invalid encoded service value: {value}")
    return service_id


def encode_service(service_id: int) -> float:
    if service_id == 0:
        return 0.0
    if not 1 <= service_id <= SERVICE_STATE_DIMENSION:
        raise ValueError(f"Invalid service id: {service_id}")
    return round(1.0 + (service_id - 0.5) / 10.0, 12)


def project_service_id(service_id: int, active_services: int) -> int:
    if service_id == 0:
        return 0
    if active_services not in ACTIVE_SERVICE_COUNTS:
        raise ValueError(f"Unsupported active service count: {active_services}")
    return 1 + (service_id - 1) % active_services


def project_dataset(dataset: dict, active_services: int) -> dict:
    projected = copy.deepcopy(dataset)
    for graph in projected.values():
        graph["graph"] = dict(graph.get("graph", {}))
        graph["graph"].update(
            {
                "service_pressure_projection": "modulo_balanced_v1",
                "source_service_state_dimension": SERVICE_STATE_DIMENSION,
                "active_services": active_services,
                "inactive_service_ids": list(
                    range(active_services + 1, SERVICE_STATE_DIMENSION + 1)
                ),
            }
        )
        for node in graph["nodes"]:
            original = decode_service(node["service"])
            node["service"] = encode_service(
                project_service_id(original, active_services)
            )
    return projected


def projected_dataset_path(active_services: int) -> Path:
    if active_services == SERVICE_STATE_DIMENSION:
        return BASE_DATASET_PATH
    return DATASET_DIR / f"dag_pegasus5_services_{active_services}.json"


def build_projected_datasets() -> dict[int, str]:
    if sha256_file(BASE_DATASET_PATH) != BASE_DATASET_SHA256:
        raise RuntimeError("Base Pegasus dataset hash mismatch")
    source = read_json(BASE_DATASET_PATH)
    hashes = {SERVICE_STATE_DIMENSION: BASE_DATASET_SHA256}
    for active_services in ACTIVE_SERVICE_COUNTS:
        if active_services == SERVICE_STATE_DIMENSION:
            continue
        path = projected_dataset_path(active_services)
        write_json(path, project_dataset(source, active_services))
        hashes[active_services] = sha256_file(path)
    return hashes


def q10_source_run(method: str, seed: int) -> Path:
    return Q10_SOURCE_ROOTS[method] / f"seed_{seed}"


def active_service_run(active_services: int, method: str, seed: int) -> Path:
    if active_services == SERVICE_STATE_DIMENSION:
        return q10_source_run(method, seed)
    return (
        RESULT_ROOT
        / "active_services"
        / f"q{active_services}"
        / method
        / "runs"
        / method
        / f"seed_{seed}"
    )


def algorithm_config(label: str) -> dict:
    return next(item for item in ALGORITHMS if item["label"] == label)


def validate_method_identity() -> None:
    ours = algorithm_config(OUR)
    dqn = algorithm_config(DQN_COORD_CACHE)
    sac = algorithm_config(COORD_SAC)
    shared = (
        "reward_mode",
        "cache_policy",
        "cache_coverage_constraint",
        "gamma",
        "n_step",
        "historical_feedback_guidance",
        "adaptive_guidance_gate",
    )
    for candidate in (dqn, sac):
        for key in shared:
            if candidate.get(key) != ours.get(key):
                raise RuntimeError(
                    f"{candidate['label']} changes shared field {key}"
                )
    if dqn["algorithm"] != "causal_telemetryDDQN":
        raise RuntimeError("DQN+CoordCache is not the flat Double DQN scheduler")
    if sac["algorithm"] != "causal_telemetryDiscreteSAC":
        raise RuntimeError("CoordCache-DiscreteSAC identity mismatch")


def validate_projected_dataset(active_services: int) -> dict:
    source = read_json(BASE_DATASET_PATH)
    path = projected_dataset_path(active_services)
    projected = read_json(path)
    observed = set()
    if source.keys() != projected.keys():
        raise RuntimeError("Projected dataset changed workflow keys")
    for key in source:
        before = source[key]
        after = projected[key]
        if before["links"] != after["links"]:
            raise RuntimeError(f"Projected dataset changed links for {key}")
        if len(before["nodes"]) != len(after["nodes"]):
            raise RuntimeError(f"Projected dataset changed node count for {key}")
        for source_node, target_node in zip(before["nodes"], after["nodes"]):
            source_rest = {
                name: value
                for name, value in source_node.items()
                if name != "service"
            }
            target_rest = {
                name: value
                for name, value in target_node.items()
                if name != "service"
            }
            if source_rest != target_rest:
                raise RuntimeError(f"Projected dataset changed task data for {key}")
            service_id = decode_service(target_node["service"])
            if service_id:
                observed.add(service_id)
    expected = set(range(1, active_services + 1))
    if observed != expected:
        raise RuntimeError(
            f"Projected dataset active services {observed} != {expected}"
        )
    return {
        "active_services": active_services,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "observed_services": sorted(observed),
    }


def validate_q10_sources() -> list[dict]:
    records = []
    for method in METHODS:
        for seed in SEEDS:
            run = q10_source_run(method, seed)
            summary = read_json(run / "summary.json")
            config = read_json(run / "config.json")["arguments"]
            complete = (
                summary.get("status") == "complete"
                and summary.get("evaluation_scenario_count")
                == EVALUATION_EPISODES
                and config.get("dag_dataset_sha256") == BASE_DATASET_SHA256
                and config.get("num_users") == USERS
                and config.get("num_servers") == SERVERS
                and config.get("num_services") == SERVICE_STATE_DIMENSION
                and config.get("num_tasks") == TASK_LIMIT_INCLUDING_DUMMY
                and float(config.get("bandwidth")) == BANDWIDTH_HZ
                and sorted(config.get("server_capacity_multiset", []))
                == sorted(CAPACITY_MULTISET)
            )
            if method in LEARNING_METHODS:
                complete = complete and bool(
                    summary.get("eligible_for_comparison")
                    and summary.get("convergence", {}).get("reached")
                )
            if not complete:
                raise RuntimeError(f"Invalid Q=10 source run: {run}")
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "run": str(run.resolve()),
                    "selected_checkpoint_sha256": summary.get(
                        "selected_checkpoint_sha256"
                    ),
                }
            )
    return records


def validate_protocol() -> dict:
    validate_method_identity()
    observed_hashes = build_projected_datasets()
    dataset_records = []
    for active_services in ACTIVE_SERVICE_COUNTS:
        record = validate_projected_dataset(active_services)
        expected_hash = EXPECTED_PROJECTED_SHA256[active_services]
        if expected_hash != "TO_BE_FILLED" and record["sha256"] != expected_hash:
            raise RuntimeError(
                f"Projected Q={active_services} dataset hash mismatch"
            )
        dataset_records.append(record)
    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "base_dataset_path": str(BASE_DATASET_PATH.resolve()),
        "base_dataset_sha256": BASE_DATASET_SHA256,
        "projected_dataset_sha256": {
            str(key): value for key, value in observed_hashes.items()
        },
        "active_service_counts": list(ACTIVE_SERVICE_COUNTS),
        "service_size_multipliers": list(SERVICE_SIZE_MULTIPLIERS),
        "service_projection": (
            "map original service q to 1 + (q - 1) mod Q_active; "
            "keep the network state dimension fixed at 10"
        ),
        "users": USERS,
        "servers": SERVERS,
        "service_state_dimension": SERVICE_STATE_DIMENSION,
        "task_limit_including_dummy": TASK_LIMIT_INCLUDING_DUMMY,
        "bandwidth_hz": BANDWIDTH_HZ,
        "capacity_multiset": list(CAPACITY_MULTISET),
        "capacity_assignment_namespace": CAPACITY_NAMESPACE,
        "workflow_families": list(FAMILIES),
        "methods": list(METHODS),
        "display_names": dict(DISPLAY_NAMES),
        "seeds": list(SEEDS),
        "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "q10_source_runs": validate_q10_sources(),
        "claim_scope": (
            "three_seed_control_sensitivity; descriptive trend evidence only"
        ),
        "method_identity": {
            DQN_COORD_CACHE: (
                "flat Double DQN scheduler with OUR's coordinated cache, "
                "causal state, reward and information protocol"
            ),
            COORD_SAC: (
                "Discrete SAC scheduler with the same coordinated cache, "
                "causal state, reward and information protocol"
            ),
        },
    }
    specification["specification_sha256"] = canonical_hash(specification)
    return specification
