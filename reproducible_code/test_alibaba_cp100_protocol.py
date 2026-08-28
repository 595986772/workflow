import copy
import io
import unittest

from alibaba_cp100_protocol import (
    BASELINE_RANDOM_DRAW_CAPACITY,
    CAPACITY_ASSIGNMENT_NAMESPACE,
    CAPACITY_MULTISET,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    EXPECTED_GRAPH_COUNT,
    TOTAL_CACHE_BUDGET,
    validate_protocol,
)
from input import INPUT_DICT, learning_arg
from run_independent_experiment import (
    base_scenario_fingerprint,
    scenario_snapshot,
    seed_everything,
)
from run_alibaba_cp100_experiment import merge_completed_stages
from simulator import (
    DEFAULT_DAG_DATASET_PATH,
    MEC_Simulator,
    load_dag_dataset,
)


def dataset_simulator(seed, algorithm="nearest_with_service"):
    seed_everything(seed)
    config = copy.deepcopy(INPUT_DICT)
    config.update(
        {
            "alg": algorithm,
            "Number of users": 3,
            "Number of servers": 10,
            "Number of services": 10,
            "Number of tasks for each user": 10,
            "dag dataset path": str(DATASET_PATH),
            "dag dataset sha256": EXPECTED_DATASET_SHA256,
            "server capacity": 2,
            "server capacity multiset": CAPACITY_MULTISET,
            "capacity assignment namespace": (
                CAPACITY_ASSIGNMENT_NAMESPACE
            ),
            "baseline server capacity": (
                BASELINE_RANDOM_DRAW_CAPACITY
            ),
            "save topology figure": False,
            "seed": seed,
        }
    )
    return MEC_Simulator(
        outputfile=io.StringIO(),
        Input_dict=config,
        learning_arguments=copy.deepcopy(learning_arg),
        filename_png="/tmp",
    )


class AlibabaCP100ProtocolTest(unittest.TestCase):
    def test_resume_state_accumulates_completed_stages(self):
        from pathlib import Path
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.json"
            state_path.write_text(
                json.dumps(
                    {"completed_stages": ["smoke", "converged"]}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                merge_completed_stages(state_path, ("tests",)),
                ["tests", "smoke", "converged"],
            )

    def test_protocol_preserves_original_budget_and_role(self):
        protocol = validate_protocol()
        self.assertEqual(sum(CAPACITY_MULTISET), TOTAL_CACHE_BUDGET)
        self.assertEqual(TOTAL_CACHE_BUDGET, 20)
        self.assertEqual(set(CAPACITY_MULTISET), {0, 1, 2, 3, 4})
        self.assertFalse(
            protocol["dataset"]["formal_unbiased_holdout"]
        )
        self.assertEqual(
            protocol["experiment"]["claim_scope"],
            "mechanism_stress_test_only",
        )

    def test_dataset_loader_validates_all_graphs(self):
        graphs, eligible, digest = load_dag_dataset(
            DATASET_PATH,
            number_of_tasks=10,
            number_of_services=10,
        )
        self.assertEqual(digest, EXPECTED_DATASET_SHA256)
        self.assertEqual(len(graphs), EXPECTED_GRAPH_COUNT)
        self.assertEqual(len(eligible), EXPECTED_GRAPH_COUNT)
        self.assertTrue(
            all(len(graph.nodes) <= 10 for graph in graphs.values())
        )

    def test_explicit_dataset_is_in_snapshot_and_pairs_methods(self):
        daoc = dataset_simulator(21)
        ours = dataset_simulator(21, "causal_telemetryPD3QN")
        self.assertEqual(
            base_scenario_fingerprint(daoc),
            base_scenario_fingerprint(ours),
        )
        snapshot = scenario_snapshot(ours)
        self.assertEqual(
            snapshot["dag_dataset"]["sha256"],
            EXPECTED_DATASET_SHA256,
        )
        self.assertEqual(
            snapshot["dag_dataset"]["eligible_graph_count"],
            EXPECTED_GRAPH_COUNT,
        )
        self.assertEqual(
            sorted(server.capacity for server in ours.servers.values()),
            CAPACITY_MULTISET,
        )

    def test_explicit_default_path_preserves_default_snapshot(self):
        seed_everything(4)
        implicit_config = copy.deepcopy(INPUT_DICT)
        implicit_config.update(
            {
                "alg": "nearest_with_service",
                "Number of users": 3,
                "Number of servers": 3,
                "Number of services": 3,
                "Number of tasks for each user": 6,
                "save topology figure": False,
            }
        )
        implicit = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=implicit_config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png="/tmp",
        )
        seed_everything(4)
        explicit_config = copy.deepcopy(implicit_config)
        explicit_config["dag dataset path"] = str(
            DEFAULT_DAG_DATASET_PATH
        )
        explicit = MEC_Simulator(
            outputfile=io.StringIO(),
            Input_dict=explicit_config,
            learning_arguments=copy.deepcopy(learning_arg),
            filename_png="/tmp",
        )
        self.assertNotIn("dag_dataset", scenario_snapshot(implicit))
        self.assertNotIn("dag_dataset", scenario_snapshot(explicit))
        self.assertEqual(
            base_scenario_fingerprint(implicit),
            base_scenario_fingerprint(explicit),
        )

    def test_wrong_dataset_hash_is_rejected(self):
        seed_everything(1)
        config = copy.deepcopy(INPUT_DICT)
        config.update(
            {
                "alg": "nearest_with_service",
                "Number of users": 1,
                "Number of servers": 10,
                "Number of services": 10,
                "Number of tasks for each user": 10,
                "dag dataset path": str(DATASET_PATH),
                "dag dataset sha256": "0" * 64,
                "save topology figure": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            MEC_Simulator(
                outputfile=io.StringIO(),
                Input_dict=config,
                learning_arguments=copy.deepcopy(learning_arg),
                filename_png="/tmp",
            )


if __name__ == "__main__":
    unittest.main()
