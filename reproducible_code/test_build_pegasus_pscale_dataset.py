import hashlib
import json
from pathlib import Path
import unittest

import networkx as nx

from build_pegasus_cross_dataset import (
    SOURCE_FILES,
    local_name,
    parse_dax,
    scale_log as legacy_scale_log,
)
from build_pegasus_pscale_dataset import (
    EXPECTED_PROGRAM_TYPES,
    TASK_LIMIT_INCLUDING_DUMMY,
    build_dataset,
)


class PegasusPScaleDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = (
            Path(__file__).resolve().parent
            / "datasets"
            / "pegasus_cross_topology"
            / "raw"
        )
        cls.dataset, cls.metadata = build_dataset(cls.raw_dir)

    def test_keeps_all_five_complete_workflows(self):
        self.assertEqual(len(self.dataset), 5)
        task_counts = {
            row["family"]: row["real_tasks"]
            for row in self.metadata["graphs"]
        }
        self.assertEqual(
            task_counts,
            {
                "Montage": 25,
                "CyberShake": 30,
                "Epigenomics": 24,
                "Inspiral": 30,
                "Sipht": 29,
            },
        )
        for graph_data in self.dataset.values():
            graph = nx.node_link_graph(graph_data)
            self.assertLessEqual(
                graph.number_of_nodes(),
                TASK_LIMIT_INCLUDING_DUMMY,
            )
            self.assertIn("0", graph)
            self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_fixed_mapping_covers_39_types_and_10_services(self):
        mapping = self.metadata["type_mapping"][
            "program_type_to_service"
        ]
        self.assertEqual(len(mapping), EXPECTED_PROGRAM_TYPES)
        self.assertEqual(set(mapping.values()), set(range(1, 11)))
        self.assertEqual(
            self.metadata["type_mapping"]["program_type_count"],
            EXPECTED_PROGRAM_TYPES,
        )

    def test_preserves_legacy_single_task_runtime_scaling(self):
        lower, upper = self.metadata["normalization"][
            "runtime_log_bounds"
        ]
        for family, source_path in SOURCE_FILES.items():
            _, jobs = parse_dax(
                self.raw_dir / local_name(source_path),
                family,
            )
            graph = nx.node_link_graph(
                self.dataset[f"pegasus_full_{family.lower()}"]
            )
            for node_id, attributes in graph.nodes(data=True):
                if node_id == "0":
                    continue
                expected = legacy_scale_log(
                    jobs[attributes["source_job_id"]]["runtime"],
                    lower,
                    upper,
                )
                self.assertEqual(attributes["cpucycle"], expected)

    def test_build_is_byte_deterministic(self):
        second_dataset, second_metadata = build_dataset(self.raw_dir)
        first = json.dumps(
            {"dataset": self.dataset, "metadata": self.metadata},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        second = json.dumps(
            {"dataset": second_dataset, "metadata": second_metadata},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(first).hexdigest(),
            hashlib.sha256(second).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
