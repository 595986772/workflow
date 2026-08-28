import copy
import unittest
from types import SimpleNamespace

import numpy as np

from sample_efficient_guidance import (
    HistoricalFeedbackGuide,
    adaptive_guidance_probability,
    compute_nominal_upward_ranks,
    q_margin_confidence,
)


def make_task(
    task_id,
    cpu_cycle,
    input_data=0.0,
    predecessors=None,
    successors=None,
    outputs=None,
):
    return SimpleNamespace(
        task_number=task_id,
        cpu_cycle=cpu_cycle,
        input_data_length=input_data,
        predecessors=list(predecessors or []),
        successors=list(successors or []),
        outputs_length=dict(outputs or {}),
        result=SimpleNamespace(finish_time=float("inf")),
    )


class SampleEfficientGuidanceTest(unittest.TestCase):
    def setUp(self):
        self.tasks = {
            "1": make_task(
                "1",
                50e6,
                input_data=20.0,
                successors=["2"],
                outputs={"2": 30.0},
            ),
            "2": make_task("2", 25e6),
        }
        self.ranks, self.maximum_rank = compute_nominal_upward_ranks(
            self.tasks,
            max_cpu_cycles=100,
            max_data_length=100,
        )

    def test_nominal_rank_follows_dag_direction(self):
        self.assertGreater(self.ranks["1"], self.ranks["2"])
        self.assertAlmostEqual(self.maximum_rank, self.ranks["1"])

    def test_adaptive_gate_prefers_critical_low_confidence_tasks(self):
        uncertain_critical = adaptive_guidance_probability(
            q_confidence=0.1,
            task_criticality=1.0,
            expert_confidence=1.0,
            maximum_probability=0.9,
        )
        confident_noncritical = adaptive_guidance_probability(
            q_confidence=0.8,
            task_criticality=0.2,
            expert_confidence=1.0,
            maximum_probability=0.9,
        )
        self.assertGreater(uncertain_critical, confident_noncritical)
        handed_off = adaptive_guidance_probability(
            q_confidence=0.1,
            task_criticality=1.0,
            expert_confidence=1.0,
            maximum_probability=0.9,
            handoff_factor=0.0,
        )
        self.assertEqual(handed_off, 0.0)

    def test_history_guide_learns_from_completed_samples(self):
        guide = HistoricalFeedbackGuide(
            number_of_servers=3,
            alpha=0.5,
            min_samples=2,
        )
        for _ in range(3):
            guide.update(
                server_id=0,
                cache_hit=True,
                normalized_cpu=0.5,
                normalized_input=0.2,
                normalized_remote_data=0.0,
                observed_path_delay=0.1,
            )
            guide.update(
                server_id=1,
                cache_hit=True,
                normalized_cpu=0.5,
                normalized_input=0.2,
                normalized_remote_data=0.0,
                observed_path_delay=0.5,
            )

        before = copy.deepcopy(guide.state_dict())
        recommendation = guide.recommend(
            cache_hits=np.ones(3, dtype=bool),
            normalized_cpu=0.5,
            normalized_input=0.2,
            remote_data_by_server=np.zeros(3),
        )
        self.assertTrue(recommendation["ready"])
        self.assertEqual(recommendation["action"], 0)
        self.assertEqual(before, guide.state_dict())

    def test_q_margin_confidence_increases_with_margin(self):
        low = q_margin_confidence([0.2, 0.19, 0.1])
        high = q_margin_confidence([0.9, 0.1, 0.0])
        self.assertGreater(high, low)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
