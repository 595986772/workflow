import math
import unittest
from types import SimpleNamespace

import numpy as np

from hindsight_critical_path import (
    bottleneck_contribution_scores,
    posterior_critical_path,
)


def make_completed_task(
    task_id,
    finish_time,
    assigned_server,
    predecessors=None,
    successors=None,
    outputs=None,
):
    return SimpleNamespace(
        task_number=task_id,
        predecessors=list(predecessors or []),
        successors=list(successors or []),
        outputs_length=dict(outputs or {}),
        assigned_server=assigned_server,
        done=True,
        result=SimpleNamespace(finish_time=finish_time),
    )


class PosteriorCriticalPathTest(unittest.TestCase):
    def make_diamond(self):
        return {
            "a": make_completed_task(
                "a",
                finish_time=2.0,
                assigned_server=0,
                successors=["b", "c"],
                outputs={"b": 0.0, "c": 0.0},
            ),
            "b": make_completed_task(
                "b",
                finish_time=5.0,
                assigned_server=0,
                predecessors=["a"],
                successors=["d"],
                outputs={"d": 0.0},
            ),
            "c": make_completed_task(
                "c",
                finish_time=3.0,
                assigned_server=1,
                predecessors=["a"],
                successors=["d"],
                outputs={"d": 0.0},
            ),
            "d": make_completed_task(
                "d",
                finish_time=7.0,
                assigned_server=0,
                predecessors=["b", "c"],
            ),
        }

    def test_backtracks_realized_critical_predecessors(self):
        posterior = posterior_critical_path(
            self.make_diamond(),
            between_server_costs=np.zeros((2, 2)),
            temperature=0.1,
        )

        self.assertEqual(posterior["path"], ("a", "b", "d"))
        self.assertEqual(posterior["makespan"], 7.0)
        self.assertEqual(posterior["scores"]["a"], 1.0)
        self.assertEqual(posterior["scores"]["b"], 1.0)
        self.assertEqual(posterior["scores"]["d"], 1.0)
        self.assertGreater(posterior["slack"]["c"], 0.0)
        self.assertLess(posterior["scores"]["c"], 1.0)
        self.assertTrue(
            math.isclose(
                posterior["scores"]["c"],
                math.exp(-2.0 / 0.7),
            )
        )

    def test_rejects_unfinished_dag(self):
        tasks = self.make_diamond()
        tasks["d"].done = False

        with self.assertRaisesRegex(
            ValueError,
            "completed tasks",
        ):
            posterior_critical_path(
                tasks,
                between_server_costs=np.zeros((2, 2)),
            )

    def test_accounts_for_realized_inter_server_transfer(self):
        tasks = {
            "a": make_completed_task(
                "a",
                finish_time=2.0,
                assigned_server=0,
                successors=["c"],
                outputs={"c": 2.0},
            ),
            "b": make_completed_task(
                "b",
                finish_time=3.0,
                assigned_server=1,
                successors=["c"],
                outputs={"c": 1.0},
            ),
            "c": make_completed_task(
                "c",
                finish_time=7.0,
                assigned_server=1,
                predecessors=["a", "b"],
            ),
        }
        costs = np.asarray([[0.0, 1.0], [1.0, 0.0]])

        posterior = posterior_critical_path(tasks, costs)

        self.assertEqual(posterior["path"], ("a", "c"))
        self.assertEqual(posterior["local_latency"]["c"], 3.0)

    def test_bottleneck_scores_keep_only_top_local_contributors(self):
        posterior = posterior_critical_path(
            self.make_diamond(),
            between_server_costs=np.zeros((2, 2)),
            temperature=0.1,
        )

        bottlenecks = bottleneck_contribution_scores(
            posterior,
            top_fraction=0.25,
        )

        self.assertEqual(bottlenecks["selected"], ("b",))
        self.assertEqual(bottlenecks["scores"]["b"], 1.0)
        self.assertEqual(bottlenecks["scores"]["a"], 0.0)
        self.assertEqual(bottlenecks["scores"]["c"], 0.0)
        self.assertEqual(bottlenecks["scores"]["d"], 0.0)
        self.assertEqual(bottlenecks["selected_fraction"], 0.25)

    def test_bottleneck_scores_reject_invalid_fraction(self):
        posterior = posterior_critical_path(
            self.make_diamond(),
            between_server_costs=np.zeros((2, 2)),
        )

        with self.assertRaisesRegex(ValueError, "top_fraction"):
            bottleneck_contribution_scores(
                posterior,
                top_fraction=0.0,
            )


if __name__ == "__main__":
    unittest.main()
