import unittest
from types import SimpleNamespace

from critical_path_reward import (
    CausalMakespanIncrementReward,
    CausalCriticalPathReward,
    CriticalPathPotential,
    compose_reward,
)


def make_task(
    task_id,
    cpu_cycle,
    predecessors=None,
    successors=None,
    outputs=None,
):
    return SimpleNamespace(
        task_number=task_id,
        cpu_cycle=cpu_cycle,
        input_data_length=0.0,
        predecessors=list(predecessors or []),
        successors=list(successors or []),
        outputs_length=dict(outputs or {}),
        result=SimpleNamespace(finish_time=float("inf")),
    )


class CriticalPathPotentialTest(unittest.TestCase):
    def make_chain(self):
        return {
            "1": make_task(
                "1",
                50e6,
                successors=["2"],
                outputs={"2": 20.0},
            ),
            "2": make_task(
                "2",
                50e6,
                predecessors=["1"],
            ),
        }

    def test_initial_potential_is_normalized(self):
        tracker = CriticalPathPotential(
            self.make_chain(),
            deadline=1.0,
            max_cpu_cycles=100,
            max_data_length=100,
        )
        self.assertAlmostEqual(tracker.initial, 1.0)

    def test_rewards_telescope_to_negative_finish_time(self):
        tasks = self.make_chain()
        tracker = CriticalPathPotential(
            tasks,
            deadline=1.0,
            max_cpu_cycles=100,
            max_data_length=100,
        )

        tasks["1"].result.finish_time = 0.4
        reward_1, _, _ = tracker.advance({"1": tasks["1"]})
        tasks["2"].result.finish_time = 0.75
        reward_2, _, final = tracker.advance(
            tasks,
            terminal_finish_time=0.75,
        )

        self.assertAlmostEqual(final, 0.75)
        self.assertAlmostEqual(
            reward_1 + reward_2,
            tracker.expected_return(0.75),
        )

    def test_observed_slow_critical_task_reduces_reward(self):
        tasks = self.make_chain()
        tracker = CriticalPathPotential(
            tasks,
            deadline=1.0,
            max_cpu_cycles=100,
            max_data_length=100,
        )
        tasks["1"].result.finish_time = 2.0
        reward, before, after = tracker.advance({"1": tasks["1"]})
        self.assertGreater(after, before)
        self.assertLess(reward, 0.0)

    def test_cycle_is_rejected(self):
        tasks = {
            "1": make_task(
                "1",
                10e6,
                predecessors=["2"],
                successors=["2"],
            ),
            "2": make_task(
                "2",
                10e6,
                predecessors=["1"],
                successors=["1"],
            ),
        }
        with self.assertRaises(ValueError):
            CriticalPathPotential(
                tasks,
                deadline=1.0,
                max_cpu_cycles=100,
                max_data_length=100,
            )

    def test_hybrid_reward_preserves_terminal_signal(self):
        self.assertAlmostEqual(
            compose_reward(
                base_reward=1.0,
                potential_reward=0.4,
                reward_mode="terminal_plus_potential",
                potential_weight=0.5,
            ),
            1.2,
        )
        self.assertAlmostEqual(
            compose_reward(
                base_reward=-1.0,
                potential_reward=0.4,
                reward_mode="terminal_plus_potential",
                potential_weight=0.5,
            ),
            -0.8,
        )


class CausalCriticalPathRewardTest(unittest.TestCase):
    def test_rewards_telescope_to_negative_observed_makespan(self):
        tracker = CausalCriticalPathReward(deadline=2.0)
        first = make_task("1", 10e6)
        second = make_task("2", 900e6)
        first.result.finish_time = 0.6
        second.result.finish_time = 1.4

        reward_1, _, _ = tracker.advance({"1": first})
        reward_2, _, final = tracker.advance(
            {"1": first, "2": second},
            terminal_finish_time=1.4,
        )

        self.assertAlmostEqual(final, 0.7)
        self.assertAlmostEqual(
            reward_1 + reward_2,
            tracker.expected_return(1.4),
        )

    def test_nonzero_origin_telescopes_to_negative_response_time(self):
        tracker = CausalCriticalPathReward(
            deadline=2.0,
            application_origin=5.0,
        )
        first = make_task("1", 10e6)
        second = make_task("2", 900e6)
        first.result.finish_time = 5.6
        second.result.finish_time = 6.4

        reward_1, _, _ = tracker.advance({"1": first})
        reward_2, _, final = tracker.advance(
            {"1": first, "2": second},
            terminal_finish_time=6.4,
        )

        self.assertAlmostEqual(final, 0.7)
        self.assertAlmostEqual(reward_1 + reward_2, -0.7)
        self.assertAlmostEqual(
            reward_1 + reward_2,
            tracker.expected_return(6.4),
        )

    def test_reward_is_invariant_to_a_common_time_shift(self):
        base_task = make_task("1", 10e6)
        shifted_task = make_task("1", 10e6)
        base_task.result.finish_time = 0.8
        shifted_task.result.finish_time = 7.8

        base_reward, _, base_final = CausalCriticalPathReward(
            deadline=2.0,
        ).advance({"1": base_task})
        shifted_reward, _, shifted_final = CausalCriticalPathReward(
            deadline=2.0,
            application_origin=7.0,
        ).advance({"1": shifted_task})

        self.assertAlmostEqual(base_reward, shifted_reward)
        self.assertAlmostEqual(base_final, shifted_final)

    def test_finish_before_application_origin_is_rejected(self):
        tracker = CausalCriticalPathReward(
            deadline=1.0,
            application_origin=2.0,
        )
        task = make_task("1", 10e6)
        task.result.finish_time = 1.9

        with self.assertRaises(ValueError):
            tracker.advance({"1": task})

    def test_reward_does_not_read_unfinished_task_metadata(self):
        observed = make_task("1", 10e6)
        observed.result.finish_time = 0.8
        future_light = make_task("2", 1.0)
        future_heavy = make_task("2", 1e15)
        future_light.input_data_length = 0.0
        future_heavy.input_data_length = 1e15

        tracker_a = CausalCriticalPathReward(deadline=1.0)
        tracker_b = CausalCriticalPathReward(deadline=1.0)
        reward_a, _, _ = tracker_a.advance({"1": observed})
        reward_b, _, _ = tracker_b.advance({"1": observed})

        self.assertAlmostEqual(reward_a, reward_b)
        self.assertNotEqual(
            future_light.cpu_cycle,
            future_heavy.cpu_cycle,
        )

    def test_compose_reward_accepts_causal_mode(self):
        self.assertAlmostEqual(
            compose_reward(
                base_reward=1.0,
                potential_reward=-0.3,
                reward_mode="causal_critical_path",
            ),
            -0.3,
        )

    def test_lean_reward_alias_has_identical_objective(self):
        old_tracker = CausalCriticalPathReward(deadline=2.0)
        lean_tracker = CausalMakespanIncrementReward(deadline=2.0)
        task = make_task("1", 10e6)
        task.result.finish_time = 1.2

        old_reward, _, _ = old_tracker.advance(
            {"1": task},
            terminal_finish_time=1.2,
        )
        lean_reward, _, _ = lean_tracker.advance(
            {"1": task},
            terminal_finish_time=1.2,
        )

        self.assertAlmostEqual(old_reward, lean_reward)
        self.assertAlmostEqual(
            compose_reward(
                base_reward=1.0,
                potential_reward=lean_reward,
                reward_mode="causal_makespan_increment",
            ),
            lean_reward,
        )


if __name__ == "__main__":
    unittest.main()
