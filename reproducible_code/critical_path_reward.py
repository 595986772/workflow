"""Objective-aligned reward shaping for partially executed task DAGs."""

import math


EPSILON = 1e-12
CAUSAL_MAKESPAN_REWARD_MODES = {
    "causal_critical_path",
    "causal_makespan_increment",
}


def compose_reward(
    base_reward,
    potential_reward,
    reward_mode,
    potential_weight=1.0,
):
    """Compose terminal and dense rewards without hiding the ablation."""
    if reward_mode == "critical_path_potential":
        return float(potential_reward)
    if reward_mode in CAUSAL_MAKESPAN_REWARD_MODES:
        return float(potential_reward)
    if reward_mode == "terminal_plus_potential":
        return (
            float(base_reward)
            + max(float(potential_weight), 0.0)
            * float(potential_reward)
        )
    if reward_mode == "terminal_binary":
        return float(base_reward)
    raise ValueError(f"Unknown reward mode: {reward_mode}")


class CausalCriticalPathReward:
    """Track the observed response-time frontier without future metadata.

    The reward at a decision is the negative increment of the latest observed
    task response time. Its undiscounted sum is exactly the negative
    application response makespan when the DAG has completed.
    """

    def __init__(self, deadline, application_origin=0.0):
        if deadline <= 0:
            raise ValueError("deadline must be positive")
        application_origin = float(application_origin)
        if not math.isfinite(application_origin) or application_origin < 0.0:
            raise ValueError(
                "application origin must be finite and non-negative"
            )
        self.deadline = float(deadline)
        self.application_origin = application_origin
        self.initial = 0.0
        self.current = 0.0

    def _response_time(self, finish_time):
        finish_time = float(finish_time)
        if not math.isfinite(finish_time):
            raise ValueError("finish time must be finite")
        response_time = finish_time - self.application_origin
        if response_time < -EPSILON:
            raise ValueError(
                "finish time must not precede the application origin"
            )
        return max(response_time, 0.0)

    def advance(self, done_tasks, terminal_finish_time=None):
        before = self.current
        observed_response = max(
            (
                self._response_time(task.result.finish_time)
                for task in done_tasks.values()
            ),
            default=0.0,
        )
        if terminal_finish_time is not None:
            observed_response = max(
                observed_response,
                self._response_time(terminal_finish_time),
            )
        after = max(before, observed_response / self.deadline)
        self.current = after
        return before - after, before, after

    def expected_return(self, finish_time):
        return -self._response_time(finish_time) / self.deadline


# The old name remains import-compatible for historical experiment configs.
CausalMakespanIncrementReward = CausalCriticalPathReward


class CriticalPathPotential:
    """Estimate the normalized DAG makespan as task outcomes are observed.

    Unfinished tasks use only static DAG metadata. Completed tasks replace the
    corresponding estimate with their observed finish time. The potential is
    one before the first task and equals the normalized application finish time
    at termination, so undiscounted shaped rewards telescope exactly.
    """

    def __init__(
        self,
        tasks,
        deadline,
        max_cpu_cycles,
        max_data_length,
    ):
        if not tasks:
            raise ValueError("tasks must not be empty")
        if deadline <= 0:
            raise ValueError("deadline must be positive")

        self.tasks = tasks
        self.deadline = float(deadline)
        self.cpu_scale = max(float(max_cpu_cycles) * 1e6, EPSILON)
        self.data_scale = max(float(max_data_length), EPSILON)
        self.order = self._topological_order()
        self.sinks = [
            task_id
            for task_id, task in tasks.items()
            if not any(
                successor_id in tasks
                for successor_id in task.successors
            )
        ]
        if not self.sinks:
            raise ValueError("the application DAG must contain a sink")

        self.own_work = {
            task_id: (
                float(task.cpu_cycle) / self.cpu_scale
                + float(task.input_data_length) / self.data_scale
            )
            for task_id, task in tasks.items()
        }
        self.edge_work = {
            (task_id, successor_id): (
                float(task.outputs_length.get(successor_id, 0.0))
                / self.data_scale
            )
            for task_id, task in tasks.items()
            for successor_id in task.successors
            if successor_id in tasks
        }
        self.nominal_makespan = self._nominal_makespan()
        self.current = self.estimate({})
        self.initial = self.current

    def _topological_order(self):
        indegree = {
            task_id: sum(
                predecessor_id in self.tasks
                for predecessor_id in task.predecessors
            )
            for task_id, task in self.tasks.items()
        }
        ready = [
            task_id
            for task_id in self.tasks
            if indegree[task_id] == 0
        ]
        order = []
        while ready:
            task_id = ready.pop(0)
            order.append(task_id)
            for successor_id in self.tasks[task_id].successors:
                if successor_id not in indegree:
                    continue
                indegree[successor_id] -= 1
                if indegree[successor_id] == 0:
                    ready.append(successor_id)
        if len(order) != len(self.tasks):
            raise ValueError("the application graph must be acyclic")
        return order

    def _nominal_makespan(self):
        finish = {}
        for task_id in self.order:
            task = self.tasks[task_id]
            predecessor_finish = max(
                (
                    finish[predecessor_id]
                    + self.edge_work.get(
                        (predecessor_id, task_id),
                        0.0,
                    )
                    for predecessor_id in task.predecessors
                    if predecessor_id in finish
                ),
                default=0.0,
            )
            finish[task_id] = predecessor_finish + self.own_work[task_id]
        return max(
            max((finish[task_id] for task_id in self.sinks), default=0.0),
            EPSILON,
        )

    def estimate(self, done_tasks):
        """Return the current normalized partial-DAG completion estimate."""
        finish = {}
        for task_id in self.order:
            if task_id in done_tasks:
                observed = float(
                    done_tasks[task_id].result.finish_time
                )
                if not math.isfinite(observed):
                    raise ValueError(
                        f"task {task_id!r} has no finite finish time"
                    )
                finish[task_id] = observed / self.deadline
                continue

            task = self.tasks[task_id]
            predecessor_finish = max(
                (
                    finish[predecessor_id]
                    + self.edge_work.get(
                        (predecessor_id, task_id),
                        0.0,
                    )
                    / self.nominal_makespan
                    for predecessor_id in task.predecessors
                    if predecessor_id in finish
                ),
                default=0.0,
            )
            finish[task_id] = (
                predecessor_finish
                + self.own_work[task_id] / self.nominal_makespan
            )

        return max(finish[task_id] for task_id in self.sinks)

    def advance(self, done_tasks, terminal_finish_time=None):
        """Observe completed tasks and return ``(reward, before, after)``."""
        before = self.current
        if terminal_finish_time is None:
            after = self.estimate(done_tasks)
        else:
            terminal_finish_time = float(terminal_finish_time)
            if not math.isfinite(terminal_finish_time):
                raise ValueError("terminal finish time must be finite")
            after = terminal_finish_time / self.deadline
        self.current = after
        return before - after, before, after

    def expected_return(self, finish_time):
        """Return the shaped episodic return implied by the true objective."""
        return self.initial - float(finish_time) / self.deadline
