from types import SimpleNamespace

import numpy as np

from regenerate_critical_path_latency_figure import (
    COMPONENT_FIELDS,
    extract_critical_path_components,
)


def result(pred, data, service, waiting, computation, finish):
    return SimpleNamespace(
        pred_latency=pred,
        data_transfer_latency=data,
        service_latency=service,
        waiting_latency=waiting,
        computing_latency=computation,
        finish_time=finish,
    )


def task(task_id, predecessors, server, outputs, values):
    return SimpleNamespace(
        task_number=task_id,
        predecessors=predecessors,
        assigned_server=server,
        outputs_length=outputs,
        result=values,
    )


def test_realized_branch_uses_only_finish_determining_predecessor():
    # Task 1 finishes later in isolation, but task 2's transfer makes it the
    # predecessor that determines task 3's ready frontier.
    tasks = {
        "1": task("1", [], 0, {"3": 2.0}, result(0, 0.1, 0.2, 0.1, 0.6, 1.0)),
        "2": task("2", [], 1, {"3": 5.0}, result(0, 0.2, 0.1, 0.1, 0.4, 0.8)),
        "3": task("3", ["1", "2"], 0, {}, result(1.3, 0.0, 0.2, 0.1, 0.4, 2.0)),
    }
    costs = np.asarray([[0.0, 0.0], [0.1, 0.0]])
    user = SimpleNamespace(
        id=0,
        done_tasks=tasks,
        tasks_init=tasks,
        task_completion_counts={key: 1 for key in tasks},
        exit_task_ids=("3",),
        finish_time_of_application=2.0,
        arrival_time=0.0,
    )
    simulator = SimpleNamespace(dynamic_queueing=False, between_server_costs=costs)
    trace = extract_critical_path_components(user, simulator)
    assert trace["critical_path_task_ids"] == "2->3"
    assert np.isclose(trace["dependency_transfer_s"], 0.5)
    assert np.isclose(sum(trace[field] for field in COMPONENT_FIELDS), 2.0)
    assert abs(trace["identity_residual_s"]) < 1e-12


def test_multiple_exits_selects_latest_completion():
    tasks = {
        "1": task("1", [], 0, {}, result(0, 0.1, 0.1, 0.1, 0.7, 1.0)),
        "2": task("2", [], 0, {}, result(0, 0.2, 0.1, 0.1, 0.8, 1.2)),
    }
    user = SimpleNamespace(
        id=0,
        done_tasks=tasks,
        tasks_init=tasks,
        task_completion_counts={"1": 1, "2": 1},
        exit_task_ids=("1", "2"),
        finish_time_of_application=1.2,
        arrival_time=0.0,
    )
    simulator = SimpleNamespace(
        dynamic_queueing=False,
        between_server_costs=np.zeros((1, 1)),
    )
    trace = extract_critical_path_components(user, simulator)
    assert trace["critical_path_task_ids"] == "2"
    assert np.isclose(trace["completion_time_s"], 1.2)
    assert np.isclose(trace["component_sum_s"], 1.2)
