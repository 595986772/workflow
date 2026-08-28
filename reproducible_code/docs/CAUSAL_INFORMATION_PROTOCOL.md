# Causal information protocol for `critical_path_joint`

## Status

`critical_path_joint` now uses the `causal_history_only_v1` information
regime. Cache decisions at the end of window \(k\) use only measurements from
completed tasks in windows \(1,\ldots,k\), and the resulting placement is used
for the next window.

Results generated before this change used privileged current-episode
information and are diagnostic only. They are not eligible for a DAOC-vs-OUR
paper comparison.

## Information boundary

Allowed at decision time:

- Metadata of an admitted DAG: dependencies, service identifiers, and static
  task metadata. This is the dependency-aware DAG model already required by
  CPR and should be stated explicitly in the paper.
- Current cache contents and configured network transfer costs.
- CPU cycles of tasks that have completed.
- Computing and waiting latency reported after a task has completed.
- Counts of completed service requests in previous cache windows.

Forbidden at decision time:

- Aggregating `users[*].tasks_init` to inspect unexecuted task CPU cycles.
- Reading `users[*].numberoftasks` to obtain the exact current-window demand.
- Reading exact `server.load` or `server.frequency` for cache placement.
- Using future DAG arrivals or future task outcomes.

The correct claim is therefore *causal online adaptation with admitted-DAG
metadata*. It is not a fully blind workload model, and it does not assume that
future application arrivals are known.

## Causal estimators

For completed cache window \(k\), let \(\mathcal{O}_k\) contain only tasks that
finished in that window.

The request-count estimate is:

\[
\hat N_k = (1-\alpha)\hat N_{k-1}
  + \alpha |\mathcal{O}_k|.
\]

The workload estimate is:

\[
\hat C_k = (1-\alpha)\hat C_{k-1}
  + \alpha \frac{1}{|\mathcal{O}_k|}
    \sum_{i\in\mathcal{O}_k} C_i.
\]

For server \(s\), the observed execution sample is
\(L_i^{exec}=L_i^{compute}+L_i^{wait}\). Its server-specific window mean is
used to update:

\[
\hat L_{s,k} = (1-\alpha)\hat L_{s,k-1}
  + \alpha \bar L_{s,k}^{exec}.
\]

No sample is added before task completion. A server without observations uses
the global completed-task latency EMA as a causal fallback. Before the first
completed window, every server has neutral quality 1.

The compute-aware placement multiplier is:

\[
Q_{s,k} =
\left(
  \frac{\min_j \hat L_{j,k}}{\hat L_{s,k}}
\right)^{
  \omega \cdot
  \operatorname{clip}(\hat C_k/C_{\max},0,1)
}.
\]

Here \(C_{\max}\) is a configured workload bound, not a current-episode
measurement.

## Audit artifacts

Every new run records:

- `information_protocol_version=causal_cache_v1`
- `cache_information_regime=causal_history_only_v1`
- `cache_history_windows`
- `cache_expected_requests_ema`
- `cache_mean_cpu_cycles_ema`
- `cache_global_execution_latency_ema`
- the per-server latency EMA and last decision context in `summary.json`
- the complete causal history state in model checkpoints

`test_critical_path_cache.py` also constructs servers whose `load` and
`frequency` properties raise an exception if read. The joint decision must
complete without a `users` object and without touching either property.

## Required rerun

At minimum, rerun DAOC, `cpr_joint_cache`, and OUR with paired seeds and the
same convergence protocol. Do not merge old joint-cache/OUR scores with new
causal scores. Recommended ablations are:

- `critical_path_coordinated`: no server-quality multiplier.
- causal joint cache with `--cache-compute-weight 0`.
- causal joint cache with the default historical latency quality.
- history sensitivity with `--cache-history-alpha` in `{0.05, 0.1, 0.2}`.
