# Reproduction workflow

## What is implemented

The reproduction path uses one fresh Python process for every
`algorithm x seed` pair. This isolates the model, replay buffers, topology,
cache estimates, random-number streams, and adaptive deadlines.

Each run has two phases:

1. Training with exploration, replay updates, target-network updates,
   adaptive deadlines, and online caching.
2. Evaluation with greedy DQN actions and frozen model weights, replay
   buffers, epsilon values, deadlines, cache estimates, and cached services.

The runner verifies the frozen state after evaluation and fails if any
protected value changes.

## Algorithm mapping

| Label | Code algorithm | Role |
|---|---|---|
| `random` | `random` | Random offloading |
| `nearest` | `nearest_server` | Nearest cloudlet |
| `greedy` | `nearest_with_service` | Nearest cloudlet caching the service |
| `basic_dqn` | `simpleDQN` | Task features only |
| `service_dqn` | `justserviceDQN` | Current and successor service features |
| `dependency_dqn` | `prev_serversDQN` | Service and predecessor destinations |
| `unguided_full` | `prev_servers_plus_service_per_serverDQN`, beta 0 | Full state without guidance |
| `guided_full` | `prev_servers_plus_service_per_serverDQN`, beta 0.1 | Full state with service-guided exploration |

The repository default
`nearestserver_prev_servers_plus_service_per_serverDQN` adds another
nearest-server one-hot vector. It is not used in the initial paper-method
mapping because the state described in the repository README matches
`prev_servers_plus_service_per_serverDQN`.

## Commands

Fast integration validation:

```bash
.venv/bin/python run_reproduction_suite.py \
  --profile quick \
  --suite-dir results/reproduction_quick_validation
```

Three-seed pilot:

```bash
.venv/bin/python run_reproduction_suite.py \
  --profile pilot \
  --suite-dir results/reproduction_pilot
```

Resume completed runs:

```bash
.venv/bin/python run_reproduction_suite.py \
  --profile pilot \
  --suite-dir results/reproduction_pilot \
  --resume
```

Paper-scale audit:

```bash
.venv/bin/python run_reproduction_suite.py \
  --profile paper_audit \
  --suite-dir results/reproduction_paper_audit_30k \
  --workers 8 \
  --keep-going
```

Re-run the integrity and conclusion audit without repeating training:

```bash
.venv/bin/python audit_paper_results.py \
  --suite-dir results/reproduction_paper_audit_30k
```

Regenerate statistics and figures without rerunning experiments:

```bash
.venv/bin/python aggregate_reproduction_results.py \
  --suite-dir results/reproduction_pilot
```

## Output contract

Each run writes:

- `config.json`: command and simulator configuration.
- `scenario_initial.json`: generated topology, DAGs, service sizes, and resources.
- `scenario_after_training.json`: learned cache placement and final scenario state.
- `episodes.csv`: one structured row per training or evaluation episode.
- `checkpoint.pt`: final per-cloudlet DQN weights for learning algorithms.
- `summary.json`: seed-level metrics and frozen-state status.
- `run.log`: complete process output.

The suite writes:

- `aggregate_summary.csv`: absolute seed-level means and 95% t-CIs.
- `paired_comparisons.csv`: matched-seed ratios, wins, and improvements.
- `training_convergence.{png,pdf}`.
- `evaluation_finish_time.{png,pdf}`.
- `paired_relative_performance.{png,pdf}`.
- `guidance_ablation.{png,pdf}`.
- `latency_breakdown.{png,pdf}`.
- `cache_metrics.{png,pdf}`.

## Current pilot result

Configuration: 5 users, 5 servers, 5 services, DAG size at most 10,
3000 training episodes, 200 frozen evaluation episodes, and seeds 1-3.

The guided full method has the lowest arithmetic mean evaluation finish
time, 0.1507 s, but three seeds are not enough for a broad superiority
claim. Absolute topology difficulty varies substantially across seeds.

The clean result is the matched guidance ablation:

| Seed | Unguided full | Guided full |
|---|---:|---:|
| 1 | 0.4198 s | 0.2339 s |
| 2 | 0.2768 s | 0.1338 s |
| 3 | 0.6282 s | 0.0844 s |

Guidance wins all three matched seeds. Mean paired improvement is 60.8%
and median paired improvement is 51.7%. This is promising pilot evidence,
not a final paper result.

Against the nearest baseline, guided full wins only one of three matched
seeds and has a median finish-time ratio of 1.067. Therefore the current
pilot does not establish that the full method consistently beats all
simple baselines.

## Paper-scale audit result

The formal audit uses the paper-scale default environment: 20 users,
10 cloudlets, 10 services, 10 independent topology seeds, 30,000 training
episodes, and 500 frozen evaluation episodes. The learning configuration
uses two 64-node hidden layers, epsilon 0.01, batch size 1024, and 15 kHz
bandwidth. It compares:

- Nearest.
- Nearest + Service using the paper's nearest-candidate definition.
- Unguided Full.
- Guided Full with fixed beta 0.1, matching the public code default.
- Guided Full with beta decaying from 0.9 to 0.1 at 0.995 per episode,
  matching the paper formula.

All 50 runs completed and passed frozen-state validation. Initial topology
and workload snapshots match exactly across methods for every paired seed.

Frozen-policy mean application finishing times are:

| Method | Mean (s) | 95% CI half-width |
|---|---:|---:|
| Nearest | 1.6389 | 0.0989 |
| Nearest + Service | 1.6062 | 0.0984 |
| Unguided Full | 1.5415 | 0.1538 |
| Guided Full, fixed beta | 1.3842 | 0.0620 |
| Guided Full, paper decay | 1.3864 | 0.0618 |

Both guided variants beat Nearest, Nearest + Service, and Unguided Full on
all 10 matched frozen-evaluation seeds. Fixed-beta paired mean improvements
are 15.16%, 13.33%, and 9.09%, respectively. Under the paper-style
late-training metric, fixed beta beats the two heuristics on 10/10 seeds
and Unguided Full on 9/10 seeds.

The paper beta schedule does not improve on fixed beta. Paper Decay beats
Fixed Beta on only 2/10 frozen-evaluation seeds; its paired mean change is
-0.16% +/- 0.66%. The two variants are effectively tied.

Late-training movement is small: the final 5,000 episodes improve over
episodes 20,001-25,000 by about 0.67% for Fixed Beta and 0.51% for Paper
Decay. The 30,000-episode conclusion is therefore not primarily explained
by obvious non-convergence.

Detailed integrity checks and statistics are in
`results/reproduction_paper_audit_30k/AUDIT_REPORT.md`.

## Interpretation

The paper's core default-setting ranking is reproduced under this audit:
guided action shaping is useful and consistently beats the three key
comparators. The earlier three-seed pilot failed because it was a reduced,
underpowered configuration and should not be used to judge the paper.

Two limitations remain:

- This audit covers one default configuration, not every parameter sweep
  behind the paper's "various conditions" claim.
- The local improvement is about 13-15% against the heuristic baselines,
  not the roughly 80% ablation effect described in the paper. The paper-code
  discrepancies in bandwidth, training duration, and released beta logic
  still warrant a targeted sensitivity audit.
