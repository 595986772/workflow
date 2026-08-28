# Alibaba-CP100

Alibaba-CP100 is a 100-DAG mechanism and development stress set for DAOC.
It is deliberately constructed to expose dependency-locality, critical-path,
and service-caching effects. It is not an unbiased Alibaba holdout and must not
be presented as one.

## Files

- `dag_alibaba_cp100.json`: DAOC-compatible NetworkX node-link graphs.
- `selection_manifest.json`: source Job IDs, task-ID mappings, selected paths,
  service assignments, and per-DAG selection metrics.
- `selection_summary.json`: source checksum, scan counts, thresholds,
  transformation protocol, and aggregate stress metrics.
- `../../tools/build_alibaba_cp100.py`: deterministic dataset builder.

## Source

The builder streams `batch_task.csv` from the official Alibaba Cluster Trace
2018 archive without extracting it. The expected archive SHA-256 is:

`7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6`

The source trace provides task dependencies, execution times, and planned CPU
values. It does not provide MEC service IDs or dependency payload sizes.

## Selection Protocol

Every selected Job:

- has 7, 8, or 9 real tasks and therefore no more than 10 nodes after adding
  DAOC's dummy source node `0`;
- contains only completed tasks with parseable and complete dependencies;
- is acyclic and contains at least one fork and one join;
- has structural depth of at least four;
- has a weighted critical path containing at least four but not all tasks;
- has critical-path work fraction of at least 0.50 and a weighted gap of at
  least 0.08 over the second path;
- does not derive path dominance from a single task contributing more than
  60 percent of the critical-path weight.

The final quotas are 33 seven-task DAGs, 33 eight-task DAGs, and 34 nine-task
DAGs. Within each quota, the highest deterministic stress scores are retained.

## Synthetic MEC Attributes

Task weights are computed as:

`log1p((end_time - start_time) * plan_cpu)`

They are normalized to DAOC's `cpucycle` interval. Critical-path tasks receive
hot service IDs 1-3, while off-path tasks receive deterministic service IDs
4-10. Critical-path edges receive larger normalized payloads than other edges.
These correlations are intentional stress factors and are fully recorded in
the summary and manifest.

## Proper Use

Use this dataset to test whether a method's dependency-aware cache and
offloading mechanisms behave as intended. Train and evaluate every compared
method on identical graphs and seeds.

Do not use Alibaba-CP100 alone for a general superiority claim. A formal paper
should additionally report results on a preregistered, job-disjoint,
distribution-preserving Alibaba dataset that was not selected to favor any
method.

## A0 Service-Alignment Control

`dag_alibaba_cp100_a0.json` is a deterministic control variant that removes
the original alignment between critical-path membership and hot services.
It permutes service labels within each DAG while preserving the DAG topology,
CPU cycles, edge payloads, graph order, and every per-DAG service multiset.
Consequently, both per-DAG and global service request counts are identical to
the base dataset.

`service_alignment_a0_manifest.json` records the base and derived checksums,
the complete task-level mapping, integrity checks, and before/after association
metrics. A0 is still a controlled mechanism dataset, not an unbiased holdout.

Rebuild the variant with:

```bash
python tools/build_alibaba_cp100_a0.py
```

## Rebuild

From the DAOC repository root:

```bash
python tools/build_alibaba_cp100.py
```
