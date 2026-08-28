# Pegasus P-Scale DAG Dataset

This controlled benchmark keeps each complete WorkflowSim/Pegasus workflow
and tests larger DAGs on the fixed DAOC MEC infrastructure. It is not an
unbiased MEC trace.

- Source: https://github.com/WorkflowSim/WorkflowSim-1.0
- Pinned source commit: `d3ea21afd8ce6479bd292d3bd7469045d7a36089`
- Families: Montage, CyberShake, Epigenomics, Inspiral, SIPHT
- Graphs: one complete workflow per family
- Real tasks: `[25, 30, 24, 30, 29]`; one dummy source is added to each graph
- Task limit including dummy source: `31`
- Program types: `39` mapped deterministically to 10 services
- Type mapping: `sha256_namespace_name_mod10_v1`
- CPU and edge data: `pegasus_global_log_minmax_v1`
- Dataset SHA-256: `0671d8ea1ecdd8165062e19733e4859edb8e5ce87ecdd054bcef290abc49d5a5`

Regenerate with:

```bash
/opt/anaconda3/envs/dl/bin/python build_pegasus_pscale_dataset.py
```
