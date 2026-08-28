# Pegasus Cross-Topology DAG Benchmark

This dataset is a deterministic, post-hoc cross-topology benchmark for h8v1.
It is not an unbiased MEC trace or an independent preregistered holdout.

- Source: https://github.com/WorkflowSim/WorkflowSim-1.0
- Pinned source commit: `d3ea21afd8ce6479bd292d3bd7469045d7a36089`
- Families: Montage, CyberShake, Epigenomics, Inspiral, SIPHT
- Output: 20 connected sub-DAGs per family, 100 graphs total
- Size: 9 real tasks plus one dummy source per graph
- Services: stable hash of the original Pegasus job type into 10 service IDs
- CPU and edge data: global log scaling into the existing DAOC normalized ranges
- Dataset SHA-256: `68750cde2903985532506066afb574867bf8af1e3eaaf1b5cb33bbc8354ae6d2`

The five source DAX files and upstream license notices are stored under `raw/`.
Regenerate with:

```bash
python build_pegasus_cross_dataset.py
```
