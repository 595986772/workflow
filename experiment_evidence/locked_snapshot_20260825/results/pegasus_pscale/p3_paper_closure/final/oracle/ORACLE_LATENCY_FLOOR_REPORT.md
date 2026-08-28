# Clairvoyant Capacity-Aware Reference

## Protocol

- Recreated all frozen evaluation scenarios from their deterministic seeds.
- Verified every reconstructed fingerprint against both DAOC and OUR.
- The primary Oracle sees the complete workload and true server/link parameters.
- Its service placement enforces every per-server K_s constraint.
- Assignment branches and queue coupling are relaxed.
- Cache placement is a capacity-feasible clairvoyant greedy solution, not a globally certified joint optimum.
- The separate perfect-cache recurrence is the certified latency floor.

## Result

| Method | Mean finish time (s) |
|---|---:|
| DAOC | 0.773611 |
| Best validated OUR | 0.266788 |
| Capacity-feasible clairvoyant reference | 0.041933 |

- OUR improves over DAOC by 65.51%.
- OUR remains 0.224855 s above this diagnostic reference.
- The observed OUR-to-reference gap is 84.28% of OUR latency.
- OUR has closed 69.27% of the DAOC-to-reference gap.

## Exact Check

- MILP-validated scenarios: 0.
- Mean DP relaxation gap to the exact optimistic MILP: not run.

The capacity-aware result is clairvoyant and diagnostic only. It uses future workload, greedy cache placement, and relaxed queue coupling, so it must not be presented as an online method or a certified global lower bound.
