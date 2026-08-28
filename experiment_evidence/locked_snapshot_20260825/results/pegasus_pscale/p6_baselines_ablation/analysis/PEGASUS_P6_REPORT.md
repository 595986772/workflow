# Pegasus-B8 Baseline and Mechanism Ablation Report

- Protocol: `pegasus_p6_baselines_ablation_v1`.
- Seeds 51-60 are paired final confirmation seeds, not a new holdout.
- Integrity passed: `True`.

## Main Baselines

| Method | Mean (s) | P95 (s) | Cache hit | Coverage | Remote load | Waiting (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1.829013 | 2.118873 | 0.0969 | 0.4200 | 0.9031 | 0.163896 |
| Nearest | 1.585909 | 2.128461 | 0.0952 | 0.4200 | 0.9048 | 0.167644 |
| Nearest-with-Service | 1.646163 | 2.186300 | 0.4206 | 0.3900 | 0.5794 | 0.181285 |
| DQN-WDSA | 1.130813 | 1.469796 | 0.0964 | 0.5100 | 0.9036 | 0.118224 |
| DAOC-paper | 0.773611 | 1.061017 | 0.1815 | 0.5600 | 0.8185 | 0.120138 |
| Centralized-Greedy-DQN | 0.374959 | 0.558572 | 0.2919 | 0.8000 | 0.7081 | 0.122090 |
| CoordCache-DiscreteSAC | 0.300081 | 0.446341 | 0.4212 | 0.8000 | 0.5788 | 0.104277 |
| OUR | 0.266788 | 0.433570 | 0.5889 | 0.8000 | 0.4111 | 0.108087 |

## OUR Paired Comparisons

- OUR vs Random: improvement `85.414%`, wins `10/10`, 95% CI `[1.303773, 1.820677] s`, p=`0.000976562`, pass=`True`.
- OUR vs Nearest: improvement `83.178%`, wins `10/10`, 95% CI `[1.062462, 1.575781] s`, p=`0.000976562`, pass=`True`.
- OUR vs Nearest-with-Service: improvement `83.793%`, wins `10/10`, 95% CI `[1.145767, 1.612984] s`, p=`0.000976562`, pass=`True`.
- OUR vs DQN-WDSA: improvement `76.407%`, wins `10/10`, 95% CI `[0.695258, 1.032793] s`, p=`0.000976562`, pass=`True`.
- OUR vs DAOC-paper: improvement `65.514%`, wins `10/10`, 95% CI `[0.390582, 0.623065] s`, p=`0.000976562`, pass=`True`.
- OUR vs Centralized-Greedy-DQN: improvement `28.849%`, wins `10/10`, 95% CI `[0.072407, 0.143935] s`, p=`0.000976562`, pass=`True`.
- OUR vs CoordCache-DiscreteSAC: improvement `11.095%`, wins `10/10`, 95% CI `[0.017521, 0.049065] s`, p=`0.000976562`, pass=`True`.

## Core 2x2

- coord_cache_effect_flat_ddqn: improvement `65.543%`, wins `10/10`, 95% CI `[0.520468, 0.841120] s`, p=`0.000976562`, pass=`True`.
- coord_cache_effect_pairwise_pd3qn: improvement `63.818%`, wins `10/10`, 95% CI `[0.321623, 0.619481] s`, p=`0.000976562`, pass=`True`.
- pairwise_effect_standard_cache: improvement `29.013%`, wins `9/10`, 95% CI `[0.138156, 0.464568] s`, p=`0.001953125`, pass=`True`.
- pairwise_effect_coordinated_cache: improvement `25.459%`, wins `10/10`, 95% CI `[0.073100, 0.109140] s`, p=`0.000976562`, pass=`True`.

## Mechanism Ablations

- OUR vs OUR-noTaskDependency: improvement `10.494%`, wins `10/10`, 95% CI `[0.008683, 0.053874] s`, p=`0.000976562`, pass=`True`.
- OUR vs OUR-noDependencyCache: improvement `2.080%`, wins `5/10`, 95% CI `[-0.009741, 0.021078] s`, p=`0.422851562`, pass=`False`.
- OUR vs OUR-TerminalReward: improvement `13.971%`, wins `8/10`, 95% CI `[0.017071, 0.069581] s`, p=`0.004882812`, pass=`True`.

## Evidence Decision

- `integrity_passed`: `True`
- `our_beats_daoc`: `True`
- `our_beats_centralized_greedy`: `True`
- `our_beats_coord_cache_discrete_sac`: `True`
- `coordinated_cache_supported_in_both_learners`: `True`
- `pairwise_pd3qn_supported_in_both_cache_regimes`: `True`
- `task_dependency_state_supported`: `True`
- `cache_dependency_weighting_supported`: `False`
- `causal_makespan_reward_supported`: `True`

A mechanism is retained as a primary paper claim only when its paired test passes. Failed ablations are reported as unsupported rather than being hidden or retuned on these seeds.
