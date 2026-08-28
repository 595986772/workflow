# Pegasus-B8 26,000轮统一训练预算报告

- 全部主对比学习方法统一到26,000轮。
- 在线稳态指标固定使用第25,001–26,000轮原始日志。
- 冻结评估使用每seed 100个完全配对场景。
- Seeds 51–60是最终确认seed，不称为独立holdout。

## 完整性审计

- `all_runs_complete`: `True`
- `all_training_ends_at_26000`: `True`
- `all_tail_windows_have_1000_rows`: `True`
- `all_evaluations_have_100_rows`: `True`
- `all_evaluation_states_frozen`: `True`
- `all_scenario_banks_paired`: `True`
- `new_methods_use_fixed_final_checkpoint`: `True`

## 在线训练稳态尾部

| 方法 | 平均DAG完成时间 (s) | 95% CI | P95 (s) |
|---|---:|---:|---:|
| DAOC | 0.707946 | [0.557889, 0.858002] | 1.233779 |
| DQN-WDSA | 1.063861 | [0.874213, 1.253509] | 1.844741 |
| DAOC+CoordCache | 0.272644 | [0.193717, 0.351571] | 0.583875 |
| OUR | 0.222293 | [0.145356, 0.299230] | 0.456743 |
| CoordCache-SAC | 0.488313 | [0.400805, 0.575822] | 0.808192 |
| DiscreteSAC+Std | 1.529187 | [1.272999, 1.785376] | 2.419834 |

### OUR配对优势

- OUR vs DAOC：68.600%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs DQN-WDSA：79.105%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs DAOC+CoordCache：18.468%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs CoordCache-SAC：54.477%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs DiscreteSAC+Std：85.463%，10/10 seed获胜，p=0.000976562，formal=True。

## 冻结配对评估

| 方法 | 平均DAG完成时间 (s) | 95% CI | P95 (s) |
|---|---:|---:|---:|
| DAOC | 0.773611 | [0.661145, 0.886076] | 1.061017 |
| DQN-WDSA | 1.130813 | [0.938013, 1.323614] | 1.469796 |
| DAOC+CoordCache | 0.377305 | [0.273155, 0.481456] | 0.574989 |
| OUR | 0.263429 | [0.174018, 0.352841] | 0.434430 |
| CoordCache-SAC | 0.305716 | [0.216651, 0.394781] | 0.458749 |
| DiscreteSAC+Std | 1.161505 | [0.938394, 1.384616] | 1.496105 |

### OUR配对优势

- OUR vs DAOC：65.948%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs DQN-WDSA：76.704%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs DAOC+CoordCache：30.181%，10/10 seed获胜，p=0.000976562，formal=True。
- OUR vs CoordCache-SAC：13.832%，9/10 seed获胜，p=0.00195312，formal=True。
- OUR vs DiscreteSAC+Std：77.320%，10/10 seed获胜，p=0.000976562，formal=True。
