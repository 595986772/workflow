# Source Provenance

打包日期：2026-08-28

## Reproducible Code

代码来自 2026-08-25 最终 Pro 复评包中的 `03_REPRODUCIBLE_CODE`。打包前已逐项核对以下核心文件与当前工作目录版本，SHA-256 均一致：

`agent.py`, `broker.py`, `capacity_protocol.py`, `critical_path_cache.py`,
`critical_path_reward.py`, `critical_path_rl.py`, `discrete_sac.py`, `dqn.py`,
`information_protocol.py`, `run_independent_experiment.py`,
`run_reproduction_suite.py`, `server.py`, `simulator.py`, `task.py`, `user.py`。

该代码快照包含：

- Pegasus 五类工作流数据及转换结果；
- Alibaba-CP100 受控实验数据；
- DAOC、DQN、Discrete SAC、DCC 与 OUR 相关训练/评估入口；
- 容量异构、DAG 完成语义、因果信息协议和主要实验协议测试；
- 复现说明与依赖清单。

不包含历史 checkpoint。旧 checkpoint 生成后，代码中的到达时刻、奖励和 DAG 终止语义曾有修订，因此不将旧权重作为与当前代码严格锁定的模型发布。

## Pro Reviews

- `OUR_前四章与DAOC写作模仿评阅方案_20260819.md`
- `OUR_二次Pro严格复评_20260821.md`
- `OUR_投稿前最终严格审计_20260825.md`

三份文件按原文复制，未对审阅意见进行改写。

## Manuscript

稿件来自 `最新版分章节论文_20260825` 工作目录在 2026-08-28 的当前状态，包括中文摘要、第一至第六章、参考文献、版本清单和章节校验值。

第一章已使用指定的混合双时间尺度架构图；第三、四章保留当前公式对象；第五章为现有正式实验叙事版本。稿件中的 `OUR` 仍为方法占位名称。

## Paper Assets

第一章架构图采用以下最终文件：

- `paper_assets/architecture/hero_dag_system_model_polished_v8_scheduler_gap_plus30.svg`
- `paper_assets/architecture/hero_dag_system_model_polished_v8_scheduler_gap_plus30_editable.drawio`

Draw.io源文件内嵌11个图像对象，不依赖本机绝对路径。`paper_assets/formal_figures/`只保留当前第五章实际引用的8张SVG，不纳入历史候选图和废弃样式。

## Experiment Evidence

`experiment_evidence/locked_snapshot_20260825/`来自最终Pro复评包的关键结果目录，包含正式主结果、seed级统计、协议锁、作图CSV及在线收敛曲线源数据。它用于核对论文数字和重新绘图，不等同于完整训练工作目录。

## Publication Boundary

本仓库是研究工作快照，不等同于最终投稿归档版本。正式投稿前仍需完成全文合稿、英文改写、统一编号、引用核查和PDF逐页检查。DAOC原文PDF未收入公开仓库，需由作者从合法来源单独保存。
