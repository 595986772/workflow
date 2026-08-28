# Workspace Handoff

更新时间：2026-08-28

## 1. 当前论文身份

研究问题是固定全网缓存预算下，容量异构边缘服务器中的服务副本放置与DAG任务卸载。当前方法由低频中央DCC副本协调和高频入口服务器级Pairwise PD3QN卸载组成；`OUR`仍是待替换的方法占位名称。

主环境采用20个用户、10台服务器、10种服务、15 kHz带宽和总预算8，服务器容量多重集为`[0,0,0,0,1,1,1,1,2,2]`。正式主比较使用seeds 51--60，每seed 100个配对冻结场景。

## 2. 继续工作的入口

1. 论文正文：`manuscript/latest_chapters/`
2. 方法和协议事实：`manuscript/supporting_notes/`
3. 最终架构图与论文图：`paper_assets/`
4. 正文数字与seed级证据：`experiment_evidence/locked_snapshot_20260825/`
5. 可执行代码与测试：`reproducible_code/`
6. 作者原始DAOC源码：`upstream/daoc`（Git submodule）
7. 审稿风险和未决项：`pro_reviews/OUR_投稿前最终严格审计_20260825.md`

## 3. 当前核心结果边界

- OUR平均DAG完成时间为0.2668 s，DAOC为0.7736 s，SAC + DCC为0.3001 s。
- OUR相对DAOC和SAC + DCC的平均完成时间改善分别为65.51%和11.09%，两组均为10/10 seed获胜。
- OUR与SAC + DCC的平均完成时间差异达到显著，但P95差异未达到显著，不能声称P95显著优于该强基线。
- 容量异构性、服务压力等三seed实验只用于趋势和机制解释，不能写成普遍最优证明。

## 4. 投稿前剩余工作

- 确定正式方法名称并统一替换`OUR`；
- 以最新Pro审计为依据完成论文、源码、标签和参数的最终事实核对；
- 合并中文分章，完成英文改写、统一图表/公式/引用编号；
- 按目标期刊模板导出PDF并逐页检查公式、中文字体和矢量图；
- 投稿前再次检查公开仓库是否符合双盲和期刊匿名要求。

## 5. 未纳入Git的材料

- DAOC受限出版PDF及其他受版权限制的论文；`references/DAOC_REFERENCE.md`已保存官方与作者机构获取入口；
- 历史checkpoint和完整逐episode训练目录；
- 旧稿、废弃图、候选图标库和已判定无效的诊断结果；
- 本机Python环境和Codex账户级配置。

上述材料并非继续写作和复核当前结论的必要依赖。需要重新训练时，应按`reproducible_code/REPRODUCE_CN.md`重建环境，而不是复用与当前代码不一致的旧checkpoint。
