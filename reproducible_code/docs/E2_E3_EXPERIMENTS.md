# E2/E3优先修正版实验手册

## 研究问题

固定全网缓存预算为10，但服务器容量异构：

```text
[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
```

容量按seed确定性随机分配，与算力、位置和初始负载使用独立RNG。`K=0`节点仍可计算，但不能缓存真实服务。

- E2：异构缓存、静态负载。
- E3：沿用E2 checkpoint，窗口51起对`K=0/1/2`各一台服务器施加4倍负载。
- E3-S：最终阶段使用相同checkpoint测试负载倍率2、4、6。

## 正式阶段

所有正式结果写入`results/e2_e3/e2r0/`。旧`r0`和`/tmp`结果不复用。

```bash
cd "$(git rev-parse --show-toplevel)/reproducible_code"

.venv/bin/python run_e2_e3_suite.py --stage smoke \
  --revision-id e2r0 --revision-number 0 --workers 2

.venv/bin/python run_e2_e3_suite.py --stage screen \
  --revision-id e2r0 --revision-number 0 --workers 2

.venv/bin/python run_e2_e3_suite.py --stage converged \
  --revision-id e2r0 --revision-number 0 --workers 2

.venv/bin/python run_e2_e3_suite.py --stage dynamic \
  --revision-id e2r0 --revision-number 0 --workers 2

.venv/bin/python run_e2_e3_suite.py --stage ablation \
  --revision-id e2r0 --revision-number 0 --workers 2

.venv/bin/python run_e2_e3_suite.py --stage final \
  --revision-id e2r0 --revision-number 0 --workers 2

.venv/bin/python run_e2_e3_suite.py --stage e0_audit \
  --revision-id e2r0 --revision-number 0 --workers 2
```

中断后只能对相同命令添加`--resume`。`dynamic`不训练模型；`ablation`复用DAOC和OUR；`final`复用seeds 1–3，只训练缺少的run。

## 阶段门槛

- Smoke：全部容量、公平性、场景配对和因果信息审计通过。
- Screen：OUR平均完成时间更低，至少2/3 seed获胜。
- Converged：双方收敛，OUR完成时间3/3获胜，平均改善至少5%，平均P95更低。
- Dynamic：OUR至少2/3 seed恢复更快，且至少2/3 seed累计Oracle regret更低。
- 严格恢复胜场无法低于0窗口。若DAOC具有正恢复延迟的seed不足2/3，
  则保持E3门槛失败并停止算法修订；不得把平局改算获胜，也不得在看到
  结果后更换恢复统计口径。
- Final：seed级配对改善95% CI下界大于0，单侧Wilcoxon `p<0.05`，至少7/10 seed获胜。

通过Dynamic后生成`results/e2_e3/FROZEN_ALGORITHM.json`。Final首次打开时生成`results/e2_e3/FINAL_LOCK.json`，完成后不能调参重跑。

## 单模块修订

开发失败时最多创建`e2r1`至`e2r3`。每轮只修改诊断报告指定的一个现有模块，不新增模块，并从smoke重新执行。例如：

```bash
.venv/bin/python run_e2_e3_suite.py --stage smoke \
  --revision-id e2r1 --revision-number 1 \
  --revision-parent e2r0 \
  --revision-reason "diagnosed reason" \
  --changed-module actor_server_features \
  --expected-metric remote_loading_rate \
  --workers 2
```

诊断顺序固定为：收敛、K=0动作、缓存价值、服务覆盖、副本边际收益、迁移稳定性、Oracle headroom、EMA质量、actor响应、缓存更新频率和协调开销。三轮仍失败即停止最终扩展。

## 输出

每个阶段保存：

- 算法、协议、环境和配置SHA256；
- checkpoint、完整DAG场景指纹和配对审计；
- 逐服务器容量、缓存矩阵、容量利用率和副本数；
- 远程加载、迁移、动作分布、EMA质量和协调开销；
- 英文与中文诊断报告。

关键图包括E2主结果、缓存热力图、E3恢复曲线、E3-S强度曲线和最小消融。

容量感知clairvoyant Oracle是诊断参考，不是认证全局最优；perfect-cache recurrence才作为严格乐观下限。

## E0补证

`e0_audit`先比较旧`telemetry_pd3qn`与`lean_our`的网络、奖励、缓存、信息和训练协议。两者当前奖励定义不同，因此旧结果不得复用；仅在最终E2/E3通过后补跑原始E0中DAOC与OUR的seeds 1–3。E1不再运行。

Seeds 1–3参与开发，所以最终结果称为“十seed最终确认”，不称为独立holdout。
