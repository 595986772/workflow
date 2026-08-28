# Workflow Scheduling Paper Materials

本仓库保存当前边缘工作流调度论文的可复现实验代码、Pro 模型评阅记录和最新版分章节中文稿件。

## Repository Structure

- `reproducible_code/`: 可执行实验代码、测试、协议脚本和论文使用的数据集。
- `pro_reviews/`: 2026-08-19、2026-08-21 和 2026-08-25 三轮 Pro 评阅原文。
- `manuscript/latest_chapters/`: 2026-08-28 整理的最新版分章节中文稿件。
- `SOURCE_PROVENANCE.md`: 本次打包来源、完整性和适用边界。
- `SHA256SUMS.txt`: 仓库内交付文件的 SHA-256 清单。

## Reproduction

已验证软件环境：

- Python 3.12.8
- PyTorch 2.7.0
- NumPy 1.26.4
- SciPy 1.15.1
- SimPy 4.1.2

进入代码目录并安装依赖：

```bash
cd reproducible_code
python -m pip install -r requirements.txt
```

运行核心回归测试：

```bash
python -m unittest -q \
  test_information_protocol.py \
  test_capacity_protocol.py \
  test_dag_completion_semantics.py \
  test_critical_path_cache.py \
  test_critical_path_reward.py \
  test_critical_path_rl.py \
  test_pegasus_pscale_protocol.py \
  test_pegasus_b8_heterogeneity.py
```

运行轻量训练 smoke test：

```bash
python run_reproduction_suite.py \
  --profile pegasus_pscale_p2_smoke \
  --suite-dir results/repro_smoke \
  --seeds 41 \
  --labels daoc_paper,lean_our \
  --dag-dataset-path datasets/pegasus_pscale/dag_pegasus5_full31.json \
  --dag-dataset-sha256 0671d8ea1ecdd8165062e19733e4859edb8e5ce87ecdd054bcef290abc49d5a5 \
  --eval-dag-families Montage,CyberShake,Epigenomics,Inspiral,Sipht \
  --server-capacity-multiset 0,0,0,0,1,1,1,1,2,2 \
  --capacity-assignment-namespace pegasus_pscale_p2 \
  --workers 2
```

完整协议、正式训练入口和结果边界见 [`reproducible_code/REPRODUCE_CN.md`](reproducible_code/REPRODUCE_CN.md)。

## Scope

本次提交不包含历史 checkpoint、完整训练中间日志、旧稿或已判定无效的性能结果。正式训练成本较高，smoke test 只验证执行链完整性，不能作为论文性能结论。

论文仍处于投稿前修订阶段。`pro_reviews/` 中的内容是审阅意见和风险清单，不代表所有问题已经关闭。

## License

本仓库尚未附加开源许可证。仓库公开可见不等于授予复制、修改或再发布许可；第三方数据文件仍遵循其目录中附带的原始许可说明。
