# 本机部署验收记录

验收日期：2026-10-17。工作目录：`D:\My codes\Katago_Training`。

**结论：9×9 b6c96 迁移续训和完整操作流程已在本机运行通过。当前候选尚无棋力增强证据。**

| 验收项 | 实际结果 | 证据 |
|---|---|---|
| 本机 GPU / CUDA | RTX 5070 Laptop 8GB，PyTorch 实际 CUDA 运算和反向传播成功 | `runs/b6c96-9x9/environment.json` |
| 官方资产 | 固定源码 commit 与 b6c96 原始 ZIP 已下载并校验 SHA256 | `deployment-assets.json` |
| 继承已有权重 | 提取 66 个 TF 张量；目标参数约 99.43% 从原参数赋值或复制 | `models/pretrained/b6c96-pytorch/initial.migration.json` |
| 转换正确性 | 3 个 9×9 局面，主策略/胜负/目数矩/领先/ownership 在引擎打印精度内一致 | `runs/b6c96-9x9/migration-verification.json` |
| 真实训练数据 | 24 局产生 1,449 行训练数据；另外 8 局产生 490 行验证数据 | `runs/b6c96-9x9/selfplay/` |
| 保存、重启与继续训练 | 样本计数 0 → 672 → 1344 → 2784 → 4224；最终 AdamW 更新计数 100 | `runs/b6c96-9x9/train/` 与 `logs/` |
| 参数确实更新 | 首次训练后 67 个参数张量改变；后续训练 68 个张量改变；所有最终参数有限 | `runs/b6c96-9x9/training-verification.json` |
| 断点恢复优化器 | 首次检查更新计数 13 → 26；最终为 100，未重置到零 | 同上及训练日志 |
| 一键完整流程 | `run.ps1 cycle -Smoke` 完成数据生成 → shuffle → 训练 → 导出 → 新模型自对弈 | `runs/b6c96-9x9/last-cycle.json` |
| 常规训练参数 | AdamW、FP16、batch32、lr-scale0.1，关闭长初始 warmup，实际训练成功 | `runs/b6c96-9x9/logs/20261017-220646-677383-train.log` |
| 无新数据时退出 | 训练预算为零时正常退出，未继续更新 checkpoint | `runs/b6c96-9x9/logs/20261017-221129-349624-train.log` |
| 最终模型推理 | 最终导出模型完成另外 2 局自对弈，生成 112 行数据 | `runs/b6c96-9x9/logs/20261017-221147-397491-selfplay-candidate-probe.log` |
| 对战流程 | 最终模型 32 visits，4 局，2 胜 2 负，执黑/执白各 2 局 | `runs/b6c96-9x9/last-evaluation.json` |
| 重建入口 | `setup.ps1` 幂等复核通过，`bootstrap.py --offline` 校验通过 | `setup.ps1`、`scripts/bootstrap.py` |

当前训练 checkpoint：

`runs/b6c96-9x9/train/checkpoint.ckpt`

SHA256：`ce37526c5caacdea67e025fd8f63c46a490ea2fe294ee523feb16642082f8f81`

对应最终引擎模型：

`runs/b6c96-9x9/exports/ce37526c5caa/model.bin.gz`

SHA256：`4b1070ca695efaa6f573d45d0722155c36f6ef14a1982f3c6a776805a135dc30`

当前分数分布等少量辅助头是迁移时新初始化的，不属于完整 TF 训练状态等价迁移。
4 局对战的近似 95% 区间约为 15%–85%，仅验证对战和统计正常，不支持棋力提升结论。
原始 b6 继续作为数据生成基线；候选尚未晋级。

此记录固定描述验收时点。以后运行训练会更新 checkpoint；以 `last-export.json`、
对应 `provenance.json` 和最新训练日志确认后续模型版本。运行计划和所有命令见 `README.md`。
