# KataGo Training 实验记录

本仓库记录 KataGo 神经网络训练实验，目标包含更大模型的继续训练与结构实验。
当前已经部署并验证的阶段是 **9×9 b6c96 卷积网络续训预演**；仓库名称中的
`b4tf12c1024` 是后续实验方向，不代表当前已部署该 Transformer 模型。

## 9×9 b6c96 续训预演

本项目已在 RTX 5070 Laptop 8GB 上部署 Windows 原生训练，目的是预演以后在
19×19 上续训 tf3-b11 所需的完整操作流程。采用官方自对弈数据、官方 PyTorch
训练器和官方模型导出器，外层用 Python/PowerShell 编排有限任务。

**开始实际训练用 `.\run.ps1 cycle`。复查短流程用 `.\run.ps1 cycle -Smoke`。**
每次命令都会结束；当前没有注册自动任务，也没有留下无限循环训练进程。

## 已部署的内容

| 项目 | 本机配置 |
|---|---|
| GPU | RTX 5070 Laptop，8GB，已验证实际 CUDA 矩阵运算与训练 |
| 训练运行时 | Python 3.14.0，PyTorch 2.11.0+cu128，NumPy 2.4.6 |
| 训练环境 | `.venv`，继承本机已有 CUDA PyTorch；没有改动系统包 |
| 旧权重转换环境 | `.venv-convert`，Python 3.10.11，独立 NumPy/protobuf |
| 引擎 | 已有 KataGo v1.18.2 CUDA 12.8/cuDNN 9.8 安装，路径见 `configs/local.json` |
| 官方源码 | commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`，位于 `vendor/` |
| 原始权重 | 官方 `b6c96-s175395328-d26788732`，保留原始下载与 SavedModel 参数 |
| 可训练起点 | `models/pretrained/b6c96-pytorch/initial.ckpt` |
| 当前训练 checkpoint | `runs/b6c96-9x9/train/checkpoint.ckpt`，包括优化器状态 |
| 最近导出的引擎模型 | 读取 `runs/b6c96-9x9/last-export.json` |

## 权重兼容性与迁移边界

历史 b6c96 是 TF 模型格式 v8，当前上游同名 `b6c96` 配置是另一个版本。
直接使用 `-model-kind b6c96` 并不能继续训练历史 `.txt.gz`。

本项目从官方完整参数 checkpoint 提取了 66 个张量，建立 v11 PyTorch 配置：

- 保留旧模型 5×5 输入卷积、6 个残差块及全局池化分支、主策略与对手策略、
  胜负、目数/领先预测、ownership、scoring、future-position 和 seki 参数。
- 新增的 soft-policy 两路由原有策略两路复制初始化。
- 目标网络 1,035,149 个参数中，1,029,293 个从原参数赋值或复制，占约 99.43%。
- 新版分数分布头的结构、短期误差头和新增 TD 输出无法完整对应，明确重新初始化。
  旧方差时间输出虽保留权重，但新版缩放从 150 变为 40，需要适应。
- TF 优化器历史没有恢复；新的 AdamW 状态从零开始。此后 PyTorch checkpoint
  的保存与恢复会保留优化器、样本计数等训练状态。

所以这是继承已有主干和主要能力的迁移续训，附带少量辅助头适配。
完整映射与未使用参数见 `models/pretrained/b6c96-pytorch/initial.migration.json`。
唯一的上游代码兼容修改见 `patches/legacy-b6.md`；可由 bootstrap 重建。

已通过同一个 KataGo 引擎的 `kata-raw-nn` 对照：固定对称变换、FP32、9×9 空盘和
两个非空局面，迁移前后的主策略（含 pass）、胜负概率、目数矩、领先和 ownership
在引擎文本输出精度内完全一致。辅助头不在等价声明范围内。
证据：`runs/b6c96-9x9/migration-verification.json`。

## 运行方式

在 PowerShell 中：

```powershell
Set-Location 'D:\My codes\Katago_Training'

# 检查本机 GPU / CUDA / 引擎，保存 environment.json
.\run.ps1 doctor

# 短流程：12 局训练数据 + 4 局独立验证数据 → shuffle → 训练 1 epoch
# → 导出 → 新模型再完成 2 局自对弈。已有 checkpoint 会自动恢复。
.\run.ps1 cycle -Smoke

# 预演常规流程：128 局训练数据 + 16 局独立验证数据 → 训练 → 导出 → 新模型试运行
.\run.ps1 cycle

# 用最近一次导出的候选模型与当前自对弈基线比赛
.\run.ps1 evaluate -Games 200 -Visits 128

# 达到评测门槛后，手动将候选设为后续自对弈的模型
.\run.ps1 promote

# 将后续自对弈恢复为原始 b6 基线，保留所有训练与导出文件
.\run.ps1 rollback
```

中断任务可按 Ctrl+C；编排器会终止它启动的当前子进程。训练只能从最后一次完整
保存的 checkpoint 恢复。不要同时启动两个会写同一 `run_dir` 的训练/流程命令。
原始权重、转换起点和已有引擎安装保留。

分步操作：

```powershell
.\run.ps1 selfplay -Games 128
.\run.ps1 selfplay -Games 16 -Label val
.\run.ps1 shuffle
.\run.ps1 train -Epochs 1
.\run.ps1 export

# 指定候选评测，而非最近导出
$candidate = (Get-Content .\runs\b6c96-9x9\last-export.json -Raw | ConvertFrom-Json).model
.\run.ps1 evaluate -Model $candidate -Games 200 -Visits 128
```

每一步的完整命令、标准输出与错误会保存到 `runs/b6c96-9x9/logs/`。
PowerShell 会显示 `cycle 1/6` 到 `6/6` 的阶段提示，以及当前任务、已用时间和
日志解析出的状态；静默阶段每约 10 秒更新一次提示。自对弈的“已启动”局数包含
尚未结束的棋局，不代表完成数。小 epoch 内没有批次日志时会显示训练阶段和计时，
不会估算虚假的百分比或剩余时间。失败时会自动显示最后 20 行日志。
checkpoint 与引擎模型是不同文件：`.ckpt` 用于训练，`.bin.gz` 用于 KataGo 推理。
外层脚本支持 Python 入口：`python scripts/pipeline.py --help`。

## 当前参数与调整顺序

| 参数 | 短流程 | 常规预演 |
|---|---:|---:|
| 棋盘和数据张量边长 | 9 | 9 |
| 自对弈搜索 visits | 32 | 128 |
| 自对弈并行局数 | 4 | 8 |
| cheap-search | 关闭 | 75% 概率 32 visits；这些局面目标权重为 0 |
| batch size | 32 | 32 |
| 每 epoch 目标样本数 | 256 | 1024 |
| 每次训练 epoch 数 | 1 | 1 |
| 优化器/精度 | AdamW / FP16 / 不编译 | 同左 |
| lr-scale | 0.1 | 0.1 |
| 初始训练 warmup | 使用上游 warmup | 取消随机初始化用的长 warmup，使用保守续训 LR |
| 数据窗口上限 | 250,000 行 | 250,000 行 |

`lr-scale=0.1` 是对官方学习率公式的乘数，**不是学习率 0.1**。
batch32 的常规主干 LR 约为 `1.41e-6`，输出头约为其一半；短流程的初始 warmup
再缩小 20 倍。这里先追求稳定和可验证；它不是已经证明最优的调参结果。

官方训练器按完整数据文件消费样本，目标 epoch 样本数可能被超过；验收中目标 256
实际先后消费了 672、1440 个样本，属于当前文件粒度行为。样本计数也不等于成功
优化器更新数，FP16 GradScaler 在溢出时会跳过更新，初始几步已有这种现象。

常规模式启用数据桶：随新数据增加训练预算，没有足够预算时退出，避免只靠重复
同一小批数据一直更新。预算按照 epoch 目标近似计费，仍有上述文件粒度误差。
短流程为验证恢复允许复用小样本，不能用其 loss 变化评价实际训练收益。

修改运行量和 batch/LR：编辑 `configs/local.json`。
修改搜索和规则：编辑 `configs/selfplay9.cfg`。`configure`/`setup.ps1` 会从官方模板
重新生成此配置，因此应先备份手动调整，或同步修改 `scripts/pipeline.py` 的配置生成项。
8GB 本机先保持 batch32；确认实际显存余量后只改变一个参数试 batch64。

## 训练计划和验收

1. **流程预演（已完成）**：CUDA、自对弈 NPZ、权重迁移、真实反向传播、保存、
   重启恢复、导出、新模型自对弈、对战和 SGF 结果统计。
2. **9×9 小规模稳定性阶段**：先按常规参数跑若干有限 cycle，积累数千至数万条
   有效新局面。记录新数据量、实际样本数、更新数、策略/价值分项 loss 与显存；
   确认新辅助头适应、无非有限值、没有只反复记忆旧小样本。
3. **候选筛选阶段**：固定规则、贴目和 visits，对基线至少跑 200 局。
   当前脚本要求执黑执白均衡，近似 95% Wilson 区间下界大于 50% 才允许 promote。
   这是操作门槛，不是多重试验下的棋力证明。接近门槛的结果应使用新对局复核，
   并加入不同规则/visits 和固定开局集合；现有比赛随机开局不做成对镜像。
4. **19×19 tf3-b11 迁移**：另建配置、数据目录、run_dir，使用官方原生 PyTorch
   checkpoint；无需沿用本次旧 TF 转换。所有 `dataBoardLen`、棋盘尺寸和 `-pos-len`
   一起设为 19，重新测显存和吞吐，先完成有限 smoke，再扩数据、visits 和 batch。
   当前编排器专用于 9×9，会拒绝仅修改 `board_size` 后直接启动；19×19 需要连同
   配置生成、训练和评测的参数一起适配。在云 GPU 上先测自对弈和训练各自的
   成本，再决定算力分配。

9×9 用于确认流程、统计、恢复和算力利用；9×9 棋力增益不用于预测 19×19 tf3-b11
是否增强。当前 b6 没有改深度/宽度，也没有把架构搜索混入流程验收。

候选默认保持独立。`cycle` 会让它试运行，但通过评测并执行 `promote` 前，后续
训练数据仍由当前基线产生。因此可以把数据生成和参数适配稳定下来，再逐步闭环。

## 文件与恢复

```text
configs/                    本机路径、自对弈、GTP 配置
scripts/                    下载校验、参数转换、流程、评测与验收
patches/                    上游兼容修改说明
downloads/                  固定版本的原始 ZIP
vendor/KataGo-<commit>/      官方源码与本地输入卷积兼容补丁
models/pretrained/          原始参数、转换 checkpoint、迁移审计
runs/b6c96-9x9/
  train/checkpoint.ckpt     最近训练状态，恢复入口
  verification/             首次训练 checkpoint 和验收证据
  selfplay/train/           训练数据
  selfplay/val/             独立生成的验证数据
  selfplay/candidate-probe/ 新模型试运行数据，不混入 train/val
  shuffleddata/             只有完整写完的目录才发布给训练器
  exports/                  由 checkpoint SHA 命名的模型和溯源
  evaluation/               每次比赛的配置、日志、SGF 和结果
  logs/                     每一步完整子进程日志
  last-export.json          最近导出的模型位置
  last-cycle.json           最近完整 cycle 的状态
```

继续运行同一训练命令会优先加载 `train/checkpoint.ckpt`，只有它不存在才加载
`initial.ckpt`。不要使用 `-always-initial-checkpoint` 来做恢复。
如果要丢弃实验从预训练起点重新开始，建议在 `configs/local.json` 中换一个新的
`run_dir`，保留当前目录以便复现与比较。

## 重建和以后搬到云端

当前机器无需重复安装。需要核对/补齐时运行 `.\setup.ps1`；只校验已下载文件可运行：

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap.py --offline
```

`setup.ps1` 会校验官方资产 SHA256，不覆盖已有转换权重或训练 checkpoint。
新机器需要修改 `configs/local.json` 中引擎路径，准备相应的 CUDA KataGo 发行版，
并在训练 Python 环境安装对应显卡的 CUDA PyTorch。训练脚本的本机可用版本见
`environment.json`；本机 `.venv` 继承系统包，不能直接复制到另一台机器使用。

云端可用 Python 3.12/3.14 与兼容的 CUDA PyTorch，在新 venv 安装 PyTorch、NumPy、
packaging、sgfmill，然后校验 bootstrap、配置、doctor。Python 编排无需 Bash。
本次提取后的 `initial.ckpt` 可直接携带，云端续训无需旧转换环境；若恢复本次
进度，还需携带完整 `train/`、对应数据和 `configs/`。下载/复制模型时同时保留
`initial.migration.json`、`deployment-assets.json` 和导出的 `provenance.json`。
实际云端兼容性和吞吐仍需在目标机器验证，本次只完成本机部署。

## 官方依据

- [KataGo 自行训练说明](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/SelfplayTraining.md)
- [官方权重下载列表](https://katagotraining.org/networks/)
- [原始 b6c96 checkpoint](https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b6c96-s175395328-d26788732.zip)
- [模型版本与输入特征对应](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/neuralnet/modelversion.cpp)
- [历史 TF 模型实现](https://github.com/lightvector/KataGo/blob/v1.4.5/python/model.py)
- [当前 PyTorch 训练器](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/python/train.py)

之前收集的大量资料和专题分析保存在 `research/`，本文件描述实际部署。

## GitHub 备份范围

仓库包含训练编排、PowerShell 入口、实时进度提示、配置、固定版本资产校验信息、
资料汇总与部署验收记录。验收 JSON 快照保存在 `docs/verification/`，记录的是
2026-10-17 验收时点，不是后台实时更新的训练进度。

`.gitignore` 排除 `.venv*`、`downloads/`、`vendor/`、`models/` 和 `runs/`。
因此 clone 本仓库不会自带训练 checkpoint、模型权重、棋谱数据或 CUDA 引擎。
部署时可通过 `setup.ps1` 与 `scripts/bootstrap.py` 重建依赖和官方预训练起点；
要恢复本机已经训练的具体进度，还需另行携带对应 checkpoint 和数据。

第三方源码资料的来源和许可见 `THIRD_PARTY_NOTICES.md`。
