# GR00T Baseline SFT：Sequence128 与 H40

本目录保存两组等有效窗口预算实验的代码身份、实际配置、TensorBoard 和训练日志快照，供检查训练条件和复现实验使用。快照日期为 2026-10-19；文件级时间、原始大小及 SHA-256 见 `snapshot_manifest.json`。

## 实验状态

两组目标均为 10,000 次 optimizer update，但这份快照并非训练完成结果。

| 项目 | Sequence128 | Independent H40 |
|---|---:|---:|
| 快照中最新完成的训练步 | 6788 | 6777 |
| 当前运行来源 | 从完整 6000 状态恢复 | 从完整 6000 状态恢复 |
| 续训退出码 | 134 | 1 |
| 续训中断日志 | SIGTERM | SIGTERM |

两组日志均在 2026-10-19 06:29:25 UTC（北京时间 14:29:25）记录 SIGTERM。信号来源尚未确定，不能据此认定人为关闭、OOM 或数值异常。此前首次运行因磁盘配额错误中断，完整原日志保存在 `*.before_resume6000` 中。

仓库提供的是代码及记录，不包含模型权重、训练 checkpoint、数据集或 Python 环境。上传过程不修改服务器原文件，也不重启训练。

## 对照条件

| 条件 | Sequence128 | Independent H40 |
|---|---|---|
| GPU | 4 × H800，物理卡 0–3 | 4 × H800，物理卡 4–7 |
| 基础模型 | GR00T-N1.7-3B + Cosmos-Reason2-2B | 相同 |
| 数据 | BowlAndCup LeRobot，104 episodes | 相同 |
| 样本 | 有序 anchor 序列，stride 8，最多 128 anchors | 所有合法起点中的独立 H40 窗口 |
| 末尾右对齐概率 | 0 | 0 |
| 动作监督 horizon | 40 | 40 |
| 每卡 micro batch | 1 条序列 | 16 个窗口 |
| global batch | 8 条序列 | 每步动态匹配序列组的有效窗口数 N_t |
| 梯度累积 | 2 次序列 microbatch | ceil(N_t / 64) 次窗口 microbatch |
| padding | 形状补齐，监督掩码为零 | 尾部补齐，监督掩码为零 |
| loss 权重 | 每条序列内平均，再按序列平均 | 全局有效窗口平均 |
| TTT / register tokens | false / 0 | 相同 |
| 精度 | FP32 主参数、FSDP BF16 计算、FP32 reduce/buffer，额外 AMP 关闭 | 相同 |
| optimizer | AdamW，峰值 LR 2.5e-5，betas 0.9/0.95，eps 1e-8，weight decay 1e-5，clip 1.0 | 相同 |
| scheduler | 200 步预热 + 总计 10K 步余弦衰减 | 相同 |
| seed | 42 | 42 |

H40 配置中的 `actor.global_batch_size=1024` 是容量设置，不代表每步固定使用 1024 个有效窗口。两组预先确定的 10K 步预算总计为 **9,726,167 个有效窗口**；计划 SHA-256 为 `50e992ad432e046094c601c9e2aaa77f3c772166e2da5ef459cb7a620049f3a1`。实际运行尚未消耗完整预算。

等预算指同一步有效窗口数、optimizer 更新次数及 LR 计划相同，不代表样本身份、loss 权重、FLOPs 或墙钟时间相同。本实验比较采样与训练组织方式，两组均未启用 TTT，不能用于直接证明 TTT 的收益。

## 代码与依赖身份

- RLinf 基线 commit：`7e90641c4135229db6d3426950cdb45cee6259a6`。
- 原生 Isaac-GR00T：`https://github.com/NVIDIA/Isaac-GR00T`，commit `23ace64f17aa5015259b8609d371eb61a357c776`，服务器该依赖工作区干净。
- Python 和关键包版本：`source_environment.json`。
- 模型和预处理文件身份：`source_models.sha256`；逐次启动验证输出见两组 `logs/assets_verified.log*`。
- 实际启动命令、完整配置、源码哈希、GPU 映射和预算列表：两组 `logs/started.json*`。

代码调用链从仓库根目录的 `examples/sft/train_vla_sft.py` 开始，进入 `FSDPGr00tSequenceSftWorker`。`sequence_dataset.py` 按配置选择序列数据或 `window_dataset.py` 的独立窗口数据；`window_sampling.py` 负责预算和可恢复采样游标。两组通过同一个 `gr00t_n1d7/sft.py` 调用原生 GR00T 模型，使用同一个 SFT runner 和 FSDP checkpoint 链路。

## 文件用途

- `<group>/config.yaml`：6K 续训时的完整 Hydra 配置，含原服务器绝对路径。
- `<group>/config.yaml.before_resume6000`：首次运行配置，`resume_dir=null`。
- `<group>/logs/sequence_metrics.jsonl`：保留首次运行 1–6000 步，再追加续训记录；`step` 是已完成 update 数。
- `<group>/logs/*.before_resume6000*`：完整保留旧分支，可能包含旧的 6000 之后记录；不要直接与当前分支拼接。
- `<group>/logs/sequence_samples_rank*.gz`：逐 rank 原始采样日志的无损 gzip；解压内容可按 manifest 的原始 SHA-256 校验。
- `<group>/tensorboard/`：原始 event 文件、配置和重启标记。TensorBoard 横轴从 0 计步，6000 对应第 6001 次更新；重启标记隐藏旧分支 6000 及之后的事件，原文件保留。
- `run_experiment.sh`：原服务器的首次启动脚本快照，包含固定路径及防覆盖检查；不适用于直接恢复已有输出。
- `migration.md`：历史迁移记录；较早章节包含已被后续“Exact window-budget restart”取代的固定 1024 窗口方案，以本说明、实际配置和启动身份为准。

## 查看曲线与日志

在仓库根目录运行：

```bash
python -m tensorboard.main --logdir_spec=sequence128:experiments/baseline-sft-1019/sequence128/tensorboard,h40:experiments/baseline-sft-1019/h40/tensorboard --host 127.0.0.1 --port 6006
```

该命令从这份快照展示两组曲线，保留 TensorBoard 默认的 orphan-event purge 行为。检查原始曲线时将 smoothing 调为 0；不要把平滑后的显示当作逐步原始值。

```bash
gzip -cd experiments/baseline-sft-1019/h40/logs/sequence_samples_rank0.jsonl.gz | head -n 1
```

该命令查看一个采样记录而不覆盖或解压原文件到磁盘。

## 在另一台机器复现

先准备上述固定版本的 RLinf、Isaac-GR00T、模型、数据及兼容环境，并核对资产哈希。将 `config.yaml.before_resume6000` 复制为新运行目录中的 `config.yaml`，再明确设置数据、模型、依赖和输出路径，以及物理 GPU placement。两组并行运行需要互不重叠的四卡资源和独立 Ray 实例；仅设置 `CUDA_VISIBLE_DEVICES` 不足以替代 RLinf 的 placement 配置。

完成路径适配后，在仓库根目录调用：

```bash
python examples/sft/train_vla_sft.py --config-path /absolute/path/to/new-run --config-name config
```

该命令读取新目录的配置开始训练。环境变量和原服务器依赖路径可参考 `run_experiment.sh` 与 `logs/started.json`，不要直接执行包含旧输出路径的快照启动脚本。此快照未附完整环境安装锁文件；包版本记录不等于跨机器无条件可运行保证。

恢复 6K 时，必须提供具有 `_SUCCESS` 的完整 checkpoint，并恢复模型、AdamW、scheduler、数据游标及 RNG，不能只加载 `full_weights.pt`。`runner.resume_dir` 指向 `global_step_6000`，不是其 `actor` 子目录。源码、数据及优化协议的严格身份校验应保留；迁移路径后若校验失败，先核对原因，不绕过检查。

本次恢复保持 `max_steps=10000` 和原 scheduler，仅将保存间隔改为 2000，后续计划保存 8K、10K。已实测第 6000 步 LR 为 `8.947246897150762e-6`，续训第 6001 步为 `8.943405191530474e-6`，与旧记录对应步一致；没有重新预热或将学习率提升至 1e-5。日志 LR 有浮点归约舍入。

## 验证范围

2026-10-19 在原服务器固定环境下，隐藏 GPU 并关闭 pytest cache 运行四个相关测试文件：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' USE_TF=0 python -m pytest -p no:cacheprovider -q tests/unit_tests/test_gr00t_sequence_sft.py tests/unit_tests/test_gr00t_window_dataset.py tests/unit_tests/test_gr00t_window_integration.py tests/unit_tests/test_gr00t_window_sampling.py
```

结果为 **49 passed、2 skipped**；两项 CUDA FSDP 测试因 GPU 不可见而跳过，未执行整个 RLinf 测试套件。发布时保持实验源码原样，不为格式化修改已运行的代码。发布整理使用了 AI 助手。

## English Summary

This snapshot contains two native GR00T N1.7 SFT baselines: ordered sequences of up to 128 anchors and independent H40 windows matched to the sequence group's valid-window budget at each update. Both disable TTT. Window counts, optimizer-step budgets and LR schedules match; loss weighting, sampled windows and compute time differ. The runs resumed from full step-6000 state and stopped at steps 6788 and 6777 after simultaneous SIGTERM signals of unverified origin. Read the actual configs, startup identities and snapshot manifest for provenance. Model weights, datasets, checkpoints and environment binaries are external assets. Tests for the four affected components returned 49 passed and two CUDA-dependent skips; this is not a full-suite or completed-10K result.
