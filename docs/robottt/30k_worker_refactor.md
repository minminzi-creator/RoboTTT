# 30K 训练、train/eval 配置复用与独立 RoboTTT worker

本轮在已有原生 RLinf 集成上继续修改，回应三个要求：默认 30K、复用 Hydra 配置、
将序列训练迁入 `fsdp_vla_robottt_worker.py`。两份 eval YAML 也改为 defaults 复用；模型数学和评测交互没有在本轮改写。
完整仓库的逐文件职责见 [file_changes.md](file_changes.md)。

## 1. 配置如何复用

两份主配置仍位于 `examples/sft/config/`，均使用：

```yaml
defaults:
  - model/gr00t_n1d7@actor.model
  - hybrid_engines/fsdp@actor.fsdp_config
  - robottt/bowlandcup_h100@_global_
  - _self_
  - override hydra/job_logging: stdout
```

训练组合顺序是 **GR00T model → 原有 FSDP → BowlAndCup 公共实验（含共享模型适配组）→ 实验组差异**。

| 文件 | 承担的配置 | 不在两份主 YAML 重复的原因 |
|---|---|---|
| `examples/sft/config/model/gr00t_n1d7.yaml` | GR00T 序列 SFT、FP32 参数、4 次去噪、执行 8、TBPTT 16、TTT 架构及初始化；默认关闭 TTT/register | 两组必须使用同一个网络和前向协议；保留 guozhen 的 model group 扩展位置，去除与当前 SFT 无关的旧 PPO/value/noise 字段 |
| `examples/sft/config/hybrid_engines/fsdp.yaml` | 仓库原有 FSDP strategy、sharding、offload、prefetch、mixed precision、AMP、GradScaler 默认值 | 本文件未修改，直接复用；不再把每一个默认值复制进实验 YAML |
| `examples/sft/config/robottt/bowlandcup_model.yaml` | H100 checkpoint/Cosmos 路径、BowlAndCup embodiment/模态维度 | 训练与评测共同加载，避免各自维护一份模型适配参数 |
| `examples/sft/config/robottt/bowlandcup_h100.yaml` | H100 路径、104 episodes、H40/stride8/128 anchors、8 卡 batch、30K optimizer 日程、日志/保存与必要 FSDP 覆盖 | 保证数据、优化器、精度和采样对两组完全一致；同一处修改即可同时生效 |
| `examples/sft/config/robocasa_bowlandcup_gr00t_baseline_h100.yaml` | defaults、日志路径和实验名 | 继承 `gr00t_n1d7`、`use_ttt: false`、0 register |
| `examples/sft/config/robocasa_bowlandcup_robottt_h100.yaml` | defaults、日志路径/实验名、`model_type: robottt`、`use_ttt: true`、16 register | 仅列出完整 RoboTTT 实验相对 baseline 的差异 |

公共实验仅覆盖 FSDP 所需差异：启用 gradient checkpointing、`use_orig_params`、
`limit_all_gathers`、BF16 autocast；关闭 checkpoint reentrant 与输入自动转换；
启用完整权重导出。三种 mixed-precision dtype 的 null 值、BF16 autocast 的 dtype、
关闭 GradScaler 等直接来自原有 FSDP group。最终参数/梯度/Adam/fast state/通信仍为 FP32。

`_self_` 排在公共配置之后，主 YAML 和命令行覆盖都能生效。配置不依赖新增环境变量
或运行时手工 merge。`bash examples/sft/run_vla_sft.sh xxx.yaml`、文件全路径、Hydra
overrides 三种既有调用均保留。

### 评测也复用原生环境和训练模型

两份 `evaluations/robocasa365/robocasa365_bowlandcup_*_h100.yaml` 都使用：

```yaml
defaults:
  - env/robocasa365@env.eval
  - model/gr00t_n1d7@rollout.model
  - robottt/bowlandcup_model@rollout.model
  - robottt/bowlandcup_eval@_global_
  - _self_
  - override hydra/job_logging: stdout
```

| 评测配置来源 | 用途 |
|---|---|
| 原有 `examples/embodiment/config/env/robocasa365.yaml` | 直接继承 robot、相机、观察 key map、action space 和一般环境行为；此原文件不修改 |
| SFT 的 `model/gr00t_n1d7.yaml` | 训练/评测共用 GR00T/TTT 架构、精度、denoising、memory 写入设置 |
| SFT 的 `robottt/bowlandcup_model.yaml` | 训练/评测共用 checkpoint/Cosmos 路径、12维动作和3相机等适配字段 |
| `evaluations/robocasa365/robottt/bowlandcup_eval.yaml` | 公共 runner/rollout 与完整 BowlAndCup env 覆盖：8env、一轮、task horizon 2400、相机256、16维state、seed7/global_unique、视频和动作二值转换约定 |
| 两份 eval 主 YAML | 各自最终 8K checkpoint、实验名和日志路径；RoboTTT 另启用 model alias、TTT 和16register |
| `evaluations/run_bowlandcup_full_eval.sh` | 激活统一环境，验证 8 张可见 GPU 和六个 checkpoint，随后按 2K/4K/8K 逐 step 配对、严格串行运行 baseline 与 RoboTTT |

训练公共实验通过自身的 `defaults` 加载共享 `bowlandcup_model@actor.model`，
评测主配置加载同一文件至 `rollout.model`。字段级审计要求每组
`actor.model == rollout.model`，并要求 eval 重构前后的所有解析字段完全相等。
不把环境基类的全部默认值复制到公共评测文件，只写实际差异。

原 `evaluations/run_eval.sh` 已提供 `REPO_PATH` 与 `EMBODIED_PATH`；主 eval YAML
的 `hydra.searchpath` 先搜索 `${REPO_PATH}/examples/sft/config/`，再搜索
`${EMBODIED_PATH}/config/`。顺序确保选择共享序列模型，而不会误选同名 PPO 模型。
Hydra 要求 searchpath 在主配置声明，故这两行保留在两个主 YAML 中。没有新增环境
变量要求；单组仍以 `bash evaluations/run_eval.sh xxx.yaml` 启动，两组正式比较以
`bash evaluations/run_bowlandcup_full_eval.sh` 启动。

两份主 YAML 默认指向已经完成的 global_step_8000 `full_weights.pt`。可继续通过
`runner.ckpt_path=.../actor/model_state_dict/full_weights.pt` 显式覆盖。

## 2. 30K 日程的具体定义

| 项目 | 上一版默认 | 当前默认 | 理由 |
|---|---:|---:|---|
| `runner.max_steps` | 500 | 30000 | 用户指定正式训练长度 |
| `actor.optim.total_training_steps` | 50000 | 30000 | cosine 衰减周期随正式训练长度调整 |
| `actor.optim.lr_warmup_steps` | 200 | 120 | 保留既定 warmup 比例：200/50000 = 120/30000 = 0.4% |
| `actor.optim.lr` | 5e-5 | 5e-5 | 全参数训练和 global batch 8 未变；训练步数缩短本身不构成线性缩放峰值 LR 的依据 |
| `runner.save_interval` | 250 | 2000 | 沿用参考长训的保存间隔，避免对完整模型/Adam 过于频繁地保存；30000 可被 2000 整除 |

AdamW betas=(0.9,0.95)、eps=1e-8、weight decay=1e-5、clip norm=1 保持一致。
这些是训练协议选择，不能把短 smoke 的通过解释为证明 120-step warmup 最优。

guozhen 最早提交中的 1e-4 / 500-step warmup / 50K，配合当时的 TTT-only 训练配置；
我们先前确认的协议是全参数 5e-5 / 200-step warmup / 50K。本次按后者比例调整，
不因恢复其代码结构而改变已经确认的训练范围或照搬其优化器超参数。

令 `s` 为已完成的 optimizer update 数，下一次更新使用的 LR 是：

```text
s < 120:    lr(s) = 5e-5 * (s + 1) / 120
s >= 120:   lr(s) = 5e-5 * [1 + cos(pi * min((s - 120) / 29880, 1))] / 2
```

首步为 `4.1666666667e-7`；第 120 次更新达到峰值，第 121 次仍在 cosine 起点；
随后下降；完成第 30000 次更新后 scheduler 的下一步 LR 为 0。
保留既有 scheduler 的边界约定，没有另写 scheduler。

`runner.max_steps` 是停止点，`actor.optim.total_training_steps` 是 schedule 总长，
两者默认都是静态 30000。**短验收只覆盖停止点**，不会把 cosine 压成 3 step。
如果将来修改正式总长，应同时修改这两个字段并明确 warmup。

## 3. worker 继承、职责和与参考提交的差异

```text
SFTRunner
  └─ FSDPVlaRoboTTTWorker
       └─ FSDPVlaSftWorker
            └─ FSDPSftWorker
                 └─ FSDPModelManager + Worker
```

`examples/sft/train_vla_sft.py:get_sft_worker_class` 根据 `sequence_training` 路由：
GR00T baseline 和 RoboTTT 两组均使用 `FSDPVlaRoboTTTWorker`；不启用序列训练的
其他 VLA 配置继续使用原生 `FSDPVlaSftWorker`。对不支持的 sequence model 早报错。
这样 baseline 也采用相同 episode 窗口、有效 anchor 分母、TBPTT 分段和 RNG 策略。

`rlinf/workers/sft/fsdp_vla_sft_worker.py` 恢复为基线提交
`3cb54c2eb2e355378ba049d50f7ec0cb8acc2da0` 中的原文件。该通用类不再包含
RoboTTT 分支。它原有的模型数据集分派、StatefulDataLoader `data.pt` 和每 rank
`rng.pt` 保存/恢复逻辑都由子类通过 `super()` 复用。

新子类只实现以下边界：

| 方法 | 专用职责 | 复用的父类职责 |
|---|---|---|
| `model_provider_func` | 共同初始化种子、wrap 前参数统计 | 调用公共 `get_model` factory 和原有模型构建 |
| `init_worker` | rank 独立编译缓存、FP32 matmul、rank RNG、写入解析配置与源码摘要 | `setup_model_and_optimizer`、FSDP 包裹、Adam、scheduler |
| `build_dataloader` | 接入 `build_gr00t_sequence_dataloader` | 原生 StatefulDataLoader 与父类数据迭代 |
| `forward_backward_batch` | 16-anchor TBPTT、全序列分母、段间 detach、非有限 loss 集体检查、采样/精度记录 | `before_micro_batch`、GradScaler；optimizer 不在此执行 |
| `run_training` | 在父类返回后追加 sequence JSONL 指标 | `super().run_training()` 执行迭代、梯度裁剪、一次 optimizer update、scheduler 和跨 rank 指标归约 |
| `save_checkpoint` | 在父类保存后追加 processor、metadata、epoch/offset/step | 原生 DCP 模型/Adam/scheduler，通用 VLA 的 data/RNG 保存 |
| `load_checkpoint` | 父类加载前核对协议/源码/数据和必需文件；加载后恢复序列计数 | 原生 DCP、StatefulDataLoader、每 rank RNG 恢复 |

审阅了 `55ed493` 与 `9d0e6d6` 的 `FSDPRoboTTTSftWorker(FSDPVlaSftWorker)`、
`train_vla_sft.py` 和 model/defaults 配置。保留其“子类扩展 + 入口选择”的结构；
文件命名按本次要求使用 `fsdp_vla_robottt_worker.py`。

参考实现把中间段 backward 放进 `get_train_model_output`，最后一段交给父类。
当前使用此前已验证的 `forward_backward_batch` hook：每段 forward 和 backward
处在同一个 `before_micro_batch` 中；仅最后一个 micro-batch 的最后一段同步。
因为 FSDP 在 forward 安装 backward hooks，只在 backward 外围套 `no_sync`
不足以表达该语义。loss 使用整条序列真实监督 anchor 数为分母，不能用固定 segment
长度重新加权短轨迹。fast state 显式传入/返回，不恢复隐式可变 context。

## 4. 审计与恢复兼容性

YAML 变为组合文件后，`yaml.safe_load` / `OmegaConf.load` 只能看到叶子字段，
不会展开 defaults。`sequence_utils.load_sft_config` 使用 Hydra compose；
`audit_sampler.py`、`audit_video_alignment.py`、`diagnose_history.py` 统一调用它。
训练本身仍由原生 Hydra 入口解析，不增加第二套配置合并规则。

源码摘要新增独立 worker 和入口文件。checkpoint metadata 另外记录 `worker_class`
及完整 `resolved_config`，其 `protocol.actor/data` 用于严格恢复核对；runner 停止点
和日志路径可以改变。optimizer schedule 属于 actor 协议，不能在严格恢复中悄悄修改。

因此旧 50K schedule 的原生 checkpoint 不能直接当作当前 30K 协议的完整恢复，
旧版本源码也会被摘要检查拒绝。这是有意的严格契约，不会放宽检查或自动重置 optimizer。
历史完整模型权重仍可由既有评测入口加载；本轮不实现跨日程的训练状态迁移。

## 5. 启动与检查

```bash
source /home/xuzhexuan/project/.venv_robottt_robocasa/bin/activate
cd /home/xuzhexuan/project/repos/robottt

# 正式默认 30K；两组各占全部 8 卡，需要依次运行。
bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_gr00t_baseline_h100.yaml
bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_robottt_h100.yaml

# 只打印实际继承结果，不启动模型训练。
bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_robottt_h100.yaml --cfg job --resolve
bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_robottt_h100.yaml --info defaults

# 3-step smoke；保持完整数据窗口、32 层 DiT、30K scheduler。
LOG_PATH=/absolute/run_dir bash examples/sft/run_vla_sft.sh \
  robocasa_bowlandcup_robottt_h100.yaml runner.max_steps=3 runner.save_interval=2

# 同一代码/协议下从 step 2 恢复到 step 3。
LOG_PATH=/absolute/resume_dir bash examples/sft/run_vla_sft.sh \
  robocasa_bowlandcup_robottt_h100.yaml runner.max_steps=3 runner.save_interval=2 \
  runner.resume_dir=/absolute/run_dir/robottt/checkpoints/global_step_2

USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 python -m pytest -q \
  tests/unit_tests/test_robottt_sft_config.py \
  tests/unit_tests/test_robottt_rlinf_integration.py \
  tests/unit_tests/test_robottt_reproduction.py

# 评测默认及配置继承检查。
bash evaluations/run_eval.sh robocasa365_bowlandcup_robottt_h100.yaml --cfg job --resolve
bash evaluations/run_eval.sh robocasa365_bowlandcup_robottt_h100.yaml --info defaults
```

本轮证据存放 `artifacts/worker30k/`，对应训练目录是
`/home/xuzhexuan/project/robottt_runs/worker30k/`。原 `artifacts/reorganization/`
和 `artifacts/acceptance/` 保持历史含义；其中的 500-step/50K 日程不能标成 30K 新运行。
本次只进行必要短运行验证，没有启动 30K 长训练。
