# RoboTTT 2.0

`robottt2.0` 是 RoboTTT 在 RLinf / GR00T N1.7 上的原生训练与闭环评测版本。
本版本不再维护一套独立的 RoboTTT trainer 和 evaluator，而是把序列训练、TTT
fast state、FSDP、checkpoint、RoboCasa365 环境交互和视频评测接入 RLinf 原生组件。

- Git 分支：`robottt2.0`
- 版本快照：`aa0ad850aef9456c009ad58ac48ca4f38beb76c2`
- 当前任务：RoboCasa365 `BowlAndCup`
- 当前训练配置：8 张 A800、8,000 steps、104 episodes
- 模型：GR00T N1.7 + Cosmos-Reason2-2B + RoboTTT

## 相较旧版代码的主要变化

### 1. 从独立实现迁移到 RLinf 原生架构

旧版 RoboTTT 使用独立的训练器、评测器和启动脚本，模型状态、分布式训练和环境
rollout 之间存在重复实现。2.0 将这些职责重新放回 RLinf 对应模块：

| 方面 | 旧版 | RoboTTT 2.0 |
|---|---|---|
| 训练入口 | 独立 RoboTTT trainer | `train_vla_sft.py` + `SFTRunner` |
| 训练 worker | 单独维护完整 worker | `FSDPVlaRoboTTTWorker` 继承 `FSDPVlaSftWorker` |
| 分布式训练 | 自定义处理较多 | 复用 RLinf FSDP、AdamW、scheduler 和 DCP |
| 评测 | 独立 evaluator | `EmbodiedEvalRunner` + 原生 env/rollout worker |
| 环境交互 | 独立 rollout 状态 | RLinf channel、stage 和 batch 路由 |
| 工具目录 | `rlinf/robottt`、`scripts/robottt` | 核心归入原模块，审计工具集中到 `toolkits/robottt` |

旧的活动目录和重复入口已经移除。这样做的目的不是改变 RoboTTT 数学，而是减少两套
训练/推理协议长期漂移的风险。

### 2. TTT memory 改为显式所有权

TTT fast state 不再隐藏在模型内部作为隐式上下文。2.0 明确区分：

- 模型接收 incoming memory，并返回 candidate memory；
- rollout worker 按 `(mode, stage_id)` 持有状态；
- 每个 action chunk 只在 worker 中提交一次；
- `reset_mask` 只重置真正发生环境 reset 的 batch 行；
- `done` 与 `reset_mask` 分离，避免把“episode 已结束但环境尚未 reset”误判为新轨迹；
- 切换 checkpoint 时清空旧 memory，防止跨权重污染。

评测日志会记录 position、forward、commit、reset、动作形状和有限性，可审计状态是否
重复写入或漏重置。

### 3. TTT/DiT/flow 协议统一

2.0 在 GR00T N1.7 的 32 层 DiT 中，在 attention 与 FFN 之间插入 TTT residual，
并实现：

- 多头两层 GELU fast MLP；
- LayerNorm 残差重建；
- 可学习 inner learning rate；
- prefix dual update 和 meta-gradient；
- 16 个 register tokens；
- 4 次 Euler denoising；
- final prediction 额外完整 forward；
- 仅 final prediction 可以产生一次 memory commit candidate。

训练、恢复和评测共用同一模型 factory、processor bundle 与 memory 语义，避免训练和
推理采用不同的 TTT 写入规则。

### 4. 数值精度和 FSDP 行为重新约束

旧版中 fast state、输入 cast、register 参数和 FSDP 包装之间容易产生隐式精度变化。
2.0 的约束为：

- 模型 master parameters 为 FP32；
- 外层主干使用 BF16 autocast；
- TTT inner loop、fast state 和 optimizer moments 保持 FP32；
- TBPTT 每 16 个 anchors detach 一次状态；
- 一条序列只执行一次 optimizer update；
- 非最后 micro-batch 的 forward/backward 均处于 `no_sync`；
- 非有限梯度直接失败，不自动掩盖或重新初始化；
- checkpoint 使用 RLinf 原生 DCP，并额外导出 `full_weights.pt`。

register tokens 作为普通 Parameter 交给 FSDP 管理，不再旁路分片或手写梯度
all-reduce。

### 5. 数据读取改为真实序列监督

训练数据不再把单步样本伪装成序列。当前实现直接读取 LeRobot episode，并使用：

- 三路相机；
- 16 维机器人状态；
- 12 维 RoboCasa365 原生动作；
- 真实 40-step action horizon；
- stride 8；
- 每条 episode 最多 128 个 anchors；
- 104 条 episode 等权采样；
- 确定性的 episode 顺序、窗口选择和图像增强；
- 独立的 `anchor_valid`、`supervision` 和 `action_mask`。

所有有效 anchor 必须监督完整的 `40 x 12` 动作；数据不足时不允许静默 padding。
StatefulDataLoader 会保存 sampler epoch/order 和已消费 cursor，使恢复位置不受 worker
预取影响。

### 6. RoboCasa365 观察和动作对齐

2.0 统一训练与评测的模态定义：

- 相机：左侧 agent view、右侧 agent view、eye-in-hand；
- 状态：base pose、base-to-EEF pose、gripper qpos，共 16 维；
- 动作：EEF position/rotation、gripper、base motion、control mode，共 12 维；
- gripper 与 control mode 在阈值 `0.0` 上投影为 `-1/+1`；
- 每次执行前检查动作 shape 和 NaN/Inf。

该转换直接输出 robosuite controller 所需的原生动作顺序，避免旧版训练字段顺序与
仿真动作顺序不一致。

### 7. 闭环评测可复现并保存视频

评测现在复用 RLinf 原生 `EnvWorker` 和 `MultiStepRolloutWorker`，支持：

- checkpoint 自带 processor bundle；
- 固定 seed 和 reset state；
- 使用 reset observation SHA256 核对不同模型是否看到同一初始场景；
- 2400-step BowlAndCup horizon；
- 逐 checkpoint 独立输出指标、TTT state 日志和 MP4；
- `success_once`、`success_at_end`、return、episode length；
- 视频帧数、分辨率和完整解码审计。

已经完成的 seed-7 单环境检查中，1K 和 2K checkpoint 均能稳定运行 2400 步且动作
全部有限。2K 可以将杯放入碗并把碗移动到目标台面，但尚未稳定学会最后的夹爪撤离，
因此该次 `success_once=0`。单 seed 结果只用于闭环诊断，不代表统计成功率。

### 8. 配置改为可组合的 Hydra defaults

训练和评测不再各自复制一份完整 YAML。公共内容拆分为：

- GR00T/RoboTTT 模型配置；
- BowlAndCup 模态与 checkpoint 路径；
- A800/H100 资源配置；
- baseline 与 RoboTTT 的最小差异；
- 公共评测环境配置。

这使 baseline 和 RoboTTT 可以共享数据、optimizer、相机、动作和 horizon，只保留
TTT/register/checkpoint 等必要差异。

## 当前训练配置

当前 `robocasa_bowlandcup_robottt_a800.yaml` 的关键参数如下：

| 参数 | 值 |
|---|---:|
| GPUs | 8 x A800 |
| Episodes | 104 |
| Max steps | 8,000 |
| Global batch size | 8 |
| Micro batch size | 1 |
| Peak learning rate | `5e-5` |
| Warmup | 200 steps |
| Scheduler | linear warmup + cosine |
| Action horizon | 40 |
| Execution chunk | 8 |
| Sequence stride | 8 |
| Max anchors | 128 |
| TBPTT segment | 16 |
| Register tokens | 16 |

## 快速开始

### 训练

```bash
source /root/workspace/RLinf-gr00t-robocasa/.venv/bin/activate
cd /path/to/RoboTTT

bash examples/sft/run_vla_sft.sh \
  robocasa_bowlandcup_robottt_a800.yaml \
  runner.logger.log_path=/path/to/output
```

### 从 checkpoint 恢复

```bash
bash examples/sft/run_vla_sft.sh \
  robocasa_bowlandcup_robottt_a800.yaml \
  runner.resume_dir=/path/to/checkpoint \
  runner.logger.log_path=/path/to/output
```

恢复会严格检查数据摘要、processor、模型参数统计、源码摘要、optimizer、scheduler、
RNG 和 dataloader cursor。

### 评测并保存视频

```bash
bash evaluations/run_eval.sh \
  robocasa365_bowlandcup_robottt_h100.yaml \
  runner.ckpt_path=/path/to/checkpoint/actor/model_state_dict/full_weights.pt \
  runner.logger.log_path=/path/to/eval_output \
  env.eval.video_cfg.video_base_dir=/path/to/eval_output/video
```

完整的多 checkpoint 评测入口为：

```bash
bash evaluations/run_bowlandcup_full_eval.sh
```

## 代码位置

| 功能 | 位置 |
|---|---|
| TTT、DiT、flow、policy adapter | `rlinf/models/embodiment/gr00t/gr00t_n1d7/` |
| RoboCasa365 模态转换 | `rlinf/models/embodiment/gr00t/simulation_io.py` |
| 序列数据集与 sampler | `rlinf/data/datasets/gr00t_n1d7/` |
| RoboTTT SFT worker | `rlinf/workers/sft/fsdp_vla_robottt_worker.py` |
| Rollout memory 所有权 | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| 环境 reset_mask | `rlinf/workers/env/env_worker.py` |
| 训练配置 | `examples/sft/config/` |
| 评测配置 | `evaluations/robocasa365/` |
| 单元测试 | `tests/unit_tests/test_robottt_*.py` |
| 审计工具 | `toolkits/robottt/` |

## 已知限制

### 终止阶段采样不足

BowlAndCup 的 104 条训练轨迹都包含成功撤离，末尾均有 16–29 个成功帧；但当前
`max_anchors=128` 的连续窗口会低估长轨迹末段。按实际 seed 重放前 182 个 epoch：

- 仅约 `0.776%` 的 action windows 覆盖首次成功转换；
- 每个 epoch 平均约 49/104 条轨迹采到成功末段；
- 65 条长轨迹中，成功转换被采到的 epoch 数中位数为 20/182；
- 最后一帧被监督的 episode-epoch 比例约为 `14.88%`。

因此 2.0 已经保留终止数据，但终止动作权重仍偏低。后续版本应在保持连续 TTT
context 的前提下加入右对齐/terminal-balanced window，或提高可覆盖的 anchors 数量。

### 评测统计量仍不足

当前 1K/2K 结论来自单 seed 闭环诊断。正式比较 checkpoint 时应使用多个配对 seed，
报告置信区间，并同时查看 `success_once` 与 `success_at_end`。

## 更多文档

- [逐文件修改说明](docs/robottt/file_changes.md)
- [原生 RLinf 组件重组](docs/robottt/native_integration.md)
- [数学协议、运行与恢复](docs/robottt/reproduction.md)
- [配置与 worker 重构](docs/robottt/30k_worker_refactor.md)
- [参考提交审查](docs/robottt/reference_review.md)
- [逐文件参考映射](docs/robottt/reference_commit_map.csv)

训练集 loss、短 rollout 和单 seed 视频仅能证明执行链路与局部行为，不能单独作为
泛化性能证据。
