# RoboTTT 数学与原生 RLinf 运行协议

训练使用 `FSDPVlaRoboTTTWorker(FSDPVlaSftWorker)`，评测使用 `EnvWorker` / `MultiStepRolloutWorker` /
`Robocasa365Env`；共用 RLinf `get_model` → GR00T factory 和 checkpoint processor。
组件归属及参考 FSDP 决策见 [native_integration.md](native_integration.md)。

## H100 运行命令

```bash
source /home/xuzhexuan/project/.venv_robottt_robocasa/bin/activate
cd /home/xuzhexuan/project/repos/robottt

bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_gr00t_baseline_h100.yaml
bash examples/sft/run_vla_sft.sh robocasa_bowlandcup_robottt_h100.yaml

# 只覆盖停止点，不压缩默认 30K schedule。
LOG_PATH=/absolute/run_dir bash examples/sft/run_vla_sft.sh \
  robocasa_bowlandcup_robottt_h100.yaml runner.max_steps=3 runner.save_interval=2

# 恢复目录是 global_step_N，内部包含 actor/；用新日志目录保留原记录。
LOG_PATH=/absolute/resume_run bash examples/sft/run_vla_sft.sh \
  robocasa_bowlandcup_robottt_h100.yaml \
  runner.resume_dir=/absolute/run_dir/robottt/checkpoints/global_step_2

# 串行评测 baseline/RoboTTT 的 2K、4K、8K checkpoint；每次占用全部 8 卡。
bash evaluations/run_bowlandcup_full_eval.sh

# 仍可单独运行，并通过 Hydra 覆盖 checkpoint 或输出目录。
LOG_PATH=/absolute/eval_output bash evaluations/run_eval.sh \
  robocasa365_bowlandcup_robottt_h100.yaml \
  runner.ckpt_path=/absolute/run_dir/robottt/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

# 各项命令依次运行；默认每个训练/评测占用全部 8 卡。
USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 python -m pytest -q \
  tests/unit_tests/test_robottt_reproduction.py \
  tests/unit_tests/test_robottt_rlinf_integration.py \
  tests/unit_tests/test_robottt_sft_config.py
PYTHONPATH=. python toolkits/robottt/audit_sampler.py \
  --config examples/sft/config/robocasa_bowlandcup_robottt_h100.yaml --output /absolute/sampler.json
PYTHONPATH=. python toolkits/robottt/audit_video_alignment.py \
  --config examples/sft/config/robocasa_bowlandcup_robottt_h100.yaml --output /absolute/video_alignment.json
PYTHONPATH=. python toolkits/robottt/audit_resume.py \
  --uninterrupted /absolute/original/global_step_3/actor \
  --resumed /absolute/resumed/global_step_3/actor --output /absolute/resume.json
PYTHONPATH=. python toolkits/robottt/diagnose_history.py \
  --config examples/sft/config/robocasa_bowlandcup_robottt_h100.yaml \
  --checkpoint /absolute/full_weights.pt --output /absolute/history.json --anchors 32
```

每个完整 BowlAndCup 实验运行 8 个并行环境、每个环境最多 2400 个 simulator step
（300 次 8-action policy call），每个 checkpoint 共 8 个 trial。六个实验总计 48 个
trial，按 2K baseline、2K RoboTTT、4K baseline、4K RoboTTT、8K baseline、8K
RoboTTT 严格串行。成功会提前结束；失败在 2400 步按 task horizon 截断。8 个环境
使用 seed 7..14，两种模型在同一 step 得到同一组确定性初始状态；视频默认保存并
叠加 success/episode length 信息。
可用 Hydra `env.eval.video_cfg.video_base_dir=...` 指定评测视频目录。
`runner.logger.log_path` 默认由 bash 入口按时间生成，`LOG_PATH` 可固定它。
训练 YAML 通过 model/FSDP/公共实验 defaults 复用；默认 30K、warmup 120。
评测 YAML 继承原生 RoboCasa365 env、共享训练 model/模态和公共 eval 组。
[配置继承与逐字段说明](30k_worker_refactor.md)；[逐文件修改目录](file_changes.md)。
训练为全参数、seed 42、global batch 8、micro batch 1、TBPTT 16、104 episode。

## 数学与状态契约

一个 anchor 的 token 顺序为 16 register、1 state、40 action。VL token 只参与
单步 cross-attention。32 层 DiT 的每层均为原 attention residual → TTT residual
→ 原 FFN residual。baseline 保持 41 个 token，不含 register 或 TTT。

快速模型为按头分组的两层 MLP，hidden ratio 4，tanh 近似 GELU。令
`f_W(x) = GELU(x W1 + b1) W2 + b2`，inner loss 是

\[
\ell_j(W)=\tfrac12\|\mathrm{LN}(f_W(k_j))-(v_j-k_j)\|_2^2.
\]

所有 token 的梯度均在同一个 incoming W 处求值；第 s 个 query 使用 inclusive
prefix 的 W。学习率为

\[
\eta_{sj}=\max\{(s+1)^{-1}+b_s,0\}\frac{0.1}{d_h}
\sigma(\theta_\eta x_j+c_\eta).
\]

`dual_update` 通过解析的 LayerNorm/GELU 导数计算梯度，保留完整计算图；没有
detach inner gradient 或 first-order 近似。独立 primal 测试逐 token 调用
`autograd.grad(create_graph=True)`，对齐输出、最终 state、输入梯度及所有慢参数
梯度。跨 anchor 必须递归使用前一个 anchor 的最终 prefix；16-anchor TBPTT
segment 不是一次 TTT mini-batch。

H100 配置启用 `compile_inner: true`，仅用 `torch.compile(fullgraph=True)` 融合纯函数
`scan_update`；递归和 padding 选择不变，缓存只保存算子、不保存 memory。CPU 数学
测试始终使用 eager 参考。真实 `[2,16,57,1536]` GPU 验证覆盖 checkpoint 重算、
外层梯度、detach/reset 和全 padding 行：FP32 最大相对 L2 误差小于 `8e-7`；
BF16 外层的最大参数梯度差为 0.151%，fast state 差小于 `5e-7`。BF16 舍入边界
会放大微小 FP32 差异，故不承诺两种实现逐位相同。TF32 仍关闭。

运行 `PYTHONPATH=. python toolkits/robottt/audit_compilation.py --output AUDIT.json`
可重做该审计。该脚本的 GPU、完整尺寸和编译耗时使其不属于 CPU 单元测试。

gate 按 [RoboTTT Eq. (3)](https://arxiv.org/html/2607.15275#S3.SS1)
设置为每层 1536 维向量，所有元素初值 0.001。`o_proj` 标准差 0.02，保证初始
W0/QKV/inner-LR 的 outer gradient 通路存在。fast W1/W2 标准差 0.02，bias 为零；
inner LN gamma/beta 为 1/0。保留原 action position embedding。
沿用参考代码的 `alpha=tanh(gate)` 参数化；日志记录的是未经过 tanh 的 gate。

训练 flow path 为 `x_tau=(1-tau)*noise+tau*GT`，target 为 `GT-noise`，每个
anchor 独立采样 `tau=.999*(1-U**(1/1.5))`。这是 Beta(1.5,1) 的逆 CDF。
训练 candidate state 来自这次 noisy-GT forward。

`FastState` 仅包含四个 fast tensor；`Memory` 包含各层 state 与每条轨迹已读入的
有效 anchor 数。`forward(batch, incoming)` 返回 loss 与 candidate state，不能
修改 incoming。提供 initialize/reset/detach/reorder；FSDP 训练用 incoming=None
在各层参数 unshard 后初始化，避免访问分片参数。`detach` 只截断图。

默认位置为 `57*anchor+slot`。中间或尾部 padding 都不更新 state 或位置。另实现
shared-temporal 与 slot-only 模式并覆盖测试；本轮不训练这些变体。

推理的四次 Euler forward 都读取同一个 incoming state。RoboTTT 在最终 x1 上
以 bucket 999 额外前向一次，忽略动作输出，返回 x1 和唯一 candidate。调用者只
提交一次。baseline 没有 memory，因此只做四次前向。`denoise_step` 支持 -1 或
0..N-1，切换写入来源不改变本次动作。checkpoint 重算不做持久提交。
评测使用 Hydra 字段 `rollout.model.memory_write_source` 和
`rollout.model.memory_write_denoise_index` 覆盖写入选项，实际前向数写入 rollout state 审计。

## 数据与 loss

104 条 episode 全部训练，不划验证集。每 epoch 对全体 episode 做一次确定性
打乱，每条 episode 取一个窗口，再按 rank 切分；104 可被 8 整除，不通过复制
样本填满 DistributedSampler。窗口与图像增强种子由 seed、epoch、episode 及
用途决定，worker 数和预取不改变结果。

\[
T_i=\min\{128,\lfloor(L_i-40)/8\rfloor+1\},\qquad
s_i\sim\mathrm{UniformInteger}[0,L_i-40-8(T_i-1)].
\]

监督帧是 `s_i+8*t+[0,...,39]`；`allow_padding=False` 用于真实数据读取。
不足 128 个 anchor 的部分仅复制处理后的 payload，三个 mask 明确区分：
anchor_valid 控制 memory/位置，supervision 控制监督，action_mask 控制有效动作维度。
padding 不贡献 loss。真实无监督 context 可以更新 memory。

每个 anchor 先对有效动作值求 MSE，再按每条序列的监督 anchor 数平均，最后对
batch 平均。每个 TBPTT segment 使用整条序列的分母，因此短轨迹与长轨迹权重相同。
各 rank 固定执行 8 个 segment，包括完全 padding 的 segment；只在整条序列结束
后更新一次 optimizer。

相机顺序为 left/right/wrist；state 为 base position、base quaternion、相对 EEF
position、相对 EEF quaternion、gripper qpos，共 16 维。动作按数据 modality
顺序拼接为 12 维，解码后按 RoboCasa 的 EEF position、rotation、gripper、base
motion、control mode 顺序执行。全数据审计确认两个离散控制量只取 -1/+1；主评测
以 0 为阈值做最近邻投影。参考 RoboCasa converter 的 0.5 阈值更偏向 -1，保留
`rollout.model.binary_action_threshold=0.5` 显式兼容选项。执行配置随 Hydra 日志保存。

归一化使用数据集提供的 percentile statistics，保留 GR00T 的图像裁剪、颜色增强
和 state dropout。训练与评测使用 checkpoint 中同一份完整 processor bundle。
基础配置的 processor 与 action head 各保留 0.2 的 state dropout；独立执行时
总丢弃概率为 0.36，两组一致。

另对全部 312 段视频做只读容器/索引审计：三路相机均直接按名称匹配，不使用位置
fallback；帧数等于对应 Parquet 行数，分辨率 256×256，起点为 0、帧率为 20 fps。
Parquet frame_index 从 0 连续递增，timestamp 相对 frame_index/20 的最大误差为
`3.052e-6` 秒。这验证加载索引的对齐，不代表独立测量了原始传感器的物理同步。

## 精度、初始化与恢复

FP32 参数、梯度、Adam moments、fast state；FSDP mixed_precision=None，参数
通信保留 FP32，避免递归转换 incoming state。BF16 autocast 用于主干及外层线性
算子；inner-loop、flow time、bucket、ODE 累加和 loss 显式保持 FP32。
FP32 matmul precision 为 highest。原生首步记录参数、梯度、Adam moments 与 fast-state dtype；数学与编译审计
另外覆盖 inner/meta-gradient。逐步 loss、grad norm、LR、有效 anchor、fast-state 更新
写入 RLinf logger 和 sequence_metrics.jsonl（有效 anchor 是 rank 均值）。
TorchInductor 缓存按 rank 分目录，避免其 pad-matmul JSON 缓存的并发写入冲突。

基础 checkpoint 严格完整加载后才创建 TTT/register。没有参数大小启发式、自动
重新初始化、非有限值替零或 fast-state 自动恢复。非有限 loss 先跨 rank 归约，再报错停止；原生 optimizer 对非有限 grad norm
停止更新，rollout 对非有限动作和 candidate state 报错；异常保留在原生运行日志。

checkpoint 沿用原生 RLinf：`global_step_N/actor/dcp_checkpoint` 保存 model、Adam、
scheduler、RNG；`model_state_dict/full_weights.pt` 是标准完整模型导出。额外保存
`data.pt`、每 rank `rng.pt`、`sequence_progress.pt`，以及 `sequence_metadata.json`
和 `processor_bundle.json`。恢复严格核对 actor/data 配置、processor、数据摘要、
源码摘要、world size、参数统计，并加载全部状态。

当前默认 30K；学习率首个更新为 `5e-5/120`，第 120 个更新达到峰值，
第 121 个更新位于 cosine 起点；完成 30000 次更新后 scheduler 的下一步 LR 为 0。
旧 50K schedule checkpoint 不自动迁移为 30K 完整恢复；日程与源码检查都会严格拒绝。
原生 optimizer/scheduler/clip/grad reduction 均由 RLinf 执行。重组前 500-step
实现使用 fused AdamW；原生 FSDP 采用框架已有 AdamW 策略，不能据此宣称新旧完整
训练轨迹逐位相等。原生内部的 checkpoint 续训有单独逐位审计。旧 500-step 模型权重仍可评测；
旧独立 trainer 的 optimizer 状态不自动迁移为 RLinf DCP，也不当作原生完整恢复。

## 比较与结论边界

比较“原始 GR00T”与“TTT + register”完整方案，不推断两个组件各自的贡献。保留该
GR00T checkpoint 配置的完整 Cosmos backbone 和 32 层 DiT。词嵌入/LM head 在基础
safetensors 中重复存储同一份权重；参数量同时区分 tensor 存储计数与共享参数去重计数。

这不是论文完整数据、16-layer head 或 8K-context pretraining 的复刻。短 rollout
每卡 2 个 seed、每个 episode 最多 64 个环境 step，用于闭环、状态 carry/reset 和
视频解码验收；其 success 不应解释为完整任务成功率。
历史 500-step 评测记录了初始观察哈希和执行动作。原生重组评测复用框架视频、
seed/task/episode 统计，并额外记录 incoming/committed position、reset mask、实际
前向数与 commit 次数；两种证据的运行版本分开标注。

仍未解决的分布偏移必须保留：训练 memory 写入 noisy GT future chunk，而推理写入
预测 chunk，其中后 32 步尚未执行。独立 flow time 不会消除这个差异。replay 诊断
固定示范观察序列与噪声，比较两类历史下的 memory、当前动作误差和 flow loss；这只是
训练集诊断，不能证明泛化能力。

H40 / stride 8 还意味着相邻监督块重叠 32 步。除第一个 anchor 外，当前监督的
80% 曾出现在前一块的 noisy-GT 更新中；最前面的执行 8 步在足够长的历史中最多
出现过四次。这是训练历史中存在的额外目标信息，不因各 anchor 独立采样 flow time
而消失。诊断分别报告执行 0:8、重叠 0:32、新出现 32:40 的误差；若 GT 历史的
优势集中在重叠区间，应将其视为分布偏移证据，不能将低训练 loss 解释为闭环提升。

另外保留原 GR00T 的 132 维 action-head 容量，只监督有效 12 维。剩余 120 维的训练
输入是衰减噪声，推理输出没有直接监督；诊断同时记录其 RMS，不将这一潜在偏移静默
改成不同于 baseline 的算法。图像增强沿用每 anchor 的上游行为，未额外添加 temporal
augmentation 一致性假设。

## 主要依据

- [RoboTTT](https://arxiv.org/html/2607.15275)：token 组成、插入位置、逐通道 tanh gate、独立 flow time 与 TBPTT。
- [TTT 原论文](https://arxiv.org/html/2407.04620)：mini-batch prefix、dual form、meta-gradient。
- [TTT 官方代码 cd831db](https://github.com/test-time-training/ttt-lm-pytorch/blob/cd831db10c8c9a0f6340f02da5613316a8a92b67/ttt.py)：MLP/GELU/LN 与学习率参数化。
- [Flow Matching](https://arxiv.org/abs/2210.02747)：条件路径与速度回归。
- h100 上的 Isaac-GR00T `23ace64`：保留的预训练网络、processor 与基础动作头。

展平 RoPE、一个 anchor 一个 TTT mini-batch、最终预测写入及非零 o_proj 初始化属于
本次明确选择的实现规格，不声称论文披露了这些全部细节。
