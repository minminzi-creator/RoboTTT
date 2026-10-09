# RoboTTT 逐文件修改说明

仓库基线为 `3cb54c2eb2e355378ba049d50f7ec0cb8acc2da0`。下表列出当前复现相对该
基线的活动源码、配置、工具和文档，并标注最新一轮 30K/worker 拆分涉及的文件。
“此前”表示改动来自已完成的原生集成，本轮保留；“本轮”表示继续修改或新增。
完整参考提交逐项采用/修正/舍弃理由见 [reference_commit_map.csv](reference_commit_map.csv)。

本轮补充：训练和评测 defaults 复用、独立 worker、30K 日程以及配置加载工具修复。
TTT 更新公式、DiT 插入位置、推理写入、数据采样、评测交互没有在本轮改变。

## 训练入口与配置

| 文件 | 阶段 | 修改及目的 |
|---|---|---|
| `examples/sft/run_vla_sft.sh` | 此前 | 统一 `bash ... xxx.yaml` 调用；支持 basename、完整路径和 Hydra overrides；保留环境准备、日志重定向，使用原训练入口 |
| `examples/sft/train_vla_sft.py` | 本轮 | 新增 `get_sft_worker_class`；sequence GR00T/RoboTTT 使用专用子类，其他模型仍用通用 VLA worker；保留原 Cluster、placement、WorkerGroup、SFTRunner |
| `examples/sft/config/model/gr00t_n1d7.yaml` | 本轮新增 | 集中可复用 GR00T 序列/TTT 架构、FP32、去噪和 memory 配置；默认无 TTT、0 register；恢复参考提交的 model group 位置 |
| `examples/sft/config/robottt/bowlandcup_model.yaml` | 本轮新增 | 训练/评测共用的 checkpoint/Cosmos 路径、BowlAndCup embodiment、action维度、相机数；分别挂载到 actor.model 与 rollout.model |
| `examples/sft/config/robottt/bowlandcup_h100.yaml` | 本轮新增 | 两组共享的数据/资源路径、batch、optimizer、30K/warmup120、保存间隔和少量 FSDP 覆盖；避免 baseline 与 RoboTTT 参数漂移 |
| `examples/sft/config/robocasa_bowlandcup_gr00t_baseline_h100.yaml` | 本轮 | 由完整平铺改成 model/FSDP/公共实验 defaults 和日志差异；沿用原始 GR00T、0 register |
| `examples/sft/config/robocasa_bowlandcup_robottt_h100.yaml` | 本轮 | 同一组 defaults，仅启用 RoboTTT model alias、TTT 与 16 register；训练其余字段与 baseline 相同 |

`examples/sft/config/hybrid_engines/fsdp.yaml` **没有修改**；两份主配置直接继承这个
已有文件。模型组按原 guozhen 的目录位置恢复，字段按当前显式 state 实现整理，
不复制已经弃用的隐式 context、TTT-only 和 PPO noise/value-head 开关。

## 训练 worker 与 FSDP

| 文件 | 阶段 | 修改及目的 |
|---|---|---|
| `rlinf/workers/sft/fsdp_sft_worker.py` | 此前 | 增加可覆写的 `forward_backward_batch` hook；默认一批一次前后向，no_sync 包含 forward/backward；保留唯一 optimizer update、梯度裁剪、scheduler、数据迭代和指标归约；分别记录本步实际 LR 与下一步 LR |
| `rlinf/workers/sft/fsdp_vla_robottt_worker.py` | 本轮新增 | `FSDPVlaRoboTTTWorker(FSDPVlaSftWorker)` 接管序列 dataloader、TBPTT、state detach、严格协议恢复及序列审计；父类继续负责训练基础设施；baseline 也使用该类以保持数据和更新协议一致 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | 本轮恢复基线 | 移出全部序列分支，文件与基线原件一致；其原生 StatefulDataLoader/RNG 保存恢复由子类继承，避免重复实现 |
| `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` | 此前 | 对当前配置启用 `fail_on_nonfinite`，非有限梯度停止训练；保留框架 AdamW/DCP，不添加自动修复或重新初始化 |
| `rlinf/hybrid_engines/fsdp/strategy/fsdp.py` | 此前 | 三种 dtype 均为空时使用 `mixed_precision=None`；允许显式关闭 forward/root 输入 cast，避免将 FP32 fast state 转成 BF16；沿用标准 register 参数管理 |
| `rlinf/hybrid_engines/fsdp/utils.py` | 此前 | 修改后采用 guozhen 的 wrap-class 兼容处理：只过滤自动推断出的缺失可选类，显式配置仍严格校验；改善报错；提供线性 warmup/cosine 日程，当前仅在 YAML 将参数改成 120/30000 |

参考 `55ed493`/`9d0e6d6` 的继承方式已恢复；不恢复仅对 backward 设置 no_sync、
手写 register all_reduce、ignored-module 旁路或按数值猜测参数是否损坏。
具体理由、方法级父子职责和恢复顺序见 [30k_worker_refactor.md](30k_worker_refactor.md)。

## 模型注册、TTT 与前向

| 文件 | 阶段 | 修改及目的 |
|---|---|---|
| `rlinf/config.py` | 此前 | 注册 `robottt` 模型类型，允许原生配置校验与 worker 选择识别它 |
| `rlinf/models/__init__.py` | 此前 | 为 GR00T/RoboTTT 使用同一模型 factory，训练与推理复用构建路径 |
| `rlinf/models/embodiment/gr00t/__init__.py` | 此前 | 将 `robottt` alias 路由到 GR00T N1.7 模块，不创建另一套模型加载器 |
| `rlinf/models/embodiment/gr00t/gr00t_n1d7/__init__.py` | 此前 | `sequence_training` 分支构造共享 SequencePolicy adapter，校验 FP32；保留原非序列 PPO/GR00T 路径 |
| `rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py` | 此前 | 增加 `GR00T_N1_7_SequencePolicy` BasePolicy adapter：SFT ForwardType、仿真观察转换、checkpoint processor 加载、显式推理 memory；原模型类继续保留 |
| `rlinf/models/embodiment/gr00t/gr00t_n1d7/ttt.py` | 此前新增 | 多头两层 GELU fast MLP、LayerNorm 残差重建、可学习 inner LR、共同 incoming 权重处的 prefix dual 更新、meta-gradient；提供 FastState/Memory 初始化/reset/detach/reorder；全程 FP32 |
| `rlinf/models/embodiment/gr00t/gr00t_n1d7/ttt_dit.py` | 此前新增 | 保留全 Cosmos 与 32 层 DiT，在原 attention 与 FFN 之间插入 TTT residual；16 register 与非零 gate/o_proj 初始化；按 anchor 递归、TBPTT loss 分母、padding mask、checkpoint 纯函数语义；共同模型构建和参数统计 |
| `rlinf/models/embodiment/gr00t/gr00t_n1d7/flow.py` | 此前新增 | 实现 FP32 时间/离散 bucket 与 flow target；4 次 Euler 使用同一 incoming，最终预测在 bucket999 额外一次完整前向，只返回唯一 candidate；写入模式/index 对照不改变已生成动作 |
| `rlinf/models/embodiment/gr00t/gr00t_n1d7/sequence_utils.py` | 本轮 | 保留 train/eval 的配置适配、runtime、checkpoint bundle；新增 Hydra `load_sft_config` 给审计工具展开 defaults；源码摘要新增独立 worker 和训练入口，防止旧源码静默续训 |
| `rlinf/models/embodiment/gr00t/simulation_io.py` | 此前 | 沿参考 eval 的位置加入 RoboCasa365 三相机、16 维 state、12 维 action 的转换；控制量阈值可配置并检查非有限动作 |

核心没有移动回 `rlinf/robottt`，也没有在 worker 中复制 TTT、DiT 或 flow 数学。

## 数据与序列采样

| 文件 | 阶段 | 修改及目的 |
|---|---|---|
| `rlinf/data/datasets/gr00t_n1d7/__init__.py` | 此前新增 | GR00T 序列数据包入口，遵循框架 dataset 目录 |
| `rlinf/data/datasets/gr00t_n1d7/sequence_dataset.py` | 此前新增 | 读取真实 40 步动作；全部 104 episodes、stride8/最多128 anchor、短轨迹保留；确定性 episode/window/增强、rank 等权切分；分离 anchor/supervision/action mask；共同 processor/归一化、分段 batch 和数据摘要 |
| `rlinf/data/datasets/gr00t_n1d7/dataloader.py` | 此前新增 | 接入 StatefulDataLoader，worker spawn/预取；sampler 保存 epoch/order，iterator 保存已消费 cursor，避免恢复到预取位置；把 metadata/processor 返回原生 worker |

## 评测入口与环境交互

评测实现沿用此前原生组件；本轮重组两份 eval YAML、新增公共配置，保留解析后的
全部评测字段和 checkpoint 加载方式。

| 文件 | 阶段 | 修改及目的 |
|---|---|---|
| `evaluations/run_eval.sh` | 此前 | 支持 `bash evaluations/run_eval.sh xxx.yaml`；保持原 benchmark、Hydra、eval entrypoint 与环境准备 |
| `evaluations/run_bowlandcup_full_eval.sh` | 完整评测 | 默认暴露全部8卡，预检 baseline/RoboTTT 的2K、4K、8K六个checkpoint，按step配对串行评测并写入六个独立日志/视频目录；额外 Hydra 参数传给全部实验 |
| `evaluations/robocasa365/robottt/bowlandcup_eval.yaml` | 本轮新增，完整评测更新 | 两组共享 runner/rollout 和 BowlAndCup 环境覆盖；8 个不同 deterministic seed、2400-step task horizon、完整失败计数及默认视频；直接继承原生 env/robocasa365 的 robot/camera/key-map/action-space 默认值 |
| `evaluations/robocasa365/robocasa365_bowlandcup_gr00t_baseline_h100.yaml` | 本轮 | 改用 env/共享训练model/共享模态/公共eval defaults，仅保留路径和实验名；默认加载已完成 global_step_8000 完整权重 |
| `evaluations/robocasa365/robocasa365_bowlandcup_robottt_h100.yaml` | 本轮 | 同一组 defaults，仅另行启用TTT/register/model alias和路径差异；匹配 baseline 的评测配置，启用 TTT/register 和最终预测 memory 写入；记录每次实际 forward/commit/reset |
| `rlinf/data/schema/embodied_types.py` | 此前 | `EnvOutput.reset_mask` 与 done 分离，随 channel 序列化和 batch 合并；避免误把结束但未重置的 observation 当作新 episode |
| `rlinf/workers/env/env_worker.py` | 此前 | 在真实手工 reset 或 auto-reset 时发出 reset_mask，并通过原有 channel 传给 rollout |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 此前 | 原生 rollout worker 按 `(mode,stage)` 保存 memory，调用模型后只提交一次；支持局部 reset、checkpoint 更换清空状态、动作/状态非有限检查；记录状态位置和 reset observation 哈希 |
| `rlinf/envs/robocasa365/robocasa365_env.py` | 此前 | 采用参考 eval 的 Ray worker 内任务注册修复；继续复用原环境初始化、任务/种子、统计、chunk交互和视频 |
| `rlinf/envs/robocasa/venv.py` | 此前 | 保留已验证的 opt-in EGL 设备局部映射；单卡可见的 Ray worker 使用本地 ordinal0，避免物理卡号与进程内 EGL 编号混淆 |

## 测试与审计工具

| 文件 | 阶段 | 验证对象 |
|---|---|---|
| `tests/unit_tests/test_robottt_reproduction.py` | 此前新增 | 独立 create_graph primal 与 dual 的输出/state/输入和参数梯度；跨 anchor 因果性、位置、padding、detach/reset/reorder、flow 写入和 DiT/checkpoint 纯函数 |
| `tests/unit_tests/test_robottt_rlinf_integration.py` | 本轮 | TBPTT 测试迁到新子类并增加非最后 micro-batch；保留采样 cursor 恢复、全 padding segment、完整 loss 分母、rollout memory/reset、FSDP wrap 与历史 LR 边界测试 |
| `tests/unit_tests/test_robottt_sft_config.py` | 本轮新增 | train/eval模型完全一致及eval环境继承；两组所有解析字段的差异检查；修改共享 model/FSDP group 确认继承生效；worker 路由和非法组合；30K warmup/cosine边界；短 stop 不压缩 schedule；源码摘要覆盖新 worker/入口 |
| `toolkits/robottt/audit_compilation.py` | 此前迁移 | GPU 真尺寸 eager/compile、FP32 state、BF16 outer、checkpoint 和 meta-gradient 对齐 |
| `toolkits/robottt/audit_sampler.py` | 本轮 | 改用 Hydra 组合配置；审计 104 episode 覆盖、stride/H40、rank 重复、worker独立性、真实 parquet/模态对齐和短轨迹 |
| `toolkits/robottt/audit_video_alignment.py` | 本轮 | 改用同一配置加载；核对全部312段相机视频的帧数/分辨率/fps和 parquet 索引时间，不把容器对齐等同于传感器物理同步 |
| `toolkits/robottt/audit_resume.py` | 此前迁移 | 流式比较连续训练与恢复后的原生 DCP 模型、Adam、scheduler、RNG 以及 data/rng/progress/full_weights，验证逐位一致 |
| `toolkits/robottt/audit_parameter_updates.py` | 此前迁移 | 比较相邻 checkpoint，确认 TTT各参数组和register存在实际非零更新 |
| `toolkits/robottt/audit_evaluation.py` | 此前迁移 | 检查8卡forward/commit/reset次数、动作形状与有限性、所有视频完整解码、两组reset哈希配对 |
| `toolkits/robottt/diagnose_history.py` | 本轮 | 改用同一 Hydra 配置；保留公共 factory/checkpoint processor，在固定示范观察上比较 noisy-GT 与预测历史 memory，只作为训练集偏移诊断 |

这些工具不持有训练主循环或另写环境交互，不替代 RLinf worker/runner。

## 文档、历史目录与证据

| 文件/目录 | 用途 |
|---|---|
| `ROBOTTT.md` | 顶层启动索引、当前30K默认、核心目录和历史证据链接 |
| `docs/robottt/reproduction.md` | 数学/数据/精度/状态契约、运行与恢复命令、诊断和结论边界；本轮更新当前日程与worker名称 |
| `docs/robottt/native_integration.md` | 原生组件职责、guozhen FSDP逐项决策；本轮更新独立worker归属和30K设置 |
| `docs/robottt/reference_review.md` | 四个参考提交与未提交修复的审查概览；本轮说明继承方式和日程变更 |
| `docs/robottt/reference_commit_map.csv` | 96条参考变更逐文件的采用/修改/舍弃理由；本轮同步入口/worker/model配置目标路径，不改参考提交身份 |
| `docs/robottt/fsdp_reference.patch` | guozhen 两个提交中相关FSDP原始差异；只读审查材料，本轮未修改 |
| `docs/robottt/30k_worker_refactor.md` | 本轮配置组合顺序、全部默认值变化理由、方法级继承、LR公式、命令、恢复兼容性和验证入口 |
| `docs/robottt/file_changes.md` | 本文件，活动修改的完整职责清单 |
| `artifacts/worker30k/` | 本轮源码备份、测试、train/eval配置解析、真实数据审计、8卡短训练/续训与新权重短闭环证据及报告 |
| `artifacts/reorganization/` | 上阶段原生RLinf重组证据，保留其运行时源码和200-warmup/50K日程的原意 |
| `artifacts/acceptance/` | 更早独立实现的500-step验收证据；不能算作当前新worker已跑500或30K |

此前活动目录 `rlinf/robottt`、`examples/robottt`、`scripts/robottt` 已移除；旧源码
归档在 `artifacts/reorganization/pre_refactor_source.tar.gz`。本轮修改前的原生代码
另存 `artifacts/worker30k/pre_change_source.tar.gz`。没有删除用户原训练/评测仓库或
历史checkpoint。当前统一入口仍是 `examples/sft/run_vla_sft.sh` 与 `evaluations/run_eval.sh`。
