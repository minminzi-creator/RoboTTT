# RLinf 原生组件重组

本次按 training/eval 参考仓库原有扩展位置重组。训练与评测主 YAML 均通过
Hydra defaults 复用原有组件配置，且共享 GR00T model 与 BowlAndCup 模态组。TTT 数学、真实 H40、三种 mask、
FP32 fast state 与最终预测写入语义保留；训练和闭环执行由 RLinf 原有组件负责。

## 入口和归属

| 功能 | 当前归属 | 复用的职责 |
|---|---|---|
| SFT 启动 | `examples/sft/run_vla_sft.sh` → `train_vla_sft.py` | Hydra、Cluster、WorkerGroup、SFTRunner |
| SFT 更新 | `rlinf/workers/sft/fsdp_vla_robottt_worker.py` | 通过 FSDPVlaSftWorker 继承 FSDPSftWorker 的数据迭代、梯度裁剪、AdamW、scheduler、指标归约 |
| 分布式和保存 | `rlinf/hybrid_engines/fsdp/` | FSDPModelManager、FSDPStrategy、DCP checkpoint 和唯一 full_weights.pt 导出 |
| 序列数据 | `rlinf/data/datasets/gr00t_n1d7/` | StatefulDataLoader；只新增 episode 窗口、模态处理、mask 和确定性 sampler |
| 模型 | `rlinf/models/embodiment/gr00t/gr00t_n1d7/` | 保留 GR00T 预训练网络；TTT、DiT 插入、flow、RLinf BasePolicy adapter 在此实现 |
| 评测启动 | `evaluations/run_eval.sh` → `eval_embodied_agent.py` | 原有 EmbodiedEvalRunner 和 channel |
| 环境交互 | `env_worker.py` ↔ `huggingface_worker.py` | 原有 batch/stage 路由和 chunk 执行；额外传递 reset_mask 与 caller-owned memory |
| 仿真 | `rlinf/envs/robocasa365/robocasa365_env.py` | 原有任务、种子、相机、向量环境、统计、视频；复用参考仓库任务注册修复 |
| 模态转换 | `rlinf/models/embodiment/gr00t/simulation_io.py` | 采用参考 eval 的 RoboCasa365 三相机、16 维 state、12 维 action 适配 |
| 审计 | `toolkits/robottt/` 和 `tests/unit_tests/` | 只保留诊断、数学和集成验证，不包含独立 trainer / rollout engine |

原 `rlinf/robottt`、`examples/robottt`、`scripts/robottt` 从活动源码移除。
重组前的完整实现备份为 `artifacts/reorganization/pre_refactor_source.tar.gz`。
历史 500-step 证据仍在 `artifacts/acceptance` 和外部 run 目录中。

## 原生 worker 所需的有限扩展

`FSDPSftWorker` 提供 `forward_backward_batch` 边界，默认行为仍是一批一次前后向；
`FSDPVlaRoboTTTWorker(FSDPVlaSftWorker)` 对该边界实现 TBPTT。入口将
`sequence_training: true` 的 GR00T/RoboTTT 路由到此子类；通用 VLA worker 恢复为基线原文件。
每个 16-anchor 段使用整条序列的 loss 分母，段间只 detach state，最后一段才同步。
`before_micro_batch` 覆盖 forward 和 backward，确保 FSDP 的 no_sync 语义正确。
每条序列结束后仍由父类执行唯一一次 optimizer update。

采样器保存 epoch/order，其 iterator 独立保存 cursor。StatefulDataLoader 在已消费
batch 边界保存快照并处理预取，避免把 worker 已预取的位置误当作训练进度。
恢复使用 RLinf 原生 DCP 加载模型、Adam 和 scheduler，附带恢复 data.pt、每 rank
rng.pt、epoch/offset；严格检查配置、processor、数据摘要、参数统计和源码摘要。

`EnvOutput.reset_mask` 表示本次 observation 是否来自 reset，与终止标志分开。
auto_reset=false 的 done 不会被误当作已经重置的观察。Rollout worker 按
`(mode, stage_id)` 保存 memory，模型只返回 candidate；每次调用只在 worker 中提交。
切换 checkpoint 清空旧 state。局部 reset 只清对应 batch 行。
评测使用原生 `global_unique` seed 策略：8 个独立种子流，各 reset 两次；两组配置一致，
用 reset observation 的 SHA256 核对实际配对场景，而不只比较 seed 字段。

## guozhen1997 两个提交的 FSDP 审查

检查了 `55ed49348a29aa207c330045c817bda92ac31aeb` 和
`9d0e6d6200a95b8ab4f5595f99d10834eb7eec85`。前者在
`rlinf/hybrid_engines/fsdp` 下没有 diff；相关修改实际位于后者。
完整 diff 保存在 `fsdp_reference.patch`。

| 原修改 | 处理 | 原因 |
|---|---|---|
| 忽略模型未包含的可选 wrap class | 修改后采用到 `utils.py` | `_no_split_modules` 可包含互斥实现；只过滤自动推断的列表。用户显式配置仍严格报错，全列表缺失也报错，避免掩盖拼写错误或意外不分片。 |
| wrap class 错误信息包含具体类名 | 采用到 `utils.py` | 初始化失败可定位到实际缺失的模块。 |
| wrap 前调用 `restore_uninitialized_params_` | 以明确加载顺序替代 | 先完整加载基础 GR00T，再创建 TTT/register；不依靠 HuggingFace missing-key 存储或按数值猜测损坏。wrap 前新增参数已完成初始化。 |
| `_fsdp_ignore`、collect ignore helpers、FSDP1 ignored_modules | 不引入 | register 是普通 Parameter；根 FSDP 负责分片和梯度归约，无需旁路。 |
| FSDP2 遍历跳过 `_fsdp_ignore`，合并 ignored_params | 不引入 | 同上；现有通用 FSDP2 接口保留，本轮实际验证使用 FSDP1。 |
| 被忽略 register 的手写梯度 all_reduce | 不引入 | 与普通 FSDP 归约重复会改变梯度；原生 DCP 也应统一管理全部参数。 |

另外在原 FSDP 位置增加三项协议支持：三种 mixed-precision dtype 都为空时使用
`mixed_precision=None`；允许明确关闭输入自动转换；非有限梯度在本配置中直接失败。
linear_warmup_cosine scheduler 继续复用，当前默认改为 120-step warmup / 30K
cosine，首个更新为 5e-5/120。配置通过 model、原有 FSDP、公共实验 defaults
组合。具体日程与 worker 继承见 [30k_worker_refactor.md](30k_worker_refactor.md)。
上一阶段的 200-step warmup / 50K cosine 只属于历史验收记录。

## 验证范围

重组验证结果见 `artifacts/reorganization/reorganization_report.md`。
此前各 500-step 结果来自重组前实现，不能称为这份原生 trainer 已重跑 500 step。
原生配置保留相同数据和数学协议；native AdamW/FSDP 实现及执行顺序可能带来浮点
轨迹差异。断点恢复的逐位一致性单独在同一原生实现内验证。
