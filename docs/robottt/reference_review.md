# 参考提交审查

从 `3cb54c2eb2e355378ba049d50f7ec0cb8acc2da0` 手工重写，未 cherry-pick。
四个提交及 train/eval 本地修复的 96 条逐文件映射见
[reference_commit_map.csv](reference_commit_map.csv)。原生组件复用及 FSDP 的逐项理由见
[native_integration.md](native_integration.md)。本轮配置与 worker 继承的详细对照见
[30k_worker_refactor.md](30k_worker_refactor.md)。

| 参考 | 采用或修正 |
|---|---|
| `55ed493` / guozhen1997 | 保留 TTT-MLP、register、DiT 插入、序列数据和 TBPTT 的职责划分；核心文件仍在 gr00t_n1d7，loader 在 data，训练由独立 FSDPVlaRoboTTTWorker 继承 FSDPVlaSftWorker。以显式 state 和独立 primal/meta-gradient 测试修正隐式 context、forward monkey patch 和跨 anchor 更新语义。 |
| `9d0e6d6` / guozhen1997 | 保留 FP32 inner 和 block-owned TTT 参数。FSDP 的可选 wrap class 过滤修改后采用；参数启发式恢复替换为“基础完整加载后再创建新增模块”；register 用正常 FSDP 管理，不保留 ignored-module / 手动同步。完整 FSDP diff 单独保存。 |
| `699c972` / training | 采用 BowlAndCup 模态、数据统计、H100 路径。真实 action delta 从 8 改为 40；确定性索引、全部短轨迹、三类 mask、整条序列分母及一致同步次数。父类 optimizer/scheduler/clip/checkpoint 不再自行重写。 |
| `e15fe9c` / eval | 复用原生 EnvWorker、huggingface_worker、Robocasa365Env、simulation_io；保留 robottt model type，路由至公共 GR00T factory。参考任务注册修复直接采用。明确 reset_mask 与 done 的区别，rollout worker 是唯一 memory owner。 |
| train 未提交 YAML | 保留本机资源路径和 8 卡。两份主 YAML 复用 model/FSDP/公共实验 defaults；按后续要求改为 30K cosine、warmup 120，峰值 LR 5e-5。原 50K/stop500 仅保留在历史验收记录。 |
| eval 未提交 YAML / run_eval.sh | 配置继续位于 evaluations/robocasa365，bash 入口继续为 evaluations/run_eval.sh。支持 .yaml 后缀和全路径；通过 defaults 复用原生 env、训练共享 model/模态、公共 eval；保留 EGL 和局部设备映射。 |
| eval 未提交 robocasa/venv.py | 保留 opt-in、单 GPU 可见条件下的 EGL ordinal=0 修复。 |

不保留旧 NaN 替零、损坏参数数值猜测、自动 fast-state 重置、重复导出、备份文件和
调试 monkey patch。DAgger / PPO 专用 recurrent 路由、human-video ICL 和长训练按
既定范围另行开展；原 RLinf 通用组件仍在，本次 SFT/评测不引入这些额外流程。

TTT gate 保留每通道 tanh 参数化，初值 0.001；o_proj std 0.02 保持首步完整外层
梯度通路。展平 RoPE、每 anchor 一个 TTT mini-batch、最终预测 bucket 999 额外
前向写入、FP32 fast state 与三种 mask 均按已确认方案实现，并非照搬参考代码。
