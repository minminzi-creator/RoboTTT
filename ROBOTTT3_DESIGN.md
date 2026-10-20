# RoboTTT 3.0: Native Sequence128 Alignment

Use this contract to keep RoboTTT training and evaluation aligned with the
native sequence128 baseline. The user approved this design and named
the independent local/server code directory `RLinf-robottt3.0` on 2026-10-19.
Base snapshot: `94db2a4a99ad3c771ef67c81ca447c4c9e0c5ca5` (native baseline
experiment snapshot over RLinf `7e90641c`). No baseline checkout is modified.

## Scientific contract

- Native Isaac-GR00T commit `23ace64f17aa5015259b8609d371eb61a357c776`;
  GR00T-N1.7-3B and Cosmos-Reason2-2B assets, not the legacy realworld-pni path.
- Same 104-episode data, processor, packed sequence sampler, augmentation,
  H40/stride8/max128, terminal probability 0, seed42, sequence-equal masked loss.
- Eight workers, microbatch1/global8/accumulation1; TBPTT16. AdamW LR2.5e-5,
  betas .9/.95, eps1e-8, weight decay1e-5, clip1; warmup200/cosine10000.
- Native backbone/head use the baseline FSDP BF16 compute/FP32 master and
  reduction policy. TTT learned initial fast weights and inner updates need
  explicit FP32 execution; initialization/reset must not silently quantize W0.
- Native noise-before-time sampling and state dropout .2 are retained; nn.Dropout
  remains disabled. New policy noise must not advance supervision RNG streams.
- TTT off follows the original native forward exactly, with no registers.
  TTT on retains the existing fast-weight mathematics and 16-register design;
  it is not claimed to be a no-op initialization. Include a registers-only
  structural control and carry/reset evaluation modes to isolate memory effects.

## State and gradient contract

Each anchor reads incoming memory. Compute native supervised flow loss and
discard its candidate. Generate four-step actions without gradients using the
same incoming memory. Stop-gradient only the predicted actions, then make one
differentiable final write at bucket999 and commit its candidate once. Future
anchor loss can differentiate through earlier writes within the segment;
backward precedes detach at segment boundaries. GT affects supervision, never
the predicted-history writer. Use full model-space H40x132 predictions.

Memory is explicit input/output, not mutable module state. It resets per
episode/environment; padding neither updates memory nor advances position.
Activation recomputation cannot commit. Rollout owns state, validates finite
values, and handles environment identity/reset explicitly. Four denoising calls
and the final writer all read the same incoming state.

## Boundaries

Preserve native dependency checkout. A narrow adapter at the native attention
residual/FFN boundary is allowed, with native-module parity tests; do not import
the legacy ExplicitDiT/flow training stack. Baseline checkpoint, data, weights,
environment and evaluation outputs are never overwritten. New formal runs must
start from original pretrained weights, not resume a baseline or legacy TTT
checkpoint under a new protocol. Formal training was subsequently approved with
checkpoints at3000,5000,6000,8000,9000,10000 only. Formal evaluation requires
separate approval; output directory names require user agreement.

## Acceptance

Native TTT-off output/loss/gradient parity; register-only parity with the native
augmented-token path; genuine fast-weight update and causal/padding/reset tests;
predicted-write GT independence and future-loss gradient; train/eval writer
equivalence with fixed RNG and dropout disabled; sequence weighting and cursor
resume unchanged; checkpoint metadata identifies algorithm and source; finite
state checks; native-module CPU tests plus explicit GPU/FSDP acceptance status.
Full-size GPU capacity, throughput, durable save/resume and 24-trial simulation
are separate execution evidence, never inferred from CPU tests.

## Code and configuration entry points

Server code root: `/mnt/public/chenzm/robottt/workspace/RLinf-robottt3.0`.
Training and inference host: `root@117.74.66.77`, SSH port43545, eight A800 80GB
devices0–7, as explicitly selected by the user. This replaces the initially
examined four-GPU port41766 host; both see the same mounted code directory.
Training entry: `examples/sft/train_vla_sft.py`, using one of these config names:

| Variant | Config under `examples/sft/config` | Memory |
| --- | --- | --- |
| Native sequence128 | `robocasa_bowlandcup_gr00t_sequence128_aligned` | None, no registers |
| Structural control | `robocasa_bowlandcup_gr00t_sequence128_registers` | None, 16 registers |
| RoboTTT3 | `robocasa_bowlandcup_robottt3` | Predicted-final history, 16 registers |

All three share data, optimizer, sampling and horizon. Their output path defaults
to null and must be chosen explicitly before running. The existing `h800_sft`
asset/dependency aliases resolve to the preserved baseline runtime and assets;
these are references, not copied model weights or data.

The historical baseline used four ranks and accumulation2. The user-selected
eight-rank setup preserves global batch8 but changes rank-local RNG consumption
and reduction order. It is not a bitwise continuation of the old run. Use the
three eight-rank controls together when making a strict architecture comparison;
strict checkpoint resume rejects a changed world size.

Training calls the sequence worker, then native `sft.build_model` and
`native_sequence_forward`. Only the opt-in adapter uses `native_ttt.py` and
`ttt.py`. The worker owns memory across TBPTT segments. Its source manifest
records the added modules and mixed-precision implementation, while strict
resume checks compare the actor/data protocol and source hashes.

Inference uses `predict_with_memory` with native processor output. Its returned
actions are the complete normalized model-space chunk, not RoboCasa controls.
`NativeTTTRollout` owns detached memory by stable environment ID, supports
carry/reset modes, and validates a candidate before committing.

The simulator call chain is `evaluations/run_gr00t_native_trials.py` →
`evaluations/eval_embodied_agent.py` → `NativeEvalRolloutWorker` →
`NativeEvalPolicy` → `NativeActionAdapter` → `NativeTTTRollout` →
`native_ttt_predict`. The policy performs native observation processing and
decodes the full H40x132 prediction before selecting the first eight controls.
The rollout worker resets memory for each episode, checks termination after
each action, and does not advance memory after an environment has finished.

Evaluation uses `evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml`:
eight environment/rollout pairs on devices0–7, one scene per GPU, seeds7–14 and
trials0–2. The queue plans three eight-step preflight batches followed by three
2400-step formal batches (24 formal episodes). Policy RNG uses
`scene_seed + 1000000 * trial_id`. Retain OSMesa rendering, physical-state
pairing and the baseline image-tolerance checks. Pass the preserved baseline
evaluation root via `--reference` to make a cross-model paired comparison;
without it, checks establish repeatability within the new run only.
The reference layout defaults to eight workers. Add `--reference-width 4`
when reading the preserved Shanghai four-worker baseline results.

Use `--dry-run` to inspect the plan without creating files, loading a model or
starting Ray. Execution requires `--execute`, a complete checkpoint and a fresh,
agreed output path. The loader checks the checkpoint manifest, source hashes,
saved model protocol and processor before strict weight loading. Its conversion
to BF16 excludes the FP32 fast-weight branch. Full-size model loading, eight-GPU
Ray placement, step1 save/resume through step2, and one eight-action RoboCasa
chunk on each of eight environments passed the bounded preflight on port43545.
See `ROBOTTT3_PLAN.md` for logs and resource measurements. This does not establish
uninterrupted/resumed bitwise equivalence, real-environment memory carry across
chunks, long-run stability or task success. The user-approved formal training
uses `results/robottt3.0/train`; the paired24-trial evaluation remains a separate
execution gate. See Task5 of the plan for the formal launch record.
