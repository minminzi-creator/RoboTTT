# RoboTTT 3.0 Implementation Plan

> **For agentic workers:** Use the approved design and test-driven-development.
> Parallel workers own disjoint files; the primary agent integrates and reviews.

**Goal:** Add explicit, differentiable predicted-history TTT to the actual
native sequence128 baseline without changing the baseline-off path.

**Architecture:** Keep the native GR00T factory/processor/sampler/optimizer.
Add a functional native-block adapter, fast-weight module, explicit worker
memory lifecycle and a shared inference interface. Never mutate module memory.

**Tech Stack:** PyTorch, FSDP1, RLinf, pinned Isaac-GR00T, Hydra, pytest.

**Spec:** `ROBOTTT3_DESIGN.md` (approved conversation transcribed).

## Global Constraints

- New code root name: RLinf-robottt3.0 locally and on the Beijing server.
- Base snapshot94db2a4a; dependency23ace64f; no baseline writes or asset copies.
- H40/stride8/max128/TBPTT16/terminal0/global8; AdamW2.5e-5/warmup200/10K.
- User-selected final host: root@117.74.66.77:43545, eight A800 devices0–7;
  all new controls use eight ranks, microbatch1 and accumulation1.
- Explicit FP32 TTT state; native base BF16 compute; independent policy noise.
- No experiment output folders or formal training without agreed names.
- User-approved output root (2026-10-19):
  `/mnt/public/chenzm/robottt/workspace/results/robottt3.0`, with `preflight`,
  `train` and `eval` roles. Only bounded full-model preflight is authorized now.

## Review Focus

- B>1/time-major slicing: each anchor must read its own features and noise.
- Padding/context-only anchors: mask supervision separately from state writes.
- Checkpoint recomputation: repeated forward must not alter incoming memory.
- BF16 FSDP: learned W0, reset and fast updates must retain FP32 execution.
- Resume/provenance: no silent restoration across baseline/TTT algorithms.

## Task 1: Fast state and native block adapter

Files: new `rlinf/models/embodiment/gr00t/gr00t_n1d7/ttt.py`,
`native_ttt.py`, `tests/unit_tests/test_gr00t_native_ttt.py`.
Interfaces: `configure_native_ttt(model,cfg)`,
`native_ttt_sequence_forward(model,data,memory=None) -> dict`,
`native_ttt_predict(model,data,memory=None,**kwargs) -> dict`.

- [x] Write failing real-module tests for native no-memory block parity,
  fast-state reset/padding/causality and differentiable writer.
- [x] Run the tests and record missing-feature failures.
- [x] Implement narrow native adapter; reuse verified fast-weight mathematics.
- [x] Verify GT-independent history, one commit, future-loss gradient and RNG.

## Task 2: Factory, training lifecycle and protocol

Files: native `sft.py`, `fsdp_gr00t_sequence_sft_worker.py`, sequence dataset,
FSDP precision configuration, new named training configuration, component tests.
Consumes explicit memory API from task1; produces fresh/resumable TTT training.

- [x] Write failing tests for off/on dispatch, segment memory/detach and protocol.
- [x] Implement opt-in TTT construction and worker ownership; leave off unchanged.
- [x] Add deterministic per-anchor policy seeds without changing sampling.
- [x] Define TTT precision isolation and metadata; reject incompatible resumes.
- [x] Run baseline regressions and new integration tests.

## Task 3: Inference and reproducibility handoff

Files: shared native inference API, tests, `ROBOTTT3_DESIGN.md` and this ledger.
Consumes functional model/state contract; produces explicit carry/reset API.

- [x] Verify fixed-noise train/write versus inference/write equivalence.
- [x] Validate finite state, per-row reset/reorder and unchanged incoming state.
- [x] Wire the existing native RoboCasa policy/worker into the approved checkout,
  preserving processor/action decoding, per-action termination and paired trials.
- [x] Verify explicit output/checkpoint arguments and a side-effect-free queue
  dry-run; keep the real simulation acceptance gate separate.
- [x] Review complete diff; run related component tests and report broader limits.
- [x] Deploy only the agreed new code directory, preserving base/old directories.
- [x] Record exact tests, source identity and outstanding full-model GPU gates.

## Execution ledger

- 2026-10-19: User approved architecture and directory name; standalone local
  Git clone avoids adding an old-repository worktree dependency to the cleanup.
- Planning and verification records remain in these two root documents; no new
  experiment-directory naming was inferred. No formal run is authorized here.
- Runtime investigation: Beijing system Python lacks torch; existing baseline
  environment is now under workspace/0History and is being checked read-only.
- RED evidence: 12 missing fast-weight tests and three missing factory/worker
  contract tests failed before implementation. Configuration and output-path
  guard tests also failed before implementation.
- Runtime used: Python3.11.14, torch2.11.0+cu129, pinned Isaac-GR00T23ace64f,
  `/mnt/public/chenzm/robottt/workspace/0History/gr00t-baseline-env-20261012`.
- Initial combined CPU regression: **174 passed, 3 skipped, 46 warnings in 52.96s**.
  Files: sequence SFT, fast weights, native TTT, native rollout, window dataset,
  window integration, window sampling, and FSDP. CUDA was hidden. Two native
  baseline FSDP tests and one two-GPU FSDP test skipped for hardware conditions.
  Existing data-backed CPU window tests ran successfully. Third-party warnings
  concern torch.jit, diffusers config access and single-rank FSDP NO_SHARD.
- Post-review/eight-rank configuration regression: **179 passed, 5 skipped,
  55 warnings in 55.99s** on port43545. The additional CPU-native FSDP case and
  both four/eight-rank sampler coverage tests passed. The two gated native GPU
  cases were exercised separately on all eight GPUs, as recorded below.
- Hydra merge verification showed that `???` retains an existing parent value.
  New configs therefore set `log_path: null`; the training entry rejects a missing
  or blank path before creating distributed workers. No output name is inferred.
- Review identified an inherited native `get_action` path that can bypass TTT.
  Three corrected fixtures produced the intended missing-guard RED failures.
  DiT/block now reject calls without the explicit recurrence protocol; the
  13-test native CPU suite passed and read-only re-review closed this finding.
- Source upload for the latest native adapter/test pair was paused by the safety
  reviewer, then explicitly approved by the user for the named two files and
  server destination. The files were synchronized through SCP after approval.
- The first inherited-inference guard test attempt failed on incomplete native
  fixtures, not the intended assertion; it was corrected before implementation.
- The initial four-GPU host became occupied (about59GB each); no validation
  process was launched there and no existing process was stopped. The user then
  selected port43545 for verification, formal training and inference. Its eight
  GPUs were confirmed idle and its shared code/runtime paths are usable.
- CPU/Gloo nested native FSDP acceptance passed, including checkpointing,
  first-segment no_sync backward, detached memory carry and final synchronized
  backward. Native computation remained BF16 and fast parameters/state FP32.
  Single-GPU NO_SHARD and eight-GPU FULL_SHARD probes passed for both eager and
  compile_inner=True. All eight ranks reported nativeBF16/fastFP32, position4,
  total fast-parameter gradient L1=0.02889624424278736; compiled peak allocation
  was at most61,352,448bytes/rank. These use a two-layer hidden8 small model,
  not the complete3B architecture. All GPU processes exited and the cards were
  confirmed idle afterward. Full-size training, durable distributed save/resume
  and 24-trial simulation remain unrun.
- Baseline source has pre-existing tracked/untracked experiment changes; preserve
  them. Do not interpret that checkout's non-clean status as changes by this task.
  Read-only SHA256 checks of its training entry, sampler, native factory and
  sequence worker match the pre-implementation snapshot exactly.
- Evaluation audit: the current Shanghai OSMesa queue is
  `/mnt/public/chenzm/robottt/h800_sft/evaluation/runtime/run_queue.py`.
  It launches seeds7–14 × trials0–2, H40 predictions with eight executed actions,
  and per-action success checks. Its policy casts all parameters to BF16 and the
  queue requires TTT off; neither can be reused unchanged for a TTT checkpoint.
- Port43545 simulation dependency inspection was read-only. The training
  environment exposes Ray/MuJoCo/robosuite but not robocasa on its default import
  path. The existing `workspace/robocasa` checkout is at
  `4f8a2980def75a55dff96b990745b83540425f09`; `workspace/robosuite` and
  `workspace/robosuite-master` are clean at
  `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. RoboCasa has a pre-existing
  `bowl_and_cup.py` debug patch and untracked assets. Its inspected diff retains
  `cup_in_bowl and bowl_on_counter and gripper_far`, adding diagnostic prints.
  These files were preserved. Select dependency paths explicitly with the queue
  arguments and verify OSMesa plus paired scenes before a formal evaluation.
- Evaluation integration uses the preserved Shanghai policy/worker as its
  processor, action-conversion and trial-identity reference. Original SHA256:
  policy `27ec5a1f3364469f991f20c69627bcd1595785a3176882e5420f3474d2c4e8cc`,
  worker `a9ca4e383e69d067af1e42a5b4fed9f0a8fdf34be2ab047906e63c38934d22ec`.
  Adapted files live only in the approved new checkout. Review closed four
  findings: existing-result protection before logger initialization; checkpoint
  subtree protection before queue writes; short Ray runtime paths; and eight-rank
  reference defaults (historical four-rank references require an explicit flag).
- Native evaluation tests use an actual two-layer hidden8 head with H40x132,
  16 registers and 57 tokens. The complete native prediction, decoded first
  eight controls and each committed fast-state tensor match exactly. VLM and
  processor asset boundaries use fixtures; the native head/TTT are real.
- Final combined regression on port43545: **208 passed, 5 skipped, 57 warnings
  in 56.30s**, exit0, covering the nine files listed below. This is a targeted
  suite, not the whole repository. Two existing baseline CUDA cases, two new
  gated native CUDA probes and one existing two-GPU FSDP case skip in this
  CUDA-hidden run; both new probes have the separate eight-GPU evidence above.
- Final Ruff check and format check passed for all 15 changed Python files;
  `git diff --check` passed. All 19 changed Python/YAML files have matching local
  and server SHA256 values. The final CLI dry-run verified six batches, 24 formal
  episodes, devices0–7 and eight-worker reference defaults without creating the
  supplied output directory. No Ray cluster, full model, training run or
  simulator was launched by that dry-run. Changes remain uncommitted.

## Task 4: Bounded Full-model Preflight

Run only on port43545 and devices0–7. Keep production H40/stride8/max128,
TBPTT16/global8/micro1, native data augmentation and the 10000-step LR schedule.
Limit the runner to one optimizer step, save a complete checkpoint, then resume
that checkpoint up to step2. The run is diagnostic and is not a formal result.
Use `runner.logger.log_path` equal to the approved `preflight` directory; the
existing runner generates `robottt3/checkpoints/global_step_N` beneath it.
Keep initial and resumed launch logs separate and preserve any failure evidence.

- [x] Confirm the eight GPUs are unused, pinned SDK clean, and storage available.
- [x] Create only the approved output root and its `preflight` directory.
- [x] Resolve the production configuration with bounded runner overrides.
- [x] Run one full-model optimizer step and inspect finite loss/gradient/update.
- [x] Validate completion manifest, metadata and eight-rank sample records.
- [x] Resume to step2 and verify optimizer/data/LR progress without changing code.
- [x] Run one eight-environment/eight-action inference preflight from step2.
- [x] Inspect resource use, preserve logs and report full-model acceptance limits.

Do not start a 10K training run or 24-trial evaluation. If capacity fails under
the specified protocol, record the failure rather than shrinking the sequence or
changing TBPTT silently. Ray runtime uses a short OS temporary directory to avoid
Unix socket length limits. Do not stop unrelated Ray or GPU jobs.

Preflight review confirmed that `runner.max_steps` does not change the fixed
`actor.optim.total_training_steps=10000`, and the strict resume contract compares
actor/data/source/world-size identity rather than the runner stop limit. Launch
one Python driver; RLinf creates eight Ray actor workers from placement0–7.
Do not wrap this production entry in `torchrun`.

The first full-size run exited0. It trained 3,466,972,064 parameters using eight
workers, with loss1.29205954, grad_norm0.73607212 and TTT update RMS0.0024797544.
LR was1.25e-7 and next LR2.5e-7. Runner timing was257s for training and303s
including checkpoint save; initialization is additional and the first step
includes compilation, so these are not steady-state throughput measurements.
The complete step1 checkpoint is58,032,753,295bytes across15 manifest members.
Read-only validation confirmed progress(1,1,1), world_size8, 32 TTT layers with
512 finite FP32 tensors, registers[1,16,1536], and all floating full-state tensors
FP32. Eight sample records contain861 real anchors in total, including three
full128-anchor sequences. All GPUs were released at the end of this process.
Evidence: `preflight/train_step1.log`, `sequence_metrics.jsonl`, per-rank
`sequence_samples_rankN.jsonl`, and `gpu_step1.csv` (5s sampling, started after
initialization). Ray runtime for this first run was `/tmp/r3-uC9ZEt`.

The second process restores `global_step_1` and stops at step2, with
`runner.save_interval=1` in both processes. Its log is `train_resume.log`, GPU
samples are `gpu_resume.csv`, and Ray runtime is `/tmp/r3-qssn9N`. It reuses the
same source, assets, actor/data protocol and compilation caches.

Resume exited0 and saved a complete step2 checkpoint (58,032,753,393bytes).
Strict metadata comparison passed. Progress is(2,2,2), LR2.5e-7, next LR3.75e-7,
loss1.27763104, grad_norm0.71188712 and TTT update RMS0.0027154048. Every rank's
first two recorded (epoch,row) identities match a fresh reference sampler, so
the resumed run consumed the next batch rather than repeating the first batch.
Selected layer0 TTT gate, w1 and q_proj tensors changed by finite nonzero amounts.
Step2 training took245s, or289s including saving; more uninterrupted steps are
needed to measure steady-state throughput. Five-second nvidia-smi sampling saw
maxima of29,513MiB in run1 and38,409MiB in the resumed run. These are sampled
usage maxima, not allocator peak-memory instrumentation.

Inference preflight uses only one direct `eval_embodied_agent.py` invocation:
step2 weights, eight scenes7–14, trial0, carry memory, max_episode_steps8 and
max_steps_per_rollout_epoch8, with `rollout.preflight=true`. Output is
`preflight/eval`, stdout is `preflight/eval.log`, GPU samples are `gpu_eval.csv`,
and Ray runtime is `/tmp/r3-YzXv5b`. It is bounded to20minutes and does not invoke
the 24-trial queue. Dependency imports and OSMesa library resolution passed
before launching; the live simulation is a separate acceptance gate. This
single-batch preflight has no external paired baseline reference.

The asset-backed inference process exited0. Final `check_batch` validation passed
for all eight scene identities, strict checkpoint loads and finite first-action
chunks of shape[1,8,12]. Each episode executed8 actions. All ranks reported
BF16/FP32 parameter dtypes, FP32 buffers, TTT enabled, 16 registers and the same
processor hash. Eight videos are retained under `preflight/eval/video`.
Five-second GPU sampling recorded14,677MiB on each inference device. The final
CSV validation ran after stopping the sampler and found no incomplete rows.
Final nvidia-smi showed0MiB on all eight devices; the sampler also exited.

The complete preflight directory occupies109GiB (rounded `du -sh`), including
two checkpoints totalling116,065,506,688bytes. Evaluation artifacts occupy2.5MiB.
Preserve these files as diagnostic evidence; they are not formal checkpoints or
success-rate results. No formal `train` or `eval` output directory was created.

Acceptance is limited to two full-model optimizer steps, durable save/resume
and one simulated action chunk per environment. A resumed two-step run has not
been compared with an independent uninterrupted two-step run, so bitwise state
equivalence is untested. Real-environment memory carry across multiple chunks,
long-horizon reset behavior, steady-state throughput, long-run stability,
paired baseline comparisons and 24-trial task success remain separate gates.
The eight-step success counters and the two loss values do not establish task
performance or improvement. The user subsequently approved formal10K training
under Task5 below. The24-trial evaluation remains unstarted and requires separate
approval.

## Task 5: Formal Training with Selected Checkpoints

The user approved fresh eight-GPU training on port43545 with checkpoints only
at steps3000,5000,6000,8000,9000,10000. Preserve the Task4 scientific protocol,
the10000-step schedule and the original pretrained initialization. Use the
already approved `results/robottt3.0/train` output root; do not resume step2.
The first3000 steps have no checkpoint recovery point under this requested
schedule. Six full checkpoints are expected to occupy about348GB.

- [x] Add tests that exercise the real SFT runner with a CPU actor boundary.
- [x] Observe the old code fail the exact six-milestone requirement.
- [x] Add optional `runner.save_steps`; keep default interval behavior unchanged.
- [x] Set the RoboTTT3 list and disable periodic checkpoint saving.
- [x] Include runner and progress-helper source hashes in checkpoint metadata.
- [x] Run regression, formatting and independent review gates.
- [x] Verify configuration, free GPUs, disk space and fresh output path.
- [x] Start the formal run in a persistent session.
- [ ] Verify finite optimizer progress after initialization completes.

An explicit save-step list overrides periodic and implicit final saves;
`null` retains the existing behavior and an empty list disables scheduled saves.
The current training config disables validation/early stopping, so no best-model
checkpoint branch runs. Invalid types, nonpositive/out-of-range steps and
duplicates are rejected before training. Existing preflight artifacts are
preserved, but their old source manifest does not match the updated runner
manifest; strict loading is not weakened to reuse them.

Targeted CPU regression passed221 tests, with5 skipped and57 warnings in64.95s.
The initial schedule tests failed as expected (12failed,1passed); after the
change all13 passed. The repository-wide `pytest -q --maxfail=1` command stopped
during collection at `toolkits/realworld_check/test_gello.py` because the optional
real-robot dependency `gello` is absent (43warnings,1error in22.58s). This is not
a GR00T/TTT training dependency; no packages were installed to change the approved
environment. The targeted suite is not a claim that the whole repository passes.

Ruff lint, format and `git diff --check` passed. Independent review found no
launch-blocking issue; the best-model branch remains separate if validation is
enabled in a future configuration. Final resolved actor/data sections exactly
match the completed preflight. All eight GPUs were unused before launch and the
shared filesystem reported4.6TB available (99% used overall).

Formal launch: 2026-10-19 at approximately13:26UTC on port43545, tmux session
`robottt3-train`, initial driver PID1881740, Ray runtime `/tmp/r3-NbGdgO`.
The driver runs the existing environment's Python with
`examples/sft/train_vla_sft.py --config-name robocasa_bowlandcup_robottt3` and
`runner.logger.log_path=/mnt/public/chenzm/robottt/workspace/results/robottt3.0/train`.
It starts fresh and is not bounded by the preflight timeout. The session retains
its pane on process exit for diagnosis. Only this session was started; no
machine-wide Ray stop or cleanup was performed.

The train root contains `train.log`, a Git HEAD archive (`source_base.tar.gz`),
tracked diff (`source_changes.patch`), modified/untracked source overlay
(`source_overlay.tar.gz`) and `source_sha256.txt`. Compilation caches are under
`train/torchinductor_cache` and `train/triton_cache`; preflight caches are preserved.
Runtime uses GPUs0–7, OMP/MKL/OpenBLAS4 threads, the pinned native SDK on
PYTHONPATH, offline HF loading, disabled bytecode writes and a4GiB Ray object-store
cap. The actual launched process's RAY_TMPDIR and GPU environment were inspected.
At this launch record, the driver is alive and has generated all eight placements;
an optimizer-step result has not yet been observed.

## Repeat the Component Checks

Run these checks in the approved server checkout with the existing environment.
The CPU suite covers the training, inference and sampling contracts; the
distributed probe exercises a small native head, not the asset-backed 3B model.

```bash
cd /mnt/public/chenzm/robottt/workspace/RLinf-robottt3.0
probe_python=/mnt/public/chenzm/robottt/workspace/0History/gr00t-baseline-env-20261012/bin/python
export PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 USE_TF=0
export PYTHONPATH=/mnt/public/chenzm/robottt/workspace/RLinf-robottt3.0:/mnt/public/chenzm/robottt/h800_sft/code/Isaac-GR00T
CUDA_VISIBLE_DEVICES= "$probe_python" -m pytest -q \
  tests/unit_tests/test_gr00t_sequence_sft.py \
  tests/unit_tests/test_gr00t_fast_weights.py \
  tests/unit_tests/test_gr00t_native_ttt.py \
  tests/unit_tests/test_gr00t_native_rollout.py \
  tests/unit_tests/test_gr00t_native_eval.py \
  tests/unit_tests/test_gr00t_window_dataset.py \
  tests/unit_tests/test_gr00t_window_integration.py \
  tests/unit_tests/test_gr00t_window_sampling.py \
  tests/unit_tests/test_fsdp.py
```

After confirming all eight cards are available, run the compiled distributed
probe below. Select `False` instead of `True` for the eager probe. Both probes
must exit successfully on all ranks before claiming eight-GPU acceptance.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ROBOTTT_TINY_CUDA_TEST=1 ROBOTTT_TINY_COMPILE_TEST=1 \
"$probe_python" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=8 -m pytest -q -s --disable-warnings \
  tests/unit_tests/test_gr00t_native_ttt.py -k 'tiny_cuda_fsdp and True'
```

Full-size forward/backward, distributed checkpoint save/resume and one-chunk
OSMesa simulation on port43545 passed the bounded Task4 checks above. Keep
long-run and 24-trial evidence separate from this preflight. Use the approved
output root and agree on any additional directory names before creating them.
