# H800 SFT migration and launch ledger

Approved root: /mnt/public/chenzm/robottt/h800_sft.
Both experiments: 10000 optimizer steps, disjoint four-GPU groups, fresh weights.
Source A800 training and source assets remain unchanged.

## Approved protocol
- Sequence: global8/micro1/4GPU/accum2, H40/stride8/max128; terminal probability0.
- Padding is excluded from supervision; valid anchors average121.576923/sequence.
- H40: global1024/micro16/4GPU/accum16; uniform permutation of all112737 legal windows.
- Expected exposure: sequence about972.6 windows/step versus H401024 (~5.3% more).
- Same native GR00T N1.7, processor, action loss, AdamW2.5e-5/warmup200/cosine10K.
- FP32 masters/FSDP BF16 compute/FP32 reduce and buffers; no extra AMP; noTTT/registers.
- These compare sampling/training protocols, not a TTT contribution.

## Migration
- RLinf7e90641c4135229db6d3426950cdb45cee6259a6 copied with source uncommitted work preserved.
- Isaac-GR00T23ace64f17aa5015259b8609d371eb61a357c776 clean, copied unchanged.
- Dataset737 files SHA256 exactly match source.
- GR00T55 files SHA256 exactly match source, including both weight shards.
- Cosmos transfer interrupted by SSH broken pipe; source/target first2309430784 bytes hash matched.
- Cosmos append resumed from that byte, with SSH keepalive; small assets copied. Fullhash pending.
- Environment tar completed; standalone CPython3.11.14 copied and venv interpreter/editablefinder relocated.
- Package identity matches source; torch2.11.0+cu129/flash-attn2.8.3/CUDA12.9.
- All8 H800 GPUs passed real BF16 flash attention forward+backward with finite gradients.
- Destination Git, FFmpeg4.4 runtime libraries, libGL and libEGL installed.
- Source identities: source_models.sha256 and source_environment.json.

## H40 implementation and tests
- New window_sampling.py: legal-start index plus deterministic global permutation and rank shards.
- Sampler4 tests passed, including non-divisible lengths and iterator restore.
- New window_dataset.py: native parquet extraction, sparse video decode, same processor.
- Real data test: image/state/action values exactly equal native full-episode extraction at first/interior/final windows.
- Stateful loader resume test passed for0 and2 workers at consumed, not prefetched, boundary.
- Shared builder routes independent_h40 explicitly; unknown sampling modes fail.
- H40 epoch accounting uses legal-window count; metadata records units and both new source hashes.
- Integration4 tests RED then GREEN; T1 loss is mean across windows.
- Sequence regression:33/33 passed. One pre-existing test hardcoded the old1e-5 LR; now checks the configured warmup/peak/halfway values.
- H40 processed-batch integration:3/3 passed, including2 workers and bitwise repeated augmentation.
- Sampling4/4 and routing/loss4/4 passed;44 targeted tests total (not the entire RLinf suite).
- Independent final code review found no Critical/Important H40-specific defects.
- Minor deferred: full bitwise optimizer+processed-data resume has not been demonstrated; cursor and optimizer state are tested separately.
- GPU probe found RLinf enumerates all8 devices despite driver CUDA_VISIBLE_DEVICES.
- Ruling: explicit actor placement0-3(sequence) and4-7(H40); both actual RLinf four-worker UUID tests passed. Both Ray instances exited cleanly.
- Official HuggingFace endpoint unreachable; matching Cosmos ModelScope mirror linked SHA256 equals source and sampled bytes matched.
- Slow SSH append stopped only its identified dd PID3407; remaining bytes now fetched using validated HTTP Range, final SHA256 mandatory.
- Metrics valid_anchors is the per-rank mean after reduction; multiply by world_size4 for global effective windows.
- Cosmos completed by matching-hash ModelScope Range continuation in100.2s; full SHA256 fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede verified.
- Entire source_models.sha256 manifest passed; see models_verified.log.
- Launch script code/run_experiment.sh guards against occupied GPUs, duplicate launch and output overwrite; independently verifies assets.
- Formal fresh10K launches dispatched2026-10-17T16:04Z (2026-10-18 China time).
- sequence128 launcherPID83076, h40 launcherPID83101; both preflight passed.
- Launch verification passed: both groups completed at least5 real optimizer steps; all recorded loss/gradient/LR values finite, TTT update0.
- Step5 sequence: loss1.2728607655, grad_norm0.70005548, LR6.24999984e-7, global8, global valid anchors936.
- Step5 H40: loss1.2723076344, grad_norm0.72564572, same LR, global1024 and1024 valid anchors.
- Physical process mapping confirmed: sequence ranks PID94568/94569/94570/94573 on GPUs0-3; H40 PID107593/107594/107595/107598 on GPUs4-7.
- Each GPU currently about35-36GiB allocated by the worker; no OOM/nonfinite/traceback markers and no launcher exit_code files. Training continues independently of SSH.

## Rulings
- Keep existing shared sequence worker for H40 with T=1: preserves native model/processor/loss and avoids a second training implementation.
- Infinite global window stream crosses dataset epochs without duplicate padding/drop; save each sample epoch/start and checkpoint consumed cursor.
- No claim of exact exposure/compute equality; record actual windows and wall time.

## Exact window-budget restart (2026-10-18)
- User approved matching each H40 optimizer update to the sequence group's valid-window count.
- Sequence sampling/global8/micro1 and its per-sequence loss weighting remain unchanged.
- H40 uses micro16, a precomputed 10000-step budget, and ceil(N/(4*16)) synchronized microbatches.
- H40 padding has zero supervision; backward scale is 4*16/N before FSDP rank averaging.
- Equality means valid H40 windows, optimizer updates and LR schedule, not FLOPs, wall time, sample identity or identical loss weighting.
- Old runs stopped at sequence80/H4078, both exit134 after targeted driver termination; GPUs verified idle.
- User-approved previous_run holds old non-TensorBoard runtime records (~13MiB); active configs retained.
- Deleted exactly the two old events.out.tfevents files (78192 bytes); no checkpoints or assets deleted.
- Tests-first: four new sampler/gradient tests failed on missing APIs then passed; real padding/plan test failed before implementation then passed.
- Matched data-loader consumed-cursor recovery verified with 0 and 2 workers.
- Existing sequence_dataset.py, model sft.py and Isaac-GR00T source are unchanged by this restart.
- Ruling: reuse existing files and this ledger; user requested no unsolicited directory names.
- Fresh post-format test run: sampling8/8, real dataset5/5, integration5/5, sequence regression33/33; 51 total. Full RLinf suite was not run. Existing PyTorch deprecation/NO_SHARD warnings remain.
- Ruff full check on the six touched Python files and bash syntax check passed; git diff --check passed.
- Source hash verification confirms unchanged sequence_dataset.py, native sft.py and sequence128/config.yaml.
- Budget first10:861,979,1011,992,936,986,999,935,975,998; min825/max1024; total9726167.
- Budget SHA256:50e992ad432e046094c601c9e2aaa77f3c772166e2da5ef459cb7a620049f3a1; all80 archived sequence steps match exactly.
- Official open issue/PR search for "window budget" returned0; broad GR00T SFT search returned4 without establishing an identical implementation.
- Independent read-only final review: no critical/important/minor correctness findings.
- Review scope ruling: real GPU memory/distributed execution will be verified during launch; full model+optimizer+processed-data bitwise resume remains unproven, not a claim of this restart.
- New launches dispatched with sequence launcher4044856 and H40 launcher4044857; awaiting actual optimizer progress verification.
- Launch verification passed: both groups completed8 real optimizer updates; per-step global windows and LR matched for steps1-8 and all four rank sample logs independently agreed.
- First8 matched windows:861,979,1011,992,936,986,999,935. The actual first step includes all-padding tail ranks and completed with finite gradients.
- Sequence step8 loss1.2846151590/grad_norm0.7078156471; H40 step8 loss1.2792589664/grad_norm0.7147234082. LR9.99999997e-7 in both.
- All recorded losses/gradients finite, TTT update0, no OOM/traceback/count mismatch; 8 physical worker GPUs each~35-36GiB with correct disjoint UUID mapping.
- Drivers4044899(sequence) and4044898(H40) remain running independently of SSH; no exit_code files at verification.
- Both started.json files contain identical full10000-step budget identity; fresh TensorBoard event files are distinct from the deleted old runs.
