# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Contracts for packed episode sampling and native GR00T sequence SFT."""

import itertools

import pytest
import torch


def test_native_ttt_dispatch_carries_memory_but_baseline_uses_native_head():
    from types import SimpleNamespace

    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import native_sequence_forward

    class NativeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, backbone, action):
            return {
                "action_loss": (self.weight * action).square(),
                "action_mask": torch.ones_like(action),
            }

    head = NativeHead()
    model = SimpleNamespace(
        prepare_input=lambda d: (d["x"], d["action"]),
        backbone=torch.nn.Identity(),
        action_head=head,
    )
    data = {
        "x": torch.ones(1, 1),
        "action": torch.tensor([[[3.0]]]),
        "anchor_valid": torch.ones(1, 1, dtype=torch.bool),
        "supervision": torch.ones(1, 1, dtype=torch.bool),
        "denominator": torch.ones(1),
    }
    result = native_sequence_forward(model, ForwardType.SFT, data, memory=None)
    assert result["loss"].item() == 36.0
    result["loss"].backward()
    assert head.weight.grad.item() == 36.0


def test_policy_noise_seeds_are_identity_based_and_do_not_advance_rng():
    import numpy as np

    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import sequence_policy_seeds

    batch = {
        "anchor_valid": torch.tensor([[True, True, False], [True, True, True]]),
        "descriptions": [
            {"epoch": 2, "episode": 7, "anchors": [0, 8]},
            {"epoch": 3, "episode": 9, "anchors": [4, 12, 20]},
        ],
    }
    state = torch.random.get_rng_state().clone()
    seeds = sequence_policy_seeds(batch, 0, 3, 42, torch.device("cpu"))
    assert seeds.shape == (2, 3)
    assert seeds.dtype == torch.int64
    assert seeds[0, 2].item() == 0
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(
        seeds[:, 1:], sequence_policy_seeds(batch, 1, 3, 42, torch.device("cpu"))
    )
    changed = {**batch, "descriptions": [dict(x) for x in batch["descriptions"]]}
    changed["descriptions"][0]["epoch"] = 4
    other = sequence_policy_seeds(changed, 0, 3, 42, torch.device("cpu"))
    assert not torch.equal(seeds[0, :2], other[0, :2])
    assert torch.equal(seeds[1], other[1])
    # Independent SeedSequence construction must leave NumPy's global stream alone.
    np.random.seed(13)
    first = np.random.random()
    sequence_policy_seeds(batch, 0, 3, 42, torch.device("cpu"))
    second = np.random.random()
    np.random.seed(13)
    assert (first, second) == (np.random.random(), np.random.random())


def test_ttt_worker_detaches_after_backward_and_resets_each_sequence():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from typing import NamedTuple

    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        FSDPGr00tSequenceSftWorker,
    )

    class State(NamedTuple):
        value: torch.Tensor

        def detach(self):
            return State(self.value.detach())

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.history = []

        def forward(self, forward_type, data, memory=None):
            self.history.append(None if memory is None else memory.value.item())
            if memory is not None:
                assert memory.value.grad_fn is None
            assert data["policy_noise_seed"].shape == (1, 2)
            history = 0.0 if memory is None else memory.value
            value = self.weight + history
            return {
                "loss": value.square(),
                "memory": State(value),
                "ttt_update_rms": self.weight.detach(),
            }

    model = Policy()
    worker = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        gradient_accumulation=1,
        amp_context=nullcontext(),
        before_micro_batch=lambda model, sync: nullcontext(),
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        cfg=SimpleNamespace(
            actor=SimpleNamespace(
                micro_batch_size=1,
                seed=42,
                model=SimpleNamespace(
                    tbptt_segment_length=2, rl_head_config=SimpleNamespace(use_ttt=True)
                ),
            )
        ),
    )
    batch = {
        "action": torch.zeros(4, 1, 1),
        "anchor_valid": torch.ones(1, 4, dtype=torch.bool),
        "supervision": torch.ones(1, 4, dtype=torch.bool),
        "denominator": torch.tensor([4.0]),
        "descriptions": [{"epoch": 0, "episode": 0, "anchors": [0, 8, 16, 24]}],
    }
    for _ in range(2):
        result = FSDPGr00tSequenceSftWorker.forward_backward_batch(worker, batch, True)
    assert model.history == [None, 2.0, None, 2.0]
    assert model.weight.grad.item() == 24.0
    assert result["ttt_update_rms"] == 2.0


def test_native_variant_configs_keep_the_control_protocol_and_protect_memory():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rlinf.utils.fsdp_config import _parse_wrapper_config, build_mixed_precision

    root = Path(__file__).resolve().parents[2] / "examples/sft/config"
    variants = []
    with initialize_config_dir(config_dir=str(root), version_base="1.1"):
        for name in (
            "robocasa_bowlandcup_gr00t_sequence128_aligned",
            "robocasa_bowlandcup_robottt3",
            "robocasa_bowlandcup_gr00t_sequence128_registers",
        ):
            variants.append(compose(config_name=name))
    baseline, ttt, registers = variants
    for cfg in variants:
        assert cfg.cluster.component_placement.actor == "0-7"
        assert cfg.actor.global_batch_size == 8
        assert cfg.actor.micro_batch_size == 1
        assert cfg.actor.global_batch_size // (8 * cfg.actor.micro_batch_size) == 1
    for candidate in variants[1:]:
        assert OmegaConf.to_container(candidate.data) == OmegaConf.to_container(
            baseline.data
        )
        assert OmegaConf.to_container(candidate.actor.optim) == OmegaConf.to_container(
            baseline.actor.optim
        )
        assert candidate.actor.global_batch_size == baseline.actor.global_batch_size
    assert baseline.data.robottt.terminal_window_probability == 0.0
    assert baseline.actor.optim.lr == 2.5e-5
    assert baseline.actor.model.rl_head_config.ttt.num_register_tokens == 0
    assert not registers.actor.model.rl_head_config.use_ttt
    assert registers.actor.model.rl_head_config.ttt.num_register_tokens == 16
    assert ttt.actor.model.training_memory_source == "predicted_final"
    mp = build_mixed_precision(ttt.actor.fsdp_config.mixed_precision)
    assert mp.param_dtype == torch.bfloat16
    assert not mp.cast_root_forward_inputs and not mp.cast_forward_inputs
    specs = _parse_wrapper_config(ttt.actor.fsdp_config)
    assert len(specs) == 1
    assert specs[0].targets == ("FastWeightTTT",)
    assert specs[0].mixed_precision.param_dtype == torch.float32
    assert specs[0].mixed_precision.reduce_dtype == torch.float32
    assert not specs[0].mixed_precision.cast_forward_inputs
    assert all(cfg.runner.logger.log_path is None for cfg in variants)


def test_training_entry_requires_output_path_before_initializing_workers():
    from omegaconf import OmegaConf

    from examples.sft.train_vla_sft import require_explicit_log_path

    for value in (None, "", "   "):
        cfg = OmegaConf.create({"runner": {"logger": {"log_path": value}}})
        with pytest.raises(ValueError, match="log_path"):
            require_explicit_log_path(cfg)
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": "/chosen/output"}}})
    require_explicit_log_path(cfg)
    assert cfg.runner.logger.log_path == "/chosen/output"


@pytest.mark.parametrize("world_size", [4, 8])
def test_full_microbatches_cross_epoch_without_dropping_episodes(world_size):
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import PackedEpisodeSampler

    ranks = [
        list(
            itertools.islice(
                iter(PackedEpisodeSampler(104, 42, rank, world_size)), 208 // world_size
            )
        )
        for rank in range(world_size)
    ]
    for epoch in (0, 1):
        entries = [entry for rank in ranks for entry in rank if entry[0] == epoch]
        assert len(entries) == 104
        assert sorted(row for _, row in entries) == list(range(104))
    samples_per_rank = 128 // world_size
    assert sum(len(rank[:samples_per_rank]) for rank in ranks) == 128
    assert [epoch for epoch, _ in ranks[0][:samples_per_rank]] == [
        index // (104 // world_size) for index in range(samples_per_rank)
    ]


def test_sampler_cursor_restores_across_epoch_boundary():
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import PackedEpisodeSampler

    iterator = iter(PackedEpisodeSampler(104, 42, 2, 4))
    list(itertools.islice(iterator, 32))
    state = iterator.state_dict()
    expected = list(itertools.islice(iterator, 70))
    restored = iter(PackedEpisodeSampler(104, 42, 2, 4))
    restored.load_state_dict(state)
    assert list(itertools.islice(restored, 70)) == expected
    with pytest.raises(ValueError):
        restored.load_state_dict({"cursor": -1})


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_loader_restores_consumed_not_prefetched_position(workers):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import PackedEpisodeSampler

    def make():
        kwargs = {"prefetch_factor": 1} if workers else {}
        return StatefulDataLoader(
            IndexDataset(),
            batch_size=16,
            sampler=PackedEpisodeSampler(104, 42, 0, 4),
            num_workers=workers,
            **kwargs,
        )

    loader = make()
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected = [next(iterator) for _ in range(4)]
    restored = make()
    restored.load_state_dict(state)
    actual = iter(restored)
    for want in expected:
        got = next(actual)
        assert torch.equal(got, want)
        assert got.shape == (16, 2)


class IndexDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 104

    def __getitem__(self, index):
        return torch.tensor(index)


def test_window_has_only_real_h40_targets_and_stable_epoch_seed():
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import window_indices

    for length in (40, 63, 400, 2000):
        anchors = window_indices(length, 7, 3, 42)
        assert anchors.tolist() == window_indices(length, 7, 3, 42).tolist()
        assert len(anchors) == min(128, (length - 40) // 8 + 1)
        assert anchors[0] >= 0
        assert anchors[-1] + 40 <= length
        assert all((anchors[1:] - anchors[:-1]) == 8)
    with pytest.raises(ValueError):
        window_indices(39, 0, 0, 42)


def test_sequence_loss_ignores_padding_and_weights_episodes_equally():
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_loss

    errors = torch.tensor([1.0, 3.0, 99.0, 5.0, 99.0, 99.0], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    loss = sequence_loss(
        errors[:, None, None],
        torch.ones(6, 1, 1),
        valid,
        valid.clone(),
        torch.tensor([2.0, 1.0]),
    )
    assert loss.item() == pytest.approx(3.5)
    loss.backward()
    assert errors.grad.tolist() == [0.25, 0.25, 0.0, 0.5, 0.0, 0.0]


def test_segment_loss_and_gradients_sum_to_full_sequence():
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_loss

    errors = torch.arange(1.0, 9.0, requires_grad=True).reshape(2, 4)
    errors.retain_grad()
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    denominator = torch.tensor([4.0, 2.0])
    full = sequence_loss(
        errors.reshape(8, 1, 1), torch.ones(8, 1, 1), valid, valid, denominator
    )
    parts = [
        sequence_loss(
            errors[:, i : i + 2].reshape(4, 1, 1),
            torch.ones(4, 1, 1),
            valid[:, i : i + 2],
            valid[:, i : i + 2],
            denominator,
        )
        for i in (0, 2)
    ]
    assert sum(parts).item() == pytest.approx(full.item())
    full.backward(retain_graph=True)
    expected = errors.grad.clone()
    errors.grad.zero_()
    sum(parts).backward()
    assert torch.equal(errors.grad, expected)


def test_zero_denominator_and_unsupervised_actions_are_rejected():
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_loss

    valid = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError):
        sequence_loss(
            torch.ones(1, 1, 1), torch.ones(1, 1, 1), valid, valid, torch.zeros(1)
        )
    with pytest.raises(ValueError):
        sequence_loss(
            torch.ones(1, 1, 1), torch.zeros(1, 1, 1), valid, valid, torch.ones(1)
        )


def test_segmented_backward_accumulates_once_and_syncs_only_last_segment():
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        FSDPGr00tSequenceSftWorker,
    )

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, forward_type, data):
            from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_loss

            error = (self.weight * data["action"]).square()
            return {
                "loss": sequence_loss(
                    error,
                    data["action_mask"],
                    data["anchor_valid"],
                    data["supervision"],
                    data["denominator"],
                )
            }

    sync_calls = []

    @contextmanager
    def context(model, sync):
        sync_calls.append(sync)
        yield

    model = TinyPolicy()
    worker = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        gradient_accumulation=2,
        cfg=SimpleNamespace(
            actor=SimpleNamespace(
                micro_batch_size=2, model=SimpleNamespace(tbptt_segment_length=2)
            )
        ),
        amp_context=nullcontext(),
        before_micro_batch=context,
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
    )
    batch = {
        "action": torch.arange(1.0, 9.0).reshape(8, 1, 1),
        "action_mask": torch.ones(8, 1, 1),
        "anchor_valid": torch.ones(2, 4, dtype=torch.bool),
        "supervision": torch.ones(2, 4, dtype=torch.bool),
        "denominator": torch.tensor([4.0, 4.0]),
    }
    for last in (False, True):
        metrics = FSDPGr00tSequenceSftWorker.forward_backward_batch(worker, batch, last)
    assert sync_calls == [False, False, False, True]
    assert metrics["loss"] == pytest.approx(25.5)
    assert model.weight.grad.item() == pytest.approx(51.0)
    assert model.weight.item() == 1.0


def test_checkpoint_resume_contract_rejects_changed_training_protocol():
    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        validate_resume_metadata,
    )

    metadata = {
        "world_size": 4,
        "protocol": {"lr": 1e-5},
        "data": {},
        "processor": {},
        "source": {},
        "parameters": {},
    }
    validate_resume_metadata(metadata, metadata)
    changed = {**metadata, "protocol": {"lr": 2e-5}}
    with pytest.raises(ValueError, match="protocol"):
        validate_resume_metadata(metadata, changed)


def test_native_action_head_uses_sequence_masks_and_produces_finite_gradients():
    from types import SimpleNamespace

    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import native_sequence_forward

    class NativeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, backbone, inputs):
            errors = (inputs["action"] * self.weight).square()
            return {"action_loss": errors, "action_mask": inputs["action_mask"]}

    head = NativeHead()
    model = SimpleNamespace(
        prepare_input=lambda data: (data, data),
        backbone=lambda data: data,
        action_head=head,
    )
    data = {
        "action": torch.tensor([1.0, 2.0, 900.0, 3.0]).reshape(4, 1, 1),
        "action_mask": torch.ones(4, 1, 1),
        "anchor_valid": torch.tensor([[True, True], [False, True]]),
        "supervision": torch.ones(2, 2, dtype=torch.bool),
        "denominator": torch.tensor([2.0, 1.0]),
    }
    out = native_sequence_forward(model, ForwardType.SFT, data)
    assert out["loss"].item() == pytest.approx(23.0)
    out["loss"].backward()
    assert head.weight.grad.item() == pytest.approx(23.0)
    with pytest.raises(ValueError):
        native_sequence_forward(model, ForwardType.DEFAULT, data)


def test_checkpoint_manifest_requires_optimizer_shards_and_checks_every_file():
    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        validate_checkpoint_manifest,
    )

    sizes = {
        "model_state_dict/full_weights.pt": 100,
        "data.pt": 20,
        "rng.pt": 20,
        "sequence_metadata.json": 20,
        "processor_bundle.json": 20,
        "sequence_progress.pt": 20,
        "dcp_checkpoint/.metadata": 10,
        "dcp_checkpoint/__0_0.distcp": 100,
    }
    manifest = {"step": 10, "files": sizes}
    validate_checkpoint_manifest(manifest, sizes)
    incomplete = {k: v for k, v in sizes.items() if not k.endswith(".distcp")}
    with pytest.raises(ValueError):
        validate_checkpoint_manifest({"step": 10, "files": incomplete}, incomplete)
    truncated = {**sizes, "dcp_checkpoint/__0_0.distcp": 10}
    with pytest.raises(ValueError):
        validate_checkpoint_manifest(manifest, truncated)
    with pytest.raises(ValueError):
        validate_checkpoint_manifest(
            manifest, {k: v for k, v in sizes.items() if k != "rng.pt"}
        )


def test_training_yaml_constructs_linear_warmup_then_cosine_scheduler():
    from pathlib import Path

    from omegaconf import OmegaConf

    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        FSDPGr00tSequenceSftWorker,
    )

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/sft/config/robocasa_bowlandcup_gr00t_sequence_sft.yaml"
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=cfg.actor.optim.lr)
    scheduler = FSDPGr00tSequenceSftWorker.build_lr_scheduler(
        None, optimizer, cfg.actor.optim
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        cfg.actor.optim.lr / cfg.actor.optim.lr_warmup_steps
    )
    for _ in range(200):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.actor.optim.lr)
    for _ in range(4900):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.actor.optim.lr / 2)
    for _ in range(4900):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_resume_metadata_accepts_json_roundtrip_but_rejects_changed_labels():
    import json

    from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
        validate_resume_metadata,
    )

    current = {
        "world_size": 4,
        "protocol": {
            "effective_model": {
                "native_config": {"id2label": {0: "LABEL_0", 1: "LABEL_1"}}
            }
        },
        "data": {},
        "processor": {},
        "source": {"source_sha256": {"worker.py": "unchanged"}},
        "parameters": {},
    }
    saved = json.loads(json.dumps(current))
    validate_resume_metadata(saved, current)
    current["protocol"]["effective_model"]["native_config"]["id2label"][0] = "changed"
    with pytest.raises(ValueError, match="protocol"):
        validate_resume_metadata(saved, current)
    current = json.loads(json.dumps(saved))
    current["source"]["source_sha256"]["worker.py"] = "changed"
    with pytest.raises(ValueError, match="source"):
        validate_resume_metadata(saved, current)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA FSDP device")
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_native_sequence_fsdp_restores_adam_strictly(activation_checkpointing):
    import copy
    import socket

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_no_split_modules

    class Qwen3VLTextDecoderLayer(torch.nn.Linear):
        pass

    class Qwen3VLVisionBlock(torch.nn.Linear):
        pass

    class BasicTransformerBlock(torch.nn.Linear):
        pass

    assert not torch.distributed.is_initialized()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.distributed.init_process_group(
        "nccl", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        torch.manual_seed(42)
        block = BasicTransformerBlock(4, 4)
        if activation_checkpointing:
            block = checkpoint_wrapper(
                block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
        native = torch.nn.Sequential(
            Qwen3VLTextDecoderLayer(4, 4), Qwen3VLVisionBlock(4, 4), block
        ).cuda()
        native._no_split_modules = sequence_no_split_modules(activation_checkpointing)
        policy = get_fsdp_wrap_policy(native, model_type="gr00t_n1d7")
        model = FSDP(native, device_id=0, use_orig_params=True, auto_wrap_policy=policy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.ones(2, 4, device="cuda")

        def step():
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            optimizer.step()

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        step()
        model_state, optimizer_state = copy.deepcopy(
            get_state_dict(model, optimizer, options=options)
        )
        step()
        expected = copy.deepcopy(model.state_dict())
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
            options=options,
        )
        step()
        actual = model.state_dict()
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert {state["step"].item() for state in optimizer.state.values()} == {2}
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "args,start,count",
    [
        ((40, 0, 0, 42), 0, 1),
        ((63, 2, 1, 42), 4, 3),
        ((658, 0, 0, 42), 1, 78),
        ((2000, 7, 3, 42), 231, 128),
        ((5000, 91, 9, 42), 1266, 128),
    ],
)
def test_terminal_windows_zero_probability_preserves_legacy(args, start, count):
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import window_indices

    actual = window_indices(*args, terminal_window_probability=0.0)
    assert actual.tolist() == list(range(start, start + count * 8, 8))


@pytest.mark.parametrize("length", [40, 63, 658, 1055, 1056, 2000])
def test_terminal_windows_full_probability_ends_at_last_real_action(length):
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import window_indices

    actual = window_indices(length, 7, 3, 42, terminal_window_probability=1.0)
    count = min(128, (length - 40) // 8 + 1)
    assert actual.tolist() == list(range(length - 40 - (count - 1) * 8, length - 39, 8))
    assert actual[0] >= 0
    assert len(set(actual.tolist())) == count


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_terminal_windows_reject_invalid_probability(probability):
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import window_indices

    with pytest.raises(ValueError, match="probability"):
        window_indices(2000, 7, 3, 42, terminal_window_probability=probability)


def test_terminal_windows_mixture_is_repeatable_and_preserves_random_branch():
    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import window_indices

    selected = 0
    for epoch in range(20):
        for episode in range(104):
            legacy = window_indices(5000, episode, epoch, 42)
            mixed = window_indices(
                5000, episode, epoch, 42, terminal_window_probability=0.2
            )
            repeated = window_indices(
                5000, episode, epoch, 42, terminal_window_probability=0.2
            )
            assert mixed.tolist() == repeated.tolist()
            assert len(mixed) == 128
            assert all((mixed[1:] - mixed[:-1]) == 8)
            assert mixed[0] >= 0 and mixed[-1] + 40 <= 5000
            if mixed[-1] == 4960:
                selected += 1
            else:
                assert mixed.tolist() == legacy.tolist()
    # Random windows may also hit the endpoint; at this length that chance is tiny.
    assert 0.16 < selected / 2080 < 0.24


def test_terminal_windows_dataset_uses_probability_without_changing_valid_count():
    from types import SimpleNamespace

    from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import EpisodeDataset

    config = {
        "seed": 42,
        "data": {
            "stride": 8,
            "max_anchors": 128,
            "action_horizon": 40,
            "terminal_window_probability": 1.0,
        },
    }
    loader = SimpleNamespace(
        episodes_metadata=[{"episode_index": 0}], episode_lengths=[658]
    )
    dataset = EpisodeDataset(config, processor=None, loader=loader)
    terminal = dataset.describe((0, 0))
    assert terminal["anchors"] == list(range(2, 619, 8))
    assert terminal["valid"] == 78
    assert terminal["terminal_aligned"] is True
    config["data"]["terminal_window_probability"] = 0.0
    legacy = dataset.describe((0, 0))
    assert legacy["anchors"] == list(range(1, 618, 8))
    assert legacy["valid"] == 78
    assert legacy["terminal_aligned"] is False
    assert terminal["augmentation_seed"] == legacy["augmentation_seed"]
