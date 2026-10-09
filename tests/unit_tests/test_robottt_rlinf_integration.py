# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native RLinf boundaries: deterministic resume, TBPTT sync and rollout reset."""

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import EpisodeSampler
from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.hybrid_engines.fsdp.utils import (
    _filter_present_wrap_class_names,
    get_lr_scheduler,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import FastState, Memory


class IndexDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 104

    def __getitem__(self, item):
        return torch.tensor(item)


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_episode_sampler_restores_consumed_cursor(workers):
    def loader():
        sampler = EpisodeSampler(104, 42, 2, 3, 8)
        options = {
            "dataset": IndexDataset(),
            "sampler": sampler,
            "batch_size": 1,
            "num_workers": workers,
            "generator": torch.Generator().manual_seed(77),
        }
        if workers:
            options.update(multiprocessing_context="spawn", prefetch_factor=2)
        return StatefulDataLoader(**options)

    original = loader()
    iterator = iter(original)
    for _ in range(5):
        next(iterator)
    state = original.state_dict()
    expected = list(iterator)
    restored = loader()
    restored.load_state_dict(state)
    actual = list(restored)
    assert len(expected) == len(actual) == 8
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    restored.sampler.set_epoch(3)
    assert len(list(restored)) == 13
    assert all(index[0] == 3 for index in restored.sampler.indices)


@pytest.mark.parametrize("is_last_micro_batch", [False, True])
def test_tbptt_worker_preserves_denominator_and_covers_forward_with_no_sync(
    tmp_path, monkeypatch, is_last_micro_batch
):
    from rlinf.workers.sft.fsdp_vla_robottt_worker import FSDPVlaRoboTTTWorker

    active = [None]
    observations = []

    @contextmanager
    def context(model, is_last):
        assert active[0] is None
        active[0] = is_last
        try:
            yield
        finally:
            active[0] = None

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.25]))

        def forward(self, forward_type, data, incoming):
            assert active[0] is not None
            observations.append(active[0])
            if incoming is not None:
                assert incoming.layers[0].w1.grad_fn is None
            torch.testing.assert_close(data["denominator"], torch.tensor([3.0, 5.0]))
            prediction = self.weight - data["target"].reshape(2, -1)
            loss = (
                (prediction.square() * data["anchor_valid"]).sum(1)
                / data["denominator"]
            ).mean()
            sync = active[0]

            def check_backward(gradient):
                assert active[0] == sync
                return gradient

            loss.register_hook(check_backward)
            state = FastState(*(self.weight.expand(2, 1) for _ in range(4)))
            return {
                "loss": loss,
                "memory": Memory((state,), torch.zeros(2, dtype=torch.long)),
                "ttt_update_rms": torch.tensor(0.0),
            }

    model = Toy()
    valid = torch.arange(8)[None] < torch.tensor([3, 5])[:, None]
    target = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    batch = {
        "anchor_valid": valid,
        "supervision": valid,
        "denominator": torch.tensor([3.0, 5.0]),
        "target": target,
        "descriptions": [],
    }
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {
                    "model": {"sequence_training": True, "tbptt_segment_length": 2}
                },
                "runner": {"logger": {"log_path": str(tmp_path)}},
            }
        ),
        model=model,
        device="cpu",
        amp_context=nullcontext(),
        before_micro_batch=context,
        grad_scaler=SimpleNamespace(scale=lambda x: x),
        gradient_accumulation=2,
        global_step=1,
        _rank=0,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    result = FSDPVlaRoboTTTWorker.forward_backward_batch(
        worker, batch, is_last_micro_batch
    )
    expected = (
        ((0.25 - target.reshape(2, 8)).square() * valid)
        .sum(1)
        .div(torch.tensor([3, 5]))
        .mean()
    )
    gradient = (2 * (0.25 - target.reshape(2, 8)) * valid).sum(1).div(
        torch.tensor([3, 5])
    ).mean() / 2
    assert result["loss"] == pytest.approx(expected.item())
    torch.testing.assert_close(model.weight.grad, gradient.reshape(1))
    assert observations == [
        False,
        False,
        False,
        is_last_micro_batch,
    ]  # final segment is entirely padding


def test_rollout_memory_is_owned_once_per_worker_stage_and_resets_rows():
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    class Policy:
        supports_recurrent_memory = True
        use_ttt = True
        settings = {
            "inference": {"denoise_steps": 4, "memory_write_source": "final_prediction"}
        }

        def predict_action_batch(self, env_obs, incoming_memory, reset_mask, mode):
            position = (
                torch.zeros(2, dtype=torch.long)
                if incoming_memory is None
                else incoming_memory.position
            )
            position = torch.where(reset_mask, 0, position) + 1
            state = FastState(*(torch.ones(2, 1) for _ in range(4)))
            return np.zeros((2, 8, 12), np.float32), {
                "memory": Memory((state,), position),
                "memory_diagnostics": {"dit_forwards": 5},
                "forward_inputs": {},
            }

    worker = SimpleNamespace(
        hf_model=Policy(),
        _policy_memories={},
        cfg=OmegaConf.create({"rollout": {"record_recurrent_state": False}}),
        _train_sampling_params={},
        _eval_sampling_params={},
        model_cfg=SimpleNamespace(model_type="robottt"),
        enable_dagger=False,
        algorithm_cfg={},
        expert_model=None,
    )
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    predict(worker, {}, mode="eval", stage_id=0, reset_mask=torch.tensor([True, True]))
    previous = worker._policy_memories[("eval", 0)]
    predict(worker, {}, mode="eval", stage_id=1, reset_mask=torch.tensor([True, True]))
    predict(worker, {}, mode="eval", stage_id=0, reset_mask=torch.tensor([True, False]))
    assert previous.position.tolist() == [1, 1]
    assert worker._policy_memories[("eval", 0)].position.tolist() == [1, 2]
    assert worker._policy_memories[("eval", 1)].position.tolist() == [1, 1]


def test_reset_signal_is_independent_of_termination_in_channel_schema():
    first = EnvOutput(
        obs={"states": torch.zeros(1, 16)},
        dones=torch.ones(1, dtype=torch.bool),
        reset_mask=torch.zeros(1, dtype=torch.bool),
    ).to_dict()
    second = EnvOutput(
        obs={"states": torch.zeros(1, 16)},
        dones=torch.zeros(1, dtype=torch.bool),
        reset_mask=torch.ones(1, dtype=torch.bool),
    ).to_dict()
    merged = EnvOutput.merge_env_outputs([first, second])
    assert merged["dones"].tolist() == [True, False]
    assert merged["reset_mask"].tolist() == [False, True]


def test_optional_inferred_fsdp_targets_and_scheduler_boundaries():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.warns(UserWarning, match="absent"):
        assert _filter_present_wrap_class_names(
            model, ["Linear", "OptionalVariant"]
        ) == ["Linear"]
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="None"):
        _filter_present_wrap_class_names(model, ["Typo"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_lr_scheduler("linear_warmup_cosine", optimizer, 200, 50000)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 / 200)
    observed = []
    for step in range(202):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert observed[199] == observed[200] == 5e-5
    assert observed[201] < observed[200]
