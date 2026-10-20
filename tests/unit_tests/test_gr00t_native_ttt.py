# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native small GR00T heads exercise the recurrent adapter without model assets."""

import copy
import importlib
import importlib.util
import os

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

if importlib.util.find_spec("gr00t") is None:
    pytest.importorskip("gr00t", reason="Native GR00T SDK is an optional dependency")


def implementation():
    name = "rlinf.models.embodiment.gr00t.gr00t_n1d7.native_ttt"
    assert importlib.util.find_spec(name) is not None, (
        "Native recurrent adapter is missing"
    )
    return importlib.import_module(name)


class FeatureBackbone(nn.Module):
    """The asset-backed VLM boundary, replaced by explicit differentiable features."""

    def forward(self, data):
        return BatchFeature(
            {
                key: data[key]
                for key in (
                    "backbone_features",
                    "backbone_attention_mask",
                    "image_mask",
                )
            }
        )


class HeadOnlyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

        self.config = Gr00tN1d7Config(
            hidden_size=8,
            input_embedding_dim=8,
            backbone_embedding_dim=12,
            max_state_dim=3,
            max_action_dim=3,
            action_horizon=2,
            max_num_embodiments=2,
            state_dropout_prob=0.2,
            diffusion_model_cfg={
                "num_layers": 2,
                "num_attention_heads": 2,
                "attention_head_dim": 4,
                "positional_embeddings": None,
                "norm_type": "ada_norm",
                "dropout": 0.0,
                "final_dropout": False,
                "output_dim": 8,
                "interleave_self_attention": True,
            },
        )
        self.action_head = Gr00tN1d7ActionHead(self.config)
        self.backbone = FeatureBackbone()

    def prepare_input(self, data):
        return data, BatchFeature(
            {
                key: data[key]
                for key in ("state", "action", "action_mask", "embodiment_id")
                if key in data
            }
        )


def settings(use_ttt=True, registers=2):
    return OmegaConf.create(
        {
            "gradient_checkpointing": False,
            "training_memory_source": "predicted_final",
            "denoising_steps": 4,
            "rl_head_config": {
                "use_ttt": use_ttt,
                "ttt": {
                    "num_register_tokens": registers,
                    "hidden_size": 8,
                    "num_heads": 2,
                    "tokens_per_anchor": 3 + registers,
                    "compile_inner": False,
                },
            },
        }
    )


def batch(b=2, t=2):
    return {
        "backbone_features": torch.randn(b * t, 4, 12, requires_grad=True),
        "backbone_attention_mask": torch.ones(b * t, 4, dtype=torch.bool),
        "image_mask": torch.tensor([[True, True, False, False]]).expand(b * t, -1),
        "state": torch.randn(b * t, 1, 3),
        "action": torch.randn(b * t, 2, 3),
        "action_mask": torch.ones(b * t, 2, 3),
        "embodiment_id": torch.zeros(b * t, dtype=torch.long),
        "anchor_valid": torch.ones(b, t, dtype=torch.bool),
        "supervision": torch.ones(b, t, dtype=torch.bool),
        "denominator": torch.full((b,), float(t)),
        "policy_noise_seed": torch.arange(b * t).reshape(b, t) + 1001,
    }


def test_disabled_adapter_preserves_native_loss_gradient_and_rng():
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import native_sequence_forward

    torch.manual_seed(21)
    reference = HeadOnlyPolicy()
    model = copy.deepcopy(reference)
    data = batch()
    api = implementation()
    api.configure_native_ttt(model, settings(False, 0))
    torch.manual_seed(32)
    expected = native_sequence_forward(reference, ForwardType.SFT, data)["loss"]
    expected_rng = torch.get_rng_state().clone()
    expected.backward()
    torch.manual_seed(32)
    actual = api.native_ttt_sequence_forward(model, data)["loss"]
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert (
        dict(model.named_parameters()).keys()
        == dict(reference.named_parameters()).keys()
    )
    for (_, a), (_, e) in zip(model.named_parameters(), reference.named_parameters()):
        if e.grad is not None:
            torch.testing.assert_close(a.grad, e.grad, rtol=0, atol=0)


def test_register_only_stack_matches_native_augmented_tokens():
    torch.manual_seed(22)
    model = HeadOnlyPolicy()
    reference = copy.deepcopy(model.action_head.model)
    api = implementation()
    api.configure_native_ttt(model, settings(False, 2))
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    encoder = torch.randn(2, 4, 12)
    masks = torch.tensor([[True, True, False, False]]).expand(2, -1)
    kwargs = {
        "timestep": torch.tensor([31, 701]),
        "image_mask": masks,
        "backbone_attention_mask": torch.ones_like(masks),
    }
    expected = reference(hidden, encoder, **kwargs)
    actual, memory, update = model.action_head.model(
        hidden, encoder, anchor_valid=torch.ones(2, 1, dtype=torch.bool), **kwargs
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert memory is None and update.item() == 0


def test_predicted_writer_does_not_consume_ground_truth_or_supervision_rng():
    torch.manual_seed(23)
    model = HeadOnlyPolicy()
    reference = copy.deepcopy(model)
    data = batch()
    api = implementation()
    api.configure_native_ttt(model, settings())
    changed = {**data, "action": data["action"] + 11}
    torch.manual_seed(41)
    first = api.native_ttt_sequence_forward(model, data)
    writer_rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    second = api.native_ttt_sequence_forward(model, changed)
    assert not torch.equal(first["loss"], second["loss"])
    for a, b in zip(first["memory"].layers, second["memory"].layers):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.manual_seed(41)
    inputs, actions = reference.prepare_input(data)
    reference.action_head(reference.backbone(inputs), actions)
    assert torch.equal(writer_rng, torch.get_rng_state())
    assert first["memory"].position.tolist() == [2, 2]
    assert first["ttt_update_rms"].item() > 0


def test_future_supervision_learns_earlier_writer_and_padding_preserves_memory():
    torch.manual_seed(24)
    model = HeadOnlyPolicy()
    model.action_head.state_dropout_prob = 0
    data = batch()
    data["supervision"] = torch.tensor([[False, True], [True, False]])
    data["anchor_valid"][1, 1] = False
    data["denominator"] = torch.ones(2)
    api = implementation()
    api.configure_native_ttt(model, settings())
    torch.manual_seed(42)
    result = api.native_ttt_sequence_forward(model, data)
    result["loss"].backward()
    assert data["backbone_features"].grad[0].abs().sum().item() > 0
    assert data["backbone_features"].grad[3].count_nonzero().item() == 0
    assert result["memory"].position.tolist() == [2, 1]
    assert all(
        x.dtype == torch.float32 for layer in result["memory"].layers for x in layer
    )
    parameter = model.action_head.model.transformer_blocks[0].ttt.k_proj.weight
    assert parameter.grad is not None and parameter.grad.abs().sum().item() > 0
    altered = {**data, "backbone_features": data["backbone_features"].detach().clone()}
    altered["backbone_features"][3].add_(9)
    torch.manual_seed(42)
    other = api.native_ttt_sequence_forward(model, altered)
    for a, b in zip(result["memory"].layers, other["memory"].layers):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_training_and_inference_commit_same_writer_with_fixed_noise():
    torch.manual_seed(25)
    model = HeadOnlyPolicy()
    model.action_head.state_dropout_prob = 0
    data = batch(b=2, t=1)
    api = implementation()
    api.configure_native_ttt(model, settings())
    trained = api.native_ttt_sequence_forward(model, data)
    noise = torch.stack(
        [
            torch.randn((2, 3), generator=torch.Generator().manual_seed(int(s)))
            for s in data["policy_noise_seed"][:, 0]
        ]
    )
    model.eval()
    predicted = api.native_ttt_predict(model, data, policy_noise=noise)
    for a, b in zip(trained["memory"].layers, predicted["memory"].layers):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert predicted["memory"].position.tolist() == [1, 1]
    assert not predicted["actions"].requires_grad
    snapshot = copy.deepcopy(predicted["memory"])
    reset = api.native_ttt_predict(
        model,
        data,
        memory=predicted["memory"],
        reset_mask=torch.tensor([True, False]),
        noise=noise,
    )
    fresh = api.native_ttt_predict(model, data, noise=noise)
    assert reset["memory"].position.tolist() == [1, 2]
    for a, b, c in zip(reset["memory"].layers, fresh["memory"].layers, snapshot.layers):
        for x, y in zip(a, b):
            torch.testing.assert_close(x[0], y[0], rtol=0, atol=0)
    for a, b in zip(predicted["memory"].layers, snapshot.layers):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_policy_noise_is_required_for_predicted_training():
    model = HeadOnlyPolicy()
    data = batch()
    del data["policy_noise_seed"]
    api = implementation()
    api.configure_native_ttt(model, settings())
    with pytest.raises(ValueError, match="policy_noise_seed"):
        api.native_ttt_sequence_forward(model, data)


def test_installation_keeps_supervision_rng_unchanged():
    model = HeadOnlyPolicy()
    api = implementation()
    before = torch.get_rng_state().clone()
    api.configure_native_ttt(model, settings())
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("use_ttt", [True, False])
def test_installed_adapter_rejects_inherited_stateless_prediction(use_ttt):
    model = HeadOnlyPolicy().eval()
    api = implementation()
    api.configure_native_ttt(model, settings(use_ttt, 2))
    inputs, actions = model.prepare_input(batch(b=1, t=1))
    del actions["action"]  # Ordinary inference, not native real-time chunk inpainting.
    with pytest.raises(ValueError, match="predict_with_memory/native_ttt_predict"):
        model.action_head.get_action(model.backbone(inputs), actions)


def test_installed_block_rejects_missing_memory_protocol():
    model = HeadOnlyPolicy()
    api = implementation()
    api.configure_native_ttt(model, settings())
    block = model.action_head.model.transformer_blocks[0]
    temb = model.action_head.model.timestep_encoder(torch.tensor([31]))
    with pytest.raises(ValueError, match="predict_with_memory/native_ttt_predict"):
        block(
            torch.randn(1, 5, 8),
            encoder_hidden_states=torch.randn(1, 4, 12),
            encoder_attention_mask=torch.ones(1, 4, dtype=torch.bool),
            temb=temb,
        )


def test_checkpoint_recomputation_preserves_memory_and_writer_gradients():
    from functools import partial

    from gr00t.model.modules.dit import BasicTransformerBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    torch.manual_seed(28)
    model = HeadOnlyPolicy()
    model.action_head.state_dropout_prob = 0
    api = implementation()
    api.configure_native_ttt(model, settings())
    checked = copy.deepcopy(model)
    apply_activation_checkpointing(
        checked.action_head,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda module: isinstance(module, BasicTransformerBlock),
    )
    data = batch()
    torch.manual_seed(47)
    reference = api.native_ttt_sequence_forward(model, data)
    reference["loss"].backward()
    torch.manual_seed(47)
    result = api.native_ttt_sequence_forward(checked, data)
    result["loss"].backward()
    torch.testing.assert_close(result["loss"], reference["loss"], rtol=0, atol=0)
    assert result["memory"].position.tolist() == [2, 2]
    for first, second in zip(reference["memory"].layers, result["memory"].layers):
        for x, y in zip(first, second):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    for (_, first), (_, second) in zip(
        model.named_parameters(), checked.named_parameters()
    ):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def test_bfloat_native_head_keeps_fp32_fast_state():
    torch.manual_seed(29)
    model = HeadOnlyPolicy()
    api = implementation()
    api.configure_native_ttt(model, settings())
    for name, parameter in model.named_parameters():
        if ".ttt." not in name:
            parameter.data = parameter.data.to(torch.bfloat16)
    data = batch(b=1, t=2)
    for key in ("backbone_features", "state", "action", "action_mask"):
        data[key] = data[key].detach().to(torch.bfloat16)
    output = api.native_ttt_sequence_forward(model, data)
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert all(
        x.dtype == torch.float32 for layer in output["memory"].layers for x in layer
    )
    assert model.action_head.model.transformer_blocks[0].ttt.w1.dtype == torch.float32


def test_predict_rejects_corrupted_incoming_memory():
    torch.manual_seed(30)
    model = HeadOnlyPolicy()
    api = implementation()
    api.configure_native_ttt(model, settings())
    data = batch(b=2, t=1)
    model.eval()
    memory = api.native_ttt_predict(model, data)["memory"]
    bad = memory._replace(
        layers=tuple(
            layer._replace(w1=torch.full_like(layer.w1, float("nan")))
            for layer in memory.layers
        )
    )
    with pytest.raises(FloatingPointError, match="fast state"):
        api.native_ttt_predict(model, data, memory=bad)
    with pytest.raises(ValueError, match="reset_mask"):
        api.native_ttt_predict(
            model, data, memory=memory, reset_mask=torch.tensor([True])
        )


@pytest.mark.skipif(
    os.environ.get("ROBOTTT_TINY_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="Opt-in tiny CUDA/FSDP acceptance; requires an explicitly selected idle GPU",
)
@pytest.mark.parametrize(
    "compile_inner",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.environ.get("ROBOTTT_TINY_COMPILE_TEST") != "1",
                reason="Additional opt-in for the production compiled inner loop",
            ),
        ),
    ],
)
def test_tiny_cuda_fsdp_keeps_native_bf16_and_recurrent_fp32(tmp_path, compile_inner):
    _tiny_fsdp_probe(tmp_path, "cuda", compile_inner)


def test_tiny_cpu_fsdp_keeps_native_bf16_and_recurrent_fp32(tmp_path):
    _tiny_fsdp_probe(tmp_path, "cpu", False)


def _tiny_fsdp_probe(tmp_path, device_type, compile_inner):
    import json
    from datetime import timedelta
    from functools import partial
    from pathlib import Path

    import torch.distributed as dist
    from gr00t.model.modules.dit import BasicTransformerBlock
    from hydra import compose, initialize_config_dir
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    from rlinf.hybrid_engines.fsdp.mixed_precision import (
        build_mixed_precision,
        get_mixed_precision_wrap_policy,
    )
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy

    api = implementation()
    config_dir = Path(__file__).resolve().parents[2] / "examples/sft/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name="robocasa_bowlandcup_robottt3")
    fsdp_cfg = config.actor.fsdp_config
    tiny_cfg = OmegaConf.create(
        OmegaConf.to_container(config.actor.model, resolve=True)
    )
    tiny_cfg.rl_head_config.ttt.hidden_size = 8
    tiny_cfg.rl_head_config.ttt.num_heads = 2
    tiny_cfg.rl_head_config.ttt.tokens_per_anchor = (
        3 + tiny_cfg.rl_head_config.ttt.num_register_tokens
    )
    tiny_cfg.rl_head_config.ttt.compile_inner = compile_inner
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    init = (
        {"init_method": "env://"}
        if "RANK" in os.environ
        else {
            "init_method": f"file://{tmp_path / 'rendezvous'}",
            "rank": 0,
            "world_size": 1,
        }
    )
    assert not dist.is_initialized()
    backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(backend, timeout=timedelta(seconds=90), **init)
    try:

        class FSDPHeadPolicy(HeadOnlyPolicy):
            def prepare_input(self, data):
                # Match native Gr00tN1d7.prepare_input's parameter-dtype conversion.
                dtype = next(self.action_head.action_encoder.parameters()).dtype
                converted = {
                    key: value.to(device=device, dtype=dtype)
                    if value.is_floating_point()
                    else value.to(device)
                    for key, value in data.items()
                }
                return super().prepare_input(converted)

            def forward(self, data, memory=None):
                if memory is not None:
                    assert all(
                        value.dtype == torch.float32
                        for layer in memory.layers
                        for value in layer
                    )
                return api.native_ttt_sequence_forward(self, data, memory)

        torch.manual_seed(73)
        model = FSDPHeadPolicy()
        model.action_head.state_dropout_prob = 0
        api.configure_native_ttt(model, tiny_cfg)
        observed_native, observed_fast = [], []

        def observe_native(module, args):
            observed_native.append((args[0].dtype, module.to_q.weight.dtype))

        def observe_fast(module, args):
            # During FSDP checkpoint recomputation these are tensor views, not Parameters.
            observed_fast.append(module.w1.dtype)
            if args[1] is not None:
                assert all(value.dtype == torch.float32 for value in args[1])

        for block in model.action_head.model.transformer_blocks:
            block.attn1.register_forward_pre_hook(observe_native)
            block.ttt.register_forward_pre_hook(observe_fast)
        apply_activation_checkpointing(
            model.action_head,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: isinstance(module, BasicTransformerBlock),
        )
        model._no_split_modules = ["CheckpointWrapper"]
        mixed = build_mixed_precision(fsdp_cfg.mixed_precision)
        assert mixed.param_dtype == torch.bfloat16
        assert not mixed.cast_forward_inputs and not mixed.cast_root_forward_inputs
        base_policy = get_fsdp_wrap_policy(
            model, config=fsdp_cfg, model_type="gr00t_n1d7"
        )
        policy = get_mixed_precision_wrap_policy(model, fsdp_cfg, base_policy, mixed)
        wrapped = FSDP(
            model.to(device),
            auto_wrap_policy=policy,
            mixed_precision=mixed,
            device_id=device,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        data = {
            key: value.detach().to(device) for key, value in batch(b=1, t=2).items()
        }
        # Match production TBPTT accumulation: first segment does not synchronize.
        with wrapped.no_sync():
            first = wrapped(data=data)
            first["loss"].backward()
        memory = first["memory"].detach()
        snapshot = tuple(
            tuple(value.clone() for value in layer) for layer in memory.layers
        )
        second = wrapped(data=data, memory=memory)
        second["loss"].backward()
        assert second["memory"].position.tolist() == [4]
        for before, after in zip(snapshot, memory.layers):
            for x, y in zip(before, after):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert observed_native and set(observed_native) == {
            (torch.bfloat16, torch.bfloat16)
        }
        assert observed_fast and set(observed_fast) == {torch.float32}
        assert all(
            value.dtype == torch.float32
            for layer in second["memory"].layers
            for value in layer
        )
        gradients = [p.grad for p in wrapped.parameters() if p.grad is not None]
        assert gradients and all(
            torch.isfinite(gradient).all() for gradient in gradients
        )
        fast_grad = sum(
            (
                parameter.grad.abs().sum()
                for name, parameter in wrapped.named_parameters()
                if ".ttt." in name and parameter.grad is not None
            ),
            torch.zeros((), device=device),
        )
        dist.all_reduce(fast_grad)
        assert fast_grad.item() > 0
        print(
            json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "rank": dist.get_rank(),
                    "device": device_type,
                    "sharding": str(wrapped.sharding_strategy),
                    "native_compute": "bfloat16",
                    "fast_state": "float32",
                    "compile_inner": compile_inner,
                    "fast_gradient_l1": fast_grad.item(),
                    "carried_position": 4,
                    "peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device_type == "cuda"
                        else None
                    ),
                }
            )
        )
    finally:
        dist.destroy_process_group()
