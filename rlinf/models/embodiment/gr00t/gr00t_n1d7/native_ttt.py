# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
# Native block operations adapted from NVIDIA Isaac-GR00T (Apache-2.0).
"""Functional recurrence at the native GR00T attention/FFN boundary."""

import torch
import torch.nn.functional as F
from gr00t.model.modules.dit import AlternateVLDiT, BasicTransformerBlock, _sdpa_context
from omegaconf import OmegaConf
from torch import nn

from .ttt import FastWeightTTT, Memory, TTTConfig


class NativeTTTBlock(BasicTransformerBlock):
    """Native spatial block with an explicit, caller-owned fast state."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
        *,
        incoming=None,
        position=None,
        anchor_valid=None,
        reset_mask=None,
    ):
        if anchor_valid is None:
            raise ValueError(
                "Native TTT requires an explicit memory protocol; "
                "use predict_with_memory/native_ttt_predict"
            )
        # Keep the pinned native block's order, mask selection and squeeze behavior.
        normed = (
            self.norm1(hidden_states, temb)
            if self.norm_type == "ada_norm"
            else self.norm1(hidden_states)
        )
        if self.pos_embed is not None:
            normed = self.pos_embed(normed)
        with _sdpa_context():
            attention = self.attn1(
                normed,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=(
                    encoder_attention_mask
                    if encoder_hidden_states is not None
                    else attention_mask
                ),
            )
        if self.final_dropout:
            attention = self.final_dropout(attention)
        hidden_states = attention + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        b, t = anchor_valid.shape
        residual, candidate, update = self.ttt(
            hidden_states.reshape(b, t, *hidden_states.shape[1:]),
            incoming,
            position,
            anchor_valid,
            reset_mask,
        )
        hidden_states = hidden_states + residual.flatten(0, 1).to(hidden_states.dtype)
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states, candidate, update


class NativeTTTDiT(AlternateVLDiT):
    """Native mask routing with explicit state arguments through each block."""

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
        *,
        memory=None,
        anchor_valid=None,
        reset_mask=None,
    ):
        if anchor_valid is None:
            raise ValueError(
                "Installed TTT/registers require an explicit memory protocol; "
                "use predict_with_memory/native_ttt_predict"
            )
        if not self._native_ttt_enabled:
            result = super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask,
                return_all_hidden_states,
                image_mask,
                backbone_attention_mask,
            )
            return result, None, hidden_states.new_zeros((), dtype=torch.float32)
        if return_all_hidden_states:
            raise ValueError(
                "Recurrent DiT returns explicit memory, not hidden snapshots"
            )
        if image_mask is None or backbone_attention_mask is None:
            raise ValueError("Native alternating DiT requires image and backbone masks")
        b = anchor_valid.shape[0]
        _validate_memory(memory, b, len(self.transformer_blocks))
        position = (
            torch.zeros(b, device=hidden_states.device, dtype=torch.long)
            if memory is None
            else memory.position
        )
        if reset_mask is not None:
            position = torch.where(reset_mask, 0, position)
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        states, updates = [], []
        for index, block in enumerate(self.transformer_blocks):
            cross = index % 2 == 0
            selected_mask = (
                non_image_attention_mask
                if index % (2 * self.attend_text_every_n_blocks) == 0
                else image_attention_mask
            )
            hidden_states, candidate, update = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states if cross else None,
                encoder_attention_mask=selected_mask if cross else None,
                temb=temb,
                incoming=None if memory is None else memory.layers[index],
                position=position,
                anchor_valid=anchor_valid,
                reset_mask=reset_mask,
            )
            states.append(candidate)
            updates.append(update)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        candidate = Memory(tuple(states), position + anchor_valid.long().sum(1))
        return self.proj_out_2(hidden_states), candidate, torch.stack(updates).mean()


def configure_native_ttt(model, cfg) -> None:
    """Install only new modules after strict base loading, before FSDP/checkpointing."""
    with torch.random.fork_rng(devices=[]):
        _configure_native_ttt(model, cfg)


def _configure_native_ttt(model, cfg):
    if getattr(model, "_native_ttt_installed", False):
        raise ValueError("Native TTT is already configured")
    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    enabled = bool(config["rl_head_config"]["use_ttt"])
    options = dict(config["rl_head_config"]["ttt"])
    registers = int(options.pop("num_register_tokens"))
    if registers < 0:
        raise ValueError("Register count must be nonnegative")
    model._native_ttt_enabled = enabled
    model._native_ttt_installed = enabled or registers > 0
    if not model._native_ttt_installed:
        return
    head = model.action_head
    dit = head.model
    if not isinstance(dit, AlternateVLDiT) or not dit.config.interleave_self_attention:
        raise ValueError("Native TTT requires the pinned alternating GR00T DiT")
    if config.get("training_memory_source", "predicted_final") != "predicted_final":
        raise ValueError("Native TTT supports predicted_final training history")
    width = dit.inner_dim
    if enabled:
        options.setdefault("hidden_size", width)
        options.setdefault("tokens_per_anchor", 1 + head.action_horizon + registers)
        ttt_config = TTTConfig(**options)
        if ttt_config.hidden_size != width or ttt_config.tokens_per_anchor != (
            1 + head.action_horizon + registers
        ):
            raise ValueError(
                "TTT dimensions must match native state/action/register tokens"
            )
    # Specialize only these loaded instances; native parameters keep their names.
    dit.__class__ = NativeTTTDiT
    dit._native_ttt_enabled = enabled
    if enabled:
        for block in dit.transformer_blocks:
            if type(block) is not BasicTransformerBlock:
                raise ValueError("Install native TTT before activation/FSDP wrapping")
            block.__class__ = NativeTTTBlock
            block.ttt = FastWeightTTT(ttt_config)
            block.ttt._fsdp_wrap_name = "robottt_fast_weights"
    if registers:
        head.ttt_registers = nn.Parameter(torch.randn(1, registers, width) * 0.02)
    head._native_ttt_register_count = registers
    head.num_inference_timesteps = int(
        config.get("denoising_steps", head.num_inference_timesteps)
    )
    if head.num_inference_timesteps < 1:
        raise ValueError("Denoising steps must be positive")


def _validate_memory(memory, batch_size, layers=None):
    if memory is None:
        return
    if memory.position.shape != (batch_size,) or memory.position.dtype != torch.long:
        raise ValueError("Memory position must match the environment batch")
    if (memory.position < 0).any():
        raise ValueError("Memory positions must be nonnegative")
    if layers is not None and len(memory.layers) != layers:
        raise ValueError("Memory layer count must match the native DiT")
    if any(value.dtype != torch.float32 for state in memory.layers for value in state):
        raise TypeError(
            "Fast memory must enter the model in FP32; disable FSDP input casting"
        )
    if any(value.shape[0] != batch_size for state in memory.layers for value in state):
        raise ValueError("Every fast-state tensor must match the environment batch")


def _finite_memory(memory):
    if memory is not None and any(
        not torch.isfinite(value).all() for state in memory.layers for value in state
    ):
        raise FloatingPointError("Non-finite recurrent fast state")


def _encode(model, data, *, training):
    backbone_inputs, action_inputs = model.prepare_input(data)
    head = model.action_head
    head.set_frozen_modules_to_eval_mode()
    output = head.process_backbone_output(model.backbone(backbone_inputs))
    state = action_inputs.state
    if state.shape[1] != head.config.state_history_length:
        raise ValueError("State history does not match the native head")
    state_features = head.state_encoder(
        state.view(state.shape[0], 1, -1), action_inputs.embodiment_id
    )
    if training and head.training and head.state_dropout_prob > 0:
        dropped = (
            torch.rand(state_features.shape[0], device=state_features.device)
            < head.state_dropout_prob
        )
        dropped = dropped[:, None, None].to(state_features.dtype)
        state_features = state_features * (1 - dropped)
    features = {
        key: output[key]
        for key in ("backbone_features", "backbone_attention_mask", "image_mask")
    }
    return features, state_features, action_inputs


def _velocity(
    head, actions, buckets, embodiment, features, state, memory, valid, reset=None
):
    encoded = head.action_encoder(actions, buckets, embodiment)
    if head.config.add_pos_embed:
        slots = torch.arange(encoded.shape[1], dtype=torch.long, device=encoded.device)
        encoded = encoded + head.position_embedding(slots).unsqueeze(0)
    tokens = [state, encoded]
    if head._native_ttt_register_count:
        tokens.insert(0, head.ttt_registers.expand(actions.shape[0], -1, -1))
    output, candidate, update = head.model(
        hidden_states=torch.cat(tokens, dim=1),
        encoder_hidden_states=features["backbone_features"],
        timestep=buckets,
        image_mask=features["image_mask"],
        backbone_attention_mask=features["backbone_attention_mask"],
        memory=memory,
        anchor_valid=valid,
        reset_mask=reset,
    )
    prediction = head.action_decoder(output, embodiment)[:, -head.action_horizon :]
    return prediction, candidate, update


def _sample_actions(
    head, noise, embodiment, features, state, memory, valid, reset=None
):
    # Preserve native Euler dtype/arithmetic; only fast-weight algebra is FP32.
    actions = noise
    dt = 1.0 / head.num_inference_timesteps
    for index in range(head.num_inference_timesteps):
        bucket = int(
            index / float(head.num_inference_timesteps) * head.num_timestep_buckets
        )
        buckets = torch.full((actions.shape[0],), bucket, device=actions.device)
        velocity, _, _ = _velocity(
            head, actions, buckets, embodiment, features, state, memory, valid, reset
        )
        actions = actions + dt * velocity
    return actions


def _write(head, actions, embodiment, features, state, memory, valid, reset=None):
    buckets = torch.full(
        (actions.shape[0],),
        head.num_timestep_buckets - 1,
        device=actions.device,
        dtype=torch.long,
    )
    _, candidate, update = _velocity(
        head, actions, buckets, embodiment, features, state, memory, valid, reset
    )
    return candidate, update


def native_ttt_sequence_forward(model, data, memory=None) -> dict:
    """Native supervised samples followed by one differentiable predicted write."""
    from rlinf.models.embodiment.base_policy import ForwardType

    from .sft import native_sequence_forward, sequence_loss

    if not getattr(model, "_native_ttt_installed", False):
        result = native_sequence_forward(model, ForwardType.SFT, data)
        return {
            **result,
            "memory": None,
            "ttt_update_rms": result["loss"].new_zeros(()),
        }
    valid = data["anchor_valid"]
    b, t = valid.shape
    _validate_memory(memory, b)
    _finite_memory(memory)
    enabled = model._native_ttt_enabled
    seeds = data.get("policy_noise_seed")
    if enabled and (seeds is None or seeds.shape != (b, t)):
        raise ValueError("Expected policy_noise_seed with shape [B,T]")
    features, state, action_inputs = _encode(model, data, training=True)
    head = model.action_head
    actions = action_inputs.action
    # Exactly native sampling order and dtypes, including its CPU Beta sampler.
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    time = head.sample_time(
        actions.shape[0], device=actions.device, dtype=actions.dtype
    )
    time = time[:, None, None]
    trajectory = (1 - time) * noise + time * actions
    target = actions - noise
    buckets = (time[:, 0, 0] * head.num_timestep_buckets).long()
    if not enabled:
        prediction, _, update = _velocity(
            head,
            trajectory,
            buckets,
            action_inputs.embodiment_id,
            features,
            state,
            None,
            valid,
        )
        errors = (
            F.mse_loss(prediction, target, reduction="none") * action_inputs.action_mask
        )
        loss = sequence_loss(
            errors,
            action_inputs.action_mask,
            valid,
            data["supervision"],
            data["denominator"],
        )
        return {"loss": loss, "memory": None, "ttt_update_rms": update}

    def anchor(value, index):
        return value.reshape(b, t, *value.shape[1:])[:, index]

    losses, updates = [], []
    reset = data.get("reset_mask")
    for index in range(t):
        current_features = {
            key: anchor(value, index) for key, value in features.items()
        }
        current_state = anchor(state, index)
        embodiment = anchor(action_inputs.embodiment_id, index)
        current_valid = valid[:, index : index + 1]
        current_reset = reset if index == 0 else None
        prediction, _, _ = _velocity(
            head,
            anchor(trajectory, index),
            anchor(buckets, index),
            embodiment,
            current_features,
            current_state,
            memory,
            current_valid,
            current_reset,
        )
        mask = anchor(action_inputs.action_mask, index)
        errors = F.mse_loss(prediction, anchor(target, index), reduction="none") * mask
        losses.append(
            sequence_loss(
                errors,
                mask,
                current_valid,
                data["supervision"][:, index : index + 1],
                data["denominator"],
            )
        )
        policy_noise = torch.stack(
            [
                torch.randn(
                    actions.shape[1:],
                    device=actions.device,
                    dtype=features["backbone_features"].dtype,
                    generator=torch.Generator(device=actions.device).manual_seed(
                        int(seed)
                    ),
                )
                for seed in seeds[:, index]
            ]
        )
        with torch.no_grad():
            predicted = _sample_actions(
                head,
                policy_noise,
                embodiment,
                current_features,
                current_state,
                memory,
                current_valid,
                current_reset,
            )
        memory, update = _write(
            head,
            predicted.detach(),
            embodiment,
            current_features,
            current_state,
            memory,
            current_valid,
            current_reset,
        )
        updates.append(update)
    _finite_memory(memory)
    return {
        "loss": torch.stack(losses).sum(),
        "memory": memory,
        "ttt_update_rms": torch.stack(updates).mean(),
    }


@torch.no_grad()
def native_ttt_predict(model, data, memory=None, **kwargs) -> dict:
    """Generate a model-space action chunk and return a single detached candidate."""
    noise = kwargs.pop("policy_noise", None)
    legacy_noise = kwargs.pop("noise", None)
    if noise is not None and legacy_noise is not None:
        raise ValueError("Supply policy_noise only once")
    if noise is None:
        noise = legacy_noise
    reset = kwargs.pop("reset_mask", None)
    mode = kwargs.pop("memory_mode", "carry")
    if kwargs:
        raise TypeError(
            f"Unsupported native recurrent inference options: {sorted(kwargs)}"
        )
    if mode not in {"carry", "reset"}:
        raise ValueError("memory_mode must be carry or reset")
    if not getattr(model, "_native_ttt_installed", False):
        raise ValueError(
            "Use the original native predictor when TTT/registers are disabled"
        )
    features, state, action_inputs = _encode(model, data, training=False)
    head = model.action_head
    b = state.shape[0]
    device = state.device
    _validate_memory(memory, b)
    _finite_memory(memory)
    if mode == "reset":
        reset = torch.ones(b, device=device, dtype=torch.bool)
    elif reset is not None:
        reset = reset.to(device=device, dtype=torch.bool)
        if reset.shape != (b,):
            raise ValueError("reset_mask must have one entry per environment")
    if noise is None:
        noise = torch.randn(
            (b, head.action_horizon, head.action_dim),
            device=device,
            dtype=features["backbone_features"].dtype,
        )
    if noise.shape != (b, head.action_horizon, head.action_dim):
        raise ValueError(
            "Policy noise must match the full native action horizon and width"
        )
    if noise.device != device or noise.dtype != features["backbone_features"].dtype:
        raise ValueError("Policy noise must match the native feature device and dtype")
    if not torch.isfinite(noise).all():
        raise FloatingPointError("Non-finite policy noise")
    valid = torch.ones(b, 1, device=device, dtype=torch.bool)
    actions = _sample_actions(
        head, noise, action_inputs.embodiment_id, features, state, memory, valid, reset
    )
    if model._native_ttt_enabled:
        candidate, update = _write(
            head,
            actions,
            action_inputs.embodiment_id,
            features,
            state,
            memory,
            valid,
            reset,
        )
    else:
        candidate, update = None, actions.new_zeros((), dtype=torch.float32)
    if not torch.isfinite(actions).all():
        raise FloatingPointError("Non-finite predicted model-space actions")
    _finite_memory(candidate)
    return {
        "actions": actions,
        "memory": None if candidate is None else candidate.detach(),
        "ttt_update_rms": update,
    }
