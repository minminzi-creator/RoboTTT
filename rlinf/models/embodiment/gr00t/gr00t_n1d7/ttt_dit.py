# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""One model factory and explicit DiT implementation for training and evaluation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .flow import euler_sample, sample_path, sequence_loss
from .ttt import FastState, FastWeightTTT, Memory, TTTConfig


class RecurrentBlock(nn.Module):
    """Preserve pretrained attention/FFN, insert TTT between their residuals."""

    def __init__(self, spatial: nn.Module, ttt_config: TTTConfig | None):
        super().__init__()
        self.spatial = spatial
        self.ttt = FastWeightTTT(ttt_config) if ttt_config else None

    def forward(
        self,
        hidden: Tensor,
        encoder: Tensor | None,
        mask: Tensor | None,
        temb: Tensor,
        state: FastState | None,
        position: Tensor,
        valid: Tensor,
        reset: Tensor | None,
    ) -> tuple[Tensor, FastState | None, Tensor]:
        from gr00t.model.modules.dit import _sdpa_context

        block = self.spatial
        normed = (
            block.norm1(hidden, temb)
            if block.norm_type == "ada_norm"
            else block.norm1(hidden)
        )
        if block.pos_embed is not None:
            normed = block.pos_embed(normed)
        with _sdpa_context():
            attention = block.attn1(
                normed, encoder_hidden_states=encoder, attention_mask=mask
            )
        if block.final_dropout is not None:
            attention = block.final_dropout(attention)
        hidden = attention + hidden
        update = hidden.new_zeros((), dtype=torch.float32)
        if self.ttt is not None:
            b, t = valid.shape
            residual, state, update = self.ttt(
                hidden.reshape(b, t, *hidden.shape[1:]), state, position, valid, reset
            )
            hidden = hidden + residual.flatten(0, 1)
        return hidden + block.ff(block.norm3(hidden)), state, update


class ExplicitDiT(nn.Module):
    """Functional version of GR00T AlternateVLDiT; checkpointing cannot commit."""

    def __init__(
        self, source: nn.Module, ttt_config: TTTConfig | None, checkpoint_blocks: bool
    ):
        super().__init__()
        if not source.config.interleave_self_attention:
            raise ValueError(
                "This reproduction requires the pretrained alternating VL DiT"
            )
        self.use_ttt = ttt_config is not None
        self.checkpoint_blocks = checkpoint_blocks
        self.attend_text_every_n_blocks = source.attend_text_every_n_blocks
        self.transformer_blocks = nn.ModuleList(
            [RecurrentBlock(block, ttt_config) for block in source.transformer_blocks]
        )
        for name in ("timestep_encoder", "norm_out", "proj_out_1", "proj_out_2"):
            setattr(self, name, getattr(source, name))

    def forward(
        self,
        hidden: Tensor,
        encoder: Tensor,
        timestep: Tensor,
        image_mask: Tensor,
        attention_mask: Tensor,
        incoming: Memory | None,
        valid: Tensor,
        reset: Tensor | None = None,
    ) -> tuple[Tensor, Memory | None, Tensor]:
        b, _ = valid.shape
        position = (
            torch.zeros(b, device=hidden.device, dtype=torch.long)
            if incoming is None
            else incoming.position
        )
        if reset is not None:
            position = torch.where(reset, 0, position)
        temb = self.timestep_encoder(timestep)
        image_mask, attention_mask = image_mask.bool(), attention_mask.bool()
        masks = (image_mask & attention_mask, ~image_mask & attention_mask)
        states, updates = [], []
        for i, block in enumerate(self.transformer_blocks):
            is_cross = i % 2 == 0
            mask = (
                masks[int(i % (2 * self.attend_text_every_n_blocks) == 0)]
                if is_cross
                else None
            )
            state = (
                incoming.layers[i] if incoming is not None and self.use_ttt else None
            )
            args = (
                hidden,
                encoder if is_cross else None,
                mask,
                temb,
                state,
                position,
                valid,
                reset,
            )
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                hidden, candidate, update = checkpoint(
                    block, *args, use_reentrant=False
                )
            else:
                hidden, candidate, update = block(*args)
            if self.use_ttt:
                states.append(candidate)
            updates.append(update)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        memory = (
            Memory(tuple(states), position + valid.long().sum(1))
            if self.use_ttt
            else None
        )
        return self.proj_out_2(hidden), memory, torch.stack(updates).mean()


class SequencePolicy(nn.Module):
    """GR00T with shared SFT / inference features and explicit memory ownership."""

    def __init__(self, base: nn.Module, config: dict):
        super().__init__()
        self.base = base
        self.settings = config
        model_cfg = config["model"]
        source = base.action_head.model
        self.use_ttt = model_cfg["use_ttt"]
        ttt_cfg = None
        if self.use_ttt:
            ttt_cfg = TTTConfig(**model_cfg["ttt"])
            if ttt_cfg.hidden_size != source.inner_dim:
                raise ValueError("TTT width must match the pretrained DiT")
            expected = 1 + base.action_head.action_horizon + model_cfg["registers"]
            if ttt_cfg.tokens_per_anchor != expected:
                raise ValueError("TTT mini-batch must contain exactly one anchor")
            self.registers = nn.Parameter(
                torch.randn(1, model_cfg["registers"], source.inner_dim) * 0.02
            )
        elif model_cfg["registers"]:
            raise ValueError("The baseline has no registers")
        base.action_head.model = ExplicitDiT(
            source, ttt_cfg, model_cfg["gradient_checkpointing"]
        )
        self.base.requires_grad_(True)

    def initialize_memory(self, batch_size: int) -> Memory | None:
        """Initialize unwrapped train/eval state with gradients to learned W0.

        Under FSDP pass incoming=None to forward instead, so initialization
        occurs while each owning block has its parameters unsharded.
        """
        if not self.use_ttt:
            return None
        blocks = self.base.action_head.model.transformer_blocks
        if any(not isinstance(block, RecurrentBlock) for block in blocks):
            raise RuntimeError("Under FSDP initialize via forward(incoming=None)")
        states = tuple(block.ttt.initial_state(batch_size) for block in blocks)
        return Memory(
            states,
            torch.zeros(batch_size, device=states[0].w1.device, dtype=torch.long),
        )

    def encode(self, batch: dict) -> tuple[dict, Tensor]:
        backbone_input = self.base.backbone.prepare_input(batch)
        output = self.base.backbone(backbone_input)
        head = self.base.action_head
        output = head.process_backbone_output(output)
        state = batch["state"].reshape(batch["state"].shape[0], 1, -1)
        state_features = head.state_encoder(state, batch["embodiment_id"])
        if self.training and head.state_dropout_prob:
            keep = (
                torch.rand(state.shape[0], device=state.device)
                >= head.state_dropout_prob
            )
            state_features = state_features * keep[:, None, None]
        return output, state_features

    def velocity(
        self,
        actions: Tensor,
        buckets: Tensor,
        embodiment: Tensor,
        features: dict,
        state_features: Tensor,
        incoming: Memory | None,
        valid: Tensor,
        reset: Tensor | None = None,
    ) -> tuple[Tensor, Memory | None, Tensor]:
        head = self.base.action_head
        encoded = head.action_encoder(actions, buckets, embodiment)
        if head.config.add_pos_embed:
            positions = torch.arange(encoded.shape[1], device=actions.device)
            encoded = encoded + head.position_embedding(positions)[None]
        tokens = [state_features, encoded]
        if self.use_ttt:
            tokens.insert(0, self.registers.expand(actions.shape[0], -1, -1))
        hidden, candidate, update = head.model(
            torch.cat(tokens, 1),
            features["backbone_features"],
            buckets,
            features["image_mask"],
            features["backbone_attention_mask"],
            incoming,
            valid,
            reset,
        )
        prediction = head.action_decoder(hidden, embodiment)[:, -head.action_horizon :]
        return prediction, candidate, update

    def forward(self, batch: dict, incoming: Memory | None = None) -> dict:
        valid = batch["anchor_valid"]
        b, t = valid.shape
        features, state_features = self.encode(batch)
        actions = batch["action"].reshape(b, t, *batch["action"].shape[1:])
        xt, target, buckets = sample_path(actions)
        prediction, candidate, update = self.velocity(
            xt.flatten(0, 1),
            buckets.flatten(),
            batch["embodiment_id"],
            features,
            state_features,
            incoming,
            valid,
        )
        loss = sequence_loss(
            prediction.reshape_as(actions),
            target,
            batch["action_mask"].reshape_as(actions),
            valid,
            batch["supervision"],
            batch["denominator"],
        )
        return {"loss": loss, "memory": candidate, "ttt_update_rms": update}

    @torch.no_grad()
    def predict(
        self,
        batch: dict,
        incoming: Memory | None = None,
        reset: Tensor | None = None,
        noise: Tensor | None = None,
        diagnostics: dict | None = None,
    ) -> tuple[Tensor, Memory | None]:
        features, state_features = self.encode(batch)
        b = state_features.shape[0]
        head = self.base.action_head
        if noise is None:
            noise = torch.randn(
                b,
                head.action_horizon,
                head.action_dim,
                device=state_features.device,
                dtype=torch.float32,
            )
        valid = torch.ones(b, 1, device=noise.device, dtype=torch.bool)

        if diagnostics is not None:
            diagnostics["dit_forwards"] = 0

        def field(x, bucket, state):
            if diagnostics is not None:
                diagnostics["dit_forwards"] += 1
            prediction, candidate, _ = self.velocity(
                x,
                bucket,
                batch["embodiment_id"],
                features,
                state_features,
                state,
                valid,
                reset,
            )
            return prediction, candidate

        settings = self.settings["inference"]
        actions, candidate = euler_sample(
            field,
            noise,
            incoming,
            steps=settings["denoise_steps"],
            write_source=settings["memory_write_source"]
            if self.use_ttt
            else "denoise_step",
            write_index=settings["memory_write_denoise_index"] if self.use_ttt else -1,
        )
        return actions, candidate.detach() if candidate is not None else None


def build_model(config: dict, policy_class=SequencePolicy) -> SequencePolicy:
    """Load ALL base weights first, then initialize new modules normally in FP32."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from transformers import PreTrainedModel

    class LocalGr00t(Gr00tN1d7):
        """Explicit offline constructor; upstream dispatch hardcodes Hub names.

        Same modules and parameter names as upstream, without patching global
        from_pretrained methods or altering the configured backbone depth.
        """

        def __init__(self, config, backbone_path):
            PreTrainedModel.__init__(self, config)
            self.backbone = Qwen3Backbone(
                model_name=backbone_path,
                tune_llm=config.tune_llm,
                tune_visual=config.tune_visual,
                select_layer=config.select_layer,
                reproject_vision=config.reproject_vision,
                use_flash_attention=config.use_flash_attention,
                load_bf16=False,
                tune_top_llm_layers=config.tune_top_llm_layers,
                trainable_params_fp32=True,
                transformers_loading_kwargs={"local_files_only": True},
            )
            self.action_head = Gr00tN1d7ActionHead(config)
            self.collator = Gr00tN1d7DataCollator(
                model_name=backbone_path,
                model_type=config.backbone_model_type,
                transformers_loading_kwargs={"local_files_only": True},
            )

    cfg = Gr00tN1d7Config.from_pretrained(
        config["model"]["base_path"], local_files_only=True
    )
    cfg.load_bf16 = False
    cfg.backbone_trainable_params_fp32 = True
    base, loading = LocalGr00t.from_pretrained(
        config["model"]["base_path"],
        config=cfg,
        dtype=torch.float32,
        backbone_path=config["model"]["backbone_path"],
        local_files_only=True,
        output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise RuntimeError(
                f"Base checkpoint did not load exactly: {key}={loading[key]}"
            )
    base.float()
    if config["model"]["disable_dropout"]:
        for module in base.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    if config["model"]["gradient_checkpointing"]:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = policy_class(base, config)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError("Factory must retain FP32 master parameters")
    return model


def parameter_counts(model: nn.Module) -> dict:
    result = {
        "total": 0,
        "trainable": 0,
        "backbone": 0,
        "action_head": 0,
        "dit_spatial": 0,
        "ttt": 0,
        "registers": 0,
    }
    for name, parameter in model.named_parameters():
        n = parameter.numel()
        result["total"] += n
        result["trainable"] += n if parameter.requires_grad else 0
        if "backbone." in name:
            result["backbone"] += n
        if "action_head." in name:
            result["action_head"] += n
        if "action_head.model." in name and ".ttt." not in name:
            result["dit_spatial"] += n
        if ".ttt." in name:
            result["ttt"] += n
        if name == "registers":
            result["registers"] += n
    return result
