# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Sequence weighting for the native NVIDIA GR00T N1.7 training head."""

import torch


def sequence_loss(
    action_loss: torch.Tensor,
    action_mask: torch.Tensor,
    anchor_valid: torch.Tensor,
    supervision: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Reduce native masked action errors with equal full-sequence weight.

    Each segment retains the full sequence denominator so that segment
    gradients add to the unsplit sequence gradient.
    """
    if action_loss.shape != action_mask.shape or action_loss.ndim != 3:
        raise ValueError("Action loss and mask must have matching [B*T,H,A] shapes")
    if anchor_valid.shape != supervision.shape or anchor_valid.ndim != 2:
        raise ValueError("Anchor masks must have matching [B,T] shapes")
    b, t = anchor_valid.shape
    if action_loss.shape[0] != b * t or denominator.shape != (b,):
        raise ValueError("Action batch and denominators must match anchor masks")
    if not torch.all(denominator > 0):
        raise ValueError("Every sequence must have a positive supervision denominator")
    selected = anchor_valid & supervision
    counts = action_mask.float().sum(dim=(1, 2)).reshape(b, t)
    if torch.any(selected & (counts <= 0)):
        raise ValueError("Supervised anchors must contain valid actions")
    per_anchor = action_loss.float().sum(dim=(1, 2)).reshape(b, t) / counts.clamp_min(1)
    masked = torch.where(selected, per_anchor, torch.zeros_like(per_anchor))
    return (masked.sum(1) / denominator.float()).mean()


def native_sequence_forward(model, forward_type, data: dict, memory=None) -> dict:
    """Route only SFT through the native training head, without RL replay."""
    from rlinf.models.embodiment.base_policy import ForwardType

    if forward_type != ForwardType.SFT:
        raise ValueError("This adapter accepts only ForwardType.SFT")
    if getattr(model, "_native_ttt_installed", False):
        from .native_ttt import native_ttt_sequence_forward

        return native_ttt_sequence_forward(model, data, memory)
    if memory is not None:
        raise ValueError("The native baseline does not accept recurrent memory")
    backbone_inputs, action_inputs = model.prepare_input(data)
    output = model.action_head(model.backbone(backbone_inputs), action_inputs)
    loss = sequence_loss(
        output["action_loss"],
        output["action_mask"],
        data["anchor_valid"],
        data["supervision"],
        data["denominator"],
    )
    return {"loss": loss}


def sequence_no_split_modules(gradient_checkpointing: bool) -> list[str]:
    """Select FSDP boundaries for the native sequence policy."""
    return [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        # FSDP must enclose activation checkpointing to retain canonical Adam names.
        "CheckpointWrapper" if gradient_checkpointing else "BasicTransformerBlock",
    ]


def build_model(cfg, torch_dtype=torch.float32):
    """Load native GR00T weights strictly; keep original parameter names."""
    import json
    import subprocess
    from pathlib import Path

    import gr00t
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from safetensors.torch import load_file

    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        redirect_qwen3_backbone_to_local,
    )

    if torch_dtype != torch.float32:
        raise ValueError("Native sequence SFT requires FP32 master parameters")
    dependency = Path(gr00t.__file__).resolve().parent.parent
    if dependency != Path(cfg.dependency_path).resolve():
        raise ValueError(
            f"Expected GR00T dependency at {cfg.dependency_path}, loaded {dependency}"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(dependency), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != cfg.dependency_commit:
        raise ValueError(f"Expected GR00T commit {cfg.dependency_commit}, got {commit}")
    subprocess.run(
        ["git", "-C", str(dependency), "diff", "--quiet", "HEAD"], check=True
    )

    class NativeSequencePolicy(Gr00tN1d7):
        _no_split_modules = sequence_no_split_modules(cfg.gradient_checkpointing)

        def forward(
            self, forward_type=ForwardType.SFT, data=None, memory=None, **kwargs
        ):
            return native_sequence_forward(self, forward_type, data, memory)

        def predict_with_memory(self, data, memory=None, **kwargs):
            from .native_ttt import native_ttt_predict

            return native_ttt_predict(self, data, memory, **kwargs)

    root = Path(cfg.model_path)
    config = Gr00tN1d7Config.from_pretrained(root, local_files_only=True)
    if config.action_horizon != 40 or config.max_action_dim != 132:
        raise ValueError("Expected native H40, 132-dimensional padded action head")
    config.load_bf16 = False
    config.backbone_trainable_params_fp32 = True
    config.no_split_modules = NativeSequencePolicy._no_split_modules
    config._no_split_modules = NativeSequencePolicy._no_split_modules
    # Eager construction leaves the native Beta sampler on real CPU tensors.
    with redirect_qwen3_backbone_to_local(config.model_name, cfg.backbone_model_path):
        model = NativeSequencePolicy(
            config, transformers_loading_kwargs={"local_files_only": True}
        )
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weights = {}
    for name in sorted(set(index["weight_map"].values())):
        shard = load_file(str(root / name), device="cpu")
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensor names in base checkpoint shards")
        weights.update(shard)
    model.load_state_dict(weights, strict=True)
    del weights
    model.float()
    if cfg.rl_head_config.use_ttt or cfg.rl_head_config.ttt.num_register_tokens:
        from .native_ttt import configure_native_ttt

        # New CPU parameter initialization must not consume native training RNG.
        with torch.random.fork_rng(devices=[]):
            configure_native_ttt(model, cfg)
    if cfg.rl_head_config.disable_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    if cfg.gradient_checkpointing:
        from functools import partial

        from gr00t.model.modules.dit import BasicTransformerBlock
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        apply_activation_checkpointing(
            model.action_head,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: isinstance(module, BasicTransformerBlock),
        )
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError("Native sequence SFT master parameters must all be FP32")
    return model
