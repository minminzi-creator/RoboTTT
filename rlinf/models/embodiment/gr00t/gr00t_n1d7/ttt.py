# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Differentiable mini-batch TTT-MLP with functional recurrent state.

The dual update follows Sun et al. (2024), official ttt-lm-pytorch cd831db.
All token gradients in ONE anchor are evaluated at the incoming weights.
Queries read inclusive prefix updates. Anchors remain recurrent.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FastState(NamedTuple):
    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor

    def detach(self) -> "FastState":
        return FastState(*(x.detach() for x in self))

    def reorder(self, indices: Tensor) -> "FastState":
        return FastState(*(x.index_select(0, indices.to(x.device)) for x in self))


class Memory(NamedTuple):
    layers: tuple[FastState, ...]
    position: Tensor  # Number of real anchors consumed per environment.

    def detach(self) -> "Memory":
        return Memory(tuple(x.detach() for x in self.layers), self.position.detach())

    def reset(self, initial: "Memory", mask: Tensor) -> "Memory":
        """Restore selected environments without touching their neighbors."""
        if len(initial.layers) != len(self.layers):
            raise ValueError("Memory layer counts differ")
        return Memory(
            tuple(choose_rows(a, b, mask) for a, b in zip(self.layers, initial.layers)),
            torch.where(mask, initial.position, self.position),
        )

    def reorder(self, indices: Tensor) -> "Memory":
        return Memory(
            tuple(x.reorder(indices) for x in self.layers),
            self.position.index_select(0, indices.to(self.position.device)),
        )


@dataclass(frozen=True)
class TTTConfig:
    hidden_size: int = 1536
    num_heads: int = 32
    mlp_ratio: int = 4
    tokens_per_anchor: int = 57
    base_lr: float = 0.1
    gate_init: float = 0.001
    o_proj_std: float = 0.02
    rope_theta: float = 10000.0
    position_mode: str = "flattened"
    compile_inner: bool = False

    def __post_init__(self):
        if self.hidden_size % (2 * self.num_heads):
            raise ValueError("RoPE requires an even integral head dimension")
        if self.position_mode not in {"flattened", "shared_temporal", "slot_only"}:
            raise ValueError(f"Unknown position mode: {self.position_mode}")
        if self.tokens_per_anchor < 1 or self.mlp_ratio < 1 or self.base_lr <= 0:
            raise ValueError("Invalid TTT dimensions or learning rate")


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    centered = x - x.mean(-1, keepdim=True)
    normalized = centered * (centered.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    return normalized * gamma + beta


def reconstruction_grad(
    x: Tensor, target: Tensor, gamma: Tensor, beta: Tensor
) -> Tensor:
    """Analytic gradient of half summed squared LN reconstruction error.

    No tensors are detached: outer differentiation includes the inner Hessian.
    """
    centered = x - x.mean(-1, keepdim=True)
    inv_std = (centered.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    normalized = centered * inv_std
    grad = (normalized * gamma + beta - target) * gamma
    return inv_std * (
        grad
        - grad.mean(-1, keepdim=True)
        - normalized * (grad * normalized).mean(-1, keepdim=True)
    )


def gelu_grad(x: Tensor) -> Tensor:
    """Derivative of the official tanh-approximate GELU."""
    c = (2.0 / torch.pi) ** 0.5
    t = torch.tanh(c * (x + 0.044715 * x.pow(3)))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t.square()) * c * (1 + 0.134145 * x.square())


def dual_update(
    state: FastState,
    q: Tensor,
    k: Tensor,
    target: Tensor,
    eta: Tensor,
    gamma: Tensor,
    beta: Tensor,
) -> tuple[Tensor, FastState]:
    """One anchor; tensors [B,H,M,D], eta [B,H,M,M]."""
    w1, b1, w2, b2 = state
    z1 = k @ w1 + b1
    hidden = F.gelu(z1, approximate="tanh")
    z2 = hidden @ w2 + b2
    g2 = reconstruction_grad(z2, target, gamma, beta)
    g1 = (g2 @ w2.transpose(-1, -2)) * gelu_grad(z1)
    causal_eta = eta.tril()
    qz1 = q @ w1 + b1 - (eta * (q @ k.transpose(-1, -2)).tril()) @ g1 - causal_eta @ g1
    qhidden = F.gelu(qz1, approximate="tanh")
    qz2 = (
        qhidden @ w2
        + b2
        - (eta * (qhidden @ hidden.transpose(-1, -2)).tril()) @ g2
        - causal_eta @ g2
    )
    last = eta[..., -1, :, None]
    candidate = FastState(
        w1 - (last * k).transpose(-1, -2) @ g1,
        b1 - (last * g1).sum(-2, keepdim=True),
        w2 - (last * hidden).transpose(-1, -2) @ g2,
        b2 - (last * g2).sum(-2, keepdim=True),
    )
    return q + layer_norm(qz2, gamma, beta), candidate


def choose_rows(old: FastState, new: FastState, mask: Tensor) -> FastState:
    """Select without modifying either source; padding remains bitwise unchanged."""
    return FastState(
        *(torch.where(mask.reshape(-1, 1, 1, 1), b, a) for a, b in zip(old, new))
    )


def scan_update(
    state: FastState,
    q: Tensor,
    k: Tensor,
    target: Tensor,
    eta: Tensor,
    gamma: Tensor,
    beta: Tensor,
    valid: Tensor,
) -> tuple[Tensor, FastState]:
    """Scan recurrent anchors without merging their independent inner updates."""
    outputs = []
    for i in range(q.shape[1]):
        readout, candidate = dual_update(
            state, q[:, i], k[:, i], target[:, i], eta[:, i], gamma, beta
        )
        state = choose_rows(state, candidate, valid[:, i])
        outputs.append(readout)
    return torch.stack(outputs, 1), state


# Compilation caches kernels, never sequence state. Eager is the reference path.
compiled_scan_update = torch.compile(scan_update, fullgraph=True, dynamic=False)


def token_positions(position: Tensor, valid: Tensor, slots: int, mode: str) -> Tensor:
    anchor = position[:, None] + valid.long().cumsum(1) - valid.long()
    slot = torch.arange(slots, device=position.device)
    if mode == "flattened":
        return anchor[..., None] * slots + slot
    if mode == "shared_temporal":
        return anchor[..., None].expand(-1, -1, slots)
    if mode == "slot_only":
        return slot.expand(*anchor.shape, -1)
    raise ValueError(mode)


def rotary(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Official half-split rotation; x [B,T,H,M,D]."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angles = positions.to(x.dtype)[..., None] * inv
    angles = torch.cat((angles, angles), -1).unsqueeze(2)
    a, b = x.chunk(2, -1)
    return x * angles.cos() + torch.cat((-b, a), -1) * angles.sin()


class FastWeightTTT(nn.Module):
    """TTT branch accepting [B,T,M,D] and returning a candidate fast state."""

    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        d, h = config.hidden_size // config.num_heads, config.num_heads
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.normal_(
                layer.weight, std=config.o_proj_std if name == "o_proj" else 0.02
            )
            setattr(self, name, layer)
        self.lr_proj = nn.Linear(config.hidden_size, h)
        nn.init.normal_(self.lr_proj.weight, std=0.02)
        nn.init.zeros_(self.lr_proj.bias)
        self.token_bias = nn.Parameter(torch.zeros(config.tokens_per_anchor))
        self.w1 = nn.Parameter(torch.randn(h, d, config.mlp_ratio * d) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(h, 1, config.mlp_ratio * d))
        self.w2 = nn.Parameter(torch.randn(h, config.mlp_ratio * d, d) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(h, 1, d))
        self.gamma = nn.Parameter(torch.ones(h, 1, d))
        self.beta = nn.Parameter(torch.zeros(h, 1, d))
        self.post_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        # RoboTTT Eq. (3): alpha is a vector in R^d, one value per channel.
        self.gate = nn.Parameter(torch.full((config.hidden_size,), config.gate_init))

    def initial_state(self, batch: int) -> FastState:
        # Called inside module.forward while FSDP parameters are unsharded.
        return FastState(
            *(
                x.unsqueeze(0).expand(batch, -1, -1, -1).clone()
                for x in (self.w1, self.b1, self.w2, self.b2)
            )
        )

    def forward(
        self,
        x: Tensor,
        incoming: FastState | None,
        position: Tensor,
        valid: Tensor,
        reset: Tensor | None = None,
    ) -> tuple[Tensor, FastState, Tensor]:
        b, t, m, width = x.shape
        cfg = self.config
        if m != cfg.tokens_per_anchor:
            raise ValueError(
                f"Expected {cfg.tokens_per_anchor} tokens per anchor; got {m}"
            )
        # Outer linear projections benefit from autocast. Inner algebra is FP32.
        inner_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        def project(layer):
            return (
                layer(x)
                .reshape(b, t, m, cfg.num_heads, -1)
                .transpose(2, 3)
                .to(inner_dtype)
            )

        q, k, v = (project(layer) for layer in (self.q_proj, self.k_proj, self.v_proj))
        lr_logits = self.lr_proj(x).to(inner_dtype)
        with torch.autocast(x.device.type, enabled=False):
            state = self.initial_state(b) if incoming is None else incoming
            state = FastState(*(p.to(inner_dtype) for p in state))
            if reset is not None:
                state = choose_rows(state, self.initial_state(b), reset)
            initial = state
            positions = token_positions(position, valid, m, cfg.position_mode)
            q, k = (rotary(p, positions, cfg.rope_theta) for p in (q, k))
            target = v - k
            key_lr = lr_logits.sigmoid().permute(0, 1, 3, 2).unsqueeze(-2) * (
                cfg.base_lr / k.shape[-1]
            )
            prefix = (
                torch.arange(1, m + 1, device=x.device, dtype=inner_dtype).reciprocal()
                + self.token_bias.to(inner_dtype)
            ).clamp_min(0)
            eta = prefix.reshape(1, 1, 1, m, 1) * key_lr
            scan = (
                compiled_scan_update if cfg.compile_inner and x.is_cuda else scan_update
            )
            readout, state = scan(
                state, q, k, target, eta, self.gamma, self.beta, valid
            )
            readout = readout.transpose(2, 3).reshape(b, t, m, width)
            # Diagnostic is detached and checkpoint-safe; never repairs state.
            update_rms = (
                sum(
                    (a.detach() - z.detach()).square().sum()
                    for a, z in zip(state, initial)
                )
                / sum(a.numel() for a in state)
            ).sqrt()
        output = self.o_proj(self.post_norm(readout.to(x.dtype))) * self.gate.tanh()
        output = output * valid[..., None, None]
        return output, state, update_rms
