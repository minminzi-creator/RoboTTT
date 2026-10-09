# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Independent mathematical and state-contract tests (CPU, no checkpoints)."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
    EpisodeSampler,
    window_indices,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.flow import (
    euler_sample,
    sequence_loss,
    time_bucket,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    FastState,
    FastWeightTTT,
    Memory,
    TTTConfig,
    layer_norm,
    token_positions,
)


def primal(state, q, k, target, eta, gamma, beta):
    """Literal autograd definition, intentionally independent of dual algebra."""
    gradients = []
    for j in range(k.shape[-2]):
        prediction = (
            F.gelu(k[..., j : j + 1, :] @ state.w1 + state.b1, approximate="tanh")
            @ state.w2
            + state.b2
        )
        objective = (
            0.5
            * (layer_norm(prediction, gamma, beta) - target[..., j : j + 1, :])
            .square()
            .sum()
        )
        gradients.append(torch.autograd.grad(objective, state, create_graph=True))
    outputs = []
    for s in range(q.shape[-2]):
        prefix = FastState(
            *(
                w
                - sum(
                    eta[..., s, j, None, None] * gradients[j][i] for j in range(s + 1)
                )
                for i, w in enumerate(state)
            )
        )
        prediction = (
            F.gelu(q[..., s : s + 1, :] @ prefix.w1 + prefix.b1, approximate="tanh")
            @ prefix.w2
            + prefix.b2
        )
        outputs.append(q[..., s : s + 1, :] + layer_norm(prediction, gamma, beta))
    return torch.cat(outputs, -2), prefix


def tiny_layer(mode="flattened"):
    return FastWeightTTT(
        TTTConfig(hidden_size=8, num_heads=2, tokens_per_anchor=3, position_mode=mode)
    ).double()


@pytest.mark.parametrize("mode", ["flattened", "shared_temporal", "slot_only"])
def test_primal_dual_outer_gradients(mode):
    torch.manual_seed(41)
    dual = tiny_layer(mode)
    reference = copy.deepcopy(dual)
    x = torch.randn(2, 3, 3, 8, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    valid = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    positions = torch.tensor([0, 4])
    out, state, _ = dual(x, None, positions, valid)
    with patch("rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt.dual_update", primal):
        outr, stater, _ = reference(xr, None, positions, valid)
    torch.testing.assert_close(out, outr, atol=1e-10, rtol=1e-8)
    for a, b in zip(state, stater):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
    objective = (out * torch.linspace(0.1, 1, out.numel()).reshape_as(out)).sum() + sum(
        p.square().sum() for p in state
    )
    objective_r = (
        outr * torch.linspace(0.1, 1, outr.numel()).reshape_as(outr)
    ).sum() + sum(p.square().sum() for p in stater)
    objective.backward()
    objective_r.backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-9, rtol=1e-7)
    for (name, p), (_, r) in zip(dual.named_parameters(), reference.named_parameters()):
        assert p.grad is not None and p.grad.abs().sum() > 0, name
        torch.testing.assert_close(p.grad, r.grad, atol=1e-8, rtol=2e-7, msg=name)


def test_state_padding_reset_reorder_causality_and_tbptt():
    torch.manual_seed(5)
    layer = tiny_layer()
    x = torch.randn(2, 3, 3, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 3, dtype=torch.bool)
    pos = torch.zeros(2, dtype=torch.long)
    output, state, _ = layer(x, None, pos, valid)
    changed = x.detach().clone()
    changed[:, 2] += 30
    changed_out, _, _ = layer(changed, None, pos, valid)
    torch.testing.assert_close(output[:, :2], changed_out[:, :2], rtol=0, atol=0)
    first, s1, _ = layer(x[:, :1], None, pos, valid[:, :1])
    snapshot = tuple(z.detach().clone() for z in s1)
    rest, s2, _ = layer(x[:, 1:], s1.detach(), pos + 1, valid[:, 1:])
    torch.testing.assert_close(torch.cat((first, rest), 1), output)
    for a, b in zip(s2, state):
        torch.testing.assert_close(a, b)
    grad = torch.autograd.grad(rest.sum(), x)[0]
    assert grad[:, 0].count_nonzero() == 0
    assert grad[:, 1:].abs().sum() > 0
    for a, b in zip(s1, snapshot):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    padded, sp, _ = layer(x, s1, pos + 1, torch.zeros_like(valid))
    assert padded.count_nonzero() == 0
    for a, b in zip(sp, s1):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    mask = torch.tensor([True, False])
    reset_out, sr, _ = layer(x[:, :1], s1, torch.tensor([0, 1]), valid[:, :1], mask)
    torch.testing.assert_close(reset_out[0], first[0])
    _, continued, _ = layer(x[:, :1], s1, pos + 1, valid[:, :1])
    for a, b in zip(sr, continued):
        torch.testing.assert_close(a[1], b[1])
    memory = Memory((state,), pos + 3)
    reordered = memory.reorder(torch.tensor([1, 0]))
    torch.testing.assert_close(reordered.layers[0].w1[0], state.w1[1])
    initial = Memory((layer.initial_state(2),), pos)
    reset_memory = memory.reset(initial, mask)
    assert reset_memory.position.tolist() == [0, 3]
    for a, b, c in zip(reset_memory.layers[0], initial.layers[0], memory.layers[0]):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], c[1], rtol=0, atol=0)


def test_initial_state_does_not_alias_learned_parameters():
    layer = tiny_layer()
    assert layer.gate.shape == (layer.config.hidden_size,)
    state = layer.initial_state(1)
    snapshot = state.w1.detach().clone()
    with torch.no_grad():
        layer.w1.add_(1)
    torch.testing.assert_close(state.w1, snapshot, rtol=0, atol=0)


def test_positions_skip_padding():
    valid = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    assert token_positions(torch.tensor([2]), valid, 3, "flattened").tolist() == [
        [[6, 7, 8], [9, 10, 11], [9, 10, 11]]
    ]
    assert token_positions(torch.tensor([2]), valid, 3, "shared_temporal").tolist() == [
        [[2, 2, 2], [3, 3, 3], [3, 3, 3]]
    ]


def test_inner_matmuls_remain_fp32_under_autocast():
    from torch.utils._python_dispatch import TorchDispatchMode

    products = []

    class ObserveMatmul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.bmm.default:
                products.append((args[0].dtype, args[1].dtype, result.dtype))
            return result

    layer = tiny_layer().float()
    with ObserveMatmul(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, state, _ = layer(
            torch.randn(1, 2, 3, 8),
            None,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
        )
    assert products and set(products) == {(torch.float32, torch.float32, torch.float32)}
    assert all(p.dtype == torch.float32 for p in state)


def test_solver_branches_and_write_indices():
    noise = torch.ones(2, 40, 12)
    incoming = torch.zeros(2)
    actions = []
    for source, index in [("denoise_step", x) for x in [-1, 0, 1, 2, 3]] + [
        ("final_prediction", -50)
    ]:
        calls = []

        def velocity(x, bucket, state):
            assert state is incoming
            calls.append(bucket.clone())
            return x * 0.1 + bucket[:, None, None] / 1000, state + bucket + 1

        result, candidate = euler_sample(velocity, noise, incoming, 4, source, index)
        actions.append(result)
        assert len(calls) == (5 if source == "final_prediction" else 4)
        expected = (
            1000
            if source == "final_prediction"
            else (750 if index == -1 else index * 250) + 1
        )
        torch.testing.assert_close(
            candidate, torch.full((2,), expected, dtype=torch.float32)
        )
        assert incoming.count_nonzero() == 0
    for a in actions[1:]:
        torch.testing.assert_close(a, actions[0], atol=0, rtol=0)
    for i in [-2, 4]:
        with pytest.raises(ValueError):
            euler_sample(velocity, noise, incoming, 4, "denoise_step", i)


def test_fp32_time_and_sequence_normalization():
    tau = torch.tensor([0, 0.25, 0.5, 0.75, 0.999], dtype=torch.float32)
    assert time_bucket(tau).tolist() == [0, 250, 500, 750, 999]
    with pytest.raises(TypeError):
        time_bucket(tau.bfloat16())
    pred = torch.arange(12).float().reshape(2, 3, 2, 1).requires_grad_()
    target = torch.zeros_like(pred)
    dims = torch.ones_like(pred)
    valid = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    supervision = valid.clone()
    denominator = valid.sum(1)
    whole = sequence_loss(pred, target, dims, valid, supervision, denominator)
    parts = sum(
        sequence_loss(
            pred[:, a:b],
            target[:, a:b],
            dims[:, a:b],
            valid[:, a:b],
            supervision[:, a:b],
            denominator,
        )
        for a, b in [(0, 1), (1, 3)]
    )
    torch.testing.assert_close(whole, parts)
    whole.backward()
    assert pred.grad[1, 1:].count_nonzero() == 0


def test_sampler_global_coverage_resume_and_short_episodes():
    all_ranks = [list(EpisodeSampler(104, 42, 7, r, 8)) for r in range(8)]
    assert len(set(sum(all_ranks, []))) == 104
    for r in range(8):
        assert list(EpisodeSampler(104, 42, 7, r, 8, 24)) == all_ranks[r][3:]
    for length in [40, 41, 658, 1055, 1056, 1564]:
        anchors = window_indices(length, 12, 3, 42)
        assert len(anchors) == min(128, (length - 40) // 8 + 1)
        assert np.all(np.diff(anchors) == 8)
        assert anchors[-1] + 39 < length
        assert np.array_equal(anchors, window_indices(length, 12, 3, 42))


def test_unsupervised_context_updates_memory_and_action_mask_only_masks_loss():
    torch.manual_seed(43)
    layer = tiny_layer()
    x = torch.randn(1, 2, 3, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    position = torch.zeros(1, dtype=torch.long)
    output, _, _ = layer(x, None, position, valid)
    output.retain_grad()
    action_mask = torch.ones_like(output)
    action_mask[..., 4:] = 0
    loss = sequence_loss(
        output,
        torch.zeros_like(output),
        action_mask,
        valid,
        torch.tensor([[False, True]]),
        torch.ones(1),
    )
    loss.backward()
    assert output.grad[:, 0].count_nonzero() == 0
    assert output.grad[..., 4:].count_nonzero() == 0
    # No direct supervision, yet its fast update affects the next query.
    assert x.grad[:, 0].abs().sum() > 0
    changed = x.detach().clone()
    changed[:, 0] += 4
    changed_output, _, _ = layer(changed, None, position, valid)
    assert not torch.equal(output[:, 1], changed_output[:, 1])
    padded, _, _ = layer(x.detach(), None, position, torch.tensor([[False, True]]))
    fresh, _, _ = layer(x[:, 1:].detach(), None, position, valid[:, :1])
    # T=1 and T=2 use different projection GEMM shapes / accumulation order.
    torch.testing.assert_close(padded[:, 1:], fresh, rtol=1e-12, atol=1e-16)


def test_checkpoint_is_pure_and_baseline_dit_parity():
    from gr00t.model.modules.dit import AlternateVLDiT

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt_dit import ExplicitDiT

    torch.manual_seed(17)
    source = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        output_dim=4,
        dropout=0,
        final_dropout=False,
        norm_type="ada_norm",
        interleave_self_attention=True,
        cross_attention_dim=12,
    )
    baseline = ExplicitDiT(copy.deepcopy(source), None, False)
    hidden = torch.randn(2, 3, 8)
    encoder = torch.randn(2, 4, 12)
    timestep = torch.tensor([20, 900])
    images = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    attention = torch.ones_like(images)
    valid = torch.ones(1, 2, dtype=torch.bool)
    original = source(
        hidden,
        encoder,
        timestep=timestep,
        image_mask=images,
        backbone_attention_mask=attention,
    )
    output, _, _ = baseline(hidden, encoder, timestep, images, attention, None, valid)
    torch.testing.assert_close(original, output, rtol=0, atol=0)
    ttt = ExplicitDiT(
        copy.deepcopy(source),
        TTTConfig(hidden_size=8, num_heads=2, tokens_per_anchor=3),
        False,
    )
    checked = copy.deepcopy(ttt)
    checked.checkpoint_blocks = True
    result, memory, _ = ttt(hidden, encoder, timestep, images, attention, None, valid)
    check_result, check_memory, _ = checked(
        hidden, encoder, timestep, images, attention, None, valid
    )
    result.square().sum().backward()
    check_result.square().sum().backward()
    torch.testing.assert_close(result, check_result)
    for a, b in zip(memory.layers, check_memory.layers):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
    assert check_memory.position.tolist() == [2]
    for (name, a), (_, b) in zip(ttt.named_parameters(), checked.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, msg=name)


def test_native_action_order_and_explicit_binary_threshold():
    from rlinf.models.embodiment.gr00t.simulation_io import (
        convert_to_robocasa365_action_n1d7,
    )

    binary = np.array([-1, -0.1, 0, 0.49, 0.5, 1], dtype=np.float32)[None, :, None]
    decoded = {
        "gripper_close": binary,
        "control_mode": binary,
        "end_effector_position": np.full((1, 6, 3), 2),
        "end_effector_rotation": np.full((1, 6, 3), 3),
        "base_motion": np.full((1, 6, 4), 4),
    }
    for threshold, expected in (
        (0.0, [-1, -1, 1, 1, 1, 1]),
        (0.5, [-1, -1, -1, -1, 1, 1]),
    ):
        action = convert_to_robocasa365_action_n1d7(
            decoded, chunk_size=6, binary_threshold=threshold
        )[0]
        assert action.shape == (6, 12) and action.dtype == np.float32
        np.testing.assert_array_equal(action[:, 6], expected)
        np.testing.assert_array_equal(action[:, 11], expected)
        assert (action[:, :3] == 2).all() and (action[:, 3:6] == 3).all()
        assert (action[:, 7:11] == 4).all()
