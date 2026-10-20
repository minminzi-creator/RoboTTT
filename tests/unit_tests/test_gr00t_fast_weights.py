# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Mathematical and state contracts for the native FP32 TTT branch."""

import importlib

import pytest
import torch
import torch.nn.functional as F


def implementation():
    return importlib.import_module("rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt")


def layer():
    ttt = implementation()
    torch.manual_seed(17)
    return ttt.FastWeightTTT(
        ttt.TTTConfig(hidden_size=8, num_heads=2, tokens_per_anchor=3)
    )


def primal_update(state, q, k, target, eta, gamma, beta):
    """Independent autograd gradient descent at the incoming weights."""

    def normalize(value):
        return F.layer_norm(value, (value.shape[-1],), eps=1e-6) * gamma + beta

    gradients = []
    for j in range(k.shape[-2]):
        prediction = (
            F.gelu(k[..., j : j + 1, :] @ state.w1 + state.b1, approximate="tanh")
            @ state.w2
            + state.b2
        )
        objective = (
            0.5 * (normalize(prediction) - target[..., j : j + 1, :]).square().sum()
        )
        gradients.append(torch.autograd.grad(objective, state, create_graph=True))
    outputs = []
    for s in range(q.shape[-2]):
        prefix = type(state)(
            *(
                value
                - sum(
                    eta[..., s, j, None, None] * gradients[j][i] for j in range(s + 1)
                )
                for i, value in enumerate(state)
            )
        )
        query = q[..., s : s + 1, :]
        prediction = (
            F.gelu(query @ prefix.w1 + prefix.b1, approximate="tanh") @ prefix.w2
            + prefix.b2
        )
        outputs.append(query + normalize(prediction))
    return torch.cat(outputs, -2), prefix


def test_dual_update_matches_independent_primal_values_and_outer_gradients():
    ttt = implementation()
    torch.manual_seed(2)
    shapes = [(2, 2, 4, 8), (2, 2, 1, 8), (2, 2, 8, 4), (2, 2, 1, 4)]
    initial = [
        torch.randn(s, dtype=torch.float64).mul_(0.1).requires_grad_() for s in shapes
    ]
    values = (
        initial
        + [
            torch.randn(2, 2, 3, 4, dtype=torch.float64).requires_grad_()
            for _ in range(3)
        ]
        + [
            (torch.rand(2, 2, 3, 3, dtype=torch.float64) * 0.01).requires_grad_(),
            torch.ones(2, 1, 4, dtype=torch.float64, requires_grad=True),
            torch.zeros(2, 1, 4, dtype=torch.float64, requires_grad=True),
        ]
    )
    reference = [v.detach().clone().requires_grad_() for v in values]
    actual, state = ttt.dual_update(ttt.FastState(*values[:4]), *values[4:])
    expected, expected_state = primal_update(
        ttt.FastState(*reference[:4]), *reference[4:]
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    for got, want in zip(state, expected_state):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-9)
    weights = torch.linspace(0.1, 1, actual.numel(), dtype=actual.dtype).reshape_as(
        actual
    )
    objective = (actual * weights).sum() + sum(v.square().sum() for v in state)
    wanted = (expected * weights).sum() + sum(v.square().sum() for v in expected_state)
    grads = torch.autograd.grad(objective, values)
    wanted_grads = torch.autograd.grad(wanted, reference)
    for got, want in zip(grads, wanted_grads):
        torch.testing.assert_close(got, want, atol=1e-8, rtol=2e-7)


def test_future_anchor_cannot_change_earlier_readouts():
    model = layer()
    x = torch.randn(2, 3, 3, 8)
    valid, pos = torch.ones(2, 3, dtype=torch.bool), torch.tensor([0, 4])
    actual, _, _ = model(x, None, pos, valid)
    changed = x.clone()
    changed[:, -1] += 20
    altered, _, _ = model(changed, None, pos, valid)
    torch.testing.assert_close(actual[:, :-1], altered[:, :-1], atol=0, rtol=0)


def test_future_loss_differentiates_through_previous_write_until_detached():
    model = layer()
    x = torch.randn(1, 2, 3, 8, requires_grad=True)
    valid, pos = torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.long)
    _, first, _ = model(x[:, :1], None, pos, valid)
    output, _, _ = model(x[:, 1:], first, pos + 1, valid)
    grads = torch.autograd.grad(
        output.square().sum(), (x, model.w1, model.k_proj.weight), retain_graph=True
    )
    assert grads[0][:, 0].abs().sum() > 0
    assert grads[1].abs().sum() > 0
    assert grads[2].abs().sum() > 0
    detached, _, _ = model(x[:, 1:], first.detach(), pos + 1, valid)
    gradient = torch.autograd.grad(detached.square().sum(), x)[0]
    assert gradient[:, 0].count_nonzero() == 0
    assert gradient[:, 1].abs().sum() > 0


def test_padding_is_equivalent_to_omitting_the_anchor():
    model = layer()
    x = torch.randn(2, 3, 3, 8)
    pos = torch.tensor([2, 5])
    mask = torch.tensor([[True, False, True], [True, False, True]])
    padded, state, _ = model(x, None, pos, mask)
    compact, compact_state, _ = model(
        x[:, [0, 2]], None, pos, torch.ones(2, 2, dtype=torch.bool)
    )
    assert padded[:, 1].count_nonzero() == 0
    torch.testing.assert_close(padded[:, [0, 2]], compact)
    for actual, wanted in zip(state, compact_state):
        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)


def test_forward_is_functional_and_all_padding_preserves_state():
    model = layer()
    initial = model.initial_state(2)
    snapshot = tuple(value.clone() for value in initial)
    x = torch.randn(2, 2, 3, 8)
    valid, pos = torch.zeros(2, 2, dtype=torch.bool), torch.tensor([0, 4])
    output, candidate, rms = model(x, initial, pos, valid)
    assert output.count_nonzero() == 0
    assert rms.item() == 0
    for old, saved, new in zip(initial, snapshot, candidate):
        torch.testing.assert_close(old, saved, atol=0, rtol=0)
        torch.testing.assert_close(new, saved, atol=0, rtol=0)
    active = torch.ones_like(valid)
    first, first_state, _ = model(x, initial, pos, active)
    repeat, repeat_state, _ = model(x, initial, pos, active)
    torch.testing.assert_close(first, repeat, atol=0, rtol=0)
    for old, saved, a, b in zip(initial, snapshot, first_state, repeat_state):
        torch.testing.assert_close(old, saved, atol=0, rtol=0)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_memory_reset_and_reorder_keep_environment_identity():
    ttt = implementation()
    model = layer()
    x = torch.randn(2, 1, 3, 8)
    _, state, _ = model(
        x, None, torch.zeros(2, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool)
    )
    memory = ttt.Memory((state,), torch.tensor([5, 7]))
    initial = ttt.Memory((model.initial_state(2),), torch.zeros(2, dtype=torch.long))
    reset = memory.reset(initial, torch.tensor([True, False]))
    assert reset.position.tolist() == [0, 7]
    for actual, fresh, old in zip(reset.layers[0], initial.layers[0], state):
        torch.testing.assert_close(actual[0], fresh[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], old[1], atol=0, rtol=0)
    reordered = memory.reorder(torch.tensor([1, 0]))
    assert reordered.position.tolist() == [7, 5]
    for actual, old in zip(reordered.layers[0], state):
        torch.testing.assert_close(actual, old.flip(0), atol=0, rtol=0)
    assert all(not p.requires_grad for p in memory.detach().layers[0])


def test_initial_state_is_fp32_independent_storage_with_learned_gradient():
    model = layer()
    with torch.no_grad():
        model.w1.fill_(0.1234567)
    state = model.initial_state(2)
    assert state.w1.dtype == torch.float32
    torch.testing.assert_close(state.w1[0], model.w1, atol=0, rtol=0)
    assert not torch.equal(state.w1, state.w1.bfloat16().float())
    gradient = torch.autograd.grad(state.w1.sum(), model.w1)[0]
    torch.testing.assert_close(gradient, torch.full_like(model.w1, 2))
    saved = state.w1.detach().clone()
    with torch.no_grad():
        model.w1.add_(1)
    torch.testing.assert_close(state.w1, saved, atol=0, rtol=0)


def test_bf16_input_and_autocast_keep_inner_matmuls_and_state_fp32():
    from torch.utils._python_dispatch import TorchDispatchMode

    products = []

    class ObserveMatmul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.bmm.default:
                products.append((args[0].dtype, args[1].dtype, result.dtype))
            return result

    model = layer()
    x = torch.randn(2, 2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    with ObserveMatmul(), torch.autocast("cpu", dtype=torch.bfloat16):
        output, state, rms = model(
            x, None, torch.tensor([0, 4]), torch.ones(2, 2, dtype=torch.bool)
        )
    assert output.dtype == torch.bfloat16
    assert rms.dtype == torch.float32
    assert products and all(dtypes == (torch.float32,) * 3 for dtypes in products)
    assert all(value.dtype == torch.float32 for value in state)
    output.float().square().sum().backward()
    assert model.w1.grad is not None and torch.isfinite(model.w1.grad).all()
    assert model.w1.grad.abs().sum() > 0


def test_quantized_initial_parameters_are_rejected_before_reset():
    model = layer().bfloat16()
    with pytest.raises(TypeError, match="FP32"):
        model.initial_state(1)


def test_quantized_incoming_memory_is_rejected_not_silently_promoted():
    ttt = implementation()
    model = layer()
    incoming = ttt.FastState(*(v.bfloat16() for v in model.initial_state(1)))
    with pytest.raises(TypeError, match="FP32"):
        model(
            torch.randn(1, 1, 3, 8),
            incoming,
            torch.tensor([0]),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_direct_reset_matches_fresh_row_and_retains_neighbor():
    model = layer()
    x = torch.randn(2, 1, 3, 8)
    valid, position = torch.ones(2, 1, dtype=torch.bool), torch.tensor([0, 1])
    _, incoming, _ = model(x, None, torch.zeros(2, dtype=torch.long), valid)
    reset, reset_state, _ = model(
        x, incoming, position, valid, torch.tensor([True, False])
    )
    fresh, fresh_state, _ = model(x, None, torch.zeros(2, dtype=torch.long), valid)
    continued, continued_state, _ = model(x, incoming, position, valid)
    torch.testing.assert_close(reset[0], fresh[0])
    torch.testing.assert_close(reset[1], continued[1])
    for actual, a, b in zip(reset_state, fresh_state, continued_state):
        torch.testing.assert_close(actual[0], a[0])
        torch.testing.assert_close(actual[1], b[1])


def test_token_positions_do_not_advance_for_padding():
    ttt = implementation()
    pos, valid = torch.tensor([2]), torch.tensor([[True, False, True]])
    assert ttt.token_positions(pos, valid, 3, "flattened").tolist() == [
        [[6, 7, 8], [9, 10, 11], [9, 10, 11]]
    ]
    assert ttt.token_positions(pos, valid, 3, "shared_temporal").tolist() == [
        [[2, 2, 2], [3, 3, 3], [3, 3, 3]]
    ]
    assert ttt.token_positions(pos, valid, 3, "slot_only").tolist() == [[[0, 1, 2]] * 3]
