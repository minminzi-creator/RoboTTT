# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""State ownership exercised through a small causal fast-weight policy."""

import copy
import importlib
import importlib.util

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    FastState,
    FastWeightTTT,
    Memory,
    TTTConfig,
)


def rollout_type():
    name = "rlinf.models.embodiment.gr00t.gr00t_n1d7.rollout"
    assert importlib.util.find_spec(name) is not None, "Native rollout owner is missing"
    return importlib.import_module(name).NativeTTTRollout


class CausalPolicy(nn.Module):
    """Real TTT updates with observations as tokens and learned action readout."""

    def __init__(self):
        super().__init__()
        self.fast = FastWeightTTT(
            TTTConfig(
                hidden_size=4,
                num_heads=1,
                mlp_ratio=2,
                tokens_per_anchor=3,
                gate_init=1.0,
                o_proj_std=0.2,
            )
        )
        self.action_projection = nn.Linear(4, 2)

    def predict_with_memory(
        self, data, memory=None, policy_noise=None, reset_mask=None
    ):
        tokens = data if policy_noise is None else data + policy_noise
        size = tokens.shape[0]
        position = (
            torch.zeros(size, dtype=torch.long) if memory is None else memory.position
        )
        if reset_mask is not None:
            position = torch.where(reset_mask, 0, position)
        output, candidate, update = self.fast(
            tokens,
            None if memory is None else memory.layers[0],
            position,
            torch.ones(size, 1, dtype=torch.bool),
            reset=reset_mask,
        )
        return {
            "actions": self.action_projection(tokens[:, 0] + output[:, 0]),
            "memory": Memory((candidate,), position + 1),
            "ttt_update_rms": update,
        }


@pytest.fixture
def policy():
    torch.manual_seed(53)
    return CausalPolicy().eval()


def observations():
    return torch.tensor(
        [
            [[[0.1, 0.4, -0.3, 0.2], [0.3, 0.8, -0.6, 0.1], [0.5, -0.2, 0.7, 0.9]]],
            [[[0.2, -0.4, 0.5, 0.8], [0.7, 0.1, 0.4, -0.2], [0.4, 0.6, -0.3, 0.2]]],
        ]
    )


def assert_memory_equal(actual, expected):
    assert torch.equal(actual.position, expected.position)
    assert len(actual.layers) == len(expected.layers)
    for left, right in zip(actual.layers, expected.layers):
        for value, reference in zip(left, right):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_carry_tracks_stable_ids_across_row_reordering(policy):
    owner = rollout_type()(policy)
    reference = rollout_type()(copy.deepcopy(policy))
    data = observations()
    first = owner.predict(data, ["left", "right"])
    reference.predict(data, ["left", "right"])
    second = owner.predict(data.flip(0) * 2, ["right", "left"])
    expected = reference.predict(data * 2, ["left", "right"])
    torch.testing.assert_close(second["actions"], expected["actions"].flip(0))
    assert_memory_equal(
        second["memory"], expected["memory"].reorder(torch.tensor([1, 0]))
    )
    assert first["memory"].position.tolist() == [1, 1]
    assert second["memory"].position.tolist() == [2, 2]
    assert not torch.equal(first["memory"].layers[0].w1, second["memory"].layers[0].w1)
    assert all(
        not value.requires_grad for layer in second["memory"].layers for value in layer
    )
    assert not second["actions"].requires_grad


@pytest.mark.parametrize("via_mask", [False, True])
def test_partial_reset_restores_only_the_requested_environment(policy, via_mask):
    owner = rollout_type()(policy)
    ongoing = rollout_type()(copy.deepcopy(policy))
    fresh = rollout_type()(copy.deepcopy(policy))
    data = observations()
    owner.predict(data, [10, 20])
    ongoing.predict(data, [10, 20])
    if not via_mask:
        owner.reset([20])
    actual = owner.predict(
        data.flip(0) * 2,
        [20, 10],
        reset_mask=torch.tensor([True, False]) if via_mask else None,
    )
    continued = ongoing.predict(data * 2, [10, 20])
    restarted = fresh.predict(data.flip(0) * 2, [20, 10])
    torch.testing.assert_close(actual["actions"][0], restarted["actions"][0])
    torch.testing.assert_close(actual["actions"][1], continued["actions"][0])
    assert actual["memory"].position.tolist() == [1, 2]
    assert_memory_equal(
        actual["memory"].reorder(torch.tensor([0])),
        restarted["memory"].reorder(torch.tensor([0])),
    )


def test_reset_mode_repeats_fresh_chunk_adaptation(policy):
    owner = rollout_type()(policy, mode="reset")
    carry = rollout_type()(copy.deepcopy(policy))
    data = observations()
    first = owner.predict(data, [10, 20])
    second = owner.predict(data, [10, 20])
    carry.predict(data, [10, 20])
    continuing = carry.predict(data, [10, 20])
    torch.testing.assert_close(first["actions"], second["actions"], rtol=0, atol=0)
    assert_memory_equal(first["memory"], second["memory"])
    assert second["memory"].position.tolist() == [1, 1]
    assert second["ttt_update_rms"] > 0
    assert not torch.equal(continuing["actions"], second["actions"])


@pytest.mark.parametrize("changed_ids", [[10, 30], [10], [10, 20, 30]])
def test_active_environment_changes_require_full_reset(policy, changed_ids):
    owner = rollout_type()(policy)
    owner.predict(observations(), [10, 20])
    with pytest.raises(ValueError, match="reset"):
        owner.predict(torch.ones(len(changed_ids), 1, 3, 4), changed_ids)
    owner.reset()
    result = owner.predict(torch.ones(len(changed_ids), 1, 3, 4), changed_ids)
    assert result["memory"].position.tolist() == [1] * len(changed_ids)


def test_returned_memory_cannot_modify_future_state_or_model_weights(policy):
    owner = rollout_type()(policy)
    reference = rollout_type()(copy.deepcopy(policy))
    before = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    first = owner.predict(observations(), [10, 20])
    reference.predict(observations(), [10, 20])
    first["memory"].position.fill_(700)
    for layer in first["memory"].layers:
        for value in layer:
            value.fill_(float("nan"))
    actual = owner.predict(observations(), [10, 20])
    expected = reference.predict(observations(), [10, 20])
    assert_memory_equal(actual["memory"], expected["memory"])
    torch.testing.assert_close(actual["actions"], expected["actions"], rtol=0, atol=0)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


class FailingPolicy(CausalPolicy):
    """Policy boundary faults must leave the rollout's prior commit intact."""

    failure = None

    def predict_with_memory(self, data, memory=None, **kwargs):
        if self.failure == "exception":
            memory.layers[0].w1.add_(100)
            memory.position.add_(100)
            raise RuntimeError("Prediction failed after touching its input")
        result = super().predict_with_memory(data, memory=memory, **kwargs)
        if self.failure == "actions":
            result["actions"][0, 0, 0] = float("nan")
        elif self.failure == "state":
            result["memory"].layers[0].w1[0, 0, 0, 0] = float("inf")
        elif self.failure == "position":
            result["memory"].position.add_(4)
        elif self.failure == "precision":
            result["memory"] = Memory(
                tuple(
                    FastState(*(x.to(torch.bfloat16) for x in layer))
                    for layer in result["memory"].layers
                ),
                result["memory"].position,
            )
        elif self.failure == "missing_memory":
            result["memory"] = None
        elif self.failure == "action_rows":
            result["actions"] = result["actions"][:1]
        return result


@pytest.mark.parametrize(
    "failure",
    [
        "exception",
        "actions",
        "state",
        "position",
        "precision",
        "missing_memory",
        "action_rows",
    ],
)
def test_failed_prediction_preserves_state_order_and_pending_resets(failure):
    torch.manual_seed(53)
    policy = FailingPolicy().eval()
    owner = rollout_type()(policy)
    reference = rollout_type()(copy.deepcopy(policy))
    owner.predict(observations(), [10, 20])
    reference.predict(observations(), [10, 20])
    owner.reset([20])
    reference.reset([20])
    policy.failure = failure
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        owner.predict(observations().flip(0), [20, 10])
    policy.failure = None
    actual = owner.predict(observations(), [10, 20])
    expected = reference.predict(observations(), [10, 20])
    assert_memory_equal(actual["memory"], expected["memory"])
    torch.testing.assert_close(actual["actions"], expected["actions"], rtol=0, atol=0)
    assert actual["memory"].position.tolist() == [2, 1]


def test_fixed_policy_noise_is_used_without_consuming_global_rng(policy):
    owner = rollout_type()(policy)
    noise = torch.full_like(observations(), 0.25)
    rng = torch.get_rng_state().clone()
    actual = owner.predict(observations(), [10, 20], policy_noise=noise)
    assert torch.equal(torch.get_rng_state(), rng)
    expected = rollout_type()(copy.deepcopy(policy)).predict(
        observations() + noise, [10, 20]
    )
    torch.testing.assert_close(actual["actions"], expected["actions"], rtol=0, atol=0)
    assert_memory_equal(actual["memory"], expected["memory"])


@pytest.mark.parametrize(
    "env_ids, reset_mask",
    [
        ([10, 10], None),
        ([], None),
        ([10, 20], torch.tensor([True])),
        ([10, 20], torch.tensor([1, 0])),
    ],
)
def test_invalid_identity_or_reset_input_does_not_start_rollout(
    policy, env_ids, reset_mask
):
    owner = rollout_type()(policy)
    with pytest.raises((ValueError, TypeError)):
        owner.predict(observations(), env_ids, reset_mask=reset_mask)
    result = owner.predict(observations(), [10, 20])
    assert result["memory"].position.tolist() == [1, 1]


def test_unknown_reset_environment_is_rejected(policy):
    owner = rollout_type()(policy)
    owner.predict(observations(), [10, 20])
    with pytest.raises(ValueError, match="environment"):
        owner.reset([30])
    with pytest.raises(ValueError, match="mode"):
        rollout_type()(policy, mode="baseline")
