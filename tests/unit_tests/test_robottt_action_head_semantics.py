# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    FlowMatchingActionHeadForRLActionPrediction,
    GR00T_N1_7_ForRLActionPrediction,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    TTTContext,
    TTTState,
)

# ============================================================
# Helpers
# ============================================================

def _make_context(value=10.0):
    """
    A one-layer fake RoboTTT recurrent state.

    We only care about context/state semantics here,
    not the real TTT math, which has already been
    validated in V1.
    """
    return TTTContext(
        num_timesteps=1,
        states=[
            TTTState(
                params=(
                    torch.tensor(
                        float(value),
                        requires_grad=True,
                    ),
                ),
                seq_offset=0,
            )
        ],
    )


def _context_value(context):
    return (
        context
        .states[0]
        .params[0]
        .detach()
        .item()
    )


# ============================================================
# Dummy action head for denoising tests
# ============================================================

class _DummyDenoisingHead:
    """
    Reuse the REAL get_rl_action / forward orchestration,
    but replace GR00T/DiT numerics with tiny deterministic
    functions.

    This isolates the action-head <-> TTTContext logic.
    """

    get_rl_action = (
        FlowMatchingActionHeadForRLActionPrediction
        .get_rl_action
    )

    forward = (
        FlowMatchingActionHeadForRLActionPrediction
        .forward
    )

    def __init__(self):
        self.rl_config = {
            "joint_logprob": False,
        }

        self.config = SimpleNamespace(
            num_steps=4,
        )

        self.action_horizon = 2
        self.model_action_dim = 3
        self.env_action_dim = 3
        self.action_chunk = 1

        self.num_inference_timesteps = 4

        self.context_ids = []
        self.incoming_values = []

    def _process_backbone_output(
        self,
        backbone_output,
    ):
        return backbone_output

    def _encode_state_features(
        self,
        action_input,
        embodiment_id,
    ):
        del embodiment_id

        batch_size = action_input.state.shape[0]

        return torch.zeros(
            batch_size,
            1,
            self.model_action_dim,
        )

    def sample_mean_var_val(
        self,
        vl_embs,
        denoise_steps,
        x_t,
        embodiment_id,
        state_features,
        idx,
        mode="train",
        compute_values=False,
        backbone_output=None,
        ttt_context=None,
    ):
        del (
            vl_embs,
            denoise_steps,
            embodiment_id,
            state_features,
            idx,
            mode,
            compute_values,
            backbone_output,
        )

        assert ttt_context is not None
        assert ttt_context.states is not None

        self.context_ids.append(
            id(ttt_context)
        )

        incoming = (
            ttt_context
            .states[0]
            .params[0]
        )

        self.incoming_values.append(
            incoming.detach().item()
        )

        # IMPORTANT:
        #
        # Mimic TTTTimeMixer semantics:
        # replace context.states[layer] with a NEW TTTState.
        #
        # Do not mutate the old tensor in-place, because branch()
        # intentionally copies the states list while sharing the
        # incoming state objects.
        outgoing = incoming + 1.0

        ttt_context.states[0] = TTTState(
            params=(outgoing,),
            seq_offset=0,
        )

        # Deterministic toy denoising transition.
        return (
            x_t + 0.1,
            torch.zeros_like(x_t),
        )

    def sample_noise(
        self,
        shape,
        device,
        dtype=None,
    ):
        return torch.zeros(
            shape,
            device=device,
            dtype=dtype,
        )

    def get_logprob_norm(
        self,
        sample,
        mu,
        sigma,
    ):
        del mu, sigma
        return torch.zeros_like(sample)

    def get_value(
        self,
        vl_embs,
        state_features,
    ):
        del state_features

        return torch.zeros(
            vl_embs.shape[0],
            device=vl_embs.device,
            dtype=vl_embs.dtype,
        )


def _denoising_inputs(batch_size=2):
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(
                batch_size,
                3,
                3,
            )
        }
    )

    action_input = BatchFeature(
        data={
            "state": torch.randn(
                batch_size,
                3,
            ),
            "embodiment_id": torch.zeros(
                batch_size,
                dtype=torch.long,
            ),
        }
    )

    return backbone, action_input


# ============================================================
# V4.1
# Rollout:
# four denoising iterations == one TTT state advance
# ============================================================

def test_rollout_denoising_advances_main_ttt_context_once():
    """
    One environment timestep has four denoising iterations.

    Desired semantics:

        denoise 0 -> branch(W_t)
        denoise 1 -> branch(W_t)
        denoise 2 -> branch(W_t)
        denoise 3 -> main(W_t)

    Therefore every iteration READS W_t,
    but only the last one commits W_{t+1}.
    """

    head = _DummyDenoisingHead()

    backbone, action_input = (
        _denoising_inputs(
            batch_size=2
        )
    )

    context = _make_context(
        value=10.0
    )

    main_context_id = id(context)

    head.get_rl_action(
        backbone_output=backbone,
        action_input=action_input,
        mode="eval",
        compute_values=False,
        ttt_context=context,
    )

    # Four denoising calls happened.
    assert len(
        head.incoming_values
    ) == 4

    # Every denoising iteration must see the SAME
    # incoming fast weight W_t.
    assert head.incoming_values == [
        10.0,
        10.0,
        10.0,
        10.0,
    ]

    # First three operate on branch contexts.
    assert all(
        context_id != main_context_id
        for context_id
        in head.context_ids[:3]
    )

    # Last iteration commits to the real context.
    assert (
        head.context_ids[-1]
        == main_context_id
    )

    # Main state advances exactly once:
    #
    #     10 -> 11
    #
    # NOT:
    #     10 -> 14
    assert _context_value(
        context
    ) == pytest.approx(11.0)


# ============================================================
# V4.2
# Actor replay/recompute path must obey the same rule
# ============================================================

def test_actor_recompute_denoising_advances_main_context_once():
    """
    RL actor recomputation replays cached denoising chains.

    It must use exactly the same context semantics as rollout:
    intermediate denoising passes cannot repeatedly advance
    RoboTTT memory.
    """

    head = _DummyDenoisingHead()

    head.rl_config = {
        "joint_logprob": True,
    }

    backbone, action_input = (
        _denoising_inputs(
            batch_size=2
        )
    )

    batch_size = 2

    # 4 transitions need 5 chain states.
    chains = torch.randn(
        batch_size,
        5,
        head.action_horizon,
        head.model_action_dim,
    )

    denoise_inds = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )

    context = _make_context(
        value=20.0
    )

    main_context_id = id(context)

    head.forward(
        backbone_output=backbone,
        action_input=action_input,
        chains=chains,
        denoise_inds=denoise_inds,
        compute_values=False,
        ttt_context=context,
    )

    assert head.incoming_values == [
        20.0,
        20.0,
        20.0,
        20.0,
    ]

    assert all(
        context_id != main_context_id
        for context_id
        in head.context_ids[:3]
    )

    assert (
        head.context_ids[-1]
        == main_context_id
    )

    assert _context_value(
        context
    ) == pytest.approx(21.0)


# ============================================================
# Dummy SFT head
# ============================================================

class _DummySFTHead:
    """
    Reuse the REAL sft_forward orchestration.

    All heavy GR00T components are replaced by small functions,
    so this test focuses only on Sequence Action Forcing,
    masks and TTT context flow.
    """

    sft_forward = (
        FlowMatchingActionHeadForRLActionPrediction
        .sft_forward
    )

    _build_ttt_token_mask = (
        FlowMatchingActionHeadForRLActionPrediction
        ._build_ttt_token_mask
    )

    def __init__(
        self,
        ttt_enabled=False,
    ):
        self.training = False

        self.state_dropout_prob = 0.0

        self.config = SimpleNamespace(
            add_pos_embed=False,
        )

        self.num_timestep_buckets = 1000

        self.ttt_enabled = (
            ttt_enabled
        )

        self.sample_time_batch_sizes = []
        self.seen_discrete_times = []

    def set_frozen_modules_to_eval_mode(
        self,
    ):
        pass

    def _process_backbone_output(
        self,
        backbone_output,
    ):
        return backbone_output

    def _encode_state_features(
        self,
        action_input,
        embodiment_id,
    ):
        del embodiment_id

        batch_size = (
            action_input.action.shape[0]
        )

        hidden_dim = (
            action_input.action.shape[-1]
        )

        return torch.zeros(
            batch_size,
            1,
            hidden_dim,
            dtype=action_input.action.dtype,
            device=action_input.action.device,
        )

    def sample_time(
        self,
        batch_size,
        device,
        dtype,
    ):
        """
        Deliberately return a different tau for every
        flattened [B*T] row.

        This lets us verify whether sft_forward asks for
        B*T samples or only B samples.
        """

        self.sample_time_batch_sizes.append(
            batch_size
        )

        values = torch.arange(
            batch_size,
            device=device,
            dtype=torch.float32,
        )

        values = (
            values
            / max(batch_size, 1)
        )

        return values.to(
            dtype=dtype
        )

    def _get_component(
        self,
        name,
    ):
        del name
        return None

    def _expand_register_tokens(
        self,
        like,
    ):
        del like
        return None

    def _run_denoising_model(
        self,
        sa_embs,
        vl_embs,
        timesteps_tensor,
        backbone_output,
        ttt_context,
    ):
        del (
            vl_embs,
            backbone_output,
        )

        self.seen_discrete_times.append(
            timesteps_tensor
            .detach()
            .clone()
        )

        # Mimic one TTT state advance if a context exists.
        if (
            ttt_context is not None
            and ttt_context.states is not None
        ):
            incoming = (
                ttt_context
                .states[0]
                .params[0]
            )

            ttt_context.states[0] = TTTState(
                params=(incoming + 1.0,),
                seq_offset=0,
            )

        # Keep correct token shape.
        return sa_embs

    def _consume_ttt_fast_weight_resets(
        self,
    ):
        return 0


def _sft_inputs(
    batch_size=2,
    num_timesteps=2,
    action_horizon=3,
    hidden_dim=4,
):
    flat_batch = (
        batch_size
        * num_timesteps
    )

    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(
                flat_batch,
                2,
                hidden_dim,
            )
        }
    )

    action = torch.randn(
        flat_batch,
        action_horizon,
        hidden_dim,
    )

    action_input = BatchFeature(
        data={
            "state": torch.randn(
                flat_batch,
                hidden_dim,
            ),

            "action": action,

            "action_mask": torch.ones_like(
                action
            ),

            "embodiment_id": torch.zeros(
                flat_batch,
                dtype=torch.long,
            ),
        }
    )

    return backbone, action_input


# ============================================================
# V4.3
# Sequence Action Forcing:
# tau must be sampled per flattened timestep
# ============================================================

def test_sequence_action_forcing_samples_tau_per_timestep():
    """
    B = 2
    T = 2

    flattened batch has B*T = 4 rows.

    sft_forward MUST call:

        sample_time(batch_size=4)

    rather than:

        sample_time(batch_size=2)

    Otherwise timesteps inside one trajectory would share tau.
    """

    head = _DummySFTHead(
        ttt_enabled=False
    )

    backbone, action_input = (
        _sft_inputs(
            batch_size=2,
            num_timesteps=2,
        )
    )

    output = head.sft_forward(
        backbone_output=backbone,
        action_input=action_input,
        ttt_context=None,
        loss_mask=None,
        num_timesteps=2,
    )

    assert (
        head.sample_time_batch_sizes
        == [4]
    )

    assert len(
        head.seen_discrete_times
    ) == 1

    seen = (
        head.seen_discrete_times[0]
    )

    # Our deterministic sample_time returned:
    #
    #   [0/4, 1/4, 2/4, 3/4]
    #
    # * 1000 buckets
    #
    #   [0, 250, 500, 750]
    expected = torch.tensor(
        [0, 250, 500, 750],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        seen.cpu(),
        expected,
    )

    assert torch.isfinite(
        output["loss"]
    )


# ============================================================
# V4.4
# loss_mask=0 must NOT stop the TTT context update
# ============================================================

def test_loss_mask_zero_still_allows_ttt_state_update():
    """
    Important for context-only demonstration timesteps.

    loss_mask=0 means:

        do not optimize imitation loss for this timestep

    It must NOT mean:

        do not let this timestep enter RoboTTT memory.
    """

    head = _DummySFTHead(
        ttt_enabled=True
    )

    backbone, action_input = (
        _sft_inputs(
            batch_size=2,
            num_timesteps=2,
        )
    )

    context = _make_context(
        value=30.0
    )

    loss_mask = torch.zeros(
        2,
        2,
    )

    output = head.sft_forward(
        backbone_output=backbone,
        action_input=action_input,
        ttt_context=context,
        loss_mask=loss_mask,
        num_timesteps=2,
    )

    # Imitation objective is completely masked.
    assert output[
        "loss"
    ].item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    # But the TTT context was still passed through
    # the denoising model and advanced.
    assert _context_value(
        context
    ) == pytest.approx(
        31.0
    )


# ============================================================
# V4.5
# Token routing:
# register / state / action layout
# ============================================================

@pytest.mark.parametrize(
    "mode, expected",
    [
        (
            "state",
            [
                False,
                False,
                True,
                False,
                False,
                False,
            ],
        ),
        (
            "state_action",
            [
                False,
                False,
                True,
                True,
                True,
                True,
            ],
        ),
        (
            "register_state",
            [
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        ),
        (
            "register",
            [
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        ),
    ],
)
def test_ttt_context_token_mask_routes_expected_tokens(
    mode,
    expected,
):
    """
    Token layout:

        [R0, R1, STATE, A0, A1, A2]

    context_tokens='state':
        only STATE

    context_tokens='state_action':
        STATE + ACTION

    context_tokens='register_state':
        REGISTER + STATE

    context_tokens='register':
        REGISTER + STATE + ACTION
    """

    head = _DummySFTHead()

    head.num_register_tokens = 2
    head.action_horizon = 3
    head.ttt_context_tokens = mode

    mask = (
        head._build_ttt_token_mask()
    )

    expected_tensor = torch.tensor(
        expected,
        dtype=torch.bool,
    )

    torch.testing.assert_close(
        mask,
        expected_tensor,
    )


def test_rollout_context_resets_at_trained_horizon():
    states = [
        TTTState(
            params=(torch.zeros(2, 1),),
            seq_offset=torch.tensor([6, 5]),
        )
    ]

    class _Head:
        ttt_enabled = True
        max_rollout_context_steps = 2
        ttt_token_mask = torch.tensor([True, True, True])

        @staticmethod
        def make_ttt_context(**kwargs):
            return TTTContext(**kwargs)

    policy = SimpleNamespace(
        action_head=_Head(),
        _ttt_rollout_states={0: states},
    )

    context = GR00T_N1_7_ForRLActionPrediction._rollout_ttt_context(
        policy,
        stage_id=0,
        dones=torch.tensor([False, True]),
    )

    torch.testing.assert_close(
        context.reset_mask,
        torch.tensor([True, True]),
    )


def test_robocasa_control_mode_stabilizer_debounces_and_resets():
    policy = SimpleNamespace(
        action_head=SimpleNamespace(
            rl_config={
                "control_mode_stabilizer": {
                    "enabled": True,
                    "confirm_steps": 16,
                    "enter_base_threshold": 0.5,
                    "enter_arm_threshold": -0.5,
                    "initial_mode": -1,
                }
            }
        ),
        obs_converter_type="robocasa365",
        _control_mode_states={},
    )

    first = np.zeros((1, 8, 12), dtype=np.float32)
    first[..., 11] = 0.9
    first_out = GR00T_N1_7_ForRLActionPrediction._stabilize_robocasa_control_mode(
        policy, first, stage_id=0, dones=torch.tensor([False])
    )
    np.testing.assert_array_equal(first_out[..., 11], -1)

    second = first.copy()
    second_out = GR00T_N1_7_ForRLActionPrediction._stabilize_robocasa_control_mode(
        policy, second, stage_id=0, dones=torch.tensor([False])
    )
    np.testing.assert_array_equal(second_out[0, :7, 11], -1)
    assert second_out[0, 7, 11] == 1

    uncertain = first.copy()
    uncertain[..., 11] = np.array([0.2, -0.2] * 4)
    uncertain_out = GR00T_N1_7_ForRLActionPrediction._stabilize_robocasa_control_mode(
        policy, uncertain, stage_id=0, dones=torch.tensor([False])
    )
    np.testing.assert_array_equal(uncertain_out[..., 11], 1)

    reset_out = GR00T_N1_7_ForRLActionPrediction._stabilize_robocasa_control_mode(
        policy, first, stage_id=0, dones=torch.tensor([True])
    )
    np.testing.assert_array_equal(reset_out[..., 11], -1)
