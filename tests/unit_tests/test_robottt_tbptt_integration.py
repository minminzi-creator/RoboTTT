from contextlib import contextmanager, nullcontext

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    segment_bounds,
    slice_batch_major,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    TTTContext,
    TTTState,
)
from rlinf.workers.sft.fsdp_robottt_sft_worker import (
    FSDPRoboTTTSftWorker,
    reduce_tbptt_metrics,
)


# ============================================================
# V3.1
# [B*T] slicing must preserve trajectory ordering
# ============================================================

def test_slice_batch_major_preserves_trajectory_time_order():
    """
    输入布局：

        B = 2
        T = 5

        flattened rows:
            trajectory 0: 0 1 2 3 4
            trajectory 1: 5 6 7 8 9

    取时间窗口 [1, 4) 后应该得到：

            trajectory 0: 1 2 3
            trajectory 1: 6 7 8

    最终重新 flatten：
            [1, 2, 3, 6, 7, 8]

    这样才能保证 TBPTT 不会混淆不同 trajectory。
    """

    batch = {
        "state": torch.arange(
            10,
            dtype=torch.float32,
        ).reshape(10, 1),

        "action": (
            100
            + torch.arange(
                10,
                dtype=torch.float32,
            )
        ).reshape(10, 1),

        "num_timesteps": 5,
    }

    sliced = slice_batch_major(
        batch,
        t_start=1,
        t_end=4,
        num_timesteps=5,
    )

    expected_state = torch.tensor(
        [1, 2, 3, 6, 7, 8],
        dtype=torch.float32,
    ).reshape(-1, 1)

    expected_action = torch.tensor(
        [101, 102, 103, 106, 107, 108],
        dtype=torch.float32,
    ).reshape(-1, 1)

    torch.testing.assert_close(
        sliced["state"],
        expected_state,
    )

    torch.testing.assert_close(
        sliced["action"],
        expected_action,
    )

    assert sliced["num_timesteps"] == 3


# ============================================================
# V3.2
# segment boundaries
# ============================================================

def test_segment_bounds_cover_sequence_exactly_once():
    """
    T = 10
    segment length = 4

    应该得到：

        [0,4)
        [4,8)
        [8,10)

    不能漏 timestep，也不能重复 timestep。
    """

    bounds = segment_bounds(
        num_timesteps=10,
        segment_length=4,
    )

    assert bounds == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]

    covered = []

    for start, end in bounds:
        covered.extend(
            range(start, end)
        )

    assert covered == list(range(10))


# ============================================================
# V3.3
# TBPTT metric/loss weighting
# ============================================================

def test_tbptt_loss_is_weighted_by_segment_length():
    """
    T = 10

    segment:
        [0,4)  length 4  loss = 1
        [4,8)  length 4  loss = 2
        [8,10) length 2  loss = 4

    正确 trajectory loss：

        1 * 4/10
      + 2 * 4/10
      + 4 * 2/10

      = 2.0

    不是简单：
        (1+2+4)/3
    """

    bounds = [
        (0, 4),
        (4, 8),
        (8, 10),
    ]

    segment_losses = [
        1.0,
        2.0,
        4.0,
    ]

    per_segment = [
        {
            "foo": 1.0,
            "ttt_fast_weight_reset": 1.0,
        },
        {
            "foo": 3.0,
            "ttt_fast_weight_reset": 2.0,
        },
        {
            "foo": 5.0,
            "ttt_fast_weight_reset": 3.0,
        },
    ]

    metrics = reduce_tbptt_metrics(
        per_segment=per_segment,
        segment_losses=segment_losses,
        bounds=bounds,
        num_timesteps=10,
    )

    assert metrics["loss"] == pytest.approx(
        2.0
    )

    assert metrics["tbptt_segments"] == 3.0

    # 普通 metric 跨 segment 平均
    assert metrics["foo"] == pytest.approx(
        3.0
    )

    # fast-weight reset 是事件次数，所以应该求和
    assert metrics[
        "ttt_fast_weight_reset"
    ] == pytest.approx(6.0)


# ============================================================
# Fake model used for worker-level test
# ============================================================

class _IdentityGradScaler:
    """
    CPU toy test 不需要 AMP scaler。

    模拟：
        scaler.scale(loss)
    """

    def scale(self, loss):
        return loss


class _FakeSequenceModel(torch.nn.Module):
    """
    极小模型，用于观察 TBPTT worker 是否正确：

        segment 1
          ↓
        TTT state
          ↓
        detach
          ↓
        segment 2

    不测试 GR00T 本身。
    只测试 worker 的 orchestration。
    """

    def __init__(self):
        super().__init__()

        # 模拟一个 outer-loop trainable parameter
        self.theta = torch.nn.Parameter(
            torch.tensor(1.0)
        )

        self.incoming_values = []
        self.incoming_requires_grad = []

        self.outgoing_values = []

        self.window_lengths = []

    def forward(
        self,
        forward_type,
        data,
        ttt_context=None,
    ):
        del forward_type

        window = int(
            data["num_timesteps"]
        )

        self.window_lengths.append(
            window
        )

        # ---------------------------------
        # read incoming recurrent state
        # ---------------------------------

        if ttt_context is None:

            incoming = torch.zeros(
                (),
                dtype=torch.float32,
            )

            self.incoming_values.append(
                None
            )

            self.incoming_requires_grad.append(
                None
            )

        else:

            incoming = (
                ttt_context
                .states[0]
                .params[0]
            )

            self.incoming_values.append(
                incoming.detach().clone()
            )

            self.incoming_requires_grad.append(
                incoming.requires_grad
            )

        # ---------------------------------
        # segment computation
        # ---------------------------------

        x = data["state"].float().mean()

        local_update = (
            self.theta * x
        )

        outgoing = (
            incoming
            + local_update
        )

        self.outgoing_values.append(
            outgoing.detach().clone()
        )

        # Make the outgoing value a fake TTT recurrent state.
        new_state = TTTState(
            params=(outgoing,),
            seq_offset=0,
        )

        new_context = TTTContext(
            num_timesteps=window,
            states=[new_state],
        )

        # Toy segment objective
        loss = outgoing.square()

        return {
            "loss": loss,
            "ttt_context": new_context,

            # Check that SUM metrics are also reduced correctly.
            "ttt_fast_weight_reset": 1.0,
        }


class _DummyRoboTTTWorker:
    """
    Reuse the real FSDPRoboTTTSftWorker implementation
    without constructing Ray/FSDP/GR00T.

    This isolates the TBPTT orchestration logic.
    """

    _tbptt_segment_length = (
        FSDPRoboTTTSftWorker
        ._tbptt_segment_length
    )

    get_train_model_output = (
        FSDPRoboTTTSftWorker
        .get_train_model_output
    )

    def __init__(self):
        self.cfg = OmegaConf.create(
            {
                "actor": {
                    "model": {
                        "tbptt_segment_length": 2
                    }
                }
            }
        )

        self.model = _FakeSequenceModel()

        self.gradient_accumulation = 1

        self.grad_scaler = (
            _IdentityGradScaler()
        )

        # get_train_model_output uses:
        #
        #     with self.amp_context:
        #
        self.amp_context = nullcontext()

    @contextmanager
    def _tbptt_no_sync(self):
        """
        Real FSDP worker suppresses all-reduce for
        intermediate backwards.

        Distributed synchronization itself is not what
        this unit test is testing, so use a no-op context.
        """
        yield


# ============================================================
# V3.4
# Real TBPTT worker orchestration:
#
# value carry + gradient detach
# ============================================================

def test_tbptt_worker_carries_values_but_detaches_graph():
    """
    T = 5
    segment length = 2

    worker 应调用模型：

        segment 0: [0,2)
        segment 1: [2,4)
        segment 2: [4,5)

    即 window:
        2, 2, 1

    在 segment 边界：

        fast-weight value 必须继续

    但是：

        requires_grad 必须变 False

    这正是 truncated BPTT。
    """

    worker = _DummyRoboTTTWorker()

    batch = {
        "state": torch.tensor(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
            ]
        ),

        "num_timesteps": 5,
    }

    last_loss, metrics = (
        worker.get_train_model_output(
            batch
        )
    )

    model = worker.model

    # ---------------------------------
    # segment structure
    # ---------------------------------

    assert model.window_lengths == [
        2,
        2,
        1,
    ]

    # ---------------------------------
    # first segment starts from W0
    # ---------------------------------

    assert (
        model.incoming_values[0]
        is None
    )

    # ---------------------------------
    # value carry:
    #
    # segment 1 input
    # == segment 0 output
    #
    # segment 2 input
    # == segment 1 output
    # ---------------------------------

    torch.testing.assert_close(
        model.incoming_values[1],
        model.outgoing_values[0],
    )

    torch.testing.assert_close(
        model.incoming_values[2],
        model.outgoing_values[1],
    )

    # ---------------------------------
    # graph truncation:
    #
    # worker calls:
    #     ttt_context.detached()
    #
    # before next segment
    # ---------------------------------

    assert (
        model.incoming_requires_grad[1]
        is False
    )

    assert (
        model.incoming_requires_grad[2]
        is False
    )

    # ---------------------------------
    # The worker already backward()'d
    # intermediate segments.
    #
    # Last segment is returned for the
    # outer/base worker backward.
    # ---------------------------------

    last_loss.backward()

    assert (
        model.theta.grad
        is not None
    )

    assert torch.isfinite(
        model.theta.grad
    )

    assert (
        model.theta.grad.abs()
        > 0
    )

    # ---------------------------------
    # metric aggregation
    # ---------------------------------

    assert metrics[
        "tbptt_segments"
    ] == 3.0

    # fake model emits one reset metric
    # per segment
    assert metrics[
        "ttt_fast_weight_reset"
    ] == 3.0
