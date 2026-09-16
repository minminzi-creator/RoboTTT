import torch

from gr00t.model.modules.dit import AlternateVLDiT

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import TTTContext
from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt_dit import (
    TTTAlternateVLDiT,
)


# ============================================================
# Tiny GR00T configuration
# ============================================================

def _model_kwargs(num_layers=4):
    """
    使用一个极小的 GR00T DiT。

    真实 GR00T N1.7:
        hidden dim = 1536
        layers 很多

    这里:
        hidden dim = 2 heads * 4 dim = 8

    数学/连接关系完全相同，但 CPU 几秒即可测试。
    """
    return dict(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=num_layers,

        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu-approximate",

        upcast_attention=False,

        norm_type="ada_norm",
        norm_elementwise_affine=False,
        norm_eps=1e-5,

        max_num_positional_embeddings=32,
        compute_dtype=torch.float32,

        final_dropout=False,
        positional_embeddings=None,

        # AlternateVLDiT 要求这个开启
        interleave_self_attention=True,

        cross_attention_dim=8,

        attend_text_every_n_blocks=2,
    )


# ============================================================
# Synthetic inputs
# ============================================================

def _inputs(batch_size=2, num_timesteps=2):
    """
    GR00T/RoboTTT 的时间序列布局：

        [B, T, tokens, D]
                 ↓ flatten
        [B*T, tokens, D]

    这里：
        B = 2
        T = 2
        B*T = 4
    """
    torch.manual_seed(123)

    flat_batch = batch_size * num_timesteps

    # DiT state/action/register-like tokens
    hidden = torch.randn(
        flat_batch,
        5,
        8,
    )

    # Vision-language backbone features
    encoder_hidden = torch.randn(
        flat_batch,
        6,
        8,
    )

    timestep = torch.randint(
        low=0,
        high=100,
        size=(flat_batch,),
    )

    # 前 3 个认为是 image token
    # 后 3 个认为是 text token
    #
    # 保证 image/text 两类都非空，避免 mask edge case
    image_mask = torch.tensor(
        [[
            True, True, True,
            False, False, False,
        ]]
    ).repeat(flat_batch, 1)

    backbone_attention_mask = torch.ones(
        flat_batch,
        6,
        dtype=torch.bool,
    )

    return (
        hidden,
        encoder_hidden,
        timestep,
        image_mask,
        backbone_attention_mask,
    )


def _forward(
    model,
    hidden,
    encoder_hidden,
    timestep,
    image_mask,
    backbone_attention_mask,
    ttt_context=None,
):
    kwargs = dict(
        hidden_states=hidden,
        encoder_hidden_states=encoder_hidden,
        timestep=timestep,
        image_mask=image_mask,
        backbone_attention_mask=backbone_attention_mask,
    )

    if isinstance(model, TTTAlternateVLDiT):
        kwargs["ttt_context"] = ttt_context

    return model(**kwargs)


# ============================================================
# Build two matched models
# ============================================================

def _make_matching_models(num_layers=4):
    """
    构造：

        A: 原始 GR00T AlternateVLDiT
        B: RoboTTT TTTAlternateVLDiT

    然后把 A 的所有 GR00T 参数复制给 B。

    因此二者唯一的结构差异应该是：

        B 多出来 TTT parameters
    """

    torch.manual_seed(2026)

    base = AlternateVLDiT(
        **_model_kwargs(num_layers=num_layers)
    )

    ttt = TTTAlternateVLDiT(
        **_model_kwargs(num_layers=num_layers),
        ttt_config=dict(
            fast_model="mlp",

            # 先设成 0
            # 方便做 no-op equivalence
            gate_init=0.0,

            base_lr=0.1,
            max_mini_batch_size=16,
        ),
    )

    base.eval()
    ttt.eval()

    result = ttt.load_state_dict(
        base.state_dict(),
        strict=False,
    )

    # 原 GR00T 不应该有 TTT 不认识的东西
    assert result.unexpected_keys == []

    # B 缺失的只能是 RoboTTT 新增参数
    assert len(result.missing_keys) > 0

    for key in result.missing_keys:
        assert (
            ".ttt_layer." in key
            or key.endswith(".ttt_gate")
        ), f"Unexpected missing key: {key}"

    return base, ttt


# ============================================================
# V2.1
# Original GR00T equivalence when TTT contribution = 0
# ============================================================

def test_zero_ttt_reproduces_original_groot_dit():
    """
    核心 invariant：

    原始：
        Attention -> FFN

    RoboTTT：
        Attention -> 0 * TTT -> FFN

    必须输出一致。

    如果不一致，说明我们在拆 BasicTransformerBlock.forward
    时已经改变了 upstream GR00T 行为。
    """

    base, ttt = _make_matching_models()

    data = _inputs(
        batch_size=2,
        num_timesteps=2,
    )

    with torch.no_grad():
        base_output = _forward(
            base,
            *data,
        )

        context = TTTContext(
            num_timesteps=2,
        )

        ttt_output = _forward(
            ttt,
            *data,
            ttt_context=context,
        )

    torch.testing.assert_close(
        ttt_output,
        base_output,
        rtol=2e-5,
        atol=2e-6,
    )


# ============================================================
# V2.2
# Active TTT must actually affect GR00T
# ============================================================

def test_active_ttt_changes_dit_output():
    """
    TTT 不能只是“装上去了但没被调用”。

    人工激活：
        o_proj != 0
        gate = 0.5

    那么 RoboTTT output 必须和原始 GR00T 不同。
    """

    base, ttt = _make_matching_models()

    torch.manual_seed(7)

    with torch.no_grad():
        for block in ttt.transformer_blocks:

            # 默认 o_proj=0，
            # 所以人工打开 TTT readout
            torch.nn.init.normal_(
                block.ttt_layer.o_proj.weight,
                std=0.1,
            )

            # tanh(0.5) ≠ 0
            block.ttt_gate.fill_(0.5)

    data = _inputs(
        batch_size=2,
        num_timesteps=2,
    )

    with torch.no_grad():

        base_output = _forward(
            base,
            *data,
        )

        context = TTTContext(
            num_timesteps=2,
        )

        ttt_output = _forward(
            ttt,
            *data,
            ttt_context=context,
        )

    assert not torch.allclose(
        base_output,
        ttt_output,
        rtol=1e-5,
        atol=1e-6,
    )


# ============================================================
# V2.3
# Explicit block ordering:
#
# Attention -> TTT -> FFN
# ============================================================

def test_block_execution_order_is_attention_ttt_ffn():
    """
    直接用 forward hooks 记录真正的执行顺序。

    这个测试不是根据源码推断，而是观察 runtime。
    """

    _, ttt = _make_matching_models(
        num_layers=1,
    )

    block = ttt.transformer_blocks[0]

    order = []

    handles = [
        block.attn1.register_forward_pre_hook(
            lambda module, inputs: order.append("attention")
        ),

        block.ttt_layer.register_forward_pre_hook(
            lambda module, inputs: order.append("ttt")
        ),

        block.ff.register_forward_pre_hook(
            lambda module, inputs: order.append("ffn")
        ),
    ]

    data = _inputs(
        batch_size=1,
        num_timesteps=2,
    )

    context = TTTContext(
        num_timesteps=2,
    )

    try:
        with torch.no_grad():
            _forward(
                ttt,
                *data,
                ttt_context=context,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert order == [
        "attention",
        "ttt",
        "ffn",
    ], f"Unexpected execution order: {order}"


# ============================================================
# V2.4
# GR00T -> TTTContext state write-back
# ============================================================

def test_dit_forward_updates_every_ttt_layer_state():
    """
    每个 transformer block 都有自己的 TTT fast state。

    Forward 完以后：

        context.states[0]
        context.states[1]
        ...
        context.states[L-1]

    都应该被更新。
    """

    _, ttt = _make_matching_models(
        num_layers=4,
    )

    batch_size = 2
    num_timesteps = 2
    tokens_per_step = 5

    data = _inputs(
        batch_size=batch_size,
        num_timesteps=num_timesteps,
    )

    context = TTTContext(
        num_timesteps=num_timesteps,
    )

    with torch.no_grad():
        _forward(
            ttt,
            *data,
            ttt_context=context,
        )

    assert context.states is not None

    assert len(context.states) == 4

    for layer_idx, state in enumerate(
        context.states
    ):
        assert state is not None, (
            f"TTT layer {layer_idx} "
            "did not write state"
        )

        # 每层 TTT 看到了：
        #
        # T * tokens_per_step
        #
        # = 2 * 5
        # = 10
        expected_offset = (
            num_timesteps
            * tokens_per_step
        )

        torch.testing.assert_close(
            state.seq_offset,
            torch.full_like(
                state.seq_offset,
                expected_offset,
            ),
        )

        for fast_weight in state.params:
            assert torch.isfinite(
                fast_weight
            ).all()


# ============================================================
# V2.5
# Outer gradient must enter TTT
# ============================================================

def test_outer_gradient_reaches_ttt_parameters():
    """
    验证真正的训练梯度路径：

        loss
          ↓
        DiT output
          ↓
        FFN
          ↓
        TTT branch
          ↓
        ttt_gate
          ↓
        W0

    如果这里不通，TTT 虽然 forward 正常，
    但根本学不到东西。
    """

    _, ttt = _make_matching_models(
        num_layers=4,
    )

    ttt.train()

    torch.manual_seed(11)

    with torch.no_grad():
        for block in ttt.transformer_blocks:

            torch.nn.init.normal_(
                block.ttt_layer.o_proj.weight,
                std=0.1,
            )

            block.ttt_gate.fill_(0.5)

    data = list(
        _inputs(
            batch_size=2,
            num_timesteps=2,
        )
    )

    hidden = (
        data[0]
        .clone()
        .requires_grad_(True)
    )

    data[0] = hidden

    context = TTTContext(
        num_timesteps=2,
    )

    output = _forward(
        ttt,
        *data,
        ttt_context=context,
    )

    # toy outer objective
    loss = output.square().mean()

    loss.backward()

    # ----------------------------
    # input gradient
    # ----------------------------

    assert hidden.grad is not None

    assert torch.isfinite(
        hidden.grad
    ).all()

    # ----------------------------
    # TTT parameter gradients
    # ----------------------------

    for layer_idx, block in enumerate(
        ttt.transformer_blocks
    ):

        # gate 必须能学习
        assert (
            block.ttt_gate.grad is not None
        ), (
            f"layer {layer_idx}: "
            "ttt_gate has no gradient"
        )

        assert torch.isfinite(
            block.ttt_gate.grad
        ).all()

        # W0 必须能够 meta-learn
        assert (
            block.ttt_layer.w1_init.grad
            is not None
        ), (
            f"layer {layer_idx}: "
            "w1_init has no gradient"
        )

        assert torch.isfinite(
            block.ttt_layer.w1_init.grad
        ).all()
