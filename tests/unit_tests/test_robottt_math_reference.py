import pytest
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    TTTConfig,
    TTTLinear,
    TTTMLP,
    build_ttt_layer,
)


def _fast_forward(layer, x, params):
    """Explicit fast-model forward.

    x: [1, head_dim]
    params are one batch/head's fast weights.
    """
    if isinstance(layer, TTTLinear):
        w1, b1 = params
        return x @ w1 + b1

    assert isinstance(layer, TTTMLP)
    w1, b1, w2, b2 = params

    z1 = x @ w1 + b1
    x2 = F.gelu(z1, approximate="tanh")
    return x2 @ w2 + b2


def _explicit_autograd_reference(
    layer,
    params,
    xq,
    xk,
    xv,
    eta,
    ln_weight,
    ln_bias,
):
    """Slow reference implementation of one TTT mini-batch.

    For every key token j:
      1. evaluate the inner reconstruction loss using the fast weights
         entering the mini-batch;
      2. use torch.autograd.grad to obtain the exact gradient.

    For every query token i:
      3. construct the prefix-updated fast weights using keys j <= i;
      4. evaluate query i using those updated weights.

    This intentionally does NOT reuse RoboTTT's hand-derived gradient formulas.
    """
    batch_size, num_heads, mini_batch_size, head_dim = xq.shape

    ref_output = torch.empty_like(xq)
    ref_final_params = [torch.empty_like(p) for p in params]

    for b in range(batch_size):
        for h in range(num_heads):
            # Fast weights at the beginning of this mini-batch.
            base_params = tuple(
                p[b, h].detach().clone().requires_grad_(True)
                for p in params
            )

            gamma = ln_weight[0, h, 0]
            beta = ln_bias[0, h, 0]

            # grad_j = d L_j(W_start) / d W_start
            token_grads = []

            for j in range(mini_batch_size):
                key = xk[b, h, j].unsqueeze(0)
                target = xv[b, h, j].unsqueeze(0)

                prediction = _fast_forward(
                    layer,
                    key,
                    base_params,
                )

                prediction = F.layer_norm(
                    prediction,
                    (head_dim,),
                    weight=gamma,
                    bias=beta,
                    eps=1e-6,
                )

                # Exactly 0.5 || LN(f_W(K_j)) - V_j ||^2
                loss_j = 0.5 * (prediction - target).pow(2).sum()

                grads_j = torch.autograd.grad(
                    loss_j,
                    base_params,
                    retain_graph=False,
                    create_graph=False,
                )

                token_grads.append(
                    tuple(g.detach() for g in grads_j)
                )

            # Read every query from its prefix-updated fast weights.
            final_updated = None

            for i in range(mini_batch_size):
                updated = [
                    p.detach().clone()
                    for p in base_params
                ]

                # update-then-apply:
                # current key j=i is included.
                for j in range(i + 1):
                    lr_ij = eta[b, h, i, j]

                    for k in range(len(updated)):
                        updated[k] = (
                            updated[k]
                            - lr_ij * token_grads[j][k]
                        )

                query = xq[b, h, i].unsqueeze(0)

                query_prediction = _fast_forward(
                    layer,
                    query,
                    tuple(updated),
                )

                query_prediction = F.layer_norm(
                    query_prediction,
                    (head_dim,),
                    weight=gamma,
                    bias=beta,
                    eps=1e-6,
                )

                # RoboTTT inner readout has the Q residual.
                ref_output[b, h, i] = (
                    query + query_prediction
                ).squeeze(0)

                if i == mini_batch_size - 1:
                    final_updated = updated

            assert final_updated is not None

            for k, value in enumerate(final_updated):
                ref_final_params[k][b, h].copy_(
                    value.reshape_as(ref_final_params[k][b, h])
                )

    return tuple(ref_final_params), ref_output


def _make_case(fast_model):
    torch.manual_seed(2026)

    config = TTTConfig(
        hidden_size=8,
        num_heads=2,
        fast_model=fast_model,
        fast_mlp_ratio=2,
        base_lr=0.1,
        use_rope=False,
        max_mini_batch_size=8,
    )

    layer = build_ttt_layer(config).double()

    # Avoid a trivial LayerNorm affine.
    with torch.no_grad():
        layer.ttt_norm_weight.uniform_(0.8, 1.2)
        layer.ttt_norm_bias.uniform_(-0.1, 0.1)

    B = 2
    H = config.num_heads
    M = 3
    D = config.head_dim

    params = tuple(
        p.detach().clone()
        for p in layer._initial_params(B)
    )

    xq = torch.randn(B, H, M, D, dtype=torch.float64)
    xk = torch.randn(B, H, M, D, dtype=torch.float64)
    xv = torch.randn(B, H, M, D, dtype=torch.float64)

    # eta itself need not be triangular. The implementation applies the
    # causal prefix mask internally.
    eta = (
        0.02
        * torch.rand(
            B,
            H,
            M,
            M,
            dtype=torch.float64,
        )
    )

    ln_weight = (
        layer.ttt_norm_weight
        .reshape(1, H, 1, D)
        .double()
    )

    ln_bias = (
        layer.ttt_norm_bias
        .reshape(1, H, 1, D)
        .double()
    )

    return (
        layer,
        params,
        xq,
        xk,
        xv,
        eta,
        ln_weight,
        ln_bias,
    )


@pytest.mark.parametrize(
    "fast_model",
    ["linear", "mlp"],
)
def test_dual_form_matches_explicit_autograd(fast_model):
    (
        layer,
        params,
        xq,
        xk,
        xv,
        eta,
        ln_weight,
        ln_bias,
    ) = _make_case(fast_model)

    # RoboTTT optimized / hand-derived implementation.
    actual_params, actual_output = layer._mini_batch_step(
        params,
        xq,
        xk,
        xv,
        eta,
        ln_weight,
        ln_bias,
    )

    # Independent, deliberately slow PyTorch-autograd implementation.
    reference_params, reference_output = (
        _explicit_autograd_reference(
            layer,
            params,
            xq,
            xk,
            xv,
            eta,
            ln_weight,
            ln_bias,
        )
    )

    # MLP includes a hand-written derivative of tanh-GELU, so tolerate
    # normal floating-point accumulation differences.
    if fast_model == "linear":
        rtol, atol = 1e-8, 1e-9
    else:
        rtol, atol = 2e-6, 2e-7

    torch.testing.assert_close(
        actual_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )

    for actual, reference in zip(
        actual_params,
        reference_params,
    ):
        torch.testing.assert_close(
            actual,
            reference,
            rtol=rtol,
            atol=atol,
        )
