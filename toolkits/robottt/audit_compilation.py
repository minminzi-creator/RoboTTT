# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""H100 audit of the optional fused scan against the independently tested eager path."""

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import FastWeightTTT, TTTConfig


def error(a, b):
    difference = (a.double() - b.double()).norm()
    return float(difference / a.double().norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    eager = FastWeightTTT(TTTConfig()).cuda()
    fused = copy.deepcopy(eager)
    fused.config = replace(fused.config, compile_inner=True)
    valid = torch.zeros(2, 16, device="cuda", dtype=torch.bool)
    valid[0, :13] = True
    position = torch.tensor([7, 3], device="cuda")
    records = []
    for autocast in (False, True):
        for detached in (False, True):
            x = torch.randn(2, 16, 57, 1536, device="cuda")
            results = []
            start = time.monotonic()
            for module in (eager, fused):
                module.zero_grad(set_to_none=True)
                value = x.clone().requires_grad_()
                incoming = module.initial_state(2).detach() if detached else None
                snapshot = (
                    None if incoming is None else tuple(p.clone() for p in incoming)
                )
                reset = torch.tensor([True, False], device="cuda") if detached else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                    output, state, _ = checkpoint(
                        module,
                        value,
                        incoming,
                        position,
                        valid,
                        reset,
                        use_reentrant=False,
                    )
                    objective = output.float().square().sum() + sum(
                        p.square().sum() for p in state
                    )
                objective.backward()
                assert output[1].count_nonzero() == 0
                assert all(p.dtype == torch.float32 for p in state)
                if incoming is not None:
                    for a, b, c in zip(state, incoming, snapshot):
                        torch.testing.assert_close(a[1], c[1], rtol=0, atol=0)
                        torch.testing.assert_close(b, c, rtol=0, atol=0)
                results.append(
                    (
                        output.detach(),
                        tuple(p.detach() for p in state),
                        value.grad,
                        {name: p.grad.clone() for name, p in module.named_parameters()},
                    )
                )
            a, b = results
            errors = {
                "output": error(a[0], b[0]),
                "input_gradient": error(a[2], b[2]),
                "state": max(error(u, v) for u, v in zip(a[1], b[1])),
                "parameter_gradients": {
                    name: error(a[3][name], b[3][name]) for name in a[3]
                },
            }
            tolerance = 0.003 if autocast else 0.00002
            assert (
                max(
                    errors["output"],
                    errors["input_gradient"],
                    errors["state"],
                    *errors["parameter_gradients"].values(),
                )
                < tolerance
            ), errors
            torch.cuda.synchronize()
            records.append(
                {
                    "autocast_bfloat16": autocast,
                    "detached_incoming_and_reset": detached,
                    "relative_l2_errors": errors,
                    "tolerance": tolerance,
                    "elapsed_seconds_including_compile": time.monotonic() - start,
                }
            )
            print(json.dumps(records[-1]), flush=True)
    result = {
        "passed": True,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "shape": [2, 16, 57, 1536],
        "checkpoint_recomputation": True,
        "all_padding_row_bitwise_unchanged": True,
        "records": records,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
