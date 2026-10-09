# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Check native FSDP updates of every TTT parameter group and the registers."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first", type=Path, required=True, help="Earlier full_weights.pt"
    )
    parser.add_argument(
        "--second", type=Path, required=True, help="Later full_weights.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    first, second = [
        torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for path in (args.first, args.second)
    ]
    values = defaultdict(
        lambda: {
            "tensors": 0,
            "elements": 0,
            "changed_elements": 0,
            "delta_squared_sum": 0.0,
        }
    )
    for name in first:
        if name == "registers":
            group = "registers"
        elif ".ttt." in name:
            group = name.split(".ttt.", 1)[1].split(".")[0]
        else:
            continue
        delta = (first[name] - second[name]).double()
        record = values[group]
        record["tensors"] += 1
        record["elements"] += delta.numel()
        record["changed_elements"] += int(delta.count_nonzero())
        record["delta_squared_sum"] += delta.square().sum().item()
    assert values and "registers" in values, "Expected RoboTTT weights with registers"
    for item in values.values():
        item["update_rms"] = math.sqrt(item.pop("delta_squared_sum") / item["elements"])
        assert item["changed_elements"] > 0 and math.isfinite(item["update_rms"])
    report = {
        "passed": True,
        "first": str(args.first),
        "second": str(args.second),
        "groups": dict(values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
