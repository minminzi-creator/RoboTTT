# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Compare uninterrupted and resumed native RLinf checkpoints without GPU memory."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint import FileSystemReader


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b


def read_entry(root, storage):
    with (root / storage.relative_path).open("rb") as handle:
        handle.seek(storage.offset)
        payload = handle.read(storage.length)
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def compare_dcp(first, second):
    """Read one tensor chunk at a time, including every Adam moment and scalar."""
    left = FileSystemReader(first).read_metadata()
    right = FileSystemReader(second).read_metadata()
    assert left.storage_data.keys() == right.storage_data.keys(), "DCP fields differ"
    counts, failures = Counter(), []
    for key, source in left.storage_data.items():
        a = read_entry(first, source)
        b = read_entry(second, right.storage_data[key])
        category = key.fqn.split(".")[1]
        counts[category] += 1
        if not equal(a, b):
            failures.append(str(key))
    return {"compared_chunks": dict(counts), "unequal_chunks": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uninterrupted", type=Path, required=True, help="Checkpoint actor directory"
    )
    parser.add_argument(
        "--resumed", type=Path, required=True, help="Checkpoint actor directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {
        "uninterrupted": str(args.uninterrupted),
        "resumed": str(args.resumed),
        "dcp": compare_dcp(
            args.uninterrupted / "dcp_checkpoint", args.resumed / "dcp_checkpoint"
        ),
    }
    result["sidecars"] = {}
    for name in ("rng.pt", "data.pt", "sequence_progress.pt"):
        states = [
            torch.load(root / name, map_location="cpu", weights_only=False)
            for root in (args.uninterrupted, args.resumed)
        ]
        result["sidecars"][name] = equal(*states)
    weights = [
        torch.load(
            root / "model_state_dict/full_weights.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        for root in (args.uninterrupted, args.resumed)
    ]
    result["exported_model_equal"] = equal(*weights)
    result["passed"] = (
        not result["dcp"]["unequal_chunks"]
        and all(result["sidecars"].values())
        and result["exported_model_equal"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise AssertionError("Native checkpoint continuation is not bitwise identical")


if __name__ == "__main__":
    main()
