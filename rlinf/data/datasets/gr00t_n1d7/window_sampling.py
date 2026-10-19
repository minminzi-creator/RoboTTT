# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Frame-uniform legal windows and a checkpointable global sample stream."""

import random
from bisect import bisect_right
from itertools import accumulate


class WindowIndex:
    """Map each legal H-frame target to one (episode row, start frame)."""

    def __init__(self, lengths: list[int], horizon: int = 40):
        if horizon <= 0 or not lengths or any(n < horizon for n in lengths):
            raise ValueError(
                "Expected nonempty episodes with length >= positive horizon"
            )
        self.ends = list(accumulate(n - horizon + 1 for n in lengths))

    def __len__(self) -> int:
        return self.ends[-1]

    def locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = bisect_right(self.ends, index)
        return row, index - (self.ends[row - 1] if row else 0)


class WindowSampler:
    """Shard an infinite shuffled-window stream without dropping epoch tails."""

    def __init__(self, size: int, seed: int, rank: int, world_size: int):
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Expected positive size/world size and rank in range")
        self.size, self.seed, self.rank, self.world_size = size, seed, rank, world_size

    def __iter__(self):
        return _WindowIterator(self)

    def state_dict(self) -> dict:
        return {
            "size": self.size, "seed": self.seed, "rank": self.rank, "world_size": self.world_size
        }

    def load_state_dict(self, state: dict) -> None:
        if state != self.state_dict():
            raise ValueError("Window sampler identity changed on resume")


class _WindowIterator:
    def __init__(self, sampler: WindowSampler):
        self.sampler = sampler
        self.cursor = 0
        self.cached_epoch = -1
        self.order = []

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, int]:
        sampler = self.sampler
        position = self.cursor * sampler.world_size + sampler.rank
        epoch, offset = divmod(position, sampler.size)
        if epoch != self.cached_epoch:
            self.order = list(range(sampler.size))
            random.Random(f"{sampler.seed}:{epoch}:window").shuffle(self.order)
            self.cached_epoch = epoch
        result = epoch, self.order[offset]
        self.cursor += 1
        return result

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        cursor = state["cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Window cursor must be a nonnegative integer")
        self.cursor = cursor
        self.cached_epoch = -1


def sequence_window_budgets(
    lengths: list[int],
    seed: int,
    world_size: int,
    global_batch_size: int,
    micro_batch_size: int,
    stride: int,
    max_anchors: int,
    horizon: int,
    steps: int,
) -> list[int]:
    """Replay the reference episode sampler to count valid windows per update."""
    from .sequence_dataset import PackedEpisodeSampler

    if (
        min(
            world_size,
            global_batch_size,
            micro_batch_size,
            stride,
            max_anchors,
            horizon,
            steps,
        )
        <= 0
    ):
        raise ValueError("Reference dimensions and step count must be positive")
    if global_batch_size % (world_size * micro_batch_size):
        raise ValueError(
            "Reference global batch must divide world size times micro batch"
        )
    if not lengths or any(n < horizon for n in lengths):
        raise ValueError("All reference episodes must contain a complete action window")
    counts = [min(max_anchors, (n - horizon) // stride + 1) for n in lengths]
    streams = [
        iter(PackedEpisodeSampler(len(lengths), seed, r, world_size))
        for r in range(world_size)
    ]
    return [
        sum(
            counts[next(stream)[1]]
            for stream in streams
            for _ in range(global_batch_size // world_size)
        )
        for _ in range(steps)
    ]


def window_backward_scale(
    windows: int, world_size: int, micro_batch_size: int
) -> float:
    """Convert padded microbatch means to a global valid-window mean after FSDP."""
    if min(windows, world_size, micro_batch_size) <= 0:
        raise ValueError("Window count, world size and micro batch must be positive")
    return world_size * micro_batch_size / windows


class MatchedWindowBatchSampler(WindowSampler):
    """Emit equal-size rank microbatches with masked tails and exact step budgets."""

    def __init__(
        self,
        size: int,
        seed: int,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        budgets: list[int],
    ):
        super().__init__(size, seed, rank, world_size)
        if (
            micro_batch_size <= 0
            or not budgets
            or any(type(n) is not int or n <= 0 for n in budgets)
        ):
            raise ValueError("Expected positive micro batch and integer window budgets")
        self.micro_batch_size = micro_batch_size
        self.budgets = tuple(budgets)
        capacity = world_size * micro_batch_size
        self.micro_ends = list(
            accumulate((n + capacity - 1) // capacity for n in budgets)
        )
        self.window_ends = list(accumulate(budgets))

    def __len__(self) -> int:
        return self.micro_ends[-1]

    def __iter__(self):
        return _MatchedWindowIterator(self)

    def state_dict(self) -> dict:
        import hashlib
        import json

        return {
            **super().state_dict(),
            "micro_batch_size": self.micro_batch_size,
            "budget_sha256": hashlib.sha256(
                json.dumps(self.budgets).encode()
            ).hexdigest(),
        }


class _MatchedWindowIterator:
    def __init__(self, sampler: MatchedWindowBatchSampler):
        self.sampler = sampler
        self.cursor = 0
        self.cached_epoch = -1
        self.order = []

    def __iter__(self):
        return self

    def __next__(self) -> list[tuple[int, int, bool, int, int, int]]:
        s = self.sampler
        if self.cursor >= len(s):
            raise StopIteration
        step = bisect_right(s.micro_ends, self.cursor)
        micro = self.cursor - (s.micro_ends[step - 1] if step else 0)
        prefix = s.window_ends[step - 1] if step else 0
        batch = []
        for j in range(s.micro_batch_size):
            offset = (
                micro * s.world_size * s.micro_batch_size
                + s.rank * s.micro_batch_size
                + j
            )
            valid = offset < s.budgets[step]
            # Padding reuses a real payload but consumes no uniform-stream position.
            position = prefix + min(offset, s.budgets[step] - 1)
            epoch, within = divmod(position, s.size)
            if epoch != self.cached_epoch:
                self.order = list(range(s.size))
                random.Random(f"{s.seed}:{epoch}:window").shuffle(self.order)
                self.cached_epoch = epoch
            batch.append(
                (epoch, self.order[within], valid, step + 1, micro, s.budgets[step])
            )
        self.cursor += 1
        return batch

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        cursor = state["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= len(self.sampler):
            raise ValueError("Matched sampler cursor must be within its finite plan")
        self.cursor, self.cached_epoch = cursor, -1
