# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Independent-window coverage and resumable distributed sampling."""

import itertools
import unittest


class WindowSamplingTests(unittest.TestCase):
    def test_every_legal_window_has_exactly_one_index(self):
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import WindowIndex

        index = WindowIndex([40, 42, 41], 40)
        self.assertEqual(len(index), 6)
        self.assertEqual(
            [index.locate(i) for i in range(6)],
            [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        )
        for invalid in [-1, 6]:
            with self.assertRaises(IndexError):
                index.locate(invalid)
        with self.assertRaises(ValueError):
            WindowIndex([39], 40)

    def test_rank_stream_covers_nondivisible_epochs_without_duplicates(self):
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import WindowSampler

        streams = [iter(WindowSampler(7, 42, rank, 4)) for rank in range(4)]
        merged = [next(streams[i % 4]) for i in range(28)]
        for epoch in range(4):
            entries = [(e, i) for e, i in merged if e == epoch]
            self.assertEqual(len(entries), 7)
            self.assertEqual(sorted(i for _, i in entries), list(range(7)))

    def test_resume_restores_cursor_across_epoch_boundary(self):
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import WindowSampler

        first = iter(WindowSampler(7, 42, 3, 4))
        list(itertools.islice(first, 5))
        state = first.state_dict()
        expected = list(itertools.islice(first, 20))
        restored = iter(WindowSampler(7, 42, 3, 4))
        restored.load_state_dict(state)
        self.assertEqual(list(itertools.islice(restored, 20)), expected)
        with self.assertRaises(ValueError):
            restored.load_state_dict({"cursor": -1})
        with self.assertRaises(ValueError):
            WindowSampler(7, 42, 3, 4).load_state_dict(
                WindowSampler(7, 43, 3, 4).state_dict()
            )

    def test_invalid_sampler_identity_is_rejected(self):
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import WindowSampler

        for args in [(0, 42, 0, 4), (7, 42, 4, 4), (7, 42, 0, 0)]:
            with self.assertRaises(ValueError):
                WindowSampler(*args)

    def test_budget_matches_packed_sequence_stream(self):
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            PackedEpisodeSampler,
            window_indices,
        )
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import (
            sequence_window_budgets,
        )

        lengths = [40, 48, 56, 1200, 41, 80, 800, 2000]
        budgets = sequence_window_budgets(lengths, 42, 2, 4, 1, 8, 128, 40, 19)
        streams = [iter(PackedEpisodeSampler(8, 42, r, 2)) for r in range(2)]
        expected = []
        for _ in range(19):
            count = 0
            for stream in streams:
                for _ in range(2):
                    epoch, row = next(stream)
                    count += len(
                        window_indices(lengths[row], row, epoch, 42, 8, 128, 40)
                    )
            expected.append(count)
        self.assertEqual(budgets, expected)

    def test_matched_batches_preserve_stream_and_mask_tails(self):
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import (
            MatchedWindowBatchSampler,
            WindowSampler,
        )

        budgets = [5, 9, 1]
        streams = [
            iter(MatchedWindowBatchSampler(7, 42, r, 2, 2, budgets)) for r in range(2)
        ]
        merged = []
        for step, budget in enumerate(budgets, 1):
            count = 0
            for micro in range((budget + 3) // 4):
                for stream in streams:
                    batch = next(stream)
                    self.assertEqual(len(batch), 2)
                    for (
                        epoch,
                        flat,
                        valid,
                        actual_step,
                        actual_micro,
                        actual_budget,
                    ) in batch:
                        self.assertEqual(
                            (actual_step, actual_micro, actual_budget),
                            (step, micro, budget),
                        )
                        if valid:
                            merged.append((epoch, flat))
                            count += 1
            self.assertEqual(count, budget)
        expected = list(
            itertools.islice(iter(WindowSampler(7, 42, 0, 1)), sum(budgets))
        )
        self.assertEqual(merged, expected)
        for stream in streams:
            with self.assertRaises(StopIteration):
                next(stream)

    def test_matched_resume_and_identity(self):
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import (
            MatchedWindowBatchSampler,
        )

        sampler = MatchedWindowBatchSampler(7, 42, 1, 2, 2, [5, 9, 1])
        iterator = iter(sampler)
        next(iterator)
        state = iterator.state_dict()
        expected = list(iterator)
        restored = iter(sampler)
        restored.load_state_dict(state)
        self.assertEqual(list(restored), expected)
        with self.assertRaises(ValueError):
            restored.load_state_dict({"cursor": 999})
        changed = MatchedWindowBatchSampler(7, 42, 1, 2, 2, [5, 8, 1])
        with self.assertRaises(ValueError):
            changed.load_state_dict(sampler.state_dict())
        for budgets in ([], [0], [1.5]):
            with self.assertRaises(ValueError):
                MatchedWindowBatchSampler(7, 42, 1, 2, 2, budgets)

    def test_masked_microbatch_gradient_matches_global_mean(self):
        import torch

        from rlinf.data.datasets.gr00t_n1d7.window_sampling import window_backward_scale
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_loss

        for count in (1, 5, 9):
            gradients = []
            for rank in range(2):
                parameter = torch.tensor(2.0, requires_grad=True)
                for micro in range((count + 3) // 4):
                    offsets = torch.arange(2) + micro * 4 + rank * 2
                    valid = (offsets < count)[:, None]
                    errors = (parameter * (offsets.float() + 1)).square()[:, None, None]
                    loss = sequence_loss(
                        errors, torch.ones_like(errors), valid, valid, torch.ones(2)
                    )
                    (loss * window_backward_scale(count, 2, 2)).backward()
                gradients.append(parameter.grad)
            full = torch.tensor(2.0, requires_grad=True)
            (full * torch.arange(1, count + 1)).square().mean().backward()
            torch.testing.assert_close(sum(gradients) / 2, full.grad)


if __name__ == "__main__":
    unittest.main()
