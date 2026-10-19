# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Real-data equivalence and consumed-cursor recovery for independent windows."""

import unittest
from pathlib import Path

import numpy as np


class WindowDatasetTests(unittest.TestCase):
    def test_sparse_window_matches_native_episode(self):
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.types import EmbodimentTag, ModalityConfig

        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import modality_dict
        from rlinf.data.datasets.gr00t_n1d7.window_dataset import WindowDataset

        root = Path("/mnt/public/chenzm/robottt/h800_sft/assets/lerobot")
        if not root.exists():
            self.skipTest("BowlAndCup integration data unavailable")
        modality = {k: ModalityConfig(**v) for k, v in modality_dict().items()}
        loader = LeRobotEpisodeLoader(root, modality, video_backend="torchcodec")
        dataset = WindowDataset(
            {"seed": 42, "data": {"action_horizon": 40, "max_anchors": 1}}, None, loader
        )
        self.assertEqual(len(dataset), 112737)
        for flat in (0, 10, len(dataset) - 1):
            actual, description = dataset.raw_content((0, flat))
            row, anchor = dataset.index.locate(flat)
            expected = extract_step_data(
                loader[row], anchor, modality, EmbodimentTag.NEW_EMBODIMENT, False
            )
            self.assertEqual(description["anchors"], [anchor])
            self.assertEqual(actual.text, expected.text)
            for group in ("images", "states", "actions"):
                got, want = getattr(actual, group), getattr(expected, group)
                self.assertEqual(set(got), set(want))
                for key in want:
                    np.testing.assert_array_equal(
                        np.asarray(got[key]), np.asarray(want[key])
                    )

    def test_real_processed_batch_and_augmentation_are_reproducible(self):
        import torch
        from omegaconf import OmegaConf

        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            SequenceCollator,
            build_gr00t_sequence_dataloader,
            segment_batch,
        )

        config_path = Path("/mnt/public/chenzm/robottt/h800_sft/h40/config.yaml")
        if not config_path.exists():
            self.skipTest("H40 integration configuration unavailable")
        cfg = OmegaConf.load(config_path)
        cfg.data.num_workers = 2
        loader, metadata = build_gr00t_sequence_dataloader(cfg, 4, 0)
        self.assertEqual(metadata["data"]["sample_count"], 112737)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["anchor_valid"].shape), (16, 1))
        self.assertTrue(batch["anchor_valid"].all())
        self.assertTrue(batch["denominator"].eq(1).all())
        segment = segment_batch(batch, 0, 1, torch.device("cpu"))
        self.assertEqual(tuple(segment["action"].shape), (16, 40, 132))
        self.assertEqual(segment["action_mask"][:, :, :12].sum().item(), 16 * 40 * 12)
        self.assertTrue(torch.isfinite(segment["action"]).all())
        collate = SequenceCollator(loader.dataset.processor)
        first = collate([loader.dataset[(2, 100)]])
        second = collate([loader.dataset[(2, 100)]])
        for key in first:
            if isinstance(first[key], torch.Tensor):
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_stateful_loader_restores_consumed_cursor(self):
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader

        from rlinf.data.datasets.gr00t_n1d7.window_sampling import WindowSampler

        for workers in (0, 2):

            def make():
                return StatefulDataLoader(
                    WindowIdentityDataset(),
                    batch_size=16,
                    sampler=WindowSampler(7, 42, 2, 4),
                    num_workers=workers,
                    **({"prefetch_factor": 1} if workers else {}),
                )

            original = make()
            iterator = iter(original)
            next(iterator)
            state = original.state_dict()
            expected = [next(iterator) for _ in range(3)]
            restored = make()
            restored.load_state_dict(state)
            actual = iter(restored)
            for want in expected:
                self.assertTrue(torch.equal(next(actual), want))

    def test_matched_real_padding_and_plan(self):
        from omegaconf import OmegaConf

        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            build_gr00t_sequence_dataloader,
        )
        from rlinf.data.datasets.gr00t_n1d7.window_dataset import WindowCollator

        path = Path("/mnt/public/chenzm/robottt/h800_sft/h40/config.yaml")
        if not path.exists():
            self.skipTest("H40 integration configuration unavailable")
        cfg = OmegaConf.load(path)
        cfg.data.num_workers = 0
        cfg.data.robottt.match_sequence_budget = {
            "global_batch_size": 8,
            "micro_batch_size": 1,
            "stride": 8,
            "max_anchors": 128,
        }
        loader, metadata = build_gr00t_sequence_dataloader(cfg, 4, 0)
        plan = metadata["data"]["matched_sequence_budget"]
        self.assertEqual(len(plan["windows"]), 10000)
        self.assertEqual(plan["windows"][4], 936)
        dataset = loader.dataset
        batch = WindowCollator(dataset.processor)(
            [dataset[(0, 0, True, 1, 0, 1)], dataset[(0, 0, False, 1, 0, 1)]]
        )
        self.assertEqual(batch["anchor_valid"].tolist(), [[True], [False]])
        self.assertEqual(batch["denominator"].tolist(), [1.0, 1.0])
        self.assertEqual(batch["budget_windows"], 1)
        self.assertEqual(batch["budget_step"], 1)
        self.assertEqual(batch["budget_microbatch"], 0)
        self.assertEqual([d["valid"] for d in batch["descriptions"]], [1, 0])

    def test_matched_stateful_loader_restores_consumed_cursor(self):
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader

        from rlinf.data.datasets.gr00t_n1d7.window_sampling import (
            MatchedWindowBatchSampler,
        )

        for workers in (0, 2):

            def make():
                return StatefulDataLoader(
                    WindowIdentityDataset(),
                    batch_sampler=MatchedWindowBatchSampler(
                        7, 42, 1, 2, 2, [5, 9, 1, 7]
                    ),
                    num_workers=workers,
                    **({"prefetch_factor": 1} if workers else {}),
                )

            original = make()
            iterator = iter(original)
            next(iterator)
            state = original.state_dict()
            expected = [next(iterator) for _ in range(3)]
            restored = make()
            restored.load_state_dict(state)
            actual = iter(restored)
            for want in expected:
                self.assertTrue(torch.equal(next(actual), want))


class WindowIdentityDataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        import torch

        return torch.tensor(index)


if __name__ == "__main__":
    unittest.main()
