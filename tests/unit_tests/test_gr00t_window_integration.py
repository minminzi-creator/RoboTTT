# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Routing and reduction contracts for the shared H40 training worker."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf


class WindowIntegrationTests(unittest.TestCase):
    def test_independent_window_route(self):
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            build_gr00t_sequence_dataloader,
        )

        cfg = OmegaConf.create(
            {"data": {"robottt": {"sampling_mode": "independent_h40"}}}
        )
        with patch(
            "rlinf.data.datasets.gr00t_n1d7.window_dataset.build_gr00t_window_dataloader",
            return_value="window",
        ) as build:
            self.assertEqual(build_gr00t_sequence_dataloader(cfg, 4, 2), "window")
            build.assert_called_once_with(cfg, 4, 2)

    def test_unknown_mode_is_rejected(self):
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            build_gr00t_sequence_dataloader,
        )

        cfg = OmegaConf.create({"data": {"robottt": {"sampling_mode": "typo"}}})
        with self.assertRaisesRegex(ValueError, "sampling_mode"):
            build_gr00t_sequence_dataloader(cfg, 4, 0)

    def test_t1_loss_is_window_mean(self):
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import sequence_loss

        error = torch.tensor([1.0, 3.0, 5.0, 7.0], requires_grad=True)
        valid = torch.ones(4, 1, dtype=torch.bool)
        loss = sequence_loss(
            error[:, None, None], torch.ones(4, 1, 1), valid, valid, torch.ones(4)
        )
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        torch.testing.assert_close(error.grad, torch.full((4,), 0.25))

    def test_h40_epoch_uses_window_count(self):
        from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
            FSDPGr00tSequenceSftWorker,
        )

        cfg = OmegaConf.create(
            {
                "runner": {"max_epochs": -1, "max_steps": 10000},
                "data": {
                    "expected_episodes": 104,
                    "robottt": {"sampling_mode": "independent_h40"},
                },
            }
        )
        worker = SimpleNamespace(
            cfg=cfg,
            global_batch_size=1024,
            data_config={"data": {"sample_count": 112737}},
        )
        self.assertEqual(
            FSDPGr00tSequenceSftWorker.get_max_steps_per_epoch(worker), 110
        )

    def test_worker_uses_dynamic_microbatch_count_and_valid_window_scale(self):
        from rlinf.workers.sft.fsdp_gr00t_sequence_sft_worker import (
            FSDPGr00tSequenceSftWorker,
        )

        worker = SimpleNamespace(
            data_config={
                "data": {"matched_sequence_budget": {"windows": [936, 1024, 1]}}
            },
            global_step=0,
            gradient_accumulation=16,
            _world_size=4,
            cfg=OmegaConf.create({"actor": {"micro_batch_size": 16}}),
        )
        self.assertEqual(
            FSDPGr00tSequenceSftWorker.step_batching(worker), (15, 64 / 936, 936)
        )
        worker.global_step = 2
        self.assertEqual(FSDPGr00tSequenceSftWorker.step_batching(worker), (1, 64.0, 1))
        worker.data_config = {"data": {}}
        worker.gradient_accumulation = 2
        self.assertEqual(
            FSDPGr00tSequenceSftWorker.step_batching(worker), (2, 0.5, None)
        )


if __name__ == "__main__":
    unittest.main()
