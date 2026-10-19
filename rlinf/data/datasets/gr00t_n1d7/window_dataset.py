# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Independent legal H40 windows using the native processor and shared loss."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .sequence_dataset import (
    SequenceCollator,
    build_processor,
    data_metadata,
    dataset_settings,
    deterministic_augmentation,
    sample_seed,
)
from .window_sampling import (
    MatchedWindowBatchSampler,
    WindowIndex,
    WindowSampler,
    sequence_window_budgets,
)


class WindowDataset(Dataset):
    """Index every legal start once; decode only the observation being trained."""

    def __init__(self, config: dict, processor, loader):
        self.config, self.processor, self.loader = config, processor, loader
        if config["data"]["action_horizon"] != 40 or config["data"]["max_anchors"] != 1:
            raise ValueError(
                "Independent H40 samples require horizon 40 and one anchor"
            )
        self.index = WindowIndex(loader.episode_lengths, 40)
        self._parquet_cache = {}

    def __len__(self) -> int:
        return len(self.index)

    def raw_content(self, index: tuple[int, int]):
        """Read exactly the native observation and its next 40 real actions."""
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag

        epoch, flat = index
        row, anchor = self.index.locate(flat)
        episode_id = self.loader.episodes_metadata[row]["episode_index"]
        if episode_id not in self._parquet_cache:
            episode = self.loader._load_parquet_data(episode_id)
            if len(episode) != self.loader.episode_lengths[row]:
                raise ValueError("Metadata and parquet episode length differ")
            self._parquet_cache[episode_id] = episode
        window = (
            self._parquet_cache[episode_id]
            .iloc[anchor : anchor + 40]
            .reset_index(drop=True)
            .copy()
        )
        videos = self.loader._load_video_data(episode_id, np.asarray([anchor]))
        for camera, frames in videos.items():
            # Video delta is zero: future images are neither loaded nor supervised.
            window[f"video.{camera}"] = [frames[0]] + [None] * 39
        content = extract_step_data(
            window,
            0,
            self.loader.modality_configs,
            EmbodimentTag.NEW_EMBODIMENT,
            allow_padding=False,
        )
        description = {
            "epoch": epoch,
            "row": row,
            "episode": episode_id,
            "window_index": flat,
            "anchors": [anchor],
            "valid": 1,
            "terminal_aligned": anchor + 40 == self.loader.episode_lengths[row],
            "augmentation_seed": sample_seed(self.config["seed"], epoch, flat, 2),
        }
        return content, description

    def __getitem__(self, index: tuple[int, int] | int) -> dict:
        if isinstance(index, int):
            index = (0, index)
        content, description = self.raw_content(index[:2])
        with deterministic_augmentation(
            description["augmentation_seed"], self.processor
        ):
            step = self.processor([{"content": content}])
        if (
            tuple(step["action"].shape) != (40, 132)
            or step["action_mask"][:, :12].sum() != 480
        ):
            raise ValueError("Each window must supervise all 40 x 12 action values")
        is_valid = bool(index[2]) if len(index) == 6 else True
        description["valid"] = int(is_valid)
        if len(index) == 6:
            description.update(
                budget_step=index[3],
                budget_microbatch=index[4],
                budget_windows=index[5],
            )
        valid = torch.tensor([is_valid], dtype=torch.bool)
        return {
            "steps": [step],
            "anchor_valid": valid,
            "supervision": valid.clone(),
            "description": description,
        }


class WindowCollator(SequenceCollator):
    """Exclude padding rows while retaining a positive per-window denominator."""

    def __call__(self, samples: list[dict]) -> dict:
        output = super().__call__(samples)
        output["denominator"].clamp_min_(1)
        descriptions = output["descriptions"]
        if "budget_step" in descriptions[0]:
            for key in ("budget_step", "budget_microbatch", "budget_windows"):
                values = {d[key] for d in descriptions}
                if len(values) != 1:
                    raise ValueError(f"Mixed matched-budget microbatch: {key}")
                output[key] = values.pop()
        return output


def build_gr00t_window_dataloader(cfg, world_size: int, rank: int):
    """Build rank-sharded windows with a checkpointable consumed-sample cursor."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    settings = dataset_settings(cfg)
    processor, episode_loader, bundle = build_processor(settings)
    processor.train()
    if len(episode_loader.episodes_metadata) != cfg.data.expected_episodes:
        raise ValueError("Unexpected number of source episodes")
    dataset = WindowDataset(settings, processor, episode_loader)
    workers = cfg.data.num_workers
    metadata = data_metadata(settings)
    reference = cfg.data.robottt.get("match_sequence_budget")
    if reference is not None:
        import hashlib
        import json

        from omegaconf import OmegaConf

        reference = OmegaConf.to_container(reference, resolve=True)
        budgets = sequence_window_budgets(
            episode_loader.episode_lengths,
            cfg.actor.seed,
            world_size,
            reference["global_batch_size"],
            reference["micro_batch_size"],
            reference["stride"],
            reference["max_anchors"],
            settings["data"]["action_horizon"],
            cfg.runner.max_steps,
        )
        if max(budgets) > cfg.actor.global_batch_size:
            raise ValueError("Matched windows exceed configured global batch capacity")
        metadata["matched_sequence_budget"] = {
            "reference": reference,
            "seed": cfg.actor.seed,
            "world_size": world_size,
            "windows": budgets,
            "sha256": hashlib.sha256(json.dumps(budgets).encode()).hexdigest(),
        }
        batching = {
            "batch_sampler": MatchedWindowBatchSampler(
                len(dataset),
                cfg.actor.seed,
                rank,
                world_size,
                cfg.actor.micro_batch_size,
                budgets,
            )
        }
    else:
        batching = {
            "batch_size": cfg.actor.micro_batch_size,
            "sampler": WindowSampler(len(dataset), cfg.actor.seed, rank, world_size),
            "drop_last": False,
        }
    loader = StatefulDataLoader(
        dataset,
        **batching,
        collate_fn=WindowCollator(processor),
        num_workers=workers,
        generator=torch.Generator().manual_seed(cfg.actor.seed + rank),
        **({"prefetch_factor": cfg.data.prefetch_factor} if workers else {}),
    )
    metadata.update(sample_count=len(dataset), sample_unit="independent_h40_window")
    return loader, {"data": metadata, "processor": bundle}
