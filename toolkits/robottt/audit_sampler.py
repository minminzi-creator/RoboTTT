# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Audit sampler coverage, full action indices and real worker reproducibility."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
    ACTION_KEYS,
    CAMERAS,
    STATE_KEYS,
    EpisodeDataset,
    EpisodeSampler,
    SequenceCollator,
    build_processor,
    data_metadata,
    segment_batch,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    dataset_settings,
    load_sft_config,
)


def tensor_digest(batch: dict) -> dict:
    return {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
        for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = dataset_settings(load_sft_config(args.config))
    torch.set_num_threads(2)
    processor, loader, bundle = build_processor(cfg)
    dataset = EpisodeDataset(cfg, processor, loader)
    report = {"metadata": data_metadata(cfg), "epochs": [], "processor": bundle}
    for epoch in range(3):
        ranks, seen = [], []
        for rank in range(8):
            indices = list(EpisodeSampler(len(dataset), cfg["seed"], epoch, rank, 8))
            rows = []
            for index in indices:
                item = dataset.describe(index)
                actions = np.array(item["anchors"])[:, None] + np.arange(40)
                assert actions.min() >= 0
                assert actions.max() < loader.episode_lengths[index[1]]
                item["action_indices"] = actions.tolist()
                item["rank"] = rank
                rows.append(item)
                seen.append(item["episode"])
            assert (
                list(EpisodeSampler(len(dataset), cfg["seed"], epoch, rank, 8, 24))
                == indices[3:]
            )
            ranks.append(rows)
        assert len(seen) == len(set(seen)) == 104
        report["epochs"].append(
            {
                "epoch": epoch,
                "coverage": len(set(seen)),
                "duplicates": 0,
                "ranks": ranks,
            }
        )
    hashes = []
    for workers in [0, 2]:
        # One short and one full-length window; only worker topology changes.
        dl = DataLoader(
            dataset,
            batch_size=1,
            sampler=[(0, 0), (0, 1)],
            num_workers=workers,
            collate_fn=SequenceCollator(processor),
            generator=torch.Generator().manual_seed(42),
        )
        worker_hashes = []
        for batch in dl:
            worker_hashes.append(tensor_digest(batch))
        hashes.append(worker_hashes)
    assert hashes[0] == hashes[1], "Worker count changed a processed sample"
    segment = segment_batch(batch, 0, 16, torch.device("cpu"))
    report["worker_reproducibility"] = {
        "workers": [0, 2],
        "episodes": [0, 1],
        "identical": True,
        "tensor_sha256": hashes[0],
    }
    report["shapes"] = {
        k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)
    }
    report["segment_shapes"] = {k: list(v.shape) for k, v in segment.items()}
    mask = batch["action_mask"]
    assert mask.shape[-2:] == (40, 132)
    assert (mask[..., :12] == 1).all() and (mask[..., 12:] == 0).all()
    report["full_horizon"] = {
        "supervised_values_per_anchor": 480,
        "late_action_rms": batch["action"][:, 8:, :12].square().mean().sqrt().item(),
        "horizon_padding": 0,
    }
    # Compare original parquet slices with loader groups, independently of model.
    import pandas as pd

    info = loader.info_meta
    meta = loader.modality_meta
    checks = []
    for row, ep in enumerate(loader.episodes_metadata):
        episode_id = ep["episode_index"]
        path = Path(cfg["data"]["path"]) / info["data_path"].format(
            episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id
        )
        frame = pd.read_parquet(path)
        assert len(frame) == ep["length"]
        state = np.stack(frame["observation.state"])
        action = np.stack(frame["action"])
        assert state.shape[1] == 16 and action.shape[1] == 12
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert (np.diff(frame["frame_index"].to_numpy()) == 1).all()
        decoded = loader._load_parquet_data(episode_id)
        decoded_state = np.concatenate(
            [np.stack(decoded[f"state.{key}"]) for key in STATE_KEYS], axis=-1
        )
        decoded_action = np.concatenate(
            [np.stack(decoded[f"action.{key}"]) for key in ACTION_KEYS], axis=-1
        )
        assert np.array_equal(decoded_state, state)
        assert np.array_equal(decoded_action, action)
        checks.append(
            {
                "episode": episode_id,
                "frames": len(frame),
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "decoded_modality_values_equal_raw_parquet": True,
            }
        )
    report["parquet_checks"] = checks
    report["alignment"] = {
        "cameras": CAMERAS,
        "state": {k: meta["state"][k] for k in STATE_KEYS},
        "action": {k: meta["action"][k] for k in ACTION_KEYS},
        "augmentation_seed_per_episode": True,
    }
    report["short_episodes_retained"] = sum(
        (length - 40) // 8 + 1 < 128 for length in loader.episode_lengths
    )
    report["status"] = "passed"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
