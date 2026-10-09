# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Episode-equal deterministic windows with real 40-frame action supervision."""

import hashlib
import inspect
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
STATE_KEYS = [
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
]
ACTION_KEYS = [
    "base_motion",
    "control_mode",
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
]
LANGUAGE_KEY = "annotation.human.task_description"


def modality_dict(horizon: int = 40) -> dict:
    return {
        "video": {"delta_indices": [0], "modality_keys": CAMERAS},
        "state": {"delta_indices": [0], "modality_keys": STATE_KEYS},
        "action": {"delta_indices": list(range(horizon)), "modality_keys": ACTION_KEYS},
        "language": {"delta_indices": [0], "modality_keys": [LANGUAGE_KEY]},
    }


def sample_seed(seed: int, epoch: int, episode: int, purpose: int = 0) -> int:
    return int(
        np.random.SeedSequence([seed, epoch, episode, purpose]).generate_state(1)[0]
    )


def window_indices(
    length: int,
    episode: int,
    epoch: int,
    seed: int,
    stride: int = 8,
    maximum: int = 128,
    horizon: int = 40,
) -> np.ndarray:
    valid = min(maximum, (length - horizon) // stride + 1)
    if valid < 1:
        raise ValueError(
            f"Episode {episode} has {length} frames, shorter than horizon {horizon}"
        )
    last_start = length - horizon - (valid - 1) * stride
    start = np.random.default_rng(sample_seed(seed, epoch, episode)).integers(
        last_start + 1
    )
    return start + np.arange(valid, dtype=np.int64) * stride


class EpisodeSampler(Sampler):
    """One shared permutation, exact rank sharding; no implicit duplicate padding."""

    def __init__(
        self,
        size: int,
        seed: int,
        epoch: int,
        rank: int,
        world_size: int,
        skip_global: int = 0,
    ):
        if size % world_size or skip_global % world_size:
            raise ValueError("Episode count and resume offset must divide world size")
        self.size, self.seed, self.rank, self.world_size = size, seed, rank, world_size
        self.skip_global = skip_global
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        order = np.random.default_rng(sample_seed(self.seed, epoch, 0, 1)).permutation(
            self.size
        )
        self.indices = [
            (epoch, int(i))
            for i in order[self.skip_global :][self.rank :: self.world_size]
        ]

    def state_dict(self):
        # The sampler owns epoch/order; its iterator separately owns the cursor.
        # StatefulDataLoader snapshots both at its consumed-batch boundary.
        return {"epoch": self.epoch, "skip_global": self.skip_global}

    def load_state_dict(self, state):
        self.skip_global = state["skip_global"]
        self.set_epoch(state["epoch"])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return _EpisodeIterator(self.indices)


class _EpisodeIterator:
    """Checkpoint a cursor without making a fresh epoch inherit an old cursor."""

    def __init__(self, indices: list[tuple[int, int]]):
        self.indices = tuple(indices)
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, int]:
        if self.cursor == len(self.indices):
            raise StopIteration
        index = self.indices[self.cursor]
        self.cursor += 1
        return index

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        cursor = state["cursor"]
        if not isinstance(cursor, int) or not 0 <= cursor <= len(self.indices):
            raise ValueError("Invalid episode iterator cursor")
        self.cursor = cursor


@contextmanager
def deterministic_augmentation(seed: int, processor):
    """An item's augmentation is independent of worker count and prefetch history."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if hasattr(processor.train_image_transform, "set_random_seed"):
            processor.train_image_transform.set_random_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def build_processor(config: dict, bundle_path: str | None = None):
    """The ONLY processor factory used by dataset, eval and replay diagnostics."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.types import ModalityConfig
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    loader = None
    if bundle_path:
        kwargs = json.loads(Path(bundle_path).read_text())
    else:
        base = Path(config["model"]["base_path"])
        if (base / "processor").is_dir():
            base /= "processor"
        kwargs = json.loads((base / "processor_config.json").read_text())[
            "processor_kwargs"
        ]
        modality = modality_dict(config["data"]["action_horizon"])
        loader = LeRobotEpisodeLoader(
            config["data"]["path"],
            {k: ModalityConfig(**v) for k, v in modality.items()},
            video_backend=config["data"]["video_backend"],
        )
        kwargs["modality_configs"] = {"new_embodiment": modality}
        kwargs["statistics"] = {"new_embodiment": loader.get_dataset_statistics()}
        kwargs["embodiment_id_mapping"] = {"new_embodiment": 10}
    kwargs["model_name"] = config["model"]["backbone_path"]
    kwargs["transformers_loading_kwargs"] = {"local_files_only": True}
    for name, parameter in inspect.signature(
        Gr00tN1d7Processor.__init__
    ).parameters.items():
        if name != "self" and parameter.default is not inspect.Parameter.empty:
            kwargs.setdefault(name, parameter.default)
    processor = Gr00tN1d7Processor(**kwargs)
    return processor, loader, kwargs


class EpisodeDataset(Dataset):
    def __init__(self, config: dict, processor, loader):
        self.config = config
        self.processor = processor
        self.loader = loader
        self.episode_ids = [r["episode_index"] for r in loader.episodes_metadata]

    def __len__(self):
        return len(self.episode_ids)

    def describe(self, index: tuple[int, int]) -> dict:
        epoch, row = index
        cfg = self.config["data"]
        episode = self.episode_ids[row]
        anchors = window_indices(
            self.loader.episode_lengths[row],
            episode,
            epoch,
            self.config["seed"],
            cfg["stride"],
            cfg["max_anchors"],
            cfg["action_horizon"],
        )
        return {
            "epoch": epoch,
            "row": row,
            "episode": episode,
            "anchors": anchors.tolist(),
            "valid": len(anchors),
            "augmentation_seed": sample_seed(self.config["seed"], epoch, episode, 2),
        }

    def __getitem__(self, index: tuple[int, int] | int) -> dict:
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag

        if isinstance(index, int):
            index = (0, index)
        description = self.describe(index)
        modality = self.processor.modality_configs["new_embodiment"]
        maximum = self.config["data"]["max_anchors"]
        with deterministic_augmentation(
            description["augmentation_seed"], self.processor
        ):
            episode = self.loader[index[1]]
            if len(episode) != self.loader.episode_lengths[index[1]]:
                raise ValueError("Metadata and decoded episode length differ")
            steps = []
            for anchor in description["anchors"]:
                content = extract_step_data(
                    episode,
                    anchor,
                    modality,
                    EmbodimentTag.NEW_EMBODIMENT,
                    allow_padding=False,
                )
                step = self.processor([{"content": content}])
                if (
                    tuple(step["action"].shape) != (40, 132)
                    or step["action_mask"][:, :12].sum() != 480
                ):
                    raise ValueError(
                        "Every valid anchor must supervise all 40 x 12 action values"
                    )
                steps.append(step)
        n = len(steps)
        # Duplicate processed payload only to keep collective shapes identical.
        # Separate masks prevent this payload from affecting loss/state/position.
        steps.extend([steps[-1]] * (maximum - n))
        valid = torch.arange(maximum) < n
        return {
            "steps": steps,
            "anchor_valid": valid,
            "supervision": valid.clone(),
            "description": description,
        }


class SequenceCollator:
    def __init__(self, processor):
        self.collator = processor.collator

    def __call__(self, samples: list[dict]) -> dict:
        b, t = len(samples), len(samples[0]["steps"])
        output = dict(
            self.collator([s for sample in samples for s in sample["steps"]])["inputs"]
        )
        # Make flattened Qwen image patches sliceable along the anchor dimension.
        for key in ("pixel_values", "image_grid_thw"):
            tensor = output[key]
            if tensor.shape[0] % (b * t):
                raise ValueError(f"{key} cannot be divided into aligned observations")
            output[key] = tensor.reshape(b * t, -1, *tensor.shape[1:])
        output["anchor_valid"] = torch.stack([x["anchor_valid"] for x in samples])
        output["supervision"] = torch.stack([x["supervision"] for x in samples])
        output["denominator"] = (
            (output["anchor_valid"] & output["supervision"]).sum(1).float()
        )
        output["descriptions"] = [x["description"] for x in samples]
        return output


def segment_batch(batch: dict, start: int, end: int, device: torch.device) -> dict:
    b, t = batch["anchor_valid"].shape
    output = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in {"anchor_valid", "supervision"}:
            value = value[:, start:end]
        elif key != "denominator":
            value = value.reshape(b, t, *value.shape[1:])[:, start:end].flatten(0, 1)
            if key in {"pixel_values", "image_grid_thw"}:
                value = value.flatten(0, 1)
        output[key] = value.to(device, non_blocking=True)
    return output


def data_metadata(config: dict) -> dict:
    root = Path(config["data"]["path"]) / "meta"
    files = [
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "stats.json",
        "modality.json",
        "embodiment.json",
    ]
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }
    rows = [json.loads(x) for x in (root / "episodes.jsonl").read_text().splitlines()]
    return {
        "sha256": hashes,
        "episodes": rows,
        "camera_order": CAMERAS,
        "state_order": STATE_KEYS,
        "action_order": ACTION_KEYS,
        "modality": modality_dict(),
    }
