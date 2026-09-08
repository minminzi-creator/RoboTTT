# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Single-step SFT dataset for GR00T N1.7 on RoboCasa.

This is the pure GR00T N1.7 SFT data path used as the control baseline
against RoboTTT.

Unlike RoboTTT sequence training:
  - each sample contains exactly one observation timestep;
  - no temporal T-step window is constructed;
  - no human-video prefix is added;
  - no loss_mask or num_timesteps is produced;
  - no B*T flattening is performed.

The normal GR00T action chunk is preserved. For an observation at timestep t,
the processor still constructs the H-step action target according to the
configured action delta_indices.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _batchify_gr00t_forward_input,
    _find_processor_dir,
    redirect_qwen3_backbone_to_local,
)
from rlinf.utils.logging import get_logger


logger = get_logger()

_DEFAULT_BACKBONE_MODEL_NAME = "nvidia/Cosmos-Reason2-2B"

_EMBODIMENT_TAG_BY_CFG = {
    "libero_sim": "libero_sim",
    "libero_panda": "libero_panda",
    "libero_franka": "libero_franka",
    "isaaclab_franka": "isaaclab_franka",
    "maniskill_widowx": "maniskill_widowx",
    "robocasa_panda_omron": "robocasa_panda_omron",
    "gr1": "gr1",
    "behavior_r1_pro": "behavior_r1_pro",
    "new_embodiment": "new_embodiment",
    "so101": "new_embodiment",
    "so100": "new_embodiment",
}


def _resolve_backbone_model_path(
    backbone_model_path: Optional[str],
) -> Optional[str]:
    if backbone_model_path is None:
        return None

    resolved = str(Path(backbone_model_path).expanduser().resolve())

    if not Path(resolved).is_dir():
        raise FileNotFoundError(
            f"Backbone model path does not exist: {resolved}"
        )

    return resolved


def _load_external_modality_config(
    config_path: str,
    config_name: Optional[str] = None,
):
    """Load a ModalityConfig dictionary from an external Python file."""

    path = Path(config_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Modality config file does not exist: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        f"_rlinf_modality_{path.stem}",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import modality config from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if config_name is not None:
        if not hasattr(module, config_name):
            raise AttributeError(
                f"{path} has no modality config named {config_name}"
            )

        cfg = getattr(module, config_name)

    else:
        candidates = []

        for name, value in vars(module).items():
            if (
                isinstance(value, dict)
                and all(
                    k in value
                    for k in (
                        "video",
                        "state",
                        "action",
                        "language",
                    )
                )
            ):
                candidates.append((name, value))

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one modality config in {path}, "
                f"found {[name for name, _ in candidates]}. "
                "Set data.modality_config_name explicitly."
            )

        config_name, cfg = candidates[0]

    logger.info(
        "Loaded external modality config %s from %s",
        config_name,
        path,
    )

    return cfg


def _load_processor(
    model_path: str,
    backbone_model_path: Optional[str] = None,
):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        Gr00tN1d7Processor,
    )

    processor_dir = _find_processor_dir(Path(model_path))

    if processor_dir is None:
        raise FileNotFoundError(
            f"No GR00T N1.7 processor files under {model_path}. "
            "Expected processor_config.json, statistics.json, "
            "embodiment_id.json."
        )

    with open(processor_dir / "processor_config.json", "r") as f:
        processor_cfg = json.load(f)["processor_kwargs"]

    with open(processor_dir / "statistics.json", "r") as f:
        processor_cfg["statistics"] = json.load(f)

    with open(processor_dir / "embodiment_id.json", "r") as f:
        processor_cfg["embodiment_id_mapping"] = json.load(f)

    resolved_backbone = _resolve_backbone_model_path(
        backbone_model_path
    )

    if resolved_backbone is not None:
        processor_cfg.setdefault(
            "transformers_loading_kwargs",
            {},
        )

        processor_cfg["transformers_loading_kwargs"][
            "local_files_only"
        ] = True

        logger.info(
            "Loading processor backbone locally from %s "
            "(canonical model_name=%s)",
            resolved_backbone,
            processor_cfg.get(
                "model_name",
                _DEFAULT_BACKBONE_MODEL_NAME,
            ),
        )

    canonical_name = str(
        processor_cfg.get(
            "model_name",
            _DEFAULT_BACKBONE_MODEL_NAME,
        )
    )

    with redirect_qwen3_backbone_to_local(
        canonical_name,
        resolved_backbone,
    ):
        processor = Gr00tN1d7Processor(**processor_cfg)

    processor.training = True

    return processor


def _resolve_embodiment_tag(tag_name: str):
    from rlinf.models.embodiment.gr00t.embodiment_tags import (
        EmbodimentTag,
    )

    mapped = _EMBODIMENT_TAG_BY_CFG.get(
        tag_name,
        tag_name,
    )

    return EmbodimentTag(mapped)


class Gr00tN1d7SingleStepDataset(Dataset):
    """Sample one RoboCasa timestep for pure GR00T N1.7 SFT."""

    def __init__(
        self,
        dataset_path: str,
        processor: Any,
        embodiment_tag: Any,
        seed: int = 0,
        video_backend: str = "torchcodec",
        samples_per_epoch: Optional[int] = None,
    ):
        from gr00t.data.dataset.lerobot_episode_loader import (
            LeRobotEpisodeLoader,
        )

        self.processor = processor
        self.embodiment_tag = embodiment_tag
        self.rng = np.random.RandomState(seed)

        tag_value = embodiment_tag.value

        if tag_value not in processor.modality_configs:
            raise KeyError(
                f"Processor has no modality config for {tag_value}; "
                f"available: {list(processor.modality_configs)}"
            )

        self.modality_configs = (
            processor.modality_configs[tag_value]
        )

        # IMPORTANT:
        # This is the normal GR00T action chunk horizon H.
        # It is NOT the RoboTTT temporal context length T.
        self.action_horizon = len(
            self.modality_configs["action"].delta_indices
        )

        self.loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=self.modality_configs,
            video_backend=video_backend,
        )

        self._index = self._build_index()

        if not self._index:
            raise ValueError(
                f"No episode in {dataset_path} is long enough "
                f"for action_horizon={self.action_horizon}."
            )

        self.samples_per_epoch = (
            int(samples_per_epoch)
            if samples_per_epoch is not None
            else len(self._index)
        )

    def _build_index(
        self,
    ) -> list[tuple[int, int]]:
        """Build a deterministic global single-step index.

        Each row is:
            (episode_id, step_index)

        A timestep t is valid when the normal GR00T H-step action
        target starting at t can be constructed:

            t <= episode_length - action_horizon

        The resulting map-style dataset allows DistributedSampler to
        partition real timesteps across ranks without each rank
        independently resampling the same episode/timestep.
        """

        index: list[tuple[int, int]] = []

        for episode_id, length in enumerate(
            self.loader.episode_lengths
        ):
            max_start = int(length) - self.action_horizon

            if max_start < 0:
                continue

            for step_index in range(max_start + 1):
                index.append(
                    (
                        episode_id,
                        step_index,
                    )
                )

        return index

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """Return one deterministic single-step GR00T sample."""

        from gr00t.data.dataset.sharded_single_step_dataset import (
            extract_step_data,
        )

        # DistributedSampler owns the shuffle/partitioning.
        # Map the sampler-provided dataset index directly to a real
        # RoboCasa (episode, timestep) pair.
        sample_index = int(index) % len(self._index)
        episode_id, step_index = self._index[sample_index]

        episode_data = self.loader[episode_id]

        step = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            allow_padding=True,
        )

        # Processor constructs the normal H-step action target.
        processed = self.processor(
            [{"content": step}]
        )

        return processed


class Gr00tN1d7SingleStepCollator:
    """Collate B independent GR00T samples into [B, ...]."""

    def __init__(
        self,
        model_name: str,
        transformers_loading_kwargs: Optional[
            dict
        ] = None,
    ):
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
            Gr00tN1d7DataCollator,
        )

        self._collator = Gr00tN1d7DataCollator(
            model_name=model_name,
            transformers_loading_kwargs=(
                transformers_loading_kwargs
                or {}
            ),
        )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Unlike RoboTTT, samples are already B
        # independent single-step processor outputs.
        collated = self._collator(samples)

        batch = dict(
            collated["inputs"]
            if "inputs" in collated
            else collated
        )

        batch_size = len(samples)

        for key in (
            "pixel_values",
            "image_grid_thw",
            "image_sizes",
        ):
            if key in batch:
                batch[key] = (
                    _batchify_gr00t_forward_input(
                        key,
                        batch[key],
                        batch_size,
                    )
                )

        # Intentionally NO:
        #   loss_mask
        #   num_timesteps

        return batch


def build_gr00t_n1d7_single_step_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths,
    eval_dataset: bool = False,
):
    """Build pure GR00T N1.7 single-step SFT dataloader."""

    from rlinf.utils.patcher import Patcher

    Patcher.clear()

    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t."
        "embodiment_tags.EmbodimentTag",
    )

    Patcher.apply()

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    dataset_path = (
        data_paths
        if isinstance(data_paths, str)
        else data_paths[0]
    )

    backbone_model_path = (
        _resolve_backbone_model_path(
            model_cfg.get(
                "backbone_model_path",
                None,
            )
        )
    )

    processor = _load_processor(
        str(model_cfg.model_path),
        backbone_model_path=backbone_model_path,
    )

    embodiment_tag = _resolve_embodiment_tag(
        str(model_cfg.embodiment_tag)
    )

    # Pure GR00T SFT configuration.
    # These are intentionally no longer stored under data.robottt.
    modality_config_path = data_cfg.get(
        "modality_config_path",
        None,
    )

    if modality_config_path:
        custom_modality_cfg = (
            _load_external_modality_config(
                str(modality_config_path),
                config_name=data_cfg.get(
                    "modality_config_name",
                    None,
                ),
            )
        )

        tag_value = embodiment_tag.value

        # Processor and StateActionProcessor keep
        # separate modality-config copies.
        processor.modality_configs[
            tag_value
        ] = custom_modality_cfg

        processor.state_action_processor.modality_configs[
            tag_value
        ] = custom_modality_cfg

        logger.info(
            "Overrode processor modality config "
            "for %s from %s",
            tag_value,
            modality_config_path,
        )

    dataset = Gr00tN1d7SingleStepDataset(
        dataset_path=str(dataset_path),
        processor=processor,
        embodiment_tag=embodiment_tag,
        seed=int(
            model_cfg.get(
                "seed",
                cfg.actor.get("seed", 0),
            )
        )
        + rank,
        video_backend=str(
            data_cfg.get(
                "video_backend",
                "torchcodec",
            )
        ),
        samples_per_epoch=data_cfg.get(
            "samples_per_epoch",
            None,
        ),
    )

    # RoboCasa needs dataset-specific normalization
    # statistics instead of the checkpoint's original
    # embodiment statistics.
    if modality_config_path:
        dataset_stats = (
            dataset.loader.get_dataset_statistics()
        )

        processor.set_statistics(
            {
                embodiment_tag.value:
                dataset_stats
            },
            override=True,
        )

        logger.info(
            "Loaded dataset statistics for %s "
            "from %s",
            embodiment_tag.value,
            dataset_path,
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )

    with redirect_qwen3_backbone_to_local(
        processor.model_name,
        backbone_model_path,
    ):
        collator = (
            Gr00tN1d7SingleStepCollator(
                model_name=processor.model_name,
                transformers_loading_kwargs=getattr(
                    processor,
                    "transformers_loading_kwargs",
                    {
                        "trust_remote_code":
                        True
                    },
                ),
            )
        )

    data_loader = StatefulDataLoader(
        dataset,
        batch_size=int(
            cfg.actor.micro_batch_size
        ),
        sampler=sampler,
        num_workers=int(
            data_cfg.get(
                "num_workers",
                4,
            )
        ),
        pin_memory=True,
        drop_last=not eval_dataset,
        collate_fn=collator,
        prefetch_factor=(
            int(
                data_cfg.get(
                    "prefetch_factor",
                    2,
                )
            )
            if int(
                data_cfg.get(
                    "num_workers",
                    4,
                )
            )
            > 0
            else None
        ),
    )

    logger.info(
        "GR00T N1.7 pure SFT dataset: "
        "%d single-step samples/epoch, "
        "action_horizon=%d, from %s",
        len(dataset),
        dataset.action_horizon,
        dataset_path,
    )

    return data_loader, {
        "num_samples": len(dataset)
    }
