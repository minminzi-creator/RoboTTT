# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Configuration and checkpoint contracts shared by native SFT and rollout."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def load_sft_config(
    config: str | Path, overrides: list[str] | None = None
) -> DictConfig:
    """Compose an SFT YAML for offline audits using the training Hydra defaults.

    Accept a file path or a name under examples/sft/config. Call outside an
    active Hydra application; train/eval entrypoints already compose their cfg.
    """
    from hydra import compose, initialize_config_dir

    path = Path(config).expanduser()
    if not path.is_file() and path.parent == Path("."):
        path = Path(__file__).resolve().parents[5] / "examples/sft/config" / path
    if not path.is_file():
        raise FileNotFoundError(path)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(path.resolve().parent)
    ):
        cfg = compose(config_name=path.stem, overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


def plain(config):
    return (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else config
    )


def model_settings(config) -> dict:
    """Resolve RLinf model fields without mutating the user configuration."""
    config = plain(config)
    head = config["rl_head_config"]
    ttt = dict(head["ttt"])
    registers = ttt.pop("num_register_tokens")
    return {
        "model": {
            "base_path": config["model_path"],
            "backbone_path": config["backbone_model_path"],
            "use_ttt": head["use_ttt"],
            "registers": registers,
            "gradient_checkpointing": config.get("gradient_checkpointing", True),
            "disable_dropout": head["disable_dropout"],
            "ttt": ttt,
        },
        "inference": {
            "denoise_steps": config["denoising_steps"],
            "execute_horizon": config["num_action_chunks"],
            "memory_write_source": config["memory_write_source"],
            "memory_write_denoise_index": config["memory_write_denoise_index"],
        },
    }


def dataset_settings(config) -> dict:
    """Adapt the native SFT data schema to the deterministic sequence dataset."""
    config = plain(config)
    result = model_settings(config["actor"]["model"])
    result["seed"] = config["actor"]["seed"]
    result["data"] = dict(config["data"]["robottt"])
    result["data"].update(
        path=config["data"]["train_data_paths"],
        video_backend=config["data"]["video_backend"],
    )
    return result


def configure_runtime(cache_root: str, component: str, rank: int, threads: int = 4):
    """Separate compiler writers and keep inner FP32 products free of TF32."""
    cache = Path(cache_root) / component / str(rank)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")


def source_metadata() -> dict:
    """Version the actual model, dataset, worker and FSDP integration source."""
    root = Path(__file__).resolve().parents[5]
    names = [
        "rlinf/models/embodiment/gr00t/gr00t_n1d7/" + name
        for name in (
            "ttt.py",
            "flow.py",
            "ttt_dit.py",
            "sequence_utils.py",
            "gr00t_action_model.py",
            "__init__.py",
        )
    ] + [
        "rlinf/data/datasets/gr00t_n1d7/sequence_dataset.py",
        "rlinf/data/datasets/gr00t_n1d7/dataloader.py",
        "rlinf/workers/sft/fsdp_sft_worker.py",
        "rlinf/workers/sft/fsdp_vla_sft_worker.py",
        "rlinf/workers/sft/fsdp_vla_robottt_worker.py",
        "examples/sft/train_vla_sft.py",
        "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
        "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
        "rlinf/hybrid_engines/fsdp/utils.py",
    ]
    return {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
        },
        "torch": torch.__version__,
    }


def checkpoint_bundle(checkpoint: str | Path) -> tuple[Path, dict]:
    """Locate portable metadata for native RLinf or archived acceptance weights."""
    path = Path(checkpoint)
    directory = path if path.is_dir() else path.parent
    for candidate in (directory, directory.parent):
        bundle = candidate / "processor_bundle.json"
        if not bundle.is_file():
            continue
        metadata = candidate / "sequence_metadata.json"
        if metadata.is_file():
            return bundle, json.loads(metadata.read_text())
        legacy = candidate / "metadata.json"
        if legacy.is_file():
            return bundle, json.loads(legacy.read_text())
    raise FileNotFoundError(f"No sequence metadata / processor bundle beside {path}")
