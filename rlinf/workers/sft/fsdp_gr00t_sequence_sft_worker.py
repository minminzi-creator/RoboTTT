# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native GR00T baseline SFT with deterministic packed episode windows."""

import hashlib
import json
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.utils import seed_everything
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def validate_resume_metadata(previous: dict, current: dict) -> None:
    """Reject resume across data, code, precision or optimization protocols."""
    # Match the checkpoint JSON representation, including integer mapping keys.
    previous = json.loads(json.dumps(previous))
    current = json.loads(json.dumps(current))
    for key in ("world_size", "protocol", "data", "processor", "source", "parameters"):
        if previous[key] != current[key]:
            raise ValueError(f"Strict sequence resume mismatch: {key}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_manifest(manifest: dict, actual_sizes: dict) -> None:
    """Require both inference weights and the complete recorded resume state."""
    required = {
        "model_state_dict/full_weights.pt",
        "data.pt",
        "rng.pt",
        "sequence_metadata.json",
        "processor_bundle.json",
        "sequence_progress.pt",
        "dcp_checkpoint/.metadata",
    }
    files = manifest["files"]
    shards = {
        name
        for name in files
        if name.startswith("dcp_checkpoint/") and name.endswith(".distcp")
    }
    if type(manifest["step"]) is not int or manifest["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if not required.issubset(files) or not shards:
        raise ValueError(
            "Checkpoint must include weights, DCP shards and all resume sidecars"
        )
    for name, size in files.items():
        if type(size) is not int or size <= 0 or actual_sizes.get(name) != size:
            raise ValueError(f"Incomplete checkpoint member: {name}")


class FSDPGr00tSequenceSftWorker(FSDPVlaSftWorker):
    """Own segment backward, full-batch iteration and strict resumable state."""

    def model_provider_func(self):
        seed_everything(self.cfg.actor.seed)
        model = super().model_provider_func()
        self._parameters = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        self._effective_model = {
            "native_config": model.config.to_dict(),
            "state_dropout_prob": model.action_head.state_dropout_prob,
            "module_dropout_values": sorted(
                {
                    module.p
                    for module in model.modules()
                    if isinstance(module, torch.nn.Dropout)
                }
            ),
        }
        return model

    def build_lr_scheduler(self, optimizer, optim_config, last_epoch=-1):
        """Retain the sequence protocol's nonzero first warmup update."""
        if optim_config.get("lr_scheduler") != "linear_warmup_cosine":
            return super().build_lr_scheduler(optimizer, optim_config, last_epoch)
        import math

        warmup = int(optim_config.lr_warmup_steps)
        total = int(optim_config.total_training_steps)
        if warmup < 0 or total <= warmup:
            raise ValueError("Expected total_training_steps > lr_warmup_steps >= 0")

        def multiplier(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return (1 + math.cos(math.pi * min(progress, 1))) / 2

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, multiplier, last_epoch=last_epoch
        )

    def build_dataloader(self, data_paths, eval_dataset=False):
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
            build_gr00t_sequence_dataloader,
        )

        if eval_dataset or data_paths != self.cfg.data.train_data_paths:
            raise ValueError("Sequence SFT requires the configured training dataset")
        return build_gr00t_sequence_dataloader(self.cfg, self._world_size, self._rank)

    def get_max_steps_per_epoch(self):
        # Epochs are tracked by sample identity, not truncated optimizer updates.
        if self.cfg.runner.max_epochs > 0 or self.cfg.runner.max_steps <= 0:
            raise ValueError(
                "Packed sequence SFT requires max_steps > 0 and max_epochs <= 0"
            )
        samples = self.cfg.data.expected_episodes
        if self.cfg.data.robottt.get("sampling_mode") == "independent_h40":
            samples = self.data_config["data"]["sample_count"]
        return max(1, samples // self.global_batch_size)

    def init_worker(self):
        torch.set_num_threads(self.cfg.actor.cpu_threads)
        torch.set_float32_matmul_precision("highest")
        super().init_worker()
        seed_everything(self.cfg.actor.seed + self._rank)
        source = [self.source_metadata() if self._rank == 0 else None]
        torch.distributed.broadcast_object_list(source, src=0)
        self._sequence_metadata = {
            "format": "rlinf_native_gr00t_sequence_v1",
            "world_size": self._world_size,
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
            "protocol": {
                "actor": OmegaConf.to_container(self.cfg.actor, resolve=True),
                "data": OmegaConf.to_container(self.cfg.data, resolve=True),
                "effective_model": self._effective_model,
                "batching": "cross_epoch_full_microbatch",
                "loss": (
                    "masked_action_mean_then_window_mean"
                    if self.cfg.data.robottt.get("sampling_mode") == "independent_h40"
                    else "masked_action_mean_then_full_sequence_mean"
                ),
            },
            "data": self.data_config["data"],
            "processor": self.data_config["processor"],
            "parameters": self._parameters,
            "source": source[0],
        }

    def source_metadata(self) -> dict:
        import importlib.metadata

        import gr00t

        root = Path(__file__).resolve().parents[3]
        dependency = Path(gr00t.__file__).resolve().parent.parent
        names = [
            "examples/sft/train_vla_sft.py",
            "rlinf/data/datasets/gr00t_n1d7/sequence_dataset.py",
            "rlinf/data/datasets/gr00t_n1d7/window_dataset.py",
            "rlinf/data/datasets/gr00t_n1d7/window_sampling.py",
            "rlinf/models/embodiment/gr00t/gr00t_n1d7/sft.py",
            "rlinf/models/embodiment/gr00t/gr00t_n1d7/__init__.py",
            "rlinf/workers/sft/fsdp_gr00t_sequence_sft_worker.py",
            "rlinf/workers/sft/fsdp_sft_worker.py",
            "rlinf/workers/sft/fsdp_vla_sft_worker.py",
            "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
            "rlinf/hybrid_engines/fsdp/strategy/base.py",
            "rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
        ]
        dependency_names = [
            "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
            "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
            "gr00t/model/modules/dit.py",
            "gr00t/model/modules/qwen3_backbone.py",
            "gr00t/data/dataset/lerobot_episode_loader.py",
        ]
        assets = {}
        for label, folder in (
            ("base", self.cfg.actor.model.model_path),
            ("backbone", self.cfg.actor.model.backbone_model_path),
        ):
            directory = Path(folder)
            assets[label] = {
                p.name: sha256_file(p)
                for p in sorted(directory.iterdir())
                if p.is_file()
                and (
                    p.suffix == ".safetensors"
                    or p.name
                    in {
                        "config.json",
                        "model.safetensors.index.json",
                        "processor_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "preprocessor_config.json",
                    }
                )
            }
        return {
            "rlinf_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "gr00t_commit": subprocess.check_output(
                ["git", "-C", str(dependency), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {name: sha256_file(root / name) for name in names},
            "gr00t_sha256": {
                name: sha256_file(dependency / name) for name in dependency_names
            },
            "assets": assets,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "diffusers",
                    "torchdata",
                    "ray",
                    "albumentations",
                )
            },
        }

    def step_batching(self) -> tuple[int, float, int | None]:
        """Return synchronized microbatch count and backward scale for this step."""
        from rlinf.data.datasets.gr00t_n1d7.window_sampling import window_backward_scale

        plan = self.data_config["data"].get("matched_sequence_budget")
        if plan is None:
            return self.gradient_accumulation, 1 / self.gradient_accumulation, None
        windows = plan["windows"][self.global_step]
        capacity = self._world_size * self.cfg.actor.micro_batch_size
        return (
            (windows + capacity - 1) // capacity,
            window_backward_scale(
                windows, self._world_size, self.cfg.actor.micro_batch_size
            ),
            windows,
        )

    def forward_backward_batch(
        self, batch, is_last_micro_batch: bool, backward_scale: float | None = None
    ) -> dict:
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import segment_batch

        if backward_scale is None:
            backward_scale = 1 / self.gradient_accumulation
        b, length = batch["anchor_valid"].shape
        segment_length = self.cfg.actor.model.tbptt_segment_length
        if b != self.cfg.actor.micro_batch_size:
            raise ValueError(
                f"Expected full microbatch {self.cfg.actor.micro_batch_size}, got {b}"
            )
        if segment_length <= 0 or length % segment_length:
            raise ValueError("Sequence length must divide into complete segments")
        total_loss = 0.0
        for start in range(0, length, segment_length):
            end = start + segment_length
            segment = segment_batch(batch, start, end, self.device)
            sync = is_last_micro_batch and end == length
            with self.before_micro_batch(self.model, sync):
                with self.amp_context:
                    output = self.model(forward_type=ForwardType.SFT, data=segment)
                loss = output["loss"]
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        finite, op=torch.distributed.ReduceOp.MIN
                    )
                if not finite.item():
                    raise FloatingPointError(
                        "Non-finite sequence loss; optimizer not stepped"
                    )
                self.grad_scaler.scale(loss * backward_scale).backward()
            total_loss += loss.detach().item()
        return {
            "loss": total_loss,
            "valid_anchors": batch["anchor_valid"].sum().item(),
            "sequence_samples": b,
            "ttt_update_rms": 0.0,
        }

    def run_training(self) -> dict:
        with self.worker_timer():
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            values = []
            samples = []
            microbatches, backward_scale, budget = self.step_batching()
            for index in range(microbatches):
                batch = next(self.data_iter)
                if budget is not None and (
                    batch["budget_step"] != self.global_step + 1
                    or batch["budget_microbatch"] != index
                    or batch["budget_windows"] != budget
                ):
                    raise ValueError("H40 data cursor and optimizer budget disagree")
                values.append(
                    self.forward_backward_batch(
                        batch, index + 1 == microbatches, backward_scale
                    )
                )
                samples.extend(d for d in batch["descriptions"] if d["valid"] > 0)
                self._data_iter_offset += 1
            if budget is not None:
                actual = torch.tensor(
                    sum(x["valid_anchors"] for x in values),
                    device=self.device,
                    dtype=torch.int64,
                )
                torch.distributed.all_reduce(actual)
                if actual.item() != budget:
                    raise ValueError(
                        f"Expected {budget} global windows, got {actual.item()}"
                    )
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = self._strategy.clip_grad_norm_(model=self.model)
            finite = torch.isfinite(torch.as_tensor(grad_norm, device=self.device)).to(
                torch.int32
            )
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(
                    "Non-finite gradient norm; optimizer not stepped"
                )
            lr_used = self.optimizer.param_groups[0]["lr"]
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            metrics = {
                "loss": sum(x["loss"] for x in values) * backward_scale,
                "valid_anchors": sum(x["valid_anchors"] for x in values),
                "global_batch_size": len(samples) * self._world_size,
                "learning_rate": lr_used,
                "next_learning_rate": self.optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "ttt_update_rms": 0.0,
            }
            if budget is not None:
                metrics["budget_windows"] = budget
                metrics["microbatches"] = microbatches
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            directory = Path(self.cfg.runner.logger.log_path)
            with (directory / f"sequence_samples_rank{self._rank}.jsonl").open(
                "a"
            ) as stream:
                stream.write(
                    json.dumps({"step": self.global_step + 1, "samples": samples})
                    + "\n"
                )
            if self._rank == 0:
                with (directory / "sequence_metrics.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({"step": self.global_step + 1, **metrics}) + "\n"
                    )
            return metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        directory = Path(save_path)
        marker = directory.parent / "_SUCCESS"
        if marker.exists():
            raise FileExistsError(
                f"Refusing to overwrite complete checkpoint: {directory.parent}"
            )
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            (directory / "sequence_metadata.json").write_text(
                json.dumps(self._sequence_metadata, indent=2)
            )
            (directory / "processor_bundle.json").write_text(
                json.dumps(self.data_config["processor"], indent=2)
            )
            torch.save(
                {
                    "step": step,
                    "optimizer_steps": self.optimizer_steps,
                    "microbatches_consumed": self._data_iter_offset,
                },
                directory / "sequence_progress.pt",
            )
        torch.distributed.barrier()
        if self._rank == 0:
            sizes = {
                str(path.relative_to(directory)): path.stat().st_size
                for path in directory.rglob("*")
                if path.is_file()
            }
            manifest = {"step": step, "files": sizes}
            validate_checkpoint_manifest(manifest, sizes)
            marker.write_text(json.dumps(manifest, indent=2))
        torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        directory = Path(load_path)
        manifest = json.loads((directory.parent / "_SUCCESS").read_text())
        sizes = {
            str(path.relative_to(directory)): path.stat().st_size
            for path in directory.rglob("*")
            if path.is_file()
        }
        validate_checkpoint_manifest(manifest, sizes)
        previous = json.loads((directory / "sequence_metadata.json").read_text())
        validate_resume_metadata(previous, self._sequence_metadata)
        progress = torch.load(directory / "sequence_progress.pt", weights_only=True)
        if progress["step"] != manifest["step"]:
            raise ValueError("Checkpoint completion step and progress disagree")
        super().load_checkpoint(load_path)
        self.optimizer_steps = progress["optimizer_steps"]
        self.global_step = progress["step"]
        self._data_iter_offset = progress["microbatches_consumed"]
