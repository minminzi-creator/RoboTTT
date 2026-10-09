# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""GR00T sequence SFT and RoboTTT TBPTT on RLinf's native VLA worker."""

import json
from pathlib import Path
from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class FSDPVlaRoboTTTWorker(FSDPVlaSftWorker):
    """Train matched GR00T/RoboTTT sequences with optional recurrent fast state.

    Only sequence data, segmented forward/backward, and metadata differ from
    the parent. Data iteration, optimizer/clipping/scheduler, FSDP checkpoints,
    StatefulDataLoader snapshots, and per-rank RNG restore stay with RLinf.
    Both comparison groups use this worker so loss weighting and data order
    remain identical when TTT is disabled for the GR00T baseline.
    """

    def model_provider_func(self) -> torch.nn.Module:
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt_dit import parameter_counts
        from rlinf.utils.utils import seed_everything

        seed_everything(self.cfg.actor.seed)
        model = super().model_provider_func()
        self._sequence_parameters = parameter_counts(model)
        return model

    def init_worker(self) -> None:
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
            configure_runtime,
        )

        configure_runtime(
            self.cfg.actor.compiler_cache_dir,
            "sft",
            self._rank,
            self.cfg.actor.cpu_threads,
        )
        super().init_worker()
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
            model_settings,
            plain,
            source_metadata,
        )
        from rlinf.utils.utils import seed_everything

        seed_everything(self.cfg.actor.seed + self._rank)
        self._sequence_metadata = {
            "format": "rlinf_sequence_v1",
            "worker_class": type(self).__name__,
            "resolved_config": plain(self.cfg),
            "world_size": self._world_size,
            "model_settings": model_settings(self.cfg.actor.model),
            "model_config": plain(self.cfg.actor.model),
            "protocol": {
                "actor": plain(self.cfg.actor),
                "data": plain(self.cfg.data),
            },
            "data": self.data_config["data"],
            "processor": self.data_config["processor"],
            "parameters": self._sequence_parameters,
            "source": source_metadata(),
        }
        directory = Path(self.cfg.runner.logger.log_path)
        directory.mkdir(parents=True, exist_ok=True)
        if self._rank == 0:
            (directory / "sequence_metadata.json").write_text(
                json.dumps(self._sequence_metadata, indent=2)
            )

    def forward_backward_batch(
        self, batch: dict[str, Any], is_last_micro_batch: bool
    ) -> dict[str, float]:
        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import segment_batch

        length = batch["anchor_valid"].shape[1]
        segment_length = self.cfg.actor.model.tbptt_segment_length
        if segment_length <= 0 or length % segment_length:
            raise ValueError(
                "Sequence length must be divisible by TBPTT segment length"
            )
        memory, losses, updates = None, [], []
        for start in range(0, length, segment_length):
            end = min(start + segment_length, length)
            segment = segment_batch(batch, start, end, self.device)
            sync = is_last_micro_batch and end == length
            # no_sync covers the forward that installs FSDP's reduction hooks.
            with self.before_micro_batch(self.model, sync):
                with self.amp_context:
                    output = self.model(
                        forward_type=ForwardType.SFT, data=segment, incoming=memory
                    )
                loss = output["loss"]
                finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(
                        "Non-finite sequence loss; stopping without recovery"
                    )
                self.grad_scaler.scale(loss / self.gradient_accumulation).backward()
            memory = output["memory"]
            memory = memory.detach() if memory is not None else None
            losses.append(loss.detach().item())
            updates.append(output["ttt_update_rms"].detach().item())
        metrics = {
            "loss": sum(losses),
            "ttt_update_rms": sum(updates) / len(updates),
            "valid_anchors": batch["anchor_valid"].sum().item(),
            "tbptt_segments": len(losses),
        }
        directory = Path(self.cfg.runner.logger.log_path)
        with (directory / f"sequence_samples_rank{self._rank}.jsonl").open(
            "a"
        ) as output_file:
            output_file.write(
                json.dumps(
                    {"step": self.global_step + 1, "samples": batch["descriptions"]}
                )
                + "\n"
            )
        if self.global_step == 0:
            audit = {
                "parameters": sorted({str(p.dtype) for p in self.model.parameters()}),
                "gradients": sorted(
                    {
                        str(p.grad.dtype)
                        for p in self.model.parameters()
                        if p.grad is not None
                    }
                ),
                "adam_moments": sorted(
                    {
                        str(v.dtype)
                        for state in self.optimizer.state.values()
                        for key, v in state.items()
                        if key in ("exp_avg", "exp_avg_sq")
                    }
                ),
                "fast_state": sorted(
                    {str(p.dtype) for layer in memory.layers for p in layer}
                )
                if memory
                else [],
                "communication": "torch.float32",
            }
            for key in ("parameters", "gradients", "adam_moments"):
                if audit[key] != ["torch.float32"]:
                    raise TypeError(f"Sequence precision contract failed: {audit}")
            if audit["fast_state"] not in ([], ["torch.float32"]):
                raise TypeError(f"Fast memory was cast: {audit}")
            (directory / f"sequence_precision_rank{self._rank}.json").write_text(
                json.dumps(audit, indent=2)
            )
        return metrics

    def run_training(self) -> dict[str, float]:
        metrics = super().run_training()
        if self._rank == 0:
            path = Path(self.cfg.runner.logger.log_path) / "sequence_metrics.jsonl"
            with path.open("a") as output:
                output.write(
                    json.dumps({"step": self.global_step + 1, **metrics}) + "\n"
                )
        return metrics

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        from rlinf.data.datasets.gr00t_n1d7.dataloader import (
            build_gr00t_sequence_dataloader,
        )

        return build_gr00t_sequence_dataloader(
            self.cfg, self._world_size, self._rank, data_paths, eval_dataset
        )

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            directory = Path(save_path)
            (directory / "sequence_metadata.json").write_text(
                json.dumps(self._sequence_metadata, indent=2)
            )
            (directory / "processor_bundle.json").write_text(
                json.dumps(self.data_config["processor"], indent=2)
            )
            torch.save(
                {
                    "data_epoch": self._data_epoch,
                    "data_iter_offset": self._data_iter_offset,
                    "optimizer_steps": self.optimizer_steps,
                    "step": step,
                },
                directory / "sequence_progress.pt",
            )
        torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        directory = Path(load_path)
        previous = json.loads((directory / "sequence_metadata.json").read_text())
        for key in (
            "world_size",
            "protocol",
            "data",
            "processor",
            "parameters",
            "source",
        ):
            if previous[key] != self._sequence_metadata[key]:
                raise ValueError(f"Strict sequence resume mismatch: {key}")
        for name in ("data.pt", "rng.pt", "sequence_progress.pt"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        super().load_checkpoint(load_path)
        progress = torch.load(
            Path(load_path) / "sequence_progress.pt", weights_only=True
        )
        self._data_epoch = progress["data_epoch"]
        self._data_iter_offset = progress["data_iter_offset"]
        self.optimizer_steps = progress["optimizer_steps"]
        self.global_step = progress["step"]
