# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT worker that runs truncated BPTT through TTT fast weights."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Optional

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    segment_bounds,
    slice_batch_major,
)
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

_SKIP_OUTPUT_KEYS = {"loss", "ttt_context", "action_loss", "action_mask"}
_SUM_METRIC_KEYS = {"ttt_fast_weight_reset"}


def _scalar_metric(value: Any) -> Optional[float]:
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().item())
    if isinstance(value, (float, int, bool)):
        return float(value)
    return None


def _collect_output_metrics(output: Any) -> dict[str, float]:
    if not isinstance(output, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in output.items():
        if key in _SKIP_OUTPUT_KEYS:
            continue
        scalar = _scalar_metric(value)
        if scalar is not None:
            metrics[key] = scalar
    return metrics


def reduce_tbptt_metrics(
    per_segment: list[dict[str, float]],
    segment_losses: list[float],
    bounds: list[tuple[int, int]],
    num_timesteps: int,
) -> dict[str, float]:
    """Collapse per-window SFT stats into one training-step record.

    ``loss`` is the length-weighted trajectory objective.
    """
    metrics: dict[str, float] = {}
    if per_segment:
        keys = set().union(*per_segment)
        for key in keys:
            values = [row[key] for row in per_segment if key in row]
            if not values:
                continue
            if key in _SUM_METRIC_KEYS:
                metrics[key] = sum(values)
            else:
                metrics[key] = sum(values) / len(values)

    weights = [(end - start) / float(num_timesteps) for start, end in bounds]
    metrics["loss"] = sum(
        loss_i * weight for loss_i, weight in zip(segment_losses, weights)
    )
    metrics["tbptt_segments"] = float(len(bounds))
    return metrics


class FSDPRoboTTTSftWorker(FSDPVlaSftWorker):
    """Sequence SFT with truncated backpropagation through TTT state.

    The base SFT worker issues a single ``backward`` per micro-batch. For a
    long trajectory that would keep every TTT inner-loop iterate on the graph.
    This worker splits the ``[B * T]`` batch into segments, backpropagates
    inside each segment, then detaches the fast weights before the next one so
    peak memory tracks segment length rather than ``T``.

    Intermediate segment backwards are wrapped in FSDP ``no_sync`` (or FSDP2
    ``set_requires_gradient_sync(False)``). Only the last segment uses the
    outer ``backward_ctx``, matching PyTorch's "sync on the last backward"
    contract. Segment losses are length-weighted so the total gradient matches
    the paper's :math:`(1/T)\\sum_t \\ell_t`.
    """

    def init_worker(self):
        """Initialize FSDP training and audit TTT-only parameter freezing."""
        super().init_worker()

        ttt_cfg = self.cfg.actor.model.get("rl_head_config", {}).get("ttt", {})
        if not bool(ttt_cfg.get("train_ttt_only", False)):
            return

        trainable_named = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

        trainable_ids = {id(param) for _, param in trainable_named}
        optimizer_params = [
            param
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]
        optimizer_ids = {id(param) for param in optimizer_params}

        allowed_keywords = (
            "ttt_layer",
            "ttt_gate",
            "register_token_bank",
        )

        unexpected = [
            name
            for name, _ in trainable_named
            if not any(key in name for key in allowed_keywords)
        ]

        missing_from_optimizer = [
            name
            for name, param in trainable_named
            if id(param) not in optimizer_ids
        ]

        optimizer_extra_ids = optimizer_ids - trainable_ids

        trainable_numel = sum(param.numel() for _, param in trainable_named)
        optimizer_numel = sum(param.numel() for param in optimizer_params)

        ttt_layer_numel = sum(
            param.numel()
            for name, param in trainable_named
            if "ttt_layer" in name
        )
        gate_numel = sum(
            param.numel()
            for name, param in trainable_named
            if "ttt_gate" in name
        )
        register_numel = sum(
            param.numel()
            for name, param in trainable_named
            if "register_token_bank" in name
        )

        self._logger.info(
            "\n"
            "================ TTT-ONLY AUDIT ================\n"
            f"trainable tensors       : {len(trainable_named)}\n"
            f"trainable params        : {trainable_numel:,}\n"
            f"optimizer tensors       : {len(optimizer_params)}\n"
            f"optimizer params        : {optimizer_numel:,}\n"
            f"optimizer == trainable  : {trainable_ids == optimizer_ids}\n"
            "\n"
            f"TTT layer params        : {ttt_layer_numel:,}\n"
            f"TTT gate params         : {gate_numel:,}\n"
            f"register params         : {register_numel:,}\n"
            "\n"
            f"unexpected tensors      : {len(unexpected)}\n"
            f"missing from optimizer  : {len(missing_from_optimizer)}\n"
            f"optimizer extra tensors : {len(optimizer_extra_ids)}\n"
            "================================================"
        )

        # Print a small sample rather than hundreds of TTT tensors.
        for name, param in trainable_named[:30]:
            self._logger.info(
                "[TTT-ONLY][TRAINABLE] %s shape=%s numel=%d",
                name,
                tuple(param.shape),
                param.numel(),
            )

        if unexpected:
            for name in unexpected[:50]:
                self._logger.error("[TTT-ONLY][UNEXPECTED] %s", name)

        if missing_from_optimizer:
            for name in missing_from_optimizer[:50]:
                self._logger.error(
                    "[TTT-ONLY][MISSING_FROM_OPTIMIZER] %s", name
                )

        assert not unexpected, (
            "TTT-only audit failed: non-TTT parameters are trainable: "
            f"{unexpected[:20]}"
        )
        assert not missing_from_optimizer, (
            "TTT-only audit failed: trainable parameters missing from optimizer: "
            f"{missing_from_optimizer[:20]}"
        )
        assert not optimizer_extra_ids, (
            "TTT-only audit failed: optimizer contains parameters that are not "
            "requires_grad=True"
        )
        assert trainable_ids == optimizer_ids, (
            "TTT-only audit failed: optimizer parameter set does not exactly "
            "match trainable parameter set"
        )

    def _tbptt_segment_length(self, num_timesteps: int) -> int:
        configured = int(self.cfg.actor.model.get("tbptt_segment_length", 0) or 0)
        if configured <= 0:
            return num_timesteps
        return min(configured, num_timesteps)

    @contextmanager
    def _tbptt_no_sync(self) -> Iterator[None]:
        """Disable gradient reduction for an intermediate TBPTT backward."""
        model = self.model
        if hasattr(model, "no_sync"):
            with model.no_sync():
                yield
            return
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                # Last-segment backward is issued by the base worker; restore
                # sync so that path can all-reduce when it is the last micro-batch.
                model.set_requires_gradient_sync(True)
            return
        with nullcontext():
            yield

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        num_timesteps = int(batch.get("num_timesteps", 1) or 1)
        segment_length = self._tbptt_segment_length(num_timesteps)
        bounds = segment_bounds(num_timesteps, segment_length)
        if len(bounds) == 1:
            return super().get_train_model_output(batch)

        ttt_context = None
        last_loss = None
        segment_losses: list[float] = []
        per_segment: list[dict[str, float]] = []
        for index, (start, end) in enumerate(bounds):
            window = end - start
            segment = slice_batch_major(batch, start, end, num_timesteps)
            if ttt_context is not None:
                ttt_context = ttt_context.detached()
                ttt_context.num_timesteps = window
            with self.amp_context:
                output = self.model(
                    forward_type=ForwardType.SFT,
                    data=segment,
                    ttt_context=ttt_context,
                )
            if isinstance(output, torch.Tensor):
                loss = output
            else:
                loss = output["loss"]
                ttt_context = output.get("ttt_context", ttt_context)
            per_segment.append(_collect_output_metrics(output))
            segment_losses.append(float(loss.detach().item()))

            # Segment ``loss`` is already (1/window) sum_{t in seg} ell_t.
            # Weight by window/T so the sum of segment contributions is
            # (1/T) sum_t ell_t. The extra 1/grad_acc matches the base worker.
            scaled = loss * (window / num_timesteps) / self.gradient_accumulation
            is_last_segment = index == len(bounds) - 1
            if is_last_segment:
                last_loss = scaled * self.gradient_accumulation
            else:
                with self._tbptt_no_sync():
                    self.grad_scaler.scale(scaled).backward()

        assert last_loss is not None
        metrics = reduce_tbptt_metrics(
            per_segment, segment_losses, bounds, num_timesteps
        )
        return last_loss, metrics


# ============================================================
# BEGIN ROBOTT_V7B_WORKER_TRACE
# ============================================================

import os as _robottt_v7b_worker_os


if _robottt_v7b_worker_os.environ.get(
    "ROBOTT_V7B_TRACE", "0"
) == "1":

    _robottt_v7b_original_get_output = (
        FSDPRoboTTTSftWorker.get_train_model_output
    )

    def _robottt_v7b_get_train_model_output(
        self,
        batch,
    ):
        # Share optimizer/global step with ttt.py.
        from rlinf.models.embodiment.gr00t.gr00t_n1d7 import (
            ttt as _robottt_v7b_ttt_module,
        )

        _robottt_v7b_ttt_module._robottt_v7b_current_step = int(
            self.global_step
        )

        return _robottt_v7b_original_get_output(
            self,
            batch,
        )

    FSDPRoboTTTSftWorker.get_train_model_output = (
        _robottt_v7b_get_train_model_output
    )


# ============================================================
# END ROBOTT_V7B_WORKER_TRACE
# ============================================================
