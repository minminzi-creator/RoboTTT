# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Environment-owned, transactional memory for native TTT inference."""

from collections.abc import Hashable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .ttt import FastState, Memory


def _snapshot(memory: Memory) -> Memory:
    return Memory(
        tuple(
            FastState(*(x.detach().clone() for x in layer)) for layer in memory.layers
        ),
        memory.position.detach().clone(),
    )


def _identities(env_ids: Sequence[Hashable] | Tensor) -> tuple[Hashable, ...]:
    if isinstance(env_ids, Tensor):
        if env_ids.ndim != 1:
            raise ValueError("Expected one-dimensional environment IDs")
        env_ids = env_ids.tolist()
    if isinstance(env_ids, (str, bytes)):
        raise TypeError("Expected a sequence of environment IDs, not a string")
    ids = tuple(env_ids)
    if any(not isinstance(value, Hashable) for value in ids):
        raise TypeError("Environment IDs must be hashable")
    if len(set(ids)) != len(ids):
        raise ValueError("Environment IDs must be unique within a batch")
    return ids


def _validate_result(
    result: dict[str, Any], size: int, expected: Tensor, previous: Memory | None
) -> Memory:
    if not isinstance(result, dict):
        raise TypeError("Native prediction must return an actions/memory dictionary")
    actions = result.get("actions")
    if not isinstance(actions, Tensor) or actions.ndim < 2 or actions.shape[0] != size:
        raise ValueError("Predicted actions must have one row per environment")
    if not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("Predicted actions must be finite floating-point values")
    memory = result.get("memory")
    if not isinstance(memory, Memory) or not memory.layers:
        raise TypeError("NativeTTTRollout requires a nonempty TTT Memory candidate")
    position = memory.position
    if (
        not isinstance(position, Tensor)
        or position.shape != (size,)
        or position.dtype != torch.long
    ):
        raise ValueError("Memory position must be an int64 vector, one per environment")
    if not torch.equal(position, expected.to(position.device)):
        raise ValueError("Memory position must advance exactly once per prediction")
    if previous is not None and len(memory.layers) != len(previous.layers):
        raise ValueError("Memory layer count changed during rollout")
    for index, layer in enumerate(memory.layers):
        if not isinstance(layer, FastState):
            raise TypeError("Each memory layer must be a FastState")
        for part, value in enumerate(layer):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 4
                or value.shape[0] != size
                or value.numel() == 0
            ):
                raise ValueError("Fast memory must contain nonempty [B,H,D,D] tensors")
            if value.dtype != torch.float32:
                raise TypeError("Fast memory must retain FP32 precision")
            if not torch.isfinite(value).all():
                raise ValueError("Fast memory must contain only finite values")
            if value.device != position.device:
                raise ValueError("Fast memory and position must share a device")
            if (
                previous is not None
                and value.shape != previous.layers[index][part].shape
            ):
                raise ValueError("Fast memory tensor shape changed during rollout")
    return memory


class NativeTTTRollout:
    """Carry detached native memory by stable environment ID.

    Calls accept the same active ID set in any row order. Call ``reset()`` before
    changing that set. Selective resets are deferred to the model, where learned
    initial weights are available inside the fast module. A failed prediction
    preserves the previous memory, ID order, and pending selective resets.

    Args:
        model: Policy exposing ``predict_with_memory`` and returning TTT Memory.
        mode: ``carry`` preserves history; ``reset`` starts every chunk fresh
            while retaining adaptation within that chunk.
    """

    def __init__(self, model: nn.Module, mode: str = "carry") -> None:
        if mode not in {"carry", "reset"}:
            raise ValueError("Native rollout mode must be 'carry' or 'reset'")
        if not callable(getattr(model, "predict_with_memory", None)):
            raise TypeError("Model must expose predict_with_memory")
        self.model = model
        self._mode = mode
        self._memory: Memory | None = None
        self._env_ids: tuple[Hashable, ...] | None = None
        self._pending_resets: set[Hashable] = set()

    def reset(self, env_ids: Sequence[Hashable] | Tensor | None = None) -> None:
        """Clear all history and IDs, or schedule named environments to restart.

        Args:
            env_ids: Active IDs to reset before their next successful prediction.
                ``None`` releases all history and permits a new active ID set.
        """
        if env_ids is None:
            self._memory = None
            self._env_ids = None
            self._pending_resets.clear()
            return
        ids = _identities(env_ids)
        active = set(self._env_ids or ())
        if any(env_id not in active for env_id in ids):
            raise ValueError("Cannot reset an environment outside the active ID set")
        self._pending_resets.update(ids)

    def predict(
        self,
        data: Any,
        env_ids: Sequence[Hashable] | Tensor,
        reset_mask: Tensor | None = None,
        policy_noise: Tensor | None = None,
    ) -> dict[str, Any]:
        """Predict one chunk and atomically commit one validated memory candidate.

        Args:
            data: Native policy inputs, with rows matching ``env_ids``.
            env_ids: Unique stable IDs for this batch, including after reordering.
            reset_mask: Optional boolean vector in the current row order.
            policy_noise: Optional fixed model-space noise passed to the policy.

        Returns:
            Policy outputs with detached actions and caller-owned memory. TTT
            memory must be finite FP32 and advance exactly one anchor per row.
        """
        ids = _identities(env_ids)
        size = len(ids)
        if not size:
            raise ValueError("Native rollout requires at least one environment")
        if self._env_ids is not None and set(ids) != set(self._env_ids):
            raise ValueError("Active environment IDs changed; call reset() first")
        if reset_mask is not None and (
            not isinstance(reset_mask, Tensor)
            or reset_mask.dtype != torch.bool
            or reset_mask.shape != (size,)
        ):
            raise ValueError("reset_mask must be a boolean vector, one per environment")

        incoming = None
        device = torch.device("cpu")
        if self._memory is not None:
            device = self._memory.position.device
            old_rows = {env_id: index for index, env_id in enumerate(self._env_ids)}
            indices = torch.tensor([old_rows[env_id] for env_id in ids], device=device)
            # index_select copies tensors so a failing policy cannot edit our commit.
            incoming = self._memory.reorder(indices)
        reset = torch.tensor(
            [self._mode == "reset" or env_id in self._pending_resets for env_id in ids],
            dtype=torch.bool,
            device=device,
        )
        if reset_mask is not None:
            reset = reset | reset_mask.to(device)
        position = (
            torch.zeros(size, dtype=torch.long, device=device)
            if incoming is None
            else incoming.position.clone()
        )
        expected = torch.where(reset, 0, position) + 1
        with torch.no_grad():
            result = self.model.predict_with_memory(
                data,
                memory=incoming,
                policy_noise=policy_noise,
                reset_mask=reset if reset.any() else None,
            )
        candidate = _validate_result(result, size, expected, self._memory)
        owned = _snapshot(candidate)
        outputs = {
            **result,
            "actions": result["actions"].detach(),
            "memory": candidate.detach(),
        }
        self._memory, self._env_ids = owned, ids
        self._pending_resets.clear()
        return outputs
