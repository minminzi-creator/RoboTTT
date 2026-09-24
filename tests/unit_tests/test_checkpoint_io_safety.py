# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.strategy import base as strategy_base
from rlinf.runners.sft_runner import SFTRunner


class _CheckpointHandle:
    def __init__(self, actor_path: str, fail: bool):
        self.actor_path = Path(actor_path)
        self.fail = fail

    def wait(self):
        self.actor_path.mkdir(parents=True, exist_ok=True)
        (self.actor_path / "partial.bin").write_bytes(b"partial")
        if self.fail:
            raise RuntimeError("simulated checkpoint write failure")


class _CheckpointActor:
    def __init__(self, fail: bool):
        self.fail = fail
        self.actor_path = None

    def save_checkpoint(self, actor_path: str, _step: int):
        self.actor_path = Path(actor_path)
        return _CheckpointHandle(actor_path, self.fail)


def _runner(tmp_path: Path, *, fail: bool) -> SFTRunner:
    runner = object.__new__(SFTRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "checkpoint-test",
                }
            }
        }
    )
    runner.global_step = 3000
    runner.actor = _CheckpointActor(fail)
    return runner


def test_sft_checkpoint_failure_does_not_publish_or_leave_partial_directory(tmp_path):
    runner = _runner(tmp_path, fail=True)

    with pytest.raises(RuntimeError, match="simulated checkpoint write failure"):
        runner._save_checkpoint()

    checkpoint_root = tmp_path / "checkpoint-test" / "checkpoints"
    assert not (checkpoint_root / "global_step_3000").exists()
    assert list(checkpoint_root.glob(".global_step_3000.incomplete-*")) == []


def test_sft_checkpoint_is_published_only_after_worker_save_completes(tmp_path):
    runner = _runner(tmp_path, fail=False)

    runner._save_checkpoint()

    checkpoint_root = tmp_path / "checkpoint-test" / "checkpoints"
    final_dir = checkpoint_root / "global_step_3000"
    assert runner.actor.actor_path != final_dir / "actor"
    assert runner.actor.actor_path.parent.name.startswith(
        ".global_step_3000.incomplete-"
    )
    assert (final_dir / "actor" / "partial.bin").read_bytes() == b"partial"
    assert (final_dir / "_SUCCESS").is_file()
    assert list(checkpoint_root.glob(".global_step_3000.incomplete-*")) == []


def test_full_weight_save_retries_copy_and_never_publishes_partial_file(
    tmp_path, monkeypatch
):
    save = getattr(strategy_base.FSDPStrategyBase, "_save_full_model_weights", None)
    assert save is not None, "full-weight atomic staging helper is missing"

    staging_dir = tmp_path / "local-staging"
    destination = tmp_path / "shared" / "full_weights.pt"
    real_copyfile = strategy_base.shutil.copyfile
    attempts = 0

    def flaky_copyfile(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(target).write_bytes(b"truncated")
            raise OSError("simulated transient Lustre write failure")
        return real_copyfile(source, target)

    monkeypatch.setattr(strategy_base.shutil, "copyfile", flaky_copyfile)
    save(
        {"weight": torch.arange(8)},
        str(destination),
        staging_dir=str(staging_dir),
        io_retries=2,
        retry_delay_s=0,
    )

    assert attempts == 2
    assert torch.equal(
        torch.load(destination, weights_only=True)["weight"], torch.arange(8)
    )
    assert list(destination.parent.glob(".full_weights.pt.incomplete-*")) == []
    assert list(staging_dir.iterdir()) == []
