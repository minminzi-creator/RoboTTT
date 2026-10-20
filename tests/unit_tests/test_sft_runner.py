# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Checkpoint scheduling through the SFT runner's training loop."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.runners.sft_runner import SFTRunner


class _Result:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value

    def consume_duration(self):
        return 0.0


class _Actor:
    """In-process boundary for the remote actor worker group."""

    def get_max_steps_per_epoch(self):
        return _Result([13])

    def run_training(self):
        return _Result([{"loss": 1.0}])

    def save_checkpoint(self, path, step):
        Path(path, "step.txt").write_text(str(step))
        return _Result(None)


def _config(tmp_path, **runner_options):
    return OmegaConf.create(
        {
            "runner": {
                "max_epochs": -1,
                "max_steps": 10000,
                "val_check_interval": -1,
                "save_interval": 1000,
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "schedule",
                    "logger_backends": [],
                },
                **runner_options,
            }
        }
    )


def _saved_steps(tmp_path):
    return sorted(
        int(path.read_text())
        for path in tmp_path.glob("schedule/checkpoints/*/actor/step.txt")
    )


def test_explicit_steps_replace_periodic_checkpoint_saves(tmp_path):
    cfg = _config(tmp_path, save_steps=[3000, 5000, 6000, 8000, 9000, 10000])
    runner = SFTRunner(cfg, _Actor())
    runner.run()
    assert _saved_steps(tmp_path) == [3000, 5000, 6000, 8000, 9000, 10000]
    assert runner.global_step == 10000


def test_explicit_steps_do_not_add_an_unlisted_final_checkpoint(tmp_path):
    runner = SFTRunner(_config(tmp_path, max_steps=4, save_steps=[2]), _Actor())
    runner.run()
    assert _saved_steps(tmp_path) == [2]


@pytest.mark.parametrize("save_steps", [None, []])
def test_default_periodic_and_explicit_empty_schedules(tmp_path, save_steps):
    runner = SFTRunner(
        _config(tmp_path, max_steps=5, save_interval=2, save_steps=save_steps),
        _Actor(),
    )
    runner.run()
    assert _saved_steps(tmp_path) == ([2, 4, 5] if save_steps is None else [])


def test_resumed_loop_only_saves_remaining_explicit_steps(tmp_path):
    runner = SFTRunner(
        _config(
            tmp_path, max_steps=10, save_interval=-1, save_steps=[3, 5, 6, 8, 9, 10]
        ),
        _Actor(),
    )
    runner.global_step = 6
    runner.run()
    assert _saved_steps(tmp_path) == [8, 9, 10]


@pytest.mark.parametrize(
    "save_steps", ["3000", [0], [-1], [10001], [True], [3.5], [3, 3]]
)
def test_invalid_checkpoint_steps_fail_before_training(tmp_path, save_steps):
    with pytest.raises(ValueError, match="save_steps"):
        SFTRunner(_config(tmp_path, save_steps=save_steps), _Actor())


def test_robottt_config_saves_only_the_selected_milestones(tmp_path):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2] / "examples/sft/config"
    with initialize_config_dir(config_dir=str(root), version_base="1.1"):
        cfg = compose(config_name="robocasa_bowlandcup_robottt3")
    cfg.runner.logger.log_path = str(tmp_path)
    cfg.runner.logger.experiment_name = "schedule"
    cfg.runner.logger.logger_backends = []
    runner = SFTRunner(cfg, _Actor())
    runner.run()
    assert _saved_steps(tmp_path) == [3000, 5000, 6000, 8000, 9000, 10000]
