# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Hydra inheritance, native worker routing, and the 8K optimizer schedule."""

import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from examples.sft.train_vla_sft import get_sft_worker_class
from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    dataset_settings,
    load_sft_config,
    source_metadata,
)
from rlinf.workers.sft.fsdp_vla_robottt_worker import FSDPVlaRoboTTTWorker
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/sft/config"
BASELINE = "robocasa_bowlandcup_gr00t_baseline_h100.yaml"
ROBOTTT = "robocasa_bowlandcup_robottt_h100.yaml"


def test_composed_groups_differ_only_in_experiment_name_and_ttt():
    baseline = load_sft_config(CONFIG_DIR / BASELINE)
    robottt = load_sft_config(CONFIG_DIR / ROBOTTT)
    assert baseline.runner.max_steps == robottt.runner.max_steps == 8000
    assert baseline.actor.optim.total_training_steps == 8000
    assert baseline.actor.optim.lr_warmup_steps == 200
    assert baseline.actor.optim.lr == 5e-5
    assert baseline.runner.save_interval == 1000
    assert baseline.actor.model.rl_head_config.use_ttt is False
    assert baseline.actor.model.rl_head_config.ttt.num_register_tokens == 0
    assert robottt.actor.model.rl_head_config.use_ttt is True
    assert robottt.actor.model.rl_head_config.ttt.num_register_tokens == 16
    assert get_sft_worker_class(baseline) is FSDPVlaRoboTTTWorker
    assert get_sft_worker_class(robottt) is FSDPVlaRoboTTTWorker
    assert issubclass(FSDPVlaRoboTTTWorker, FSDPVlaSftWorker)
    # Compare every resolved field, including optimizer, data, and FSDP policy.
    robottt.runner.logger = baseline.runner.logger
    robottt.actor.model.model_type = baseline.actor.model.model_type
    robottt.actor.model.rl_head_config.use_ttt = False
    robottt.actor.model.rl_head_config.ttt.num_register_tokens = 0
    assert OmegaConf.to_container(baseline) == OmegaConf.to_container(robottt)
    config = dataset_settings(baseline)
    assert config["data"]["action_horizon"] == 40
    assert config["data"]["max_anchors"] == 128


def test_model_and_fsdp_changes_are_inherited_by_both_experiments(tmp_path):
    config_dir = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config_dir)
    model_path = config_dir / "model/gr00t_n1d7.yaml"
    model = OmegaConf.load(model_path)
    model.denoising_steps = 7
    OmegaConf.save(model, model_path)
    fsdp_path = config_dir / "hybrid_engines/fsdp.yaml"
    fsdp = OmegaConf.load(fsdp_path)
    fsdp.backward_prefetch = "pre"
    OmegaConf.save(fsdp, fsdp_path)
    for name in (BASELINE, ROBOTTT):
        config = load_sft_config(config_dir / name)
        assert config.actor.model.denoising_steps == 7
        assert config.actor.fsdp_config.backward_prefetch == "pre"
        assert config.actor.fsdp_config.use_orig_params is True
        assert config.actor.fsdp_config.amp_autocast.enabled is True
        assert config.actor.fsdp_config.amp_autocast.precision == "bf16"
        assert set(config.actor.fsdp_config.mixed_precision.values()) == {None}
        assert config.actor.fsdp_config.cast_forward_inputs is False
        assert config.actor.fsdp_config.cast_root_forward_inputs is False


@pytest.mark.parametrize("model_type", ["openpi", "dreamzero", "gr00t_n1d7"])
def test_non_sequence_configs_keep_the_native_vla_worker(model_type):
    cfg = OmegaConf.create({"actor": {"model": {"model_type": model_type}}})
    assert get_sft_worker_class(cfg) is FSDPVlaSftWorker
    cfg.actor.model.sequence_training = False
    assert get_sft_worker_class(cfg) is FSDPVlaSftWorker


def test_unsupported_sequence_model_is_rejected_before_worker_launch():
    cfg = OmegaConf.create(
        {"actor": {"model": {"model_type": "openpi", "sequence_training": True}}}
    )
    with pytest.raises(ValueError, match="Sequence SFT requires"):
        get_sft_worker_class(cfg)


def test_smoke_stop_does_not_change_the_8k_learning_rate_schedule():
    cfg = load_sft_config(ROBOTTT, ["runner.max_steps=3", "runner.save_interval=2"])
    assert cfg.runner.max_steps == 3
    assert cfg.actor.optim.total_training_steps == 8000
    optim_cfg = cfg.actor.optim
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=optim_cfg.lr)
    scheduler = get_lr_scheduler(
        optim_cfg.lr_scheduler,
        optimizer,
        optim_cfg.lr_warmup_steps,
        optim_cfg.total_training_steps,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 / 200)
    schedule = scheduler.lr_lambdas[0]
    assert schedule(199) == schedule(200) == 1.0
    assert schedule(201) < schedule(200)
    assert schedule(4100) == pytest.approx(0.5)
    assert schedule(7999) > 0
    assert schedule(8000) == schedule(9000) == 0
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5 * 4 / 200)


def test_strict_resume_tracks_the_selected_worker_and_entrypoint():
    sources = source_metadata()["source_sha256"]
    assert "rlinf/workers/sft/fsdp_vla_robottt_worker.py" in sources
    assert "rlinf/workers/sft/fsdp_vla_sft_worker.py" in sources
    assert "examples/sft/train_vla_sft.py" in sources


def compose_eval(group, monkeypatch, search_root=None):
    """Mirror the search paths supplied by evaluations/run_eval.sh."""
    from hydra import compose, initialize_config_dir

    repo = CONFIG_DIR.parents[2]
    search_root = repo if search_root is None else search_root
    monkeypatch.setenv("REPO_PATH", str(search_root))
    monkeypatch.setenv("EMBODIED_PATH", str(search_root / "examples/embodiment"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "evaluations/robocasa365")
    ):
        cfg = compose(config_name=f"robocasa365_bowlandcup_{group}_h100")
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize("group", ["gr00t_baseline", "robottt"])
def test_eval_defaults_share_the_exact_sft_model(group, monkeypatch):
    train = load_sft_config(f"robocasa_bowlandcup_{group}_h100.yaml")
    evaluation = compose_eval(group, monkeypatch)
    assert evaluation.rollout.model == train.actor.model
    assert evaluation.env.eval.env_type == "robocasa365"
    assert evaluation.env.eval.total_num_envs == 8
    assert evaluation.env.eval.seed_strategy == "global_unique"
    assert evaluation.env.eval.rollout_epoch == 1
    assert evaluation.env.eval.observation.state_layout == [
        "base_pos",
        "base_quat",
        "base_to_eef_pos",
        "base_to_eef_quat",
        "gripper_qpos",
    ]
    assert evaluation.env.eval.action_space.binarize_gripper_control is False


def test_eval_inherits_model_adapter_and_native_environment(tmp_path, monkeypatch):
    repo = CONFIG_DIR.parents[2]
    sft_config = tmp_path / "examples/sft/config"
    env_config = tmp_path / "examples/embodiment/config/env"
    shutil.copytree(CONFIG_DIR, sft_config)
    shutil.copytree(repo / "examples/embodiment/config/env", env_config)
    for relative, field, value in (
        ("model/gr00t_n1d7.yaml", "denoising_steps", 7),
        ("robottt/bowlandcup_model.yaml", "model_path", "/test/shared-checkpoint"),
    ):
        path = sft_config / relative
        config = OmegaConf.load(path)
        config[field] = value
        OmegaConf.save(config, path)
    path = env_config / "robocasa365.yaml"
    config = OmegaConf.load(path)
    config.robot_name = "InheritedRobot"
    OmegaConf.save(config, path)
    for group in ("gr00t_baseline", "robottt"):
        evaluation = compose_eval(group, monkeypatch, search_root=tmp_path)
        train = load_sft_config(sft_config / f"robocasa_bowlandcup_{group}_h100.yaml")
        assert evaluation.rollout.model == train.actor.model
        assert evaluation.rollout.model.denoising_steps == 7
        assert evaluation.rollout.model.model_path == "/test/shared-checkpoint"
        assert evaluation.env.eval.robot_name == "InheritedRobot"
        assert evaluation.env.eval.init_params.camera_heights == 256
