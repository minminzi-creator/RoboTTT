# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native evaluation boundary tests without simulator or model assets."""

import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import FastWeightTTT, TTTConfig


def api():
    name = "evaluations.gr00t_native_policy"
    assert importlib.util.find_spec(name) is not None, (
        "Native evaluation adapter is missing"
    )
    return importlib.import_module(name)


def observation():
    return {
        "states": np.arange(16, dtype=np.float32)[None],
        "main_images": np.full((1, 256, 256, 3), 11, dtype=np.uint8),
        "wrist_images": np.full((1, 256, 256, 3), 22, dtype=np.uint8),
        "extra_view_images": np.full((1, 1, 256, 256, 3), 33, dtype=np.uint8),
        "task_descriptions": ["put bowl and cup away"],
    }


def test_precision_conversion_preserves_original_fp32_fast_weights():
    fast = FastWeightTTT(TTTConfig(hidden_size=4, num_heads=1, tokens_per_anchor=3))
    model = nn.ModuleDict({"native": nn.Linear(4, 4), "fast": fast})
    before = {name: p.detach().clone() for name, p in fast.named_parameters()}
    api().set_inference_precision(model)
    assert model["native"].weight.dtype == torch.bfloat16
    for name, parameter in fast.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_conversion_keeps_cameras_state_and_controller_order():
    module = api()
    converted = module.convert_observation(observation())
    assert converted["video.robot0_agentview_left"][0, 0, 0, 0, 0] == 11
    assert converted["video.robot0_agentview_right"][0, 0, 0, 0, 0] == 33
    assert converted["video.robot0_eye_in_hand"][0, 0, 0, 0, 0] == 22
    np.testing.assert_array_equal(
        converted["state.end_effector_position_relative"], [[[7, 8, 9]]]
    )
    decoded = {
        key: np.full((1, 40, width), value, dtype=np.float32)
        for key, width, value in (
            ("end_effector_position", 3, 1),
            ("end_effector_rotation", 3, 2),
            ("gripper_close", 1, -0.1),
            ("base_motion", 4, 4),
            ("control_mode", 1, 0),
        )
    }
    actions = module.convert_action(decoded, 8, 0)
    assert actions.shape == (1, 8, 12)
    np.testing.assert_array_equal(actions[0, 0], [1, 1, 1, 2, 2, 2, -1, 4, 4, 4, 4, 1])


def test_terminal_chunk_stops_on_first_success():
    class SimulatorBoundary:
        def __init__(self):
            self.position = 0

        def step(self, action, auto_reset):
            assert auto_reset is False
            self.position += int(action[0, 0])
            return (
                {},
                torch.tensor([1.0]),
                torch.tensor([self.position >= 3]),
                torch.tensor([False]),
                {},
            )

    simulator = SimulatorBoundary()
    result = api().terminal_chunk(simulator.step, torch.ones(1, 8, 12))
    assert simulator.position == 3
    assert result[1].shape == (1, 3)
    assert result[2].tolist() == [[False, False, True]]


def test_strict_load_rejects_missing_or_nonfinite_weights():
    model = nn.Linear(3, 2)
    with pytest.raises(RuntimeError):
        api().strict_load(model, {"weight": model.weight.detach()})
    with pytest.raises(FloatingPointError):
        api().strict_load(
            model,
            {
                "weight": torch.full_like(model.weight, float("nan")),
                "bias": model.bias.detach(),
            },
        )


def test_checkpoint_requires_complete_manifest_and_exact_weights_path(tmp_path):
    checkpoint = tmp_path / "global_step_7"
    actor = checkpoint / "actor"
    names = [
        "model_state_dict/full_weights.pt",
        "data.pt",
        "rng.pt",
        "processor_bundle.json",
        "sequence_metadata.json",
        "sequence_progress.pt",
        "dcp_checkpoint/.metadata",
        "dcp_checkpoint/__0_0.distcp",
    ]
    for name in names:
        file = actor / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test")
    (checkpoint / "_SUCCESS").write_text(
        json.dumps({"step": 7, "files": dict.fromkeys(names, 4)})
    )
    weights = actor / names[0]
    assert api().check_checkpoint(weights)["step"] == 7
    with pytest.raises(ValueError):
        api().check_checkpoint(actor / "model_state_dict/other.pt")
    (actor / names[-1]).unlink()
    with pytest.raises(ValueError):
        api().check_checkpoint(weights)


def test_checkpoint_identity_rejects_changed_ttt_protocol_or_sources(tmp_path):
    source = tmp_path / "sft.py"
    source.write_text("native source")
    import hashlib

    cfg = OmegaConf.create(
        {
            "model_path": "/weights",
            "backbone_model_path": "/backbone",
            "dependency_path": "/dependency",
            "dependency_commit": "commit",
            "embodiment_tag": "new_embodiment",
            "action_dim": 12,
            "valid_action_dim": 12,
            "num_action_chunks": 8,
            "denoising_steps": 4,
            "training_memory_source": "predicted_final",
            "rl_head_config": {
                "use_ttt": True,
                "ttt": {"num_register_tokens": 16, "tokens_per_anchor": 57},
            },
        }
    )
    metadata = {
        "format": "rlinf_native_gr00t_sequence_v1",
        "source": {
            "source_sha256": {"sft.py": hashlib.sha256(source.read_bytes()).hexdigest()}
        },
        "resolved_config": {"actor": {"model": OmegaConf.to_container(cfg)}},
    }
    api().validate_training_identity(metadata, cfg, tmp_path)
    changed = copy.deepcopy(cfg)
    changed.rl_head_config.use_ttt = False
    with pytest.raises(ValueError):
        api().validate_training_identity(metadata, changed, tmp_path)
    source.write_text("changed source")
    with pytest.raises(ValueError):
        api().validate_training_identity(metadata, cfg, tmp_path)


def test_action_adapter_uses_real_native_memory_and_reset():
    if importlib.util.find_spec("gr00t") is None:
        pytest.importorskip(
            "gr00t", reason="Native GR00T SDK is an optional dependency"
        )
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.native_ttt import (
        configure_native_ttt,
        native_ttt_predict,
    )

    torch.manual_seed(84)

    class BackboneBoundary(nn.Module):
        def forward(self, data):
            # The real VLM returns fresh outputs; the native head mutates them.
            return dict(data)

    class TinyNativePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Gr00tN1d7Config(
                hidden_size=8,
                input_embedding_dim=8,
                backbone_embedding_dim=12,
                max_state_dim=3,
                max_action_dim=132,
                action_horizon=40,
                max_num_embodiments=2,
                state_dropout_prob=0,
                diffusion_model_cfg={
                    "num_layers": 2,
                    "num_attention_heads": 2,
                    "attention_head_dim": 4,
                    "norm_type": "ada_norm",
                    "dropout": 0,
                    "final_dropout": False,
                    "output_dim": 8,
                    "interleave_self_attention": True,
                },
            )
            self.action_head = Gr00tN1d7ActionHead(self.config)
            self.backbone = BackboneBoundary()

        def prepare_input(self, data):
            return data, BatchFeature(
                {key: data[key] for key in ("state", "embodiment_id")}
            )

        def predict_with_memory(self, data, **kwargs):
            return native_ttt_predict(self, data, **kwargs)

    model = TinyNativePolicy().eval()
    cfg = OmegaConf.create(
        {
            "gradient_checkpointing": False,
            "training_memory_source": "predicted_final",
            "denoising_steps": 4,
            "rl_head_config": {
                "use_ttt": True,
                "ttt": {
                    "num_register_tokens": 16,
                    "hidden_size": 8,
                    "num_heads": 2,
                    "tokens_per_anchor": 57,
                    "compile_inner": False,
                },
            },
        }
    )
    configure_native_ttt(model, cfg)
    features = {
        "backbone_features": torch.randn(1, 4, 12),
        "backbone_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "image_mask": torch.tensor([[True, True, False, False]]),
        "state": torch.randn(1, 1, 3),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }

    class ProcessorBoundary:
        def process_observation(self, obs, embodiment):
            return features

        def decode_action(self, values, embodiment, state):
            return {
                key: values[:, :, start : start + width]
                for key, start, width in (
                    ("end_effector_position", 0, 3),
                    ("end_effector_rotation", 3, 3),
                    ("gripper_close", 6, 1),
                    ("base_motion", 7, 4),
                    ("control_mode", 11, 1),
                )
            }

    adapter = api().NativeActionAdapter(
        model, ProcessorBoundary(), "robot", memory_mode="carry"
    )
    noise = torch.zeros(1, 40, 132)
    with torch.no_grad():
        complete = model.predict_with_memory(features, policy_noise=noise)
    assert complete["actions"].shape == (1, 40, 132)
    assert complete["ttt_update_rms"] > 0
    expected = api().convert_action(
        ProcessorBoundary().decode_action(
            complete["actions"].detach().numpy(), "robot", {}
        ),
        8,
        0,
    )
    first, info1 = adapter.predict_action_batch(
        observation(), env_ids=[7], policy_noise=noise
    )
    np.testing.assert_array_equal(first.numpy(), expected)
    for owned, full in zip(adapter.rollout._memory.layers, complete["memory"].layers):
        for actual, expected_state in zip(owned, full):
            torch.testing.assert_close(actual, expected_state, rtol=0, atol=0)
    _, info2 = adapter.predict_action_batch(
        observation(), env_ids=[7], policy_noise=noise
    )
    assert info1["memory_position"] == [1] and info2["memory_position"] == [2]
    adapter.reset()
    restarted, info3 = adapter.predict_action_batch(
        observation(), env_ids=[7], policy_noise=noise
    )
    torch.testing.assert_close(first, restarted, rtol=0, atol=0)
    assert first.shape == (1, 8, 12) and info3["memory_position"] == [1]


@pytest.mark.parametrize(
    "reference_options,reference_width",
    [
        ([], 8),
        (["--reference", "/paired-eight-gpu"], 8),
        (["--reference", "/paired-four-gpu", "--reference-width", "4"], 4),
    ],
)
def test_trial_queue_dry_run_never_creates_outputs_or_launches(
    tmp_path, reference_options, reference_width
):
    script = (
        Path(__file__).resolve().parents[2] / "evaluations/run_gr00t_native_trials.py"
    )
    assert script.exists(), "Native evaluation queue is missing"
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output",
            str(output),
            "--dependency",
            "/native",
            "--robocasa",
            "/robocasa",
            "--robosuite",
            "/robosuite",
            "--dry-run",
            *reference_options,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert not output.exists()
    assert plan["environment"]["MUJOCO_GL"] == "osmesa"
    assert plan["environment"]["PYOPENGL_PLATFORM"] == "osmesa"
    formal = [row for row in plan["batches"] if row["phase"] == "formal"]
    assert len(formal) == 3
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    pairs = {(scene, row["trial"]) for row in formal for scene in row["scene_seeds"]}
    assert pairs == {(scene, trial) for scene in range(7, 15) for trial in range(3)}
    assert all("env.eval.max_episode_steps=2400" in row["command"] for row in formal)
    assert all(
        f"rollout.reference_batch_size={reference_width}" in row["command"]
        for row in formal
    )


def test_evaluation_config_requires_explicit_checkpoint_and_output():
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    config = root / "evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml"
    assert config.exists(), "Native evaluation config is missing"
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    with initialize_config_dir(config_dir=str(config.parent), version_base="1.1"):
        cfg = compose(config_name=config.stem)
    assert cfg.runner.ckpt_path is None and cfg.runner.logger.log_path is None
    assert cfg.env.eval.total_num_envs == 8
    assert (
        cfg.rollout.model.num_action_chunks == 8
        and cfg.rollout.model.rl_head_config.use_ttt
    )
    with pytest.raises(ValueError, match="checkpoint|output"):
        api().validate_eval_paths(cfg)


@pytest.mark.parametrize(
    "scene,width,directory,rank",
    [
        (7, 4, "seeds_7_10", 0),
        (10, 4, "seeds_7_10", 3),
        (11, 4, "seeds_11_14", 0),
        (14, 4, "seeds_11_14", 3),
        (14, 8, "seeds_7_14", 7),
    ],
)
def test_scene_identity_maps_old_and_new_reference_workers(
    scene, width, directory, rank
):
    path, reference_rank = api().reference_for_scene(
        Path("/paired/preflight/trial_0"), scene, width
    )
    assert path == Path("/paired/preflight/trial_0") / directory
    assert reference_rank == rank
    assert api().policy_seed(scene, 2) == 2000000 + scene


def test_evaluation_guard_rejects_output_inside_checkpoint():
    cfg = OmegaConf.create(
        {
            "runner": {
                "ckpt_path": "/checkpoints/global_step_7/actor/model_state_dict/full_weights.pt",
                "logger": {"log_path": "/checkpoints/global_step_7/eval"},
            }
        }
    )
    with pytest.raises(ValueError, match="checkpoint"):
        api().validate_eval_paths(cfg)


def test_checkpoint_processor_bundle_must_match_saved_metadata():
    bundle = {"embodiment_id_mapping": {"new_embodiment": 10}, "statistics": {"a": 1}}
    api().validate_processor_bundle(bundle, copy.deepcopy(bundle))
    changed = {**bundle, "statistics": {"a": 2}}
    with pytest.raises(ValueError, match="Processor"):
        api().validate_processor_bundle(changed, bundle)


def test_direct_entry_rejects_existing_results_but_accepts_queue_log_reservation(
    tmp_path,
):
    checkpoint = (
        tmp_path / "checkpoint/global_step_7/actor/model_state_dict/full_weights.pt"
    )
    output = tmp_path / "results"
    cfg = OmegaConf.create(
        {"runner": {"ckpt_path": str(checkpoint), "logger": {"log_path": str(output)}}}
    )
    api().validate_eval_paths(cfg, require_fresh=True)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        api().validate_eval_paths(cfg, require_fresh=True)
    (output / "eval.log").write_text("queue launch log\n")
    api().validate_eval_paths(cfg, require_fresh=True)
    (output / "config.yaml").write_text("old result")
    with pytest.raises(FileExistsError):
        api().validate_eval_paths(cfg, require_fresh=True)


def test_queue_rejects_checkpoint_subtree_before_any_write(tmp_path):
    queue = importlib.import_module("evaluations.run_gr00t_native_trials")
    checkpoint = tmp_path / "global_step_7"
    output = checkpoint / "evaluation"
    with pytest.raises(ValueError, match="checkpoint"):
        queue.execute(
            {
                "checkpoint": str(
                    checkpoint / "actor/model_state_dict/full_weights.pt"
                ),
                "output": str(output),
            }
        )
    assert not checkpoint.exists()


def test_runtime_environment_uses_short_tmp_path_not_experiment_path(tmp_path):
    queue = importlib.import_module("evaluations.run_gr00t_native_trials")
    plan = {
        "output": str(tmp_path / ("long-experiment-name" * 12)),
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    }
    environment = queue.runtime_environment(plan, "/tmp/r3-test1234")
    assert environment["RAY_TMPDIR"] == "/tmp/r3-test1234"
    assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in environment
    assert not Path(plan["output"]).exists()
    with pytest.raises(ValueError, match="socket"):
        queue.runtime_environment(plan, plan["output"])


def test_executable_entry_rejects_existing_results_before_cluster_start(tmp_path):
    entry = importlib.import_module("evaluations.eval_embodied_agent")
    output = tmp_path / "old-result"
    output.mkdir()
    saved = output / "config.yaml"
    saved.write_text("preserve existing result")
    cfg = OmegaConf.create(
        {
            "rollout": {"native_gr00t_eval": True},
            "runner": {
                "ckpt_path": str(
                    tmp_path
                    / "checkpoint/global_step_7/actor/model_state_dict/full_weights.pt"
                ),
                "logger": {"log_path": str(output)},
            },
        }
    )
    with pytest.raises(FileExistsError):
        entry.main.__wrapped__(cfg)
    assert saved.read_text() == "preserve existing result"


@pytest.mark.parametrize(
    "invalid",
    ["scene", "action", "pairing", "checkpoint", "mode", "precision", "episode"],
)
def test_queue_rejects_corrupt_batch_audits(tmp_path, invalid):
    queue = importlib.import_module("evaluations.run_gr00t_native_trials")
    checkpoint = "/checkpoints/global_step_7/actor/model_state_dict/full_weights.pt"
    batch = {
        "output": str(tmp_path),
        "phase": "formal",
        "trial": 2,
        "scene_seeds": list(range(7, 15)),
    }
    for rank, scene in enumerate(batch["scene_seeds"]):
        identity = {"scene_seed": scene, "trial_id": 2, "policy_seed": 2000000 + scene}
        records = {
            "trial_audit": {
                **identity,
                "first_action_shape": [1, 8, 12],
                "first_action_finite": True,
                "pairing": {"physical_identity_equal": True},
            },
            "model": {
                "checkpoint": checkpoint,
                "strict_load": True,
                "finite_weights": True,
                "memory_mode": "carry",
                "use_ttt": True,
                "parameter_dtypes": ["torch.bfloat16", "torch.float32"],
                "buffer_dtypes": ["torch.float32"],
            },
            "episode": {**identity, "preflight": False, "episode_len": 9},
        }
        for name, record in records.items():
            (tmp_path / f"{name}_rank{rank}.json").write_text(json.dumps(record))
    plan = {"checkpoint": checkpoint, "memory_mode": "carry"}
    assert len(queue.check_batch(batch, plan)) == 8
    target, key, value = {
        "scene": ("trial_audit", "policy_seed", 7),
        "action": ("trial_audit", "first_action_finite", False),
        "pairing": ("trial_audit", "pairing", {"physical_identity_equal": False}),
        "checkpoint": ("model", "checkpoint", "/other-checkpoint.pt"),
        "mode": ("model", "memory_mode", "reset"),
        "precision": ("model", "parameter_dtypes", ["torch.bfloat16"]),
        "episode": ("episode", "episode_len", 2401),
    }[invalid]
    path = tmp_path / f"{target}_rank7.json"
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        queue.check_batch(batch, plan)
