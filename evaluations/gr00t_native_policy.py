# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native checkpoint, processor and episode-owned TTT evaluation boundary."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def create_seeded_env(factory, scene_seed):
    """Seed fixture construction inside the simulator subprocess, not the policy."""
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    return factory()


def capture_scene_identity(sim):
    """Fingerprint integration state, compiled model, and world camera poses."""
    import mujoco

    def digest_arrays(values):
        digest = hashlib.sha256()
        for name, value in values:
            value = np.ascontiguousarray(value)
            digest.update(str((name, value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    model, data = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite simulator integration state")
    model_arrays = []
    for name in sorted(dir(model)):
        if not name.startswith("_"):
            value = getattr(model, name)
            if isinstance(value, np.ndarray):
                model_arrays.append((name, value))
    return {
        "version": 1,
        "integration_state_sha256": digest_arrays([("integration", state)]),
        "model_xml_sha256": hashlib.sha256(sim.model.get_xml().encode()).hexdigest(),
        "model_arrays_sha256": digest_arrays(model_arrays),
        "camera_sha256": digest_arrays(
            [("cam_xpos", data.cam_xpos), ("cam_xmat", data.cam_xmat)]
        ),
    }


def create_audited_env(factory, scene_seed):
    """Attach read-only scene evidence to each fresh simulator reset."""
    env = create_seeded_env(factory, scene_seed)
    original_reset = env.reset

    def audited_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        obs = result[0] if isinstance(result, (tuple, list)) else result
        obs["native_scene_identity"] = json.dumps(
            capture_scene_identity(env.sim), sort_keys=True
        )
        return result

    env.reset = audited_reset
    return env


def compare_paired_observations(current, reference):
    """Require exact physical identity; permit only bounded RGB rounding noise."""
    identities = []
    for obs in (current, reference):
        value = as_numpy(obs["native_scene_identity"])
        if value.shape != (1,):
            raise ValueError("Expected one simulator identity per worker")
        identity = json.loads(str(value[0]))
        keys = {
            "integration_state_sha256",
            "model_xml_sha256",
            "model_arrays_sha256",
            "camera_sha256",
        }
        if identity.get("version") != 1 or set(identity) != keys | {"version"}:
            raise ValueError("Missing or unsupported simulator identity")
        for key in keys:
            digest = identity[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Invalid simulator fingerprint: {key}")
        identities.append(identity)
    if identities[0] != identities[1]:
        raise ValueError("Scene pairing failed: physical scene identity differs")
    for key in ("states", "task_descriptions"):
        a, b = as_numpy(current[key]), as_numpy(reference[key])
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError(f"Scene pairing failed: {key} differs")
        if key == "states" and not np.isfinite(a).all():
            raise ValueError("Nonfinite robot state")
    differences = {}
    for key in ("main_images", "wrist_images", "extra_view_images"):
        a, b = as_numpy(current[key]), as_numpy(reference[key])
        shape = (1, 1, 256, 256, 3) if key == "extra_view_images" else (1, 256, 256, 3)
        if (
            a.shape != shape
            or b.shape != shape
            or a.dtype != np.uint8
            or b.dtype != np.uint8
        ):
            raise ValueError(f"Expected uint8 camera shape {shape}: {key}")
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        count, maximum = int(np.count_nonzero(delta)), int(delta.max())
        if count > 64 or maximum > 1:
            raise ValueError(
                f"Scene pairing failed: {key} has {count} changed channels, "
                f"max difference {maximum}; limits are 64 and 1"
            )
        differences[key] = {"changed_channels": count, "max_abs": maximum}
    return {
        "protocol": "physical_exact_rgb_1_64_v1",
        "physical_identity_equal": True,
        "scene_identity": identities[0],
        "exact_observation_equal": observation_hash(current)
        == observation_hash(reference),
        "image_differences": differences,
    }


def record_initial_observation(
    output,
    rank,
    env_obs,
    scene_seed,
    reference,
    allow_render_tolerance=False,
    reference_rank=None,
):
    """Preserve reset evidence even when strict scene pairing fails."""
    arrays = {
        key: as_numpy(env_obs[key])
        for key in (
            "main_images",
            "wrist_images",
            "extra_view_images",
            "states",
            "task_descriptions",
        )
    }
    if "native_scene_identity" in env_obs:
        arrays["native_scene_identity"] = as_numpy(env_obs["native_scene_identity"])
    digest = observation_hash(arrays)
    with (Path(output) / f"reset_rank{rank}.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    if reference:
        reference_rank = rank if reference_rank is None else reference_rank
        previous = json.loads(
            (Path(reference) / f"trial_audit_rank{reference_rank}.json").read_text()
        )
        if previous["scene_seed"] != scene_seed or (
            not allow_render_tolerance
            and previous["initial_observation_sha256"] != digest
        ):
            raise ValueError(
                f"Scene pairing failed for seed {scene_seed}: expected "
                f"{previous['initial_observation_sha256']}, got {digest}; "
                f"reset evidence saved in {output}"
            )
        if allow_render_tolerance:
            with np.load(Path(reference) / f"reset_rank{reference_rank}.npz") as saved:
                if observation_hash(saved) != previous["initial_observation_sha256"]:
                    raise ValueError("Reference reset file disagrees with its audit")
                compare_paired_observations(arrays, saved)
    elif allow_render_tolerance:
        compare_paired_observations(arrays, arrays)
    return digest


def observation_hash(observation: dict) -> str:
    digest = hashlib.sha256()
    for key in (
        "main_images",
        "wrist_images",
        "extra_view_images",
        "states",
        "task_descriptions",
    ):
        value = np.ascontiguousarray(as_numpy(observation[key]))
        if value.dtype.hasobject:
            raise TypeError(f"Object dtype is not allowed for {key}")
        digest.update(str((key, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def policy_seed(scene_seed: int, trial_id: int) -> int:
    if not 0 <= scene_seed < 1000000 or trial_id not in (0, 1, 2):
        raise ValueError("Expected scene seed in [0,1000000) and trial 0, 1 or 2")
    return scene_seed + 1000000 * trial_id


def reference_for_scene(root: Path, scene: int, width: int) -> tuple[Path, int]:
    """Map a scene to either preserved four-worker or current eight-worker evidence."""
    if width not in (4, 8) or scene not in range(7, 15):
        raise ValueError("Expected scenes 7..14 and reference width four or eight")
    start = 7 + ((scene - 7) // width) * width
    return Path(root) / f"seeds_{start}_{start + width - 1}", scene - start


def convert_observation(env_obs: dict) -> dict:
    states = as_numpy(env_obs["states"])
    if states.ndim != 2 or states.shape[1] != 16 or not np.isfinite(states).all():
        raise ValueError(f"Expected finite [B,16] states, got {states.shape}")
    extra = as_numpy(env_obs["extra_view_images"])
    right = extra[:, 0] if extra.ndim == 5 else extra
    cameras = (
        as_numpy(env_obs["main_images"]),
        right,
        as_numpy(env_obs["wrist_images"]),
    )
    names = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
    result = {}
    for name, camera in zip(names, cameras):
        if camera.shape != (len(states), 256, 256, 3) or camera.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 [B,256,256,3] {name}, got {camera.shape}/{camera.dtype}"
            )
        result[f"video.{name}"] = camera[:, None]
    for name, start, end in (
        ("base_position", 0, 3),
        ("base_rotation", 3, 7),
        ("end_effector_position_relative", 7, 10),
        ("end_effector_rotation_relative", 10, 14),
        ("gripper_qpos", 14, 16),
    ):
        result[f"state.{name}"] = states[:, None, start:end].astype(
            np.float32, copy=False
        )
    result["annotation.human.task_description"] = list(env_obs["task_descriptions"])
    return result


def convert_action(decoded: dict, chunks: int, threshold: float) -> np.ndarray:
    if not 1 <= chunks <= 40:
        raise ValueError("Execution chunk must be in [1,40]")
    components = []
    for key, dim in (
        ("end_effector_position", 3),
        ("end_effector_rotation", 3),
        ("gripper_close", 1),
        ("base_motion", 4),
        ("control_mode", 1),
    ):
        value = np.asarray(decoded[key])
        if value.ndim != 3 or value.shape[1:] != (40, dim):
            raise ValueError(f"Expected [B,40,{dim}] for {key}, got {value.shape}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite decoded action: {key}")
        value = value[:, :chunks]
        if key in ("gripper_close", "control_mode"):
            value = np.where(value < threshold, -1.0, 1.0)
        components.append(value)
    return np.concatenate(components, axis=-1).astype(np.float32)


def check_checkpoint(weights_path) -> dict:
    weights = Path(weights_path).resolve()
    actor = weights.parent.parent
    if weights != actor / "model_state_dict/full_weights.pt":
        raise ValueError("Expected actor/model_state_dict/full_weights.pt checkpoint")
    marker = json.loads((actor.parent / "_SUCCESS").read_text())
    if actor.parent.name != f"global_step_{marker['step']}":
        raise ValueError("Checkpoint directory and completion step disagree")
    required = {
        "model_state_dict/full_weights.pt",
        "processor_bundle.json",
        "sequence_metadata.json",
        "sequence_progress.pt",
        "data.pt",
        "rng.pt",
        "dcp_checkpoint/.metadata",
    }
    if not required <= marker["files"].keys() or not any(
        name.startswith("dcp_checkpoint/") and name.endswith(".distcp")
        for name in marker["files"]
    ):
        raise ValueError(
            "Completion manifest is missing required checkpoint components"
        )
    for name, size in marker["files"].items():
        path = (actor / name).resolve()
        if (
            not path.is_relative_to(actor)
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise ValueError(f"Incomplete checkpoint component: {name}")
    return marker


def strict_load(model, weights: dict) -> None:
    for name, value in weights.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite checkpoint tensor: {name}")
    model.load_state_dict(weights, strict=True)


def terminal_chunk(step, actions):
    """Execute a one-environment chunk only until its first terminal step."""
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] < 1:
        raise ValueError("Expected a nonempty [1,T,A] action chunk")
    obs, rewards, terms, truncs, infos = [], [], [], [], []
    for index in range(actions.shape[1]):
        o, r, t, u, info = step(actions[:, index], auto_reset=False)
        obs.append(o)
        rewards.append(r)
        terms.append(t)
        truncs.append(u)
        infos.append(info)
        if bool((t | u).any()):
            break
    return (
        obs,
        torch.stack(rewards, 1),
        torch.stack(terms, 1),
        torch.stack(truncs, 1),
        infos,
    )


def validate_processor_bundle(bundle, saved):
    if bundle != saved:
        raise ValueError("Processor bundle disagrees with saved sequence metadata")
    if bundle["embodiment_id_mapping"].get("new_embodiment") != 10:
        raise ValueError("Expected new_embodiment slot 10")


def validate_output_destination(
    checkpoint, output, *, require_fresh=False, allow_queue_log=False
) -> None:
    """Reject absent paths, checkpoint pollution and existing evaluation results."""
    if not checkpoint or not output:
        raise ValueError(
            "Specify runner.ckpt_path checkpoint and runner.logger.log_path output"
        )
    checkpoint_root = Path(checkpoint).resolve().parent.parent.parent
    output = Path(output).resolve()
    if output.is_relative_to(checkpoint_root):
        raise ValueError("Evaluation output must be outside the checkpoint directory")
    if require_fresh and output.exists():
        # The queue reserves a fresh batch directory only to capture child stdout.
        # Empty directories and any previous result/config are not reservations.
        reservation = (
            allow_queue_log
            and output.is_dir()
            and {path.name for path in output.iterdir()} == {"eval.log"}
            and (output / "eval.log").is_file()
            and not (output / "eval.log").is_symlink()
        )
        if not reservation:
            raise FileExistsError(
                "Evaluation output already exists; choose a fresh output path"
            )


def validate_eval_paths(cfg, *, require_fresh=False) -> None:
    """Require explicit destinations before starting workers or creating outputs."""
    validate_output_destination(
        cfg.runner.get("ckpt_path"),
        cfg.runner.logger.get("log_path"),
        require_fresh=require_fresh,
        allow_queue_log=True,
    )


def validate_training_identity(metadata, model_cfg, repo: Path) -> None:
    """Require matching native implementation, normalization and TTT algorithm."""
    from omegaconf import OmegaConf

    if metadata["format"] != "rlinf_native_gr00t_sequence_v1":
        raise ValueError("Expected native GR00T sequence checkpoint")
    repo = Path(repo).resolve()
    for name, expected in metadata["source"]["source_sha256"].items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise ValueError(f"Invalid training source: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training source changed since checkpoint: {name}")
    current = OmegaConf.to_container(model_cfg, resolve=True)
    saved = metadata["resolved_config"]["actor"]["model"]
    for key in (
        "model_path",
        "backbone_model_path",
        "dependency_path",
        "dependency_commit",
        "embodiment_tag",
        "action_dim",
        "valid_action_dim",
        "num_action_chunks",
        "denoising_steps",
        "training_memory_source",
        "rl_head_config",
    ):
        if current.get(key) != saved.get(key):
            raise ValueError(f"Evaluation/training identity mismatch: {key}")
    if (
        current["rl_head_config"]["use_ttt"]
        and current.get("training_memory_source") != "predicted_final"
    ):
        raise ValueError("TTT evaluation requires predicted_final training memory")


def set_inference_precision(model: torch.nn.Module) -> None:
    """Keep the complete fast module FP32 while native parameters compute BF16."""
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import FastWeightTTT

    fast_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, FastWeightTTT)
        for parameter in module.parameters()
    }
    for parameter in model.parameters():
        if id(parameter) in fast_parameters:
            if parameter.dtype != torch.float32:
                raise TypeError(
                    "TTT initial weights must be FP32 before precision setup"
                )
        else:
            parameter.data = parameter.data.to(torch.bfloat16)


class NativeActionAdapter:
    """Convert native model-space predictions to the existing controller protocol."""

    def __init__(
        self,
        model,
        processor,
        embodiment,
        *,
        memory_mode="carry",
        chunks=8,
        threshold=0.0,
    ):
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.rollout import NativeTTTRollout

        if memory_mode not in {"carry", "reset"} or chunks != 8:
            raise ValueError("Expected carry/reset memory and eight executed actions")
        self.model, self.processor, self.embodiment = model, processor, embodiment
        self.chunks, self.threshold = chunks, threshold
        self.rollout = (
            NativeTTTRollout(model, memory_mode)
            if getattr(model, "_native_ttt_enabled", False)
            else None
        )

    def reset(self, env_ids=None) -> None:
        """Release episode memory before a fresh simulator trial."""
        if self.rollout is not None:
            self.rollout.reset(env_ids)

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs, mode="eval", *, env_ids=None, policy_noise=None
    ):
        """Run one complete H40 writer, then decode and execute its first eight actions."""
        if mode != "eval":
            raise ValueError("NativeActionAdapter is evaluation-only")
        observations = convert_observation(env_obs)
        inputs = self.processor.process_observation(observations, self.embodiment)
        position = None
        if self.rollout is not None:
            if env_ids is None:
                raise ValueError(
                    "TTT inference requires explicit stable environment IDs"
                )
            result = self.rollout.predict(inputs, env_ids, policy_noise=policy_noise)
            predicted = result["actions"]
            position = result["memory"].position.cpu().tolist()
        elif getattr(self.model, "_native_ttt_installed", False):
            predicted = self.model.predict_with_memory(
                inputs, policy_noise=policy_noise
            )["actions"]
        else:
            if policy_noise is not None:
                raise ValueError(
                    "Baseline native get_action uses its seeded native RNG"
                )
            predicted = self.model.get_action(inputs)["action_pred"]
        if (
            predicted.shape != (len(env_obs["states"]), 40, 132)
            or not torch.isfinite(predicted).all()
        ):
            raise FloatingPointError(
                "Expected finite [B,40,132] model-space predictions"
            )
        state = {k[6:]: v for k, v in observations.items() if k.startswith("state.")}
        decoded = self.processor.decode_action(
            predicted.float().cpu().numpy(), self.embodiment, state=state
        )
        actions = convert_action(decoded, self.chunks, self.threshold)
        return torch.from_numpy(actions), {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {},
            "memory_position": position,
        }


class NativeEvalPolicy(NativeActionAdapter):
    """Strictly restore a native checkpoint and its saved processor."""

    def __init__(self, model_cfg, weights_path: str, device, *, memory_mode="carry"):
        from gr00t.data.embodiment_tags import EmbodimentTag

        from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import build_processor
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import build_model

        self.marker = check_checkpoint(weights_path)
        actor = Path(weights_path).resolve().parent.parent
        metadata = json.loads((actor / "sequence_metadata.json").read_text())
        validate_training_identity(
            metadata, model_cfg, Path(__file__).resolve().parents[1]
        )
        bundle_path = actor / "processor_bundle.json"
        bundle = json.loads(bundle_path.read_text())
        validate_processor_bundle(bundle, metadata["processor"])
        self.processor, _, _ = build_processor(
            {"model": {"backbone_path": model_cfg.backbone_model_path}},
            str(bundle_path),
        )
        self.processor.eval()
        self.embodiment = EmbodimentTag.NEW_EMBODIMENT
        self.model = build_model(model_cfg, torch.float32)
        weights = torch.load(
            weights_path, map_location="cpu", weights_only=True, mmap=True
        )
        strict_load(self.model, weights)
        del weights
        self.model.requires_grad_(False).eval().to(device)
        set_inference_precision(self.model)
        self.model.action_head.num_inference_timesteps = int(model_cfg.denoising_steps)
        self.chunks = int(model_cfg.num_action_chunks)
        self.threshold = float(model_cfg.binary_action_threshold)
        super().__init__(
            self.model,
            self.processor,
            self.embodiment,
            memory_mode=memory_mode,
            chunks=self.chunks,
            threshold=self.threshold,
        )
        self.identity = {
            "step": self.marker["step"],
            "checkpoint": str(weights_path),
            "strict_load": True,
            "finite_weights": True,
            "processor_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "model_class": type(self.model).__name__,
            "action_head_class": type(self.model.action_head).__name__,
            "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()}),
            "buffer_dtypes": sorted({str(b.dtype) for b in self.model.buffers()}),
            "use_ttt": bool(model_cfg.rl_head_config.use_ttt),
            "num_register_tokens": int(
                model_cfg.rl_head_config.ttt.num_register_tokens
            ),
            "memory_mode": memory_mode,
            "training_memory_source": model_cfg.get("training_memory_source"),
            "extra_amp": False,
            "denoising_steps": int(model_cfg.denoising_steps),
        }
