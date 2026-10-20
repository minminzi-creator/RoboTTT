# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Fresh-scene repeated trials and terminal recording for native GR00T eval."""

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from evaluations.gr00t_native_policy import (
    NativeEvalPolicy,
    as_numpy,
    compare_paired_observations,
    create_audited_env,
    policy_seed,
    record_initial_observation,
    reference_for_scene,
    terminal_chunk,
    validate_eval_paths,
)
from rlinf.utils.utils import seed_everything
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)


def validate_trial(cfg):
    validate_eval_paths(cfg)
    if (
        not cfg.runner.only_eval
        or cfg.env.eval.auto_reset
        or cfg.env.eval.rollout_epoch != 1
    ):
        raise ValueError("Native trials require eval-only, one epoch and no auto-reset")
    if cfg.env.eval.total_num_envs != 8 or cfg.rollout.pipeline_stage_num != 1:
        raise ValueError(
            "Expected eight environments, one per worker, one pipeline stage"
        )
    if (
        cfg.env.eval.seed_strategy != "global_unique"
        or cfg.env.eval.seed != 7
        or list(cfg.rollout.scene_seeds) != list(range(7, 15))
    ):
        raise ValueError("Expected all eight scenes 7..14 with global_unique seeds")
    policy_seed(int(cfg.env.eval.seed), int(cfg.rollout.trial_id))


class NativeEvalRolloutWorker(MultiStepRolloutWorker):
    def init_worker(self):
        validate_trial(self.cfg)
        if (
            self._world_size != 8
            or self.per_node_eval_batch_size != 1
            or self.enable_offload
        ):
            raise ValueError(
                "Expected eight resident rollout workers, one environment each"
            )
        self.scene_seed = int(self.cfg.rollout.scene_seeds[self._rank])
        self.trial = int(self.cfg.rollout.trial_id)
        self.trial_seed = policy_seed(self.scene_seed, self.trial)
        self.output = Path(self.cfg.runner.logger.log_path)
        self.first_prediction = True
        self.hf_model = NativeEvalPolicy(
            self.model_cfg,
            self.cfg.runner.ckpt_path,
            self.device,
            memory_mode=self.cfg.rollout.memory_mode,
        )
        self.hf_model.reset()
        self.setup_sample_params()
        seed_everything(self.trial_seed)
        write_json(self.output / f"model_rank{self._rank}.json", self.hf_model.identity)

    def predict(self, env_obs, mode="eval"):
        if mode != "eval":
            raise ValueError("Native trials cannot train")
        if bool(as_numpy(env_obs.get("native_eval_done", [False])).all()):
            return torch.zeros((1, self.model_cfg.num_action_chunks, 12)), {}
        record = None
        if self.first_prediction:
            reference_root = self.cfg.rollout.get("reference_dir")
            reference, reference_rank = (
                reference_for_scene(
                    Path(reference_root),
                    self.scene_seed,
                    int(self.cfg.rollout.reference_batch_size),
                )
                if reference_root
                else (None, None)
            )
            digest = record_initial_observation(
                self.output,
                self._rank,
                env_obs,
                self.scene_seed,
                reference,
                allow_render_tolerance=True,
                reference_rank=reference_rank,
            )
            if reference:
                with np.load(
                    Path(reference) / f"reset_rank{reference_rank}.npz"
                ) as saved:
                    pairing = compare_paired_observations(env_obs, saved)
            else:
                pairing = compare_paired_observations(env_obs, env_obs)
            record = {
                "scene_seed": self.scene_seed,
                "trial_id": self.trial,
                "policy_seed": self.trial_seed,
                "initial_observation_sha256": digest,
                "pairing": pairing,
                "cpu_rng_sha256": hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()
                ).hexdigest(),
                "device_rng_sha256": hashlib.sha256(
                    self.torch_platform.get_rng_state().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        actions, info = self.hf_model.predict_action_batch(
            env_obs, mode, env_ids=[self.scene_seed]
        )
        if record is not None:
            record.update(
                first_action_sha256=hashlib.sha256(
                    actions.numpy().tobytes()
                ).hexdigest(),
                first_action_shape=list(actions.shape),
                first_action_finite=bool(torch.isfinite(actions).all()),
            )
            write_json(self.output / f"trial_audit_rank{self._rank}.json", record)
            self.first_prediction = False
        return actions, info


class NativeEvalEnvWorker(EnvWorker):
    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env

        validate_trial(self.cfg)
        if env_cls is not Robocasa365Env or num_envs_per_stage != 1:
            raise ValueError("Expected one native RoboCasa365 environment per worker")

        class TerminalEnv(Robocasa365Env):
            def reset(self, *args, **kwargs):
                obs, infos = super().reset(*args, **kwargs)
                if "native_scene_identity" in obs:
                    infos["native_scene_identity"] = obs["native_scene_identity"]
                return obs, infos

            def get_env_fns(self, env_idx=None):
                indices = (
                    np.arange(self.num_envs) if env_idx is None else np.asarray(env_idx)
                )
                return [
                    partial(create_audited_env, factory, int(self.env_seeds[index]))
                    for index, factory in zip(indices, super().get_env_fns(env_idx))
                ]

            def _wrap_obs(self, obs_list, info_list):
                obs = super()._wrap_obs(obs_list, info_list)
                if all("native_scene_identity" in raw for raw in obs_list):
                    obs["native_scene_identity"] = [
                        raw["native_scene_identity"] for raw in obs_list
                    ]
                return obs

            def chunk_step(self, actions):
                # A variable final chunk ends at the exact simulator success step.
                return terminal_chunk(self.step, actions)

        self._native_finished = {}
        return super()._setup_env_and_wrappers(TerminalEnv, env_cfg, num_envs_per_stage)

    def _build_rollout_input_data(self, env_batch):
        data = super()._build_rollout_input_data(env_batch)
        infos = env_batch.get("env_infos") or {}
        if "native_scene_identity" in infos:
            data["obs"]["native_scene_identity"] = infos["native_scene_identity"]
        if self._native_finished:
            data["obs"]["native_eval_done"] = torch.ones(1, dtype=torch.bool)
        return data

    def env_evaluate_step(self, raw_actions, stage_id):
        if stage_id in self._native_finished:
            return self._native_finished[stage_id], {}
        output, metrics = super().env_evaluate_step(raw_actions, stage_id)
        if metrics.get("episode_len") is not None and metrics["episode_len"].numel():
            output.obs["native_eval_done"] = torch.ones(1, dtype=torch.bool)
            self._native_finished[stage_id] = output
            scene = int(self.cfg.rollout.scene_seeds[self._rank])
            record = {
                "scene_seed": scene,
                "trial_id": int(self.cfg.rollout.trial_id),
                "policy_seed": policy_seed(scene, int(self.cfg.rollout.trial_id)),
                "preflight": bool(self.cfg.rollout.preflight),
                "success": bool(metrics["success_once"].item()),
                "episode_len": int(metrics["episode_len"].item()),
            }
            write_json(
                Path(self.cfg.runner.logger.log_path)
                / f"episode_rank{self._rank}.json",
                record,
            )
        return output, metrics
