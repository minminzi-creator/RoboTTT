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
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_RLINF:
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi_rlinf import (
                build_official_openpi_sft_dataloader,
            )

            return build_official_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.GR00T_N1D7:
            from rlinf.data.datasets.gr00t_n1d7 import (
                build_gr00t_n1d7_sft_dataloader,
            )

            return build_gr00t_n1d7_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        # RoboTTT inference bundle:
        # full_weights.pt already stores learned W0 and all trained model params.
        # Here we additionally package the RoboCasa interface metadata required
        # by the inference side.
        if self._rank == 0:
            import json
            import shutil

            inference_dir = os.path.join(save_path, "inference_meta")
            os.makedirs(inference_dir, exist_ok=True)

            dataset_meta_dir = (
                "/mnt/public/chenzm/robottt/datasets/"
                "robocasa365/BowlAndCup/lerobot/lerobot/meta"
            )

            modality_config_path = (
                "/mnt/public/chenzm/robottt/workspace/RLinf-robottt/"
                "examples/sft/config/robocasa365_panda_omron_config.py"
            )

            train_config_path = (
                "/mnt/public/chenzm/robottt/workspace/RLinf-robottt/"
                "examples/sft/config/"
                "robocasa_bowlandcup_sft_gr00t_n1d7_robottt_50k.yaml"
            )

            files_to_copy = {
                os.path.join(dataset_meta_dir, "modality.json"):
                    "modality.json",
                os.path.join(dataset_meta_dir, "embodiment.json"):
                    "embodiment.json",
                os.path.join(dataset_meta_dir, "stats.json"):
                    "stats.json",
                modality_config_path:
                    "robocasa365_panda_omron_config.py",
                train_config_path:
                    "train_config.yaml",
            }

            copied = []
            missing = []

            for src, dst_name in files_to_copy.items():
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(inference_dir, dst_name))
                    copied.append(dst_name)
                else:
                    missing.append(src)

            inference_config = {
                "checkpoint_step": int(step),

                "model_weights":
                    "../model_state_dict/full_weights.pt",

                "embodiment_tag": "new_embodiment",

                "state_dim": 16,
                "action_dim": 12,
                "valid_action_dim": 12,
                "image_nums": 3,

                "num_action_chunks": 8,
                "model_action_horizon": 40,
                "model_max_action_dim": 132,

                "robottt": {
                    "enabled": True,
                    "fast_model": "mlp",

                    "learned_w0_parameters": [
                        "w1_init",
                        "b1_init",
                        "w2_init",
                        "b2_init"
                    ],

                    "query_parameter": "q_proj.weight",
                    "key_parameter": "k_proj.weight",
                    "value_parameter": "v_proj.weight",

                    "inner_lr_parameters": [
                        "ttt_lr_proj.weight",
                        "ttt_lr_proj.bias",
                        "token_idx_bias"
                    ],

                    "gate_parameter": "ttt_gate",

                    "register_parameter":
                        "register_token_bank.tokens"
                },

                "copied_files": copied,
                "missing_files": missing
            }

            with open(
                os.path.join(inference_dir, "inference_config.json"),
                "w",
            ) as f:
                json.dump(inference_config, f, indent=2)

        torch.distributed.barrier()
        # Package RoboCasa / RoboTTT inference metadata.
        if self._rank == 0:
            import json
            import shutil

            inference_dir = os.path.join(save_path, "inference_meta")
            os.makedirs(inference_dir, exist_ok=True)

            dataset_meta_dir = (
                "/mnt/public/chenzm/robottt/datasets/"
                "robocasa365/BowlAndCup/lerobot/lerobot/meta"
            )

            files_to_copy = {
                os.path.join(dataset_meta_dir, "modality.json"):
                    "modality.json",
                os.path.join(dataset_meta_dir, "embodiment.json"):
                    "embodiment.json",
                os.path.join(dataset_meta_dir, "stats.json"):
                    "stats.json",
                (
                    "/mnt/public/chenzm/robottt/workspace/RLinf-robottt/"
                    "examples/sft/config/robocasa365_panda_omron_config.py"
                ):
                    "robocasa365_panda_omron_config.py",
            }

            for src, dst_name in files_to_copy.items():
                if os.path.isfile(src):
                    shutil.copy2(
                        src,
                        os.path.join(inference_dir, dst_name),
                    )

            inference_config = {
                "checkpoint_step": int(step),
                "embodiment_tag": "new_embodiment",
                "state_dim": 16,
                "action_dim": 12,
                "valid_action_dim": 12,
                "image_nums": 3,
                "num_action_chunks": 8,
                "model_action_horizon": 40,
                "model_max_action_dim": 132,
                "robottt_fast_model": "mlp",
                "robottt_w0_parameters": [
                    "w1_init",
                    "b1_init",
                    "w2_init",
                    "b2_init",
                ],
                "model_weights": "../model_state_dict/full_weights.pt",
            }

            with open(
                os.path.join(
                    inference_dir,
                    "inference_config.json",
                ),
                "w",
            ) as f:
                json.dump(
                    inference_config,
                    f,
                    indent=2,
                )

        torch.distributed.barrier()

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in (SupportedModel.OPENPI_RLINF, SupportedModel.OPENPI):
            if model_type == SupportedModel.OPENPI_RLINF:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                    is_official_openpi_sft_dataloader,
                )

                num_batches = (
                    get_official_openpi_sft_num_batches(self.data_loader)
                    if is_official_openpi_sft_dataloader(self.data_loader)
                    else len(self.data_loader)
                )
            else:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                )

                num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
