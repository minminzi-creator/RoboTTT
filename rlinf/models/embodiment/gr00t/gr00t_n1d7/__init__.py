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

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger


def _audit_runtime_model_shapes(model, cfg):
    expected = {
        "action_dim": int(cfg.action_dim),
        "valid_action_dim": int(cfg.valid_action_dim),
        "image_nums": int(cfg.image_nums),
    }
    observed = {
        "model.action_dim": getattr(model, "action_dim", None),
        "model.valid_action_dim": getattr(model, "valid_action_dim", None),
        "model.image_nums": getattr(model, "image_nums", None),
        "action_head.env_action_dim": getattr(
            model.action_head, "env_action_dim", None
        ),
        "action_head.valid_action_dim": getattr(
            model.action_head, "valid_action_dim", None
        ),
    }
    expected_by_observed_name = {
        "model.action_dim": expected["action_dim"],
        "model.valid_action_dim": expected["valid_action_dim"],
        "model.image_nums": expected["image_nums"],
        "action_head.env_action_dim": expected["action_dim"],
        "action_head.valid_action_dim": expected["valid_action_dim"],
    }
    mismatches = [
        f"{name}={observed[name]!r} (expected {wanted})"
        for name, wanted in expected_by_observed_name.items()
        if observed[name] != wanted
    ]
    if mismatches:
        raise ValueError(
            "GR00T runtime shape audit failed: " + "; ".join(mismatches)
        )
    return expected


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
    AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)

    logger = get_logger()
    logger.info(
        "Successfully registered custom architecture Gr00tN1d7, authentication passed!"
    )

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    embodiment_tag_by_cfg = {
        "libero_sim": EmbodimentTag.LIBERO_SIM,  # N1.7 specific change
        "libero_panda": EmbodimentTag.LIBERO_PANDA,
        "libero_franka": EmbodimentTag.LIBERO_FRANKA,
        "isaaclab_franka": EmbodimentTag.ISAACLAB_FRANKA,
        "maniskill_widowx": EmbodimentTag.MANISKILL_WIDOWX,
        "robocasa_panda_omron": EmbodimentTag.ROBOCASA_PANDA_OMRON,
        "gr1": EmbodimentTag.GR1,
        "behavior_r1_pro": EmbodimentTag.BEHAVIOR_R1_PRO,
        "new_embodiment": EmbodimentTag.NEW_EMBODIMENT,
        "so101": EmbodimentTag.NEW_EMBODIMENT,
        "so100": EmbodimentTag.NEW_EMBODIMENT,
    }
    emb_tag = embodiment_tag_by_cfg.get(cfg.embodiment_tag)
    if emb_tag is None:
        raise ValueError(
            f"Invalid or unsupported embodiment tag: {cfg.embodiment_tag}. "
            f"Supported tags are: {list(embodiment_tag_by_cfg.keys())}."
        )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config = Gr00tN1d7Config.from_pretrained(str(model_path))
    _action_dim = cfg.get("action_dim")
    if _action_dim is not None:
        config.action_dim = _action_dim

    # Explicit embodiment-shape overrides.
    # Useful when the checkpoint processor metadata belongs to another
    # new_embodiment dataset, e.g. PNI -> RoboCasa PandaOmron.
    _valid_action_dim = cfg.get("valid_action_dim")
    if _valid_action_dim is not None:
        config.valid_action_dim = int(_valid_action_dim)

    _image_nums = cfg.get("image_nums")
    if _image_nums is not None:
        config.image_nums = int(_image_nums)

    backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)

    model = GR00T_N1_7_ForRLActionPrediction.from_pretrained(
        config=config,
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        backbone_model_path=backbone_model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        rl_head_config=cfg.rl_head_config,
    )

    runtime_shapes = _audit_runtime_model_shapes(model, cfg)
    logger.info(
        "GR00T runtime shape audit passed: action_dim=%d, "
        "valid_action_dim=%d, image_nums=%d",
        runtime_shapes["action_dim"],
        runtime_shapes["valid_action_dim"],
        runtime_shapes["image_nums"],
    )

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
