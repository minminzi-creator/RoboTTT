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

import numpy as np
import torch
import torch.nn.functional as F


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}

    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # [B, 8] -> [B, T(1), 8]
    groot_obs["state.x"] = env_obs["states"].unsqueeze(1)[:, :, 0:1].numpy()
    groot_obs["state.y"] = env_obs["states"].unsqueeze(1)[:, :, 1:2].numpy()
    groot_obs["state.z"] = env_obs["states"].unsqueeze(1)[:, :, 2:3].numpy()
    groot_obs["state.roll"] = env_obs["states"].unsqueeze(1)[:, :, 3:4].numpy()
    groot_obs["state.pitch"] = env_obs["states"].unsqueeze(1)[:, :, 4:5].numpy()
    groot_obs["state.yaw"] = env_obs["states"].unsqueeze(1)[:, :, 5:6].numpy()
    groot_obs["state.gripper"] = env_obs["states"].unsqueeze(1)[:, :, 6:].numpy()
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],  # H
        256,
    )
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).numpy()
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t_1_7 pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action_n1d5(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T N1.5 action chunk to Libero format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d6(
    action_chunk: dict[str, np.array],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.6 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")

    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.7 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")
    # gripper conversion is handled by the shared
    # ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.

    if action_array.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim action, got {action_array.shape[-1]}")
    return action_array


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Isaaclab Stack Cube format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array




def _robocasa365_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def convert_robocasa365_obs_to_gr00t_format(env_obs):
    """Convert RLinf RoboCasa365 observation to the SFT NEW_EMBODIMENT format.

    env state layout:
      base_pos(3)
      base_quat(4)
      base_to_eef_pos(3)
      base_to_eef_quat(4)
      gripper_qpos(2)
    total = 16
    """
    main = _robocasa365_to_numpy(env_obs["main_images"])
    wrist = _robocasa365_to_numpy(env_obs["wrist_images"])
    extra = _robocasa365_to_numpy(env_obs["extra_view_images"])
    state = _robocasa365_to_numpy(env_obs["states"])

    if state.ndim != 2 or state.shape[-1] != 16:
        raise ValueError(
            f"RoboCasa365 GR00T expects states [B,16], got {state.shape}"
        )

    if extra.ndim != 5 or extra.shape[1] < 1:
        raise ValueError(
            "RoboCasa365 GR00T expects extra_view_images "
            f"[B,N,H,W,C] with N>=1, got {extra.shape}"
        )

    right = extra[:, 0]

    # [B,H,W,C] -> [B,T=1,H,W,C]
    groot_obs = {
        "video.robot0_agentview_left": main[:, None],
        "video.robot0_agentview_right": right[:, None],
        "video.robot0_eye_in_hand": wrist[:, None],

        "state.base_position": state[:, None, 0:3],
        "state.base_rotation": state[:, None, 3:7],
        "state.end_effector_position_relative": state[:, None, 7:10],
        "state.end_effector_rotation_relative": state[:, None, 10:14],
        "state.gripper_qpos": state[:, None, 14:16],

        "annotation.human.task_description": env_obs["task_descriptions"],
    }

    return groot_obs


def convert_to_robocasa365_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 8,
) -> np.ndarray:
    """Convert SFT grouped 12D action to native PandaOmron RoboCasa action.

    SFT / processor order:
      base_motion(4), control_mode(1),
      ee_position(3), ee_rotation(3), gripper(1)

    RoboCasa native order:
      ee_position(3), ee_rotation(3), gripper(1),
      base_motion(4), control_mode(1)
    """

    base = np.asarray(action_chunk["base_motion"])[:, :chunk_size]
    mode = np.asarray(action_chunk["control_mode"])[:, :chunk_size]
    pos = np.asarray(action_chunk["end_effector_position"])[:, :chunk_size]
    rot = np.asarray(action_chunk["end_effector_rotation"])[:, :chunk_size]
    grip = np.asarray(action_chunk["gripper_close"])[:, :chunk_size]

    action = np.concatenate(
        [
            pos,
            rot,
            grip,
            base,
            mode,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    if action.shape[-1] != 12:
        raise ValueError(
            f"Expected 12-dim RoboCasa365 action, got {action.shape}"
        )

    # Native RoboCasa gripper expects {-1,+1}
    action[..., 6] = np.where(action[..., 6] >= 0.0, 1.0, -1.0)

    return action


OBS_CONVERSION = {
    "robocasa365": convert_robocasa365_obs_to_gr00t_format,
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
}

ACTION_CONVERSION_N1D5 = {
    "libero": convert_to_libero_action_n1d5,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D6 = {
    "libero": convert_to_libero_action_n1d6,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D7 = {
    "robocasa365": convert_to_robocasa365_action_n1d7,
    "libero": convert_to_libero_action_n1d7,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """Cut and resize the images to the crop size."""
    images_nchw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    original_width = images_nchw.shape[-1]  # W
    start = (original_width - crop_size) // 2
    end = start + crop_size

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images_nchw[:, :, :, start:end]  # [B, C, H, crop_W]

    # Resize: interpolate to target_size x target_size
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )  # [B, C, target_size, target_size]

    # Convert back to NHWC
    resized_nhwc = resized_tensor.permute(
        0, 2, 3, 1
    ).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    return resized_nhwc


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
