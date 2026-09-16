from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


robocasa365_panda_omron_config = {
    # 3 cameras, current frame
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    ),

    # Exactly matches BowlAndCup meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_position",
            "base_rotation",
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
        ],
    ),

    # RoboCasa action chunk.
    # Keep raw RoboCasa controls; do not add relative-action transforms here.
    "action": ModalityConfig(
        delta_indices=list(range(8)),
        modality_keys=[
            "base_motion",
            "control_mode",
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
        ],
    ),

    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.task_description",
        ],
    ),
}


register_modality_config(
    robocasa365_panda_omron_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
