#!/usr/bin/env python3

"""
RoboTTT open-loop evaluation for RoboCasa365 BowlAndCup.

Purpose
-------
Evaluate a RoboTTT checkpoint on GT observations from an offline
LeRobot trajectory without executing env.step().

Two modes are compared:

1. recurrent
   TTT fast-weight state is propagated across action chunks:
       W0 -> W1 -> W2 -> W3 -> ...

2. reset
   TTT fast-weight state is reset before every action chunk:
       W0 -> W1
       W0 -> W1'
       W0 -> W1''
       ...

Fair comparison
---------------
For the same inference index, recurrent and reset use exactly the
same random seed for flow-matching / denoising. Therefore the intended
difference between the two modes is the TTT recurrent fast-weight state.

Action layout used for metric calculation is RoboCasa native 12D:

    0:3    end-effector position
    3:6    end-effector rotation
    6      gripper
    7:11   base motion
    11     control mode
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag

from rlinf.models import get_model


# ============================================================
# Constants
# ============================================================

STATE_KEYS = [
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
]

ACTION_GROUPS = {
    "ee_position": slice(0, 3),
    "ee_rotation": slice(3, 6),
    "gripper": slice(6, 7),
    "base_motion": slice(7, 11),
    "control_mode": slice(11, 12),
}

ACTION_DIM_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "gripper",
    "base_0",
    "base_1",
    "base_2",
    "base_3",
    "control_mode",
]


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_chunk(base_seed: int, infer_idx: int) -> int:
    """
    Re-seed before each policy inference.

    recurrent inference i and reset inference i therefore receive the
    same denoising randomness.
    """
    seed = int(base_seed + infer_idx)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


# ============================================================
# Dataset / observation utilities
# ============================================================

def first_time(x):
    """
    Remove the T dimension from a T=1 observation.
    """

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim >= 1 and x.shape[0] == 1:
        return x[0]

    return x


def text_to_string(x) -> str:
    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        if len(x) == 1:
            return str(next(iter(x.values())))
        return str(x)

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        return str(x[0])

    if isinstance(x, np.ndarray):
        flat = x.reshape(-1)
        if len(flat) == 0:
            return ""
        return str(flat[0])

    return str(x)


def make_env_obs(data_point) -> dict:
    """
    Convert an offline GT observation into the same observation structure
    consumed by RoboTTT closed-loop inference.

    Expected state layout:

        base_position                     3
        base_rotation                     4
        end_effector_position_relative    3
        end_effector_rotation_relative    4
        gripper_qpos                      2
                                         --
                                         16
    """

    left = first_time(
        data_point.images["robot0_agentview_left"]
    )

    right = first_time(
        data_point.images["robot0_agentview_right"]
    )

    wrist = first_time(
        data_point.images["robot0_eye_in_hand"]
    )

    state_parts = []

    for key in STATE_KEYS:
        value = first_time(data_point.states[key])

        value = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        state_parts.append(value)

    state = np.concatenate(
        state_parts,
        axis=0,
    ).astype(np.float32)

    if state.shape != (16,):
        raise ValueError(
            f"Expected RoboCasa365 state shape (16,), got {state.shape}"
        )

    task_description = text_to_string(
        data_point.text
    )

    # Match convert_robocasa365_obs_to_gr00t_format()
    #
    # main_images:
    #   [B,H,W,C]
    #
    # wrist_images:
    #   [B,H,W,C]
    #
    # extra_view_images:
    #   [B,N,H,W,C]
    #
    # states:
    #   [B,16]
    #
    return {
        "main_images": torch.from_numpy(
            np.asarray(left)[None]
        ),

        "wrist_images": torch.from_numpy(
            np.asarray(wrist)[None]
        ),

        "extra_view_images": torch.from_numpy(
            np.asarray(right)[None, None]
        ),

        "states": torch.from_numpy(
            state[None]
        ),

        "task_descriptions": [
            task_description
        ],
    }


def extract_gt_grouped_action(
    traj: pd.DataFrame,
    action_keys: list[str],
    start: int,
    horizon: int,
):
    """
    Extract dataset GT actions in the original grouped NEW_EMBODIMENT format.

    Returns:
        dict:
            action_key -> [B=1, T, D]

        valid_len:
            actual number of valid timesteps in this chunk
    """

    end = min(
        start + horizon,
        len(traj),
    )

    valid_len = end - start

    grouped = {}

    for key in action_keys:
        column = f"action.{key}"

        values = [
            np.asarray(v).reshape(-1)
            for v in traj[column].iloc[start:end]
        ]

        arr = np.stack(
            values,
            axis=0,
        ).astype(np.float32)

        grouped[key] = arr[None]

    return grouped, valid_len


def as_numpy_action(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    # [B=1,T,D] -> [T,D]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2:
        raise ValueError(
            f"Expected action [T,D], got {x.shape}"
        )

    return x.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Model
# ============================================================

def build_robottt(args):
    """
    Build the same RLinf GR00T-N1.7 RoboTTT model used by rollout eval,
    then load full_weights.pt with strict=True.
    """

    model_cfg = OmegaConf.create(
        {
            "model_type": "gr00t_n1d7",

            # Required by rlinf.models.get_model()
            "is_lora": False,

            "model_path": args.model_path,

            "backbone_model_path": (
                args.backbone_model_path
            ),

            "precision": "bf16",

            "embodiment_tag": "new_embodiment",

            "obs_converter_type": "robocasa365",

            "action_dim": 12,

            "valid_action_dim": 12,

            "image_nums": 3,

            "add_value_head": False,

            # Dataset / closed-loop effective action chunk.
            "num_action_chunks": args.horizon,

            # Same as closed-loop eval.
            "denoising_steps": 4,

            "rl_head_config": {
                "use_ttt": True,

                "add_value_head": False,

                "disable_dropout": True,

                "ttt": {
                    "fast_model": "mlp",

                    "num_register_tokens": 16,

                    "context_tokens": "register",

                    "train_ttt_only": False,

                    "gate_init": 0.001,

                    "base_lr": 0.1,
                },
            },
        }
    )

    print()
    print("============================================================")
    print("BUILD ROBOTTT MODEL")
    print("============================================================")
    print(OmegaConf.to_yaml(model_cfg))

    model = get_model(
        model_cfg
    )

    print()
    print("============================================================")
    print("LOAD CHECKPOINT")
    print("============================================================")
    print(args.ckpt)

    state_dict = torch.load(
        args.ckpt,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    model.eval()

    model.requires_grad_(False)

    print("device =", device)
    print("✅ checkpoint loaded with strict=True")

    print(
        "TTT enabled =",
        bool(model.action_head.ttt_enabled),
    )

    print(
        "num_action_chunks =",
        model.output_action_chunks,
    )

    print(
        "model action_horizon =",
        model.action_head.action_horizon,
    )

    return model


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict:
    if gt.shape != pred.shape:
        raise ValueError(
            f"GT/pred shape mismatch: {gt.shape} vs {pred.shape}"
        )

    err = pred - gt

    metrics = {
        "overall_mse": float(
            np.mean(err ** 2)
        ),

        "overall_mae": float(
            np.mean(np.abs(err))
        ),
    }

    for name, sl in ACTION_GROUPS.items():
        group_err = err[:, sl]

        metrics[f"{name}_mse"] = float(
            np.mean(group_err ** 2)
        )

        metrics[f"{name}_mae"] = float(
            np.mean(np.abs(group_err))
        )

    # Continuous motion metric excluding:
    #   gripper       index 6
    #   control_mode  index 11
    #
    continuous_idx = [
        0, 1, 2,
        3, 4, 5,
        7, 8, 9, 10,
    ]

    continuous_err = err[
        :,
        continuous_idx,
    ]

    metrics["continuous_mse"] = float(
        np.mean(
            continuous_err ** 2
        )
    )

    metrics["continuous_mae"] = float(
        np.mean(
            np.abs(continuous_err)
        )
    )

    # Gripper after RoboCasa converter is binary {-1,+1}.
    gripper_gt = gt[:, 6]
    gripper_pred = pred[:, 6]

    mismatch = np.where(
        gripper_gt != gripper_pred
    )[0]

    metrics["gripper_mismatch_count"] = int(
        len(mismatch)
    )

    metrics["gripper_mismatch_rate"] = float(
        len(mismatch) / len(gt)
    )

    metrics["gripper_mismatch_steps"] = (
        mismatch.astype(int).tolist()
    )

    return metrics


# ============================================================
# One evaluation mode
# ==========================================================

def run_mode(
    model,
    traj: pd.DataFrame,
    modality_configs,
    args,
    mode_name: str,
):
    """
    Run one open-loop mode.

    recurrent:
        keep TTT fast-weight state across chunks.

    reset:
        clear TTT state before every chunk.
    """

    if mode_name not in {
        "recurrent",
        "reset",
    }:
        raise ValueError(
            f"Unknown mode: {mode_name}"
        )

    print()
    print("============================================================")
    print(f"MODE = {mode_name}")
    print("============================================================")

    # Start both modes from exactly the same episode initial condition.
    model._ttt_rollout_states.clear()

    obs_configs = deepcopy(
        modality_configs
    )

    obs_configs.pop(
        "action"
    )

    action_keys = list(
        modality_configs[
            "action"
        ].modality_keys
    )

    actual_steps = min(
        args.steps,
        len(traj),
    )

    all_gt = []
    all_pred = []

    rows = []

    for infer_idx, step in enumerate(
        range(
            0,
            actual_steps,
            args.horizon,
        )
    ):

        # ----------------------------------------------------
        # Reset ablation
        # ----------------------------------------------------

        if mode_name == "reset":
            model._ttt_rollout_states.clear()

        # ----------------------------------------------------
        # GT observation from dataset
        # ----------------------------------------------------

        data_point = extract_step_data(
            traj,
            step,
            obs_configs,
            EmbodimentTag.NEW_EMBODIMENT,
        )

        env_obs = make_env_obs(
            data_point
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # identical denoising randomness for:
        #
        # recurrent inference i
        # reset     inference i
        #
        # Therefore the controlled difference is TTT memory.
        # ----------------------------------------------------

        chunk_seed = seed_chunk(
            args.noise_seed,
            infer_idx,
        )

        # ----------------------------------------------------
        # RoboTTT prediction
        # ----------------------------------------------------

        with torch.no_grad():
            pred_raw, _ = model.predict_action_batch(
                env_obs,
                mode="eval",
                dones=None,
                stage_id=0,
            )

        pred = as_numpy_action(
            pred_raw
        )

        # ----------------------------------------------------
        # GT action
        # ----------------------------------------------------

        gt_grouped, valid_len = (
            extract_gt_grouped_action(
                traj,
                action_keys,
                step,
                args.horizon,
            )
        )

        # Convert GT action through the exact same RoboCasa
        # action converter used by closed-loop inference.
        #
        # Dataset grouped order:
        #
        #   base_motion
        #   control_mode
        #   ee_position
        #   ee_rotation
        #   gripper
        #
        # Converted native RoboCasa order:
        #
        #   ee_position     0:3
        #   ee_rotation     3:6
        #   gripper         6
        #   base_motion     7:11
        #   control_mode    11
        #
        gt_native = model.action_convert_fn(
            gt_grouped,
            chunk_size=valid_len,
        )

        gt = as_numpy_action(
            gt_native
        )

        pred = pred[
            :valid_len
        ]

        if pred.shape != gt.shape:
            raise ValueError(
                f"step={step}: "
                f"pred={pred.shape}, "
                f"gt={gt.shape}"
            )

        # ----------------------------------------------------
        # Chunk metrics
        # ----------------------------------------------------

        err = pred - gt

        chunk_mse = float(
            np.mean(
                err ** 2
            )
        )

        chunk_mae = float(
            np.mean(
                np.abs(err)
            )
        )

        grip_bad = int(
            np.sum(
                pred[:, 6]
                !=
                gt[:, 6]
            )
        )

        rows.append(
            {
                "mode": mode_name,
                "inference_index": infer_idx,
                "env_step": step,
                "valid_len": valid_len,
                "noise_seed": chunk_seed,
                "chunk_mse": chunk_mse,
                "chunk_mae": chunk_mae,
                "gripper_mismatch": grip_bad,
            }
        )

        all_gt.append(
            gt
        )

        all_pred.append(
            pred
        )

        print(
            "[OPEN LOOP] "
            f"mode={mode_name:9s} "
            f"infer={infer_idx:03d} "
            f"step={step:04d} "
            f"seed={chunk_seed} "
            f"mse={chunk_mse:.6e} "
            f"mae={chunk_mae:.6e} "
            f"grip_bad={grip_bad}/{valid_len}"
        )

    gt_all = np.concatenate(
        all_gt,
        axis=0,
    )

    pred_all = np.concatenate(
        all_pred,
        axis=0,
    )

    metrics = calculate_metrics(
        gt_all,
        pred_all,
    )

    print()
    print(
        f"===== {mode_name.upper()} RESULT ====="
    )

    for key, value in metrics.items():
        if key == "gripper_mismatch_steps":
            print(
                f"{key:28s} = {value}"
            )
        elif isinstance(value, int):
            print(
                f"{key:28s} = {value}"
            )
        else:
            print(
                f"{key:28s} = {value:.8e}"
            )

    return {
        "gt": gt_all,
        "pred": pred_all,
        "metrics": metrics,
        "rows": rows,
    }


# ============================================================
# Plot
# ============================================================

def save_plot(
    gt: np.ndarray,
    recurrent: np.ndarray,
    reset: np.ndarray,
    output_path: Path,
) -> None:

    num_dims = gt.shape[1]

    fig, axes = plt.subplots(
        num_dims,
        1,
        figsize=(
            14,
            2.3 * num_dims,
        ),
        sharex=True,
    )

    if num_dims == 1:
        axes = [axes]

    for dim in range(
        num_dims
    ):
        ax = axes[dim]

        ax.plot(
            gt[:, dim],
            label="GT",
            linewidth=1.7,
        )

        ax.plot(
            recurrent[:, dim],
            label="RoboTTT recurrent",
            linewidth=1.0,
        )

        ax.plot(
            reset[:, dim],
            label="RoboTTT reset",
            linewidth=1.0,
            alpha=0.8,
        )

        ax.set_ylabel(
            ACTION_DIM_NAMES[dim]
        )

        ax.grid(
            alpha=0.2
        )

        if dim == 0:
            ax.legend()

    axes[-1].set_xlabel(
        "environment timestep"
    )

    fig.suptitle(
        "BowlAndCup Open-loop: GT vs RoboTTT",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Extra diagnostic
# ============================================================

def print_gripper_window(
    gt,
    recurrent,
    reset,
    start=88,
    end=136,
):
    end = min(
        end,
        len(gt),
    )

    start = min(
        start,
        end,
    )

    print()
    print("============================================================")
    print("GRIPPER WINDOW")
    print("============================================================")

    print(
        "step | GT grip | recurrent | reset | mismatch"
    )

    print(
        "-" * 54
    )

    for t in range(
        start,
        end,
    ):
        mark = []

        if (
            recurrent[t, 6]
            !=
            gt[t, 6]
        ):
            mark.append(
                "R"
            )

        if (
            reset[t, 6]
            !=
            gt[t, 6]
        ):
            mark.append(
                "S"
            )

        print(
            f"{t:4d} | "
            f"{gt[t, 6]:7.2f} | "
            f"{recurrent[t, 6]:9.2f} | "
            f"{reset[t, 6]:5.2f} | "
            f"{' '.join(mark)}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "RoboTTT BowlAndCup open-loop "
            "recurrent-vs-reset evaluation"
        )
    )

    parser.add_argument(
        "--ckpt",
        required=True,
        help="RLinf full_weights.pt",
    )

    parser.add_argument(
        "--dataset-path",
        default=(
            "/mnt/public/chenzm/robottt/"
            "datasets/robocasa365/"
            "BowlAndCup/lerobot/lerobot"
        ),
    )

    parser.add_argument(
        "--model-path",
        default=(
            "/mnt/public/chenzm/gr00t_sft/"
            "gr00t_n1d7_bowlandcup_sft50k_eval_model"
        ),
    )

    parser.add_argument(
        "--backbone-model-path",
        default=(
            "/mnt/public/guozhen/models/"
            "Cosmos-Reason2-2B"
        ),
    )

    parser.add_argument(
        "--traj-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--noise-seed",
        type=int,
        default=12345,
        help=(
            "Per-chunk seed is "
            "noise_seed + inference_index"
        ),
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    args = parser.parse_args()

    if args.horizon != 8:
        print(
            "WARNING: BowlAndCup modality config "
            "uses action horizon 8."
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        args.noise_seed
    )

    print()
    print("============================================================")
    print("ROBOTTT OPEN-LOOP EVALUATION")
    print("============================================================")

    print(
        "checkpoint       =",
        args.ckpt,
    )

    print(
        "dataset          =",
        args.dataset_path,
    )

    print(
        "trajectory       =",
        args.traj_id,
    )

    print(
        "steps            =",
        args.steps,
    )

    print(
        "action horizon   =",
        args.horizon,
    )

    print(
        "noise seed base  =",
        args.noise_seed,
    )

    print(
        "output           =",
        out_dir,
    )

    # --------------------------------------------------------
    # Modality config
    # --------------------------------------------------------

    modality_namespace = runpy.run_path(
        (
            "examples/sft/config/"
            "robocasa365_panda_omron_config.py"
        )
    )

    modality_configs = (
        modality_namespace[
            "robocasa365_panda_omron_config"
        ]
    )

    print()
    print("============================================================")
    print("LOAD DATASET")
    print("============================================================")

    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,

        modality_configs=(
            modality_configs
        ),

        video_backend="torchcodec",

        video_backend_kwargs=None,
    )

    print(
        "num trajectories =",
        len(loader),
    )

    if (
        args.traj_id < 0
        or
        args.traj_id >= len(loader)
    ):
        raise ValueError(
            f"traj_id={args.traj_id} "
            f"outside [0,{len(loader)-1}]"
        )

    traj = loader[
        args.traj_id
    ]

    print(
        "trajectory length =",
        len(traj),
    )

    print(
        "effective eval steps =",
        min(
            args.steps,
            len(traj),
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_robottt(
        args
    )

    # --------------------------------------------------------
    # Mode A: recurrent
    # --------------------------------------------------------

    recurrent = run_mode(
        model,
        traj,
        modality_configs,
        args,
        mode_name="recurrent",
    )

    # --------------------------------------------------------
    # Mode B: reset
    # --------------------------------------------------------

    reset = run_mode(
        model,
        traj,
        modality_configs,
        args,
        mode_name="reset",
    )

    # --------------------------------------------------------
    # Verify GT is identical
    # --------------------------------------------------------

    if not np.allclose(
        recurrent["gt"],
        reset["gt"],
    ):
        raise RuntimeError(
            "GT arrays differ between "
            "recurrent and reset modes."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    metrics = {
        "checkpoint": args.ckpt,

        "traj_id": args.traj_id,

        "steps": args.steps,

        "action_horizon": args.horizon,

        "noise_seed_base": args.noise_seed,

        "recurrent": recurrent[
            "metrics"
        ],

        "reset": reset[
            "metrics"
        ],
    }

    with open(
        out_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    pd.DataFrame(
        recurrent["rows"]
        +
        reset["rows"]
    ).to_csv(
        out_dir / "per_chunk_metrics.csv",
        index=False,
    )

    np.savez(
        out_dir / "pred_vs_gt.npz",

        gt=recurrent[
            "gt"
        ],

        recurrent=recurrent[
            "pred"
        ],

        reset=reset[
            "pred"
        ],
    )

    save_plot(
        recurrent["gt"],
        recurrent["pred"],
        reset["pred"],
        out_dir / "gt_vs_pred.png",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("============================================================")
    print("FINAL SUMMARY")
    print("============================================================")

    print(
        f"{'metric':28s}"
        f"{'recurrent':>16s}"
        f"{'reset':>16s}"
    )

    print(
        "-" * 60
    )

    summary_keys = [
        "overall_mse",
        "overall_mae",
        "continuous_mse",
        "continuous_mae",
        "ee_position_mse",
        "ee_position_mae",
        "ee_rotation_mse",
        "ee_rotation_mae",
        "gripper_mse",
        "gripper_mae",
        "base_motion_mse",
        "base_motion_mae",
        "control_mode_mse",
        "control_mode_mae",
    ]

    for key in summary_keys:
        r = recurrent[
            "metrics"
        ][key]

        s = reset[
            "metrics"
        ][key]

        print(
            f"{key:28s}"
            f"{r:16.8e}"
            f"{s:16.8e}"
        )

    print()
    print(
        "gripper mismatch count:"
    )

    print(
        "recurrent =",
        recurrent[
            "metrics"
        ][
            "gripper_mismatch_count"
        ],
    )

    print(
        "reset     =",
        reset[
            "metrics"
        ][
            "gripper_mismatch_count"
        ],
    )

    print()
    print(
        "recurrent mismatch steps =",
        recurrent[
            "metrics"
        ][
            "gripper_mismatch_steps"
        ],
    )

    print(
        "reset mismatch steps     =",
        reset[
            "metrics"
        ][
            "gripper_mismatch_steps"
        ],
    )

    # Diagnostic around the interesting gripper transition.
    print_gripper_window(
        recurrent["gt"],
        recurrent["pred"],
        reset["pred"],
        start=88,
        end=136,
    )

    print()
    print("============================================================")
    print("OUTPUT FILES")
    print("============================================================")

    print(
        out_dir / "metrics.json"
    )

    print(
        out_dir / "per_chunk_metrics.csv"
    )

    print(
        out_dir / "pred_vs_gt.npz"
    )

    print(
        out_dir / "gt_vs_pred.png"
    )


if __name__ == "__main__":
    main()
