# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""In-sample observation replay: noisy-GT versus predicted-history memory."""

import argparse
import json
from pathlib import Path

import torch

from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
    EpisodeDataset,
    SequenceCollator,
    build_processor,
    segment_batch,
)
from rlinf.models import get_model
from rlinf.models.embodiment.gr00t.gr00t_n1d7.flow import euler_sample, sample_path
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    dataset_settings,
    load_sft_config,
)


def load_policy(checkpoint: Path, config, device):
    """Use the same RLinf model factory and bundled processor as rollout workers."""
    if checkpoint.is_dir():
        native = checkpoint / "model_state_dict/full_weights.pt"
        checkpoint = native if native.is_file() else checkpoint / "model.pt"
    policy = get_model(config.actor.model)
    policy.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    policy.load_inference_metadata(checkpoint, config.actor.model)
    policy.to(device).eval()
    return policy, policy._modality_transform, dataset_settings(config)


def state_distance(a, b) -> dict:
    differences, magnitudes = [], []
    for la, lb in zip(a.layers, b.layers):
        differences.append(
            torch.cat([(x - y).flatten() for x, y in zip(la, lb)]).square().mean()
        )
        magnitudes.append(torch.cat([x.flatten() for x in la]).square().mean())
    rms = torch.stack(differences).mean().sqrt()
    magnitude = torch.stack(magnitudes).mean().sqrt()
    return {
        "memory_difference_rms": rms.item(),
        "gt_memory_rms": magnitude.item(),
        "relative_memory_difference": (rms / magnitude.clamp_min(1e-12)).item(),
        "per_layer_difference_rms": torch.stack(differences).sqrt().tolist(),
    }


def plot_replay(report: dict, output: Path):
    """Plot observed differences without interpreting them as generalization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = report["records"]
    x = [row["anchor"] for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for axis, prefix, title in (
        (axes[0, 0], "flow_loss", "Current flow loss"),
        (axes[0, 1], "action_mse", "Current action MSE (40 x 12)"),
    ):
        for suffix, label, color in (
            ("gt_history", "Noisy-GT history", "#0072B2"),
            ("predicted_history", "Predicted history", "#D55E00"),
        ):
            axis.plot(
                x, [row[f"{prefix}_{suffix}"] for row in rows], label=label, color=color
            )
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=8)
    axes[0, 2].plot(
        x, [row["relative_memory_difference"] for row in rows], color="#6A3D9A"
    )
    axes[0, 2].set_title("Relative memory difference after update")
    axes[1, 0].plot(x, [row["action_difference_mse"] for row in rows], color="#6A3D9A")
    axes[1, 0].set_title("Action difference MSE between histories")
    for key, label, color in (
        ("training_inactive_input_rms", "Training noisy-GT input", "#0072B2"),
        ("inactive_action_rms", "Final prediction", "#D55E00"),
    ):
        axes[1, 1].plot(x, [row[key] for row in rows], label=label, color=color)
    axes[1, 1].set_title("RMS in 120 unsupervised action dimensions")
    axes[1, 1].legend(frameon=False, fontsize=8)
    values = np.asarray([row["per_layer_difference_rms"] for row in rows]).T
    heatmap = axes[1, 2].imshow(values, aspect="auto", origin="lower", cmap="magma")
    axes[1, 2].set(title="Memory difference RMS after update", ylabel="DiT layer")
    figure.colorbar(heatmap, ax=axes[1, 2])
    for axis in axes.flat:
        axis.set_xlabel("Observation anchor")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Training-set observation replay | episode {report['episode']} | "
        "same current noise | not a generalization test"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", required=True, help="Native SFT YAML defining the training data"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    policy, processor, cfg = load_policy(
        Path(args.checkpoint), load_sft_config(args.config), device
    )
    if not policy.use_ttt:
        raise ValueError("Memory diagnostic requires RoboTTT")
    _, loader, _ = build_processor(cfg)
    dataset = EpisodeDataset(cfg, processor, loader)
    sample = dataset[(0, 0)]
    batch = SequenceCollator(processor)([sample])
    torch.manual_seed(1701)
    gt_memory = pred_memory = None
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(min(args.anchors, sample["description"]["valid"])):
            observation = segment_batch(batch, i, i + 1, device)
            features, state_features = policy.encode(observation)
            gt = observation["action"].float()
            mask = observation["action_mask"].float()
            xt, target, bucket = sample_path(gt)

            def field(x, buckets, memory):
                velocity, candidate, _ = policy.velocity(
                    x,
                    buckets,
                    observation["embodiment_id"],
                    features,
                    state_features,
                    memory,
                    observation["anchor_valid"],
                )
                return velocity, candidate

            velocity_gt, next_gt = field(xt, bucket, gt_memory)
            velocity_pred, _ = field(xt, bucket, pred_memory)
            ode_noise = torch.randn_like(gt)
            options = cfg["inference"]
            sampling = {
                "steps": options["denoise_steps"],
                "write_source": options["memory_write_source"],
                "write_index": options["memory_write_denoise_index"],
            }
            action_gt, _ = euler_sample(field, ode_noise, gt_memory, **sampling)
            action_pred, next_pred = euler_sample(
                field, ode_noise, pred_memory, **sampling
            )
            if i == 0:
                torch.testing.assert_close(velocity_gt, velocity_pred, rtol=0, atol=0)
                torch.testing.assert_close(action_gt, action_pred, rtol=0, atol=0)

            def error(a, b, region=slice(None)):
                weight = mask[:, region]
                return (
                    (
                        (a[:, region].float() - b[:, region].float()).square() * weight
                    ).sum()
                    / weight.sum()
                ).item()

            record = {
                "anchor": i,
                "raw_frame": sample["description"]["anchors"][i],
                "flow_bucket": bucket.item(),
                "flow_loss_gt_history": error(velocity_gt, target),
                "flow_loss_predicted_history": error(velocity_pred, target),
                "action_mse_gt_history": error(action_gt, gt),
                "action_mse_predicted_history": error(action_pred, gt),
                "action_difference_mse": error(action_gt, action_pred),
            }
            for label, region in (
                ("executed_0_8", slice(0, 8)),
                ("overlap_0_32", slice(0, 32)),
                ("novel_32_40", slice(32, 40)),
            ):
                record[label] = {
                    "flow_loss_gt_history": error(velocity_gt, target, region),
                    "flow_loss_predicted_history": error(velocity_pred, target, region),
                    "action_mse_gt_history": error(action_gt, gt, region),
                    "action_mse_predicted_history": error(action_pred, gt, region),
                }
            record["inactive_action_rms"] = (
                action_pred[..., 12:].float().square().mean().sqrt().item()
            )
            record["training_inactive_input_rms"] = (
                xt[..., 12:].float().square().mean().sqrt().item()
            )
            gt_memory, pred_memory = next_gt.detach(), next_pred.detach()
            record.update(state_distance(gt_memory, pred_memory))
            if any(
                not torch.isfinite(torch.tensor(x))
                for x in record.values()
                if isinstance(x, float)
            ):
                raise FloatingPointError("Non-finite replay diagnostic")
            records.append(record)
    report = {
        "checkpoint": args.checkpoint,
        "episode": sample["description"]["episode"],
        "seed": 1701,
        "anchors": len(records),
        "identical_initial_history_verified": True,
        "scope": "Training-set observation replay; no closed-loop or generalization claim",
        "gt_history": "Independent Beta flow-time noisy GT update, matching training semantics",
        "predicted_history": "Final predicted action update at bucket 999",
        "adjacent_chunk_overlap": {
            "steps": 32,
            "fraction": 0.8,
            "first_anchor_has_no_history": True,
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    plot_replay(report, output.with_suffix(".png"))


if __name__ == "__main__":
    main()
