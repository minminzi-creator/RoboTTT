# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Plan or explicitly execute paired OSMesa native GR00T trials.

Use --dry-run to inspect commands without creating paths, importing the
simulator, loading weights, or launching workers. Execution requires --execute.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def build_plan(args) -> dict:
    """Construct three eight-environment batches for each verification phase."""
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    checkpoint = str(Path(args.checkpoint).resolve())
    if len(args.gpus.split(",")) != 8 or len(set(args.gpus.split(","))) != 8:
        raise ValueError("Select exactly eight distinct GPUs for eight rollout workers")
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(repo), args.dependency, args.robocasa, args.robosuite]
        ),
        "EMBODIED_PATH": str(repo / "examples/embodiment"),
        "REPO_PATH": str(repo),
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "USE_TF": "0",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NUMBA_DISABLE_JIT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RAY_ADDRESS": "local",
        "RAY_USAGE_STATS_ENABLED": "0",
        "RLINF_CODE_WORKING_DIR": "0",
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": "4294967296",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "HYDRA_FULL_ERROR": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if args.renderer_library_path:
        environment["LD_LIBRARY_PATH"] = args.renderer_library_path
    batches = []
    for phase in ("preflight", "formal"):
        for trial in range(3):
            for seed in (7,):
                path = output / phase / f"trial_{trial}" / "seeds_7_14"
                reference = (
                    (Path(args.reference) if args.reference else output)
                    / "preflight"
                    / "trial_0"
                )
                horizon = 8 if phase == "preflight" else 2400
                command = [
                    args.python,
                    str(repo / "evaluations/eval_embodied_agent.py"),
                    "--config-path",
                    str(repo / "evaluations/robocasa365"),
                    "--config-name",
                    "bowlandcup_gr00t_native_ttt",
                    *args.override,
                    f"runner.ckpt_path={checkpoint}",
                    f"runner.logger.log_path={path}",
                    f"env.eval.seed={seed}",
                    f"rollout.trial_id={trial}",
                    f"rollout.memory_mode={args.memory_mode}",
                    f"rollout.model.dependency_path={args.dependency}",
                    f"rollout.preflight={str(phase == 'preflight').lower()}",
                    f"env.eval.max_episode_steps={horizon}",
                    f"env.eval.max_steps_per_rollout_epoch={horizon}",
                ]
                if args.reference or phase != "preflight" or trial != 0:
                    command.append(f"rollout.reference_dir={reference}")
                    command.append(
                        f"rollout.reference_batch_size={args.reference_width if args.reference else 8}"
                    )
                batches.append(
                    {
                        "phase": phase,
                        "trial": trial,
                        "seed": seed,
                        "scene_seeds": list(range(7, 15)),
                        "output": str(path),
                        "command": command,
                    }
                )
    return {
        "checkpoint": checkpoint,
        "output": str(output),
        "memory_mode": args.memory_mode,
        "formal_episodes": 24,
        "environment": environment,
        "batches": batches,
    }


def _read(path):
    return Path(path).read_text()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def check_batch(batch, plan) -> list[dict]:
    """Reject incomplete, unpaired, wrong-checkpoint or wrong-mode trial output."""
    output = Path(batch["output"])
    records = []
    for rank, scene in enumerate(batch["scene_seeds"]):
        audit = json.loads(_read(output / f"trial_audit_rank{rank}.json"))
        model = json.loads(_read(output / f"model_rank{rank}.json"))
        episode = json.loads(_read(output / f"episode_rank{rank}.json"))
        trial = batch["trial"]
        expected_seed = scene + 1000000 * trial
        if (audit["scene_seed"], audit["trial_id"], audit["policy_seed"]) != (
            scene,
            trial,
            expected_seed,
        ):
            raise ValueError("Trial identity disagrees with the planned scene and RNG")
        if (
            audit["first_action_shape"] != [1, 8, 12]
            or not audit["first_action_finite"]
        ):
            raise ValueError("Invalid first action chunk")
        if not audit["pairing"]["physical_identity_equal"]:
            raise ValueError("Scene pairing failed")
        if (
            model["checkpoint"] != plan["checkpoint"]
            or not model["strict_load"]
            or not model["finite_weights"]
            or model["memory_mode"] != plan["memory_mode"]
        ):
            raise ValueError(
                "Checkpoint identity or memory mode disagrees with the plan"
            )
        expected_dtypes = (
            ["torch.bfloat16", "torch.float32"]
            if model["use_ttt"]
            else ["torch.bfloat16"]
        )
        if model["parameter_dtypes"] != expected_dtypes or model["buffer_dtypes"] != [
            "torch.float32"
        ]:
            raise ValueError(
                "Native/TTT inference precision disagrees with the protocol"
            )
        if (episode["scene_seed"], episode["trial_id"], episode["policy_seed"]) != (
            scene,
            trial,
            expected_seed,
        ):
            raise ValueError("Episode identity disagrees with the plan")
        preflight = batch["phase"] == "preflight"
        if episode["preflight"] != preflight or not 1 <= episode["episode_len"] <= (
            8 if preflight else 2400
        ):
            raise ValueError("Episode phase or horizon disagrees with the plan")
        records.append({"audit": audit, "model": model, "episode": episode})
    return records


def runtime_environment(plan, runtime_path) -> dict:
    """Use a short OS temporary root; Ray appends session/socket path components."""
    runtime_path = str(runtime_path)
    if not Path(runtime_path).is_absolute() or len(runtime_path.encode("utf-8")) > 24:
        raise ValueError(
            "Use a short absolute Ray runtime path to fit its socket limit"
        )
    environment = {**os.environ, **plan["environment"]}
    environment.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    environment["RAY_TMPDIR"] = runtime_path
    return environment


def execute(plan) -> None:
    """Run only an explicit plan into a fresh output directory."""
    from evaluations.gr00t_native_policy import (
        check_checkpoint,
        validate_output_destination,
    )

    validate_output_destination(plan["checkpoint"], plan["output"], require_fresh=True)
    check_checkpoint(plan["checkpoint"])
    root = Path(plan["output"])
    repo = Path(__file__).resolve().parents[1]
    sources = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in (
            "evaluations/gr00t_native_policy.py",
            "evaluations/gr00t_native_worker.py",
            "evaluations/eval_embodied_agent.py",
            "evaluations/run_gr00t_native_trials.py",
            "evaluations/robocasa365/bowlandcup_gr00t_native_ttt.yaml",
            "rlinf/envs/sim/robocasa365/robocasa365_env.py",
        )
    }
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "plan.json", {**plan, "source_sha256": sources})
    # Runtime-only files live under a short OS temporary path, not the long
    # experiment root: installed Ray limits Linux Unix sockets to 107 bytes.
    runtime_path = tempfile.mkdtemp(prefix="r3-", dir="/tmp")
    environment = runtime_environment(plan, runtime_path)
    _write(root / "runtime.json", {"ray_tmpdir": runtime_path})
    preflight, episodes = {}, []
    for batch in plan["batches"]:
        for name, expected in sources.items():
            if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evaluation source changed: {name}")
        output = Path(batch["output"])
        output.mkdir(parents=True, exist_ok=False)
        with (output / "eval.log").open("x") as stream:
            result = subprocess.run(
                batch["command"],
                cwd=repo,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        _write(output / "exit.json", {"exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(
                f"Evaluation batch failed; inspect {output / 'eval.log'}"
            )
        rows = check_batch(batch, plan)
        for row in rows:
            key = (row["episode"]["scene_seed"], row["episode"]["trial_id"])
            if batch["phase"] == "preflight":
                preflight[key] = row
            else:
                for field in ("cpu_rng_sha256", "device_rng_sha256"):
                    if row["audit"][field] != preflight[key]["audit"][field]:
                        raise ValueError(
                            "Formal trial RNG disagrees with its preflight"
                        )
                episodes.append(row["episode"])
        if len(preflight) == 24 and batch["phase"] == "preflight":
            for scene in range(7, 15):
                rows = [preflight[(scene, trial)] for trial in range(3)]
                for field in ("cpu_rng_sha256", "device_rng_sha256"):
                    if len({row["audit"][field] for row in rows}) != 3:
                        raise ValueError(
                            "Trials must use three independent policy seeds"
                        )
                if len({row["model"]["processor_sha256"] for row in rows}) != 1:
                    raise ValueError("Processor changed between trials")
            _write(root / "preflight_passed.json", {"passed": True, "episodes": 24})
    if len(episodes) != 24:
        raise ValueError("Expected 24 completed formal episodes")
    _write(
        root / "results.json",
        {
            "complete": True,
            "episodes": episodes,
            "total": 24,
            "successes": sum(row["success"] for row in episodes),
            "success_rate": sum(row["success"] for row in episodes) / 24,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "output", "dependency", "robocasa", "robosuite"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--memory-mode", choices=("carry", "reset"), default="carry")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--renderer-library-path")
    parser.add_argument("--reference", help="Existing paired evaluation root")
    parser.add_argument(
        "--reference-width",
        type=int,
        choices=(4, 8),
        default=8,
        help="Reference worker count; use 4 explicitly for historical baseline trials",
    )
    parser.add_argument("--override", action="append", default=[])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dry-run", action="store_true")
    selection.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        # Direct script execution places evaluations/, not its parent, on sys.path.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        execute(plan)


if __name__ == "__main__":
    main()
