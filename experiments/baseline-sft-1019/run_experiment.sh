#!/usr/bin/env bash
# Launch one approved four-GPU experiment without overwriting prior output.
set -euo pipefail
run_root=/mnt/public/chenzm/robottt/h800_sft
case "${1:-}" in
  sequence128) run_name=sequence128; devices=0,1,2,3; ray_temp=/tmp/h800-seq ;;
  h40) run_name=h40; devices=4,5,6,7; ray_temp=/tmp/h800-h40 ;;
  *) echo "Usage: bash code/run_experiment.sh sequence128|h40" >&2; exit 2 ;;
esac
run_dir="$run_root/$run_name"
exec 9>"$run_dir/.train.lock"
flock -n 9 || { echo "This experiment already has a launcher"; exit 3; }
for artifact in started.json train.log sequence_metrics.jsonl exit_code; do
  test ! -e "$run_dir/$artifact" || { echo "Refusing overwrite: $run_dir/$artifact"; exit 4; }
done
export PATH="$run_root/env/gr00t-baseline-env-20261012/bin:$PATH"
export PYTHONPATH="$run_root/code/RLinf:$run_root/code/Isaac-GR00T"
export CUDA_VISIBLE_DEVICES="$devices"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RAY_ADDRESS=local RAY_TMPDIR="$ray_temp" RAY_USAGE_STATS_ENABLED=0
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4294967296
export RLINF_CODE_WORKING_DIR=0
export USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export RUN_DIR="$run_dir" RUN_ROOT="$run_root"
python - <<'PY'
import hashlib,json,os,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
from omegaconf import OmegaConf
root,run=Path(os.environ["RUN_ROOT"]),Path(os.environ["RUN_DIR"])
cfg=OmegaConf.load(run/"config.yaml")
assert cfg.runner.max_steps==10000 and cfg.runner.resume_dir is None
assert cfg.actor.optim.total_training_steps==10000 and cfg.actor.optim.lr==2.5e-5
assert not cfg.actor.model.rl_head_config.use_ttt
assert cfg.actor.model.rl_head_config.ttt.num_register_tokens==0
assert cfg.data.robottt.terminal_window_probability==0
assert torch.cuda.device_count()==4
from rlinf.data.datasets.gr00t_n1d7.window_sampling import sequence_window_budgets
sequence_cfg=OmegaConf.load(root/"sequence128/config.yaml")
h40_cfg=OmegaConf.load(root/"h40/config.yaml")
reference=h40_cfg.data.robottt.match_sequence_budget
assert sequence_cfg.actor.global_batch_size==reference.global_batch_size==8
assert sequence_cfg.actor.micro_batch_size==reference.micro_batch_size==1
assert sequence_cfg.data.robottt.stride==reference.stride==8
assert sequence_cfg.data.robottt.max_anchors==reference.max_anchors==128
assert sequence_cfg.data.robottt.terminal_window_probability==0
assert sequence_cfg.actor.seed==h40_cfg.actor.seed
assert sequence_cfg.data.train_data_paths==h40_cfg.data.train_data_paths
assert sequence_cfg.data.robottt.action_horizon==h40_cfg.data.robottt.action_horizon==40
assert sequence_cfg.runner.max_steps==h40_cfg.runner.max_steps==10000
assert sequence_cfg.runner.resume_dir is None and h40_cfg.runner.resume_dir is None
assert OmegaConf.to_container(sequence_cfg.actor.optim)==OmegaConf.to_container(h40_cfg.actor.optim)
assert OmegaConf.to_container(sequence_cfg.actor.fsdp_config)==OmegaConf.to_container(h40_cfg.actor.fsdp_config)
lengths=[json.loads(line)["length"] for line in
         (Path(sequence_cfg.data.train_data_paths)/"meta/episodes.jsonl").read_text().splitlines()]
budgets=sequence_window_budgets(lengths,sequence_cfg.actor.seed,4,8,1,8,128,40,10000)
budget_identity={"windows":budgets,"total_windows":sum(budgets),
                 "sha256":hashlib.sha256(json.dumps(budgets).encode()).hexdigest()}
selected=[int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
gpu_ids={int(row.split(",")[0]):row.split(",")[1].strip() for row in rows.splitlines()}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
busy=[row for row in apps.splitlines() if row.split(",")[0].strip() in {gpu_ids[i] for i in selected}]
assert not busy, f"Assigned GPUs are occupied: {busy}"
with (run/"assets_verified.log").open("x") as out:
    subprocess.run(["sha256sum","-c",str(root/"code/source_models.sha256")],cwd=root/"assets",stdout=out,stderr=subprocess.STDOUT,check=True)
files=list((root/"code/RLinf/rlinf/data/datasets/gr00t_n1d7").glob("*.py"))
files += [root/"code/RLinf/rlinf/workers/sft/fsdp_gr00t_sequence_sft_worker.py",
          root/"code/RLinf/rlinf/models/embodiment/gr00t/gr00t_n1d7/sft.py",
          root/"code/RLinf/examples/sft/train_vla_sft.py",run/"config.yaml"]
record={"started_utc":datetime.now(timezone.utc).isoformat(),"hostname":socket.gethostname(),
        "launcher_pid":os.getppid(),"cuda_visible_devices":selected,
        "gpu_uuids":[gpu_ids[i] for i in selected],"ray_tmpdir":os.environ["RAY_TMPDIR"],
        "command":["python","examples/sft/train_vla_sft.py","--config-path",str(run),"--config-name","config"],
        "config":OmegaConf.to_container(cfg,resolve=True),
        "matched_sequence_budget":budget_identity,
        "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
with (run/"started.json").open("x") as out:
    json.dump(record,out,indent=2)
print(json.dumps({"preflight":"passed","gpu_uuids":record["gpu_uuids"]}))
PY
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit_code"' EXIT
printf "%s\n" "$$" > "$run_dir/launcher.pid"
cd "$run_root/code/RLinf"
python examples/sft/train_vla_sft.py --config-path "$run_dir" --config-name config >"$run_dir/train.log" 2>&1
