#!/usr/bin/env bash
set -euo pipefail
export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH:-}:${DREAMZERO_PATH:-}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [hydra_overrides...]" >&2
    exit 1
fi
config_arg="$1"
shift
if [[ -f "$config_arg" ]]; then
    config_path="$(cd "$(dirname "$config_arg")" && pwd)"
    config_name="$(basename "$config_arg" .yaml)"
else
    config_path="${EMBODIED_PATH}/config"
    config_name="${config_arg%.yaml}"
fi
[[ -f "${config_path}/${config_name}.yaml" ]] || { echo "Missing config: ${config_path}/${config_name}.yaml" >&2; exit 1; }
log_dir="${LOG_PATH:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${config_name}}"
mkdir -p "$log_dir"
cmd=(python "${EMBODIED_PATH}/train_vla_sft.py" --config-path "$config_path" --config-name "$config_name" "runner.logger.log_path=$log_dir" "$@")
printf '%q ' "${cmd[@]}" > "${log_dir}/run_vla_sft.log"
printf '\n' >> "${log_dir}/run_vla_sft.log"
"${cmd[@]}" 2>&1 | tee -a "${log_dir}/run_vla_sft.log"
