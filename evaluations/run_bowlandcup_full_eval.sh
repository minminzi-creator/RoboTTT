#!/usr/bin/env bash

set -euo pipefail

EVALUATIONS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${EVALUATIONS_PATH}")"
VENV_PATH="${VENV_PATH:-/home/xuzhexuan/project/.venv_robottt_robocasa}"

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "RoboTTT evaluation environment does not exist: ${VENV_PATH}" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

# Use all 8 H100s by default. An explicit CUDA_VISIBLE_DEVICES may remap them,
# but it must still expose exactly eight devices for this evaluation protocol.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [ "${#VISIBLE_GPUS[@]}" -ne 8 ]; then
    echo "Full BowlAndCup evaluation requires 8 visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

BASE_RUN_DIR="${BASE_RUN_DIR:-${REPO_PATH}/logs/20260924-063706-robocasa_bowlandcup_gr00t_baseline_h100/gr00t_baseline}"
ROBOTTT_RUN_DIR="${ROBOTTT_RUN_DIR:-${REPO_PATH}/logs/20260921-235012-robocasa_bowlandcup_robottt_h100/robottt}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_PATH}/logs/eval}"
EVAL_STEPS=(2000 4000 8000)

checkpoint_path() {
    local run_dir="$1"
    local step="$2"
    echo "${run_dir}/checkpoints/global_step_${step}/actor/model_state_dict/full_weights.pt"
}

# Validate every input before starting the first multi-hour rollout. This avoids
# finishing part of the six-run matrix and only then discovering a missing file.
for step in "${EVAL_STEPS[@]}"; do
    for checkpoint in \
        "$(checkpoint_path "${BASE_RUN_DIR}" "${step}")" \
        "$(checkpoint_path "${ROBOTTT_RUN_DIR}" "${step}")"; do
        if [ ! -f "${checkpoint}" ]; then
            echo "Evaluation checkpoint does not exist: ${checkpoint}" >&2
            exit 1
        fi
        actor_dir="$(dirname "$(dirname "${checkpoint}")")"
        if [ ! -f "${actor_dir}/processor_bundle.json" ] || \
           [ ! -f "${actor_dir}/sequence_metadata.json" ]; then
            echo "Checkpoint is missing its inference metadata: ${actor_dir}" >&2
            exit 1
        fi
    done
done

EXTRA_ARGS=("$@")
cd "${REPO_PATH}"

run_one_eval() {
    local label="$1"
    local config_name="$2"
    local checkpoint="$3"
    local output_dir="$4"

    echo "Evaluating ${label} on 8 GPUs with the full BowlAndCup horizon"
    LOG_PATH="${output_dir}" \
    bash evaluations/run_eval.sh \
        "${config_name}" \
        "runner.ckpt_path=${checkpoint}" \
        "env.eval.video_cfg.video_base_dir=${output_dir}/video" \
        "${EXTRA_ARGS[@]}"
}

# Keep each training step paired: baseline first, then RoboTTT on the same eight
# deterministic initial states. All six evaluations run strictly serially.
for step in "${EVAL_STEPS[@]}"; do
    run_one_eval \
        "GR00T baseline step ${step}" \
        robocasa365_bowlandcup_gr00t_baseline_h100.yaml \
        "$(checkpoint_path "${BASE_RUN_DIR}" "${step}")" \
        "${EVAL_ROOT}/baseline_step${step}"

    run_one_eval \
        "RoboTTT step ${step}" \
        robocasa365_bowlandcup_robottt_h100.yaml \
        "$(checkpoint_path "${ROBOTTT_RUN_DIR}" "${step}")" \
        "${EVAL_ROOT}/robottt_step${step}"
done
