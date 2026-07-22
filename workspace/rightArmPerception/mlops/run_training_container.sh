#!/usr/bin/env bash
set -euo pipefail

image="${GESTURE_TRAINING_IMAGE:-stgcn_openmmlab:v1}"
gpu_devices="${GESTURE_GPU_DEVICES:-0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_root="${GESTURE_RESULTS_ROOT:-${repo_root}/.model-agent-results}"
outputs_root="${GESTURE_OUTPUTS_ROOT:-${repo_root}/.model-agent-outputs}"
datasets_root="${GESTURE_DATASETS_ROOT:-}"

mkdir -p "${results_root}" "${outputs_root}"

if [[ $# -eq 0 ]]; then
  set -- bash
fi

docker_args=(
  docker run --rm --init
  --gpus "device=${gpu_devices}"
  --shm-size 16g
  --label openhands.task=rightarm-gesture-training
  -v "${repo_root}:/workspace/rightArmPerception:rw"
  -v "${results_root}:/home/results:rw"
  -v "${outputs_root}:/home/outputs:rw"
  -w /workspace/rightArmPerception/training/mmaction2
)

if [[ -n "${datasets_root}" ]]; then
  docker_args+=(-v "${datasets_root}:/home/datasets:ro")
fi

exec "${docker_args[@]}" "${image}" "$@"
