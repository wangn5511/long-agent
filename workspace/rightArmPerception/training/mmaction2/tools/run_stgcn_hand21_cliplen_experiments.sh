#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${1:-cliplen_compare_${TIMESTAMP}}"

declare -A CONFIG_MAP=(
  [u36]="configs/skeleton/stgcn/stgcn_hand21-joint-u36-30e_custom-keypoint-2d.py"
  [u16]="configs/skeleton/stgcn/stgcn_hand21-joint-u16-30e_custom-keypoint-2d.py"
  [u8]="configs/skeleton/stgcn/stgcn_hand21-joint-u8-30e_custom-keypoint-2d.py"
)

print_usage() {
  cat <<EOF
Usage:
  bash tools/run_stgcn_hand21_cliplen_experiments.sh [run_tag]

Examples:
  bash tools/run_stgcn_hand21_cliplen_experiments.sh
  bash tools/run_stgcn_hand21_cliplen_experiments.sh exp_seed0

Environment variables:
  PYTHON_BIN   Python executable to use. Default: python
  TIMESTAMP    Override timestamp suffix used in default run_tag
EOF
}

run_train() {
  local clip_tag="$1"
  local config_rel="${CONFIG_MAP[$clip_tag]}"
  local work_dir="work_dirs/stgcn_hand21_${clip_tag}_${RUN_TAG}"

  echo "============================================================"
  echo "Training ${clip_tag}"
  echo "Config   : ${config_rel}"
  echo "Work dir : ${work_dir}"
  echo "============================================================"

  "${PYTHON_BIN}" tools/train.py "${config_rel}" --work-dir "${work_dir}"
}

run_test() {
  local clip_tag="$1"
  local config_rel="${CONFIG_MAP[$clip_tag]}"
  local work_dir="work_dirs/stgcn_hand21_${clip_tag}_${RUN_TAG}"
  local best_ckpt

  best_ckpt="$(find "${ROOT_DIR}/${work_dir}" -maxdepth 1 -name 'best_acc_top1_epoch_*.pth' | sort | tail -n 1)"
  if [[ -z "${best_ckpt}" ]]; then
    echo "No best checkpoint found for ${clip_tag} under ${work_dir}" >&2
    exit 1
  fi

  echo "------------------------------------------------------------"
  echo "Testing ${clip_tag}"
  echo "Checkpoint: ${best_ckpt}"
  echo "------------------------------------------------------------"

  "${PYTHON_BIN}" tools/test.py "${config_rel}" "${best_ckpt}" --work-dir "${work_dir}"
}

summarize_results() {
  local summary_file="work_dirs/stgcn_hand21_${RUN_TAG}_summary.txt"

  : > "${ROOT_DIR}/${summary_file}"

  for clip_tag in u36 u16 u8; do
    local work_dir="${ROOT_DIR}/work_dirs/stgcn_hand21_${clip_tag}_${RUN_TAG}"
    local best_ckpt
    best_ckpt="$(find "${work_dir}" -maxdepth 1 -name 'best_acc_top1_epoch_*.pth' | sort | tail -n 1)"

    {
      echo "[${clip_tag}]"
      echo "work_dir=${work_dir}"
      echo "best_ckpt=${best_ckpt}"
      if [[ -f "${work_dir}/confusion_matrix_test_summary.txt" ]]; then
        echo "per_class_summary=${work_dir}/confusion_matrix_test_summary.txt"
      fi
      if [[ -f "${work_dir}/misclassified_test.txt" ]]; then
        echo "misclassified=${work_dir}/misclassified_test.txt"
      fi
      if [[ -f "${work_dir}/classified_test.txt" ]]; then
        echo "classified=${work_dir}/classified_test.txt"
      fi
      echo
    } >> "${ROOT_DIR}/${summary_file}"
  done

  echo "Summary written to ${summary_file}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

cd "${ROOT_DIR}"

for clip_tag in u36 u16 u8; do
  run_train "${clip_tag}"
  run_test "${clip_tag}"
done

summarize_results
