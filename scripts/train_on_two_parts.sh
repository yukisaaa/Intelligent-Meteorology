#!/usr/bin/env bash
set -euo pipefail
# Place this file in scripts/ beside run_npy_experiment.sh.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
split_dir="${SPLIT_DIR:?Set SPLIT_DIR to the directory produced by split_train_to_npy.py}"
crop_size="${CROP_SIZE:-41}"
split_dir="$(cd "${split_dir}" && pwd)"
export TRAIN_FILE="${split_dir}/train_crop${crop_size}_inputs.npy"
export HOLDOUT_FILE="${split_dir}/fusion_eval_crop${crop_size}_inputs.npy"
if [[ ! -f "${TRAIN_FILE}" || ! -f "${HOLDOUT_FILE}" ]]; then
  echo "找不到两部分 NPY，请检查 SPLIT_DIR 与 CROP_SIZE" >&2
  exit 2
fi
# The tuning subset is drawn only from the development file, never the final set.
unset VALIDATION_FILE
exec bash "${project_dir}/scripts/run_npy_experiment.sh" "${1:-all}"
