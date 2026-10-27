#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
train_file="${TRAIN_FILE:?Set TRAIN_FILE to the training *_inputs.npy}"
vendor_root="${VENDOR_ROOT:-${project_dir}/vendor}"
output_root="${OUTPUT_ROOT:-${project_dir}/experiments/simvp-comparison-$(date +%Y%m%d-%H%M%S)}"
device="${DEVICE:-auto}"
if [[ -e "${output_root}" ]]; then
  echo "请使用新的 OUTPUT_ROOT：${output_root}" >&2
  exit 2
fi
extra=(--split-seed 42 --seed 42)
[[ -n "${MAX_SAMPLES:-}" ]] && extra+=(--max-samples "${MAX_SAMPLES}")
[[ -n "${VALIDATION_FILE:-}" ]] && extra+=(--validation-file "${VALIDATION_FILE}")
[[ -n "${HOLDOUT_FILE:-}" ]] && extra+=(--holdout-file "${HOLDOUT_FILE}")
[[ "${device}" == "cuda" ]] && extra+=(--amp)
run_dirs=()
for loss in heavy_rain mse; do
  for model in simvp simvp_scalar; do
    run_dir="${output_root}/${model}-${loss}"
    "${python_bin}" "${project_dir}/train.py" --model "${model}" --loss "${loss}" \
      --train-file "${train_file}" --vendor-root "${vendor_root}" --output-dir "${run_dir}" \
      --index-dir "${output_root}/indexes" --epochs "${EPOCHS:-100}" \
      --batch-size "${BATCH_SIZE:-8}" --crop-size "${CROP_SIZE:-41}" --device "${device}" \
      --learning-rate 1e-4 --warmup-epochs 5 --min-epochs 30 --early-stopping-patience 20 \
      "${extra[@]}"
    run_dirs+=("${run_dir}")
  done
done
"${python_bin}" "${project_dir}/compare_runs.py" "${run_dirs[@]}" --output "${output_root}/comparison.csv"
"${python_bin}" "${project_dir}/diagnose_runs.py" "${run_dirs[@]}" --output "${output_root}/diagnosis.json"
echo "四组对照完成：${output_root}"
