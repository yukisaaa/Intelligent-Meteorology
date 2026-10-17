#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
train_file="${TRAIN_FILE:-${project_dir}/../data/train_crop41_inputs.npy}"
vendor_root="${VENDOR_ROOT:-${project_dir}/vendor}"
output_root="${OUTPUT_ROOT:-${project_dir}/experiments/$(date +%Y%m%d-%H%M%S)}"
device="${DEVICE:-auto}"
model="${1:-all}"
if [[ ! -f "${train_file}" || ! -d "${vendor_root}" ]]; then
  echo "请检查 TRAIN_FILE（*_inputs.npy）与 VENDOR_ROOT；第三方源码可用 bash scripts/bootstrap_vendor.sh 下载。" >&2
  exit 2
fi
if [[ -e "${output_root}" ]]; then
  echo "输出目录已存在，请用 OUTPUT_ROOT 指定新目录：${output_root}" >&2
  exit 2
fi
models=()
case "${model}" in
  all) models=(convlstm predrnn earthformer) ;;
  convlstm|predrnn|earthformer|simvp|simvp_scalar) models=("${model}") ;;
  *) echo "用法：bash scripts/run_npy_experiment.sh [all|convlstm|predrnn|earthformer|simvp|simvp_scalar]" >&2; exit 2 ;;
esac
extra=(--split-seed 42)
[[ -n "${MAX_SAMPLES:-}" ]] && extra+=(--max-samples "${MAX_SAMPLES}")
[[ -n "${VALIDATION_FILE:-}" ]] && extra+=(--validation-file "${VALIDATION_FILE}")
[[ -n "${HOLDOUT_FILE:-}" ]] && extra+=(--holdout-file "${HOLDOUT_FILE}")
[[ -n "${TEST_A_FILE:-}" ]] && extra+=(--test-a-file "${TEST_A_FILE}")
[[ -n "${TEST_B_FILE:-}" ]] && extra+=(--test-file "${TEST_B_FILE}")
[[ "${device}" == "cuda" ]] && extra+=(--amp)
run_dirs=()
for name in "${models[@]}"; do
  case "${name}" in
    convlstm) batch_size="${BATCH_SIZE:-8}" ;;
    predrnn) batch_size="${BATCH_SIZE:-4}" ;;
    earthformer) batch_size="${BATCH_SIZE:-1}" ;;
    simvp|simvp_scalar) batch_size="${BATCH_SIZE:-8}" ;;
  esac
  "${python_bin}" "${project_dir}/train.py" --model "${name}" \
    --train-file "${train_file}" --vendor-root "${vendor_root}" \
    --output-dir "${output_root}/${name}" --index-dir "${output_root}/indexes" \
    --epochs "${EPOCHS:-100}" --batch-size "${batch_size}" \
    --crop-size "${CROP_SIZE:-41}" --device "${device}" \
    --warmup-epochs 5 --min-epochs 30 --early-stopping-patience 20 \
    "${extra[@]}"
  run_dirs+=("${output_root}/${name}")
done
"${python_bin}" "${project_dir}/compare_runs.py" "${run_dirs[@]}" \
  --output "${output_root}/model_comparison.csv"
if [[ "${model}" == "all" ]]; then
  "${python_bin}" "${project_dir}/ensemble.py" "${run_dirs[@]}" \
    --output-dir "${output_root}/ensemble"
  "${python_bin}" "${project_dir}/diagnose_runs.py" "${run_dirs[@]}" \
    --ensemble-dir "${output_root}/ensemble" --output "${output_root}/diagnosis.json"
else
  "${python_bin}" "${project_dir}/diagnose_runs.py" "${run_dirs[@]}" \
    --output "${output_root}/diagnosis.json"
fi
echo "完成：${output_root}"
