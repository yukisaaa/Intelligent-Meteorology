#!/usr/bin/env bash
set -euo pipefail

train_file="${TRAIN_FILE:-#TODO}"
test_a_file="${TEST_A_FILE:-#TODO}"
test_b_file="${TEST_B_FILE:-#TODO}"
answer_a="${ANSWER_A:-#TODO}"
answer_b="${ANSWER_B:-#TODO}"
vendor_root="${VENDOR_ROOT:-#TODO}"
output_root="${OUTPUT_ROOT:-#TODO}"

for value in "${train_file}" "${test_a_file}" "${test_b_file}" "${answer_a}" \
  "${answer_b}" "${vendor_root}" "${output_root}"; do
  if [[ "${value}" == "#TODO" ]]; then
    echo "请设置 TRAIN_FILE、TEST_A_FILE、TEST_B_FILE、ANSWER_A、ANSWER_B、VENDOR_ROOT 和 OUTPUT_ROOT" >&2
    exit 2
  fi
done

profile="${1:-core}"
device="${2:-cuda}"
crop_size="${CROP_SIZE:-41}"
run_id="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
output_base_root="${output_root}"
output_root="${output_base_root}/${run_id}"
python_bin="${PYTHON_BIN:-python}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
index_dir="${output_base_root}/indexes"
if [[ -e "${output_root}" ]]; then
  echo "输出目录已存在：${output_root}；请设置不同的 RUN_ID" >&2
  exit 2
fi
mkdir -p "${output_root}"
mkdir -p "${index_dir}"
cd "${project_dir}"

amp_args=()
if [[ "${device}" == "cuda" ]]; then
  amp_args+=(--amp)
fi

"${python_bin}" data_audit.py \
  --train-file "${train_file}" \
  --test-a-file "${test_a_file}" \
  --test-b-file "${test_b_file}" \
  --answer-a "${answer_a}" \
  --answer-b "${answer_b}" \
  --index-dir "${index_dir}" \
  --output-dir "${output_root}/data-audit"

train_model() {
  local model="$1"
  local run_name="$2"
  local batch_size="$3"
  shift 3
  "${python_bin}" train_modified.py \
    --model "${model}" \
    --train-file "${train_file}" \
    --test-file "${test_b_file}" \
    --test-a-file "${test_a_file}" \
    --vendor-root "${vendor_root}" \
    --index-dir "${index_dir}" \
    --output-dir "${output_root}/${run_name}" \
    --epochs 50 \
    --batch-size "${batch_size}" \
    --crop-size "${crop_size}" \
    --split-strategy stratified \
    --early-stopping-patience 8 \
    --device "${device}" \
    "${amp_args[@]}" \
    "$@"
}

train_model convlstm convlstm-full-s42 8 --seed 42 --hidden-dims 16,32
train_model predrnn predrnn-full-s42 4 --seed 42 --hidden-dims 32,32
train_model earthformer earthformer-full-s42 1 --seed 42 \
  --earthformer-base-units 32 --learning-rate 0.0003

run_dirs=(
  "${output_root}/convlstm-full-s42"
  "${output_root}/predrnn-full-s42"
  "${output_root}/earthformer-full-s42"
)

if [[ "${profile}" == "paper" ]]; then
  train_model convlstm convlstm-full-s2026 8 --seed 2026 --hidden-dims 16,32 --skip-test-prediction
  train_model convlstm convlstm-full-s3407 8 --seed 3407 --hidden-dims 16,32 --skip-test-prediction
  train_model convlstm convlstm-smooth-l1-s42 8 --seed 42 --hidden-dims 16,32 \
    --loss smooth_l1 --skip-test-prediction
  train_model convlstm convlstm-mse-s42 8 --seed 42 --hidden-dims 16,32 \
    --loss mse --skip-test-prediction
  train_model convlstm convlstm-no-attention-s42 8 --seed 42 --hidden-dims 16,32 \
    --head-variant no_attention --skip-test-prediction
  train_model convlstm convlstm-no-multiscale-s42 8 --seed 42 --hidden-dims 16,32 \
    --head-variant no_multiscale --skip-test-prediction
  train_model convlstm convlstm-avg-only-s42 8 --seed 42 --hidden-dims 16,32 \
    --head-variant avg_only --skip-test-prediction
  run_dirs+=(
    "${output_root}/convlstm-full-s2026"
    "${output_root}/convlstm-full-s3407"
    "${output_root}/convlstm-smooth-l1-s42"
    "${output_root}/convlstm-mse-s42"
    "${output_root}/convlstm-no-attention-s42"
    "${output_root}/convlstm-no-multiscale-s42"
    "${output_root}/convlstm-avg-only-s42"
  )
elif [[ "${profile}" != "core" ]]; then
  echo "profile 只能是 core 或 paper" >&2
  exit 2
fi

"${python_bin}" compare_runs_modified.py "${run_dirs[@]}" \
  --output "${output_root}/reports/model_comparison.csv"
"${python_bin}" report_experiments_modified.py "${run_dirs[@]}" \
  --output-dir "${output_root}/reports"
"${python_bin}" ensemble_modified.py \
  "${output_root}/convlstm-full-s42" \
  "${output_root}/predrnn-full-s42" \
  "${output_root}/earthformer-full-s42" \
  --output-dir "${output_root}/ensemble"
"${python_bin}" validate_submission.py \
  --prediction-file "${output_root}/ensemble/test_predictions.csv" \
  --answer-sample "${answer_b}" \
  --report-file "${output_root}/ensemble/submission_validation.json"
"${python_bin}" validate_submission.py \
  --prediction-file "${output_root}/ensemble/testA_predictions.csv" \
  --answer-sample "${answer_a}" \
  --report-file "${output_root}/ensemble/testA_submission_validation.json"
"${python_bin}" make_risk_report.py \
  --prediction-file "${output_root}/ensemble/test_predictions.csv" \
  --config "${output_root}/convlstm-full-s42/config.json" \
  --output-file "${output_root}/ensemble/risk_report.csv"

echo "全部实验完成：${output_root}"
