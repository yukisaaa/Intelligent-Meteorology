#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/vdb/tianchi/cikm2017/data_new}"
crop_size="${CROP_SIZE:-41}"
train_file="${TRAIN_FILE:-${data_root}/CIKM2017_train/train_crop${crop_size}_inputs.npy}"
test_a_file="${TEST_A_FILE:-${data_root}/CIKM2017_testA/testA_crop${crop_size}_inputs.npy}"
test_b_file="${TEST_B_FILE:-${data_root}/CIKM2017_testB/testB_crop${crop_size}_inputs.npy}"
answer_a="${ANSWER_A:-${data_root}/CIKM2017_testA/answer_sample.csv}"
answer_b="${ANSWER_B:-${data_root}/CIKM2017_testB/answer_sample.csv}"
vendor_root="${VENDOR_ROOT:-${project_dir}/vendor}"
output_root="${OUTPUT_ROOT:-${project_dir}/experiments}"

for value in "${train_file}" "${test_a_file}" "${test_b_file}" "${answer_a}" \
  "${answer_b}"; do
  if [[ ! -f "${value}" ]]; then
    echo "输入文件不存在：${value}" >&2
    exit 2
  fi
done
if [[ ! -d "${vendor_root}" ]]; then
  echo "第三方模型目录不存在：${vendor_root}；请先运行 scripts/bootstrap_vendor.sh" >&2
  exit 2
fi

profile="${1:-core}"
device="${2:-cuda}"
run_id="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
output_base_root="${output_root}"
output_root="${output_base_root}/${run_id}"
python_bin="${PYTHON_BIN:-python}"
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
train_model simvp simvp-full-s42 8 --seed 42 --learning-rate 0.0005
train_model tau tau-full-s42 8 --seed 42 --learning-rate 0.0005
train_model fourcastlstm fourcastlstm-full-s42 4 --seed 42 --learning-rate 0.0003

run_dirs=(
  "${output_root}/convlstm-full-s42"
  "${output_root}/simvp-full-s42"
  "${output_root}/tau-full-s42"
  "${output_root}/fourcastlstm-full-s42"
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
  "${output_root}/simvp-full-s42" \
  "${output_root}/tau-full-s42" \
  "${output_root}/fourcastlstm-full-s42" \
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
