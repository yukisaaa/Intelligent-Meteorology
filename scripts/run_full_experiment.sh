#!/usr/bin/env bash
set -euo pipefail

train_file="#TODO"
test_a_file="#TODO"
test_b_file="#TODO"
answer_a="#TODO"
answer_b="#TODO"
vendor_root="#TODO"
output_root="#TODO"

for value in "${train_file}" "${test_a_file}" "${test_b_file}" "${answer_a}" \
  "${answer_b}" "${vendor_root}" "${output_root}"; do
  if [[ "${value}" == "#TODO" ]]; then
    echo "请先填写 scripts/run_full_experiment.sh 顶部全部 #TODO" >&2
    exit 2
  fi
done

profile="${1:-core}"
device="${2:-cuda}"
python_bin="${PYTHON_BIN:-python}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
index_dir="${output_root}/indexes"
mkdir -p "${output_root}"
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
  "${python_bin}" train.py \
    --model "${model}" \
    --train-file "${train_file}" \
    --test-file "${test_b_file}" \
    --vendor-root "${vendor_root}" \
    --index-dir "${index_dir}" \
    --output-dir "${output_root}/${run_name}" \
    --epochs 50 \
    --batch-size "${batch_size}" \
    --crop-size 41 \
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
    "${output_root}/convlstm-no-attention-s42"
    "${output_root}/convlstm-no-multiscale-s42"
    "${output_root}/convlstm-avg-only-s42"
  )
elif [[ "${profile}" != "core" ]]; then
  echo "profile 只能是 core 或 paper" >&2
  exit 2
fi

"${python_bin}" compare_runs.py "${run_dirs[@]}" \
  --output "${output_root}/reports/model_comparison.csv"
"${python_bin}" report_experiments.py "${run_dirs[@]}" \
  --output-dir "${output_root}/reports"
"${python_bin}" ensemble.py \
  "${output_root}/convlstm-full-s42" \
  "${output_root}/predrnn-full-s42" \
  "${output_root}/earthformer-full-s42" \
  --output-dir "${output_root}/ensemble"
"${python_bin}" validate_submission.py \
  --prediction-file "${output_root}/ensemble/test_predictions.csv" \
  --answer-sample "${answer_b}" \
  --report-file "${output_root}/ensemble/submission_validation.json"
"${python_bin}" make_risk_report.py \
  --prediction-file "${output_root}/ensemble/test_predictions.csv" \
  --config "${output_root}/convlstm-full-s42/config.json" \
  --output-file "${output_root}/ensemble/risk_report.csv"

echo "全部实验完成：${output_root}"
