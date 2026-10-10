#!/usr/bin/env bash
set -euo pipefail

train_file="${TRAIN_FILE:-#TODO}"
test_file="${TEST_B_FILE:-#TODO}"
test_a_file="${TEST_A_FILE:-#TODO}"
vendor_root="${VENDOR_ROOT:-#TODO}"
output_root="${OUTPUT_ROOT:-#TODO}"

if [[ "${train_file}" == "#TODO" || "${test_file}" == "#TODO" || \
      "${test_a_file}" == "#TODO" || \
      "${vendor_root}" == "#TODO" || "${output_root}" == "#TODO" ]]; then
  echo "请设置 TRAIN_FILE、TEST_A_FILE、TEST_B_FILE、VENDOR_ROOT 和 OUTPUT_ROOT" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-convlstm}"
crop_size="${CROP_SIZE:-41}"
run_id="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
output_root="${output_root}/${run_id}"
if [[ -e "${output_root}" ]]; then
  echo "输出目录已存在：${output_root}；请设置不同的 RUN_ID" >&2
  exit 2
fi
cd "${project_dir}"

case "${model}" in
  convlstm)
    python train_modified.py --model convlstm --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/convlstm" \
      --epochs 30 --batch-size 8 --crop-size "${crop_size}" --hidden-dims 16,32 --device cuda
    ;;
  predrnn)
    python train_modified.py --model predrnn --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/predrnn" \
      --epochs 30 --batch-size 4 --crop-size "${crop_size}" --hidden-dims 32,32 --device cuda
    ;;
  earthformer)
    python train_modified.py --model earthformer --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/earthformer" \
      --epochs 30 --batch-size 1 --crop-size "${crop_size}" --earthformer-base-units 32 \
      --learning-rate 0.0003 --device cuda
    ;;
  *)
    echo "未知模型：${model}；可选 convlstm、predrnn、earthformer" >&2
    exit 2
    ;;
esac
