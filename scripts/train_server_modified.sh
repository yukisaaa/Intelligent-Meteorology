#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/vdb/tianchi/cikm2017/data_new}"
crop_size="${CROP_SIZE:-41}"
train_file="${TRAIN_FILE:-${data_root}/CIKM2017_train/train_crop${crop_size}_inputs.npy}"
test_file="${TEST_B_FILE:-${data_root}/CIKM2017_testB/testB_crop${crop_size}_inputs.npy}"
test_a_file="${TEST_A_FILE:-${data_root}/CIKM2017_testA/testA_crop${crop_size}_inputs.npy}"
vendor_root="${VENDOR_ROOT:-${project_dir}/vendor}"
output_root="${OUTPUT_ROOT:-${project_dir}/experiments}"

for path in "${train_file}" "${test_file}" "${test_a_file}"; do
  if [[ ! -f "${path}" ]]; then
    echo "数据文件不存在：${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${vendor_root}" ]]; then
  echo "第三方模型目录不存在：${vendor_root}；请先运行 scripts/bootstrap_vendor.sh" >&2
  exit 2
fi
model="${1:-convlstm}"
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
  simvp)
    python train_modified.py --model simvp --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/simvp" \
      --epochs 30 --batch-size 8 --crop-size "${crop_size}" --learning-rate 0.0005 --device cuda
    ;;
  tau)
    python train_modified.py --model tau --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/tau" \
      --epochs 30 --batch-size 8 --crop-size "${crop_size}" --learning-rate 0.0005 --device cuda
    ;;
  fourcastlstm)
    python train_modified.py --model fourcastlstm --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/fourcastlstm" \
      --epochs 30 --batch-size 4 --crop-size "${crop_size}" --learning-rate 0.0003 --device cuda
    ;;
  cascast)
    python train_modified.py --model cascast --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/cascast" \
      --epochs 30 --batch-size 1 --crop-size "${crop_size}" --learning-rate 0.0001 --device cuda
    ;;
  postrainbench)
    python train_modified.py --model postrainbench --train-file "${train_file}" --test-file "${test_file}" \
      --test-a-file "${test_a_file}" --vendor-root "${vendor_root}" \
      --output-dir "${output_root}/postrainbench" \
      --epochs 30 --batch-size 1 --crop-size "${crop_size}" --learning-rate 0.0001 --device cuda
    ;;
  *)
    echo "未知模型；可选 convlstm、predrnn、earthformer、simvp、tau、fourcastlstm、cascast、postrainbench" >&2
    exit 2
    ;;
esac
