#!/usr/bin/env bash
set -euo pipefail

train_file="#TODO"
test_file="#TODO"
vendor_root="#TODO"
output_root="#TODO"

if [[ "${train_file}" == "#TODO" || "${test_file}" == "#TODO" || \
      "${vendor_root}" == "#TODO" || "${output_root}" == "#TODO" ]]; then
  echo "请先把 scripts/train_server.sh 顶部所有 #TODO 替换为服务器实际路径" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-convlstm}"
cd "${project_dir}"

case "${model}" in
  convlstm)
    python train.py --model convlstm --train-file "${train_file}" --test-file "${test_file}" \
      --vendor-root "${vendor_root}" --output-dir "${output_root}/convlstm" \
      --epochs 30 --batch-size 8 --crop-size 41 --hidden-dims 16,32 --device cuda
    ;;
  predrnn)
    python train.py --model predrnn --train-file "${train_file}" --test-file "${test_file}" \
      --vendor-root "${vendor_root}" --output-dir "${output_root}/predrnn" \
      --epochs 30 --batch-size 4 --crop-size 41 --hidden-dims 32,32 --device cuda
    ;;
  earthformer)
    python train.py --model earthformer --train-file "${train_file}" --test-file "${test_file}" \
      --vendor-root "${vendor_root}" --output-dir "${output_root}/earthformer" \
      --epochs 30 --batch-size 1 --crop-size 41 --earthformer-base-units 32 \
      --learning-rate 0.0003 --device cuda
    ;;
  *)
    echo "未知模型：${model}；可选 convlstm、predrnn、earthformer" >&2
    exit 2
    ;;
esac
