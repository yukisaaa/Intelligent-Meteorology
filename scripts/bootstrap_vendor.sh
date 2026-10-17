#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor_dir="${project_dir}/vendor"
mkdir -p "${vendor_dir}"

fetch_pinned_repository() {
  local repository_url="$1"
  local destination="$2"
  local commit="$3"

  if [[ ! -d "${destination}/.git" ]]; then
    mkdir -p "${destination}"
    git -C "${destination}" init
    git -C "${destination}" remote add origin "${repository_url}"
    git -C "${destination}" fetch --depth 1 origin "${commit}"
    git -C "${destination}" checkout --detach FETCH_HEAD
  fi

  local actual_commit
  actual_commit="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${commit}" ]]; then
    echo "第三方仓库版本不匹配：${destination}" >&2
    echo "期望 ${commit}，实际 ${actual_commit}" >&2
    echo "请备份本地修改后删除该 vendor 子目录，再重新运行本脚本。" >&2
    exit 3
  fi
}

fetch_pinned_repository \
  "https://github.com/ndrplz/ConvLSTM_pytorch.git" \
  "${vendor_dir}/ConvLSTM_pytorch" \
  "9f662ba24b0a38e82cf0f5208c7f4859deb85ffe"

fetch_pinned_repository \
  "https://github.com/thuml/predrnn-pytorch.git" \
  "${vendor_dir}/predrnn-pytorch" \
  "49d3b0f3a2757ad9c299716ec455a96868455d67"

fetch_pinned_repository \
  "https://github.com/amazon-science/earth-forecasting-transformer.git" \
  "${vendor_dir}/earth-forecasting-transformer" \
  "7732b03bdb366110563516c3502315deab4c2026"

fetch_pinned_repository \
  "https://github.com/chengtan9907/OpenSTL.git" \
  "${vendor_dir}/external_models/SimVP" \
  "eecf8a3078f0a178dbc7b28723da20f94ce36985"
