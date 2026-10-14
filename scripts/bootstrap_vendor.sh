#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor_dir="${project_dir}/vendor"
mkdir -p "${vendor_dir}"

fetch_pinned_repository() {
  local repository_url="$1"
  local destination="$2"
  local commit="$3"

  mkdir -p "${destination}"
  if [[ ! -d "${destination}/.git" ]]; then
    git -C "${destination}" init
  fi
  if ! git -C "${destination}" remote get-url origin >/dev/null 2>&1; then
    git -C "${destination}" remote add origin "${repository_url}"
  fi

  local actual_commit
  actual_commit="$(git -C "${destination}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${actual_commit}" != "${commit}" ]]; then
    git -C "${destination}" fetch --depth 1 origin "${commit}"
    # These directories are pinned third-party vendor trees.  Force checkout
    # is intentional here so an unpacked source tree can be made reproducible.
    git -C "${destination}" checkout --detach -f FETCH_HEAD
  fi

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

external_vendor_dir="${vendor_dir}/external_models"
mkdir -p "${external_vendor_dir}"

fetch_pinned_repository \
  "https://github.com/xcw68/Fourier-NowCasting-LSTM.git" \
  "${external_vendor_dir}/Fourier-NowCasting-LSTM" \
  "f36d643190c95f2d9a0fdd8475bea0f82982ebfe"

fetch_pinned_repository \
  "https://github.com/chengtan9907/OpenSTL.git" \
  "${external_vendor_dir}/OpenSTL" \
  "eecf8a3078f0a178dbc7b28723da20f94ce36985"

fetch_pinned_repository \
  "https://github.com/yyyujintang/PostRainBench.git" \
  "${external_vendor_dir}/PostRainBench" \
  "ecc19249d998dd00fb0cb621aa1c28a1fc43db24"

fetch_pinned_repository \
  "https://github.com/Cmeo97/BlockGPT.git" \
  "${external_vendor_dir}/BlockGPT" \
  "9866759645f83b2bbc48c1335b41794fa608d429"

fetch_pinned_repository \
  "https://github.com/OpenEarthLab/CasCast.git" \
  "${external_vendor_dir}/CasCast" \
  "487a68b5ade9aa829fe7df2e8f6746b4d9acc233"

fetch_pinned_repository \
  "https://github.com/luochuyao/PFST-LSTM.git" \
  "${external_vendor_dir}/PFST-LSTM" \
  "585ebeec9e167abd31efd1e97362b08bdf324de6"
