#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/input/model}"

mkdir -p "${MODEL_DIR}"

if [ "$#" -eq 0 ]; then
  MODELS=("resnet50" "vgg16" "inception_v3")
else
  MODELS=("$@")
fi

download_model() {
  local name_raw="$1"
  local name
  local url
  local file

  case "${name_raw}" in
    resnet|resnet50|resnet50_v1|resnet50-stream|resnet50_stream|resnet50-sync|resnet50_sync)
      name="resnet50"
      url="https://download.pytorch.org/models/resnet50-0676ba61.pth"
      ;;
    vgg|vgg16|vgg16_bn)
      name="vgg16"
      url="https://download.pytorch.org/models/vgg16-397923af.pth"
      ;;
    inception|inception_v3|inception-v3|googlenet)
      name="inception_v3"
      url="https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
      ;;
    *)
      echo "unsupported model: ${name_raw}" >&2
      return 1
      ;;
  esac

  file="${MODEL_DIR}/${name}.pth"
  if [ -f "${file}" ]; then
    echo "already exists: ${file}"
    return 0
  fi

  echo "downloading ${name} weight to ${file}"
  curl -L --fail "${url}" -o "${file}"
  echo "saved: ${file}"
}

for model_name in "${MODELS[@]}"; do
  download_model "${model_name}"
done

