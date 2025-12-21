#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/input/model}"
WEIGHT_PATH="${MODEL_DIR}/resnet50.pth"
URL="https://download.pytorch.org/models/resnet50-0676ba61.pth"

mkdir -p "${MODEL_DIR}"

if [ -f "${WEIGHT_PATH}" ]; then
  echo "already exists: ${WEIGHT_PATH}"
  exit 0
fi

echo "downloading resnet50 weight to ${WEIGHT_PATH}"
curl -L --fail "${URL}" -o "${WEIGHT_PATH}"
echo "saved: ${WEIGHT_PATH}"

