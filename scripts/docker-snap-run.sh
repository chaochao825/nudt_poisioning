#!/usr/bin/env bash
set -euo pipefail

# Ensure we rely on the Docker binary installed via snap by default.
DOCKER_BIN=${DOCKER_BIN:-/snap/bin/docker}
IMAGE_NAME=${IMAGE_NAME:-nudt_poisoning_api:latest}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-"${HOME}/nudt_poisoning_runs"}
CONTAINER_OUTPUT=/workspace/output/docker

if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Expected docker binary at ${DOCKER_BIN}. Override DOCKER_BIN if different." >&2
  exit 1
fi

echo "[INFO] Building Docker image ${IMAGE_NAME}..."
"${DOCKER_BIN}" build -t "${IMAGE_NAME}" "${REPO_ROOT}/nudt_poisioning"

echo "[INFO] Creating output directory ${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_container() {
  local label="$1"
  shift
  echo "[INFO] Running task: ${label}"
  "${DOCKER_BIN}" run --rm \
    -v "${REPO_ROOT}":/workspace \
    -v "${OUTPUT_ROOT}":${CONTAINER_OUTPUT} \
    -w /workspace/nudt_poisioning \
    "${IMAGE_NAME}" \
    "$@"
}

run_container "batch_attacks" python run_all_attacks.py \
  --model_dir /workspace/nudt_poisioning/input/model \
  --output_path ${CONTAINER_OUTPUT}/attacks

run_container "badnets_training" python main.py \
  --input_path /workspace/nudt_poisioning/input \
  --output_path ${CONTAINER_OUTPUT}/training \
  --epochs 1 \
  --batch 16 \
  --poison_rate 0.05 \
  --method model_poisoning

for defense in isolation strip neuralcleanse; do
  run_container "defense_${defense}" python defense_simulator.py \
    --method "${defense}" \
    --output_path ${CONTAINER_OUTPUT}/defenses
done

echo "[INFO] All dockerised tasks finished. SSE summaries are under ${OUTPUT_ROOT}."
