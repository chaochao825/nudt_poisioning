#!/usr/bin/env bash
set -euo pipefail

ATTACK="${1:-BadNets}"
IMAGE="${IMAGE:-poisioning:latest}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm \
  -v "${PROJECT_ROOT}":/workspace \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "python attack_simulator.py --attack ${ATTACK} --model_dir ./input/model"

