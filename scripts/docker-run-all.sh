#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-poisioning:latest}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm \
  -v "${PROJECT_ROOT}":/workspace \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "python run_all_attacks.py --model_dir ./input/model"

