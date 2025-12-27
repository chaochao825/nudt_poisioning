#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/run_all_attacks.py" --model_dir "${ROOT_DIR}/input/model"

