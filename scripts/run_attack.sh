#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATTACK="${1:-BadNets}"

python "${ROOT_DIR}/attack_simulator.py" --attack "${ATTACK}" --model_dir "${ROOT_DIR}/input/model"

