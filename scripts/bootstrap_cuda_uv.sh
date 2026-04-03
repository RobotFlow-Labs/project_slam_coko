#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CUDA_WHL_INDEX="${CUDA_WHL_INDEX:-https://download.pytorch.org/whl/cu124}"

uv venv --python "${PYTHON_BIN}"
uv sync --extra dev --extra paper
uv pip install --python .venv/bin/python --index-url "${CUDA_WHL_INDEX}" \
  torch torchvision torchaudio

echo "CUDA-ready environment created in .venv using ${CUDA_WHL_INDEX}"
