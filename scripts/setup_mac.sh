#!/usr/bin/env bash
# One-time setup on macOS (Apple silicon: PyTorch uses the Metal/MPS backend automatically).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
PY
