#!/usr/bin/env bash
# ~2-5 minutes: checks the numpy simulator and the PyTorch implementation (downloads Fashion-MNIST once).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
python - <<'PY'
import sys, numpy as np
sys.path.insert(0, ".")
from fedweights.data import load_dataset, dirichlet_partition, Systems
from fedweights.models import Softmax
from fedweights.engine import Federation
from fedweights.hew import HEWConfig, run_hew
Xtr, ytr, Xte, yte = load_dataset("fmnist")
m = Softmax(Xtr.shape[1], 10, lam=1e-4); L = m.smoothness(Xtr)
parts = dirichlet_partition(ytr, 20, 1.0, np.random.default_rng(1000))
for w in ["p", "hew"]:
    sysm = Systems(20, np.random.default_rng(2000), H_set=(1, 2, 4, 8, 16), b_set=(8, 16, 32, 64, 128))
    log = run_hew(Federation(m, Xtr, ytr, Xte, yte, parts, sysm, seed=0), HEWConfig(rounds=30, theta=16.0, weight=w), L)
    print("numpy", w, "loss after 30 rounds:", round(log["loss"][-1], 4))
PY
python -m hew_torch.run --dataset fmnist --model cnn --n 10 --rounds 6 --eval_every 3 --lr 0.1 --method hew --eval_n 2000
echo "smoke test OK"
