"""Measure smoothness along trajectories: max over clients of lambda_max(Hessian of F_i) at iterates,
to express theta in trajectory units (Remark 'Which L')."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights.data import load_dataset, dirichlet_partition, Systems
from fedweights.models import Softmax
from fedweights.engine import Federation
from fedweights.hew import HEWConfig, run_hew
import fedweights.hew as HW

def lmax_hessian(model, w, X, y, iters=40, eps=1e-4, seed=0):
    v = np.random.default_rng(seed).standard_normal(w.size); v /= np.linalg.norm(v); lam = 0.0
    for _ in range(iters):
        Hv = (model.grad(w + eps * v, X, y) - model.grad(w - eps * v, X, y)) / (2 * eps)
        lam = float(v @ Hv); v = Hv / (np.linalg.norm(Hv) + 1e-30)
    return lam

Xtr, ytr, Xte, yte = load_dataset("fmnist"); model = Softmax(Xtr.shape[1], 10, lam=1e-4); L = model.smoothness(Xtr)
out = {}
for alpha in [np.inf, 0.1]:
    parts = dirichlet_partition(ytr, 20, alpha, np.random.default_rng(1000))
    sysm = Systems(20, np.random.default_rng(2000), H_set=(1, 2, 4, 8, 16), b_set=(8, 16, 32, 64, 128))
    fed = Federation(model, Xtr, ytr, Xte, yte, parts, sysm, seed=0)
    snaps = {}
    orig = fed.full_metrics
    def fm(w, _o=orig, _s=snaps, _c=[0]):
        _s[_c[0]] = w.copy(); _c[0] += 10; return _o(w)
    fed.full_metrics = fm
    run_hew(fed, HEWConfig(rounds=300, theta=16.0, weight="hew", eval_every=10), L)
    res = {}
    for r in [0, 20, 100, 300]:
        lam = max(lmax_hessian(model, snaps[r], fed.Xc[i], fed.yc[i]) for i in range(20))
        res[r] = lam
    out["inf" if not np.isfinite(alpha) else str(alpha)] = res
    print(alpha, {k: round(v, 4) for k, v in res.items()}, "L_worst=", round(L, 4), flush=True)
json.dump(dict(L=L, lmax=out), open(os.path.join(os.path.dirname(__file__), "..", "results", "smoothness.json"), "w"))
