"""E3: improvements of debiased HEW (numpy simulator, Fashion-MNIST softmax / MLP).

Usage:
  python e3_improvements.py <suite> [--seeds 0 1] [--shard k K] [--dataset fmnist] [--model softmax]
Suites:
  sketch    gradient-sketch heterogeneity (RE / GLS / path) vs cross-round pseudo-gradient Gram
  step      adaptive server step (cross-round SNR + debiased curvature) vs fixed and decaying steps
  partial   partial participation (30% of clients per round)
  scaffold  control variates combined with HEW weights
  full      everything above on the full grid (alpha in {inf,1,0.3,0.1}, theta in {4,16}) -- serious runs
Each (config, seed, method) is written to results/e3/<suite>_<dataset>_<model>/<key>.json (resumable).
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights.data import load_dataset, dirichlet_partition, Systems
from fedweights.models import Softmax, MLP
from fedweights.engine import Federation
from fedweights.hew import HEWConfig, run_hew

ROOT = os.path.join(os.path.dirname(__file__), "..")

M = {
    "p":            dict(weight="p"),
    "fedavg":       dict(weight="p", local="raw"),
    "scaffold":     dict(weight="p", control=True),
    "iv":           dict(weight="iv"),
    "hew":          dict(weight="hew", het="cross", rule="re", timing="delayed"),      # paper's method
    "hew_post":     dict(weight="hew", het="cross", rule="re", timing="post"),
    "hew_sk":       dict(weight="hew", het="sketch", rule="re", timing="delayed"),
    "hew_sk_gls":   dict(weight="hew", het="sketch", rule="gls", timing="delayed"),
    "hew_sk_path":  dict(weight="hew", het="sketch", rule="path", timing="delayed"),
    "hew_sk_full":  dict(weight="hew", het="sketch", rule="re", timing="delayed", sketch_k=-1),
    "p_adapt":      dict(weight="p", step="adaptive"),
    "p_decay":      dict(weight="p", step="decay"),
    "hew_adapt":    dict(weight="hew", het="cross", rule="re", timing="delayed", step="adaptive"),
    "hew_decay":    dict(weight="hew", het="cross", rule="re", timing="delayed", step="decay"),
    "hew_sk_adapt": dict(weight="hew", het="sketch", rule="path", timing="delayed", step="adaptive"),
    "p_adsk":       dict(weight="p", step="adaptive_sk"),
    "hew_adsk":     dict(weight="hew", het="cross", rule="re", timing="delayed", step="adaptive_sk"),
    "hew_sk_gls_v1":  dict(weight="hew", het="sketch", rule="gls", timing="delayed", var_weight=1.0),
    "hew_gls_v1":     dict(weight="hew", het="cross", rule="gls", timing="delayed", var_weight=1.0),
    "hew_path_v1":    dict(weight="hew", het="cross", rule="path", timing="delayed", var_weight=1.0),
    "hew_v1":         dict(weight="hew", het="cross", rule="re", timing="delayed", var_weight=1.0),
    "hew_vauto":      dict(weight="hew", het="cross", rule="re", timing="delayed", var_weight="auto"),
    "hew_path_vauto": dict(weight="hew", het="cross", rule="path", timing="delayed", var_weight="auto"),
    "hew_om":         dict(weight="hew", het="cross", rule="re", timing="delayed", var_weight="omega"),
    "hew_gls_om":     dict(weight="hew", het="cross", rule="gls", timing="delayed", var_weight="omega"),
    "hew_path_om":    dict(weight="hew", het="cross", rule="path", timing="delayed", var_weight="omega"),
    "hew_sk_gls_om":  dict(weight="hew", het="sketch", rule="gls", timing="delayed", var_weight="omega"),
    "scaffold_hew": dict(weight="hew", het="cross", rule="re", timing="delayed", control=True),
}


def suite_configs(suite):
    A3 = [np.inf, 1.0, 0.1]
    if suite == "sketch":
        for a in A3:
            yield dict(alpha=a, theta=16.0, part=1.0, methods=["hew", "hew_sk", "hew_sk_gls", "hew_sk_path"])
    elif suite == "step":
        for a in A3:
            yield dict(alpha=a, theta=16.0, part=1.0, methods=["p", "p_adapt", "p_adsk", "p_decay", "hew_adapt", "hew_adsk", "hew_decay"])
    elif suite == "steptest":
        for a in [np.inf, 0.1]:
            yield dict(alpha=a, theta=16.0, part=1.0, methods=["p_adsk", "p_decay"])
    elif suite == "varw":
        for a in [1.0, 0.1, np.inf]:
            yield dict(alpha=a, theta=16.0, part=1.0, methods=["hew_path_v1", "hew_gls_v1", "hew_v1", "hew_vauto", "hew_path_vauto", "hew_sk_gls_v1"])
    elif suite == "omega":
        for a in [1.0, 0.1, np.inf]:
            yield dict(alpha=a, theta=16.0, part=1.0, methods=["hew_om", "hew_gls_om", "hew_path_om", "hew_sk_gls_om"])
    elif suite == "seeds":
        for a in [np.inf, 1.0, 0.1]:
            yield dict(alpha=a, theta=16.0, part=1.0, methods=["p", "hew", "fedavg", "scaffold", "scaffold_hew"])
    elif suite == "partial":
        for a in [1.0, 0.1]:
            yield dict(alpha=a, theta=16.0, part=0.3, methods=["p", "hew", "fedavg", "scaffold"])
    elif suite == "scaffold":
        for a in [1.0, 0.1]:
            for th in [4.0, 16.0]:
                yield dict(alpha=a, theta=th, part=1.0, methods=["scaffold", "scaffold_hew"])
    elif suite == "full":
        for a in [np.inf, 1.0, 0.3, 0.1]:
            for th in [4.0, 16.0]:
                for part in [1.0, 0.3]:
                    yield dict(alpha=a, theta=th, part=part, methods=list(M))
    else:
        raise ValueError(suite)


def fstar(model, X, y, cache):
    from scipy.optimize import minimize
    if os.path.exists(cache):
        return float(np.load(cache)[0])
    r = minimize(lambda w: (model.loss(w, X, y), model.grad(w, X, y)), np.zeros(model.dim),
                 jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10))
    np.save(cache, np.array([r.fun])); return float(r.fun)


def summarize(log, Fs):
    loss = np.array(log["loss"]); g = np.array(log["gnorm2"]); acc = np.array(log["test_acc"])
    k = max(1, len(loss) // 4)
    out = dict(rounds=log["round"], loss=loss.tolist(), gnorm2=g.tolist(), test_acc=acc.tolist(),
               floor_gnorm2=float(g[-k:].mean()), final_acc=float(acc[-k:].mean()),
               gamma=log["gamma"], tau2=log["tau2"], weights_last=log["weights"][-1] if log["weights"] else None,
               s2_last=log["s2_last"], H=log["H"], b=log["b"], m=log["m"], time=log["time"][-1])
    if Fs is not None:
        out["gap"] = (loss - Fs).tolist(); out["floor_gap"] = float((loss - Fs)[-k:].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dataset", default="fmnist")
    ap.add_argument("--model", default="softmax")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", "e3", f"{args.suite}_{args.dataset}_{args.model}")
    os.makedirs(outdir, exist_ok=True)
    Xtr, ytr, Xte, yte = load_dataset(args.dataset)
    if args.model == "softmax":
        model = Softmax(Xtr.shape[1], 10, lam=1e-4); L = model.smoothness(Xtr)
        Fs = fstar(model, Xtr, ytr, os.path.join(ROOT, "results", f"fstar_{args.dataset}_softmax_1e-4.npy"))
    else:
        model = MLP(Xtr.shape[1], 10, h=32, lam=1e-4); L = 1.0; Fs = None
    idx = 0
    for cfg in suite_configs(args.suite):
        for seed in args.seeds:
            idx += 1
            if idx % args.shard[1] != args.shard[0]:
                continue
            parts = dirichlet_partition(ytr, args.n, cfg["alpha"], np.random.default_rng(1000 + seed))
            for mname in cfg["methods"]:
                key = f"a{cfg['alpha']}_th{cfg['theta']}_part{cfg['part']}_s{seed}_{mname}"
                path = os.path.join(outdir, key + ".json")
                if os.path.exists(path):
                    continue
                sysm = Systems(args.n, np.random.default_rng(2000 + seed), H_set=(1, 2, 4, 8, 16),
                               b_set=(8, 16, 32, 64, 128), participation=cfg["part"])
                fed = Federation(model, Xtr, ytr, Xte, yte, parts, sysm, seed=seed)
                kw = dict(M[mname])
                if kw.get("sketch_k") == -1:
                    kw["sketch_k"] = model.dim
                hc = HEWConfig(rounds=args.rounds, theta=cfg["theta"], seed=seed, **kw)
                t0 = time.time()
                log = run_hew(fed, hc, L)
                res = summarize(log, Fs)
                res.update(dict(alpha=cfg["alpha"] if np.isfinite(cfg["alpha"]) else "inf", theta=cfg["theta"],
                                part=cfg["part"], seed=seed, method=mname, L=L))
                json.dump(res, open(path, "w"))
                print(f"{key}: gap={res.get('floor_gap', float('nan'))*1e3:.1f}e-3 "
                      f"g2={res['floor_gnorm2']*1e3:.3f}e-3 acc={res['final_acc']:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
