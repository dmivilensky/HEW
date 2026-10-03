"""E2-E4: real-data experiments (FMNIST / MNIST; softmax regression and tanh-MLP).

Usage:
  python e2_grid.py <suite> [--seeds 0 1] [--dataset fmnist] [--model softmax]
Suites:
  grid       regime map: Dirichlet alpha x step size, all weighting rules (E2)
  decay      decaying step size: crossover from inverse-variance to objective weights (E3)
  partial    partial participation + stragglers (E4)
  mlp        nonconvex model, subset of rules (E2b)
Each (config, seed, method) result is written to results/e2/<suite>/<key>.json, so
interrupted runs resume where they stopped.
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights.data import load_dataset, dirichlet_partition, Systems
from fedweights.models import Softmax, MLP
from fedweights.engine import Federation, run
from scipy.optimize import minimize

ROOT = os.path.join(os.path.dirname(__file__), "..")

METHODS = {
    "p":          dict(method="p"),
    "work":       dict(method="work"),
    "iv":         dict(method="iv"),
    "re":         dict(method="re"),
    "re_oracle":  dict(method="re_oracle"),
    "gls_oracle": dict(method="gls_oracle"),
    "posthoc_hew": dict(method="posthoc_hew"),
    "fedadp":     dict(method="fedadp"),
    "scaffold":   dict(method="p", corrected=True),
    "scaffold_iv": dict(method="iv", corrected=True),
    "scaffold_re": dict(method="re", corrected=True),
    "fedavg":     dict(method="p", raw=True),
    "gls":        dict(method="gls"),
    "relmax":     dict(method="relmax"),
    "re_lmax_oracle": dict(method="re_lmax_oracle"),
    "p_log":      dict(method="p"),
    "hew":        dict(method="hew"),
    "hew_gls":    dict(method="hew_gls"),
    "hew_fresh":  dict(method="hew_fresh"),
    "hew_nodebias": dict(method="hew_nodebias"),
    "hewx":       dict(method="hewx"),
    "hewx_gls":   dict(method="hewx_gls"),
    "hewx_reg":   dict(method="hewx_reg"),
    "phew":       dict(method="phew"),
    "phew_raw":   dict(method="phew_raw"),
    "phew_psd":   dict(method="phew_psd"),
    "hewx_lag":   dict(method="hewx_lag"),
}


def fstar(model, X, y, cache):
    if os.path.exists(cache):
        return float(np.load(cache)[0])
    r = minimize(lambda w: (model.loss(w, X, y), model.grad(w, X, y)), np.zeros(model.dim),
                 jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10))
    np.save(cache, np.array([r.fun]))
    return float(r.fun)


def summarize(log, Fs):
    loss = np.array(log["loss"]); g = np.array(log["gnorm2"]); acc = np.array(log["test_acc"])
    k = max(1, len(loss) // 4)
    out = dict(rounds=log["round"], loss=loss.tolist(), gnorm2=g.tolist(), test_acc=acc.tolist(),
               floor_gnorm2=float(g[-k:].mean()), final_acc=float(acc[-k:].mean()),
               val_pred=log["val_pred"], tau2_hat=log["tau2_hat"], trZ=log["trZ"], lmaxZ=log["lmaxZ"],
               risk_w=log["risk_w"], risk_opt=log["risk_opt"],
               weights_last=log["weights"][-1] if log["weights"] else None,
               weights=log["weights"], s2_oracle=log["s2_oracle"], s2_hat=log["s2_hat"],
               time=log["time"][-1])
    if Fs is not None:
        out["gap"] = (loss - Fs).tolist()
        out["floor_gap"] = float((loss - Fs)[-k:].mean())
    return out


def suite_configs(suite):
    if suite == "grid":
        for alpha in [np.inf, 1.0, 0.3, 0.1]:
            for theta in [4.0, 16.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300,
                           methods=["p", "work", "iv", "re", "gls_oracle", "posthoc_hew",
                                    "fedadp", "scaffold", "scaffold_iv", "fedavg"])
    elif suite == "grid2":
        for alpha in [np.inf, 1.0, 0.3, 0.1]:
            for theta in [4.0, 16.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300,
                           methods=["gls", "relmax", "re_lmax_oracle", "p_log"])
    elif suite == "hew":
        for alpha in [np.inf, 1.0, 0.1]:
            for theta in [16.0, 4.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300,
                           methods=["hew", "hew_gls", "hew_fresh", "hew_nodebias"])
    elif suite == "hewx":
        for alpha in [np.inf, 1.0, 0.1]:
            for theta in [16.0, 4.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300, methods=["hewx", "hewx_gls"])
    elif suite == "hewreg":
        for alpha in [np.inf, 0.1, 1.0]:
            for theta in [16.0, 4.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300, methods=["hewx_reg"])
    elif suite == "phew":
        for alpha in [np.inf, 0.1, 1.0]:
            for theta in [16.0, 4.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300, methods=["phew"])
    elif suite == "phew2":
        for alpha in [0.1, 1.0]:
            yield dict(alpha=alpha, theta=16.0, sys="Hb", rounds=300, methods=["phew_psd"])
    elif suite == "hewlag":
        for alpha in [np.inf, 0.1, 1.0]:
            for theta in [16.0, 4.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300, methods=["hewx_lag"])
    elif suite == "tune":
        for alpha in [np.inf, 1.0, 0.1]:
            for theta in [8.0, 32.0]:
                yield dict(alpha=alpha, theta=theta, sys="Hb", rounds=300,
                           methods=["p", "relmax", "gls", "posthoc_hew", "scaffold", "fedavg"])
    elif suite == "decay":
        for alpha in [1.0, 0.3]:
            yield dict(alpha=alpha, theta="decay", sys="Hb", rounds=600,
                       methods=["p", "iv", "relmax", "gls", "gls_oracle"])
    elif suite == "partial":
        for alpha in [np.inf, 0.3]:
            yield dict(alpha=alpha, theta=8.0, sys="Hb_straggle_partial", rounds=400,
                       methods=["p", "work", "iv", "gls", "relmax", "posthoc_hew", "fedadp", "scaffold",
                                "fedavg"])
    elif suite == "mlp":
        for alpha in [np.inf, 0.3]:
            yield dict(alpha=alpha, theta=6.0, sys="Hb", rounds=300,
                       methods=["p", "iv", "relmax", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"])


def make_systems(kind, n, rng):
    if kind == "Hb":
        return Systems(n, rng, H_set=(1, 2, 4, 8, 16), b_set=(8, 16, 32, 64, 128))
    if kind == "Hb_straggle_partial":
        return Systems(n, rng, H_set=(1, 2, 4, 8, 16), b_set=(8, 16, 32, 64, 128), redraw=True,
                       participation=0.3)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dataset", default="fmnist")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], help="k K: run configs with idx%%K==k")
    args = ap.parse_args()
    model_kind = args.model or ("mlp" if args.suite == "mlp" else "softmax")
    sdir = "grid" if args.suite in ("grid2", "tune", "hew", "hewx", "hewreg", "phew", "phew2", "hewlag") else args.suite
    outdir = os.path.join(ROOT, "results", "e2", f"{sdir}_{args.dataset}_{model_kind}")
    os.makedirs(outdir, exist_ok=True)
    Xtr, ytr, Xte, yte = load_dataset(args.dataset)
    if model_kind == "softmax":
        model = Softmax(Xtr.shape[1], 10, lam=1e-4)
        L = model.smoothness(Xtr)
        Fs = fstar(model, Xtr, ytr, os.path.join(ROOT, "results", f"fstar_{args.dataset}_softmax_1e-4.npy"))
    else:
        model = MLP(Xtr.shape[1], 10, h=32, lam=1e-4)
        L = 1.0                                   # step scale absorbed in theta
        Fs = None
    idx = 0
    for cfg in suite_configs(args.suite):
        for seed in args.seeds:
            idx += 1
            if idx % args.shard[1] != args.shard[0]:
                continue
            rng = np.random.default_rng(1000 + seed)
            parts = dirichlet_partition(ytr, args.n, cfg["alpha"], rng)
            sysm_state = np.random.default_rng(2000 + seed)
            theta = cfg["theta"]
            if theta == "decay":
                theta_fn = lambda t: 16.0 / np.sqrt(1.0 + t / 20.0)
            else:
                theta_fn = float(theta)
            for mname in cfg["methods"]:
                key = f"a{cfg['alpha']}_th{theta}_{cfg['sys']}_s{seed}_{mname}"
                path = os.path.join(outdir, key + ".json")
                if os.path.exists(path):
                    continue
                sysm = make_systems(cfg["sys"], args.n, np.random.default_rng(2000 + seed))
                fed = Federation(model, Xtr, ytr, Xte, yte, parts, sysm, seed=seed)
                t0 = time.time()
                log = run(fed, rounds=cfg["rounds"], theta=theta_fn, L=L, eval_every=10, seed=seed,
                          **METHODS[mname])
                res = summarize(log, Fs)
                res.update(dict(alpha=cfg["alpha"] if np.isfinite(cfg["alpha"]) else "inf", theta=theta,
                                sys=cfg["sys"], seed=seed, method=mname, H=sysm.H.tolist(), b=sysm.b.tolist(),
                                m=fed.m.tolist(), L=L))
                json.dump(res, open(path, "w"))
                print(f"{key}: floor_g2={res['floor_gnorm2']:.3e} "
                      f"gap={res.get('floor_gap', float('nan')):.4f} acc={res['final_acc']:.4f} "
                      f"valpred={np.mean(res['val_pred'][-5:]):.2f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
