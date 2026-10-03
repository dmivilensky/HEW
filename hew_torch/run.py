"""Federated simulation with debiased HEW aggregation (PyTorch; CPU, CUDA or Apple Metal/MPS).

Datasets are downloaded automatically on first use (torchvision) into --data_root.

Examples
--------
# Apple silicon (Metal), Fashion-MNIST CNN, 20 clients, Dirichlet(0.3), unequal horizons/batches
python -m hew_torch.run --dataset fmnist --model cnn --alpha 0.3 --rounds 300 --lr 0.1 --method hew

# CIFAR-10 / ResNet-18 (GroupNorm), 100 clients, 20% participation
python -m hew_torch.run --dataset cifar10 --model resnet18 --n 100 --participation 0.2 --alpha 0.3 \
    --rounds 1000 --lr 0.5 --wd 5e-4 --method hew --out results/c10_hew.json

--lr is the total local step theta/L: client i uses lr/H_i per step, so every client moves a total of lr.
--device auto picks mps (Apple GPU) > cuda > cpu.

The JSON log contains, at every evaluation: train loss (fixed subset), test loss/accuracy, squared gradient
norm, server step multiplier, the variance weight theta, the weights w and their total-variation distance
to p, the heterogeneity level tau^2, the shrinkage eps, and the per-client noise estimates s_i^2; at the end,
p, H, b and tail averages (last quarter of the evaluations) of all scalar metrics.
"""
import argparse, json, time, os
import numpy as np
import torch
import torch.nn.functional as Fn
from .server import HEWServer, ServerConfig
from .client import local_train, flat_params, set_flat
from .data import load, labels_of, dirichlet_partition, Systems, ClientSampler
from .models import make_model
from .sketch import make_sketch

# "raw": unnormalized local steps (FedAvg); "control": SCAFFOLD control variates;
# "theta": variance weight of the round risk: a number (fixed), "lmax" (lr * lambda_max of the Hessian, re-estimated
# every 25 rounds; theta = lr * L as in the theory) or absent (online secant estimate)
PRESETS = {
    "p":            dict(weight="p"),
    "fedavg":       dict(weight="p", raw=True),
    "scaffold":     dict(weight="p", control=True),
    "iv":           dict(weight="iv"),
    "hew":          dict(weight="hew", het="cross", rule="re", timing="delayed"),
    "hew_t4":       dict(weight="hew", het="cross", rule="re", timing="delayed", theta=4.0),
    "hew_lm":       dict(weight="hew", het="cross", rule="re", timing="delayed", theta="lmax"),
    "hew_post":     dict(weight="hew", het="cross", rule="re", timing="post", theta="lmax"),
    "hew_sk":       dict(weight="hew", het="sketch", rule="re", timing="delayed", theta="lmax"),
    "hew_sk_gls":   dict(weight="hew", het="sketch", rule="gls", timing="delayed", theta="lmax"),
    "hew_sk_path":  dict(weight="hew", het="sketch", rule="path", timing="delayed", theta="lmax"),
    "p_adapt":      dict(weight="p", step="adaptive"),
    "hew_adapt":    dict(weight="hew", het="cross", rule="re", timing="delayed", step="adaptive", theta="lmax"),
    "hew_adsk":     dict(weight="hew", het="cross", rule="re", timing="delayed", step="adaptive_sk", theta="lmax"),
    "scaffold_hew": dict(weight="hew", het="cross", rule="re", timing="delayed", control=True, theta="lmax"),
}
SCALARS = ["train_loss", "test_loss", "test_acc", "gnorm2", "gamma", "theta", "tv_wp", "tau2", "eps"]


def pick_device(name):
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, x, ds, idx, bs=1000):
    set_flat(model, x); model.eval(); tot, correct, loss = 0, 0, 0.0
    for s in range(0, len(idx), bs):
        xb, yb = ds.batch(idx[s:s + bs], train=False)
        out = model(xb)
        loss += float(Fn.cross_entropy(out, yb, reduction="sum")); correct += int((out.argmax(1) == yb).sum()); tot += len(yb)
    model.train(); return loss / tot, correct / tot


def grad_norm2(model, x, ds, idx, wd, bs=1000):
    set_flat(model, x); model.zero_grad(); n = len(idx)
    for s in range(0, n, bs):
        xb, yb = ds.batch(idx[s:s + bs], train=False)
        (Fn.cross_entropy(model(xb), yb, reduction="sum") / n).backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()]) + wd * x
    model.zero_grad(); return float(g @ g)


def _lmax_exact(model, x, xb, yb, iters, seed):
    set_flat(model, x); params = list(model.parameters())
    loss = Fn.cross_entropy(model(xb), yb)
    g = torch.cat([t.reshape(-1) for t in torch.autograd.grad(loss, params, create_graph=True)])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(g.numel(), generator=gen).to(g.device); v = v / v.norm(); lam = 0.0
    for _ in range(iters):
        hv = torch.cat([t.reshape(-1) for t in torch.autograd.grad(g @ v, params, retain_graph=True)])
        lam = float(v @ hv); nv = float(hv.norm())
        if not np.isfinite(nv) or nv == 0:
            break
        v = hv / nv
    model.zero_grad(set_to_none=True)
    return abs(lam)


def hessian_lmax(model, x, ds, idx, iters=30, seed=0):
    """Largest Hessian eigenvalue (in absolute value) of the training loss at x on the fixed subset idx, by power
    iteration with exact Hessian-vector products (double backward). Some backends lack double backward for some
    layers (e.g. max-pooling on MPS); then the computation is repeated on a CPU copy of the model, which is exact
    and takes a few seconds for 1024 samples. Used to set theta = lr * L as in the theory (L = smoothness)."""
    xb, yb = ds.batch(idx, train=False)
    try:
        return _lmax_exact(model, x, xb, yb, iters, seed)
    except Exception as e:
        import copy
        if not getattr(hessian_lmax, "_warned", False):
            print(f"lambda_max: double backward unavailable on {x.device} ({type(e).__name__}); using a CPU copy", flush=True)
            hessian_lmax._warned = True
        mc = copy.deepcopy(model).to("cpu")
        lam = _lmax_exact(mc, x.detach().cpu(), xb.cpu(), yb.cpu(), iters, seed)
        set_flat(model, x)
        return lam


def _f(v):
    v = float(v); return v if np.isfinite(v) else None


def tail_summary(log, frac=0.25):
    k = max(1, int(round(frac * len(log["round"]))))
    out = {}
    for key in SCALARS:
        vals = [v for v in log[key][-k:] if v is not None]
        out[key] = float(np.mean(vals)) if vals else None
    out["rounds_from"] = log["round"][-k]
    return out


def save(log, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f)
    os.replace(tmp, path)                 # never leaves a half-written file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fmnist"); ap.add_argument("--model", default="cnn")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--alpha", type=float, default=float("inf"))
    ap.add_argument("--participation", type=float, default=1.0); ap.add_argument("--stragglers", action="store_true")
    ap.add_argument("--H_set", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--b_set", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--cv", default="first", choices=["first", "avg"],
                    help="SCAFFOLD control variate: gradient at the server point (first) or path average (avg)")
    ap.add_argument("--theta", type=float, default=None,
                    help="variance weight lr*L of the round risk; default: preset value, else estimated online")
    ap.add_argument("--method", default="hew", choices=list(PRESETS))
    ap.add_argument("--sketch", default="count", choices=["count", "gaussian"]); ap.add_argument("--sketch_k", type=int, default=256)
    ap.add_argument("--ema", type=float, default=0.8)
    ap.add_argument("--eval_every", type=int, default=5); ap.add_argument("--eval_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
    ap.add_argument("--data_root", default="./data"); ap.add_argument("--out", default=None)
    ap.add_argument("--skip_existing", action="store_true", help="do nothing if --out already holds a finished run")
    a = ap.parse_args(argv)
    if a.skip_existing and a.out and os.path.exists(a.out):
        try:
            if json.load(open(a.out)).get("finished"):
                print(f"skip (exists): {a.out}", flush=True); return None
        except Exception:
            pass
    dev = pick_device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f"device={dev} method={a.method} alpha={a.alpha} seed={a.seed}", flush=True)

    train, test, C = load(a.dataset, a.data_root, device=dev)
    y = labels_of(train); parts = dirichlet_partition(y, a.n, a.alpha, seed=1000 + a.seed)
    p = np.array([len(q) for q in parts], float); p /= p.sum()
    sysm = Systems(a.n, tuple(a.H_set), tuple(a.b_set), a.participation, a.stragglers, seed=2000 + a.seed)
    model = make_model(a.model, tuple(train.X.shape[1:]), C).to(dev)
    x = flat_params(model).to(dev); dim = x.numel()
    preset = dict(PRESETS[a.method]); raw = preset.pop("raw", False); control = preset.pop("control", False)
    theta = a.theta if a.theta is not None else preset.pop("theta", None)
    preset.pop("theta", None)
    theta_lmax = theta == "lmax"                 # theta = lr * lambda_max(Hessian), re-estimated every 25 rounds
    ev_lm = torch.as_tensor(np.random.default_rng(1).choice(len(train), min(1024, len(train)), replace=False), device=dev)
    cfg = ServerConfig(ema=a.ema, **preset)
    server = HEWServer(a.n, p, dim, a.lr, cfg, device=dev)
    need_sketch = cfg.het == "sketch" or cfg.step == "adaptive_sk"
    sketch = make_sketch(a.sketch, dim, a.sketch_k, seed=a.seed + 777, device=dev) if need_sketch else None
    server.sketch = sketch
    samplers = [ClientSampler(train, parts[i], int(sysm.b[i]), seed=a.seed * 1000 + i) for i in range(a.n)]
    # one contiguous tensor per client: row views of an (n x dim) MPS tensor are not safe to use in arithmetic
    c = [torch.zeros(dim, device=dev) for _ in range(a.n)] if control else None
    cbar = torch.zeros(dim, device=dev) if control else None
    ev_tr = torch.as_tensor(np.random.default_rng(0).choice(len(train), min(a.eval_n, len(train)), replace=False), device=dev)
    ev_te = torch.arange(len(test), device=dev)
    loss_fn = lambda o, yy: Fn.cross_entropy(o, yy)

    log = dict(args=vars(a), method=a.method, device=dev, dim=dim, p=p.tolist(), H=[int(h) for h in sysm.H],
               b=[int(v) for v in sysm.b], sizes=[len(q) for q in parts], round=[], time=[], w=[], s2=[],
               finished=False, diverged=None, **{k: [] for k in SCALARS})
    t0 = time.time()
    for t in range(a.rounds + 1):
        if t % a.eval_every == 0 or t == a.rounds:
            trl, _ = evaluate(model, x, train, ev_tr); tel, tea = evaluate(model, x, test, ev_te)
            g2 = grad_norm2(model, x, train, ev_tr, a.wd)
            wl = getattr(server, "last_weights", None); dg = server.diag
            if wl:
                pa = {i: p[i] / sum(p[j] for j in wl) for i in wl}
                tv = 0.5 * sum(abs(wl[i] - pa[i]) for i in wl)
                wfull = [wl.get(i) for i in range(a.n)]
            else:
                tv, wfull = float("nan"), None
            s2 = [_f(server.v2[i] / (sysm.H[i] * sysm.b[i])) for i in range(a.n)]
            row = dict(train_loss=trl, test_loss=tel, test_acc=tea, gnorm2=g2, gamma=server.gamma,
                       theta=server.theta_used, tv_wp=tv, tau2=dg.get("tau2", float("nan")), eps=dg.get("eps", float("nan")))
            log["round"].append(t); log["time"].append(time.time() - t0); log["w"].append(wfull); log["s2"].append(s2)
            for k in SCALARS:
                log[k].append(_f(row[k]))
            print(f"[{a.method}] t={t} train_loss={trl:.4f} test_acc={tea:.4f} |grad|^2={g2:.3e} "
                  f"gamma={server.gamma:.3f} theta={server.theta_used:.3g} TV(w,p)={tv:.3f} "
                  f"tau2={row['tau2']:.3g} eps={row['eps']:.3g} ({time.time() - t0:.0f}s)", flush=True)
        if t == a.rounds:
            break
        if theta_lmax and t % 25 == 0:
            try:
                lam = hessian_lmax(model, x, train, ev_lm, seed=t)
            except Exception as e:                   # e.g. double backward unsupported on a backend
                print(f"lambda_max estimate failed ({e}); falling back to online theta", flush=True); lam = None
            theta = None if lam is None else float(np.clip(a.lr * lam, 1e-3, 64.0))
            theta_lmax = lam is not None
            log.setdefault("lmax", []).append([t, lam])
        act, H, b = sysm.draw()
        server.begin_round(list(act), [int(H[i]) for i in act], [int(b[i]) for i in act], theta=theta)
        msgs = []
        try:
            for i in act:
                base = a.lr if not raw else a.lr * H[i] / np.mean(sysm.H)      # FedAvg: equal per-step lr for all
                msg, cnew = local_train(int(i), model, x, samplers[i], int(H[i]), int(b[i]), base, loss_fn,
                                        sketch=sketch, c_i=c[int(i)] if control else None, c_bar=cbar,
                                        weight_decay=a.wd, clip=a.clip, cv=a.cv)
                msgs.append(msg)
                if control:
                    c[int(i)] = cnew.detach().clone()
            x = x + server.end_round(msgs)
        except FloatingPointError as e:
            print(f"diverged at round {t}: {e}", flush=True); log["diverged"] = t; break
        if control:
            cbar = sum(float(p[j]) * c[j] for j in range(a.n))
        if not torch.isfinite(x).all():
            print("diverged (non-finite parameters); try a smaller --lr", flush=True); log["diverged"] = t; break
        if a.out and t % 50 == 49:
            save(log, a.out)              # checkpoint of the log (a crash keeps the curves so far)
    log["finished"] = True
    log["tail"] = tail_summary(log)
    log["wall_time"] = time.time() - t0
    save(log, a.out)
    return log


if __name__ == "__main__":
    main()
