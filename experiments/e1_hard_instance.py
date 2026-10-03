"""E1: validation of the theory on the Gaussian random-effects hard family.
F_i(x) = 0.5||x - theta_i||^2, theta_i = mu + zeta_i, zeta = (I - 1 p^T) eta, eta_i ~ N(0, tau2/d I).
Query noise N(0, v_i^2/d I) per sample; client i uses H_i steps with minibatch b_i per round.

E1a  attainability: plain normalized local SGD + Polyak averaging with RE*(R) weights vs the
     Bayes lower bound (any algorithm), and the standard choices p / work / IV.
E1b  SCAFFOLD neutralization (H_i = 1, heterogeneous b_i): risk independent of weights.
E1c  value of weights: predicted (closed form) vs measured, over a (tau2, spread) grid.
E1d  post-hoc (same-round) HEW rule in high dimension: weights increase with noise.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights.theory import re_weights, re_value, simplex_qp

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def make_family(rng, S, n, d, p, tau2, D2):
    eta = rng.standard_normal((S, n, d)) * np.sqrt(tau2 / d)
    zeta = eta - np.einsum("j,sjd->sd", p, eta)[:, None, :]
    mu = rng.standard_normal((S, d)) * np.sqrt(D2 / d)
    return mu[:, None, :] + zeta, mu            # theta (S,n,d), mu (S,d)


def alphas(theta_step, H):
    return 1 - (1 - theta_step / H) ** H


def run_plain(rng, theta, mu, v2, H, b, w, step, R, checkpoints, correct=True, burn=0):
    """Normalized local SGD on the hard family. Returns E||xbar_R - mu||^2 at checkpoints.
    If correct, server uses w_i ∝ wt_i/alpha_i so that the fixed point weights equal wt."""
    S, n, d = theta.shape
    al = alphas(step, H)
    if correct:
        ww = w / al
        ww = ww / ww.sum()
    else:
        ww = w
    x = np.zeros((S, d))
    xsum = np.zeros((S, d))
    out = {}
    for t in range(1, R + 1):
        upd = np.zeros((S, d))
        for i in range(n):
            if ww[i] == 0:
                continue
            eta = step / H[i]
            y = x.copy()
            for _ in range(H[i]):
                xi = rng.standard_normal((S, d)) * np.sqrt(v2[i] / (d * b[i]))
                y -= eta * (y - theta[:, i, :] + xi)
            upd += ww[i] * (y - x)
        x = x + upd
        if t > burn:
            xsum += x
        if t in checkpoints:
            xbar = xsum / (t - burn)
            out[t] = float(np.mean(np.sum((xbar - mu) ** 2, 1)))
    return out


def run_scaffold(rng, theta, mu, v2, b, w, p, step, R, checkpoints):
    """SCAFFOLD with H_i = 1, option-II control variates, server cv c = sum_j p_j c_j."""
    S, n, d = theta.shape
    x = np.zeros((S, d)); xsum = np.zeros((S, d))
    c = np.zeros((S, n, d)); cbar = np.zeros((S, d))
    out = {}
    for t in range(1, R + 1):
        g = x[:, None, :] - theta + rng.standard_normal((S, n, d)) * np.sqrt(v2 / (d * b))[None, :, None]
        direc = g - c + cbar[:, None, :]
        x = x - step * np.einsum("i,sid->sd", w, direc)
        c = g
        cbar = np.einsum("i,sid->sd", p, c)
        xsum += x
        if t in checkpoints:
            out[t] = float(np.mean(np.sum((xsum / t - mu) ** 2, 1)))
    return out


def lower_bound(p, tau2, v2, N, D2):
    lb0 = re_value(p, tau2, v2 / N)
    return lb0 / (1 + lb0 / D2)


def e1a(seed=0):
    rng = np.random.default_rng(seed)
    n, d, S, D2, step = 20, 50, 120, 0.1, 0.3
    H = rng.choice([1, 2, 4, 8, 16], n); b = np.ones(n, int)
    v2 = np.exp(rng.uniform(np.log(0.5), np.log(50), n))       # 100x spread in noise
    p = np.ones(n) / n
    Rs = [20, 50, 100, 200, 500, 1000, 2000]
    res = {}
    for tau2 in [0.0, 1e-3, 1e-2]:
        theta, mu = make_family(rng, S, n, d, p, tau2, D2)
        rec = {"R": Rs, "LB": [lower_bound(p, tau2, v2, R * H * b, D2) for R in Rs]}
        # fixed-weight rules: one run to Rmax
        for name, w in [("p", p), ("work", H * b / (H * b).sum()),
                        ("iv", (H * b / v2) / (H * b / v2).sum())]:
            o = run_plain(rng, theta, mu, v2, H, b, w, step, max(Rs), set(Rs))
            rec[name] = [o[R] for R in Rs]
        # budget-optimal RE weights, one run per R
        rec["re_opt"] = []
        for R in Rs:
            w = re_weights(p, tau2, v2 / (R * H * b)) if tau2 > 0 else (H * b / v2) / (H * b / v2).sum()
            o = run_plain(rng, theta, mu, v2, H, b, w, step, R, {R})
            rec["re_opt"].append(o[R])
        res[str(tau2)] = rec
        print("tau2", tau2, {k: np.round(np.array(v) / np.array(rec["LB"]), 3).tolist()
                             for k, v in rec.items() if k not in ("R", "LB")}, flush=True)
    res["meta"] = dict(n=n, d=d, S=S, D2=D2, step=step, H=H.tolist(), v2=v2.tolist())
    json.dump(res, open(os.path.join(OUT, "e1a.json"), "w"))


def e1b(seed=1):
    rng = np.random.default_rng(seed)
    n, d, S, D2, step = 20, 100, 300, 1.0, 0.2
    b = rng.choice([1, 2, 4, 8, 16, 32], n); H = np.ones(n, int)
    v2 = np.exp(rng.uniform(np.log(0.5), np.log(50), n))
    p = np.ones(n) / n
    Rs = [20, 50, 100, 200, 500, 1000, 2000]
    iv = (b / v2) / (b / v2).sum()
    res = {}
    for tau2 in [0.0, 1e-2]:
        theta, mu = make_family(rng, S, n, d, p, tau2, D2)
        rec = {"R": Rs, "LB": [lower_bound(p, tau2, v2, R * b, D2) for R in Rs]}
        for name, w in [("p", p), ("iv", iv), ("work", b / b.sum())]:
            o = run_scaffold(rng, theta, mu, v2, b, w, p, step, max(Rs), set(Rs))
            rec["scaffold_" + name] = [o[R] for R in Rs]
            o = run_plain(rng, theta, mu, v2, H, b, w, step, max(Rs), set(Rs))
            rec["plain_" + name] = [o[R] for R in Rs]
        rec["plain_re_opt"] = []
        for R in Rs:
            w = re_weights(p, tau2, v2 / (R * b)) if tau2 > 0 else iv
            rec["plain_re_opt"].append(run_plain(rng, theta, mu, v2, H, b, w, step, R, {R})[R])
        res[str(tau2)] = rec
        print("tau2", tau2, {k: np.round(np.array(v) / np.array(rec["LB"]), 3).tolist()
                             for k, v in rec.items() if k not in ("R", "LB")}, flush=True)
    res["meta"] = dict(n=n, d=d, S=S, step=step, b=b.tolist(), v2=v2.tolist())
    json.dump(res, open(os.path.join(OUT, "e1b.json"), "w"))


def e1c(seed=2):
    """Measured value of weights vs closed form, grid over tau2 and noise spread."""
    rng = np.random.default_rng(seed)
    n, d, S, D2, step, R = 20, 50, 100, 0.1, 0.3, 300
    p = np.ones(n) / n
    H = rng.choice([1, 2, 4, 8, 16], n); b = np.ones(n, int)
    rows = []
    for spread in [3, 10, 30, 100]:
        u = rng.uniform(0, 1, n)
        v2 = np.exp(np.log(0.5) + u * np.log(spread))
        sig2 = v2 / ((R // 2) * H * b)          # tail averaging over the last R/2 rounds
        for tau2 in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
            theta, mu = make_family(rng, S, n, d, p, tau2, D2)
            w_opt = re_weights(p, tau2, sig2) if tau2 > 0 else (1 / sig2) / (1 / sig2).sum()
            r_p = run_plain(rng, theta, mu, v2, H, b, p, step, R, {R}, burn=R // 2)[R]
            r_o = run_plain(rng, theta, mu, v2, H, b, w_opt, step, R, {R}, burn=R // 2)[R]
            pred = (np.sum(p ** 2 * sig2)) / re_value(p, tau2, sig2)
            rows.append(dict(spread=spread, tau2=tau2, measured=r_p / r_o, predicted=pred))
            print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(OUT, "e1c.json"), "w"))


def e1d(seed=3):
    """Same-round post-hoc HEW rule (anchor = mean pseudo-gradient of the round) in high dim."""
    rng = np.random.default_rng(seed)
    n, d, S, step, R, Lam = 20, 2000, 4, 0.2, 60, 1.5
    p = np.ones(n) / n
    H = np.full(n, 4); b = np.ones(n, int)
    v2 = np.exp(np.linspace(np.log(0.5), np.log(50), n))
    theta, mu = make_family(rng, S, n, d, p, 0.0, 1.0)
    W = np.zeros(n); cnt = 0
    x = np.zeros((S, d))
    for t in range(R):
        Ds = np.zeros((S, n, d))
        for i in range(n):
            y = x.copy(); eta = step / H[i]
            for _ in range(H[i]):
                y -= eta * (y - theta[:, i, :] + rng.standard_normal((S, d)) * np.sqrt(v2[i] / d))
            Ds[:, i] = y - x
        for s in range(S):
            Da = Ds[s]
            gbar = -(Da / step).mean(0)
            w = simplex_qp(0.5 * Lam * Da @ Da.T, Da @ gbar)
            x[s] = x[s] + w @ Da
            if t >= 10:
                W += w; cnt += 1
    W /= cnt
    s2 = v2 / H
    kappa = 1.0 / (n * Lam * step)
    json.dump(dict(v2=v2.tolist(), s2=s2.tolist(), mean_weight=W.tolist(), kappa=kappa,
                   iv=((1 / s2) / (1 / s2).sum()).tolist()), open(os.path.join(OUT, "e1d.json"), "w"))
    print("corr(weight, s2) =", np.corrcoef(W, s2)[0, 1], "weights:", np.round(W, 3), flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c", "d"]
    for w in which:
        t = time.time(); globals()["e1" + w](); print("done e1" + w, time.time() - t, flush=True)
