"""Federated local-SGD simulator with horizon normalization, optional control variates,
and pluggable aggregation-weight rules (predictable or post-hoc)."""
from __future__ import annotations
import time
import numpy as np
from .theory import re_weights, gls_oracle_weights, simplex_qp, risk

PREDICTABLE = {"p", "uniform", "work", "iv", "re", "re_oracle", "re_lmax_oracle", "gls_oracle",
               "re_fixed", "gls", "relmax"}
POSTHOC = {"posthoc_hew", "fedadp", "hew", "hew_gls", "hew_fresh", "hew_nodebias", "hewx", "hewx_gls", "hewx_reg", "phew", "phew_raw", "phew_psd", "hewx_lag"}


class Federation:
    def __init__(self, model, Xtr, ytr, Xte, yte, parts, systems, seed=0):
        self.model, self.Xtr, self.ytr, self.Xte, self.yte = model, Xtr, ytr, Xte, yte
        self.parts, self.sys = parts, systems
        self.n = len(parts)
        self.m = np.array([len(p) for p in parts])
        self.p = self.m / self.m.sum()          # FedAvg objective weights
        self.Xc = [Xtr[p] for p in parts]
        self.yc = [ytr[p] for p in parts]
        self.rng = np.random.default_rng(seed + 12345)

    # ---------- oracle quantities at x (F_t-measurable) ----------
    def oracle(self, w):
        G, V = [], []
        for Xi, yi in zip(self.Xc, self.yc):
            g, v = self.model.grad_and_persample_var(w, Xi, yi)
            G.append(g)
            V.append(v)
        G, V = np.array(G), np.array(V)
        gF = self.p @ G
        Zeta = G - gF
        Z = Zeta @ Zeta.T
        return gF, Z, V

    def full_metrics(self, w):
        gF, _ = self.model.grad_and_persample_var(w, self.Xtr, self.ytr)
        return dict(loss=self.model.loss(w, self.Xtr, self.ytr), gnorm2=float(gF @ gF),
                    test_acc=self.model.accuracy(w, self.Xte, self.yte))

    # ---------- local training ----------
    def local(self, i, w, H, b, eta, c_i=None, cbar=None):
        Xi, yi = self.Xc[i], self.yc[i]
        y = w.copy()
        vsum = 0.0
        gsum = np.zeros_like(w)
        half = b // 2
        for _ in range(H):
            idx = self.rng.integers(0, len(yi), size=2 * half)
            gA = self.model.grad(y, Xi[idx[:half]], yi[idx[:half]])
            gB = self.model.grad(y, Xi[idx[half:]], yi[idx[half:]])
            g = 0.5 * (gA + gB)
            d = gA - gB
            vsum += (2 * half) * (d @ d) / 4.0       # unbiased per-sample variance estimate
            gsum += g
            if c_i is not None:
                g = g - c_i + cbar
            y -= eta * g
        return y - w, vsum / H, gsum / H


def run(fed: Federation, method: str, rounds: int, theta, L, corrected=False, raw=False,
        eval_every=10, ema=0.8, c_theta=1.0, Lam_ratio=1.5, tau_mode="mean", seed=0,
        re_fixed=None, log_weights=True, verbose=False):
    """theta: float or callable t->theta_t (normalized step, lr = theta/L).
    method: weight rule name.  corrected: SCAFFOLD-type control variates.
    raw: FedAvg with un-normalized local steps (eta = lr/mean(H))."""
    rng = np.random.default_rng(seed)
    n, p, model = fed.n, fed.p, fed.model
    w = model.init(rng)
    sched = theta if callable(theta) else (lambda t: theta)
    v2_hat = np.full(n, np.nan)                 # per-sample variance estimates (EMA)
    tau2_hat = None
    Zhat = None                                 # plug-in heterogeneity Gram (EMA), used with lag
    Eprev = None; Zx = None; dev2 = None; w_next = None                     # cross-round Gram estimator (debiasing-free)
    c = np.zeros((n, model.dim)) if corrected else None
    cbar = np.zeros(model.dim) if corrected else None
    ang = np.zeros(n); cnt = np.zeros(n)        # FedAdp state
    log = dict(round=[], loss=[], gnorm2=[], test_acc=[], weights=[], tau2_hat=[], lmaxZ=[],
               trZ=[], s2_oracle=[], s2_hat=[], val_pred=[], risk_w=[], risk_opt=[], time=[])
    wts = np.ones(n) / n
    t0 = time.time()
    for t in range(rounds + 1):
        th = sched(t)
        lr = th / L
        need_oracle = method in ("re_oracle", "re_lmax_oracle", "gls_oracle") or (t % eval_every == 0)
        if t % eval_every == 0 or t == rounds:
            met = fed.full_metrics(w)
            for k in ("loss", "gnorm2", "test_acc"):
                log[k].append(met[k])
            log["round"].append(t)
            log["time"].append(time.time() - t0)
            if verbose:
                print(t, met, flush=True)
        if t == rounds:
            break
        H, b, active = fed.sys.draw()
        if need_oracle:
            gF, Z, V = fed.oracle(w)
            s2_or = V / (H * b)
        # ------------ predictable weights ------------
        if method in PREDICTABLE:
            wts = predictable_weights(method, p, active, H, b, th, v2_hat, tau2_hat,
                                      Z if need_oracle else None,
                                      s2_or if need_oracle else None, c_theta, tau_mode, re_fixed, Zhat)
        v2_lag = v2_hat.copy()                      # noise estimates up to round t-1 (for predictable candidates)
        # ------------ local rounds ------------
        Deltas = np.zeros((n, model.dim))
        for i in active:
            eta = (lr / np.mean(fed.sys.H)) if raw else (lr / H[i])
            if corrected:
                D, v2, gavg = fed.local(i, w, H[i], b[i], eta, c[i], cbar)
            else:
                D, v2, gavg = fed.local(i, w, H[i], b[i], eta)
            Deltas[i] = D
            v2_hat[i] = v2 if np.isnan(v2_hat[i]) else ema * v2_hat[i] + (1 - ema) * v2
            if corrected:
                c[i] = gavg
        # ------------ round statistics (used post-locally by HEW, with lag by relmax/gls) ------------
        G = -Deltas[active] / lr
        pa = p[active] / p[active].sum()
        gbar_p = pa @ G
        T = np.sum((G - gbar_p) ** 2)
        s2 = v2_hat[active] / (H[active] * b[active])
        est = T - np.sum(s2 * (1 - 2 * pa)) - len(active) * np.sum(pa ** 2 * s2)
        est = max(est, 0.0) / max(len(active) - 1, 1)
        tau2_hat = est if tau2_hat is None else ema * tau2_hat + (1 - ema) * est
        # plug-in Gram of heterogeneity: centered pseudo-gradient Gram minus noise covariance
        E = G - gbar_p
        Zt = E @ E.T
        Zt -= np.diag(s2) - np.outer(s2 * pa, np.ones(len(pa))) - np.outer(np.ones(len(pa)), s2 * pa) \
            + np.sum(pa ** 2 * s2)
        if Zhat is None:
            Zhat = np.zeros((n, n)); seen = np.zeros((n, n), bool)
        ix = np.ix_(active, active)
        Zhat[ix] = np.where(seen[ix], ema * Zhat[ix] + (1 - ema) * Zt, Zt)
        seen[ix] = True
        # ------------ HEW-x: cross-round heterogeneity Gram (no noise subtraction needed) ------------
        if method in ("hewx", "hewx_gls", "hewx_reg", "phew", "phew_raw", "phew_psd", "hewx_lag"):
            Efull = np.zeros((n, model.dim)); Efull[active] = E
            if Eprev is None:
                wts = np.zeros(n); wts[active] = pa
            else:
                C = Efull @ Eprev.T
                Zt_x = 0.5 * (C + C.T)
                k = len(active)
                P = np.eye(k) - np.ones((k, k)) / k
                if Zx is not None:                   # fluctuation level of the EMA estimate (for eps_t)
                    dv = np.linalg.norm(P @ (Zt_x - Zx)[ix] @ P, 2) ** 2
                    dev2 = dv if dev2 is None else ema * dev2 + (1 - ema) * dv
                Zx = Zt_x if Zx is None else ema * Zx + (1 - ema) * Zt_x
                Zc = _psd(Zx[ix])
                x = th * s2
                if method in ("phew", "phew_raw", "phew_psd"):
                    # path-restricted HEW: candidates = RE path from lagged noise, select by cross-round risk
                    sl = v2_lag[active] / (H[active] * b[active])
                    if np.any(np.isnan(sl)):
                        sl = s2
                    xl = th * sl
                    ks = [0.0] + list(np.median(xl) * 2.0 ** np.arange(-8, 9))
                    cands = [re_weights(pa, k2, xl) for k2 in ks] + [pa.copy()]
                    Zr = Zc if method == "phew_psd" else Zx[ix]
                    best, wa = np.inf, pa
                    for cw in cands:
                        dvec = cw - pa
                        bias = dvec @ Zr @ dvec
                        if method == "phew":
                            bias = max(bias, 0.0)
                        crit = bias + np.sum(cw ** 2 * x)
                        if crit < best:
                            best, wa = crit, cw
                elif method == "hewx_gls":
                    wa = simplex_qp(Zc + np.diag(x), -2 * Zc @ pa)
                else:
                    tau2 = max(np.linalg.eigvalsh(P @ Zc @ P)[-1], 0.0)
                    if method in ("hewx_reg", "hewx_lag") and dev2 is not None:
                        tau2 += np.sqrt(0.5 * (1 - ema) * dev2)     # eps_t: shrink toward p by the error level
                    wa = re_weights(pa, tau2, x)
                wts = np.zeros(n); wts[active] = wa
            Eprev = Efull
            if method == "hewx_lag":
                w_apply = w_next if w_next is not None else np.zeros(n)
                if w_next is None:
                    w_apply[active] = pa
                w_next = wts.copy()
                wts = w_apply
        # ------------ HEW: debiased post-local certificate ------------
        if method in ("hew", "hew_gls", "hew_fresh", "hew_nodebias"):
            if method == "hew_nodebias":           # original quadratic model with Lambda = L/theta, p-anchor
                M = E @ E.T
                qa = -2 * M @ pa
                wa = simplex_qp(M, qa)
            else:
                Zc = _psd(Zt if method == "hew_fresh" else Zhat[ix])
                x = th * s2
                if method == "hew_gls":
                    M = Zc + np.diag(x)
                    wa = simplex_qp(M, -2 * Zc @ pa)
                else:
                    k = len(active)
                    P = np.eye(k) - np.ones((k, k)) / k
                    tau2 = max(np.linalg.eigvalsh(P @ Zc @ P)[-1], 0.0)
                    wa = re_weights(pa, tau2, x)
            wts = np.zeros(n); wts[active] = wa
        # ------------ post-hoc weights ------------
        if method == "posthoc_hew":
            Da = Deltas[active]
            gbar = -(Da / lr).mean(0)              # same-round anchor (old HEW, hom. branch)
            Lam = Lam_ratio * L
            wa = simplex_qp(0.5 * Lam * Da @ Da.T, Da @ gbar)
            wts = np.zeros(n); wts[active] = wa
        elif method == "fedadp":
            G = -Deltas[active] / lr
            gagg = p[active] @ G / p[active].sum()
            cos = G @ gagg / (np.linalg.norm(G, axis=1) * np.linalg.norm(gagg) + 1e-12)
            th_i = np.arccos(np.clip(cos, -1, 1))
            cnt[active] += 1
            ang[active] = ang[active] + (th_i - ang[active]) / cnt[active]
            alpha = 5.0
            f = alpha * (1 - np.exp(-np.exp(-alpha * (ang[active] - 1))))
            wa = p[active] * np.exp(f); wa /= wa.sum()
            wts = np.zeros(n); wts[active] = wa
        # ------------ aggregate ------------
        w = w + wts @ Deltas
        if corrected:
            cbar = p @ c
        if t % eval_every == 0:
            if log_weights:
                log["weights"].append(wts.tolist())
            log["tau2_hat"].append(tau2_hat)
            ev = np.linalg.eigvalsh(Z)
            log["lmaxZ"].append(float(ev[-1]))
            log["trZ"].append(float(np.trace(Z)))
            log["s2_oracle"].append(s2_or.tolist())
            log["s2_hat"].append((v2_hat / (H * b)).tolist())
            x = th * s2_or
            wopt = gls_oracle_weights(p, Z, x)
            r_opt = risk(wopt, p, x, Z=Z)
            r_p = risk(p, p, x, Z=Z)
            log["risk_opt"].append(float(r_opt))
            log["risk_w"].append(float(risk(wts, p, x, Z=Z)))
            log["val_pred"].append(float(r_p / max(r_opt, 1e-30)))
    return log


def _psd(M):
    M = 0.5 * (M + M.T)
    ev, U = np.linalg.eigh(M)
    return (U * np.maximum(ev, 0)) @ U.T


def predictable_weights(method, p, active, H, b, th, v2_hat, tau2_hat, Z, s2_or, c_theta,
                        tau_mode, re_fixed, Zhat=None):
    n = len(p)
    wts = np.zeros(n)
    if method == "p":
        wts[active] = p[active] / p[active].sum()
    elif method == "uniform":
        wts[active] = 1.0 / len(active)
    elif method == "work":
        a = (H * b)[active].astype(float); wts[active] = a / a.sum()
    elif method in ("gls", "relmax"):
        if Zhat is None or np.any(np.isnan(v2_hat[active])):
            wts[active] = p[active] / p[active].sum()
            return wts
        s2 = v2_hat / (H * b)
        Zp = _psd(Zhat)
        if method == "gls":
            wts = gls_oracle_weights(p, Zp, c_theta * th * s2, active)
        else:
            P = np.eye(n) - np.ones((n, n)) / n
            tau2 = np.linalg.eigvalsh(P @ Zp @ P)[-1]
            wts = re_weights(p, tau2, c_theta * th * s2, active)
    elif method in ("iv", "re", "re_fixed"):
        if np.all(np.isnan(v2_hat[active])) or (method == "re" and tau2_hat is None):
            wts[active] = p[active] / p[active].sum()
            return wts
        v2 = v2_hat.copy()
        v2[np.isnan(v2)] = np.nanmean(v2)
        s2 = v2 / (H * b)
        if method == "iv":
            a = 1.0 / s2[active]; wts[active] = a / a.sum()
        else:
            tau2 = tau2_hat if method == "re" else re_fixed
            wts = re_weights(p, tau2, c_theta * th * s2, active)
    elif method in ("re_oracle", "re_lmax_oracle"):
        P = np.eye(n) - np.ones((n, n)) / n
        tau2 = np.linalg.eigvalsh(P @ Z @ P)[-1] if method == "re_lmax_oracle" else np.trace(Z) / (n - 1)
        wts = re_weights(p, tau2, c_theta * th * s2_or, active)
    elif method == "gls_oracle":
        wts = gls_oracle_weights(p, Z, c_theta * th * s2_or, active)
    return wts
