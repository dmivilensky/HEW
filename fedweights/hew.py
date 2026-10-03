"""Debiased HEW-Local SGD: clean reference implementation (numpy simulator).

One configurable server covering every variant used in the paper:

  local      : "normalized" (eta_i = theta/(L H_i))  |  "raw" (FedAvg, eta = theta/(L mean H))
  control    : SCAFFOLD-type control variates (bool)
  weight     : "p" | "iv" | "hew"
  het        : heterogeneity estimate used by "hew"
                 "cross"  - cross-round Gram of centered pseudo-gradients  (noise-free in mean)
                 "sketch" - same-round Gram of first-step *gradients* from the two minibatch halves,
                            compressed by a shared JL sketch of size k      (unbiased, no saturation)
  rule       : "re" (lambda_max random effects, closed form) | "gls" (QP) | "path" (select on RE path)
  timing     : "delayed" (weights for round t from statistics <= t-1; predictable)  |  "post"
  shrink     : add eps_t (calibrated from round-to-round variability of the Gram estimate)
  step       : "fixed" | "decay" (theta_t = theta/sqrt(1+t/t0)) | "adaptive" (cross-round SNR step)

Communication per client and round: the update, one scalar (split-batch noise estimate) and,
for het="sketch", 2k numbers.  Partial participation is supported through Systems.draw().
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from .theory import re_weights, simplex_qp


@dataclass
class HEWConfig:
    rounds: int = 300
    theta: float = 16.0
    local: str = "normalized"
    control: bool = False
    weight: str = "hew"
    het: str = "cross"
    rule: str = "re"
    timing: str = "delayed"
    shrink: bool = True
    step: str = "fixed"
    decay_t0: float = 20.0
    sketch_k: int = 128
    sketch_fresh: bool = True        # fresh JL sketch per round for the Gram (as in the proposition)
    ema: float = 0.8
    gamma_min: float = 0.02
    var_weight: object = None        # variance weight in the round risk: None -> theta; float; 'auto' -> min(1, lr*secant curvature)
    s2_floor: float = 1e-12
    eval_every: int = 10
    seed: int = 0
    log_weights: bool = True
    extra: dict = field(default_factory=dict)


def _psd(M):
    M = 0.5 * (M + M.T)
    ev, U = np.linalg.eigh(M)
    return (U * np.maximum(ev, 0.0)) @ U.T


def _lmax_centered(M):
    k = M.shape[0]
    P = np.eye(k) - np.ones((k, k)) / k
    return float(np.linalg.eigvalsh(P @ M @ P)[-1])


class _EMAGram:
    """EMA of an n x n Gram estimate updated on the active block; tracks round-to-round variability."""

    def __init__(self, n, beta):
        self.Z = np.zeros((n, n)); self.seen = np.zeros((n, n), bool)
        self.dev2 = None; self.beta = beta

    def update(self, act, Zt):
        ix = np.ix_(act, act); b = self.beta
        if self.seen[ix].all():
            k = len(act); P = np.eye(k) - np.ones((k, k)) / k
            dv = np.linalg.norm(P @ (Zt - self.Z[ix]) @ P, 2) ** 2
            self.dev2 = dv if self.dev2 is None else b * self.dev2 + (1 - b) * dv
        self.Z[ix] = np.where(self.seen[ix], b * self.Z[ix] + (1 - b) * Zt, Zt)
        self.seen[ix] = True

    def ready(self, act):
        return self.seen[np.ix_(act, act)].all()

    def eps(self):
        # Var(EMA) = (1-b)/(1+b) Var(single), E||Z_t - EMA_{t-1}||^2 ~ 2/(1+b) Var(single)
        return 0.0 if self.dev2 is None else float(np.sqrt(0.5 * (1 - self.beta) * self.dev2))


def local_round(fed, i, w, H, b, eta, c_i=None, cbar=None, want_first=False):
    """H local SGD steps with split minibatches. Returns update, per-sample variance estimate,
    average local gradient (for control variates) and optionally the two first-step half gradients."""
    Xi, yi = fed.Xc[i], fed.yc[i]
    y = w.copy(); vsum = 0.0; gsum = np.zeros_like(w); half = max(b // 2, 1); first = None
    for k in range(H):
        idx = fed.rng.integers(0, len(yi), size=2 * half)
        gA = fed.model.grad(y, Xi[idx[:half]], yi[idx[:half]])
        gB = fed.model.grad(y, Xi[idx[half:]], yi[idx[half:]])
        if k == 0 and want_first:
            first = (gA, gB)
        g = 0.5 * (gA + gB); d = gA - gB
        vsum += (2 * half) * (d @ d) / 4.0
        gsum += g
        if c_i is not None:
            g = g - c_i + cbar
        y -= eta * g
    return y - w, vsum / H, gsum / H, first


def weights_from_gram(cfg, pa, Zc, x, eps, x_path=None):
    """HEW weight rules on the active block. Zc: Gram estimate (may be indefinite), x = theta*s2."""
    if cfg.rule == "re":
        tau2 = max(_lmax_centered(_psd(Zc)), 0.0) + (eps if cfg.shrink else 0.0)
        return re_weights(pa, tau2, x)
    if cfg.rule == "gls":
        M = _psd(Zc) + (eps if cfg.shrink else 0.0) * np.eye(len(pa))
        return simplex_qp(M + np.diag(x), -2 * M @ pa)
    if cfg.rule == "path":
        xl = x if x_path is None else x_path
        grid = [0.0] + list(np.median(xl) * 2.0 ** np.arange(-8, 9))
        best, wbest = np.inf, pa
        for k2 in grid + [None]:
            cw = pa.copy() if k2 is None else re_weights(pa, k2, xl)
            d = cw - pa
            crit = max(d @ Zc @ d, 0.0) + (eps if cfg.shrink else 0.0) * (d @ d) + np.sum(cw ** 2 * x)
            if crit < best:
                best, wbest = crit, cw
        return wbest
    raise ValueError(cfg.rule)


def run_hew(fed, cfg: HEWConfig, L: float, verbose=False):
    rng = np.random.default_rng(cfg.seed)
    n, p, model = fed.n, fed.p, fed.model
    x = model.init(rng)
    v2 = np.full(n, np.nan)                       # EMA of per-sample variance estimates
    gram = _EMAGram(n, cfg.ema)
    g_last = np.full((n, model.dim), np.nan)      # last pseudo-gradient seen per client (cross-round)
    c = np.zeros((n, model.dim)) if cfg.control else None
    cbar = np.zeros(model.dim) if cfg.control else None
    S = None                                      # fixed JL sketch (step statistics; Gram if not fresh)
    need_first = cfg.het == "sketch" or cfg.step == "adaptive_sk"
    if need_first:
        S = np.random.default_rng(cfg.seed + 777).standard_normal((cfg.sketch_k, model.dim)) / np.sqrt(cfg.sketch_k)
    rng_sk = np.random.default_rng(cfg.seed + 999)
    # adaptive server step state
    st = dict(sig=None, mom=None, cnum=None, cden=None, gprev=None, xprev=None, eta_prev=None, V_prev=None)
    gamma = 1.0
    log = dict(round=[], loss=[], gnorm2=[], test_acc=[], weights=[], gamma=[], tau2=[], time=[])
    t0 = time.time()

    def theta_t(t):
        return cfg.theta / np.sqrt(1.0 + t / cfg.decay_t0) if cfg.step == "decay" else cfg.theta

    kst = dict(num=None, den=None, g=None, dx=None, eta=None, V=None)   # always-on debiased secant curvature

    def var_weight(th):
        if cfg.var_weight is None:
            return th
        if cfg.var_weight == "omega":
            # stationary variance weight omega = (1-mu)/(1+mu), mu = 1 - r the per-round contraction along the
            # steps, r = lr * (debiased secant curvature) clipped to [1e-3, 1]  (Prop. stationary risk)
            r = float(np.clip((th / L) * max(kst["num"] / kst["den"], 1e-12), 1e-3, 1.0)) if kst["den"] else min(th, 1.0)
            return r / (2.0 - r)
        if cfg.var_weight == "auto":
            if kst["den"]:
                return float(np.clip((th / L) * max(kst["num"] / kst["den"], 1e-12), 1e-3, 1.0))
            return min(th, 1.0)
        return float(cfg.var_weight)

    def current_weights(act, H, b, th_eff):
        pa = p[act] / p[act].sum()
        if cfg.weight == "p" or np.any(np.isnan(v2[act])):
            return pa, None
        s2 = np.maximum(v2[act] / (H[act] * b[act]), cfg.s2_floor)
        xx = th_eff * s2
        if cfg.weight == "iv":
            a = 1.0 / s2; return a / a.sum(), None
        if not gram.ready(act):
            return pa, None
        Zc = gram.Z[np.ix_(act, act)]
        return weights_from_gram(cfg, pa, Zc, xx, gram.eps()), _lmax_centered(_psd(Zc))

    for t in range(cfg.rounds + 1):
        th = theta_t(t)
        if t % cfg.eval_every == 0 or t == cfg.rounds:
            met = fed.full_metrics(x)
            for k in ("loss", "gnorm2", "test_acc"):
                log[k].append(met[k])
            log["round"].append(t); log["time"].append(time.time() - t0); log["gamma"].append(gamma)
            if verbose:
                print(t, met, gamma, flush=True)
        if t == cfg.rounds:
            break
        H, b, act = fed.sys.draw()
        # ---- adaptive server step (predictable: statistics of rounds <= t-1) ----
        if cfg.step in ("adaptive", "adaptive_sk") and st["sig"] is not None and st["cden"] and st["cden"] > 0:
            ell = max(st["cnum"] / st["cden"], 1e-12)
            eta_star = max(st["sig"], 0.0) / (ell * st["mom"])
            gamma = float(np.clip(eta_star / (th / L), cfg.gamma_min, 1.0))
        eta_srv = gamma * th / L                   # server step multiplies the normalized update
        th_eff = gamma * var_weight(th)            # variance multiplier in the round risk
        # ---- delayed (predictable) weights ----
        tau2 = None
        if cfg.timing == "delayed":
            wa, tau2 = current_weights(act, H, b, th_eff)
        # ---- local rounds ----
        D = np.zeros((len(act), model.dim)); firsts = []
        for j, i in enumerate(act):
            eta = (th / L / np.mean(fed.sys.H)) if cfg.local == "raw" else (th / L / H[i])
            Di, vv, gavg, first = local_round(fed, i, x, H[i], b[i], eta,
                                              c[i] if cfg.control else None, cbar,
                                              want_first=need_first)
            D[j] = Di; firsts.append(first)
            v2[i] = vv if np.isnan(v2[i]) else cfg.ema * v2[i] + (1 - cfg.ema) * vv
            if cfg.control:
                c[i] = gavg
        pa = p[act] / p[act].sum()
        G = -D / (th / L)                          # pseudo-gradients
        # ---- heterogeneity statistics of this round ----
        if cfg.weight == "hew":
            if cfg.het == "cross":
                prev = g_last[act]
                if not np.any(np.isnan(prev)):
                    E = G - pa @ G; Ep = prev - pa @ prev
                    C = E @ Ep.T; gram.update(act, 0.5 * (C + C.T))
            elif cfg.het == "sketch":
                St = (rng_sk.standard_normal((cfg.sketch_k, model.dim)) / np.sqrt(cfg.sketch_k)
                      if cfg.sketch_fresh else S)
                A = np.array([St @ f[0] for f in firsts]); B = np.array([St @ f[1] for f in firsts])
                EA = A - pa @ A; EB = B - pa @ B
                C = EA @ EB.T; gram.update(act, 0.5 * (C + C.T))
        g_last[act] = G
        # ---- post-local weights ----
        if cfg.timing == "post":
            wa, tau2 = current_weights(act, H, b, th_eff)
        if cfg.weight == "p" or wa is None:
            wa = pa
        wts = np.zeros(n); wts[act] = wa
        # ---- server update ----
        g_agg = wa @ G
        x_new = x - eta_srv * g_agg
        if cfg.control:
            cbar = p @ c
        # ---- always-on secant curvature along server steps (debiased) ----
        s2k = np.maximum(v2[act] / (H[act] * b[act]), cfg.s2_floor); Vk = float(np.sum(wa ** 2 * s2k)); bt = cfg.ema
        if kst["g"] is not None and len(kst["g"]) == len(g_agg):
            numk = float((g_agg - kst["g"]) @ kst["dx"]) - kst["eta"] * kst["V"]; denk = float(kst["dx"] @ kst["dx"])
            kst["num"] = numk if kst["num"] is None else bt * kst["num"] + (1 - bt) * numk
            kst["den"] = denk if kst["den"] is None else bt * kst["den"] + (1 - bt) * denk
        kst["g"], kst["dx"], kst["eta"], kst["V"] = g_agg.copy(), -eta_srv * g_agg, eta_srv, Vk
        # ---- adaptive-step statistics (used from round t+1 on) ----
        if cfg.step == "adaptive":
            s2a = np.maximum(v2[act] / (H[act] * b[act]), cfg.s2_floor)
            V = float(np.sum(wa ** 2 * s2a))
            bt = cfg.ema
            mom = float(g_agg @ g_agg)
            st["mom"] = mom if st["mom"] is None else bt * st["mom"] + (1 - bt) * mom
            if st["gprev"] is not None:
                sig = float(g_agg @ st["gprev"])
                st["sig"] = sig if st["sig"] is None else bt * st["sig"] + (1 - bt) * sig
                dx = x - st["xprev"]
                num = float((g_agg - st["gprev"]) @ dx) - st["eta_prev"] * st["V_prev"]
                den = float(dx @ dx)
                st["cnum"] = num if st["cnum"] is None else bt * st["cnum"] + (1 - bt) * num
                st["cden"] = den if st["cden"] is None else bt * st["cden"] + (1 - bt) * den
            st["gprev"], st["xprev"], st["eta_prev"], st["V_prev"] = g_agg.copy(), x.copy(), eta_srv, V
        if cfg.step == "adaptive_sk":
            # all statistics from first-step gradient sketches at x_t (no saturation, no cross-round lag)
            A = np.array([S @ f[0] for f in firsts]); B = np.array([S @ f[1] for f in firsts])
            sig = 0.5 * (float((pa @ A) @ (wa @ B)) + float((pa @ B) @ (wa @ A)))
            h = 0.5 * (pa @ A + pa @ B)
            mom = float(g_agg @ g_agg); bt = cfg.ema
            st["sig"] = sig if st["sig"] is None else bt * st["sig"] + (1 - bt) * sig
            st["mom"] = mom if st["mom"] is None else bt * st["mom"] + (1 - bt) * mom
            if st["gprev"] is not None:
                sdx = S @ (x - st["xprev"])
                num = float((h - st["gprev"]) @ sdx); den = float(sdx @ sdx)
                st["cnum"] = num if st["cnum"] is None else bt * st["cnum"] + (1 - bt) * num
                st["cden"] = den if st["cden"] is None else bt * st["cden"] + (1 - bt) * den
            st["gprev"], st["xprev"] = h, x.copy()
        x = x_new
        if t % cfg.eval_every == 0:
            if cfg.log_weights:
                log["weights"].append(wts.tolist())
            log["tau2"].append(tau2)
    log["H"] = fed.sys.H.tolist(); log["b"] = fed.sys.b.tolist(); log["m"] = fed.m.tolist()
    log["s2_last"] = (v2 / (fed.sys.H * fed.sys.b)).tolist()
    return log
