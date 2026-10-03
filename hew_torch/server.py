"""Server side of debiased HEW-Local SGD (model-agnostic: works on flattened update vectors).

Protocol per round t (see run.py for a full loop):
    w, gamma = server.begin_round(active, H, b)       # predictable quantities ("delayed" timing)
    ... clients train locally with lr_i = base_lr / H_i and return ClientMessage ...
    delta = server.end_round(messages)                 # aggregated model update (already scaled by gamma)
    params += delta

All statistics use only what the clients send: the update Delta_i, one scalar (split-batch noise
estimate) and, if het == "sketch" or step == "adaptive_sk", two k-dimensional sketches of the
first-step half-batch gradients.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ServerConfig:
    weight: str = "hew"          # "p" | "iv" | "hew"
    het: str = "cross"           # "cross" (cross-round pseudo-gradient Gram) | "sketch" (first-step gradient sketch)
    rule: str = "re"             # "re" | "gls" | "path"
    timing: str = "delayed"      # "delayed" (predictable) | "post"
    shrink: bool = True          # add eps_t calibrated from round-to-round variability
    step: str = "fixed"          # "fixed" | "adaptive" (cross-round SNR + secant curvature) | "adaptive_sk"
    ema: float = 0.8
    gamma_min: float = 0.02
    theta_max: float = 16.0      # upper clip of the online estimate of theta
    s2_floor: float = 1e-12


@dataclass
class ClientMessage:
    cid: int
    delta: torch.Tensor          # flattened model update y^H - x
    v2: float                    # per-sample variance estimate from split minibatches
    H: int
    b: int
    skA: torch.Tensor | None = None   # sketch of first-step half-batch gradient A
    skB: torch.Tensor | None = None   # sketch of first-step half-batch gradient B


# ---------------- small dense helpers (n x n, numpy) ----------------
def _psd(M):
    M = 0.5 * (M + M.T); ev, U = np.linalg.eigh(M); return (U * np.maximum(ev, 0)) @ U.T


def _lmax_centered(M):
    k = M.shape[0]; P = np.eye(k) - np.ones((k, k)) / k
    return float(np.linalg.eigvalsh(P @ M @ P)[-1])


def re_weights(p, tau2, x):
    """argmin_{w in simplex} tau2 ||w-p||^2 + sum w_i^2 x_i  (closed form, Prop. RE)."""
    D = tau2 + x; c = (1.0 - tau2 * np.sum(p / D)) / np.sum(1.0 / D)
    return (c + tau2 * p) / D


def simplex_qp(M, q, iters=600):
    """min_w w^T M w + q^T w over the simplex (FISTA with sort-based projection)."""
    n = len(q); Lg = 2 * np.linalg.eigvalsh(M).max() + 1e-12
    def proj(v):
        u = np.sort(v)[::-1]; css = np.cumsum(u); r = np.nonzero(u * np.arange(1, n + 1) > css - 1)[0][-1]
        return np.maximum(v - (css[r] - 1) / (r + 1), 0)
    w = np.ones(n) / n; y = w.copy(); tk = 1.0
    for _ in range(iters):
        w_new = proj(y - (2 * M @ y + q) / Lg); tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = w_new + (tk - 1) / tk1 * (w_new - w); w, tk = w_new, tk1
    return w


class HEWServer:
    def __init__(self, n, p, dim, base_lr, cfg: ServerConfig, device="cpu"):
        self.n, self.p, self.dim, self.base_lr, self.cfg, self.device = n, np.asarray(p, float), dim, base_lr, cfg, device
        self.v2 = np.full(n, np.nan)
        self.Z = np.zeros((n, n)); self.seen = np.zeros((n, n), bool); self.dev2 = None
        self.g_last = {}                     # cid -> last pseudo-gradient (torch, on device)
        self.gamma = 1.0
        self.st = dict(sig=None, mom=None, cnum=None, cden=None, gprev=None, xprev_delta=None, eta_prev=None, V_prev=None)
        self._pending = None
        self.theta_scale = 1.0
        self.diag = {}
        self.tau2_s = None; self.dev2_s = None
        self.theta_used = float("nan")

    # ---------- Gram EMA ----------
    def _gram_update(self, act, Zt):
        if not np.all(np.isfinite(Zt)):          # skip corrupted statistics
            return
        ix = np.ix_(act, act); b = self.cfg.ema
        if self.seen[ix].all():
            k = len(act); P = np.eye(k) - np.ones((k, k)) / k
            D = P @ (Zt - self.Z[ix]) @ P
            dv = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (D + D.T))))) ** 2
            self.dev2 = dv if self.dev2 is None else b * self.dev2 + (1 - b) * dv
        self.Z[ix] = np.where(self.seen[ix], b * self.Z[ix] + (1 - b) * Zt, Zt); self.seen[ix] = True
        # scalar heterogeneity level lambda_max(P Z P) of the current round, smoothed over rounds. Used by the RE rule
        # when the active clients were never all observed together (partial participation), so that the Gram
        # matrix of this subset is incomplete; its round-to-round spread gives the shrinkage in that case.
        t2 = max(_lmax_centered(_psd(Zt)), 0.0)
        if self.tau2_s is None:
            self.tau2_s = t2
        else:
            self.dev2_s = (t2 - self.tau2_s) ** 2 if self.dev2_s is None else b * self.dev2_s + (1 - b) * (t2 - self.tau2_s) ** 2
            self.tau2_s = b * self.tau2_s + (1 - b) * t2

    def _eps(self):
        return 0.0 if (self.dev2 is None or not self.cfg.shrink) else float(np.sqrt(0.5 * (1 - self.cfg.ema) * self.dev2))

    # ---------- weights ----------
    def _weights(self, act, H, b, theta_eff):
        act = np.asarray(act); pa = self.p[act] / self.p[act].sum()
        if self.cfg.weight == "p" or np.any(np.isnan(self.v2[act])):
            return pa
        s2 = np.maximum(self.v2[act] / (np.asarray(H) * np.asarray(b)), self.cfg.s2_floor)
        x = theta_eff * s2
        if self.cfg.weight == "iv":
            a = 1 / s2; return a / a.sum()
        if not self.seen[np.ix_(act, act)].all():
            # partial participation: this subset was never observed together. The RE rule needs only the scalar
            # tau^2, so use its smoothed estimate; the other rules fall back to p.
            if self.cfg.rule != "re" or self.tau2_s is None:
                return pa
            eps = 0.0 if (self.dev2_s is None or not self.cfg.shrink) else float(np.sqrt(0.5 * (1 - self.cfg.ema) * self.dev2_s))
            self.diag.update(tau2=self.tau2_s, eps=eps, x_med=float(np.median(x)), x_min=float(x.min()), x_max=float(x.max()))
            return re_weights(pa, self.tau2_s + eps, x)
        Zc = self.Z[np.ix_(act, act)]; eps = self._eps()
        tau2 = max(_lmax_centered(_psd(Zc)), 0.0)
        self.diag.update(tau2=tau2, eps=eps, x_med=float(np.median(x)), x_min=float(x.min()), x_max=float(x.max()))
        if self.cfg.rule == "re":
            return re_weights(pa, tau2 + eps, x)
        if self.cfg.rule == "gls":
            M = _psd(Zc) + eps * np.eye(len(pa)); return simplex_qp(M + np.diag(x), -2 * M @ pa)
        best, wb = np.inf, pa
        for k2 in [0.0] + list(np.median(x) * 2.0 ** np.arange(-8, 9)) + [None]:
            cw = pa.copy() if k2 is None else re_weights(pa, k2, x); d = cw - pa
            crit = max(d @ Zc @ d, 0.0) + eps * (d @ d) + np.sum(cw ** 2 * x)
            if crit < best:
                best, wb = crit, cw
        return wb

    def curvature(self):
        """Debiased secant estimate of the curvature along the server steps (None before 2 rounds)."""
        st = self.st
        if st.get("knum") is None or not st.get("kden"):
            return None
        return max(st["knum"] / st["kden"], 1e-12)

    def begin_round(self, act, H, b, theta=None):
        """Call before local training. theta = base_lr * L (variance weight of the round risk).
        If None, it is estimated online as base_lr * (debiased secant curvature), clipped to [1e-3, theta_max]."""
        c, st = self.cfg, self.st
        if theta is None:
            ell = self.curvature()
            theta = 1.0 if ell is None else float(np.clip(self.base_lr * ell, 1e-3, c.theta_max))
        self.theta_used = theta
        if c.step in ("adaptive", "adaptive_sk") and st["sig"] is not None and st["cden"]:
            ell = max(st["cnum"] / st["cden"], 1e-12)
            eta_star = max(st["sig"], 0.0) / (ell * st["mom"])
            self.gamma = float(np.clip(eta_star / self.base_lr, c.gamma_min, 1.0))
        self._pending = (list(act), list(H), list(b), theta)
        w = self._weights(act, H, b, self.gamma * theta) if c.timing == "delayed" else None
        self._w_delayed = w
        return w, self.gamma

    def end_round(self, msgs: list[ClientMessage]):
        """Aggregate. Returns the model delta to add to the parameters."""
        c = self.cfg; act, H, b, theta = self._pending
        order = {m.cid: m for m in msgs}; msgs = [order[i] for i in act]
        pa = self.p[act] / self.p[act].sum()
        for m in msgs:
            if not np.isfinite(m.v2):
                continue
            i = m.cid; self.v2[i] = m.v2 if np.isnan(self.v2[i]) else c.ema * self.v2[i] + (1 - c.ema) * m.v2
        G = torch.stack([-m.delta / self.base_lr for m in msgs])                     # pseudo-gradients
        if not torch.isfinite(G).all():
            bad = [m.cid for m in msgs if not torch.isfinite(m.delta).all()]
            raise FloatingPointError(f"non-finite client updates from clients {bad}; "
                                     f"local training diverged - reduce --lr")
        pa_t = torch.tensor(np.asarray(pa, dtype=np.float32), device=G.device)
        if c.weight == "hew":
            if c.het == "cross":
                if all(i in self.g_last for i in act):
                    Gp = torch.stack([self.g_last[i] for i in act])
                    E = G - pa_t @ G; Ep = Gp - pa_t @ Gp
                    C = (E @ Ep.T).detach().cpu().double().numpy(); self._gram_update(act, 0.5 * (C + C.T))
            else:
                A = torch.stack([m.skA for m in msgs]); B = torch.stack([m.skB for m in msgs])
                EA = A - pa_t @ A; EB = B - pa_t @ B
                C = (EA @ EB.T).detach().cpu().double().numpy(); self._gram_update(act, 0.5 * (C + C.T))
        for j, i in enumerate(act):
            self.g_last[i] = G[j].detach().clone()
        w = self._w_delayed if c.timing == "delayed" else self._weights(act, H, b, self.gamma * theta)
        w_t = torch.tensor(np.asarray(w, dtype=np.float32), device=G.device)
        g_agg = w_t @ G
        eta = self.gamma * self.base_lr
        delta = -eta * g_agg
        self._step_stats(msgs, w, pa, g_agg, delta, H, b)
        self.last_weights = dict(zip(act, w.tolist()))
        return delta

    # ---------- adaptive step statistics (used from next round on) ----------
    def _step_stats(self, msgs, w, pa, g_agg, delta, H, b):
        c, st = self.cfg, self.st
        bt = c.ema
        s2 = np.maximum(np.array([self.v2[m.cid] for m in msgs]) / (np.array(H) * np.array(b)), c.s2_floor)
        V = float(np.sum(w ** 2 * s2)); mom = float(g_agg @ g_agg)
        upd = lambda key, val: val if st.get(key) is None else bt * st[key] + (1 - bt) * val
        # debiased secant curvature along server steps (always tracked; used for theta=None)
        if st.get("k_g") is not None:
            dx = st["k_dx"]
            st["knum"] = upd("knum", float((g_agg - st["k_g"]) @ dx) - st["k_eta"] * st["k_V"])
            st["kden"] = upd("kden", float(dx @ dx))
        st["k_g"], st["k_dx"], st["k_eta"], st["k_V"] = g_agg.clone(), delta.clone(), self.gamma * self.base_lr, V
        if c.step not in ("adaptive", "adaptive_sk"):
            return
        st["mom"] = upd("mom", mom)
        if c.step == "adaptive":
            if st["gprev"] is not None:
                st["sig"] = upd("sig", float(g_agg @ st["gprev"]))
                dx = st["xprev_delta"]                               # x_t - x_{t-1}
                num = float((g_agg - st["gprev"]) @ dx) - st["eta_prev"] * st["V_prev"]
                st["cnum"] = upd("cnum", num); st["cden"] = upd("cden", float(dx @ dx))
            st["gprev"], st["eta_prev"], st["V_prev"] = g_agg.clone(), self.gamma * self.base_lr, V
            st["xprev_delta"] = delta.clone()
        else:
            A = torch.stack([m.skA for m in msgs]); B = torch.stack([m.skB for m in msgs])
            w_t = torch.tensor(np.asarray(w, dtype=np.float32), device=A.device); p_t = torch.tensor(np.asarray(pa, dtype=np.float32), device=A.device)
            sig = 0.5 * (float((p_t @ A) @ (w_t @ B)) + float((p_t @ B) @ (w_t @ A)))
            h = 0.5 * (p_t @ A + p_t @ B)
            st["sig"] = upd("sig", sig)
            if st["gprev"] is not None:
                sdx = st["xprev_delta"]                              # sketch of x_t - x_{t-1}
                st["cnum"] = upd("cnum", float((h - st["gprev"]) @ sdx)); st["cden"] = upd("cden", float(sdx @ sdx))
            st["gprev"] = h
            st["xprev_delta"] = self.sketch(delta) if hasattr(self, "sketch") else None
