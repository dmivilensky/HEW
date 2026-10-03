"""Closed-form quantities from the theory: risk functional, optimal (random-effects /
GLS) weights, Bayes lower bound, value of weights, crossover budget."""
from __future__ import annotations
import numpy as np


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(v) + 1)
    r = np.nonzero(u * k > (css - 1))[0][-1]
    return np.maximum(v - (css[r] - 1) / (r + 1), 0.0)


def simplex_qp(M, q, iters=600, w0=None):
    """min_w  w^T M w + q^T w  over the probability simplex (M PSD). FISTA."""
    n = len(q)
    Lg = 2 * np.linalg.eigvalsh(M).max() + 1e-12
    w = np.ones(n) / n if w0 is None else w0.copy()
    z, tk = w.copy(), 1.0
    for _ in range(iters):
        wn = proj_simplex(z - (2 * M @ z + q) / Lg)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        z = wn + (tk - 1) / tn * (wn - w)
        w, tk = wn, tn
    return w


def re_weights(p, tau2, x, active=None):
    """Minimizer over the simplex (support = active) of tau2*||w-p||^2 + sum_i w_i^2 x_i.
    Closed form: w_i = (c + tau2 p_i)/(tau2 + x_i), c >= 0 by normalization."""
    n = len(p)
    act = np.arange(n) if active is None else np.asarray(active)
    D = tau2 + x[act]
    c = (1.0 - tau2 * np.sum(p[act] / D)) / np.sum(1.0 / D)
    w = np.zeros(n)
    w[act] = (c + tau2 * p[act]) / D
    return w


def risk(w, p, x, tau2=None, Z=None):
    a = w - p
    bias = a @ Z @ a if Z is not None else tau2 * a @ a
    return bias + np.sum(w ** 2 * x)


def re_value(p, tau2, x):
    """Optimal value of tau2||w-p||^2 + sum w^2 x over the affine set sum w = 1
    (= Bayes lower bound on the random-effects hard family when x_i = v_i^2/N_i)."""
    w = re_weights(p, tau2, x)
    return risk(w, p, x, tau2=tau2)


def gls_oracle_weights(p, Z, x, active=None, w0=None):
    """Minimizer over the simplex (support=active) of (w-p)^T Z (w-p) + sum w^2 x."""
    n = len(p)
    act = np.arange(n) if active is None else np.asarray(active)
    M = Z[np.ix_(act, act)] + np.diag(x[act])
    # (w-p)^T Z (w-p) with w supported on act: w_act^T Z_aa w_act - 2 w_act^T (Z p)_act + const
    q = -2 * (Z @ p)[act]
    wa = simplex_qp(M, q, w0=None if w0 is None else w0[act] / max(w0[act].sum(), 1e-12))
    w = np.zeros(n)
    w[act] = wa
    return w


def value_of_weights(p, tau2, x):
    """Risk(p)/Risk(optimal) for the random-effects functional."""
    return risk(p, p, x, tau2=tau2) / re_value(p, tau2, x)


def value_bounds_uniform(tau2, x):
    """Two-sided bounds on Val-1 for uniform p (Theorem: phase transition)."""
    xm, var = x.mean(), x.var()
    lo = tau2 * var / ((tau2 + x.max()) ** 2 * xm)
    hi = var / ((tau2 + x.min()) * x.min())
    return lo, hi


def crossover_budget(tau2, v2_over_Hb):
    """R_x ~ Var_i(v_i^2/(H_i b_i)) / (tau^2 * mean_i(v_i^2/(H_i b_i)))."""
    return v2_over_Hb.var() / (tau2 * v2_over_Hb.mean())
