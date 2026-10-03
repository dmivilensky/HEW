"""Figures and LaTeX tables for the real-data experiments (E2-E4)."""
import sys, os, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights.plots import setup, line, plt, COLORS, LABELS, MUTED, INK

ROOT = os.path.join(os.path.dirname(__file__), "..")
F_ = os.path.join(ROOT, "figures")
T_ = os.path.join(ROOT, "paper", "tables")
os.makedirs(F_, exist_ok=True); os.makedirs(T_, exist_ok=True)
setup()

SHORT = {"gls": "GLS (ours)", "relmax": "RE-$\\lambda_{\\max}$ (ours)", "re_lmax_oracle": "RE-$\\lambda_{\\max}$ oracle",
         "p": "$p$ (FedNova)", "work": "work", "iv": "inv.-var.", "re": "RE-mean", "re_oracle": "RE oracle",
         "gls_oracle": "GLS oracle", "posthoc_hew": "post-hoc", "fedadp": "FedAdp", "scaffold": "SCAFFOLD",
         "scaffold_iv": "SCAF.+iv", "scaffold_re": "SCAF.+RE", "fedavg": "FedAvg"}


def load(suite):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", "e2", suite, "*.json")):
        r = json.load(open(f))
        out[(str(r["alpha"]), str(r["theta"]), r["method"])].append(r)
    return out


def agg(rs, key):
    v = np.array([r[key] for r in rs])
    return v.mean(), v.std()


def table(data, methods, regimes, key, caption, label, fname, rel_to="p", fmt="{:.2f}"):
    """Rows = methods, columns = regimes; entries = key(method)/key(rel_to), mean over seeds of ratio."""
    cols = []
    for (a, th) in regimes:
        if (a, th, rel_to) not in data:
            continue
        cols.append((a, th))
    lines = [r"\begin{table}[t]\centering\small", r"\caption{" + caption + "}\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}\\toprule"]
    lines.append("method & " + " & ".join(
        (r"$\alpha=\infty$" if a == "inf" else f"$\\alpha={a}$") + (f", $\\vartheta={th}$" if th not in ("decay",) else "")
        for a, th in cols) + r"\\\midrule")
    for m in methods:
        row = [SHORT.get(m, m)]
        best_cells = []
        for (a, th) in cols:
            rs, base = data.get((a, th, m)), data.get((a, th, rel_to))
            if not rs or not base:
                row.append("--"); continue
            bseed = {r["seed"]: r[key] for r in base}
            ratios = [r[key] / bseed[r["seed"]] for r in rs if r["seed"] in bseed]
            row.append(fmt.format(np.mean(ratios)) + (r"{\scriptsize$\pm$" + fmt.format(np.std(ratios)) + "}" if len(ratios) > 1 else ""))
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(T_, fname), "w").write("\n".join(lines))


def curves(data, regimes, methods, key, ylabel, fname, logy=True):
    regs = [r for r in regimes if any((r[0], r[1], m) in data for m in methods)]
    if not regs:
        return
    fig, axes = plt.subplots(1, len(regs), figsize=(2.6 * len(regs), 2.3), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (a, th) in zip(axes, regs):
        for m in methods:
            rs = data.get((a, th, m))
            if not rs:
                continue
            x = np.array(rs[0]["rounds"])
            y = np.mean([np.array(r[key]) for r in rs], 0)
            line(ax, x, y, m)
        if logy:
            ax.set_yscale("log")
        ttl = (r"$\alpha=\infty$" if a == "inf" else f"$\\alpha={a}$") + ("" if th == "decay" else f", $\\vartheta={th}$")
        ax.set_title(ttl); ax.set_xlabel("round")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(os.path.join(F_, fname)); plt.close(fig)


# ------------------------------------------------------------------ grid
G = load("grid_fmnist_softmax")
if G:
    regimes = [(a, th) for th in ["4.0", "16.0"] for a in ["inf", "1.0", "0.3", "0.1"]]
    methods = ["p", "work", "iv", "re", "relmax", "re_lmax_oracle", "gls", "gls_oracle", "posthoc_hew", "fedadp",
               "fedavg", "scaffold", "scaffold_iv"]
    table(G, methods, regimes, "floor_gnorm2",
          r"Softmax regression on Fashion-MNIST, $n=20$, heterogeneous $H_i\in\{1,\dots,16\}$ and $b_i\in\{8,\dots,128\}$. "
          r"Stationary squared gradient norm (mean of the last quarter of 300 rounds) relative to objective weights $p$; "
          r"lower is better; mean$\pm$std over seeds of the per-seed ratio.", "tab:grid_g2", "grid_g2.tex")
    table(G, methods, regimes, "final_acc",
          r"Same runs: final test accuracy relative to objective weights $p$ (higher is better).",
          "tab:grid_acc", "grid_acc.tex", fmt="{:.3f}")
    curves(G, [("inf", "16.0"), ("1.0", "16.0"), ("0.3", "16.0"), ("0.1", "16.0")],
           ["p", "work", "iv", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"],
           "gnorm2", r"$\|\nabla F(x_t)\|^2$", "e2_curves_th16.pdf")
    curves(G, [("inf", "4.0"), ("1.0", "4.0"), ("0.3", "4.0"), ("0.1", "4.0")],
           ["p", "work", "iv", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"],
           "gnorm2", r"$\|\nabla F(x_t)\|^2$", "e2_curves_th4.pdf")

    # predicted value (oracle quantities) vs observed gain, across regimes and seeds
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    seq = {"4.0": "#9ec5f0", "16.0": "#2a78d6"}
    pts = []
    for (a, th) in regimes:
        base, best = G.get((a, th, "p_log")), G.get((a, th, "gls_oracle"))
        if not base or not best:
            continue
        bs = {r["seed"]: r for r in base}
        for r in best:
            if r["seed"] not in bs:
                continue
            k = max(1, len(r["val_pred"]) // 4)
            pred = np.mean(bs[r["seed"]]["val_pred"][-k:])
            obs = bs[r["seed"]]["floor_gnorm2"] / r["floor_gnorm2"]
            pts.append((pred, obs, th, a))
    for th in ["4.0", "16.0"]:
        pp = [q for q in pts if q[2] == th]
        if pp:
            ax.scatter([q[0] for q in pp], [q[1] for q in pp], s=24, color=seq[th], edgecolor="white",
                       linewidth=0.6, label=f"$\\vartheta={th}$", zorder=3)
    if pts:
        lim = [0.8, max(max(q[0] for q in pts), max(q[1] for q in pts)) * 1.2]
        ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("predicted value of weights"); ax.set_ylabel(r"observed gain: floor($p$)/floor(GLS)")
        ax.legend(loc="upper left")
        fig.savefig(os.path.join(F_, "e2_value_scatter.pdf"))
    plt.close(fig)

    # estimator accuracy: tau2_hat vs oracle mean eigenvalue, s2_hat vs s2 oracle
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
    xs, ys, xa, ya = [], [], [], []
    for (a, th) in regimes:
        for r in G.get((a, th, "gls"), []):
            n = len(r["s2_oracle"][0])
            tr = np.array(r["trZ"]) / (n - 1); th_ = np.array(r["tau2_hat"])
            xs += tr[2:].tolist(); ys += th_[2:].tolist()
            so, sh = np.array(r["s2_oracle"][-1]), np.array(r["s2_hat"][-1])
            xa += so.tolist(); ya += sh.tolist()
    if xs:
        axes[0].scatter(xs, ys, s=6, color=COLORS["re"], alpha=0.5)
        lim = [min(xs + ys) * 0.8, max(xs + ys) * 1.2]
        axes[0].plot(lim, lim, color=MUTED, lw=0.8, ls="--"); axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel(r"oracle $\mathrm{tr}\,Z/(n-1)$"); axes[0].set_ylabel(r"moment estimate $\hat\tau^2$")
        axes[1].scatter(xa, ya, s=8, color=COLORS["re"], alpha=0.6)
        lim = [min(xa + ya) * 0.8, max(xa + ya) * 1.2]
        axes[1].plot(lim, lim, color=MUTED, lw=0.8, ls="--"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel(r"oracle $s_i^2$"); axes[1].set_ylabel(r"split-batch estimate $\hat s_i^2$")
        fig.savefig(os.path.join(F_, "e2_estimators.pdf"))
    plt.close(fig)

    # weights vs noise for post-hoc and RE on real data (theta=16, iid and alpha=0.3)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True)
    for ax, a in zip(axes, ["inf", "0.3"]):
        for m in ["gls", "posthoc_hew", "p"]:
            for r in G.get((a, "16.0", m), [])[:1]:
                W = np.array(r["weights"][len(r["weights"]) // 2:]).mean(0)
                s2 = np.array(r["s2_oracle"][len(r["s2_oracle"]) // 2:]).mean(0)
                o = np.argsort(s2)
                ax.plot(s2[o], W[o], "o-", color=COLORS[m], label=LABELS[m], ms=3, lw=1)
        ax.set_xscale("log"); ax.set_xlabel(r"client noise $s_i^2$ (oracle)")
        ax.set_title(r"$\alpha=\infty$" if a == "inf" else f"$\\alpha={a}$")
    axes[0].set_ylabel("mean weight"); axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(os.path.join(F_, "e2_weights_vs_noise.pdf")); plt.close(fig)

# ------------------------------------------------------------------ decay
D = load("decay_fmnist_softmax")
if D:
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.2), sharex=True)
    for j, a in enumerate(["1.0", "0.3"]):
        ax = axes[0, j]
        for m in ["p", "iv", "relmax", "gls", "gls_oracle"]:
            rs = D.get((a, "decay", m))
            if not rs:
                continue
            x = np.array(rs[0]["rounds"]); y = np.mean([np.array(r["gnorm2"]) for r in rs], 0)
            line(ax, x, y, m)
        ax.set_yscale("log"); ax.set_title(f"$\\alpha={a}$, $\\vartheta_t=16/\\sqrt{{1+t/20}}$")
        ax.set_ylabel(r"$\|\nabla F(x_t)\|^2$")
        ax = axes[1, j]
        rs = D.get((a, "decay", "gls_oracle"))
        if rs:
            x = np.array(rs[0]["rounds"])
            vp = np.mean([np.array(r["val_pred"]) for r in rs], 0)
            ax.plot(x, vp, color=INK, lw=1.4, label="predicted value (oracle)")
            b = D.get((a, "decay", "p")); o = D.get((a, "decay", "gls_oracle"))
            if b and o:
                ratio = np.mean([np.array(bb["gnorm2"]) for bb in b], 0) / np.mean([np.array(oo["gnorm2"]) for oo in o], 0)
                k = 5
                sm = np.convolve(ratio, np.ones(k) / k, mode="same")
                ax.plot(x, sm, color=COLORS["re"], lw=1.2, label=r"observed $\|\nabla F\|^2$ ratio $p$/GLS")
            ax.axhline(1, color=MUTED, lw=0.8, ls="--"); ax.set_yscale("log")
            ax.set_xlabel("round"); ax.set_ylabel("value of weights")
            ax.legend(loc="upper right")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(os.path.join(F_, "e3_decay.pdf")); plt.close(fig)

# ------------------------------------------------------------------ partial
P = load("partial_fmnist_softmax")
if P:
    regimes = [("inf", "8.0"), ("0.3", "8.0")]
    PM = ["p", "work", "iv", "relmax", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"]
    table(P, PM, regimes,
          "floor_gnorm2", r"Partial participation (6 of 20 clients per round) with stragglers (horizons re-drawn every round): "
          r"stationary $\|\nabla F\|^2$ relative to $p$.", "tab:partial_g2", "partial_g2.tex")
    table(P, PM, regimes,
          "final_acc", r"Partial participation with stragglers: final test accuracy relative to $p$.",
          "tab:partial_acc", "partial_acc.tex", fmt="{:.3f}")
    curves(P, regimes, ["p", "work", "iv", "relmax", "posthoc_hew", "fedadp", "scaffold", "fedavg"], "gnorm2",
           r"$\|\nabla F(x_t)\|^2$", "e4_partial.pdf")

# ------------------------------------------------------------------ mlp
M = load("mlp_fmnist_mlp")
if M:
    regimes = [("inf", "6.0"), ("0.3", "6.0")]
    table(M, ["p", "iv", "relmax", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"], regimes, "floor_gnorm2",
          r"Tanh-MLP (nonconvex) on Fashion-MNIST: stationary $\|\nabla F\|^2$ relative to $p$.", "tab:mlp_g2", "mlp_g2.tex")
    table(M, ["p", "iv", "relmax", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"], regimes, "final_acc",
          r"Tanh-MLP on Fashion-MNIST: final test accuracy relative to $p$.", "tab:mlp_acc", "mlp_acc.tex", fmt="{:.3f}")
    curves(M, regimes, ["p", "iv", "relmax", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"], "gnorm2",
           r"$\|\nabla F(x_t)\|^2$", "e2_mlp_curves.pdf")
    curves(M, regimes, ["p", "iv", "relmax", "gls", "posthoc_hew", "fedadp", "scaffold", "fedavg"], "test_acc",
           "test accuracy", "e2_mlp_acc.pdf", logy=False)
print("E2 figures/tables written")

# ------------------------------------------------------------------ tuned comparison (best over theta)
if G:
    thetas = ["4.0", "8.0", "16.0", "32.0"]
    TM = ["p", "relmax", "gls", "posthoc_hew", "scaffold", "fedavg"]
    rows = []
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Tuned comparison: for each method and heterogeneity level, the best loss gap $F(x)-F_\star$ "
             r"(mean of the last quarter of 300 rounds, averaged over seeds) over $\vartheta\in\{4,8,16,32\}$, "
             r"relative to the best objective-weight ($p$) run. The selected $\vartheta$ is in parentheses.}\label{tab:tuned}",
             r"\begin{tabular}{l" + "c" * 3 + r"}\toprule",
             r"method & $\alpha=\infty$ & $\alpha=1$ & $\alpha=0.1$\\\midrule"]
    def best(a, m):
        cands = []
        for th in thetas:
            rs = G.get((a, th, m))
            if rs:
                cands.append((np.mean([r["floor_gap"] for r in rs]), th, len(rs)))
        return min(cands) if cands else None
    for m in TM:
        row = [SHORT.get(m, m)]
        for a in ["inf", "1.0", "0.1"]:
            b, bp = best(a, m), best(a, "p")
            row.append("--" if (b is None or bp is None) else f"{b[0] / bp[0]:.2f} ({float(b[1]):g})")
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    open(os.path.join(T_, "tuned_gap.tex"), "w").write("\n".join(lines))
    print("tuned table written")
