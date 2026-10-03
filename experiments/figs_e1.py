"""Figures for E1 (hard-instance validation)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights.plots import setup, line, plt, COLORS, LABELS, MUTED, INK

R_ = os.path.join(os.path.dirname(__file__), "..", "results")
F_ = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F_, exist_ok=True)
setup()

# ---------- E1a: risk / lower bound vs budget ----------
d = json.load(open(os.path.join(R_, "e1a.json")))
taus = [k for k in d if k != "meta"]
fig, axes = plt.subplots(1, len(taus), figsize=(2.6 * len(taus), 2.4), sharey=True)
for ax, tk in zip(axes, taus):
    rec = d[tk]; R = np.array(rec["R"]); LB = np.array(rec["LB"])
    for key in ["p", "work", "iv", "re_opt"]:
        line(ax, R, np.array(rec[key]) / LB, key, marker="o")
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$\\tau^2={float(tk):g}$"); ax.set_xlabel("rounds $R$")
axes[0].set_ylabel("risk / lower bound")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
fig.savefig(os.path.join(F_, "e1a_attain.pdf")); plt.close(fig)

# ---------- E1b: SCAFFOLD neutralization ----------
d = json.load(open(os.path.join(R_, "e1b.json")))
taus = [k for k in d if k != "meta"]
fig, axes = plt.subplots(1, len(taus), figsize=(2.9 * len(taus), 2.4), sharey=True)
for ax, tk in zip(axes, taus):
    rec = d[tk]; R = np.array(rec["R"]); LB = np.array(rec["LB"])
    for key in ["plain_p", "plain_re_opt"]:
        line(ax, R, np.array(rec[key]) / LB, key, marker="o", zorder=2)
    for key, lw in [("scaffold_p", 3.0), ("scaffold_iv", 1.8), ("scaffold_work", 1.4)]:
        line(ax, R, np.array(rec[key]) / LB, key, lw=lw, zorder=3, alpha=0.9)
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$H_i\\equiv1$, $\\tau^2={float(tk):g}$"); ax.set_xlabel("rounds $R$")
axes[0].set_ylabel("risk / lower bound")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
fig.savefig(os.path.join(F_, "e1b_scaffold.pdf")); plt.close(fig)

# ---------- E1c: value of weights, predicted vs measured ----------
rows = json.load(open(os.path.join(R_, "e1c.json")))
fig, ax = plt.subplots(figsize=(2.8, 2.6))
spreads = sorted(set(r["spread"] for r in rows))
seq = ["#9ec5f0", "#5a9be6", "#2a78d6", "#154a8a"]          # one hue, light -> dark = spread
for c, sp in zip(seq, spreads):
    rr = [r for r in rows if r["spread"] == sp]
    ax.scatter([r["predicted"] for r in rr], [r["measured"] for r in rr], s=22, color=c,
               edgecolor="white", linewidth=0.6, label=f"noise spread {sp}$\\times$", zorder=3)
lim = [0.95, max(r["predicted"] for r in rows) * 1.15]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted value (Thm. value)"); ax.set_ylabel("measured risk ratio $p$ / RE")
ax.legend(loc="upper left")
fig.savefig(os.path.join(F_, "e1c_value.pdf")); plt.close(fig)

# ---------- E1d: post-hoc rule weights vs noise ----------
dd = json.load(open(os.path.join(R_, "e1d.json")))
s2 = np.array(dd["s2"]); W = np.array(dd["mean_weight"]); iv = np.array(dd["iv"])
# theoretical limit (Thm anti-IV): w_i = (kappa - lam/s_i^2)_+, kappa = L q_i/(Lam*theta)
kappa = dd["kappa"]
lo, hi = 0.0, kappa * s2.max()
for _ in range(200):
    lam = 0.5 * (lo + hi)
    w = np.maximum(kappa - lam / s2, 0)
    lo, hi = (lam, hi) if w.sum() > 1 else (lo, lam)
fig, ax = plt.subplots(figsize=(2.9, 2.4))
ax.plot(s2, W, "o", color=COLORS["posthoc_hew"], label="post-hoc projection (measured)", zorder=3)
ax.plot(s2, w, "-", color=COLORS["posthoc_hew"], lw=1.2, label="Thm. limit $(\\kappa-\\lambda/s_i^2)_+$")
ax.plot(s2, iv, "-", color=COLORS["iv"], label="inverse-variance (optimal, $\\tau=0$)")
ax.set_xscale("log"); ax.set_xlabel("client noise $s_i^2$"); ax.set_ylabel("aggregation weight")
ax.legend(loc="upper center")
fig.savefig(os.path.join(F_, "e1d_antiiv.pdf")); plt.close(fig)
print("E1 figures written")
