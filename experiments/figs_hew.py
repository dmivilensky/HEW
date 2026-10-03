"""Figure for the HEW paper: convergence (IID / alpha=0.1, theta=16) and learned weights vs noise."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedweights import plots
import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e2", "grid_fmnist_softmax")
plots.COLORS.update({"hewx_lag": "#2a78d6", "hew_nodebias": "#8a8984"})
plots.STYLE.update({"fedavg": "--", "scaffold": ":"})
plots.LABELS.update({"hewx_lag": "debiased HEW (Alg. 1)", "posthoc_hew": "plain HEW, $\\Lambda=1.5L$",
                     "p": "normalized, $w=p$", "fedavg": "FedAvg", "scaffold": "SCAFFOLD"})
def load(tag, m, s):
    return json.load(open(f"{D}/{tag}_th16.0_Hb_s{s}_{m}.json"))
plots.setup()
fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
for k, (tag, title) in enumerate([("ainf", "IID, $\\vartheta=16$"), ("a0.1", "Dirichlet $\\alpha=0.1$, $\\vartheta=16$")]):
    for m in ["fedavg", "scaffold", "posthoc_hew", "p", "hewx_lag"]:
        g = np.mean([load(tag, m, s)["gap"] for s in (0, 1)], 0); r = load(tag, m, 0)["rounds"]
        plots.line(ax[k], r[1:], g[1:], m)
    ax[k].set_yscale("log"); ax[k].set_title(title); ax[k].set_xlabel("round"); ax[k].set_ylabel("$F(x_t)-F_{\\inf}$")
    ax[k].set_ylim(3e-3, 3e-1)
ax[0].legend(loc="upper right")
# weights vs per-round noise level, seed 0, last logged round
for tag, mk, lab in [("ainf", "o", "IID"), ("a0.1", "s", "$\\alpha=0.1$")]:
    d = load(tag, "hewx_lag", 0); s2 = np.array(d["s2_hat"][-1]); w = np.array(d["weights_last"])
    ax[2].scatter(s2, w * len(w), marker=mk, s=18, color="#2a78d6" if tag == "ainf" else "#eb6834", label=f"debiased HEW, {lab}")
d = load("ainf", "hewx_lag", 0); s2 = np.array(d["s2_hat"][-1]); p = np.array(d["m"]) / np.sum(d["m"])
iv = (1 / s2) / np.sum(1 / s2)
o = np.argsort(s2); ax[2].plot(s2[o], iv[o] * len(iv), color="#52514e", lw=1, ls="--", label="inverse-variance")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("per-round noise $\\hat s_i^2$"); ax[2].set_ylabel("$n\\,w_i$")
ax[2].set_title("learned weights (last round)"); ax[2].legend(loc="lower left")
os.makedirs("figures", exist_ok=True); os.makedirs("paper_aistats/figures", exist_ok=True); fig.tight_layout(); fig.savefig("figures/hew_main.pdf"); fig.savefig("paper_aistats/figures/hew_main.pdf"); fig.savefig("figures/hew_main.png")
print("ok")
