"""Table of tail-averaged optimality gaps for the HEW suite (+ baselines from grid/tune)."""
import json, glob, os, sys, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e2", "grid_fmnist_softmax")
M = ["hewx_reg", "hewx_lag", "phew", "fedavg", "p", "iv", "relmax", "posthoc_hew", "fedadp", "scaffold", "gls_oracle",
     "hew_nodebias", "hew", "hew_gls", "hew_fresh", "hewx", "hewx_gls"]
A = [None, 1.0, 0.1]; T = [4.0, 16.0]
def get(a, th, m, key="floor_gap"):
    tag = "ainf" if a is None else f"a{a}"
    fs = [f for f in glob.glob(f"{D}/{tag}_th{th}_Hb_s*_{m}.json") if os.path.basename(f).split("_", 4)[4] == m + ".json"]
    v = [json.load(open(f))[key] for f in fs]
    return (np.mean(v), np.std(v), len(v)) if v else None
if __name__ == "__main__":
    hdr = [f"{'iid' if a is None else a}/{int(t)}" for a in A for t in T]
    print(f"{'method':14s}" + "".join(f"{h:>16s}" for h in hdr))
    for m in M:
        row = f"{m:14s}"
        for a in A:
            for t in T:
                r = get(a, t, m)
                row += f"{'-':>16s}" if r is None else f"{r[0]:10.4f}({r[2]})    "
        print(row)

NAMES = {"fedavg": "FedAvg (unnormalized, $w=p$)", "p": "Normalized, $w=p$ (FedNova-type)",
         "iv": "Inverse-variance (plug-in)", "fedadp": "FedAdp (angle-based)", "scaffold": "SCAFFOLD, $w=p$",
         "posthoc_hew": "Original HEW (post-local, $\\Lambda=1.5L$)",
         "hew_nodebias": "HEW certificate, $\\Lambda=L/\\vartheta$, no debiasing",
         "gls_oracle": "GLS weights, oracle $Z$ (reference)",
         "hewx_reg": "\\textbf{Debiased RE-HEW} (ours)", "hewx_lag": "One-round-delayed RE-HEW (\\cref{thm:lag})", "phew": "RE path, selected by estimated round risk", "hewx": "RE-HEW, no shrinkage ($\\epsilon_t=0$)", "hewx_gls": "GLS-HEW (cross-round)",
         "hew": "RE-HEW, same-round debiasing (EMA)", "hew_fresh": "RE-HEW, same-round debiasing, no EMA",
         "relmax": "Lagged RE-HEW, same-round (predictable)", "hew_gls": "GLS-HEW, same-round debiasing"}
GROUPS = [["fedavg", "p", "iv", "fedadp", "scaffold", "posthoc_hew"],
          ["hewx_reg", "hewx_lag", "hewx", "hewx_gls", "phew"],
          ["hew_nodebias", "hew", "hew_fresh", "relmax", "hew_gls", "gls_oracle"]]

def latex(path="paper/tables/hew_main.tex"):
    cols = [(a, t) for a in A for t in T]
    best = {}
    for c in cols:
        vals = [get(*c, m) for g in GROUPS for m in g if m != "gls_oracle"]
        best[c] = min(v[0] for v in vals if v)
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             " & \\multicolumn{2}{c}{IID ($\\alpha=\\infty$)} & \\multicolumn{2}{c}{$\\alpha=1$} & \\multicolumn{2}{c}{$\\alpha=0.1$}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Aggregation & $\\vartheta=4$ & $\\vartheta=16$ & $\\vartheta=4$ & $\\vartheta=16$ & $\\vartheta=4$ & $\\vartheta=16$\\\\", "\\midrule"]
    for gi, g in enumerate(GROUPS):
        for m in g:
            row = NAMES[m]
            for c in cols:
                r = get(*c, m)
                if r is None: row += " & --"; continue
                s = f"{1e3*r[0]:.1f}"
                if m != "gls_oracle" and r[0] <= best[c] * 1.03: s = "\\textbf{" + s + "}"
                row += " & " + s
            lines.append(row + "\\\\")
        if gi < len(GROUPS) - 1: lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")

if __name__ == "__main__" and "--tex" in sys.argv:
    latex()

def latex_gnorm(path="paper/tables/hew_gnorm.tex"):
    ms = ["p", "fedavg", "scaffold", "posthoc_hew", "hewx_lag", "hewx_reg", "phew", "gls_oracle"]
    cols = [(a, t) for a in A for t in T]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             " & \\multicolumn{2}{c}{IID} & \\multicolumn{2}{c}{$\\alpha=1$} & \\multicolumn{2}{c}{$\\alpha=0.1$}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Aggregation & $\\vartheta=4$ & $\\vartheta=16$ & $\\vartheta=4$ & $\\vartheta=16$ & $\\vartheta=4$ & $\\vartheta=16$\\\\", "\\midrule"]
    for m in ms:
        row = NAMES[m]
        for c in cols:
            r = get(*c, m, key="floor_gnorm2")
            row += " & --" if r is None else f" & {1e3*r[0]:.2f}"
        lines.append(row + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")

if __name__ == "__main__" and "--tex" in sys.argv:
    latex_gnorm()
