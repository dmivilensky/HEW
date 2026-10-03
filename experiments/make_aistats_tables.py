"""Generate all tables of the AISTATS submission (paper_aistats/tables/*.tex) from results/."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import table_hew as T
import table_e3 as E
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_aistats", "tables")
os.makedirs(OUT, exist_ok=True)
cols = [(a, t) for a in T.A for t in T.T]
HEAD = ["\\begin{tabular}{lcccccc}", "\\toprule",
        " & \\multicolumn{2}{c}{IID} & \\multicolumn{2}{c}{$\\alpha=1$} & \\multicolumn{2}{c}{$\\alpha=0.1$}\\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Aggregation & $\\vartheta=4$ & $\\vartheta=16$ & $\\vartheta=4$ & $\\vartheta=16$ & $\\vartheta=4$ & $\\vartheta=16$\\\\", "\\midrule"]
N = {"fedavg": "FedAvg", "p": "Normalized, $w=p$", "iv": "Inverse-variance", "fedadp": "FedAdp", "scaffold": "SCAFFOLD",
     "posthoc_hew": "Plain HEW, $\\Lambda=1.5L$", "hew_nodebias": "Plain HEW, $\\Lambda=L/\\vartheta$",
     "hewx_lag": "\\textbf{Debiased HEW} (Alg.~\\ref{alg:hew})", "hewx_reg": "\\quad post-local variant", "hewx": "\\quad no shrinkage",
     "hewx_gls": "\\quad GLS instead of RE", "phew": "\\quad path selection", "gls_oracle": "GLS, oracle $Z$ (reference)"}

def main_table():
    groups = [["fedavg", "p", "iv", "fedadp", "scaffold", "posthoc_hew", "hew_nodebias"],
              ["hewx_lag", "hewx_reg", "hewx", "hewx_gls", "phew"], ["gls_oracle"]]
    best = {c: min(T.get(*c, m)[0] for g in groups[:2] for m in g) for c in cols}
    L = list(HEAD)
    for gi, g in enumerate(groups):
        for m in g:
            row = N[m]
            for c in cols:
                v = f"{1e3 * T.get(*c, m)[0]:.1f}"
                if gi < 2 and T.get(*c, m)[0] <= best[c] * 1.03:
                    v = "\\textbf{" + v + "}"
                row += " & " + v
            L.append(row + "\\\\")
        if gi < 2:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "main_compact.tex"), "w").write("\n".join(L) + "\n")

def gnorm_table():
    ms = ["fedavg", "p", "scaffold", "posthoc_hew", "hewx_lag", "hewx_reg", "phew", "gls_oracle"]
    names = dict(N, hewx_lag="Debiased HEW (Alg.~\\ref{alg:hew})")
    L = list(HEAD)
    for m in ms:
        L.append(names[m] + "".join(f" & {1e3 * T.get(*c, m, key='floor_gnorm2')[0]:.2f}" for c in cols) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "hew_gnorm.tex"), "w").write("\n".join(L) + "\n")

def e3_compact():
    def cell(suite, a, part, m):
        g = E.get(suite, a, 16.0, part, m, "floor_gap"); q = E.get(suite, a, 16.0, part, m, "floor_gnorm2")
        return f"{1e3 * g[0]:.1f} / {1e3 * q[0]:.2f}"
    L = ["\\begin{tabular}{lcc}", "\\toprule", "& $\\alpha=1$ & $\\alpha=0.1$\\\\", "\\midrule",
         "\\multicolumn{3}{l}{\\emph{Full participation}}\\\\",
         f"Normalized, $w=p$ & {cell('step','1.0',1.0,'p')} & {cell('step','0.1',1.0,'p')}\\\\"]
    for lab, s, m in [("Debiased HEW", "sketch", "hew"), ("\\quad + sketch, GLS", "sketch", "hew_sk_gls"),
                      ("\\quad + sketch, path", "sketch", "hew_sk_path"), ("SCAFFOLD", "scaffold", "scaffold"),
                      ("SCAFFOLD + HEW", "scaffold", "scaffold_hew")]:
        L.append(f"{lab} & {cell(s,'1.0',1.0,m)} & {cell(s,'0.1',1.0,m)}\\\\")
    L.append("\\midrule\\multicolumn{3}{l}{\\emph{30\\% participation}}\\\\")
    for lab, m in [("Normalized, $w=p$", "p"), ("Debiased HEW", "hew"), ("FedAvg", "fedavg"), ("SCAFFOLD", "scaffold")]:
        L.append(f"{lab} & {cell('partial','1.0',0.3,m)} & {cell('partial','0.1',0.3,m)}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "e3_compact.tex"), "w").write("\n".join(L) + "\n")

def seeds_table():
    names = {"p": "Normalized, $w=p$", "hew": "Debiased HEW (Alg.~\\ref{alg:hew})", "fedavg": "FedAvg",
             "scaffold": "SCAFFOLD", "scaffold_hew": "SCAFFOLD + debiased HEW"}
    A = [("inf", "IID"), ("1.0", "$\\alpha=1$"), ("0.1", "$\\alpha=0.1$")]
    L = ["\\begin{tabular}{l" + "c" * len(A) + "}", "\\toprule", "Method & " + " & ".join(l for _, l in A) + "\\\\", "\\midrule"]
    stats = {}
    for m in names:
        row = names[m]
        for a, _ in A:
            fs = sorted(glob.glob(f"{E.R}/seeds_fmnist_softmax/a{a}_th16.0_part1.0_s*_{m}.json"))
            fs = [f for f in fs if f.endswith(f"_{m}.json") and os.path.basename(f).split("_s", 1)[1].split("_", 1)[1] == m + ".json"]
            v = np.array([json.load(open(f))["floor_gap"] for f in fs]) * 1e3
            stats[(m, a)] = v
            row += f" & {v.mean():.1f} $\\pm$ {v.std(ddof=1):.1f}" if len(v) > 1 else " & --"
        L.append(row + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "seeds.tex"), "w").write("\n".join(L) + "\n")
    return stats

if __name__ == "__main__":
    main_table(); gnorm_table(); e3_compact(); E.tex()
    st = seeds_table()
    for (m, a), v in st.items():
        print(m, a, np.round(v, 1))
