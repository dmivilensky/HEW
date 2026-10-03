"""Tables for E3 (improvements). Usage: python experiments/table_e3.py [--tex]"""
import json, glob, os, sys, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e3")
def get(suite, a, th, part, m, key):
    fs = glob.glob(f"{R}/{suite}_fmnist_softmax/a{a}_th{th}_part{part}_s*_{m}.json")
    fs = [f for f in fs if os.path.basename(f).split("_s", 1)[1].split("_", 1)[1] == m + ".json"]
    v = [json.load(open(f))[key] for f in fs]
    return (float(np.mean(v)), len(v)) if v else None
NAMES = {"hew": "Debiased HEW (Alg.~\\ref{alg:hew})", "hew_sk": "Debiased HEW, gradient sketch",
         "hew_sk_gls": "GLS-HEW, gradient sketch", "hew_sk_path": "Path selection, gradient sketch",
         "p": "Normalized, $w=p$", "p_adapt": "$w=p$, SNR step (cross-round)", "p_adsk": "$w=p$, SNR step (sketch)",
         "p_decay": "$w=p$, $\\vartheta_t=\\vartheta/\\sqrt{1+t/20}$", "hew_adapt": "Debiased HEW, SNR step (cross-round)",
         "hew_adsk": "Debiased HEW, SNR step (sketch)", "hew_decay": "Debiased HEW, $\\vartheta_t=\\vartheta/\\sqrt{1+t/20}$",
         "hew_v1": "Debiased HEW, variance weight $1$", "hew_vauto": "Debiased HEW, weight $\\min(1,\\hat r)$",
         "hew_gls_v1": "GLS-HEW, weight $1$", "hew_path_v1": "Path, weight $1$", "hew_path_vauto": "Path, weight $\\min(1,\\hat r)$",
         "hew_sk_gls_v1": "GLS-HEW sketch, weight $1$",
         "hew_om": "Debiased HEW, weight $\\hat\\omega$", "hew_gls_om": "GLS-HEW, weight $\\hat\\omega$",
         "hew_path_om": "Path, weight $\\hat\\omega$", "hew_sk_gls_om": "GLS-HEW sketch, weight $\\hat\\omega$",
         "fedavg": "FedAvg", "scaffold": "SCAFFOLD, $w=p$", "scaffold_hew": "SCAFFOLD + debiased HEW"}
SUITES = {"sketch": ([("inf", 16.0), ("1.0", 16.0), ("0.1", 16.0)], 1.0, ["hew", "hew_sk", "hew_sk_gls", "hew_sk_path"]),
          "step": ([("inf", 16.0), ("1.0", 16.0), ("0.1", 16.0)], 1.0, ["p", "p_adapt", "p_adsk", "p_decay", "hew_adapt", "hew_adsk", "hew_decay"]),
          "varw": ([("inf", 16.0), ("1.0", 16.0), ("0.1", 16.0)], 1.0, ["hew_v1", "hew_vauto", "hew_gls_v1", "hew_path_v1", "hew_path_vauto", "hew_sk_gls_v1"]),
          "omega": ([("inf", 16.0), ("1.0", 16.0), ("0.1", 16.0)], 1.0, ["hew_om", "hew_gls_om", "hew_path_om", "hew_sk_gls_om"]),
          "partial": ([("1.0", 16.0), ("0.1", 16.0)], 0.3, ["p", "hew", "fedavg", "scaffold"]),
          "scaffold": ([("1.0", 4.0), ("1.0", 16.0), ("0.1", 4.0), ("0.1", 16.0)], 1.0, ["scaffold", "scaffold_hew"])}
def show():
    for s, (cols, part, ms) in SUITES.items():
        print(f"== {s} (gap x1e3 | gnorm x1e3)")
        for m in ms:
            row = f"{m:12s}"
            for a, th in cols:
                g = get(s, a, th, part, m, "floor_gap"); q = get(s, a, th, part, m, "floor_gnorm2")
                row += f"  {a}/{int(th)}: " + ("  -  " if g is None else f"{1e3*g[0]:6.1f} | {1e3*q[0]:6.3f} ({g[1]})")
            print(row)
def lab(a, th):
    A = 'IID' if a == 'inf' else '$\\alpha=' + a.rstrip('0').rstrip('.') + '$'
    return A + ', $\\vartheta=' + str(int(th)) + '$'

def tex():
    for s, (cols, part, ms) in SUITES.items():
        lines = ["\\begin{tabular}{l" + "cc" * len(cols) + "}", "\\toprule",
                 " & " + " & ".join("\\multicolumn{2}{c}{" + lab(a, th) + "}" for a, th in cols) + "\\\\",
                 "".join(f"\\cmidrule(lr){{{2+2*j}-{3+2*j}}}" for j in range(len(cols))),
                 "Method" + " & gap & $\\norm{\\nabla F}^2$" * len(cols) + "\\\\", "\\midrule"]
        for m in ms:
            row = NAMES[m]
            for a, th in cols:
                g = get(s, a, th, part, m, "floor_gap"); q = get(s, a, th, part, m, "floor_gnorm2")
                row += " & -- & --" if g is None else f" & {1e3*g[0]:.1f} & {1e3*q[0]:.2f}"
            lines.append(row + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_aistats", "tables"); os.makedirs(_out, exist_ok=True)
        open(os.path.join(_out, f"e3_{s}.tex"), "w").write("\n".join(lines) + "\n")
if __name__ == "__main__":
    tex() if "--tex" in sys.argv else show()
