"""Figures and LaTeX tables for the hew_torch experiments.

Usage: python experiments/plot_torch.py [results/torch] [--no-smooth]
Reads <root>/<suite>_<setting>/<method>_s<seed>.json (finished runs only) and writes to <root>/figures, for every
suite (= directory prefix, e.g. fmnist_mlp_hn):
  <suite>_curves.pdf/png    train loss (log) and test accuracy vs round, one column per data regime;
                            rounds >= 0.2 R, mean over seeds (3-point moving average), band = ± sd over seeds
  <suite>_relative.pdf/png  tail train loss of every method relative to w = p on the same seeds (paired);
                            dot = mean ratio, bar = min..max over seeds; < 1 is better than p
  <suite>_variants.pdf/png  the HEW variants (theta choice, sketch, ...) against w = p
  <suite>_weights.pdf/png   HEW diagnostics: TV(w_t, p), theta_t, learned weights vs noise level
  <suite>_table.tex         tail averages (last quarter of the evaluations), mean ± sd over seeds
Every method present in the results appears in at least one figure and in the table. Legends are drawn in their
own row above the panels, so they never cover data. Diverged runs are drawn up to their last evaluation and
marked with a cross; in the table they are flagged with a dagger.
"""
import os, sys, collections, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from summarize_torch import load_root, tail, methods_sorted

# ---------------------------------------------------------------- style
# categorical slots in fixed order (validated palette); colour follows the method, never its rank. Every method
# also has its own marker and line style (secondary encoding for colour-vision deficiency and grayscale print).
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SPEC = {  # method: (colour, line style, marker, label)
    "hew_lm":       (BLUE,    "-",            "o", "debiased HEW (ours)"),
    "p":            (ORANGE,  "-",            "s", "normalized, $w=p$"),
    "iv":           (AQUA,    "-",            "^", "inverse-variance"),
    "fedavg":       (YELLOW,  "-.",           "v", "FedAvg"),
    "scaffold":     (VIOLET,  "-",            "D", "SCAFFOLD"),
    "scaffold_hew": (GREEN,   "--",           "P", "SCAFFOLD + HEW (ours)"),
    "hew":          (MAGENTA, ":",            "X", "HEW, online $\\vartheta$"),
    "hew_t4":       (GREEN,   "--",           "d", "HEW, $\\vartheta=4$"),
    "hew_sk":       (VIOLET,  (0, (5, 1, 1, 1)), "*", "HEW, gradient sketch"),
    "hew_post":     (MAGENTA, "--",           "h", "HEW, post-local weights"),
    "hew_adapt":    (GREEN,   "-.",           "p", "HEW + SNR server step"),
    "p_adapt":      (RED,     ":",            "<", "$w=p$ + SNR server step"),
    "hew_sk_gls":   (RED,     "--",           ">", "HEW, sketch, GLS"),
    "hew_sk_path":  (YELLOW,  ":",            "8", "HEW, sketch, path"),
    "hew_adsk":     (RED,     "-.",           "H", "HEW + sketch SNR step"),
}
OURS = {"hew_lm", "scaffold_hew"}
MAIN_METHODS = ["fedavg", "p", "iv", "scaffold", "scaffold_hew", "hew_lm"]          # drawn last = on top
VARIANT_METHODS = ["p", "hew", "hew_t4", "hew_sk", "hew_sk_gls", "hew_sk_path", "hew_lm"]
SETTING_LABEL = {"iid": "IID", "a1": "Dirichlet $\\alpha=1$", "a03": "Dirichlet $\\alpha=0.3$",
                 "a01": "Dirichlet $\\alpha=0.1$"}
SETTING_ORDER = ["iid", "a1", "a03", "a01"]
SETTING_SPEC = {"iid": (BLUE, "o"), "a1": (AQUA, "^"), "a03": (ORANGE, "s"), "a01": (VIOLET, "D")}
SUITE_TITLE = {"fmnist_mlp_hn": "Fashion-MNIST, MLP, noise-heterogeneous systems ($b_i\\in\\{2,\\dots,256\\}$)",
               "fmnist_cnn_main": "Fashion-MNIST, CNN ($b_i\\in\\{8,\\dots,128\\}$)",
               "fmnist_mlp_pp": "Fashion-MNIST, MLP, 50 clients, 30% participation",
               "fmnist_mlp_strag": "Fashion-MNIST, MLP, stragglers",
               "fmnist_mlp_abl": "Fashion-MNIST, MLP, ablations",
               "c10_r18": "CIFAR-10, ResNet-18, 20% participation"}
SMOOTH = "--no-smooth" not in sys.argv
ZOOM = 0.2          # curves show rounds >= ZOOM * R (the initial transient is common to all methods)


def spec(m):
    return SPEC.get(m, (INK, "-", ".", m))


def label(m):
    return spec(m)[3]


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.4, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 220, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlepad": 4, "mathtext.fontset": "dejavusans",
    })


# ---------------------------------------------------------------- data helpers
def split(name):
    parts = name.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (name, "")


def ordered_settings(names):
    return sorted(names, key=lambda n: SETTING_ORDER.index(split(n)[1]) if split(n)[1] in SETTING_ORDER else 99)


def smooth(y, k=3):
    if not SMOOTH or len(y) < k:
        return y
    out = y.copy(); h = k // 2
    for i in range(len(y)):
        w = y[max(0, i - h): i + h + 1]; w = w[np.isfinite(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def curve(runs, key):
    """Mean and sd over seeds on the union of logged rounds. Seeds that stopped early (divergence) contribute
    up to their last evaluation, so a method is never dropped because one seed is shorter."""
    L = max(len(d["round"]) for d in runs)
    r = np.array(max(runs, key=lambda d: len(d["round"]))["round"], float)
    Y = np.full((len(runs), L), np.nan)
    for i, d in enumerate(runs):
        v = np.array([np.nan if x is None else x for x in d[key]], float); Y[i, :len(v)] = v
    n = np.sum(np.isfinite(Y), 0)
    mu = np.nanmean(Y, 0); sd = np.where(n > 1, np.nanstd(Y, 0, ddof=1), 0.0)
    return r, smooth(mu), smooth(sd)


def diverged(runs):
    return [d for d in runs if d.get("diverged") is not None]


def draw(ax, runs, key, m, band=True, every=None):
    col, ls, mk, _ = spec(m)
    r, mu, sd = curve(runs, key)
    ok = np.isfinite(mu)
    if not ok.any():
        return None
    ours = m in OURS
    every = every or max(1, ok.sum() // 7)
    ax.plot(r[ok], mu[ok], color=col, ls=ls, lw=2.0 if ours else 1.25, marker=mk, markersize=3.6 if ours else 3.0,
            markevery=every, markerfacecolor=col if ours else "white", markeredgewidth=0.9,
            zorder=5 if ours else 3, solid_capstyle="round")
    if band and len(runs) > 1:
        ax.fill_between(r[ok], (mu - sd)[ok], (mu + sd)[ok], color=col, alpha=0.10, lw=0, zorder=1)
    for d in diverged(runs):                       # mark where a seed diverged
        ax.plot(d["round"][-1], mu[min(len(d["round"]), len(mu)) - 1], marker="x", color=col, ms=6, mew=1.4, zorder=6)
    return r[ok], mu[ok]


def handle(m):
    col, ls, mk, lab = spec(m); ours = m in OURS
    return Line2D([], [], color=col, ls=ls, lw=2.0 if ours else 1.25, marker=mk, ms=4 if ours else 3.4,
                  markerfacecolor=col if ours else "white", markeredgewidth=0.9, label=lab)


def top_legend(fig, handles, ncol=None, y=1.0):
    ncol = ncol or min(len(handles), 4)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
               handlelength=2.6, columnspacing=1.4, borderaxespad=0.2)


def robust_limits(series, log=False, pad=0.06, start_frac=0.0):
    """Limits that contain every drawn line from start_frac of the rounds on (nothing is clipped there)."""
    vals = []
    for r, y in series:
        if len(r) == 0:
            continue
        cut = r[0] + start_frac * (r[-1] - r[0]); v = y[r >= cut]; v = v[np.isfinite(v)]
        if log:
            v = v[v > 0]
        vals.append(v)
    v = np.concatenate(vals) if vals else np.array([])
    if len(v) == 0:
        return None
    lo, hi = v.min(), v.max()
    if log:
        lo, hi = np.log10(lo), np.log10(hi); d = max(hi - lo, 0.05)
        return 10 ** (lo - pad * d), 10 ** (hi + pad * d)
    d = max(hi - lo, 1e-3)
    return lo - pad * d, hi + pad * d


def log_ticks(ax):
    """Readable labels on a log axis that spans less than a decade (1-2-3-5 steps, plain numbers)."""
    lo, hi = ax.get_ylim()
    subs = (1.0, 2.0, 5.0) if hi / lo > 8 else (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) if hi / lo < 2.5 else (1.0, 2.0, 3.0, 5.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs, numticks=20))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------- figures
def fig_curves(pre, settings, data, out, methods, name):
    present = [m for m in methods if any(m in data[s] for s in settings)]
    if not present:
        return False
    ncol = len(settings)
    single = ncol == 1
    if single:   # one data regime: one full-width row (loss | accuracy | paired ratio to p), never a lone column
        fig, a3 = plt.subplots(1, 3, figsize=(6.9, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
        ax = np.array([[a3[0]], [a3[1]]])
    else:
        fig, ax = plt.subplots(2, ncol, figsize=(2.3 * ncol, 3.9), squeeze=False)
    for j, s in enumerate(settings):
        loss_series, acc_series = [], []
        for m in present:
            if m not in data[s]:
                continue
            runs = list(data[s][m].values())
            a = draw(ax[0, j], runs, "train_loss", m); b = draw(ax[1, j], runs, "test_acc", m)
            if a: loss_series.append(a)
            if b: acc_series.append(b)
        n_seeds = max(len(v) for v in data[s].values())
        stitle = f"{SETTING_LABEL.get(split(s)[1], s)}  ({n_seeds} seed{'s' if n_seeds > 1 else ''})"
        ax[0, j].set_title(stitle)
        ax[0, j].set_yscale("log")
        # both rows zoom on rounds >= ZOOM * R, where the methods separate; the limits contain every line there
        for i, ser, lg in ((0, loss_series, True), (1, acc_series, False)):
            lim = robust_limits(ser, log=lg, start_frac=ZOOM)
            if lim: ax[i, j].set_ylim(*lim)
            if ser:
                r0 = max((a for a, _ in ser), key=len)
                ax[i, j].set_xlim(r0[0] + ZOOM * (r0[-1] - r0[0]), r0[-1] + 0.01 * (r0[-1] - r0[0]))
        ax[1, j].set_xlabel("round")
        if single:
            ax[0, j].set_xlabel("round"); ax[0, j].set_title("train loss"); ax[1, j].set_title("test accuracy")
        log_ticks(ax[0, j])
        ax[1, j].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0, 0].set_ylabel("train loss"); ax[1, 0].set_ylabel("test accuracy")
    if single:
        rows, rms = relative_rows(settings, data, [m for m in present if m != "p"])
        if rows:
            draw_relative(a3[2], rows, rms, settings, "tail loss / tail loss of $w=p$")
            a3[2].set_yticklabels([])      # rows are identified by colour + marker, as in the legend
            a3[2].tick_params(axis="y", length=0)
            a3[2].set_title("paired ratio to $w=p$ ($<1$ better)")
        else:
            a3[2].axis("off")
        a3[0].set_ylabel("train loss"); a3[1].set_ylabel("test accuracy")
        fig.text(0.5, -0.02, stitle, ha="center", va="top", fontsize=8, color=MUTED)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    top_legend(fig, [handle(m) for m in present], ncol=min(len(present), 3 if ncol == 1 else 6 if ncol >= 3 else 4))
    save(fig, out, name)
    return True


def relative_rows(settings, data, ms):
    """(method, setting, mean, min, max, n) of the paired tail train-loss ratio to w = p."""
    rows = []
    for m in ms:
        for s in settings:
            if m in data[s] and "p" in data[s]:
                seeds = sorted(set(data[s][m]) & set(data[s]["p"]))
                rr = [tail(data[s][m][k], "train_loss") / tail(data[s]["p"][k], "train_loss") for k in seeds
                      if tail(data[s][m][k], "train_loss") and tail(data[s]["p"][k], "train_loss")]
                if rr:
                    rows.append((m, s, float(np.mean(rr)), float(np.min(rr)), float(np.max(rr)), len(rr)))
    return rows, [m for m in ms if any(r[0] == m for r in rows)]


def draw_relative(ax, rows, ms, settings, xlabel):
    off = np.linspace(-0.27, 0.27, len(settings)) if len(settings) > 1 else [0.0]
    for (m, s, mu, lo, hi, k) in rows:
        y = ms.index(m) + off[settings.index(s)]
        col, mk = SETTING_SPEC.get(split(s)[1], (INK, "o")) if len(settings) > 1 else (spec(m)[0], spec(m)[2])
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.8, solid_capstyle="round", zorder=2)
        ax.plot(mu, y, marker=mk, color=col, ms=4.5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.axvline(1.0, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([label(m) for m in ms]); ax.set_ylim(len(ms) - 0.5, -0.5)
    ax.set_xscale("log"); ax.grid(axis="y", visible=False)
    vals = np.array([v for r in rows for v in r[3:5]])
    lo, hi = min(vals.min(), 0.95), max(vals.max(), 1.05); d = np.log10(hi / lo)
    ax.set_xlim(lo / 10 ** (0.08 * d), hi * 10 ** (0.08 * d))
    cand = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 5, 10]
    lo_, hi_ = ax.get_xlim(); ticks = [t for t in cand if lo_ <= t <= hi_]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(xlabel)


def fig_relative(pre, settings, data, out):
    """Paired tail train-loss ratio to w = p, per method and data regime (one axis, reference line at 1)."""
    ms = [m for m in methods_sorted(set().union(*[data[s].keys() for s in settings])) if m != "p"]
    rows, ms = relative_rows(settings, data, ms)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2 + 0.3 * len(settings), 0.34 * len(ms) + 0.9))
    draw_relative(ax, rows, ms, settings, "tail train loss / tail train loss of $w=p$  (paired seeds; $<1$ is better)")
    fig.tight_layout()
    if len(settings) > 1:
        handles = [Line2D([], [], color=SETTING_SPEC[split(s)[1]][0], marker=SETTING_SPEC[split(s)[1]][1], ls="-", lw=1.2,
                          ms=4.5, label=SETTING_LABEL[split(s)[1]]) for s in settings if split(s)[1] in SETTING_SPEC]
        top_legend(fig, handles, ncol=len(handles))
    else:
        ax.set_title(SETTING_LABEL.get(split(settings[0])[1], settings[0]))
    save(fig, out, f"{pre}_relative")


def fig_weights(pre, settings, data, out):
    main = next((m for m in ["hew_lm", "hew", "hew_t4", "hew_sk"] if any(m in data[s] for s in settings)), None)
    if main is None:
        return
    hs = [s for s in settings if main in data[s]]
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.35))
    for s in hs:
        col, mk = SETTING_SPEC.get(split(s)[1], (INK, "o"))
        runs = list(data[s][main].values())
        for key, a in (("tv_wp", ax[0]), ("theta", ax[1])):
            r, mu, sd = curve(runs, key); ok = np.isfinite(mu)
            if ok.any():
                a.plot(r[ok], mu[ok], color=col, marker=mk, ms=3, markevery=max(1, ok.sum() // 7),
                       markerfacecolor="white", markeredgewidth=0.9, lw=1.4)
                if len(runs) > 1:
                    a.fill_between(r[ok], (mu - sd)[ok], (mu + sd)[ok], color=col, alpha=0.10, lw=0)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("TV$(w_t,\\,p)$"); ax[0].set_title("distance of the weights from $p$")
    ax[0].set_ylim(bottom=0)
    ax[1].set_xlabel("round"); ax[1].set_ylabel("$\\vartheta_t$"); ax[1].set_yscale("log")
    ax[1].set_title("variance weight $\\vartheta_t$")
    # learned weights vs noise at the last evaluation (lowest seed), IID and the most heterogeneous regime
    shown = [hs[0], hs[-1]] if len(hs) > 1 else hs
    allx = []
    for s in shown:
        col, mk = SETTING_SPEC.get(split(s)[1], (INK, "o"))
        d = data[s][main][min(data[s][main])]
        w = np.array([np.nan if v is None else v for v in d["w"][-1]], float) if d["w"][-1] else np.array([])
        s2 = np.array([np.nan if v is None else v for v in d["s2"][-1]], float)
        if len(w) != len(s2):
            continue
        ok = np.isfinite(w) & np.isfinite(s2) & (w > 0) & (s2 > 0)
        ax[2].scatter(s2[ok], ok.sum() * w[ok] / w[ok].sum(), marker=mk, s=16, facecolor=col, edgecolor="white",
                      linewidth=0.5, zorder=3)
        allx.append(s2[ok])
    if allx:
        x = np.concatenate(allx); g = np.geomspace(x.min(), x.max(), 50)
        hm = 1 / np.mean(1 / allx[0]) if len(allx[0]) else 1
        ax[2].plot(g, hm / g, color=MUTED, lw=0.9, ls="--", zorder=2)      # inverse-variance: n w_i = HM(s^2)/s_i^2
        ax[2].axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=2)
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("noise level $\\hat s_i^2$")
    ax[2].set_ylabel("$n\\,w_i$"); ax[2].set_title("learned weights (last round)")
    fig.tight_layout(w_pad=1.0)
    handles = [Line2D([], [], color=SETTING_SPEC[split(s)[1]][0], marker=SETTING_SPEC[split(s)[1]][1], lw=1.4, ms=3.6,
                      markerfacecolor="white", markeredgewidth=0.9, label=SETTING_LABEL[split(s)[1]])
               for s in hs if split(s)[1] in SETTING_SPEC]
    handles += [Line2D([], [], color=MUTED, ls="--", lw=0.9, label="inverse-variance"),
                Line2D([], [], color=MUTED, ls=":", lw=0.8, label="$w=p$ (uniform $p$)")]
    top_legend(fig, handles, ncol=len(handles))
    save(fig, out, f"{pre}_weights")


def table(pre, settings, data, out):
    ms = methods_sorted(set().union(*[data[s].keys() for s in settings]))

    def stat(s, m, key):
        if m not in data[s]:
            return None
        v = np.array([tail(d, key) for d in data[s][m].values() if tail(d, key) is not None], float)
        if not len(v):
            return None
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(diverged(list(data[s][m].values())))
    best = {}
    for s in settings:
        tl = {m: stat(s, m, "train_loss") for m in ms}; ta = {m: stat(s, m, "test_acc") for m in ms}
        tl = {m: v for m, v in tl.items() if v and v[2] == 0}; ta = {m: v for m, v in ta.items() if v and v[2] == 0}
        best[s] = (min(tl, key=lambda m: tl[m][0]) if tl else None, max(ta, key=lambda m: ta[m][0]) if ta else None)
    L = ["\\begin{tabular}{l" + "cc" * len(settings) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SETTING_LABEL.get(split(s)[1], s)}}}" for s in settings) + "\\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(settings))),
         "Aggregation & " + " & ".join(["loss & acc.\\ (\\%)"] * len(settings)) + "\\\\", "\\midrule"]
    for m in ms:
        cells = []
        for s in settings:
            a, b = stat(s, m, "train_loss"), stat(s, m, "test_acc")
            dag = "$^\\dagger$" if a and a[2] else ""
            fa = "--" if a is None else f"{a[0]:.3f}$\\pm${a[1]:.3f}{dag}"
            fb = "--" if b is None else f"{100 * b[0]:.1f}$\\pm${100 * b[1]:.1f}"
            if best[s][0] == m: fa = f"\\textbf{{{fa}}}"
            if best[s][1] == m: fb = f"\\textbf{{{fb}}}"
            cells += [fa, fb]
        L.append(label(m).replace(" (ours)", "") + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "% tail = mean over the last quarter of the evaluations; mean $\\pm$ sd over seeds; "
          "bold = best among runs without divergence; $^\\dagger$ = at least one seed diverged"]
    open(os.path.join(out, f"{pre}_table.tex"), "w").write("\n".join(L) + "\n")


def main(root):
    data = load_root(root)
    if not data:
        print(f"no finished runs under {root}"); return
    out = os.path.join(root, "figures"); os.makedirs(out, exist_ok=True)
    setup()
    groups = collections.defaultdict(list)
    for name in data:
        groups[split(name)[0]].append(name)
    for pre, names in sorted(groups.items()):
        settings = ordered_settings(names)
        present = set().union(*[data[s].keys() for s in settings])
        made = []
        other = [m for m in methods_sorted(present) if m not in MAIN_METHODS and m not in VARIANT_METHODS]
        main_ms = [m for m in MAIN_METHODS if m in present]
        # main figure (methods that belong to no other figure, e.g. ablations, are added so that none is lost);
        # our method is drawn last, i.e. on top
        curve_ms = [m for m in main_ms if m != "hew_lm"] + other + (["hew_lm"] if "hew_lm" in present else [])
        if curve_ms and fig_curves(pre, settings, data, out, curve_ms, f"{pre}_curves"):
            made.append("curves")
        variants = [m for m in VARIANT_METHODS if m in present]
        if any(m not in ("p", "hew_lm") for m in variants):
            if fig_curves(pre, settings, data, out, variants, f"{pre}_variants"):
                made.append("variants")
        fig_relative(pre, settings, data, out); made.append("relative")
        fig_weights(pre, settings, data, out); made.append("weights")
        table(pre, settings, data, out); made.append("table")
        missing = [m for m in present if m not in MAIN_METHODS + VARIANT_METHODS + other]
        assert not missing, missing
        print(f"{pre}: {', '.join(made)} -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "results/torch")
