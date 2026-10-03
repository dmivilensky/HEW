"""Shared figure style (static paper figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots in fixed order; color follows the method, never its rank
COLORS = {
    "gls": "#2a78d6", "relmax": "#2a78d6", "re_lmax_oracle": "#2a78d6", "p_log": "#eb6834",
    "re": "#2a78d6", "re_oracle": "#2a78d6", "gls_oracle": "#2a78d6", "re_opt": "#2a78d6",
    "plain_re_opt": "#2a78d6",
    "p": "#eb6834", "plain_p": "#eb6834",
    "iv": "#1baf7a", "plain_iv": "#1baf7a",
    "work": "#eda100", "plain_work": "#eda100",
    "posthoc_hew": "#e87ba4",
    "fedadp": "#008300",
    "scaffold": "#4a3aa7", "scaffold_p": "#4a3aa7", "scaffold_iv": "#4a3aa7", "scaffold_re": "#4a3aa7",
    "scaffold_work": "#4a3aa7",
    "fedavg": "#e34948",
}
STYLE = {"re": ":", "relmax": "--", "re_lmax_oracle": "--", "re_oracle": "--", "gls_oracle": (0, (1, 1)), "scaffold_iv": "--", "scaffold_re": ":", "scaffold_work": "-."}
LABELS = {
    "gls": "GLS weights (ours, plug-in)", "relmax": "RE, $\\lambda_{\\max}$ plug-in",
    "re_lmax_oracle": "RE, $\\lambda_{\\max}$ oracle", "p_log": "objective weights $p$",
    "re": "RE, mean-eigenvalue plug-in", "re_oracle": "RE weights (oracle)", "gls_oracle": "GLS weights (oracle)",
    "re_opt": "RE weights (budget-optimal)", "plain_re_opt": "RE weights (budget-optimal)",
    "p": "objective weights $p$ (FedNova-type)", "plain_p": "objective weights $p$",
    "iv": "inverse-variance", "plain_iv": "inverse-variance",
    "work": "work-proportional ($\\propto H_ib_i$)", "plain_work": "work-proportional",
    "posthoc_hew": "post-hoc projection (old HEW)", "fedadp": "FedAdp (angle)",
    "scaffold": "SCAFFOLD", "scaffold_p": "SCAFFOLD, $w=p$", "scaffold_iv": "SCAFFOLD, inverse-var.",
    "scaffold_re": "SCAFFOLD, RE weights", "scaffold_work": "SCAFFOLD, work-prop.",
    "fedavg": "FedAvg (unnormalized)",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
    })


def line(ax, x, y, key, **kw):
    ax.plot(x, y, color=COLORS.get(key, INK), linestyle=STYLE.get(key, "-"),
            label=LABELS.get(key, key), **kw)
