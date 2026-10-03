"""Summarize hew_torch logs: tail averages (last quarter of the evaluations), mean ± sd over seeds.

Usage: python experiments/summarize_torch.py [results/torch]
Layout: <root>/<setting>/<method>_s<seed>.json. Prints one table per setting and paired comparisons
(same seed) of the HEW variants against their fixed-weight counterparts.
"""
import sys, os, json, glob, re, collections
import numpy as np

ORDER = ["fedavg", "p", "iv", "hew_lm", "hew", "hew_t4", "hew_sk", "hew_post", "hew_adapt", "p_adapt", "scaffold", "scaffold_hew"]
PAIRS = [("hew_lm", "p"), ("hew_lm", "iv"), ("hew", "p"), ("hew_t4", "p"), ("hew_sk", "p"), ("iv", "p"),
         ("scaffold_hew", "scaffold"), ("hew_post", "hew_lm"), ("hew_adapt", "hew_lm"), ("p_adapt", "p")]


def load_root(root):
    data = collections.defaultdict(lambda: collections.defaultdict(dict))   # setting -> method -> seed -> log
    for f in glob.glob(os.path.join(root, "*", "*.json")):
        mt = re.match(r"(.+)_s(\d+)\.json$", os.path.basename(f))
        if not mt:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("finished"):
            continue
        data[os.path.basename(os.path.dirname(f))][mt.group(1)][int(mt.group(2))] = d
    return data


def tail(d, key, frac=0.25):
    if "tail" in d and key in d["tail"] and frac == 0.25:
        return d["tail"][key]
    v = [x for x in d[key] if x is not None]; k = max(1, int(round(frac * len(v))))
    return float(np.mean(v[-k:])) if v else None


def methods_sorted(ms):
    return sorted(ms, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))


def main(root):
    data = load_root(root)
    if not data:
        print(f"no finished runs under {root}"); return
    for setting in sorted(data):
        print(f"\n== {setting} ==  (tail = last quarter of the evaluations; mean ± sd over seeds)")
        print(f"{'method':14s} {'train loss':>16s} {'test acc':>16s} {'|grad|^2':>18s} {'TV(w,p)':>8s} {'theta':>7s} seeds div")
        for m in methods_sorted(data[setting]):
            runs = data[setting][m]
            def col(key):
                v = np.array([tail(d, key) for d in runs.values() if tail(d, key) is not None], float)
                return v
            tl, ta, g2, tv, th = col("train_loss"), col("test_acc"), col("gnorm2"), col("tv_wp"), col("theta")
            sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
            ndiv = sum(1 for d in runs.values() if d.get("diverged") is not None)
            print(f"{m:14s} {tl.mean():.4f}±{sd(tl):.4f}  {ta.mean():.4f}±{sd(ta):.4f}  {g2.mean():.3e}±{sd(g2):.1e} "
                  f"{(tv.mean() if len(tv) else float('nan')):8.3f} {(th.mean() if len(th) else float('nan')):7.2f} "
                  f"{len(runs):5d} {ndiv:3d}")
        for a, b in PAIRS:
            if a in data[setting] and b in data[setting]:
                seeds = sorted(set(data[setting][a]) & set(data[setting][b]))
                if not seeds:
                    continue
                r = [tail(data[setting][a][s], "train_loss") / tail(data[setting][b][s], "train_loss") for s in seeds]
                acc = [tail(data[setting][a][s], "test_acc") - tail(data[setting][b][s], "test_acc") for s in seeds]
                wins = sum(x < 1 for x in r)
                print(f"  {a} vs {b}: train-loss ratio {np.mean(r):.3f} (lower in {wins}/{len(seeds)} seeds), "
                      f"test-acc diff {100 * np.mean(acc):+.2f} pp")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/torch")
