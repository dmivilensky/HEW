"""Delete runs whose theta = lr * lambda_max estimate failed (they fell back to the online theta).
Usage: python scripts/clean_bad_lmax.py [results/torch]   (then re-run ./scripts/run_torch_mac.sh)"""
import sys, glob, json, os
root = sys.argv[1] if len(sys.argv) > 1 else "results/torch"
for f in glob.glob(os.path.join(root, "*", "*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    lm = d.get("lmax")
    uses_lmax = os.path.basename(f).startswith(("hew_lm_", "scaffold_hew_"))
    if uses_lmax and (not lm or any(v is None for _, v in lm)):
        os.remove(f); print("removed", f)
