"""Step 10 -- summarise the L1-only generator ablation (paired by seed).

Reads results/expA_<arm>_b<N>_s<seed>_stepmatched.json for the arms
none / classical / pix2pix / l1only, uses the seeds present in ALL arms being
compared, and prints paired differences with bootstrap CIs.

    python scripts/10_ablation_summary.py
    python scripts/10_ablation_summary.py --budgets 100 200 --seeds 0 1 2 3 4

Equivalence margin (default 0.01 mIoU): if the whole 95% CI of
(pix2pix - l1only) lies inside +/- margin, the discriminator is not adding
measurable value. If the CI lies above +margin, the adversarial term helps.
Anything else is inconclusive at this number of seeds.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoseg import config as C

ARMS = ["none", "classical", "l1only", "pix2pix"]
PAIRS = [("pix2pix", "l1only"), ("l1only", "classical"), ("l1only", "none"),
         ("pix2pix", "classical"), ("pix2pix", "none")]


def load(arm, b, s):
    f = C.RESULTS_DIR / f"expA_{arm}_b{b}_s{s}_stepmatched.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    return {"mIoU": r["test_test1"]["mean_iou"], "BF1": r["test_test1"]["boundary_f1"],
            "ECE": r["expB"]["ece_mc"]}


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()

    rows = []
    for b in args.budgets:
        seeds = [s for s in args.seeds if all(load(a, b, s) for a in ARMS)]
        if not seeds:
            print(f"N={b}: no seed has all four arms yet"); continue
        print(f"\n=== N={b}  seeds used: {seeds} ===")
        print("means:  " + "   ".join(
            f"{a}: mIoU {np.mean([load(a,b,s)['mIoU'] for s in seeds]):.4f} "
            f"ECE {np.mean([load(a,b,s)['ECE'] for s in seeds]):.4f}" for a in ARMS))
        for A, B in PAIRS:
            for m in ("mIoU", "BF1", "ECE"):
                d = np.array([load(A, b, s)[m] - load(B, b, s)[m] for s in seeds])
                lo, hi = boot(d)
                better = (d < 0).sum() if m == "ECE" else (d > 0).sum()
                rows.append({"N": b, "A": A, "B": B, "metric": m, "n_seeds": len(d),
                             "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "A_better_in": f"{better}/{len(d)}"})
                tag = ""
                if (A, B) == ("pix2pix", "l1only") and m == "mIoU":
                    if lo > args.margin: tag = "  -> adversarial term helps"
                    elif lo > -args.margin and hi < args.margin: tag = "  -> EQUIVALENT within margin"
                    elif hi < -args.margin: tag = "  -> L1-only is better"
                    else: tag = "  -> inconclusive"
                print(f"  {A:9s}-{B:9s} {m:5s} {d.mean():+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
                      f"A better {better}/{len(d)}{tag}")
    out = C.RESULTS_DIR / "ablation_l1only_summary.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\nCSV ->", out)


if __name__ == "__main__":
    main()
