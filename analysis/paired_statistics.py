"""Paired differences between training-data arms.

Reads the per-checkpoint metric tables written by scripts/11_uncertainty_quality.py
(<RESULTS_DIR>/expB_quality_full, expB_quality_test2) and writes
  paired_differences.csv     mean paired difference, 95% percentile bootstrap CI
  paired_percent_gains.csv   percentage gain (mean over seeds of the paired ratio minus 1)

Pairing is by seed (same label subset and initialisation in both arms); the bootstrap
resamples the seed-level differences 20,000 times.

    python analysis/paired_statistics.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoseg import config as C

R = C.RESULTS_DIR
QDIR = {"test1": "expB_quality_full", "test2": "expB_quality_test2"}   # folders written by 11_uncertainty_quality.py
Q = {t: pd.read_csv(R / QDIR[t] / "quality_per_checkpoint.csv") for t in ("test1", "test2")}
B = 20000

def boot_ci(d, rng):
    d = np.asarray(d)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), int((d > 0).sum()), len(d)

# 1. absolute paired differences, all metrics
rng = np.random.default_rng(0)
rows = []
for t in ("test1", "test2"):
    for meth in ("det", "mc"):
        d = Q[t][Q[t].method == meth]
        for N in (50, 100, 200):
            for a, b in [("pix2pix", "none"), ("pix2pix", "classical"), ("classical", "none"),
                         ("pix2pix", "l1only"), ("l1only", "classical")]:
                for met in ["mIoU", "pixel_acc", "ece", "nll", "aurc", "auroc_ent"] + (["auroc_mi"] if meth == "mc" else []):
                    A = d[(d.arm == a) & (d.N == N)].set_index("seed")[met]
                    Bm = d[(d.arm == b) & (d.N == N)].set_index("seed")[met]
                    m, lo, hi, pos, n = boot_ci((A - Bm).dropna().values, rng)
                    rows.append(dict(test=t, method=meth, N=N, A=a, B=b, metric=met, mean=m, lo=lo, hi=hi, pos=pos, n=n))
pd.DataFrame(rows).to_csv(R / "paired_differences.csv", index=False)

# 2. percentage gains in mIoU (deterministic networks)
rng = np.random.default_rng(7)
rows = []
for a, b in [("classical", "none"), ("pix2pix", "none"), ("pix2pix", "classical"), ("pix2pix", "l1only")]:
    for N in (50, 100, 200):
        for t in ("test1", "test2"):
            d = Q[t][Q[t].method == "det"]
            A = d[(d.arm == a) & (d.N == N)].set_index("seed").mIoU
            Bm = d[(d.arm == b) & (d.N == N)].set_index("seed").mIoU
            g = ((A / Bm - 1) * 100).values
            m = g[rng.integers(0, len(g), (B, len(g)))].mean(1)
            rows.append(dict(test=t, N=N, A=a, B=b, gain_pct=g.mean(), lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5)))
pd.DataFrame(rows).to_csv(R / "paired_percent_gains.csv", index=False)
print("wrote paired_differences.csv and paired_percent_gains.csv")
