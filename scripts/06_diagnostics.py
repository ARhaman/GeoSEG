"""Step 6 -- diagnostics that turn the raw numbers into diagnostics.

Costs no GPU time; reads only what steps 03 and 04 already wrote.

    python scripts\\06_diagnostics.py

Produces:
  * per-class epistemic uncertainty vs class frequency  -> Figure 8b
  * matched-training-set-size synthetic-vs-real table   -> Figure 4b
  * the exchange rate: what one synthetic patch is worth in real patches
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C

plt.rcParams.update({"font.size": 9, "savefig.dpi": 400,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------
def per_class_uncertainty():
    """Does uncertainty track class rarity rather than boundary geometry?"""
    cands = sorted(C.RESULTS_DIR.glob("expB_maps_*.npz"))
    if not cands:
        print("[diag] no Experiment B maps found -- run step 04 first")
        return None
    d = np.load(cands[0])
    Y, epi, ale = d["Y"], d["epistemic"], d["aleatoric"]

    counts = np.load(C.PATCH_CACHE / "class_counts_train.npy")
    freq = 100 * counts / counts.sum()

    rows = []
    for c in range(C.N_CLASSES):
        m = (Y == c)
        if m.sum() == 0:
            continue
        rows.append({"cls": c, "name": C.CLASS_NAMES[c], "freq": float(freq[c]),
                     "epistemic": float(epi[m].mean()),
                     "aleatoric": float(ale[m].mean()),
                     "n_px": int(m.sum())})

    print("\n--- Uncertainty by facies class ---")
    print(f"{'class':<18}{'train %':>9}{'epistemic':>12}{'aleatoric':>12}")
    for r in rows:
        print(f"{r['name']:<18}{r['freq']:>9.2f}{r['epistemic']:>12.4f}"
              f"{r['aleatoric']:>12.4f}")

    f = np.array([r["freq"] for r in rows])
    e = np.array([r["epistemic"] for r in rows])
    # Spearman rank correlation without a scipy.stats import
    rf, re_ = np.argsort(np.argsort(f)), np.argsort(np.argsort(e))
    rho = float(np.corrcoef(rf, re_)[0, 1])
    print(f"\nSpearman rho (class frequency vs epistemic uncertainty) = {rho:+.3f}")
    if rho < -0.5:
        print("  -> strongly negative: rarer facies carry higher model uncertainty.")
        print("     Uncertainty tracks class imbalance, not boundary geometry.")
    elif rho > 0.5:
        print("  -> positive: common facies are the uncertain ones (unexpected).")
    else:
        print("  -> no clear monotonic relationship.")

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.scatter(f, e, s=70, c=[C.CLASS_COLORS[r["cls"]] for r in rows],
               edgecolor="white", zorder=3)
    for r in rows:
        ax.annotate(r["name"], (r["freq"], r["epistemic"]), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Share of annotated pixels in training volume (%)")
    ax.set_ylabel("Mean epistemic uncertainty (nats)")
    ax.set_title(f"Model uncertainty concentrates in rare facies (ρ = {rho:+.2f})")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURE_DIR / f"fig08b_uncertainty_vs_frequency.{ext}",
                    bbox_inches="tight")
    print("wrote fig08b_uncertainty_vs_frequency")
    return rows


# --------------------------------------------------------------------------
def matched_size_comparison():
    """What is a synthetic patch worth, measured in real patches?

    pix2pix@N trains on N real + N synthetic = 2N patches.
    none@2N   trains on 2N real patches.
    Same training-set size; the only difference is real vs synthetic.
    """
    rows = []
    for p in sorted(C.RESULTS_DIR.glob("expA_*.json")):
        r = json.loads(p.read_text())
        t = r.get("test_test1") or r.get("test_test2")
        if t:
            rows.append((r["arm"], r["budget"], t["mean_iou"]))

    def mean_of(arm, budget):
        v = [m for a, b, m in rows if a == arm and b == budget]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else None

    print("\n--- Matched training-set size: synthetic vs real ---")
    print(f"{'total patches':>14}{'pix2pix@N':>12}{'none@2N':>12}{'delta':>9}")
    pairs = [(50, 100), (100, 200), (200, 500)]   # 500 is the nearest available
    out = []
    for n, n2 in pairs:
        a, b = mean_of("pix2pix", n), mean_of("none", n2)
        if not a or not b:
            continue
        note = "" if n2 == 2 * n else f"  (none@{n2}, not 2N)"
        print(f"{2*n:>14}{a[0]:>12.4f}{b[0]:>12.4f}{a[0]-b[0]:>+9.4f}{note}")
        out.append((n, a[0], b[0], a[0] - b[0]))

    # exchange rate from the real-only learning curve
    curve = [(b, mean_of("none", b)[0]) for b in C.LABEL_BUDGETS if mean_of("none", b)]
    if len(curve) >= 2:
        xs = np.log2([c[0] for c in curve])
        ys = np.array([c[1] for c in curve])
        slope = float(np.polyfit(xs, ys, 1)[0])
        print(f"\nReal-label learning curve: {slope:+.4f} mIoU per doubling of labels")
        for n, p2p, non, delta in out:
            if slope > 1e-6:
                eq = (2 ** (delta / slope)) * n
                print(f"  at N={n}: {n} synthetic patches were worth "
                      f"~{eq - n:+.0f} real patches")
    return out


# --------------------------------------------------------------------------
def significance_note():
    """Blunt reminder of what 3 seeds can and cannot support."""
    rows = {}
    for p in sorted(C.RESULTS_DIR.glob("expA_*.json")):
        r = json.loads(p.read_text())
        t = r.get("test_test1") or r.get("test_test2")
        if t:
            rows.setdefault((r["arm"], r["budget"]), []).append(t["mean_iou"])

    print("\n--- Effect sizes vs seed noise (n=3) ---")
    print(f"{'budget':>7}{'best arm':>14}{'gain over none':>16}{'pooled SD':>11}{'ratio':>8}")
    for b in C.LABEL_BUDGETS:
        base = rows.get(("none", b))
        if not base:
            continue
        best_arm, best_mu = None, -1
        for arm in ("classical", "pix2pix"):
            v = rows.get((arm, b))
            if v and np.mean(v) > best_mu:
                best_arm, best_mu = arm, float(np.mean(v))
        if best_arm is None:
            continue
        gain = best_mu - float(np.mean(base))
        sd = float(np.sqrt((np.var(base) + np.var(rows[(best_arm, b)])) / 2))
        ratio = gain / sd if sd > 0 else float("nan")
        flag = "" if abs(ratio) > 2 else "   <- within noise"
        print(f"{b:>7}{best_arm:>14}{gain:>+16.4f}{sd:>11.4f}{ratio:>8.2f}{flag}")
    print("\nWith 3 seeds, treat |gain| < 2 SD as suggestive only, not significant.")


def main():
    C.ensure_dirs()
    per_class_uncertainty()
    matched_size_comparison()
    significance_note()


if __name__ == "__main__":
    main()
