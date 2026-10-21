"""Step 1 -- Figure 2: class imbalance in F3.

    python scripts/01_fig2_class_imbalance.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 400, "figure.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    C.ensure_dirs()
    counts = np.load(C.PATCH_CACHE / "class_counts_train.npy")
    pct = 100 * counts / counts.sum()
    order = np.argsort(-pct)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    names = [C.CLASS_NAMES[i] for i in order]
    vals = pct[order]
    cols = [C.CLASS_COLORS[i] for i in order]
    bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Share of annotated pixels (%)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Facies class imbalance in the F3 Netherlands training volume")

    imb = vals.max() / vals.min()
    ax.annotate(f"imbalance ratio {imb:.0f}:1",
                xy=(len(vals) - 1.4, max(vals) * 0.72),
                fontsize=8.5, style="italic", color="#444444")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURE_DIR / f"fig02_class_imbalance.{ext}", bbox_inches="tight")
    print("wrote", C.FIGURE_DIR / "fig02_class_imbalance.png")
    print(f"imbalance ratio = {imb:.1f}:1  "
          f"(largest {names[0]} {vals[0]:.2f}%, smallest {names[-1]} {vals[-1]:.2f}%)")


if __name__ == "__main__":
    main()
