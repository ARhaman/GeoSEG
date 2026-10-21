"""Step 5 -- assemble Figures 4, 5, 8, 9 and the LaTeX/Markdown results table.

    python scripts/05_make_figures.py

Reads only the JSON/NPZ written by steps 3 and 4, so it is safe to re-run at
any time and cheap to iterate on for figure polish.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "savefig.dpi": 400, "figure.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARM_STYLE = {
    "none":      ("No augmentation",        "#7A7A7A", "--", "o"),
    "classical": ("Classical augmentation", "#0072B2", "-",  "s"),
    "pix2pix":   ("Pix2Pix augmentation",   "#E69F00", "-",  "^"),
    "ldm":       ("Diffusion augmentation", "#D55E00", "-",  "D"),
}
FACIES_CMAP = ListedColormap(C.CLASS_COLORS)


def load_expA():
    rows = []
    for p in sorted(C.RESULTS_DIR.glob("expA_*.json")):
        r = json.loads(p.read_text())
        test = r.get("test_test1") or r.get("test_test2")
        if test is None:
            continue
        rows.append({"arm": r["arm"], "budget": r["budget"], "seed": r["seed"],
                     "miou": test["mean_iou"], "dice": test["mean_dice"],
                     "bf1": test["boundary_f1"], "fwiou": test["freq_weighted_iou"]})
    return rows


def agg(rows, key="miou"):
    out = {}
    for r in rows:
        out.setdefault(r["arm"], {}).setdefault(r["budget"], []).append(r[key])
    return out


# --------------------------------------------------------------------------
def fig4_learning_curve(rows):
    if not rows:
        print("[fig4] no Experiment A results yet -- skipping")
        return
    a = agg(rows, "miou")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for arm in ["none", "classical", "pix2pix", "ldm"]:
        if arm not in a:
            continue
        label, col, ls, mk = ARM_STYLE[arm]
        bs = sorted(a[arm])
        mu = np.array([np.mean(a[arm][b]) for b in bs])
        sd = np.array([np.std(a[arm][b]) for b in bs])
        ax.plot(bs, mu, ls, color=col, marker=mk, ms=5, lw=1.6, label=label)
        ax.fill_between(bs, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)

    ax.set_xscale("log")
    ax.set_xticks(C.LABEL_BUDGETS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of labelled 2D patches")
    ax.set_ylabel("Mean IoU on F3 test set")
    ax.set_title("Effect of augmentation strategy under a shrinking label budget")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURE_DIR / f"fig04_learning_curve.{ext}", bbox_inches="tight")
    print("wrote fig04_learning_curve")


def fig5_synthetic_grid(budget=None, seed=0, n_cols=4):
    tr = C.PATCH_CACHE / "train_patches.npz"
    if budget is not None:
        p = C.SYNTH_DIR / f"pix2pix_b{budget}_s{seed}.npz"
    else:  # use the largest budget available
        cands = sorted(C.SYNTH_DIR.glob("pix2pix_b*.npz"),
                       key=lambda q: int(q.stem.split("_b")[1].split("_")[0]))
        p = cands[-1] if cands else C.SYNTH_DIR / "missing.npz"
        budget = p.stem.split("_b")[1].split("_")[0] if cands else "?"
    if not p.exists() or not tr.exists():
        print("[fig5] synthetic or train patches missing -- skipping")
        return
    syn = np.load(p); real = np.load(tr)
    rng = np.random.default_rng(0)
    ri = rng.choice(len(real["X"]), n_cols, replace=False)
    si = rng.choice(len(syn["X"]), n_cols, replace=False)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.1 * n_cols, 6.4))
    for c in range(n_cols):
        axes[0, c].imshow(real["X"][ri[c]].T, cmap="gray", aspect="auto")
        axes[1, c].imshow(syn["X"][si[c]].T, cmap="gray", aspect="auto")
        axes[2, c].imshow(syn["Y"][si[c]].T, cmap=FACIES_CMAP, aspect="auto",
                          vmin=0, vmax=C.N_CLASSES - 1, interpolation="nearest")
        for r in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for r, lab in enumerate(["Real F3", "Pix2Pix synthetic", "Conditioning mask"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle(f"Real vs. Pix2Pix-synthesised seismic patches "
                 f"(budget = {budget} labels)", y=0.995)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURE_DIR / f"fig05_synthetic_grid.{ext}", bbox_inches="tight")
    print("wrote fig05_synthetic_grid")


def fig8_uncertainty(tag=None):
    cands = sorted(C.RESULTS_DIR.glob("expB_maps_*.npz"))
    if not cands:
        print("[fig8] no Experiment B maps -- skipping")
        return
    p = cands[0] if tag is None else C.RESULTS_DIR / f"expB_maps_{tag}.npz"
    d = np.load(p)
    k = int(np.argmax(d["epistemic"].mean(axis=(1, 2))))  # most uncertain patch

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    axes[0].imshow(d["X"][k].T, cmap="gray", aspect="auto")
    axes[0].set_title("A  Seismic section")
    axes[1].imshow(d["pred"][k].T, cmap=FACIES_CMAP, aspect="auto",
                   vmin=0, vmax=C.N_CLASSES - 1, interpolation="nearest")
    axes[1].set_title("B  Predicted facies")
    for ax, key, t in ((axes[2], "epistemic", "C  Epistemic (model)"),
                       (axes[3], "aleatoric", "D  Aleatoric (data)")):
        im = ax.imshow(d[key][k].T, cmap="magma", aspect="auto")
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="nats")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURE_DIR / f"fig08_uncertainty.{ext}", bbox_inches="tight")
    print("wrote fig08_uncertainty")


def fig9_reliability():
    cands = sorted(C.RESULTS_DIR.glob("expB_*.json"))
    cands = [c for c in cands if "maps" not in c.name]
    if not cands:
        print("[fig9] no Experiment B results -- skipping")
        return
    r = json.loads(cands[0].read_text())
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for key, lab, col in (("reliability_det", "Deterministic U-Net", "#7A7A7A"),
                          ("reliability_mc", "MC-Dropout (N=50)", "#0072B2")):
        curve = [b for b in r[key] if b["n"] > 0]
        ax.plot([b["conf"] for b in curve], [b["acc"] for b in curve],
                "-o", ms=4, color=col, lw=1.5, label=lab)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration on F3 test set")
    ece_d = r["deterministic"]["ece"]; ece_m = r["mc_dropout"]["ece"]
    ax.text(0.04, 0.92, f"ECE  deterministic {ece_d:.3f}\nECE  MC-Dropout   {ece_m:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURE_DIR / f"fig09_reliability.{ext}", bbox_inches="tight")
    print("wrote fig09_reliability")


def results_table(rows):
    if not rows:
        return
    lines = ["| Arm | Labels | mIoU | Dice | Boundary F1 |",
             "| --- | --- | --- | --- | --- |"]
    for arm in ["none", "classical", "pix2pix", "ldm"]:
        a = agg(rows, "miou").get(arm)
        if not a:
            continue
        for b in sorted(a):
            sel = [r for r in rows if r["arm"] == arm and r["budget"] == b]
            f = lambda k: (np.mean([s[k] for s in sel]), np.std([s[k] for s in sel]))
            mi, ms = f("miou"); di, ds = f("dice"); bi, bs_ = f("bf1")
            lines.append(f"| {ARM_STYLE[arm][0]} | {b} | {mi:.3f} ± {ms:.3f} "
                         f"| {di:.3f} ± {ds:.3f} | {bi:.3f} ± {bs_:.3f} |")
    txt = "\n".join(lines)
    (C.RESULTS_DIR / "table_experiment_a.md").write_text(txt)
    print("\n" + txt)


def main():
    C.ensure_dirs()
    rows = load_expA()
    print(f"loaded {len(rows)} Experiment A runs")
    fig4_learning_curve(rows)
    fig5_synthetic_grid()
    fig8_uncertainty()
    fig9_reliability()
    results_table(rows)
    print(f"\nFigures -> {C.FIGURE_DIR}")


if __name__ == "__main__":
    main()
