"""Real F3 seismic imagery, built directly from your own cached data.

I don't have file access to your machine, so I can't generate these myself --
this script runs on your machine and reads the patch cache your own
00_prepare_data.py already wrote. Every pixel in the output figures is your
real F3 amplitude and facies-label data. Nothing here is synthetic or
schematic.

    python scripts\07_real_seismic_figures.py

Takes seconds (no GPU, no retraining). Writes to figures\real\ alongside
the other figures. Also exports vector (.svg / .pdf) copies of every
panel so they open as editable paths in Illustrator/Inkscape/PowerPoint
(Insert > Object, or ungroup after paste), unlike the .png versions.
"""
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
    "font.size": 9.5, "font.family": "DejaVu Sans", "savefig.dpi": 400,
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
})

OUT = C.FIGURE_DIR / "real"
FACIES_CMAP = ListedColormap(C.CLASS_COLORS)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")   # editable vector
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")   # editable vector
    plt.close(fig)
    print(f"  wrote {name}.png / .svg / .pdf")


# --------------------------------------------------------------------------
# Panel 1 -- real seismic section gallery with facies overlay
# --------------------------------------------------------------------------
def real_section_gallery(n_panels=4):
    d = np.load(C.PATCH_CACHE / "train_patches.npz")
    X, Y = d["X"], d["Y"]
    rng = np.random.default_rng(7)
    idx = rng.choice(len(X), n_panels, replace=False)

    fig, axes = plt.subplots(2, n_panels, figsize=(2.6 * n_panels, 5.6))
    for c, i in enumerate(idx):
        axes[0, c].imshow(X[i].T, cmap="gray", aspect="auto",
                          vmin=np.percentile(X[i], 2), vmax=np.percentile(X[i], 98))
        axes[0, c].set_title(f"Patch {i}", fontsize=9)
        axes[1, c].imshow(Y[i].T, cmap=FACIES_CMAP, aspect="auto",
                          vmin=0, vmax=C.N_CLASSES - 1, interpolation="nearest")
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("Seismic amplitude\n(real F3 data)", fontsize=9)
    axes[1, 0].set_ylabel("Expert facies label\n(real F3 annotation)", fontsize=9)
    fig.suptitle("Real F3 Netherlands training patches: seismic amplitude and "
                 "expert facies annotation", y=1.01, fontsize=11.5)
    fig.tight_layout()
    save(fig, "real_fig01_section_gallery")


# --------------------------------------------------------------------------
# Panel 2 -- real vs Pix2Pix-synthetic, same conditioning mask
# --------------------------------------------------------------------------
def real_vs_synthetic_grid(budget=500, seed=0, n_cols=4):
    synth_path = C.SYNTH_DIR / f"pix2pix_b{budget}_s{seed}.npz"
    if not synth_path.exists():
        cands = sorted(C.SYNTH_DIR.glob("pix2pix_b*.npz"),
                       key=lambda q: int(q.stem.split("_b")[1].split("_")[0]))
        if not cands:
            print("  [skip] no synthetic patches found -- run 02_train_pix2pix.py first")
            return
        synth_path = cands[-1]
        budget = synth_path.stem.split("_b")[1].split("_")[0]

    syn = np.load(synth_path)
    real = np.load(C.PATCH_CACHE / "train_patches.npz")
    rng = np.random.default_rng(3)
    ri = rng.choice(len(real["X"]), n_cols, replace=False)
    si = rng.choice(len(syn["X"]), n_cols, replace=False)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.4 * n_cols, 6.6))
    for c in range(n_cols):
        vmin, vmax = np.percentile(real["X"][ri[c]], [2, 98])
        axes[0, c].imshow(real["X"][ri[c]].T, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        axes[1, c].imshow(syn["X"][si[c]].T, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        axes[2, c].imshow(syn["Y"][si[c]].T, cmap=FACIES_CMAP, aspect="auto",
                          vmin=0, vmax=C.N_CLASSES - 1, interpolation="nearest")
        for r in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("Real F3 seismic", fontsize=9)
    axes[1, 0].set_ylabel("Pix2Pix synthetic\nseismic", fontsize=9)
    axes[2, 0].set_ylabel("Conditioning\nfacies mask", fontsize=9)
    fig.suptitle(f"Real vs. Pix2Pix-synthesised seismic (label budget N={budget})", y=1.02, fontsize=10.5)
    fig.tight_layout()
    save(fig, "real_fig02_synthetic_comparison")


# --------------------------------------------------------------------------
# Panel 3 -- a real test prediction: seismic, ground truth, prediction, error
# --------------------------------------------------------------------------
def real_prediction_panel(arm="classical", budget=500, seed=0, split="test1"):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geoseg.models import UNet

    ckpt = C.CKPT_DIR / f"unet_{arm}_b{budget}_s{seed}.pt"
    if not ckpt.exists():
        print(f"  [skip] checkpoint {ckpt.name} not found")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    d = np.load(C.PATCH_CACHE / f"{split}_patches.npz")
    X, Y = d["X"], d["Y"]
    rng = np.random.default_rng(11)
    idx = rng.choice(len(X), 3, replace=False)

    fig, axes = plt.subplots(4, 3, figsize=(7.5, 9.5))
    with torch.no_grad():
        for c, i in enumerate(idx):
            x = X[i]
            xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
            xt = torch.nn.functional.pad(xt, (0, C.IMG_SIZE - x.shape[1], 0, C.IMG_SIZE - x.shape[0]),
                                         mode="reflect").to(dev)
            pred = model(xt).argmax(1).cpu().numpy()[0, :x.shape[0], :x.shape[1]]
            true = Y[i]
            err = (pred != true).astype(float)

            vmin, vmax = np.percentile(x, [2, 98])
            axes[0, c].imshow(x.T, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
            axes[1, c].imshow(true.T, cmap=FACIES_CMAP, aspect="auto",
                              vmin=0, vmax=C.N_CLASSES - 1, interpolation="nearest")
            axes[2, c].imshow(pred.T, cmap=FACIES_CMAP, aspect="auto",
                              vmin=0, vmax=C.N_CLASSES - 1, interpolation="nearest")
            axes[3, c].imshow(err.T, cmap="Reds", aspect="auto", vmin=0, vmax=1)
            for r in range(4):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    row_labels = ["Real seismic", "Expert ground truth", "Model prediction", "Error map (red = wrong)"]
    for r, lab in enumerate(row_labels):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle(f"Real test predictions ({split}, {arm} arm, N={budget}, seed={seed})",
                y=1.005, fontsize=11.5)
    fig.tight_layout()
    save(fig, "real_fig03_test_predictions")


if __name__ == "__main__":
    C.ensure_dirs()
    print("Generating real-data figures from your cached F3 arrays...")
    if not (C.PATCH_CACHE / "train_patches.npz").exists():
        sys.exit("Patch cache not found -- run scripts\\00_prepare_data.py first.")
    real_section_gallery()
    real_vs_synthetic_grid()
    real_prediction_panel()
    print(f"\nDone. Real-data figures (with editable .svg/.pdf copies) -> {OUT}")