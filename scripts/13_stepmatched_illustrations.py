r"""Fig. 5 (test predictions) and Fig. 9 (spatial uncertainty) from a STEP-MATCHED checkpoint.

    python scripts\13_stepmatched_illustrations.py                       # classical, N=200, seed 0
    python scripts\13_stepmatched_illustrations.py --arm pix2pix --budget 100 --seed 1

Needs checkpoints\unet_<arm>_b<N>_s<seed>_stepmatched.pt and patches\test1_patches.npz.
Writes figures\real\real_fig03_test_predictions_stepmatched.(png|svg|pdf) and
real_fig09_uncertainty_stepmatched.(png|svg|pdf). GPU optional (3 patches, T=50).
"""
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from matplotlib.colors import ListedColormap
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoseg import config as C
from geoseg.models import UNet, enable_mc_dropout

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="classical"); ap.add_argument("--budget", type=int, default=200)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--split", default="test1")
ap.add_argument("--passes", type=int, default=50)
ap.add_argument("--worst", type=int, default=None, help="skip the scan and use this patch index (found previously: 5)")
a = ap.parse_args()

plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "savefig.dpi": 400})
CM = ListedColormap(C.CLASS_COLORS); OUT = C.FIGURE_DIR / "real"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = C.CKPT_DIR / f"unet_{a.arm}_b{a.budget}_s{a.seed}_stepmatched.pt"
if not ck.exists(): sys.exit(f"missing {ck}")
model = UNet().to(dev); model.load_state_dict(torch.load(ck, map_location=dev))

d = np.load(C.PATCH_CACHE / f"{a.split}_patches.npz"); X, Y = d["X"], d["Y"]
idx = np.random.default_rng(11).choice(len(X), 3, replace=False)   # same sections as the original Fig. 5

def prep(x):
    t = torch.from_numpy(x).float()[None, None]
    return torch.nn.functional.pad(t, (0, C.IMG_SIZE - x.shape[1], 0, C.IMG_SIZE - x.shape[0]), mode="reflect").to(dev)

def save(fig, n):
    for e in ("png", "svg", "pdf"): fig.savefig(OUT / f"{n}.{e}", bbox_inches="tight")
    plt.close(fig); print("wrote", n)

# Fig. 5: deterministic predictions
model.eval(); h, w = X[0].shape
fig, ax = plt.subplots(4, 3, figsize=(7.5, 9.5))
with torch.no_grad():
    for c, i in enumerate(idx):
        p = model(prep(X[i])).argmax(1).cpu().numpy()[0, :h, :w]
        lo, hi = np.percentile(X[i], [2, 98])
        ax[0, c].imshow(X[i].T, cmap="gray", aspect="auto", vmin=lo, vmax=hi)
        ax[1, c].imshow(Y[i].T, cmap=CM, aspect="auto", vmin=0, vmax=C.N_CLASSES-1, interpolation="nearest")
        ax[2, c].imshow(p.T, cmap=CM, aspect="auto", vmin=0, vmax=C.N_CLASSES-1, interpolation="nearest")
        ax[3, c].imshow((p != Y[i]).T.astype(float), cmap="Reds", aspect="auto", vmin=0, vmax=1)
        for r in range(4): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
for r, t in enumerate(["Seismic amplitude", "Expert labels", "Model prediction", "Error map (red: wrong)"]):
    ax[r, 0].set_ylabel(t)
fig.tight_layout(); save(fig, "real_fig03_test_predictions_stepmatched")

# Fig. 9 (main) and Fig. A.3 (appendix): MC-Dropout maps
from matplotlib.colors import PowerNorm
ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-3)   # class axis = dim -3
def mc_maps(x):
    model.eval(); enable_mc_dropout(model)
    with torch.no_grad():
        ps = torch.stack([torch.softmax(model(prep(x)), 1) for _ in range(a.passes)])
    P = ps.mean(0); alea = ent(ps).mean(0); tot = ent(P)
    cut = lambda t: t.cpu().numpy()[0, :h, :w]
    return cut(P.argmax(1).float()).astype(int), cut((tot - alea).clamp_min(0)), cut(alea), cut(tot)

# rule: the section with the highest mean mutual information among ALL test patches of the split
if a.worst is None:
    NSEL = len(X); mi_mean = []
    for i in range(NSEL):
        mi_mean.append(mc_maps(X[i])[1].mean())
        if i % 50 == 0: print("scanning", i, "/", NSEL)
    k_worst = int(np.argmax(mi_mean)); print("highest-MI section:", k_worst, "mean MI", mi_mean[k_worst], "| mean over all:", float(np.mean(mi_mean)))
else:
    k_worst = a.worst

def uq_figure(k, name, with_error):
    torch.manual_seed(0)   # fixed MC-Dropout masks so the reported numbers are reproducible
    x = X[k]; pred, epi, alea, tot = mc_maps(x); err = (pred != Y[k]).astype(float)
    print(f"{name}: section {k} | mean MI {epi.mean():.4f} | mean aleatoric {alea.mean():.4f} | mean entropy {tot.mean():.4f} | pixel acc {1-err.mean():.4f}")
    print(f"   entropy on wrong px {tot[err==1].mean():.3f} vs correct {tot[err==0].mean():.3f}; MI {epi[err==1].mean():.4f} vs {epi[err==0].mean():.4f}")
    lo, hi = np.percentile(x, [2, 98])
    pan = [("Seismic amplitude", x, dict(cmap="gray", vmin=lo, vmax=hi), False),
           ("Predicted facies", pred, dict(cmap=CM, vmin=0, vmax=C.N_CLASSES-1, interpolation="nearest"), False)]
    if with_error: pan.append(("Error (red: wrong)", err, dict(cmap="Reds", vmin=0, vmax=1, interpolation="nearest"), False))
    pan += [("Epistemic (mutual information, nats)", epi, dict(cmap="YlOrRd", norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(epi, 99.5))), True),
            ("Aleatoric (mean entropy, nats)", alea, dict(cmap="YlOrRd", norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(alea, 99.5))), True)]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.6)); ax = axs.ravel()
    from matplotlib.patches import Patch
    ax[-1].axis('off'); ax[-1].legend(handles=[Patch(facecolor=C.CLASS_COLORS[q], edgecolor='k', label=C.CLASS_NAMES[q]) for q in range(C.N_CLASSES)], title='Facies', loc='center', fontsize=11, title_fontsize=12, frameon=False)
    for j, (t, im, kw, cb) in enumerate(pan):
        m = ax[j].imshow(im.T, aspect="auto", **kw); ax[j].set_xticks([]); ax[j].set_yticks([]); ax[j].set_title(t, fontsize=10)
        if cb: fig.colorbar(m, ax=ax[j], fraction=0.046, pad=0.02).ax.tick_params(labelsize=9)
    fig.tight_layout(); save(fig, name)  # saved at 400 dpi

uq_figure(k_worst, "real_fig09_uncertainty_stepmatched", with_error=True)
uq_figure(int(idx[0]), "real_figA3_uncertainty_typical_stepmatched", with_error=True)
print(f"checkpoint: {ck.name}  split: {a.split}")
