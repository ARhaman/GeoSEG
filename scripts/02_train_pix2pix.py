"""Step 2 -- train Pix2Pix (facies mask -> seismic) and emit synthetic patches.

One generator is trained PER LABEL BUDGET PER SEED, using only the patches
that budget is allowed to see. This is the single most important fairness
constraint in the whole benchmark: a generator trained on all 500 patches and
then used to augment the 50-patch arm would leak information and produce a
result that looks great and means nothing.

    python scripts/02_train_pix2pix.py --budgets 50 100 200 500 --seeds 0 1 2

Ablation (is it the GAN or just mask-conditioned L1 regression?):
    python scripts/02_train_pix2pix.py --l1-only --budgets 100 200 --seeds 0 1 2 3 4
trains the SAME generator (same init, label subset, epochs, synthesis seed) with
the L1 term only -- no discriminator, no adversarial loss -- and writes
synthetic/l1only_b<N>_s<seed>.npz. Existing pix2pix files are never touched.

Roughly 8-15 min per (budget, seed) on an RTX 4500 Ada at 300 epochs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C
from geoseg.data import PatchDataset, sample_label_budget
from geoseg.models import P2PDiscriminator, P2PGenerator, one_hot
from geoseg.train import device, set_seed
from torch.utils.data import DataLoader


def train_pix2pix(X, Y, seed, epochs=300, lam_l1=100.0, lr=2e-4, verbose=True, adversarial=True):
    set_seed(seed)
    dev = device()
    G = P2PGenerator().to(dev)          # G is built first, so its init is identical
    D = P2PDiscriminator().to(dev)      # with and without the discriminator
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()

    loader = DataLoader(PatchDataset(X, Y), batch_size=8, shuffle=True,
                        num_workers=0, drop_last=len(X) > 8)

    for ep in range(1, epochs + 1):
        gl = dl = nb = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            m = one_hot(y)
            fake = G(m)

            if adversarial:
                optD.zero_grad(set_to_none=True)
                d_real = D(m, x)
                d_fake = D(m, fake.detach())
                loss_d = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                                bce(d_fake, torch.zeros_like(d_fake)))
                loss_d.backward()
                optD.step()
                d_fake2 = D(m, fake)
                adv = bce(d_fake2, torch.ones_like(d_fake2))
            else:
                loss_d = torch.zeros(())
                adv = 0.0

            optG.zero_grad(set_to_none=True)
            loss_g = adv + lam_l1 * l1(fake, x)
            loss_g.backward()
            optG.step()

            gl += loss_g.detach().item(); dl += loss_d.detach().item(); nb += 1
        if verbose and (ep % 50 == 0 or ep == epochs):
            print(f"    p2p ep {ep:4d}  G {gl/nb:.3f}  D {dl/nb:.3f}", flush=True)
    return G


@torch.no_grad()
def synthesise(G, Y_masks, n_out, seed=0, jitter=True):
    """Generate synthetic seismic conditioned on real facies masks.

    Masks are recycled from the labelled budget (we do not invent new
    stratigraphy -- that would risk exactly the geological hallucination the
    earlier analysis warned about). Diversity comes from generator noise
    injected as small input dropout on the one-hot mask.
    """
    dev = device()
    G.eval()
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    idx = rng.integers(0, len(Y_masks), size=n_out)
    for i in range(0, n_out, 8):
        batch = idx[i:i + 8]
        y = torch.from_numpy(np.stack([Y_masks[j] for j in batch])).long()
        y = torch.nn.functional.pad(y[:, None].float(),
                                    (0, C.IMG_SIZE - y.shape[2],
                                     0, C.IMG_SIZE - y.shape[1]),
                                    mode="replicate")[:, 0].long().to(dev)
        m = one_hot(y)
        if jitter:
            m = m * (torch.rand_like(m[:, :1]) > 0.05).float()
        out = G(m).cpu().numpy()[:, 0]
        Xs.append(out)
        Ys.append(y.cpu().numpy())
    return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=C.LABEL_BUDGETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--l1-only", action="store_true",
                    help="ablation: generator trained with the L1 loss only (no discriminator)")
    args = ap.parse_args()
    prefix = "l1only" if args.l1_only else "pix2pix"

    C.ensure_dirs()
    d = np.load(C.PATCH_CACHE / "train_patches.npz")
    X_all, Y_all = d["X"], d["Y"]
    print(f"train pool: {len(X_all)} patches")

    for seed in args.seeds:
        for budget in args.budgets:
            tag = f"b{budget}_s{seed}"
            out = C.SYNTH_DIR / f"{prefix}_{tag}.npz"
            if out.exists():
                print(f"[skip] {tag} already generated")
                continue
            print(f"\n=== {prefix}  budget={budget}  seed={seed} ===")
            idx = sample_label_budget(len(X_all), budget, seed)
            G = train_pix2pix(X_all[idx], Y_all[idx], seed, epochs=args.epochs,
                              adversarial=not args.l1_only)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n_out = int(round(budget * C.SYNTH_RATIO))
            Xs, Ys = synthesise(G, Y_all[idx], n_out, seed=seed)
            np.savez_compressed(out, X=Xs, Y=Ys)
            torch.save(G.state_dict(), C.CKPT_DIR / f"p2p_G_{prefix}_{tag}.pt" if args.l1_only else C.CKPT_DIR / f"p2p_G_{tag}.pt")
            print(f"  -> {n_out} synthetic patches  {out.name}")


if __name__ == "__main__":
    main()
