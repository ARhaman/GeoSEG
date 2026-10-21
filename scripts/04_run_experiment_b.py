"""Step 4 -- Experiment B: MC-Dropout uncertainty + calibration.

Reuses a checkpoint from Experiment A -- no retraining, so this is cheap
(~10-20 min total). Produces the arrays behind Figures 8 and 9.

    python scripts/04_run_experiment_b.py --arm classical --budget 500 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C
from geoseg.metrics import calibration, decompose_uncertainty, metrics_from_cm, confusion_matrix
from geoseg.models import UNet, enable_mc_dropout
from geoseg.train import device, save_json


@torch.no_grad()
def mc_predict(model, X, dev, passes=C.MC_PASSES, batch=8):
    """Return (T, N, C, H, W) softmax samples -- memory-aware: we accumulate
    per-patch so the full tensor is only (T, C, H, W) at a time."""
    n_drop = enable_mc_dropout(model)
    print(f"  MC-Dropout active on {n_drop} layers, {passes} passes")
    means, totals, aleas, epis, preds = [], [], [], [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1)
        xb = torch.nn.functional.pad(
            xb, (0, C.IMG_SIZE - xb.shape[3], 0, C.IMG_SIZE - xb.shape[2]),
            mode="reflect").to(dev)
        samples = []
        for _ in range(passes):
            samples.append(torch.softmax(model(xb), 1).cpu().numpy())
        s = np.stack(samples)                       # (T, B, C, H, W)
        for b in range(s.shape[1]):
            mean_p, tot, ale, epi = decompose_uncertainty(s[:, b])
            means.append(mean_p); totals.append(tot)
            aleas.append(ale); epis.append(epi)
            preds.append(mean_p.argmax(0))
    return (np.stack(means), np.stack(totals),
            np.stack(aleas), np.stack(epis), np.stack(preds))


@torch.no_grad()
def deterministic_probs(model, X, dev, batch=8):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1)
        xb = torch.nn.functional.pad(
            xb, (0, C.IMG_SIZE - xb.shape[3], 0, C.IMG_SIZE - xb.shape[2]),
            mode="reflect").to(dev)
        out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="classical")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=C.MC_PASSES)
    ap.add_argument("--n-patches", type=int, default=24,
                    help="test patches to analyse (keeps arrays small)")
    ap.add_argument("--split", default="test1")
    args = ap.parse_args()

    C.ensure_dirs()
    tag = f"{args.arm}_b{args.budget}_s{args.seed}"
    ckpt = C.CKPT_DIR / f"unet_{tag}.pt"
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found -- run Experiment A for this configuration first")

    dev = device()
    model = UNet().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    d = np.load(C.PATCH_CACHE / f"{args.split}_patches.npz")
    X, Y = d["X"], d["Y"].astype(np.int64)
    sel = np.linspace(0, len(X) - 1, min(args.n_patches, len(X))).astype(int)
    X, Y = X[sel], Y[sel]
    Yp = np.stack([np.pad(y, ((0, C.IMG_SIZE - y.shape[0]),
                              (0, C.IMG_SIZE - y.shape[1])), mode="edge") for y in Y])
    print(f"analysing {len(X)} patches from {args.split}, checkpoint {tag}")

    mean_p, tot, ale, epi, pred = mc_predict(model, X, dev, passes=args.passes)
    det_p = deterministic_probs(model, X, dev)
    det_pred = det_p.argmax(1)

    np.savez_compressed(
        C.RESULTS_DIR / f"expB_maps_{tag}.npz",
        X=X, Y=Yp, mean_prob=mean_p.astype(np.float16), pred=pred.astype(np.uint8),
        total=tot.astype(np.float32), aleatoric=ale.astype(np.float32),
        epistemic=epi.astype(np.float32), det_pred=det_pred.astype(np.uint8))

    flat_y = Yp.ravel()
    cal_mc = calibration(mean_p.transpose(0, 2, 3, 1).reshape(-1, C.N_CLASSES), flat_y)
    cal_det = calibration(det_p.transpose(0, 2, 3, 1).reshape(-1, C.N_CLASSES), flat_y)

    m_mc = metrics_from_cm(confusion_matrix(pred.ravel(), flat_y))
    m_det = metrics_from_cm(confusion_matrix(det_pred.ravel(), flat_y))

    # Does uncertainty actually sit on geological boundaries? (Figure 8 claim)
    from geoseg.metrics import _boundaries
    bmask = np.stack([_boundaries(y) for y in Yp])
    boundary_ratio = float(epi[bmask].mean() / (epi[~bmask].mean() + 1e-12))

    rec = {"experiment": "B", "checkpoint": tag, "passes": args.passes,
           "split": args.split, "n_patches": int(len(X)),
           "mc_dropout": {**m_mc, "ece": cal_mc["ece"]},
           "deterministic": {**m_det, "ece": cal_det["ece"]},
           "reliability_mc": cal_mc["reliability"],
           "reliability_det": cal_det["reliability"],
           "epistemic_boundary_vs_interior_ratio": boundary_ratio}
    rec["mc_dropout"].pop("confusion_matrix", None)
    rec["deterministic"].pop("confusion_matrix", None)
    save_json(rec, C.RESULTS_DIR / f"expB_{tag}.json")

    print(f"\n  deterministic  mIoU {m_det['mean_iou']:.4f}  ECE {cal_det['ece']:.4f}")
    print(f"  MC-Dropout     mIoU {m_mc['mean_iou']:.4f}  ECE {cal_mc['ece']:.4f}")
    print(f"  epistemic uncertainty boundary/interior ratio = {boundary_ratio:.2f}x")
    print("  (>1 supports the claim that uncertainty concentrates at geological "
          "boundaries rather than facies interiors)")


if __name__ == "__main__":
    main()
