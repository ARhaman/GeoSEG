"""Step 12 -- backbone robustness: repeat the key arm comparison with a second network.

Same data, same label budgets and seeds, same augmentation arms, same step-matched
training length (default 1,860 optimiser steps), same validation-based checkpoint
selection -- only the segmentation network changes (U-Net -> DeepLabV3+-style,
geoseg/models_alt.py, trained from scratch). Synthetic patches come from the SAME
generators as the U-Net runs, so nothing but the backbone differs.

Per run: test1 mIoU / BF1 (all patches) and uncertainty quality (ECE, NLL, AUROC,
AURC; deterministic and MC-Dropout) on test1.

    python scripts/12_second_backbone.py --arms none --budgets 100 --seeds 0 --tag timing --n-patches 40 --passes 5
    python scripts/12_second_backbone.py                     # none/classical/pix2pix, N=100,200, 5 seeds
    python scripts/12_second_backbone.py --summarize-only

Results -> results/expC_<backbone>_<arm>_b<N>_s<seed>.json   (resumable)
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from geoseg import config as C
from geoseg.augment import build_augment
from geoseg.models import UNet
from geoseg.models_alt import BACKBONES
from geoseg.train import device, evaluate, make_loader, save_json, train_model


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def steps_per_epoch(n):
    return n // C.BATCH_SIZE if n > C.BATCH_SIZE else 1


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(d, size=(n, len(d))).mean(1), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="deeplab", choices=list(BACKBONES))
    ap.add_argument("--arms", nargs="+", default=["none", "classical", "pix2pix"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ref-steps", type=int, default=1860)
    ap.add_argument("--passes", type=int, default=C.MC_PASSES)
    ap.add_argument("--n-patches", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    pre = f"expC_{args.backbone}" + (f"_{args.tag}" if args.tag else "")
    if args.summarize_only:
        return summarize(pre)

    A = _load("expA", "03_run_experiment_a.py")
    Q = _load("quality", "11_uncertainty_quality.py")
    model_fn = BACKBONES[args.backbone]
    n_alt = sum(p.numel() for p in model_fn().parameters())
    n_unet = sum(p.numel() for p in UNet().parameters())
    print(f"parameters: {args.backbone} {n_alt/1e6:.2f} M | U-Net {n_unet/1e6:.2f} M", flush=True)

    X_all, Y_all = A.load_split("train")
    X_va, Y_va = A.load_split("val")
    Xt, Yt = A.load_split("test1")
    dev = device()
    sel = np.linspace(0, len(Xt) - 1, min(args.n_patches, len(Xt))).astype(int)
    Xq = Xt[sel].astype(np.float32); Yq = Yt[sel].astype(np.int64)

    for seed in args.seeds:
        for budget in args.budgets:
            for arm in args.arms:
                tag = f"{arm}_b{budget}_s{seed}"
                out = C.RESULTS_DIR / f"{pre}_{tag}.json"
                if out.exists():
                    print(f"[skip] {tag}"); continue
                X, Y, n_real, n_syn = A.build_training_set(X_all, Y_all, budget, arm, seed)
                spe = steps_per_epoch(len(X))
                epochs = max(1, math.ceil(args.ref_steps / spe))
                log_every = max(10, epochs // 6)
                print(f"\n=== {args.backbone} | {tag} | {len(X)} patches -> {epochs} epochs "
                      f"({epochs * spe} steps) ===", flush=True)
                model, info = train_model(X, Y, X_va, Y_va, build_augment(arm, seed=seed), seed=seed,
                                          epochs=epochs, log_every=log_every, model_fn=model_fn)
                m = evaluate(model, make_loader(Xt, Yt, None, shuffle=False), dev)
                print(f"  test1: mIoU {m['mean_iou']:.4f}  BF1 {m['boundary_f1']:.4f}", flush=True)
                rec, _ = Q.run_model(model, tag, Xq, Yq, dev, args.passes, 8, keep_det=False)
                print(f"  det ECE {rec['det']['ece']:.4f} AUROC {rec['det']['auroc_ent']:.3f} | "
                      f"MC ECE {rec['mc']['ece']:.4f} AUROC(MI) {rec['mc']['auroc_mi']:.3f}", flush=True)
                torch.save(model.state_dict(), C.CKPT_DIR / f"{pre}_{tag}.pt")
                m.pop("confusion_matrix", None)
                save_json({"experiment": "C-backbone", "backbone": args.backbone, "arm": arm,
                           "budget": budget, "seed": seed, "epochs": epochs,
                           "optimizer_steps": epochs * spe, "n_real": n_real, "n_synthetic": n_syn,
                           "train_seconds": info["train_seconds"], "test_test1": m,
                           "quality": {k: rec[k] for k in ("det", "mc")}}, out)
    summarize(pre)


def summarize(pre):
    import pandas as pd
    rows = []
    for f in sorted(C.RESULTS_DIR.glob(f"{pre}_*_b*_s*.json")):
        r = json.load(open(f))
        row = {"arm": r["arm"], "N": r["budget"], "seed": r["seed"],
               "mIoU": r["test_test1"]["mean_iou"], "BF1": r["test_test1"]["boundary_f1"]}
        for meth in ("det", "mc"):
            for k in ("ece", "nll", "auroc_ent", "aurc"):
                row[f"{meth}_{k}"] = r["quality"][meth][k]
        rows.append(row)
    if not rows:
        sys.exit("no results yet")
    df = pd.DataFrame(rows); df.to_csv(C.RESULTS_DIR / f"{pre}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== mean over seeds ===")
    print(df.groupby(["N", "arm"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
    print("\n=== paired differences over seeds (A - B), 95% bootstrap CI ===")
    cols = ["mIoU", "BF1", "det_ece", "det_nll", "det_auroc_ent", "det_aurc", "mc_ece"]
    for N in sorted(df.N.unique()):
        d = df[df.N == N]
        for A, B in [("pix2pix", "none"), ("pix2pix", "classical"), ("classical", "none")]:
            a = d[d.arm == A].set_index("seed"); b = d[d.arm == B].set_index("seed")
            s = sorted(set(a.index) & set(b.index))
            if len(s) < 3:
                continue
            line = f"  N={N:<4d} {A:9s}-{B:9s} (n={len(s)}): "
            for c in cols:
                x = (a.loc[s, c] - b.loc[s, c]).values; lo, hi = boot(x)
                line += f"{c} {x.mean():+.4f}[{lo:+.4f},{hi:+.4f}]  "
            print(line)


if __name__ == "__main__":
    main()
