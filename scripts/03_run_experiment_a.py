"""Step 3 -- Experiment A: label budget x augmentation arm x seed.

Fixed-epoch benchmark (first stage; superseded by 09_step_matched_control.py for all reported comparisons).

    python scripts/03_run_experiment_a.py --arms none classical pix2pix
    python scripts/03_run_experiment_a.py --budgets 50 --seeds 0   # smoke test

Every run writes results/expA_<arm>_b<budget>_s<seed>.json and skips itself if
that file already exists, so the sweep is fully resumable after a crash or a
Windows reboot.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C
from geoseg.augment import build_augment, load_synthetic
from geoseg.data import sample_label_budget
from geoseg.train import device, evaluate, make_loader, save_json, train_model


def load_split(name):
    d = np.load(C.PATCH_CACHE / f"{name}_patches.npz")
    return d["X"], d["Y"]  # Y stays uint8


def build_training_set(X_all, Y_all, budget, arm, seed):
    idx = sample_label_budget(len(X_all), budget, seed)
    X, Y = X_all[idx], Y_all[idx]
    n_real, n_syn = len(X), 0
    if arm in ("pix2pix", "ldm", "l1only"):
        p = C.SYNTH_DIR / f"{arm}_b{budget}_s{seed}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run scripts/02_train_pix2pix.py first")
        Xs, Ys = load_synthetic(p)
        # synthetic patches are already padded to IMG_SIZE; crop back so the
        # arrays concatenate cleanly with the real PATCH_SIZE patches
        Xs = Xs[:, :X.shape[1], :X.shape[2]]
        Ys = Ys[:, :Y.shape[1], :Y.shape[2]]
        X = np.concatenate([X, Xs]).astype(np.float32)
        Y = np.concatenate([Y, Ys]).astype(np.uint8)
        n_syn = len(Xs)
    return X, Y, n_real, n_syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=C.LABEL_BUDGETS)
    ap.add_argument("--arms", nargs="+", default=C.AUG_ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--test-split", default="test1", choices=["test1", "test2", "both"])
    args = ap.parse_args()

    C.ensure_dirs()
    X_all, Y_all = load_split("train")
    X_va, Y_va = load_split("val")
    test_splits = ["test1", "test2"] if args.test_split == "both" else [args.test_split]
    tests = {t: load_split(t) for t in test_splits}
    dev = device()
    print(f"device={dev}  train pool={len(X_all)}  val={len(X_va)}")

    for seed in args.seeds:
        for budget in args.budgets:
            for arm in args.arms:
                tag = f"{arm}_b{budget}_s{seed}"
                out = C.RESULTS_DIR / f"expA_{tag}.json"
                if out.exists():
                    print(f"[skip] {tag}")
                    continue
                print(f"\n=== Experiment A | arm={arm} budget={budget} seed={seed} ===")
                X, Y, n_real, n_syn = build_training_set(X_all, Y_all, budget, arm, seed)
                print(f"  training patches: {n_real} real + {n_syn} synthetic")
                aug = build_augment(arm, seed=seed)
                model, info = train_model(X, Y, X_va, Y_va, aug,
                                          seed=seed, epochs=args.epochs)

                rec = {"experiment": "A", "arm": arm, "budget": budget,
                       "seed": seed, "n_real": n_real, "n_synthetic": n_syn,
                       **info}
                for t, (Xt, Yt) in tests.items():
                    m = evaluate(model, make_loader(Xt, Yt, None, shuffle=False), dev)
                    rec[f"test_{t}"] = m
                    print(f"  {t}: mIoU {m['mean_iou']:.4f}  "
                          f"Dice {m['mean_dice']:.4f}  BF1 {m['boundary_f1']:.4f}")
                save_json(rec, out)
                torch.save(model.state_dict(), C.CKPT_DIR / f"unet_{tag}.pt")

    print("\nAll runs complete ->", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
