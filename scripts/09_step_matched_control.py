"""Step 9 -- step-matched control for Experiment A / B.

Problem: Experiment A trains every run for 60 epochs, so optimiser steps scale
with the number of training patches (N=50 -> ~180 steps, N=500 -> ~1860). Any
effect of N is therefore confounded with training length.

Control: retrain the SMALL budgets with enough epochs that every run gets the
same number of optimiser steps as the reference (default: N=500, 60 epochs).
Same data, same seeds, same augmentation, same validation selection scheme.

    python scripts/09_step_matched_control.py --arms none --budgets 50 --seeds 0 --tag smoke   # time one
    python scripts/09_step_matched_control.py --arms none                                      # none: N=50,100,200 x 3 seeds
    python scripts/09_step_matched_control.py --arms pix2pix                                   # then pix2pix
    python scripts/09_step_matched_control.py --summarize-only

Outputs (never overwrite the original runs):
    results/expA_<arm>_b<N>_s<seed>_stepmatched.json
    checkpoints/unet_<arm>_b<N>_s<seed>_stepmatched.pt
    results/stepmatched_summary.csv
Resumable: finished runs are skipped.

How to read it:
  * If step-matched mIoU at small N rises toward the N=500 value, the flat
    learning curve in Experiment A was partly a training-length artefact.
  * If it stays flat, the plateau is a data/domain-shift property and the
    original conclusion is strengthened.
  * ECE / epistemic share are recomputed on the same 200 test1 patches as
    08_experiment_b_full.py so they are directly comparable.
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
from geoseg.metrics import _boundaries
from geoseg.train import device, evaluate, make_loader, save_json, train_model


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def steps_per_epoch(n):
    # mirrors train.make_loader: drop_last when len > batch_size
    return n // C.BATCH_SIZE if n > C.BATCH_SIZE else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["none"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--ref-budget", type=int, default=500)
    ap.add_argument("--ref-epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--ref-steps", type=int, default=None,
                    help="override the target step count directly")
    ap.add_argument("--passes", type=int, default=C.MC_PASSES)
    ap.add_argument("--n-patches", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    sfx = "_stepmatched" + (f"_{args.tag}" if args.tag else "")
    if args.summarize_only:
        return summarize(sfx)

    A = _load("expA", "03_run_experiment_a.py")
    B = _load("expB", "08_experiment_b_full.py")

    ref_steps = args.ref_steps or steps_per_epoch(args.ref_budget) * args.ref_epochs
    print(f"target optimiser steps per run: {ref_steps}")

    X_all, Y_all = A.load_split("train")
    X_va, Y_va = A.load_split("val")
    Xt, Yt = A.load_split("test1")
    dev = device()

    sel = np.linspace(0, len(Xt) - 1, min(args.n_patches, len(Xt))).astype(int)
    Xb = Xt[sel].astype(np.float32); Ycb = Yt[sel].astype(np.int64)
    bmasks = np.stack([_boundaries(y) for y in Ycb])

    for seed in args.seeds:
        for budget in args.budgets:
            for arm in args.arms:
                tag = f"{arm}_b{budget}_s{seed}"
                out = C.RESULTS_DIR / f"expA_{tag}{sfx}.json"
                if out.exists():
                    print(f"[skip] {tag}"); continue
                X, Y, n_real, n_syn = A.build_training_set(X_all, Y_all, budget, arm, seed)
                spe = steps_per_epoch(len(X))
                epochs = max(1, math.ceil(ref_steps / spe))
                log_every = max(10, epochs // 6)   # keep ~6 validation checks, as in the 60-epoch runs
                print(f"\n=== step-matched | {tag} | {len(X)} patches, {spe} steps/epoch "
                      f"-> {epochs} epochs ({epochs * spe} steps), val every {log_every} ===",
                      flush=True)
                aug = build_augment(arm, seed=seed)
                model, info = train_model(X, Y, X_va, Y_va, aug, seed=seed,
                                          epochs=epochs, log_every=log_every)
                rec = {"experiment": "A-stepmatched", "arm": arm, "budget": budget,
                       "seed": seed, "n_real": n_real, "n_synthetic": n_syn,
                       "epochs": epochs, "optimizer_steps": epochs * spe, **info}
                m = evaluate(model, make_loader(Xt, Yt, None, shuffle=False), dev)
                rec["test_test1"] = m
                print(f"  test1: mIoU {m['mean_iou']:.4f}  BF1 {m['boundary_f1']:.4f}", flush=True)
                ck = C.CKPT_DIR / f"unet_{tag}{sfx}.pt"
                torch.save(model.state_dict(), ck)
                # uncertainty on the same patches as Experiment B
                brec, _ = B.run_checkpoint(f"{tag}{sfx}", Xb, Ycb, bmasks, dev,
                                           args.passes, 8)
                al = brec["per_class"][-1]
                rec["expB"] = {"ece_mc": brec["ece_mc"], "miou_det": brec["miou_det"],
                               "epistemic": al["epistemic"], "total": al["total"],
                               "epistemic_share": al["epistemic_share"]}
                print(f"  ECE {brec['ece_mc']:.4f}  epi share {al['epistemic_share']:.4f}", flush=True)
                save_json(rec, out)
    summarize(sfx)


def summarize(sfx):
    import pandas as pd
    rows = []
    for f in sorted(C.RESULTS_DIR.glob(f"expA_*{sfx}.json")):
        r = json.load(open(f))
        orig = C.RESULTS_DIR / f"expA_{r['arm']}_b{r['budget']}_s{r['seed']}.json"
        o = json.load(open(orig)) if orig.exists() else None
        ob = C.RESULTS_DIR / "expB_full" / f"ckpt_{r['arm']}_b{r['budget']}_s{r['seed']}.json"
        ob = json.load(open(ob)) if ob.exists() else None
        rows.append({
            "arm": r["arm"], "budget": r["budget"], "seed": r["seed"],
            "epochs_sm": r["epochs"], "steps_sm": r["optimizer_steps"],
            "miou_orig": o["test_test1"]["mean_iou"] if o else np.nan,
            "miou_sm": r["test_test1"]["mean_iou"],
            "bf1_orig": o["test_test1"]["boundary_f1"] if o else np.nan,
            "bf1_sm": r["test_test1"]["boundary_f1"],
            "ece_orig": ob["ece_mc"] if ob else np.nan, "ece_sm": r["expB"]["ece_mc"],
            "epishare_orig": ob["per_class"][-1]["epistemic_share"] if ob else np.nan,
            "epishare_sm": r["expB"]["epistemic_share"]})
    if not rows:
        sys.exit("no step-matched results yet")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / f"stepmatched_summary{sfx}.csv", index=False)
    g = df.groupby(["arm", "budget"]).agg(["mean", "std"])
    cols = ["miou_orig", "miou_sm", "ece_orig", "ece_sm", "epishare_orig", "epishare_sm"]
    print("\n=== Original (60 epochs) vs step-matched, test1; mean / SD over seeds ===")
    print(g[cols].round(4).to_string())
    # reference: N=500 originals
    ref = []
    for f in sorted(C.RESULTS_DIR.glob("expA_*_b500_s?.json")):
        r = json.load(open(f)); ref.append((r["arm"], r["test_test1"]["mean_iou"]))
    if ref:
        rd = pd.DataFrame(ref, columns=["arm", "miou"]).groupby("arm")["miou"].agg(["mean", "std"])
        print("\n=== Reference: N=500 at 60 epochs (the step target) ===")
        print(rd.round(4).to_string())
    d = df.assign(gain=df.miou_sm - df.miou_orig)
    print("\n=== mIoU gain from step matching (per seed, same labels and seed) ===")
    for (a, b), x in d.groupby(["arm", "budget"]):
        print(f"  {a:8s} N={b:<4d} gain per seed = {np.round(x.gain.values, 4).tolist()}  mean {x.gain.mean():+.4f}")
    print("\nCSV ->", C.RESULTS_DIR / f"stepmatched_summary{sfx}.csv")


if __name__ == "__main__":
    main()
