"""Step 8 -- Experiment B, full version: all 36 checkpoints, fixed test patches.

Replaces the single-checkpoint analysis of 04_run_experiment_b.py. For every
(arm, budget, seed) checkpoint it runs MC-Dropout on the SAME fixed set of
test1 patches and records, per ground-truth class:
    aleatoric, epistemic, total entropy (means), epistemic share, ECE, mIoU.
For every (arm, budget) it also forms the 3-seed ensemble and records the
ensemble mutual information (disagreement between seeds) as an independent
epistemic estimate.

    python scripts/08_experiment_b_full.py                  # everything, resumable
    python scripts/08_experiment_b_full.py --n-patches 40 --passes 10 --budgets 50 --arms none --tag smoke
    python scripts/08_experiment_b_full.py --summarize-only

Cost: dominated by T x n_patches forward passes per checkpoint. Entropy is
computed on the GPU with running sums, so no (T,C,H,W) tensors are stored.
Time one checkpoint first (--budgets 50 --arms none --seeds 0) and scale by 36.

Caveats to carry into the paper:
  * Seeds change both initialisation AND the sampled label subset, so ensemble
    disagreement mixes the two sources of variance.
  * Dropout sits in only 5 deep blocks (p=0.2); MC-Dropout is known to
    understate epistemic uncertainty. Treat absolute levels with caution; the
    trend with N is the testable claim.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C
from geoseg.metrics import _boundaries, confusion_matrix, metrics_from_cm
from geoseg.models import UNet, enable_mc_dropout
from geoseg.train import device

EPS = 1e-12
NB = C.CALIBRATION_BINS
P = C.PATCH_SIZE


def ent(p, dim=1):
    return -(p * torch.log(p + EPS)).sum(dim)


def pad_batch(xb):
    xb = torch.from_numpy(xb).float().unsqueeze(1)
    return torch.nn.functional.pad(
        xb, (0, C.IMG_SIZE - xb.shape[3], 0, C.IMG_SIZE - xb.shape[2]), mode="reflect")


class Acc:
    """Streaming per-class accumulators (class index C.N_CLASSES = 'all')."""
    def __init__(self):
        K = C.N_CLASSES + 1
        self.n = np.zeros(K)
        self.ale = np.zeros(K); self.epi = np.zeros(K); self.tot = np.zeros(K)
        self.bnd_epi = 0.0; self.bnd_n = 0.0; self.int_epi = 0.0; self.int_n = 0.0
        self.cm_mc = np.zeros((C.N_CLASSES,) * 2, dtype=np.int64)
        self.cm_det = np.zeros((C.N_CLASSES,) * 2, dtype=np.int64)
        self.bin_n = np.zeros(NB); self.bin_conf = np.zeros(NB); self.bin_acc = np.zeros(NB)

    def add_unc(self, y, ale, epi, tot, bmask):
        for c in range(C.N_CLASSES):
            m = y == c
            k = m.sum()
            if k:
                self.n[c] += k; self.ale[c] += ale[m].sum()
                self.epi[c] += epi[m].sum(); self.tot[c] += tot[m].sum()
        self.n[-1] += y.size; self.ale[-1] += ale.sum()
        self.epi[-1] += epi.sum(); self.tot[-1] += tot.sum()
        self.bnd_epi += epi[bmask].sum(); self.bnd_n += bmask.sum()
        self.int_epi += epi[~bmask].sum(); self.int_n += (~bmask).sum()

    def add_cal(self, conf, correct):
        b = np.minimum((conf * NB).astype(int), NB - 1)
        self.bin_n += np.bincount(b, minlength=NB)
        self.bin_conf += np.bincount(b, weights=conf, minlength=NB)
        self.bin_acc += np.bincount(b, weights=correct, minlength=NB)

    def ece(self):
        m = self.bin_n > 0
        w = self.bin_n[m] / self.bin_n.sum()
        return float(np.sum(w * np.abs(self.bin_acc[m] / self.bin_n[m]
                                       - self.bin_conf[m] / self.bin_n[m])))


@torch.no_grad()
def run_checkpoint(tag, X, Yc, bmasks, dev, passes, batch):
    """Returns (record dict, mean-prob array (N,C,P,P) float16 for ensembling)."""
    model = UNet().to(dev)
    model.load_state_dict(torch.load(C.CKPT_DIR / f"unet_{tag}.pt", map_location=dev))
    model.eval()
    det_cm = np.zeros((C.N_CLASSES,) * 2, dtype=np.int64)
    # deterministic pass (dropout off) for reference mIoU
    for i in range(0, len(X), batch):
        lo = model(pad_batch(X[i:i + batch]).to(dev))[:, :, :P, :P]
        det_cm += confusion_matrix(lo.argmax(1).cpu().numpy().ravel(),
                                   Yc[i:i + batch].ravel(), C.N_CLASSES)
    enable_mc_dropout(model)
    torch.manual_seed(0)  # same dropout-mask stream for every checkpoint
    acc = Acc(); acc.cm_det = det_cm
    mean_store = np.zeros((len(X), C.N_CLASSES, P, P), dtype=np.float16)
    for i in range(0, len(X), batch):
        xb = pad_batch(X[i:i + batch]).to(dev)
        sum_p = None; sum_h = None
        for _ in range(passes):
            p = torch.softmax(model(xb), 1)[:, :, :P, :P]
            h = ent(p)
            sum_p = p if sum_p is None else sum_p + p
            sum_h = h if sum_h is None else sum_h + h
        mean_p = sum_p / passes
        tot = ent(mean_p); ale = sum_h / passes
        epi = torch.clamp(tot - ale, min=0)
        conf, pred = mean_p.max(1)
        y = Yc[i:i + batch]
        acc.cm_mc += confusion_matrix(pred.cpu().numpy().ravel(), y.ravel(), C.N_CLASSES)
        acc.add_cal(conf.cpu().numpy().ravel(),
                    (pred.cpu().numpy() == y).astype(np.float64).ravel())
        acc.add_unc(y, ale.cpu().numpy(), epi.cpu().numpy(), tot.cpu().numpy(),
                    bmasks[i:i + batch])
        mean_store[i:i + batch] = mean_p.cpu().numpy().astype(np.float16)
    rows = []
    names = C.CLASS_NAMES + ["ALL"]
    for k, nm in enumerate(names):
        if acc.n[k] == 0:
            continue
        rows.append({"class": nm, "n_pix": int(acc.n[k]),
                     "aleatoric": acc.ale[k] / acc.n[k], "epistemic": acc.epi[k] / acc.n[k],
                     "total": acc.tot[k] / acc.n[k],
                     "epistemic_share": acc.epi[k] / max(acc.tot[k], EPS)})
    rec = {"tag": tag, "passes": passes, "n_patches": int(len(X)),
           "miou_mc": metrics_from_cm(acc.cm_mc)["mean_iou"],
           "miou_det": metrics_from_cm(acc.cm_det)["mean_iou"],
           "ece_mc": acc.ece(),
           "boundary_interior_epi_ratio": (acc.bnd_epi / acc.bnd_n) / ((acc.int_epi / acc.int_n) + EPS),
           "per_class": rows}
    return rec, mean_store


def ensemble_stats(mean_stores, Yc, bmasks, batch=16):
    """Seed-ensemble mutual information: H[mean_s p_s] - mean_s H[p_s]."""
    acc = Acc(); cm = np.zeros((C.N_CLASSES,) * 2, dtype=np.int64)
    for i in range(0, len(Yc), batch):
        ps = torch.stack([torch.from_numpy(m[i:i + batch].astype(np.float32))
                          for m in mean_stores])        # (S,B,C,H,W)
        mp = ps.mean(0); tot = ent(mp)
        ale = ent(ps, dim=2).mean(0)
        mi = torch.clamp(tot - ale, min=0)
        y = Yc[i:i + batch]
        cm += confusion_matrix(mp.argmax(1).numpy().ravel(), y.ravel(), C.N_CLASSES)
        acc.add_unc(y, ale.numpy(), mi.numpy(), tot.numpy(), bmasks[i:i + batch])
    rows = []
    for k, nm in enumerate(C.CLASS_NAMES + ["ALL"]):
        if acc.n[k]:
            rows.append({"class": nm, "ens_mi": acc.epi[k] / acc.n[k],
                         "ens_total": acc.tot[k] / acc.n[k],
                         "ens_share": acc.epi[k] / max(acc.tot[k], EPS)})
    return {"miou_ensemble": metrics_from_cm(cm)["mean_iou"], "per_class": rows}


def summarize(out_dir, tag):
    import pandas as pd
    from scipy.stats import spearmanr
    rows, ens = [], []
    for f in sorted(out_dir.glob("ckpt_*.json")):
        r = json.load(open(f))
        arm, b, s = r["tag"].split("_")
        for pc in r["per_class"]:
            rows.append({"arm": arm, "budget": int(b[1:]), "seed": int(s[1:]),
                         "miou_mc": r["miou_mc"], "miou_det": r["miou_det"],
                         "ece_mc": r["ece_mc"],
                         "bnd_int_ratio": r["boundary_interior_epi_ratio"], **pc})
    for f in sorted(out_dir.glob("ens_*.json")):
        r = json.load(open(f))
        arm, b = r["group"].split("_")
        for pc in r["per_class"]:
            ens.append({"arm": arm, "budget": int(b[1:]),
                        "miou_ensemble": r["miou_ensemble"], **pc})
    if not rows:
        sys.exit("no per-checkpoint results found")
    df = pd.DataFrame(rows); df.to_csv(out_dir / f"expB_full_percheckpoint{tag}.csv", index=False)
    all_ = df[df["class"] == "ALL"]
    print("\n=== Epistemic share of total entropy (ALL pixels), mean +/- SD over seeds ===")
    t = all_.groupby(["arm", "budget"])["epistemic_share"].agg(["mean", "std", "count"])
    print(t.round(4).to_string())
    print("\n=== Prediction: epistemic share rises as N falls (paired by seed, N=50 minus N=500) ===")
    verdict = []
    for arm, g in all_.groupby("arm"):
        a = g[g.budget == 50].set_index("seed")["epistemic_share"]
        b = g[g.budget == 500].set_index("seed")["epistemic_share"]
        d = (a - b).dropna()
        if len(d):
            pos = int((d > 0).sum())
            print(f"  {arm:10s} delta per seed = {np.round(d.values, 4).tolist()}  "
                  f"mean {d.mean():+.4f}  positive in {pos}/{len(d)} seeds")
            verdict.append((arm, pos, len(d)))
    print("  -> supported only if the delta is positive in all seeds for all arms; "
          "otherwise soften the mechanistic claim.")
    print("\n=== Epistemic mean (ALL pixels) vs budget ===")
    print(all_.groupby(["arm", "budget"])["epistemic"].mean().round(5).to_string())
    if ens:
        e = pd.DataFrame(ens); e.to_csv(out_dir / f"expB_full_ensemble{tag}.csv", index=False)
        ea = e[e["class"] == "ALL"].set_index(["arm", "budget"])
        mc = all_.groupby(["arm", "budget"])["epistemic"].mean()
        j = ea.join(mc.rename("mc_epistemic"))
        print("\n=== Seed-ensemble MI vs MC-Dropout epistemic (ALL pixels) ===")
        print(j[["ens_mi", "ens_share", "mc_epistemic", "miou_ensemble"]].round(5).to_string())
        if len(j) >= 4:
            rho, p = spearmanr(j["ens_mi"], j["mc_epistemic"])
            print(f"  Spearman rho (n={len(j)} arm-budget points) = {rho:.2f}, p = {p:.3f}")
            print("  (weak agreement means MC-Dropout is not tracking seed-level disagreement)")
        # does ensemble MI also grow as N falls?
        for arm, g in ea.reset_index().groupby("arm"):
            g = g.sort_values("budget")
            print(f"  ensemble MI, {arm}: " + ", ".join(
                f"N={int(r.budget)}: {r.ens_mi:.4f}" for r in g.itertuples()))
    print(f"\nCSV files written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=C.AUG_ARMS)
    ap.add_argument("--budgets", type=int, nargs="+", default=C.LABEL_BUDGETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--passes", type=int, default=C.MC_PASSES)
    ap.add_argument("--n-patches", type=int, default=200,
                    help="fixed, evenly spaced test1 patches (same for every checkpoint)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="test1")
    ap.add_argument("--tag", default="", help="suffix for the output folder (use for smoke tests)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    out_dir = C.RESULTS_DIR / ("expB_full" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    if args.summarize_only:
        return summarize(out_dir, tag)

    d = np.load(C.PATCH_CACHE / f"{args.split}_patches.npz")
    Xall, Yall = d["X"], d["Y"]
    sel = np.linspace(0, len(Xall) - 1, min(args.n_patches, len(Xall))).astype(int)
    X = Xall[sel].astype(np.float32); Yc = Yall[sel].astype(np.int64)
    bmasks = np.stack([_boundaries(y) for y in Yc])
    np.save(out_dir / "patch_indices.npy", sel)
    dev = device()
    print(f"device={dev}  patches={len(X)} of {len(Xall)}  passes={args.passes}")

    for arm in args.arms:
        for b in args.budgets:
            group = f"{arm}_b{b}"
            stores, ok = [], True
            for s in args.seeds:
                tg = f"{arm}_b{b}_s{s}"
                jf = out_dir / f"ckpt_{tg}.json"
                ck = C.CKPT_DIR / f"unet_{tg}.pt"
                if not ck.exists():
                    print(f"[missing] {ck.name} -- skipped"); ok = False; continue
                npy = out_dir / f"mean_{tg}.npy"
                if jf.exists() and npy.exists():
                    print(f"[skip] {tg}"); stores.append(np.load(npy)); continue
                print(f"=== {tg} ===", flush=True)
                rec, ms = run_checkpoint(tg, X, Yc, bmasks, dev, args.passes, args.batch)
                np.save(npy, ms)       # float16 cache; delete when the study is finished
                json.dump(rec, open(jf, "w"), indent=2)
                stores.append(ms)
                al = rec["per_class"][-1]
                print(f"  mIoU det {rec['miou_det']:.4f} mc {rec['miou_mc']:.4f}  "
                      f"ECE {rec['ece_mc']:.4f}  epi share {al['epistemic_share']:.4f}", flush=True)
            ef = out_dir / f"ens_{group}.json"
            if ok and len(stores) == len(args.seeds) and len(stores) > 1 and not ef.exists():
                r = ensemble_stats(stores, Yc, bmasks)
                r["group"] = group; r["seeds"] = args.seeds
                json.dump(r, open(ef, "w"), indent=2)
                print(f"  ensemble {group}: mIoU {r['miou_ensemble']:.4f}")
            del stores
    summarize(out_dir, tag)


if __name__ == "__main__":
    main()
