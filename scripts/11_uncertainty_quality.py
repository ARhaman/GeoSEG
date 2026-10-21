"""Step 11 -- Experiment B (revised): uncertainty QUALITY on step-matched checkpoints.

Replaces the single-checkpoint, fixed-epoch analysis. Question: does generative
augmentation make the model's uncertainty more reliable, and is a seed ensemble
a better uncertainty source than MC-Dropout?

For every step-matched checkpoint (arms none / classical / l1only / pix2pix,
N = 50, 100, 200, every available seed) on the SAME fixed test1 patches:
  * deterministic softmax, MC-Dropout (T passes), and -- per (arm, N) -- the
    seed ensemble (deterministic members, deep-ensemble style)
  * accuracy:    mIoU
  * calibration: ECE (10 bins), NLL, Brier
  * usefulness:  AUROC of error detection (pixel misclassified?) using
                 predictive entropy and, where defined, mutual information
                 (MC-Dropout epistemic / ensemble disagreement);
                 AURC (area under the risk-coverage curve) and accuracy at 80% coverage
Everything is accumulated in streaming histograms, so memory stays small.

    python scripts/11_uncertainty_quality.py --arms none --budgets 100 --seeds 0 --tag timing
    python scripts/11_uncertainty_quality.py                 # everything, resumable
    python scripts/11_uncertainty_quality.py --summarize-only

Needs the checkpoints written by 09_step_matched_control.py
(unet_<arm>_b<N>_s<seed>_stepmatched.pt). Results -> results/expB_quality/.
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
from geoseg.metrics import confusion_matrix, metrics_from_cm
from geoseg.models import UNet, enable_mc_dropout
from geoseg.train import device

EPS = 1e-12
P = C.PATCH_SIZE
K = C.N_CLASSES
NB_ECE = C.CALIBRATION_BINS
NB_H = 8000
HI = float(np.log(K))
SFX = "_stepmatched"
ARMS = ["none", "classical", "l1only", "pix2pix"]


def ent(p, dim=1):
    return -(p * torch.log(p + EPS)).sum(dim)


def pad_batch(xb):
    xb = torch.from_numpy(xb).float().unsqueeze(1)
    return torch.nn.functional.pad(
        xb, (0, C.IMG_SIZE - xb.shape[3], 0, C.IMG_SIZE - xb.shape[2]), mode="reflect")


class Scorer:
    """Streaming calibration + error-detection metrics for one probability source."""
    def __init__(self, dev):
        self.dev = dev
        z = lambda n: torch.zeros(n, dtype=torch.float64, device=dev)
        self.nll = torch.zeros((), dtype=torch.float64, device=dev)
        self.brier = torch.zeros((), dtype=torch.float64, device=dev)
        self.n = 0
        self.cm = np.zeros((K, K), dtype=np.int64)
        self.ece_n, self.ece_conf, self.ece_acc = z(NB_ECE), z(NB_ECE), z(NB_ECE)
        self.hc = {"ent": z(NB_H), "mi": z(NB_H)}
        self.he = {"ent": z(NB_H), "mi": z(NB_H)}
        self.has_mi = False
        self.epi_sum = 0.0

    def update(self, p, y, mi=None):
        """p: (B,K,H,W) predictive probs; y: (B,H,W) long; mi: (B,H,W) or None."""
        conf, pred = p.max(1)
        wrong = pred != y
        ptrue = p.gather(1, y.unsqueeze(1)).squeeze(1)
        self.nll += (-torch.log(ptrue.clamp_min(EPS))).sum().double()
        oh = torch.nn.functional.one_hot(y, K).permute(0, 3, 1, 2).float()
        self.brier += ((p - oh) ** 2).sum(1).sum().double()
        self.n += y.numel()
        self.cm += confusion_matrix(pred.cpu().numpy().ravel(), y.cpu().numpy().ravel(), K)
        b = (conf * NB_ECE).long().clamp(0, NB_ECE - 1).flatten()
        self.ece_n += torch.bincount(b, minlength=NB_ECE).double()
        self.ece_conf += torch.bincount(b, weights=conf.flatten().double(), minlength=NB_ECE)
        self.ece_acc += torch.bincount(b, weights=(~wrong).flatten().double(), minlength=NB_ECE)
        self._hist("ent", ent(p), wrong)
        if mi is not None:
            self.has_mi = True
            self.epi_sum += float(mi.sum())
            self._hist("mi", mi, wrong)

    def _hist(self, name, s, wrong):
        idx = (s / HI * NB_H).long().clamp(0, NB_H - 1).flatten()
        w = wrong.flatten()
        self.he[name] += torch.bincount(idx[w], minlength=NB_H).double()
        self.hc[name] += torch.bincount(idx[~w], minlength=NB_H).double()

    @staticmethod
    def _auroc(hc, he):
        hc, he = hc.cpu().numpy(), he.cpu().numpy()
        C_, E_ = hc.sum(), he.sum()
        if C_ == 0 or E_ == 0:
            return float("nan")
        below = np.cumsum(hc) - hc            # correct pixels with strictly lower score
        return float((he * (below + 0.5 * hc)).sum() / (C_ * E_))

    def result(self):
        m = self.ece_n.cpu().numpy(); ok = m > 0
        conf = self.ece_conf.cpu().numpy(); acc = self.ece_acc.cpu().numpy()
        ece = float(np.sum((m[ok] / m.sum()) * np.abs(acc[ok] / m[ok] - conf[ok] / m[ok])))
        hc, he = self.hc["ent"].cpu().numpy(), self.he["ent"].cpu().numpy()
        tot = hc + he
        cum_n, cum_e = np.cumsum(tot), np.cumsum(he)
        risk = cum_e / np.maximum(cum_n, 1)
        cov = cum_n / cum_n[-1]
        w = tot / tot.sum()
        aurc = float((risk * w).sum())
        k80 = int(np.searchsorted(cov, 0.8))
        r = {"mIoU": metrics_from_cm(self.cm)["mean_iou"],
             "pixel_acc": float(np.trace(self.cm) / self.cm.sum()),
             "ece": ece, "nll": float(self.nll / self.n), "brier": float(self.brier / self.n),
             "auroc_ent": self._auroc(self.hc["ent"], self.he["ent"]),
             "aurc": aurc, "acc_at_80cov": float(1 - risk[min(k80, NB_H - 1)])}
        if self.has_mi:
            r["auroc_mi"] = self._auroc(self.hc["mi"], self.he["mi"])
            r["mean_mi"] = self.epi_sum / self.n
        return r


def load_model(tag, dev):
    m = UNet().to(dev)
    m.load_state_dict(torch.load(C.CKPT_DIR / f"unet_{tag}.pt", map_location=dev))
    return m


@torch.no_grad()
def det_probs_batch(model, xb):
    model.eval()
    return torch.softmax(model(xb), 1)[:, :, :P, :P]


def run_checkpoint(tag, X, Yc, dev, passes, batch, keep_det=True):
    return run_model(load_model(tag, dev), tag, X, Yc, dev, passes, batch, keep_det)


@torch.no_grad()
def run_model(model, tag, X, Yc, dev, passes, batch, keep_det=True):
    """Backbone-agnostic: any nn.Module whose dropout layers are nn.Dropout/Dropout2d."""
    sd, sm = Scorer(dev), Scorer(dev)
    store = np.zeros((len(X), K, P, P), dtype=np.float16) if keep_det else None
    torch.manual_seed(0)
    for i in range(0, len(X), batch):
        xb = pad_batch(X[i:i + batch]).to(dev)
        y = torch.from_numpy(Yc[i:i + batch]).to(dev)
        pd_ = det_probs_batch(model, xb)
        sd.update(pd_, y)
        if keep_det:
            store[i:i + batch] = pd_.cpu().numpy().astype(np.float16)
        enable_mc_dropout(model)
        sp = sh = None
        for _ in range(passes):
            p = torch.softmax(model(xb), 1)[:, :, :P, :P]
            h = ent(p)
            sp = p if sp is None else sp + p
            sh = h if sh is None else sh + h
        mp = sp / passes
        mi = torch.clamp(ent(mp) - sh / passes, min=0)
        sm.update(mp, y, mi=mi)
    return {"tag": tag, "passes": passes, "n_patches": int(len(X)),
            "det": sd.result(), "mc": sm.result()}, store


@torch.no_grad()
def ensemble_result(stores, Yc, dev, batch=8):
    sc = Scorer(dev)
    for i in range(0, len(Yc), batch):
        ps = torch.stack([torch.from_numpy(s[i:i + batch].astype(np.float32)) for s in stores]).to(dev)
        mp = ps.mean(0)
        mi = torch.clamp(ent(mp) - ent(ps, dim=2).mean(0), min=0)
        sc.update(mp, torch.from_numpy(Yc[i:i + batch]).to(dev), mi=mi)
    return sc.result()


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(d, size=(n, len(d))).mean(1), [2.5, 97.5])


def summarize(out_dir):
    import pandas as pd
    from scipy.stats import wilcoxon
    rows, ens = [], []
    for f in sorted(out_dir.glob("ckpt_*.json")):
        r = json.load(open(f)); arm, b, s = r["tag"].replace(SFX, "").split("_")
        for method in ("det", "mc"):
            rows.append({"arm": arm, "N": int(b[1:]), "seed": int(s[1:]), "method": method, **r[method]})
    for f in sorted(out_dir.glob("ens_*.json")):
        r = json.load(open(f)); arm, b = r["group"].split("_")
        ens.append({"arm": arm, "N": int(b[1:]), "n_members": r["n_members"], **r["ensemble"]})
    if not rows:
        sys.exit("no per-checkpoint results found")
    df = pd.DataFrame(rows); df.to_csv(out_dir / "quality_per_checkpoint.csv", index=False)
    cols = ["mIoU", "ece", "nll", "brier", "auroc_ent", "auroc_mi", "aurc", "acc_at_80cov"]
    pd.set_option("display.width", 200)
    for method, label in (("det", "deterministic softmax"), ("mc", "MC-Dropout")):
        t = df[df.method == method].groupby(["N", "arm"])[cols].mean()
        print(f"\n=== {label}: mean over seeds ===\n{t.round(4).to_string()}")
    if ens:
        e = pd.DataFrame(ens); e.to_csv(out_dir / "quality_ensembles.csv", index=False)
        print("\n=== Seed ensemble (deterministic members) ===")
        print(e.set_index(["N", "arm"])[["n_members"] + cols].round(4).to_string())
        # ensemble vs average single member
        print("\n=== Ensemble minus mean single member (same arm, N) ===")
        for _, r in e.iterrows():
            s = df[(df.method == "det") & (df.arm == r.arm) & (df.N == r.N)]
            print(f"  N={r.N:<4d} {r.arm:9s} " + "  ".join(
                f"{c}:{r[c] - s[c].mean():+.4f}" for c in ("mIoU", "ece", "nll", "auroc_ent", "aurc")))
    print("\n=== Paired differences over seeds (A - B), 95% bootstrap CI ===")
    pairs = [("pix2pix", "none"), ("pix2pix", "classical"), ("pix2pix", "l1only"),
             ("l1only", "classical"), ("classical", "none")]
    for method in ("det", "mc"):
        for N in sorted(df.N.unique()):
            d = df[(df.method == method) & (df.N == N)]
            for A, B in pairs:
                a = d[d.arm == A].set_index("seed"); b = d[d.arm == B].set_index("seed")
                seeds = sorted(set(a.index) & set(b.index))
                if len(seeds) < 3:
                    continue
                line = f"  [{method}] N={N:<4d} {A:9s}-{B:9s} (n={len(seeds)}): "
                for c in ("mIoU", "ece", "nll", "auroc_ent", "aurc"):
                    diff = (a.loc[seeds, c] - b.loc[seeds, c]).values
                    lo, hi = boot(diff)
                    line += f"{c} {diff.mean():+.4f}[{lo:+.4f},{hi:+.4f}]  "
                print(line)
    mc = df[df.method == "mc"].dropna(subset=["auroc_mi"])
    if len(mc):
        d = (mc.auroc_mi - mc.auroc_ent).values
        print(f"\nMC-Dropout epistemic MI vs predictive entropy as error detector: "
              f"mean AUROC diff {d.mean():+.4f} over {len(d)} checkpoints; "
              f"MI better in {(d > 0).sum()}/{len(d)} (negative = entropy is the better signal)")
    print(f"\nCSV files in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--passes", type=int, default=C.MC_PASSES)
    ap.add_argument("--n-patches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="test1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    out_dir = C.RESULTS_DIR / ("expB_quality" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        return summarize(out_dir)

    d = np.load(C.PATCH_CACHE / f"{args.split}_patches.npz")
    sel = np.linspace(0, len(d["X"]) - 1, min(args.n_patches, len(d["X"]))).astype(int)
    X = d["X"][sel].astype(np.float32); Yc = d["Y"][sel].astype(np.int64)
    np.save(out_dir / "patch_indices.npy", sel)
    dev = device()
    print(f"device={dev}  patches={len(X)}  passes={args.passes}", flush=True)

    for arm in args.arms:
        for b in args.budgets:
            group = f"{arm}_b{b}"
            stores, n_ok = [], 0
            avail = [s for s in args.seeds if (C.CKPT_DIR / f"unet_{arm}_b{b}_s{s}{SFX}.pt").exists()]
            ef = out_dir / f"ens_{group}.json"
            ens_done = ef.exists() and json.load(open(ef)).get("n_members") == len(avail)
            for s in args.seeds:
                tg = f"{arm}_b{b}_s{s}{SFX}"
                if not (C.CKPT_DIR / f"unet_{tg}.pt").exists():
                    continue
                jf = out_dir / f"ckpt_{tg}.json"
                need_ens = not ens_done
                if jf.exists() and not need_ens:
                    print(f"[skip] {tg}"); n_ok += 1; continue
                if jf.exists():          # metrics done; only the det probs are needed for the ensemble
                    m = load_model(tg, dev); st = np.zeros((len(X), K, P, P), dtype=np.float16)
                    with torch.no_grad():
                        for i in range(0, len(X), args.batch):
                            st[i:i + args.batch] = det_probs_batch(
                                m, pad_batch(X[i:i + args.batch]).to(dev)).cpu().numpy().astype(np.float16)
                    stores.append(st); n_ok += 1; print(f"[ens-only] {tg}"); continue
                print(f"=== {tg} ===", flush=True)
                rec, st = run_checkpoint(tg, X, Yc, dev, args.passes, args.batch)
                json.dump(rec, open(jf, "w"), indent=2)
                stores.append(st); n_ok += 1
                print(f"  det: mIoU {rec['det']['mIoU']:.4f} ECE {rec['det']['ece']:.4f} "
                      f"AUROC {rec['det']['auroc_ent']:.3f} | MC: ECE {rec['mc']['ece']:.4f} "
                      f"AUROC(ent) {rec['mc']['auroc_ent']:.3f} AUROC(MI) {rec['mc']['auroc_mi']:.3f}", flush=True)
            if len(stores) >= 3 and not ens_done:
                r = ensemble_result(stores, Yc, dev)
                json.dump({"group": group, "n_members": len(stores), "ensemble": r}, open(ef, "w"), indent=2)
                print(f"  ensemble {group} ({len(stores)} members): mIoU {r['mIoU']:.4f} "
                      f"ECE {r['ece']:.4f} AUROC(MI) {r['auroc_mi']:.3f}", flush=True)
            del stores
    summarize(out_dir)


if __name__ == "__main__":
    main()
