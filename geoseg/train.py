"""Training / evaluation engine for the segmentation backbone."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .data import PatchDataset
from .metrics import boundary_f1, confusion_matrix, metrics_from_cm
from .models import UNet


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, Y, augment, batch_size=C.BATCH_SIZE, shuffle=True,
                num_workers=C.NUM_WORKERS):
    ds = PatchDataset(X, Y, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle and len(ds) > batch_size)


def class_weights(Y, n_classes=C.N_CLASSES, scheme="inv_sqrt_freq"):
    """Shale-dominated F3 needs reweighting or minority facies collapse.
    inverse-sqrt-frequency is a middle ground between none and full inverse
    frequency (which destabilises training at small label budgets)."""
    counts = np.bincount(Y.ravel().astype(np.int64), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if scheme == "none":
        w = np.ones(n_classes)
    elif scheme == "inv_freq":
        w = counts.sum() / (n_classes * counts)
    else:
        w = np.sqrt(counts.sum() / (n_classes * counts))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, dev, n_classes=C.N_CLASSES, want_bf1=True):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    bf1s = []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        true = y.numpy()
        cm += confusion_matrix(pred.ravel(), true.ravel(), n_classes)
        if want_bf1:
            for p, t in zip(pred, true):
                bf1s.append(boundary_f1(p, t))
    out = metrics_from_cm(cm)
    out["boundary_f1"] = float(np.mean(bf1s)) if bf1s else None
    out["confusion_matrix"] = cm.tolist()
    return out


def train_model(X_tr, Y_tr, X_val, Y_val, augment, seed=0,
                epochs=C.EPOCHS, lr=C.LR, log_every=10, verbose=True, model_fn=None):
    set_seed(seed)
    dev = device()
    model = (model_fn() if model_fn is not None else UNet()).to(dev)
    w = class_weights(Y_tr).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=C.AMP and dev.type == "cuda")

    tr_loader = make_loader(X_tr, Y_tr, augment, shuffle=True)
    va_loader = make_loader(X_val, Y_val, None, shuffle=False)

    best = {"mean_iou": -1.0}
    best_state, history = None, []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=C.AMP and dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach().item()
            nb += 1
        sched.step()

        if ep % log_every == 0 or ep == epochs:
            va = evaluate(model, va_loader, dev, want_bf1=False)
            history.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                            "val_miou": va["mean_iou"]})
            if va["mean_iou"] > best["mean_iou"]:
                best = va
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep {ep:3d}  loss {tot/max(nb,1):.4f}  "
                      f"val mIoU {va['mean_iou']:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": best, "history": history,
                   "train_seconds": time.time() - t0,
                   "n_train_patches": int(len(X_tr))}


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
