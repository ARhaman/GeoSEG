"""Evaluation metrics.

Beyond mIoU (which the paper treats a necessary but
boundary-blind baseline) we compute:
  * per-class IoU and Dice
  * Boundary F1 at a tolerance of `tol` pixels -- catches the errors mIoU hides
  * Expected Calibration Error + reliability curve -- for Figure 9
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config as C


# --------------------------------------------------------------------------
# Confusion-matrix based
# --------------------------------------------------------------------------
def confusion_matrix(pred, true, n_classes=C.N_CLASSES):
    k = (true >= 0) & (true < n_classes)
    return np.bincount(
        n_classes * true[k].astype(int) + pred[k].astype(int),
        minlength=n_classes ** 2,
    ).reshape(n_classes, n_classes)


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    present = (cm.sum(1) > 0)
    return {
        "iou_per_class": np.where(present, iou, np.nan).tolist(),
        "dice_per_class": np.where(present, dice, np.nan).tolist(),
        "precision_per_class": np.where(present, prec, np.nan).tolist(),
        "recall_per_class": np.where(present, rec, np.nan).tolist(),
        "mean_iou": float(np.nanmean(iou[present])),
        "mean_dice": float(np.nanmean(dice[present])),
        "pixel_acc": float(tp.sum() / cm.sum()),
        # class-frequency-weighted accuracy: reported alongside mIoU because
        # F3 is heavily imbalanced and the two can move in opposite directions
        "freq_weighted_iou": float(
            np.nansum((cm.sum(1) / cm.sum()) * np.where(present, iou, 0.0))
        ),
    }


# --------------------------------------------------------------------------
# Boundary F1
# --------------------------------------------------------------------------
def _boundaries(mask):
    """Binary map of pixels adjacent to a different label."""
    b = np.zeros_like(mask, dtype=bool)
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    b[1:, :] |= mask[:-1, :] != mask[1:, :]
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    b[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return b


def boundary_f1(pred, true, tol=2):
    """F1 between predicted and true boundary pixels, allowing `tol` px slack.
    Reported per 2D section then averaged."""
    bp, bt = _boundaries(pred), _boundaries(true)
    if not bp.any() and not bt.any():
        return 1.0
    if not bp.any() or not bt.any():
        return 0.0
    dt_true = ndimage.distance_transform_edt(~bt)
    dt_pred = ndimage.distance_transform_edt(~bp)
    precision = (dt_true[bp] <= tol).mean()
    recall = (dt_pred[bt] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration(probs, true, n_bins=C.CALIBRATION_BINS):
    """probs: (N, C) softmax; true: (N,) int.
    Returns ECE plus the reliability curve for Figure 9."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == true).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            curve.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                          "conf": np.nan, "acc": np.nan})
            continue
        c, a = conf[m].mean(), correct[m].mean()
        ece += (m.sum() / len(conf)) * abs(a - c)
        curve.append({"bin_lo": float(lo), "bin_hi": float(hi),
                      "n": int(m.sum()), "conf": float(c), "acc": float(a)})
    return {"ece": float(ece), "reliability": curve}


def entropy(probs, axis=0, eps=1e-12):
    return -(probs * np.log(probs + eps)).sum(axis)


def decompose_uncertainty(prob_samples):
    """prob_samples: (T, C, H, W) softmax outputs from T stochastic passes.

    Total predictive entropy = aleatoric + epistemic (mutual information):
        H[E_t p]              -> total
        E_t H[p]              -> aleatoric (irreducible data noise)
        H[E_t p] - E_t H[p]   -> epistemic (reducible; shrinks with more data)

    This decomposition is what makes Figure 8 panels C and D meaningfully
    different rather than two views of the same number.
    """
    mean_p = prob_samples.mean(0)
    total = entropy(mean_p, axis=0)
    aleatoric = np.stack([entropy(p, axis=0) for p in prob_samples]).mean(0)
    epistemic = np.clip(total - aleatoric, 0, None)
    return mean_p, total, aleatoric, epistemic
