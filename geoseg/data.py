"""F3 Netherlands data handling.

The Alaudah et al. (2019) release ships four arrays:
    train/train_seismic.npy       (401, 701, 255)   inline, crossline, depth
    train/train_labels.npy        (401, 701, 255)   int, 0..5
    test_once/test1_seismic.npy   (601, 200, 255)
    test_once/test1_labels.npy
    test_once/test2_seismic.npy   (200, 601, 255)
    test_once/test2_labels.npy

Nothing below hard-codes those shapes; we infer them at load time so the code
still works if you sub-sample or re-export the volumes.

Split discipline (important for the the benchmark's credibility):
  * Patches are cut from the TRAIN volume only.
  * A fixed 15% of train inlines is held out as a validation set for early
    stopping. Those inlines never appear in any label budget.
  * test1/test2 are touched exactly once, at the end, by evaluate().
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C


# --------------------------------------------------------------------------
# Volume loading
# --------------------------------------------------------------------------
def load_volume(split: str):
    """split in {'train', 'test1', 'test2'} -> (seismic, labels) float32/int64."""
    if split == "train":
        s = np.load(C.TRAIN_DIR / "train_seismic.npy")
        l = np.load(C.TRAIN_DIR / "train_labels.npy")
    elif split in ("test1", "test2"):
        s = np.load(C.TEST_DIR / f"{split}_seismic.npy")
        l = np.load(C.TEST_DIR / f"{split}_labels.npy")
    else:
        raise ValueError(f"unknown split {split!r}")
    # The released volumes are float64 (~573 MB for train). Cast on load so we
    # never hold two copies of a double-precision volume. Labels stay uint8:
    # as int64 the cached train patches alone would occupy ~1.7 GB for no gain,
    # since PatchDataset casts to long per sample anyway.
    return s.astype(np.float32, copy=False), l.astype(np.uint8, copy=False)


def normalise(seismic: np.ndarray, stats: tuple[float, float] | None = None):
    """Zero-mean/unit-std using TRAIN statistics only (passed in for test)."""
    if stats is None:
        mu, sd = float(seismic.mean()), float(seismic.std())
    else:
        mu, sd = stats
    return (seismic - mu) / (sd + 1e-8), (mu, sd)


# --------------------------------------------------------------------------
# Patch extraction
# --------------------------------------------------------------------------
def _sections(vol: np.ndarray, axis: int):
    """Yield 2D sections along `axis` (0 = inline, 1 = crossline)."""
    n = vol.shape[axis]
    for i in range(n):
        yield i, (vol[i] if axis == 0 else vol[:, i])


def choose_axis(vol, patch=C.PATCH_SIZE):
    """Pick the slicing axis whose 2D sections are widest laterally.

    F3 volumes are not all oriented the same way: train is (401, 701, 255) and
    test1 is (200, 701, 255), so inline slicing gives 701-wide sections. But
    test2 is (601, 200, 255) -- inline slicing there gives sections only 200
    wide, narrower than the 255-pixel patch, and every section would be
    discarded. Slicing test2 along axis 1 gives 601-wide sections instead.
    Choosing automatically avoids a silent empty-dataset failure.
    """
    lat0, lat1 = vol.shape[1], vol.shape[0]
    return 0 if lat0 >= lat1 else 1


def _pad_section(sec_s, sec_l, patch):
    """Reflect-pad a section that is narrower than one patch."""
    need = patch - sec_s.shape[0]
    sec_s = np.pad(sec_s, ((0, need), (0, 0)), mode="reflect")
    sec_l = np.pad(sec_l, ((0, need), (0, 0)), mode="edge")
    return sec_s, sec_l


def extract_patches(seismic, labels, axis="auto", indices=None,
                    patch=C.PATCH_SIZE, stride=C.PATCH_STRIDE):
    """Tile 2D sections into (patch x depth) windows.

    axis="auto" picks the wider lateral axis (see choose_axis).
    Returns X (N,H,W), Y (N,H,W), meta (section_idx, offset), and the axis used.
    """
    if axis == "auto":
        axis = choose_axis(seismic, patch)
    X, Y, meta = [], [], []
    n_sections = seismic.shape[axis]
    idx_iter = range(n_sections) if indices is None else indices
    for i in idx_iter:
        sec_s = seismic[i] if axis == 0 else seismic[:, i]
        sec_l = labels[i] if axis == 0 else labels[:, i]
        if sec_s.shape[0] < patch:
            sec_s, sec_l = _pad_section(sec_s, sec_l, patch)
        lat = sec_s.shape[0]
        for off in range(0, lat - patch + 1, stride):
            X.append(sec_s[off:off + patch, :])
            Y.append(sec_l[off:off + patch, :])
            meta.append((int(i), int(off)))
        # always include the flush-right window so the section edge is covered
        if lat > patch and (lat - patch) % stride != 0:
            X.append(sec_s[lat - patch:, :])
            Y.append(sec_l[lat - patch:, :])
            meta.append((int(i), int(lat - patch)))
    if not X:
        raise RuntimeError("no patches extracted -- check patch/stride vs volume shape")
    return np.stack(X), np.stack(Y), np.array(meta), axis


def train_val_section_split(n_sections: int, val_frac=0.15, seed=1234):
    """Contiguous-block validation split, so val sections are not adjacent to
    train sections (adjacent seismic inlines are near-duplicates -- a random
    split leaks and inflates every number)."""
    n_val = max(1, int(round(n_sections * val_frac)))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_sections - n_val))
    val = list(range(start, start + n_val))
    # drop a 3-section buffer either side of the val block
    buf = set(range(max(0, start - 3), min(n_sections, start + n_val + 3)))
    train = [i for i in range(n_sections) if i not in buf]
    return train, val


def sample_label_budget(n_available: int, budget: int, seed: int):
    """Pick `budget` patch indices. Nested: the budget-50 set is a subset of
    budget-100, which makes the learning curve monotone in *data*, not in
    *which* data -- removing a large confound."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_available)
    return np.sort(perm[:min(budget, n_available)])


# --------------------------------------------------------------------------
# Torch datasets
# --------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, X, Y, augment=None, img_size=C.IMG_SIZE):
        self.X, self.Y = X, Y
        self.augment = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i].copy(), self.Y[i].copy()
        if self.augment is not None:
            x, y = self.augment(x, y)
        x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(y)).long()
        x, y = _pad_to(x, y, self.img_size)
        return x, y


def _pad_to(x, y, size):
    """Reflect-pad H and W up to `size` (F3 patches are 255x255 -> 256x256)."""
    _, h, w = x.shape
    ph, pw = size - h, size - w
    if ph or pw:
        x = torch.nn.functional.pad(x[None], (0, pw, 0, ph), mode="reflect")[0]
        y = torch.nn.functional.pad(y[None, None].float(), (0, pw, 0, ph),
                                    mode="replicate")[0, 0].long()
    return x, y


def class_pixel_counts(labels: np.ndarray, n_classes=C.N_CLASSES):
    return np.bincount(labels.ravel().astype(np.int64), minlength=n_classes)
