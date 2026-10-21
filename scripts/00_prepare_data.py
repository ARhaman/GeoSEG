"""Step 0 -- verify the F3 volumes, extract 2D patches, cache them.

Run this first. It fails loudly if the data layout is wrong, which is much
better than discovering it three experiments later.

    python scripts/00_prepare_data.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg import config as C
from geoseg.data import (choose_axis, class_pixel_counts, extract_patches,
                         load_volume, normalise, train_val_section_split)


def check_files():
    need = [
        C.TRAIN_DIR / "train_seismic.npy",
        C.TRAIN_DIR / "train_labels.npy",
        C.TEST_DIR / "test1_seismic.npy",
        C.TEST_DIR / "test1_labels.npy",
        C.TEST_DIR / "test2_seismic.npy",
        C.TEST_DIR / "test2_labels.npy",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        print("MISSING FILES:")
        for p in missing:
            print("   ", p)
        print("\nUnzip both Google Drive archives so that you have:")
        print(f"   {C.TRAIN_DIR}\\train_seismic.npy  (and train_labels.npy)")
        print(f"   {C.TEST_DIR}\\test1_seismic.npy   (and test1/test2 labels)")
        print("\nThen re-run. If your folder names differ, edit "
              "geoseg/config.py -> DATA_ROOT / TRAIN_DIR / TEST_DIR.")
        sys.exit(1)
    print("All six .npy files found.")


def main():
    C.ensure_dirs()
    check_files()

    s, l = load_volume("train")
    print(f"\ntrain_seismic {s.shape} {s.dtype}   "
          f"range [{s.min():.3f}, {s.max():.3f}]")
    print(f"train_labels  {l.shape} {l.dtype}   "
          f"unique {np.unique(l).tolist()}")

    if s.shape != l.shape:
        sys.exit("seismic and label volumes disagree in shape -- stop and check")
    n_lab = int(l.max()) + 1
    if n_lab != C.N_CLASSES:
        print(f"WARNING: found {n_lab} classes, config says {C.N_CLASSES}")

    # Class balance over the whole training volume (feeds Figure 2)
    counts = class_pixel_counts(l)
    pct = 100 * counts / counts.sum()
    print("\nClass balance (train volume):")
    for i, (n, p) in enumerate(zip(counts, pct)):
        name = C.CLASS_NAMES[i] if i < len(C.CLASS_NAMES) else f"class {i}"
        print(f"  {i}  {name:<16s} {p:6.2f}%   ({n:,} px)")
    np.save(C.PATCH_CACHE / "class_counts_train.npy", counts)

    # Train/val split along whichever axis gives the wider sections
    ax = choose_axis(s)
    n_sec = s.shape[ax]
    tr_idx, va_idx = train_val_section_split(n_sec)
    print(f"\nslicing axis {ax} ({'inline' if ax == 0 else 'crossline'}), "
          f"{n_sec} sections")
    print(f"split: {len(tr_idx)} train / {len(va_idx)} val "
          f"(val block = {va_idx[0]}..{va_idx[-1]}, 3-section buffer dropped)")

    s_n, stats = normalise(s)
    np.save(C.PATCH_CACHE / "norm_stats.npy", np.array(stats))
    print(f"normalisation from TRAIN only: mean={stats[0]:.4f} std={stats[1]:.4f}")

    for name, idx in (("train", tr_idx), ("val", va_idx)):
        X, Y, meta, used = extract_patches(s_n, l, axis=ax, indices=idx)
        np.savez_compressed(C.PATCH_CACHE / f"{name}_patches.npz",
                            X=X.astype(np.float32), Y=Y.astype(np.uint8), meta=meta)
        print(f"  {name}: {X.shape[0]} patches of {X.shape[1]}x{X.shape[2]} "
              f"(axis {used})")
    del s_n

    # Test patches (test1 + test2), touched only at final evaluation
    for split in ("test1", "test2"):
        ts, tl = load_volume(split)
        print(f"\n{split}_seismic {ts.shape}  labels {tl.shape}")
        ts, _ = normalise(ts, stats)
        X, Y, meta, used = extract_patches(ts, tl, axis="auto")
        np.savez_compressed(C.PATCH_CACHE / f"{split}_patches.npz",
                            X=X.astype(np.float32), Y=Y.astype(np.uint8), meta=meta)
        print(f"  {split}: {X.shape[0]} patches (axis {used})")
        del ts, tl, X, Y

    print(f"\nDone. Cache -> {C.PATCH_CACHE}")


if __name__ == "__main__":
    main()
