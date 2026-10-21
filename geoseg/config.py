"""Central configuration for the F3 generative-augmentation benchmark.

Paths are read from environment variables so no file needs editing:

    GEOSEG_DATA_ROOT   folder holding the F3 benchmark `train/` and `test_once/` folders
    GEOSEG_RUN_ROOT    where patches, checkpoints, results and figures are written
    GEOSEG_TRAIN_DIR / GEOSEG_TEST_DIR   optional, if the two folders are not directly
                                         under GEOSEG_DATA_ROOT
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("GEOSEG_DATA_ROOT", "data/f3"))
RUN_ROOT = Path(os.environ.get("GEOSEG_RUN_ROOT", "runs"))

TRAIN_DIR = Path(os.environ.get("GEOSEG_TRAIN_DIR", DATA_ROOT / "train"))
TEST_DIR = Path(os.environ.get("GEOSEG_TEST_DIR", DATA_ROOT / "test_once"))

PATCH_CACHE = RUN_ROOT / "patches"        # extracted 2D patches (.npz)
CKPT_DIR = RUN_ROOT / "checkpoints"
RESULTS_DIR = RUN_ROOT / "results"        # per-run metrics .json
FIGURE_DIR = RUN_ROOT / "figures"
SYNTH_DIR = RUN_ROOT / "synthetic"        # pix2pix-generated patches

# --------------------------------------------------------------------------
# Dataset definition (Alaudah et al. 2019, F3 Netherlands)
# --------------------------------------------------------------------------
N_CLASSES = 6
CLASS_NAMES = [
    "Upper N.S.",
    "Middle N.S.",
    "Lower N.S.",
    "Rijnland/Chalk",
    "Scruff",
    "Zechstein",
]
# Colour-blind-safe qualitative palette (Okabe-Ito), used in every figure.
CLASS_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
]

PATCH_SIZE = 255           # depth axis of F3 is 255 samples; keep square-ish
PATCH_STRIDE = 128         # lateral tiling stride. 128 yields ~1.7k train patches
                           # -- ample for a 500-patch budget, and keeps the cached
                           # arrays near 0.5 GB instead of 2.5 GB.
IMG_SIZE = 256             # everything is resized/padded to this before the net

# --------------------------------------------------------------------------
# Experiment A grid
# --------------------------------------------------------------------------
LABEL_BUDGETS = [50, 100, 200, 500]        # number of labelled 2D patches
AUG_ARMS = ["none", "classical", "pix2pix"]  # add "ldm" only if time allows
SEEDS = [0, 1, 2]          # 3 seeds -> error bars on the learning curve

# --------------------------------------------------------------------------
# Training hyper-parameters (tuned for a 24 GB card)
# --------------------------------------------------------------------------
UNET_BASE = 32            # U-Net width; 32 fits 24 GB comfortably at 256^2
P2P_BASE = 64             # Pix2Pix width
BATCH_SIZE = 16
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.2            # kept active at inference for MC-Dropout
SYNTH_RATIO = 1.0          # 1 synthetic patch per real patch in the pix2pix arm
NUM_WORKERS = 4            # set to 0 if Windows dataloader workers misbehave
AMP = True                 # mixed precision

# --------------------------------------------------------------------------
# Experiment B (uncertainty)
# --------------------------------------------------------------------------
MC_PASSES = 50
CALIBRATION_BINS = 10


def ensure_dirs():
    for d in (RUN_ROOT, PATCH_CACHE, CKPT_DIR, RESULTS_DIR, FIGURE_DIR, SYNTH_DIR):
        d.mkdir(parents=True, exist_ok=True)
