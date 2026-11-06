"""Portable settings shared by the original experiment scripts."""

import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

if os.environ.get("GRAPHCONTROL_SHOW", "0") != "1":
    matplotlib.use("Agg")

seed = int(os.environ.get("GRAPHCONTROL_SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(int(os.environ.get("GRAPHCONTROL_THREADS", "4")))

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("GRAPHCONTROL_OUTPUT_DIR", ROOT / "outputs" / "latest"))


def figure_dir(part):
    path = OUTPUT / f"part{part}" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + os.sep


def experiment_root(part):
    return str(OUTPUT / f"part{part}" / "experiments")


def training_epochs():
    return int(os.environ.get("GRAPHCONTROL_EPOCHS", "50"))
