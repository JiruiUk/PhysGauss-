"""Run an official entrypoint with an explicit seed and unchanged training code."""
import os
from pathlib import Path
import random
import runpy
import sys
import numpy as np
import torch

repo = Path(__file__).resolve().parents[2] / "third_party" / "PIDG"
sys.path.insert(0, str(repo))
os.chdir(repo)
from utils import general_utils

original_safe_state = general_utils.safe_state


def safe_state(silent):
    original_safe_state(silent)
    seed = int(os.environ.get("PIDG_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


general_utils.safe_state = safe_state
script = sys.argv[1]
if script not in {"train_pidg.py", "render.py", "metrics.py"}:
    raise ValueError("Only fixed official entrypoints are allowed")
sys.argv = [str(repo / script), *sys.argv[2:]]
runpy.run_path(str(repo / script), run_name="__main__")
