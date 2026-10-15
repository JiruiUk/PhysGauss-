"""Execute an explicitly selected PIDG source tree with legacy seed semantics."""
import argparse
import os
from pathlib import Path
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("script", choices=("train_pidg.py", "render.py", "metrics.py"))
    args, remaining = parser.parse_known_args()
    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    sys.path.append(str(Path(__file__).resolve().parents[1] / "reproduction/scripts"))
    os.chdir(repo)
    import numpy as np
    import torch
    from utils import general_utils
    if Path(general_utils.__file__).resolve().parent.parent != repo:
        raise RuntimeError("Imported PIDG code from a different source tree")
    original = general_utils.safe_state

    def safe_state(silent):
        original(silent)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    general_utils.safe_state = safe_state
    sys.argv = [str(repo / args.script), *remaining]
    runpy.run_path(str(repo / args.script), run_name="__main__")


if __name__ == "__main__":
    main()
