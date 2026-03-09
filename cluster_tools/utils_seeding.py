"""
Small reproducibility helpers.

Keep this dependency-free and safe to import from cluster scripts.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    Parameters
    ----------
    seed:
        Base RNG seed.
    deterministic:
        If True, enables deterministic (but sometimes slower) behavior in PyTorch.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True avoids hard-crashes if an op has no deterministic impl.
        torch.use_deterministic_algorithms(True, warn_only=True)