"""Utilities for reproducible experiment seeds."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for Python, NumPy, and PyTorch."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
