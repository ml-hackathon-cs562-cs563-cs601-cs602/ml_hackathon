"""Reproducibility and random seed utilities for PathSnap.

Owner: Member C
Phase 1: Shared harness
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy and other libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional torch seed if installed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
