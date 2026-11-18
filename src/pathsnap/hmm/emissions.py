"""Emission log-likelihood computation for HMM map matching.

Owner: Member B
"""

from __future__ import annotations
from typing import Dict, Union
import numpy as np


def prob_matrix(tab: Dict[str, Any], p: np.ndarray) -> np.ndarray:
    """Format 1D prediction probabilities into an (N, K) observation-candidate matrix."""
    N, K = tab["cand"].shape
    pm = np.full((N, K), 1e-4, dtype=np.float32)
    pm[tab["ri"], tab["rk"]] = p
    return pm


def compute_emissions(
    pm: np.ndarray,
    cand: np.ndarray,
    w_e: float = 1.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute weighted emission log-scores, heavily penalizing absent candidates."""
    em = w_e * np.log(np.clip(pm, eps, 1.0))
    em[cand < 0] = -1e9
    return em
