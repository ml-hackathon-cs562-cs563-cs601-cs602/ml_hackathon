"""Emission log-likelihood computation for HMM map matching.

Owner: Member B
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def prob_matrix(tab: Dict[str, Any], p: np.ndarray) -> np.ndarray:
    """Format flat classifier probabilities into an (N, K) matrix."""
    candidates = np.asarray(tab["cand"])
    probabilities = np.asarray(p, dtype=np.float64)

    if candidates.ndim != 2:
        raise ValueError("tab['cand'] must be a 2D array.")

    if probabilities.ndim != 1:
        raise ValueError("p must be a 1D probability array.")

    if len(probabilities) != len(tab["ri"]) or len(probabilities) != len(tab["rk"]):
        raise ValueError("Probability and candidate index lengths must match.")

    if np.any(~np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    n_rows, n_candidates = candidates.shape
    result = np.full((n_rows, n_candidates), 1e-6, dtype=np.float64)

    row_indices = np.asarray(tab["ri"], dtype=int)
    candidate_indices = np.asarray(tab["rk"], dtype=int)

    if np.any((row_indices < 0) | (row_indices >= n_rows)):
        raise ValueError("Row indices are outside the candidate matrix.")

    if np.any((candidate_indices < 0) | (candidate_indices >= n_candidates)):
        raise ValueError("Candidate indices are outside the candidate matrix.")

    result[row_indices, candidate_indices] = probabilities
    return result


def compute_emissions(
    pm: np.ndarray,
    cand: np.ndarray,
    w_e: float = 0.357,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calculate weighted classifier log-probability emission scores."""
    pm = np.asarray(pm, dtype=np.float64)
    cand = np.asarray(cand)

    if pm.ndim != 2 or cand.ndim != 2:
        raise ValueError("pm and cand must be 2D arrays.")

    if pm.shape != cand.shape:
        raise ValueError("pm and cand must have identical shapes.")

    if not np.isfinite(w_e) or not 0.0 <= w_e <= 1.0:
        raise ValueError("w_e must be between 0 and 1.")

    if not np.isfinite(eps) or not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5.")

    if np.any(~np.isfinite(pm)):
        raise ValueError("Probabilities must be finite.")

    if np.any((pm < 0) | (pm > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    emissions = w_e * np.log(np.clip(pm, eps, 1.0))
    emissions[cand < 0] = -1e9

    return emissions
