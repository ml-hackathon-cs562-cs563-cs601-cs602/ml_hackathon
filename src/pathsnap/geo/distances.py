"""Geometric distance calculations between GPS points and road segments.

Owner: Member B
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def seg_dist(
    P: np.ndarray, A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal/clipped distance from points P (N,2) to segments A->B (N,2).
    
    Returns:
        dist: Euclidean distance in metres from point P to the closest point on segment AB.
        t: Relative projection position along segment AB, normalized to [0, 1].
    """
    AB = B - A
    L2 = (AB ** 2).sum(axis=1)
    L2 = np.where(L2 == 0, 1e-9, L2)
    t = np.clip(((P - A) * AB).sum(axis=1) / L2, 0.0, 1.0)
    C = A + AB * t[:, None]
    dist = np.sqrt(((P - C) ** 2).sum(axis=1))
    return dist, t
