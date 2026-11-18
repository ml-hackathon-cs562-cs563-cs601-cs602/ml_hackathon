"""Road network candidate edge retrieval for GPS observations.

Owner: Member B
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from pathsnap.geo.distances import seg_dist
from pathsnap.ingest.data_loader import Network

K_CAND = 10
RADIUS = 80.0  # Metres


def candidates(
    net: Network,
    P: np.ndarray,
    k: int = K_CAND,
    radius: float = RADIUS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve up to K nearest candidate road arcs within search radius for each GPS point.
    
    Args:
        net: Road network instance.
        P: (N, 2) Projected GPS coordinates in metres.
        k: Maximum number of candidate edges per GPS point.
        radius: Spatial search radius in metres.
        
    Returns:
        cand: (N, k) integer array of candidate arc IDs (-1 padded).
        dist: (N, k) float array of perpendicular distances (inf padded).
        tpos: (N, k) float array of relative projection positions along the arc.
    """
    tree = net.tree()
    N = len(P)
    cand = -np.ones((N, k), dtype=int)
    dist = np.full((N, k), np.inf, dtype=np.float32)
    tpos = np.zeros((N, k), dtype=np.float32)

    hits = tree.query_ball_point(P, radius)
    for i, h in enumerate(hits):
        if not h:
            continue
        a = np.unique(net._ids[h])
        d, t = seg_dist(np.repeat(P[i:i + 1], len(a), axis=0), net.A[a], net.B[a])
        o = np.argsort(d)[:k]
        cand[i, :len(o)] = a[o]
        dist[i, :len(o)] = d[o]
        tpos[i, :len(o)] = t[o]

    return cand, dist, tpos


def candidate_miss_rate(cand: np.ndarray, reference_arcs: np.ndarray) -> float:
    """Compute candidate miss rate (fraction of observations where reference arc is absent)."""
    misses = [
        reference_arcs[i] not in cand[i]
        for i in range(len(reference_arcs))
        if reference_arcs[i] >= 0
    ]
    return float(np.mean(misses)) if misses else 0.0
