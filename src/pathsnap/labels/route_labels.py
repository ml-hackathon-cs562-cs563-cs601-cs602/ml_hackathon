"""Route-to-observation alignment and label generation.

Owner: Member B
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from pathsnap.geo.distances import seg_dist
from pathsnap.ingest.data_loader import Network


def align_labels(
    net: Network,
    P: np.ndarray,
    route: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align sequence of GPS points to ordered reference route using monotone DP.
    
    Each GPS observation is assigned to a reference route edge. Route indices are
    constrained to be non-decreasing (allowing forward progression and skipping up to 3 arcs).
    
    Args:
        net: Road network instance.
        P: (N, 2) Projected GPS coordinates.
        route: (M,) Ordered sequence of road arc IDs.
        
    Returns:
        matched_arcs: (N,) Reference road arc ID aligned to each GPS point.
        matched_costs: (N,) Distance cost in metres.
    """
    N, M = len(P), len(route)
    if N == 0 or M == 0:
        return np.full(N, -1, dtype=int), np.full(N, np.inf, dtype=np.float32)

    A, B = net.A[route], net.B[route]
    cost = np.empty((N, M), dtype=np.float32)

    # Compute distances in chunks for vectorization
    for s in range(0, N, 512):
        chunk = P[s:s + 512]
        cost[s:s + 512] = [
            seg_dist(np.repeat(q[None], M, axis=0), A, B)[0] for q in chunk
        ]

    cost = np.minimum(cost, 200.0)
    dp = np.full(M, np.inf, dtype=np.float32)
    dp[:min(4, M)] = cost[0, :min(4, M)]
    bp = np.zeros((N, M), dtype=np.int8)

    for i in range(1, N):
        best = dp.copy()
        arg = np.zeros(M, dtype=np.int8)
        for s in (1, 2, 3):
            if s >= M:
                continue
            sh = np.full(M, np.inf, dtype=np.float32)
            sh[s:] = dp[:-s]
            m = sh < best
            best = np.where(m, sh, best)
            arg = np.where(m, s, arg)
        dp = best + cost[i]
        bp[i] = arg

    j = int(np.argmin(dp))
    idx = np.zeros(N, dtype=int)
    for i in range(N - 1, -1, -1):
        idx[i] = j
        j -= int(bp[i, j])

    matched_arcs = route[idx]
    matched_costs = cost[np.arange(N), idx]
    return matched_arcs, matched_costs
