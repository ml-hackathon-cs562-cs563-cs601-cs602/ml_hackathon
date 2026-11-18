"""Directed road-network transition modeling and hop-distance calculation.

Owner: Member B
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np


class Hop:
    """Calculate and cache directed node reachability within H hops."""

    def __init__(self, arcs: np.ndarray, H: int = 4) -> None:
        arcs = np.asarray(arcs)

        if arcs.ndim != 2 or arcs.shape[1] != 2:
            raise ValueError("arcs must have shape (E, 2).")

        if not isinstance(H, (int, np.integer)) or H < 0:
            raise ValueError("H must be a non-negative integer.")

        if not np.issubdtype(arcs.dtype, np.integer):
            if not np.all(np.isfinite(arcs)) or not np.all(arcs == np.floor(arcs)):
                raise ValueError("Node IDs must be finite integers.")

        arcs = arcs.astype(np.int64)

        if np.any(arcs < 0):
            raise ValueError("Node IDs must be non-negative.")

        self.arcs = arcs
        self.H = int(H)

        # Dictionary adjacency supports sparse and non-contiguous node IDs.
        self.adj: Dict[int, List[int]] = {}

        for source, destination in arcs:
            source = int(source)
            destination = int(destination)

            self.adj.setdefault(source, []).append(destination)

        self.memo: Dict[int, Dict[int, int]] = {}

    def reach(self, u: int) -> Dict[int, int]:
        """
        Return {reachable_node: minimum_hops} from node u.

        The search follows directed edges only and includes the starting
        node with distance zero.
        """

        if not isinstance(u, (int, np.integer)) or u < 0:
            raise ValueError("Starting node must be a non-negative integer.")

        u = int(u)

        if u in self.memo:
            return self.memo[u]

        distances = {u: 0}
        queue = deque([u])

        while queue:
            current = queue.popleft()
            current_distance = distances[current]

            if current_distance >= self.H:
                continue

            for neighbour in self.adj.get(current, []):
                if neighbour not in distances:
                    distances[neighbour] = current_distance + 1
                    queue.append(neighbour)

        self.memo[u] = distances
        return distances
