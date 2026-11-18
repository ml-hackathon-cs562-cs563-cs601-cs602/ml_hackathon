"""Road network transition modeling and hop distance calculation.

Owner: Member B
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional
import numpy as np


class Hop:
    """Computes and memoizes network hop-reachability up to H intermediate nodes."""

    def __init__(self, arcs: np.ndarray, H: int = 4) -> None:
        self.arcs = arcs
        self.H = H
        n_nodes = int(arcs.max()) + 1 if len(arcs) > 0 else 0
        self.adj: List[List[int]] = [[] for _ in range(n_nodes)]
        for f, t in arcs:
            if f < n_nodes:
                self.adj[f].append(int(t))
        self.memo: Dict[int, Dict[int, int]] = {}

    def reach(self, u: int) -> Dict[int, int]:
        """Return dict of {reachable_node: hop_distance} from node u within H hops."""
        r = self.memo.get(u)
        if r is None:
            r = {u: 0}
            q = deque([u])
            while q:
                x = q.popleft()
                d = r[x]
                if d >= self.H:
                    continue
                if x < len(self.adj):
                    for y in self.adj[x]:
                        if y not in r:
                            r[y] = d + 1
                            q.append(y)
            self.memo[u] = r
        return r
