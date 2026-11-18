"""Network representation and data loading utilities.

Owner: Member A & B
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from pathsnap.geo.projection import Proj
from pathsnap.ingest.parsers import load_trip


class Network:
    """Road network graph with spatial KD-Tree and successor indexing."""

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray, proj: Proj) -> None:
        self.proj = proj
        self.xy = proj(nodes[:, 0], nodes[:, 1])
        self.arcs = arcs
        self.A = self.xy[arcs[:, 0]]
        self.B = self.xy[arcs[:, 1]]
        self.len = np.sqrt(((self.B - self.A) ** 2).sum(axis=1))
        d = self.B - self.A
        self.bear = np.arctan2(d[:, 0], d[:, 1])  # Radians from north

        # Successor lists for network traversal
        n_nodes = len(nodes)
        order = np.argsort(arcs[:, 0], kind="stable")
        self.out_start = np.searchsorted(arcs[order, 0], np.arange(n_nodes + 1))
        self.out_arcs = order
        self._tree = None
        self._ids = None

    def succ(self, a: int) -> np.ndarray:
        """Return array of outgoing arc indices starting at the destination of arc a."""
        v = self.arcs[a, 1]
        return self.out_arcs[self.out_start[v]:self.out_start[v + 1]]

    def tree(self) -> cKDTree:
        """Build and cache a cKDTree over discretized segment points for fast candidate retrieval."""
        if self._tree is None:
            n = np.maximum(1, np.ceil(self.len / 20.0).astype(int)) + 1
            ids = np.repeat(np.arange(len(self.len)), n)
            first = np.cumsum(n) - n
            k = np.arange(len(ids)) - np.repeat(first, n)
            f = (k / np.repeat(n - 1, n))[:, None]
            pts = self.A[ids] + (self.B[ids] - self.A[ids]) * f
            self._ids = ids
            self._tree = cKDTree(pts)
        return self._tree
