"""Data parsers for PathSnap raw trip files (.track, .nodes, .arcs, .route).

Owner: Member A
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


def load_trip(
    root: Union[str, Path],
    tid: str,
    with_route: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Parse track, nodes, arcs, and optional reference route for a trip.
    
    Args:
        root: Directory containing trip folders.
        tid: Trip ID string (e.g. '00000001').
        with_route: Whether to load the .route file if present.
        
    Returns:
        track: (N, 3) GPS observations [lon, lat, timestamp].
        nodes: (V, 2) Road network node coordinates [lon, lat].
        arcs: (E, 2) Directed road network edges [src_node, dst_node].
        route: (R,) Reference route arc indices, or None.
    """
    d = Path(root) / tid
    track_path = d / f"{tid}.track"
    nodes_path = d / f"{tid}.nodes"
    arcs_path = d / f"{tid}.arcs"
    route_path = d / f"{tid}.route"

    if not track_path.exists():
        raise FileNotFoundError(f"Missing track file: {track_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    if not arcs_path.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_path}")

    track = np.loadtxt(track_path, ndmin=2)
    nodes = np.loadtxt(nodes_path, ndmin=2)
    arcs = np.loadtxt(arcs_path, dtype=int, ndmin=2)

    route = None
    if with_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=int, ndmin=1).reshape(-1)

    return track, nodes, arcs, route
