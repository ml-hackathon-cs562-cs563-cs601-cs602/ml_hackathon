"""Unit tests for ingest/parsers.py: column counts, types, arc_id = row index.

Owner: Member A
"""

import numpy as np
import pytest
from pathlib import Path

from pathsnap.ingest.parsers import load_trip


def test_parsers_structure_and_types(tmp_path: Path):
    tid = "test_trip_01"
    trip_dir = tmp_path / tid
    trip_dir.mkdir()

    # Create dummy .track (lon, lat, ts)
    track_data = np.array([
        [-1.55, 53.80, 0.0],
        [-1.54, 53.81, 1.0],
        [-1.53, 53.82, 2.0],
    ])
    np.savetxt(trip_dir / f"{tid}.track", track_data)

    # Create dummy .nodes (lon, lat)
    nodes_data = np.array([
        [-1.55, 53.80],
        [-1.54, 53.81],
        [-1.53, 53.82],
    ])
    np.savetxt(trip_dir / f"{tid}.nodes", nodes_data)

    # Create dummy .arcs (src, dst)
    arcs_data = np.array([
        [0, 1],
        [1, 2],
    ], dtype=int)
    np.savetxt(trip_dir / f"{tid}.arcs", arcs_data, fmt="%d")

    # Create dummy .route (arc indices)
    route_data = np.array([0, 1], dtype=int)
    np.savetxt(trip_dir / f"{tid}.route", route_data, fmt="%d")

    # Load with route
    track, nodes, arcs, route = load_trip(tmp_path, tid, with_route=True)

    assert track.shape == (3, 3)
    assert track.dtype in (np.float64, float)
    assert nodes.shape == (3, 2)
    assert arcs.shape == (2, 2)
    assert np.issubdtype(arcs.dtype, np.integer)
    assert route is not None
    assert len(route) == 2

    # Arc ID = row index
    for arc_id in range(len(arcs)):
        src, dst = arcs[arc_id]
        assert 0 <= src < len(nodes)
        assert 0 <= dst < len(nodes)

    # Test with_route=False
    _, _, _, route_none = load_trip(tmp_path, tid, with_route=False)
    assert route_none is None


def test_parsers_missing_file_raises(tmp_path: Path):
    tid = "missing_trip"
    trip_dir = tmp_path / tid
    trip_dir.mkdir()
    (trip_dir / f"{tid}.track").write_text("0 0 0\n")

    with pytest.raises(FileNotFoundError):
        load_trip(tmp_path, tid)
