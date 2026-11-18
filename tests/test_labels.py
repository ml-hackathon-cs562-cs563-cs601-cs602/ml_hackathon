"""Unit tests for labels/route_labels.py: positive and negative labels, unknown arc IDs.

Owner: Member B
"""

import numpy as np
import pytest

from pathsnap.geo.projection import Proj
from pathsnap.ingest.data_loader import Network
from pathsnap.labels.route_labels import align_labels


def test_align_labels_linear_route():
    # 3 nodes along x axis: (0,0), (100,0), (200,0)
    # 2 arcs: arc 0: 0->1, arc 1: 1->2
    nodes = np.array([
        [-1.55, 53.80],
        [-1.54, 53.80],
        [-1.53, 53.80],
    ])
    arcs = np.array([
        [0, 1],
        [1, 2],
    ], dtype=int)

    proj = Proj(nodes[:, 0].mean(), nodes[:, 1].mean())
    net = Network(nodes, arcs, proj)

    # GPS points near arc 0 and arc 1
    # Point 0 near arc 0, Point 1 near arc 1
    p0 = (net.A[0] + net.B[0]) / 2.0
    p1 = (net.A[1] + net.B[1]) / 2.0
    P = np.array([p0, p1])

    route = np.array([0, 1], dtype=int)
    matched_arcs, matched_costs = align_labels(net, P, route)

    assert len(matched_arcs) == 2
    assert matched_arcs[0] == 0
    assert matched_arcs[1] == 1
    assert matched_costs[0] < 5.0
    assert matched_costs[1] < 5.0


def test_align_labels_empty_and_unknown():
    nodes = np.array([
        [-1.55, 53.80],
        [-1.54, 53.80],
    ])
    arcs = np.array([[0, 1]], dtype=int)
    proj = Proj(nodes[:, 0].mean(), nodes[:, 1].mean())
    net = Network(nodes, arcs, proj)

    # Empty points
    empty_p = np.empty((0, 2))
    route = np.array([0], dtype=int)
    m_arcs, m_costs = align_labels(net, empty_p, route)
    assert len(m_arcs) == 0

    # Empty route
    P = np.array([[0.0, 0.0]])
    empty_route = np.array([], dtype=int)
    m_arcs, m_costs = align_labels(net, P, empty_route)
    assert len(m_arcs) == 1
    assert m_arcs[0] == -1


def test_positive_and_negative_candidate_labels():
    # Verify label assignment logic:
    # A candidate matching the reference arc gets label 1, others get 0
    ref_arc = 42
    candidates = np.array([10, 25, 42, 80, -1], dtype=int)

    labels = (candidates == ref_arc).astype(int)
    assert np.array_equal(labels, [0, 0, 1, 0, 0])
    assert labels.sum() == 1

    # Candidate miss case
    ref_arc_miss = 99
    labels_miss = (candidates == ref_arc_miss).astype(int)
    assert np.array_equal(labels_miss, [0, 0, 0, 0, 0])
    assert labels_miss.sum() == 0
