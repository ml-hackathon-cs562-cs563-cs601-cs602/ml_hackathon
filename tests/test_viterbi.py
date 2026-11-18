"""Unit tests for hmm/viterbi.py: missing candidates, disconnected arcs, gap handling.

Owner: Member B
"""

import numpy as np
import pytest

from pathsnap.hmm.viterbi import decode


def test_decode_connected_chain():
    # 3 arcs in a linear path: arc 0 (0->1), arc 1 (1->2), arc 2 (2->3)
    arcs = np.array([
        [0, 1],  # arc 0
        [1, 2],  # arc 1
        [2, 3],  # arc 2
        [3, 0],  # arc 3 (loop back)
    ], dtype=int)

    # 3 observations, K=2 candidates each
    cand = np.array([
        [0, 3],  # Obs 0: arc 0 vs arc 3
        [1, 3],  # Obs 1: arc 1 vs arc 3
        [2, 3],  # Obs 2: arc 2 vs arc 3
    ], dtype=int)

    # Probabilities slightly favoring candidate 0 (arc 0, 1, 2)
    pm = np.array([
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2],
    ], dtype=np.float32)

    ts = np.array([0.0, 1.0, 2.0])

    pred = decode(arcs, cand, pm, ts)
    assert np.array_equal(pred, [0, 1, 2])


def test_decode_missing_candidates_imputation():
    # Arc 0 (0->1), Arc 1 (1->2)
    arcs = np.array([
        [0, 1],
        [1, 2],
    ], dtype=int)

    # Obs 1 has NO candidates (-1 padded)
    cand = np.array([
        [0, -1],
        [-1, -1],  # Missing candidate point
        [1, -1],
    ], dtype=int)

    pm = np.array([
        [0.9, 0.0],
        [0.0, 0.0],
        [0.9, 0.0],
    ], dtype=np.float32)

    ts = np.array([0.0, 1.0, 2.0])

    pred = decode(arcs, cand, pm, ts)
    assert len(pred) == 3
    assert pred[0] == 0
    assert pred[2] == 1
    # Missing observation is filled from neighbor
    assert pred[1] in (0, 1)


def test_decode_gap_handling():
    # Disconnected arcs: arc 0 (0->1), arc 1 (10->11)
    arcs = np.array([
        [0, 1],
        [10, 11],
    ], dtype=int)

    cand = np.array([
        [0, -1],
        [1, -1],
    ], dtype=int)

    pm = np.array([
        [0.9, 0.0],
        [0.9, 0.0],
    ], dtype=np.float32)

    # Time delta = 100s > gap_s (30s)
    ts = np.array([0.0, 100.0])

    # With gap reset, jump between disconnected arcs is permitted without failing
    pred = decode(arcs, cand, pm, ts, params={"gap_s": 30.0})
    assert np.array_equal(pred, [0, 1])


def test_decode_all_missing_returns_negative_ones():
    arcs = np.array([[0, 1]], dtype=int)
    cand = np.array([[-1, -1], [-1, -1]], dtype=int)
    pm = np.zeros((2, 2), dtype=np.float32)
    ts = np.array([0.0, 1.0])

    pred = decode(arcs, cand, pm, ts)
    assert np.array_equal(pred, [-1, -1])
