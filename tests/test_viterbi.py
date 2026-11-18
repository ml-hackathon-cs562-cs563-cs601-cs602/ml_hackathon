import numpy as np
import pytest

from pathsnap.hmm.viterbi import decode


ARCS = np.array([
    [0, 1],    # arc 0
    [1, 2],    # arc 1
    [2, 3],    # arc 2
    [10, 11],  # arc 3: disconnected
])


def test_decoder_returns_correct_shape():
    cand = np.array([[0, 1], [1, 2]])
    pm = np.array([[0.8, 0.2], [0.7, 0.3]])
    ts = np.array([0.0, 1.0])

    result = decode(
        ARCS, cand, pm, ts,
        params={"w_e": 1.0},
    )

    assert result.shape == (2,)


def test_decoder_prefers_high_probability_candidate():
    cand = np.array([[0, 1]])
    pm = np.array([[0.2, 0.8]])
    ts = np.array([0.0])

    result = decode(
        ARCS, cand, pm, ts,
        params={"w_e": 1.0},
    )

    assert result[0] == 1


def test_decoder_prefers_connected_sequence():
    cand = np.array([
        [0, 1],
        [1, 3],
    ])

    pm = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
    ])

    ts = np.array([0.0, 1.0])

    result = decode(
        ARCS,
        cand,
        pm,
        ts,
        params={
            "w_e": 1.0,
                "hop_base": 0.1,
            "hop_step": 0.1,
            "invalid": 20.0,
        },
    )

    # Arc 1 is connected to arc 0; arc 3 is disconnected.
    assert result.tolist() == [0, 1]

def test_decoder_penalizes_disconnected_transition():
    cand = np.array([
        [0],
        [3],
    ])

    pm = np.array([
        [0.99],
        [0.99],
    ])

    ts = np.array([0.0, 1.0])

    result = decode(
        ARCS,
        cand,
        pm,
        ts,
        params={"invalid": 15.0},
    )

    # The only available candidates are selected, even if disconnected.
    assert result.tolist() == [0, 3]


def test_decoder_resets_after_large_gps_gap():
    cand = np.array([
        [0, 1],
        [3, 2],
    ])

    pm = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
    ])

    ts = np.array([0.0, 100.0])

    result = decode(
        ARCS,
        cand,
        pm,
        ts,
        params={
            "w_e": 1.0,
            "gap_s": 30.0,
        },
    )

    assert result.tolist() == [0, 2]


def test_decoder_preserves_missing_observations():
    cand = np.array([
        [0, 1],
        [-1, -1],
        [1, 2],
    ])

    pm = np.array([
        [0.8, 0.2],
        [0.0, 0.0],
        [0.7, 0.3],
    ])

    ts = np.array([0.0, 1.0, 2.0])

    result = decode(
        ARCS, cand, pm, ts,
        params={"w_e": 1.0},
    )

    assert result[0] >= 0
    assert result[1] == -1
    assert result[2] >= 0


def test_decoder_handles_no_candidates():
    cand = np.full((3, 2), -1)
    pm = np.zeros((3, 2))
    ts = np.array([0.0, 1.0, 2.0])

    result = decode(
        ARCS, cand, pm, ts,
        params={"w_e": 1.0},
    )

    assert np.all(result == -1)


def test_decoder_rejects_unknown_arc_id():
    cand = np.array([[99]])
    pm = np.array([[1.0]])
    ts = np.array([0.0])

    with pytest.raises(ValueError):
        decode(ARCS, cand, pm, ts)


def test_decoder_rejects_unsorted_timestamps():
    cand = np.array([[0], [1]])
    pm = np.array([[0.8], [0.7]])
    ts = np.array([2.0, 1.0])

    with pytest.raises(ValueError):
        decode(ARCS, cand, pm, ts)


def test_decoder_rejects_mismatched_probability_shape():
    cand = np.array([[0], [1]])
    pm = np.array([[0.8]])
    ts = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        decode(ARCS, cand, pm, ts)


def test_decoder_accepts_sparse_node_ids():
    sparse_arcs = np.array([
        [1000, 2000],
        [2000, 3000],
    ])

    cand = np.array([
        [0],
        [1],
    ])

    pm = np.array([
        [0.9],
        [0.9],
    ])

    ts = np.array([0.0, 1.0])

    result = decode(sparse_arcs, cand, pm, ts)

    assert result.tolist() == [0, 1]
