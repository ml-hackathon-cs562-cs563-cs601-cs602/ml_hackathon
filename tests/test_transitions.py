import numpy as np
import pytest

from pathsnap.hmm.transitions import Hop


def make_graph():
    # 10 -> 20 -> 30 -> 40
    # 10 -> 50
    # 60 is disconnected
    return np.array([
        [10, 20],
        [20, 30],
        [30, 40],
        [10, 50],
    ])


def test_direct_reachability():
    hop = Hop(make_graph(), H=4)

    result = hop.reach(10)

    assert result[10] == 0
    assert result[20] == 1
    assert result[30] == 2
    assert result[40] == 3
    assert result[50] == 1


def test_hop_limit():
    hop = Hop(make_graph(), H=2)

    result = hop.reach(10)

    assert 20 in result
    assert 30 in result
    assert 40 not in result


def test_directed_graph():
    hop = Hop(make_graph(), H=4)

    result = hop.reach(40)

    assert result == {40: 0}
    assert 10 not in result


def test_disconnected_node():
    hop = Hop(make_graph(), H=4)

    assert hop.reach(60) == {60: 0}


def test_sparse_node_ids():
    arcs = np.array([
        [1000, 5000],
        [5000, 9000],
    ])

    hop = Hop(arcs, H=2)
    result = hop.reach(1000)

    assert result == {
        1000: 0,
        5000: 1,
        9000: 2,
    }


def test_empty_graph():
    hop = Hop(np.empty((0, 2), dtype=int), H=3)

    assert hop.reach(100) == {100: 0}


def test_zero_hop_limit():
    hop = Hop(make_graph(), H=0)

    assert hop.reach(10) == {10: 0}


def test_cache_is_reused():
    hop = Hop(make_graph(), H=3)

    first = hop.reach(10)
    second = hop.reach(10)

    assert first is second


def test_invalid_graph_shape():
    with pytest.raises(ValueError):
        Hop(np.array([1, 2, 3]), H=2)


def test_negative_node_id():
    with pytest.raises(ValueError):
        Hop(np.array([[-1, 2]]), H=2)


def test_invalid_hop_limit():
    with pytest.raises(ValueError):
        Hop(make_graph(), H=-1)
