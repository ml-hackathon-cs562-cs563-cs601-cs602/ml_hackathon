import numpy as np
import pytest

from pathsnap.hmm.emissions import compute_emissions, prob_matrix


def test_prob_matrix_formats_flat_probabilities():
    tab = {
        "cand": np.array([[0, 1], [2, 3]]),
        "ri": np.array([0, 0, 1]),
        "rk": np.array([0, 1, 0]),
    }
    p = np.array([0.8, 0.6, 0.9])

    result = prob_matrix(tab, p)

    assert result.shape == (2, 2)
    assert np.allclose(result, [[0.8, 0.6], [0.9, 1e-6]])


def test_emissions_increase_with_classifier_probability():
    cand = np.array([[0, 1]])
    pm = np.array([[0.2, 0.8]])

    result = compute_emissions(pm, cand, w_e=0.357)

    assert result[0, 1] > result[0, 0]


def test_emissions_match_weighted_log_probability():
    cand = np.array([[0]])
    pm = np.array([[0.8]])

    result = compute_emissions(pm, cand, w_e=0.357)

    expected = 0.357 * np.log(0.8)
    assert np.isclose(result[0, 0], expected)


def test_invalid_candidates_receive_large_negative_score():
    cand = np.array([[0, -1]])
    pm = np.array([[0.8, 0.9]])

    result = compute_emissions(pm, cand)

    assert result[0, 1] == -1e9


def test_emissions_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_emissions(np.array([[0.8]]), np.array([[0, 1]]))


def test_emissions_reject_invalid_probability():
    with pytest.raises(ValueError):
        compute_emissions(np.array([[1.2]]), np.array([[0]]))


def test_emissions_reject_invalid_weight():
    with pytest.raises(ValueError):
        compute_emissions(np.array([[0.8]]), np.array([[0]]), w_e=1.5)
