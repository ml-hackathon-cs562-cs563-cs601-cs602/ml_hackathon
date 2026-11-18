"""Tests for evaluation/metrics.py against hand-computed small examples.

Owner: Member C
"""

import math
import numpy as np
import pytest

from pathsnap.evaluation.metrics import (
    candidate_accuracy,
    candidate_coverage,
    candidate_log_loss,
    candidate_macro_f1,
    candidate_miss_rate,
    candidate_precision,
    candidate_recall,
    compute_candidate_metrics,
    compute_observation_predictions,
    compute_route_metrics,
    observation_macro_f1,
    observation_top1_accuracy,
    route_continuity,
    sequence_exact_match,
)


def test_candidate_classification_metrics_hand_computed():
    """Hand-computed test:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    
    Confusion matrix:
      TN=1 (ix 0), FP=1 (ix 1), FN=1 (ix 2), TP=1 (ix 3)
      Accuracy = (1 + 1) / 4 = 0.5
      Precision = TP / (TP + FP) = 1 / 2 = 0.5
      Recall = TP / (TP + FN) = 1 / 2 = 0.5
      F1_pos = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
      F1_neg: precision_neg = 1/2=0.5, recall_neg = 1/2=0.5 -> F1_neg = 0.5
      Macro F1 = (0.5 + 0.5) / 2 = 0.5
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    assert candidate_accuracy(y_true, y_pred) == pytest.approx(0.5)
    assert candidate_precision(y_true, y_pred) == pytest.approx(0.5)
    assert candidate_recall(y_true, y_pred) == pytest.approx(0.5)
    assert candidate_macro_f1(y_true, y_pred) == pytest.approx(0.5)


def test_candidate_perfect_classification():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 0]

    assert candidate_accuracy(y_true, y_pred) == 1.0
    assert candidate_precision(y_true, y_pred) == 1.0
    assert candidate_recall(y_true, y_pred) == 1.0
    assert candidate_macro_f1(y_true, y_pred) == 1.0


def test_candidate_log_loss_hand_computed():
    """Hand-computed log loss:
    y_true = [1, 0]
    y_prob = [0.8, 0.1]
    Loss = - (ln(0.8) + ln(0.9)) / 2
    ln(0.8) ~= -0.22314355
    ln(0.9) ~= -0.10536052
    Loss ~= -(-0.32850407)/2 = 0.16425203
    """
    y_true = [1, 0]
    y_prob = [0.8, 0.1]
    expected_loss = -(math.log(0.8) + math.log(0.9)) / 2.0
    assert candidate_log_loss(y_true, y_prob) == pytest.approx(expected_loss, rel=1e-4)


def test_observation_top1_accuracy_and_miss_rate():
    """true_arcs has one missing candidate (-1).
    true_arcs = [10, 20, -1, 40]
    pred_arcs = [10, 25, 30, 40]
    
    Observation 0: 10 == 10 -> Correct
    Observation 1: 20 != 25 -> Wrong
    Observation 2: -1 != 30 -> Wrong (reference arc was not among candidates)
    Observation 3: 40 == 40 -> Correct
    
    Top-1 accuracy = 2 / 4 = 0.5
    Candidate miss rate = 1 / 4 = 0.25
    Candidate coverage = 3 / 4 = 0.75
    """
    true_arcs = [10, 20, -1, 40]
    pred_arcs = [10, 25, 30, 40]

    assert observation_top1_accuracy(true_arcs, pred_arcs) == pytest.approx(0.5)
    assert candidate_miss_rate(true_arcs) == pytest.approx(0.25)
    assert candidate_coverage(true_arcs) == pytest.approx(0.75)


def test_route_continuity_hand_computed():
    """Directed edge continuity:
    Edge 0: node 1 -> node 2 (dst=2)
    Edge 1: node 2 -> node 3 (src=2, dst=3) -> match!
    Edge 2: node 4 -> node 5 (src=4, dst=5) -> mismatch!
    Edge 3: node 5 -> node 6 (src=5, dst=6) -> match!
    
    Pairs evaluated: (0, 1), (1, 2), (2, 3) -> 3 transitions.
    Matches: 2 / 3 ~= 0.6667
    """
    pred_src = [1, 2, 4, 5]
    pred_dst = [2, 3, 5, 6]

    assert route_continuity(pred_src, pred_dst) == pytest.approx(2.0 / 3.0)


def test_route_continuity_single_observation():
    assert route_continuity([1], [2]) == 1.0


def test_sequence_exact_match():
    route_a = [101, 102, 103]
    route_b = [101, 102, 103]
    route_c = [101, 999, 103]

    assert sequence_exact_match(route_a, route_b) == 1.0
    assert sequence_exact_match(route_a, route_c) == 0.0


def test_compute_observation_predictions():
    """Two observations, 2 candidates each:
    Obs 0:
      Candidate 0: arc=101, src=1, dst=2, y=1, prob=0.7
      Candidate 1: arc=102, src=1, dst=3, y=0, prob=0.3
      -> Best: arc 101, true=101
    Obs 1:
      Candidate 0: arc=201, src=2, dst=4, y=0, prob=0.2
      Candidate 1: arc=202, src=2, dst=5, y=0, prob=0.8
      -> Best: arc 202, true=-1 (no candidate was on route)
    """
    obs_ids = np.array([0, 0, 1, 1])
    arc_ids = np.array([101, 102, 201, 202])
    src_nodes = np.array([1, 1, 2, 2])
    dst_nodes = np.array([2, 3, 4, 5])
    y_true = np.array([1, 0, 0, 0])
    probabilities = np.array([0.7, 0.3, 0.2, 0.8])

    true_arcs, pred_arcs, pred_src, pred_dst = compute_observation_predictions(
        obs_ids, arc_ids, src_nodes, dst_nodes, y_true, probabilities
    )

    np.testing.assert_array_equal(true_arcs, [101, -1])
    np.testing.assert_array_equal(pred_arcs, [101, 202])
    np.testing.assert_array_equal(pred_src, [1, 2])
    np.testing.assert_array_equal(pred_dst, [2, 5])
    assert route_continuity(pred_src, pred_dst) == 1.0  # dst 2 == src 2
