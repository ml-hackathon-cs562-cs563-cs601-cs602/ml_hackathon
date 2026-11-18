"""Tests for evaluation/cv.py GroupKFold splitting and leakage validation.

Owner: Member C
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pathsnap.evaluation.cv import (
    cross_validate,
    get_group_kfold,
    split_by_trip,
    validate_no_leakage,
)


def test_validate_no_leakage_passes_when_disjoint():
    groups = np.array(["trip_1", "trip_1", "trip_2", "trip_2", "trip_3", "trip_3"])
    train_idx = np.array([0, 1, 2, 3])
    val_idx = np.array([4, 5])
    # Should not raise
    validate_no_leakage(groups, train_idx, val_idx)


def test_validate_no_leakage_raises_on_overlap():
    groups = np.array(["trip_1", "trip_1", "trip_2", "trip_2", "trip_3", "trip_3"])
    train_idx = np.array([0, 1, 2])  # includes trip_2 ix 2
    val_idx = np.array([3, 4, 5])    # includes trip_2 ix 3
    with pytest.raises(AssertionError, match="Trip leakage detected"):
        validate_no_leakage(groups, train_idx, val_idx)


def test_split_by_trip_zero_leakage():
    # 5 trips, 10 samples each
    groups = np.repeat([f"trip_{i}" for i in range(5)], 10)
    X = np.random.randn(len(groups), 4)
    y = np.random.choice([0, 1], len(groups))

    splits = list(split_by_trip(X, y, groups, n_splits=5))
    assert len(splits) == 5

    for train_idx, val_idx, fold in splits:
        train_trips = set(groups[train_idx])
        val_trips = set(groups[val_idx])
        assert len(train_trips.intersection(val_trips)) == 0
        assert len(train_idx) + len(val_idx) == len(groups)


def test_cross_validate_structure():
    groups = np.repeat([f"trip_{i}" for i in range(6)], 20)
    X = np.random.randn(len(groups), 4)
    y = np.random.choice([0, 1], len(groups))

    model = LogisticRegression()
    res = cross_validate(model, X, y, groups, n_splits=3)

    assert "cv_macro_f1_mean" in res
    assert "cv_macro_f1_std" in res
    assert "cv_train_macro_f1_mean" in res
    assert "fold_rows" in res
    assert len(res["fold_rows"]) == 3
    for fold_dict in res["fold_rows"]:
        assert "train_candidate_macro_f1" in fold_dict
        assert "validation_candidate_macro_f1" in fold_dict
