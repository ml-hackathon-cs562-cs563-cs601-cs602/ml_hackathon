"""Group-aware Cross Validation for PathSnap.

Owner: Member C
Phase 1: Shared harness

Implements GroupKFold splitting by trip_id to prevent data leakage across trips.
Guarantees that observations from the same trip never appear in both train and validation splits.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.evaluation.metrics import candidate_macro_f1, compute_candidate_metrics


def get_group_kfold(n_splits: int = 5) -> GroupKFold:
    """Return a configured GroupKFold splitter."""
    return GroupKFold(n_splits=n_splits)


def validate_no_leakage(
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Assert that no group (trip_id) exists in both train and validation sets."""
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = train_groups.intersection(val_groups)
    if overlap:
        raise AssertionError(
            f"Trip leakage detected in GroupKFold split! Overlapping trips: {overlap}"
        )


def split_by_trip(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """Yield leakage-free (train_idx, val_idx, fold) for GroupKFold."""
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError(
            f"GroupKFold requires at least 2 distinct groups, found {len(unique_groups)}"
        )

    gkf = GroupKFold(n_splits=folds)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        validate_no_leakage(groups, train_idx, val_idx)
        yield train_idx, val_idx, fold


def cross_validate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    predict_proba_fn: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run GroupKFold cross-validation on candidate data.
    
    Args:
        model: Sklearn-compatible estimator or wrapper with fit and predict_proba.
        X: Feature matrix.
        y: Binary labels (0 or 1).
        groups: Array of trip_ids.
        n_splits: Number of CV folds (default 5).
        predict_proba_fn: Optional custom function to extract positive class probabilities.
        
    Returns:
        Dictionary containing:
        - cv_macro_f1_mean: Mean validation candidate macro F1 across folds.
        - cv_macro_f1_std: Standard deviation across folds.
        - cv_train_macro_f1_mean: Mean training candidate macro F1.
        - fold_rows: List of per-fold metric dictionaries.
        - folds_df: DataFrame of per-fold metrics.
    """
    fold_rows = []
    val_scores = []
    train_scores = []

    for train_idx, val_idx, fold in split_by_trip(X, y, groups, n_splits=n_splits):
        est = clone(model) if hasattr(model, "get_params") else model
        est.fit(X[train_idx], y[train_idx])

        # Get probabilities
        if predict_proba_fn is not None:
            p_val = predict_proba_fn(est, X[val_idx])
            p_tr = predict_proba_fn(est, X[train_idx])
        elif hasattr(est, "predict_proba"):
            p_raw_val = est.predict_proba(X[val_idx])
            p_val = p_raw_val[:, 1] if p_raw_val.ndim == 2 else p_raw_val
            p_raw_tr = est.predict_proba(X[train_idx])
            p_tr = p_raw_tr[:, 1] if p_raw_tr.ndim == 2 else p_raw_tr
        else:
            p_val = est.predict(X[val_idx])
            p_tr = est.predict(X[train_idx])

        val_metrics = compute_candidate_metrics(y[val_idx], p_val)
        tr_metrics = compute_candidate_metrics(y[train_idx], p_tr)

        val_f1 = val_metrics["candidate_macro_f1"]
        tr_f1 = tr_metrics["candidate_macro_f1"]
        val_scores.append(val_f1)
        train_scores.append(tr_f1)

        fold_rows.append({
            "fold": fold,
            "train_trips": len(set(groups[train_idx])),
            "validation_trips": len(set(groups[val_idx])),
            "train_candidate_macro_f1": float(tr_f1),
            "validation_candidate_macro_f1": float(val_f1),
            "validation_candidate_log_loss": float(val_metrics["candidate_log_loss"]),
            "validation_candidate_accuracy": float(val_metrics["candidate_accuracy"]),
        })

    return {
        "cv_macro_f1_mean": float(np.mean(val_scores)),
        "cv_macro_f1_std": float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0,
        "cv_train_macro_f1_mean": float(np.mean(train_scores)),
        "fold_rows": fold_rows,
        "folds_df": pd.DataFrame(fold_rows),
    }
