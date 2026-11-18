"""PathSnap Evaluation Metrics.

Owner: Member C
Phase 1: Shared harness

Implements standard metric functions computed consistently across all models:
- Candidate-level: Macro F1, Precision, Recall, Accuracy, Log Loss
- Observation-level: Top-1 Accuracy, Candidate Miss Rate, Observation Macro F1
- Sequence-level: Route Continuity, Sequence Exact Match
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def candidate_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Compute candidate-level macro F1 score."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    return float(f1_score(y_t, y_p, average="macro", zero_division=0))


def candidate_precision(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Compute candidate-level precision for positive class (reference edge)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    return float(precision_score(y_t, y_p, zero_division=0))


def candidate_recall(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Compute candidate-level recall for positive class (reference edge)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    return float(recall_score(y_t, y_p, zero_division=0))


def candidate_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Compute candidate-level binary classification accuracy."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    return float(accuracy_score(y_t, y_p))


def candidate_log_loss(
    y_true: Sequence[int],
    y_prob: Union[Sequence[float], np.ndarray],
    eps: float = 1e-15,
) -> float:
    """Compute binary cross-entropy log loss over candidate probabilities."""
    y_t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if p.ndim == 1:
        p = np.clip(p, eps, 1.0 - eps)
        prob_matrix = np.column_stack([1.0 - p, p])
    else:
        prob_matrix = np.clip(p, eps, 1.0 - eps)
        # Normalize rows to sum to 1
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    return float(log_loss(y_t, prob_matrix, labels=[0, 1]))


def observation_top1_accuracy(
    true_arcs: Sequence[int],
    pred_arcs: Sequence[int],
) -> float:
    """Compute observation-level Top-1 accuracy.
    
    A reference edge missing from the candidate set is encoded as -1 in true_arcs
    and counts as an incorrect prediction.
    """
    t = np.asarray(true_arcs, dtype=int)
    p = np.asarray(pred_arcs, dtype=int)
    if len(t) == 0:
        return 0.0
    return float(np.mean(t == p))


def candidate_miss_rate(
    true_arcs: Sequence[int],
) -> float:
    """Fraction of GPS observations where the true reference arc is missing (< 0)."""
    t = np.asarray(true_arcs, dtype=int)
    if len(t) == 0:
        return 0.0
    return float(np.mean(t < 0))


def candidate_coverage(
    true_arcs: Sequence[int],
) -> float:
    """Fraction of GPS observations with a valid candidate on reference route (>= 0)."""
    return 1.0 - candidate_miss_rate(true_arcs)


def observation_macro_f1(
    true_arcs: Sequence[int],
    pred_arcs: Sequence[int],
) -> float:
    """Compute macro F1 over local edge IDs across observations (including -1)."""
    t = np.asarray(true_arcs, dtype=int)
    p = np.asarray(pred_arcs, dtype=int)
    if len(t) == 0:
        return 0.0
    labels = np.unique(np.concatenate([t, p]))
    return float(f1_score(t, p, labels=labels, average="macro", zero_division=0))


def route_continuity(
    pred_src: Sequence[int],
    pred_dst: Sequence[int],
) -> float:
    """Directed edge continuity: fraction of consecutive predicted edges where
    the previous edge destination node equals the next edge source node.
    """
    src = np.asarray(pred_src, dtype=int)
    dst = np.asarray(pred_dst, dtype=int)
    if len(src) <= 1:
        return 1.0
    matches = dst[:-1] == src[1:]
    return float(np.mean(matches))


def sequence_exact_match(
    true_route: Sequence[int],
    pred_route: Sequence[int],
) -> float:
    """Check if the entire predicted route sequence exactly matches the true route."""
    t = np.asarray(true_route, dtype=int)
    p = np.asarray(pred_route, dtype=int)
    if len(t) != len(p):
        return 0.0
    return float(np.array_equal(t, p))


def compute_candidate_metrics(
    y_true: Sequence[int],
    y_prob: Union[Sequence[float], np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all candidate-level metrics in one call."""
    y_t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if p.ndim == 2:
        p_pos = p[:, 1]
    else:
        p_pos = p
    y_pred = (p_pos >= threshold).astype(int)

    return {
        "candidate_macro_f1": candidate_macro_f1(y_t, y_pred),
        "candidate_precision": candidate_precision(y_t, y_pred),
        "candidate_recall": candidate_recall(y_t, y_pred),
        "candidate_accuracy": candidate_accuracy(y_t, y_pred),
        "candidate_log_loss": candidate_log_loss(y_t, p_pos),
    }


def compute_observation_predictions(
    obs_ids: np.ndarray,
    arc_ids: np.ndarray,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the candidate with highest positive probability for each observation.
    
    Returns:
        (true_arcs, pred_arcs, pred_src, pred_dst)
    """
    unique_obs = np.unique(obs_ids)
    pred_arcs, true_arcs, pred_src, pred_dst = [], [], [], []
    for oid in unique_obs:
        ix = np.flatnonzero(obs_ids == oid)
        best_local = int(np.argmax(probabilities[ix]))
        best_idx = ix[best_local]

        # Check positive candidate for this observation
        positives = ix[y_true[ix] == 1]
        true_idx = positives[0] if len(positives) > 0 else None

        pred_arcs.append(int(arc_ids[best_idx]))
        pred_src.append(int(src_nodes[best_idx]))
        pred_dst.append(int(dst_nodes[best_idx]))
        true_arcs.append(int(arc_ids[true_idx]) if true_idx is not None else -1)

    return (
        np.asarray(true_arcs, dtype=int),
        np.asarray(pred_arcs, dtype=int),
        np.asarray(pred_src, dtype=int),
        np.asarray(pred_dst, dtype=int),
    )


def compute_route_metrics(
    metas: Sequence[Dict[str, Any]],
    probability_chunks: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Compute observation-level and route metrics across multiple trips."""
    all_top1 = []
    all_macro = []
    all_continuity = []
    all_true = []
    all_pred = []

    for meta, probs in zip(metas, probability_chunks):
        M = meta["M"]
        obs_ids = M[:, 0].astype(int)
        arc_ids = M[:, 1].astype(int)
        src_nodes = M[:, 2].astype(int)
        dst_nodes = M[:, 3].astype(int)
        y_true = np.asarray(meta["y"], dtype=int)

        t_arc, p_arc, p_src, p_dst = compute_observation_predictions(
            obs_ids, arc_ids, src_nodes, dst_nodes, y_true, probs
        )

        all_top1.extend((t_arc == p_arc).astype(float).tolist())
        labels = np.unique(np.concatenate([t_arc, p_arc]))
        all_macro.append(float(f1_score(t_arc, p_arc, labels=labels, average="macro", zero_division=0)))
        all_true.extend(t_arc.tolist())
        all_pred.extend(p_arc.tolist())

        if len(p_src) > 1:
            all_continuity.extend((p_dst[:-1] == p_src[1:]).astype(float).tolist())

    return {
        "student_test_top1_accuracy": float(np.mean(all_top1)) if all_top1 else 0.0,
        "student_test_observation_macro_f1": float(np.mean(all_macro)) if all_macro else 0.0,
        "route_continuity": float(np.mean(all_continuity)) if all_continuity else 0.0,
        "candidate_coverage": float(np.mean(np.asarray(all_true) >= 0)) if all_true else 0.0,
        "candidate_miss_rate": float(np.mean(np.asarray(all_true) < 0)) if all_true else 0.0,
    }
