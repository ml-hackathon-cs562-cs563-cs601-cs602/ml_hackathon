"""Viterbi decoding over candidate road-edge hidden states.

Owner: Member B
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pathsnap.hmm.emissions import compute_emissions
from pathsnap.hmm.transitions import Hop


DEFAULT_PARAMS: Dict[str, Any] = {
    "w_e": 0.357,
    "hop_base": 1.089,
    "hop_step": 1.962,
    "invalid": 17.302,
    "uturn": 5.066,
    "H": 4,
    "gap_s": 5.0,
    "stay": 0.0,
}


def _transition_matrix(
    previous_candidates: np.ndarray,
    current_candidates: np.ndarray,
    arcs: np.ndarray,
    hop: Hop,
    params: Dict[str, Any],
) -> np.ndarray:
    """Calculate transition scores between two consecutive observations."""

    previous_count = len(previous_candidates)
    current_count = len(current_candidates)

    scores = np.full(
        (previous_count, current_count),
        -float(params["invalid"]),
        dtype=np.float64,
    )

    for i, previous_arc in enumerate(previous_candidates):
        previous_start, previous_end = arcs[previous_arc]
        reachable = hop.reach(int(previous_end))

        for j, current_arc in enumerate(current_candidates):
            current_start, current_end = arcs[current_arc]

            if previous_arc == current_arc:
                scores[i, j] = -float(params["stay"])
                continue

            hops = reachable.get(int(current_start))

            if hops is not None:
                scores[i, j] = -(
                    float(params["hop_base"])
                    + float(params["hop_step"]) * hops
                )

            # Penalize an immediate reversal along the same directed edge.
            if (
                current_start == previous_end
                and current_end == previous_start
            ):
                scores[i, j] -= float(params["uturn"])

    return scores


def _decode_segment(
    indices: np.ndarray,
    candidates: np.ndarray,
    emissions: np.ndarray,
    arcs: np.ndarray,
    hop: Hop,
    timestamps: np.ndarray,
    params: Dict[str, Any],
    output: np.ndarray,
) -> None:
    """Decode one uninterrupted sequence of observations."""

    first = int(indices[0])

    valid_candidates = np.flatnonzero(candidates[first] >= 0)
    current_ids = candidates[first, valid_candidates]

    score = emissions[first, valid_candidates].astype(np.float64)
    score -= score.max()

    backpointers = []

    for previous_index, current_index in zip(indices[:-1], indices[1:]):
        previous_index = int(previous_index)
        current_index = int(current_index)

        previous_slots = np.flatnonzero(
            candidates[previous_index] >= 0
        )
        current_slots = np.flatnonzero(
            candidates[current_index] >= 0
        )

        previous_ids = candidates[previous_index, previous_slots]
        current_ids = candidates[current_index, current_slots]

        transition = _transition_matrix(
            previous_ids,
            current_ids,
            arcs,
            hop,
            params,
        )

        previous_emission = score[:, None]
        total = previous_emission + transition

        best_previous = np.argmax(total, axis=0)

        current_score = (
            total[best_previous, np.arange(len(current_ids))]
            + emissions[current_index, current_slots]
        )

        current_score -= current_score.max()

        backpointers.append(
            (current_index, current_slots, best_previous)
        )

        score = current_score

    best_state = int(np.argmax(score))

    last_index = int(indices[-1])
    last_slots = np.flatnonzero(candidates[last_index] >= 0)

    output[last_index] = candidates[last_index, last_slots[best_state]]

    for observation_index, slots, previous_states in reversed(backpointers):
        best_state = int(previous_states[best_state])

        previous_index = int(
            indices[np.searchsorted(indices, observation_index) - 1]
        )

        output[previous_index] = candidates[
            previous_index,
            np.flatnonzero(candidates[previous_index] >= 0)[best_state],
        ]


def decode(
    arcs: np.ndarray,
    cand: np.ndarray,
    pm: np.ndarray,
    ts: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    hop: Optional[Hop] = None,
) -> np.ndarray:
    """
    Decode the most likely road-arc sequence using Viterbi.

    Args:
        arcs: (E, 2) directed road edges [source_node, destination_node].
        cand: (N, K) candidate arc IDs, with negative values as padding.
        pm: (N, K) classifier probabilities.
        ts: (N,) timestamps in seconds.
        params: optional HMM configuration.
        hop: optional precomputed directed reachability object.

    Returns:
        (N,) decoded arc IDs. Observations without candidates remain -1.
    """

    p = {**DEFAULT_PARAMS, **(params or {})}

    arcs = np.asarray(arcs)
    cand = np.asarray(cand)
    pm = np.asarray(pm, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("arcs must have shape (E, 2).")

    if len(arcs) == 0:
        raise ValueError("The road network must contain at least one arc.")

    if not np.issubdtype(arcs.dtype, np.integer):
        if not np.all(np.isfinite(arcs)) or not np.all(arcs == np.floor(arcs)):
            raise ValueError("Road-network node IDs must be integers.")

    arcs = arcs.astype(np.int64)

    if np.any(arcs < 0):
        raise ValueError("Road-network node IDs must be non-negative.")

    if cand.ndim != 2:
        raise ValueError("cand must be a 2D array.")

    if not np.issubdtype(cand.dtype, np.integer):
        if not np.all(np.isfinite(cand)) or not np.all(cand == np.floor(cand)):
            raise ValueError("Candidate arc IDs must be integers.")

    cand = cand.astype(np.int64)

    if pm.shape != cand.shape:
        raise ValueError("pm and cand must have identical shapes.")

    if ts.ndim != 1 or len(ts) != len(cand):
        raise ValueError("ts must contain one timestamp per observation.")

    if np.any(~np.isfinite(ts)):
        raise ValueError("Timestamps must be finite.")

    if np.any(np.diff(ts) < 0):
        raise ValueError("Timestamps must be in non-decreasing order.")

    if np.any(cand >= len(arcs)):
        raise ValueError("A candidate references an unknown arc ID.")

    for key in ("w_e", "hop_base", "hop_step", "invalid", "uturn", "stay", "gap_s"):
        value = float(p[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative.")

    if not isinstance(p["H"], (int, np.integer)) or p["H"] < 0:
        raise ValueError("H must be a non-negative integer.")

    emissions = compute_emissions(
        pm,
        cand,
        w_e=float(p["w_e"]),
    )

    output = np.full(len(cand), -1, dtype=np.int64)

    # An observation is valid if it has at least one candidate in any slot.
    valid_indices = np.flatnonzero(np.any(cand >= 0, axis=1))

    if len(valid_indices) == 0:
        return output

    hop = hop or Hop(arcs, H=int(p["H"]))

    # Split the sequence at long GPS gaps. Each segment is decoded independently.
    segment_start = 0

    for position in range(1, len(valid_indices) + 1):
        split = (
            position == len(valid_indices)
            or ts[valid_indices[position]]
            - ts[valid_indices[position - 1]]
            > float(p["gap_s"])
        )

        if split:
            segment = valid_indices[segment_start:position]

            _decode_segment(
                segment,
                cand,
                emissions,
                arcs,
                hop,
                ts,
                p,
                output,
            )

            segment_start = position

    return output
