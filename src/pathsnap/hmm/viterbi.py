"""Viterbi decoding algorithm over candidate road-edge hidden states.

Owner: Member B
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from pathsnap.hmm.emissions import compute_emissions, prob_matrix
from pathsnap.hmm.transitions import Hop

DEFAULT_PARAMS: Dict[str, Any] = {
    "w_e": 1.0,
    "hop_base": 1.0,
    "hop_step": 0.5,
    "invalid": 8.0,
    "uturn": 2.0,
    "H": 4,
    "gap_s": 30.0,
    "stay": 0.0,
}


def decode(
    arcs: np.ndarray,
    cand: np.ndarray,
    pm: np.ndarray,
    ts: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    hop: Optional[Hop] = None,
) -> np.ndarray:
    """Run Viterbi sequence decoding over candidate states.
    
    Args:
        arcs: (E, 2) Road network directed edges [src, dst].
        cand: (N, K) Candidate arc IDs (-1 padded).
        pm: (N, K) Classifier probabilities.
        ts: (N,) Timestamps in seconds.
        params: HMM transition & emission parameters.
        hop: Optional memoized Hop reachability instance.
        
    Returns:
        out: (N,) Predicted road arc index per observation (-1 where undecidable).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    hop = hop or Hop(arcs, int(p["H"]))
    N, K = cand.shape
    out = np.full(N, -1, dtype=int)

    valid = np.nonzero(cand[:, 0] >= 0)[0]
    if len(valid) == 0:
        return out

    fr, to = arcs[:, 0], arcs[:, 1]
    em = compute_emissions(pm, cand, w_e=p["w_e"])

    prev = valid[0]
    score = em[prev].copy()
    back = []

    for i in valid[1:]:
        ca, pa = cand[i], cand[prev]
        gap = (ts[i] - ts[prev]) > p["gap_s"]
        T = np.zeros((K, K), dtype=np.float32)

        if not gap:
            for a_i in range(K):
                a = pa[a_i]
                if a < 0:
                    T[a_i] = -1e9
                    continue
                r = hop.reach(to[a])
                for b_i in range(K):
                    b = ca[b_i]
                    if b < 0:
                        T[a_i, b_i] = -1e9
                        continue
                    if a == b:
                        T[a_i, b_i] = -p["stay"]
                        continue
                    h = r.get(fr[b])
                    if h is None:
                        t_cost = -p["invalid"]
                    else:
                        t_cost = -(p["hop_base"] + p["hop_step"] * h)
                    # U-turn penalty
                    if to[b] == fr[a] and fr[b] == to[a]:
                        t_cost -= p["uturn"]
                    T[a_i, b_i] = t_cost
        else:
            # Time gap reset: transition scores are all 0
            T[:] = np.where(pa[:, None] >= 0, 0.0, -1e9)

        cand_sc = score[:, None] + T
        bp = cand_sc.argmax(axis=0)
        score = cand_sc.max(axis=0) + em[i]
        score -= score.max()  # Prevent numerical underflow
        back.append((i, bp))
        prev = i

    best_k = int(score.argmax())
    out[prev] = cand[prev, best_k]

    for i, bp in reversed(back):
        best_k = int(bp[best_k])
        j = valid[np.searchsorted(valid, i) - 1]
        out[j] = cand[j, best_k]

    # Forward fill missing observations
    last = -1
    for i in range(N):
        if out[i] < 0:
            out[i] = last
        else:
            last = out[i]

    # Backward fill remaining start observations
    nxt = -1
    for i in range(N - 1, -1, -1):
        if out[i] < 0:
            out[i] = nxt
        else:
            nxt = out[i]

    return out
