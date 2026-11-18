"""Per-candidate feature extraction for PathSnap candidate scoring.

Owner: Member B
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from pathsnap.candidates.generate import candidates
from pathsnap.geo.projection import Proj
from pathsnap.ingest.data_loader import Network
from pathsnap.ingest.parsers import load_trip
from pathsnap.labels.route_labels import align_labels

FEATS = [
    "dist",
    "rank",
    "dist_rel",
    "dist_gap",
    "t_proj",
    "arc_len",
    "bear_diff_cos",
    "bear_diff_abs",
    "speed",
    "n_cand",
    "dist_x_cos",
    "perp_signed",
    "prev_conn",
    "next_conn",
    "dt",
]


def bearing_from_track(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute heading angle and displacement magnitude along GPS trajectory."""
    d = np.zeros_like(P)
    if len(P) > 2:
        d[1:-1] = P[2:] - P[:-2]
        d[0] = P[1] - P[0]
        d[-1] = P[-1] - P[-2]
    elif len(P) == 2:
        d[0] = P[1] - P[0]
        d[1] = d[0]
    sp = np.sqrt((d ** 2).sum(axis=1))
    return np.arctan2(d[:, 0], d[:, 1]), sp


def build_features(
    net: Network,
    P: np.ndarray,
    ts: np.ndarray,
    cand: np.ndarray,
    dist: np.ndarray,
    tpos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build candidate tabular feature matrix.
    
    Returns:
        X: (num_candidate_rows, 15) float32 feature matrix.
        rows_i: Observation index for each candidate row.
        rows_k: Rank index within candidate set [0..k-1].
        arc: Arc ID for each candidate row.
    """
    N, k = cand.shape
    hb, sp = bearing_from_track(P)
    dt = np.r_[1.0, np.diff(ts)]
    dt = np.minimum(dt, 60.0)
    denom = np.maximum(1.0, np.where(np.r_[True, np.diff(ts) > 0], np.r_[1.0, np.diff(ts)], 1.0))
    spd = sp / denom

    valid = cand >= 0
    rows_i, rows_k = np.nonzero(valid)
    a = cand[rows_i, rows_k]
    dmin = dist[:, 0]
    d = dist[rows_i, rows_k]

    bd = hb[rows_i] - net.bear[a]
    bd = (bd + np.pi) % (2 * np.pi) - np.pi
    moving = (spd[rows_i] > 1.0).astype(float)
    cosb = np.cos(bd) * moving

    ab = net.B[a] - net.A[a]
    ap = P[rows_i] - net.A[a]
    cross = (ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]) / np.maximum(net.len[a], 1e-6)
    ncand = valid.sum(axis=1)

    # 1-step lookahead connectivity
    prev_conn = np.zeros(len(a), dtype=np.float32)
    next_conn = np.zeros(len(a), dtype=np.float32)
    to_node = net.arcs[:, 1]
    fr_node = net.arcs[:, 0]

    for r, (i, kk) in enumerate(zip(rows_i, rows_k)):
        if i > 0:
            pc = cand[i - 1]
            pc = pc[pc >= 0]
            if len(pc) > 0:
                prev_conn[r] = float(np.any((pc == a[r]) | (to_node[pc] == fr_node[a[r]])))
        if i < N - 1:
            nc = cand[i + 1]
            nc = nc[nc >= 0]
            if len(nc) > 0:
                next_conn[r] = float(np.any((nc == a[r]) | (fr_node[nc] == to_node[a[r]])))

    nxt = np.where(rows_k + 1 < k, dist[rows_i, np.minimum(rows_k + 1, k - 1)], 200.0)
    nxt = np.where(np.isfinite(nxt), nxt, 200.0)
    gap = nxt - d

    X = np.column_stack([
        d,
        rows_k.astype(np.float32),
        d - dmin[rows_i],
        gap,
        tpos[rows_i, rows_k],
        net.len[a],
        cosb,
        np.abs(bd) * moving,
        spd[rows_i],
        ncand[rows_i].astype(np.float32),
        d * cosb,
        cross,
        prev_conn,
        next_conn,
        dt[rows_i],
    ])

    return X.astype(np.float32), rows_i, rows_k, a


def trip_table(root: str, tid: str, with_route: bool = True) -> Dict[str, Any]:
    """Generate complete candidate representation and features for a trip."""
    tr, nd, ar, rt = load_trip(root, tid, with_route=with_route)
    proj = Proj(tr[:, 0].mean(), tr[:, 1].mean())
    net = Network(nd, ar, proj)
    P = proj(tr[:, 0], tr[:, 1])
    ts = tr[:, 2]

    cand, dist, tpos = candidates(net, P)
    X, ri, rk, a = build_features(net, P, ts, cand, dist, tpos)

    out = {
        "tid": tid,
        "net": net,
        "tr": tr,
        "nd": nd,
        "ar": ar,
        "P": P,
        "ts": ts,
        "cand": cand,
        "dist": dist,
        "X": X,
        "ri": ri,
        "rk": rk,
        "arc": a,
        "route": rt,
    }

    if rt is not None and len(rt) > 0:
        ref, rcost = align_labels(net, P, rt)
        out["ref"] = ref
        out["ref_cost"] = rcost
        out["y"] = (a == ref[ri]).astype(np.int8)

    return out
