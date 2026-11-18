"""Inference pipeline for PathSnap FastAPI backend.

Owner: Member A & B
"""

from pathlib import Path
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pathsnap.candidates.generate import candidates
from pathsnap.features.build import build_features
from pathsnap.geo.projection import Proj
from pathsnap.hmm.emissions import prob_matrix
from pathsnap.hmm.viterbi import decode
from pathsnap.ingest.data_loader import Network
from pathsnap.ingest.parsers import load_trip

# Model paths
MODEL_PATH = ROOT / "models" / "final_model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "artifacts" / "randomforest.joblib"

HMM_PARAMS_PATH = ROOT / "models" / "hmm_params.json"


def get_model():
    if MODEL_PATH.suffix == ".pkl":
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        import joblib
        return joblib.load(MODEL_PATH)


def get_hmm_params() -> dict:
    if HMM_PARAMS_PATH.exists():
        with open(HMM_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    from pathsnap.hmm.viterbi import DEFAULT_PARAMS
    return DEFAULT_PARAMS


_MODEL = None
_HMM_PARAMS = None


def init_artifacts():
    global _MODEL, _HMM_PARAMS
    if _MODEL is None:
        _MODEL = get_model()
    if _HMM_PARAMS is None:
        _HMM_PARAMS = get_hmm_params()


def list_trips(root: str) -> List[str]:
    """List available trip IDs in data directory."""
    p = Path(root)
    if not p.exists():
        return []
    return sorted(
        d.name
        for d in p.iterdir()
        if d.is_dir() and (d / f"{d.name}.track").exists()
    )


def predict(root: str, tid: str) -> Dict[str, Any]:
    """Run candidate generation, feature engineering, classification, and HMM/Viterbi.
    
    Reference route is NEVER read during prediction.
    """
    init_artifacts()
    t0 = time.perf_counter()

    # Load without route
    tr, nd, ar, _ = load_trip(root, tid, with_route=False)
    proj = Proj(tr[:, 0].mean(), tr[:, 1].mean())
    net = Network(nd, ar, proj)
    P = proj(tr[:, 0], tr[:, 1])
    ts = tr[:, 2]

    cand, dist, tpos = candidates(net, P)
    X, ri, rk, a = build_features(net, P, ts, cand, dist, tpos)

    # Positive class probabilities
    p = _MODEL.predict_proba(X)
    p_pos = p[:, 1] if p.ndim == 2 else p

    tab = {"P": P, "cand": cand, "ri": ri, "rk": rk, "arc": a}
    from pathsnap.evaluation.metrics import compute_observation_predictions
    c_pred = np.array([
        a[ix[np.argmax(p_pos[ix])]] if len(ix) > 0 else -1
        for ix in [np.flatnonzero(ri == i) for i in range(len(P))]
    ], dtype=int)

    pm = prob_matrix(tab, p_pos)
    pred_hmm = decode(ar, cand, pm, ts, _HMM_PARAMS)

    return {
        "tr": tr,
        "nd": nd,
        "ar": ar,
        "net": net,
        "clf": c_pred,
        "pred": pred_hmm,
        "secs": time.perf_counter() - t0,
        "n": len(P),
    }


def route_polyline(res: dict) -> list:
    """Format ordered, de-duplicated predicted arcs as [lat, lon] line segments."""
    pred = res["pred"]
    valid = pred[pred >= 0]
    if len(valid) == 0:
        return []
    seq = valid[np.r_[True, valid[1:] != valid[:-1]]]
    nd, ar = res["nd"], res["ar"]
    return [
        [
            [float(nd[ar[a, 0], 1]), float(nd[ar[a, 0], 0])],
            [float(nd[ar[a, 1], 1]), float(nd[ar[a, 1], 0])],
        ]
        for a in seq
        if 0 <= a < len(ar)
    ]
