"""Inference pipeline with multi-model support for PathSnap FastAPI backend.

Supports:
- LightGBM (gradient boosting)
- MLP (neural network)
- Random Forest
- Logistic Regression
- Decision Tree

Owner: Member A, B, D
"""

from pathlib import Path
import json
import os
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
from pathsnap.features.build import build_features, FEATS
from pathsnap.geo.projection import Proj
from pathsnap.hmm.emissions import prob_matrix
from pathsnap.hmm.viterbi import decode
from pathsnap.ingest.data_loader import Network
from pathsnap.ingest.parsers import load_trip

# Model paths configuration
MODEL_PATHS = {
    "lightgbm": ROOT / "results" / "model4" / "outputs_lgbm" / "artifacts" / "lgbm_untuned.txt",
    "mlp": ROOT / "results" / "model5" / "outputs_mlp" / "artifacts" / "mlp_untuned.joblib",
    "randomforest": ROOT / "artifacts" / "randomforest.joblib",
    "logisticregression": ROOT / "artifacts" / "logisticregression.joblib",
    "decisiontree": ROOT / "artifacts" / "decisiontree.joblib",
}

SCALER_PATHS = {
    "mlp": ROOT / "results" / "model5" / "outputs_mlp" / "artifacts" / "scaler.joblib",
}

HMM_PARAMS_PATH = ROOT / "models" / "hmm_params.json"

# Loaded models cache
_LOADED_MODELS = {}
_LOADED_SCALERS = {}
_HMM_PARAMS = None


def load_model(model_name: str):
    """Load a model by name."""
    import joblib
    
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]
    
    # Find model path
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not model_path.exists():
        # Try fallback paths
        fallback = ROOT / "artifacts" / f"{model_name}.joblib"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model not found: {model_name}")
    
    log_msg = f"Loading model: {model_path}"
    print(log_msg)
    
    if model_path.suffix == ".txt":
        # LightGBM model
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
    elif model_path.suffix in [".joblib", ".pkl"]:
        # sklearn model
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    
    _LOADED_MODELS[model_name] = model
    return model


def load_scaler(model_name: str):
    """Load scaler for a model (if needed)."""
    import joblib
    
    if model_name in _LOADED_SCALERS:
        return _LOADED_SCALERS[model_name]
    
    scaler_path = SCALER_PATHS.get(model_name)
    if not scaler_path or not scaler_path.exists():
        return None
    
    scaler = joblib.load(scaler_path)
    _LOADED_SCALERS[model_name] = scaler
    return scaler


def get_hmm_params() -> dict:
    """Load HMM parameters."""
    global _HMM_PARAMS
    if _HMM_PARAMS is None:
        if HMM_PARAMS_PATH.exists():
            with open(HMM_PARAMS_PATH, "r") as f:
                _HMM_PARAMS = json.load(f)
        else:
            from pathsnap.hmm.viterbi import DEFAULT_PARAMS
            _HMM_PARAMS = DEFAULT_PARAMS
    return _HMM_PARAMS


def init_artifacts():
    """Initialize all artifacts (called at startup)."""
    # Pre-load default model
    try:
        load_model("lightgbm")
        print("Default model (LightGBM) loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
    
    # Load HMM params
    get_hmm_params()
    print("HMM parameters loaded")


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


def predict(root: str, tid: str, model_name: str = "lightgbm") -> Dict[str, Any]:
    """Run the full prediction pipeline.
    
    Args:
        root: Data directory path
        tid: Trip ID
        model_name: Model to use for prediction
        
    Returns:
        Dictionary with prediction results
    """
    t0 = time.perf_counter()
    
    # Load model and scaler
    model = load_model(model_name)
    scaler = load_scaler(model_name)
    hmm_params = get_hmm_params()
    
    # Load trip data (without reference route)
    tr, nd, ar, _ = load_trip(root, tid, with_route=False)
    proj = Proj(tr[:, 0].mean(), tr[:, 1].mean())
    net = Network(nd, ar, proj)
    P = proj(tr[:, 0], tr[:, 1])
    ts = tr[:, 2]
    
    # Generate candidates and features
    cand, dist, tpos = candidates(net, P)
    X, ri, rk, a = build_features(net, P, ts, cand, dist, tpos)
    
    # Scale features if needed
    if scaler is not None:
        X = scaler.transform(X)
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        # sklearn-style model
        p = model.predict_proba(X)
        p_pos = p[:, 1] if p.ndim == 2 else p
    elif hasattr(model, 'predict'):
        # LightGBM
        p_pos = model.predict(X, num_iteration=model.best_iteration)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
    # Classifier-only predictions (argmax per GPS point)
    tab = {"P": P, "cand": cand, "ri": ri, "rk": rk, "arc": a}
    from pathsnap.evaluation.metrics import compute_observation_predictions
    c_pred = np.array([
        a[ix[np.argmax(p_pos[ix])]] if len(ix) > 0 else -1
        for ix in [np.flatnonzero(ri == i) for i in range(len(P))]
    ], dtype=int)
    
    # HMM/Viterbi decoding
    pm = prob_matrix(tab, p_pos)
    pred_hmm = decode(ar, cand, pm, ts, hmm_params)
    
    elapsed = time.perf_counter() - t0
    
    return {
        "tr": tr,
        "nd": nd,
        "ar": ar,
        "net": net,
        "clf": c_pred,
        "pred": pred_hmm,
        "probs": p_pos,
        "secs": elapsed,
        "n": len(P),
    }


def route_polyline(res: dict) -> list:
    """Format predicted arcs as [lat, lon] line segments."""
    pred = res["pred"]
    valid = pred[pred >= 0]
    if len(valid) == 0:
        return []
    
    # Remove consecutive duplicates
    seq = valid[np.r_[True, valid[1:] != valid[:-1]]]
    nd, ar = res["nd"], res["ar"]
    
    return [
        [
            [float(nd[ar[a, 0], 1]), float(nd[ar[a, 0], 0])],  # start (lat, lon)
            [float(nd[ar[a, 1], 1]), float(nd[ar[a, 1], 0])],  # end (lat, lon)
        ]
        for a in seq
        if 0 <= a < len(ar)
    ]
