"""API routes with model selection support for PathSnap.

Endpoints:
- GET /models - List available models with metrics
- POST /select_model - Select active model for predictions
- GET /trips - List available trips
- POST /predict - Run prediction with selected model
- POST /evaluate/{trip_id} - Evaluate prediction against ground truth

Owner: Member D
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import evaluation_service, pipeline

router = APIRouter()
log = logging.getLogger("pathsnap.api")

ROOT = Path(__file__).resolve().parents[1]

# Available models configuration
AVAILABLE_MODELS = [
    {
        "name": "lightgbm",
        "display_name": "LightGBM",
        "family": "Gradient Boosting",
        "path": "results/model4/outputs_lgbm/artifacts/lgbm_untuned.txt",
        "scaler": None,
        "metrics": {"top1": 0.848, "f1": 0.904, "log_loss": 0.104},
        "calibrated": False,
    },
    {
        "name": "mlp",
        "display_name": "MLP Neural Network",
        "family": "Neural Network",
        "path": "results/model5/outputs_mlp/artifacts/mlp_untuned.joblib",
        "scaler": "results/model5/outputs_mlp/artifacts/scaler.joblib",
        "metrics": {"top1": 0.848, "f1": 0.911, "log_loss": 0.098},
        "calibrated": False,
    },
    {
        "name": "randomforest",
        "display_name": "Random Forest",
        "family": "Tree Ensemble (Bagging)",
        "path": "artifacts/randomforest_config.json",
        "scaler": None,
        "metrics": {"top1": 0.690, "f1": 0.815, "log_loss": 0.222},
        "calibrated": False,
    },
    {
        "name": "logisticregression",
        "display_name": "Logistic Regression",
        "family": "Linear Model",
        "path": "artifacts/logisticregression_config.json",
        "scaler": None,
        "metrics": {"top1": 0.672, "f1": 0.733, "log_loss": 0.348},
        "calibrated": False,
    },
    {
        "name": "decisiontree",
        "display_name": "Decision Tree",
        "family": "Tree-Based",
        "path": "artifacts/decisiontree_config.json",
        "scaler": None,
        "metrics": {"top1": 0.676, "f1": 0.788, "log_loss": 0.397},
        "calibrated": False,
    },
]

# Selected model (default: LightGBM as best performer)
SELECTED_MODEL = "lightgbm"


def get_data_dir() -> Path:
    """Get data directory based on environment."""
    import os
    env_dir = os.environ.get("EVAL_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p
    
    # Check common locations
    for loc in [
        ROOT / "data" / "raw" / "evaluator_test",
        ROOT / "api" / "server_data",
        ROOT / "data" / "raw" / "student_test",
        ROOT / "data" / "raw" / "train",
    ]:
        if loc.exists() and any(loc.iterdir()):
            return loc
    
    return ROOT / "data" / "raw" / "student_test"


class PredictRequest(BaseModel):
    trip_id: str
    model: Optional[str] = None


class ModelSelectRequest(BaseModel):
    model: str


@router.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    data_dir = get_data_dir()
    trips = pipeline.list_trips(str(data_dir))
    return {
        "status": "ok",
        "data_dir": str(data_dir),
        "n_trips": len(trips),
        "selected_model": SELECTED_MODEL,
        "available_models": len(AVAILABLE_MODELS),
    }


@router.get("/models")
def list_models() -> Dict[str, List[Dict]]:
    """List all available models with their metrics."""
    return {"models": AVAILABLE_MODELS}


@router.post("/select_model")
def select_model(req: ModelSelectRequest) -> Dict[str, Any]:
    """Select the active model for predictions."""
    global SELECTED_MODEL
    
    model_names = [m["name"] for m in AVAILABLE_MODELS]
    if req.model not in model_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: {model_names}"
        )
    
    SELECTED_MODEL = req.model
    log.info(f"Selected model changed to: {SELECTED_MODEL}")
    
    return {
        "status": "ok",
        "selected_model": SELECTED_MODEL,
        "message": f"Model '{SELECTED_MODEL}' selected for predictions"
    }


@router.get("/selected_model")
def get_selected_model() -> Dict[str, Any]:
    """Get the currently selected model."""
    model_info = next(
        (m for m in AVAILABLE_MODELS if m["name"] == SELECTED_MODEL),
        AVAILABLE_MODELS[0]
    )
    return {"selected_model": model_info}


@router.get("/trips")
def list_trips() -> Dict[str, List[str]]:
    """List available trip IDs."""
    data_dir = get_data_dir()
    trips = pipeline.list_trips(str(data_dir))
    return {"trips": trips}


@router.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """Run prediction on a trip with the selected model."""
    tid = req.trip_id
    model_name = req.model or SELECTED_MODEL
    
    data_dir = get_data_dir()
    available = pipeline.list_trips(str(data_dir))
    if tid not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Trip '{tid}' not found in {data_dir}"
        )
    
    log.info(f"POST /predict trip_id={tid} model={model_name}")
    
    # Get model info
    model_info = next(
        (m for m in AVAILABLE_MODELS if m["name"] == model_name),
        None
    )
    if not model_info:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    # Run prediction
    res = pipeline.predict(str(data_dir), tid, model_name)
    
    # Score if ground truth available
    sc = evaluation_service.score_prediction(str(data_dir), tid, res)
    
    # Build response
    response = {
        "trip_id": tid,
        "model_name": model_info["display_name"],
        "n_gps": res["n"],
        "inference_seconds": round(res["secs"], 3),
        "gps": [[float(a), float(b)] for a, b in res["tr"][:, [1, 0]]],  # [lat, lon]
        "predicted_route": pipeline.route_polyline(res),
        "predicted_arc_per_point": [int(a) for a in res["pred"]],
    }
    
    if sc is not None:
        response["metrics"] = {
            "top1_classifier": sc.get("clf", {}).get("top1", 0),
            "top1_hmm": sc.get("pred", {}).get("top1", 0),
            "macro_f1": sc.get("pred", {}).get("macro_f1", 0),
            "route_continuity": sc.get("route_continuity", 0),
            "log_loss": sc.get("pred", {}).get("log_loss", 0),
        }
    
    return response


@router.post("/evaluate/{trip_id}")
def evaluate(trip_id: str) -> Dict[str, Any]:
    """Run prediction and evaluate against ground truth (server-side only)."""
    data_dir = get_data_dir()
    available = pipeline.list_trips(str(data_dir))
    if trip_id not in available:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found")
    
    log.info(f"POST /evaluate/{trip_id}")
    res = pipeline.predict(str(data_dir), trip_id, SELECTED_MODEL)
    sc = evaluation_service.score_prediction(str(data_dir), trip_id, res)
    
    return {
        "trip_id": trip_id,
        "model": SELECTED_MODEL,
        "n_gps": res["n"],
        "inference_seconds": round(res["secs"], 3),
        "predicted_arc_per_point": [int(a) for a in res["pred"]],
        "metrics": sc,
    }
