"""API routes for PathSnap inference and evaluation service.

GET /trips returns the available trip IDs (no labels).
POST /predict runs the full prediction pipeline on a trip ID without reference route.
POST /evaluate/{trip_id} runs prediction and scores against server-side ground truth.
Owner: Member A & C
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import evaluation_service, pipeline

router = APIRouter()
log = logging.getLogger("pathsnap.api")

ROOT = Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    env_dir = os.environ.get("EVAL_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p
    eval_trips = ROOT / "data" / "eval_trips"
    if eval_trips.exists() and any(eval_trips.iterdir()):
        return eval_trips
    server_data = ROOT / "api" / "server_data"
    if server_data.exists() and any(d.is_dir() for d in server_data.iterdir()):
        return server_data
    # Fallback to student test set during development
    student_test = ROOT / "data" / "raw" / "student_test"
    if student_test.exists():
        return student_test
    return ROOT / "data" / "raw" / "train"


class PredictRequest(BaseModel):
    trip_id: str


@router.get("/health")
def health() -> Dict[str, Any]:
    data_dir = get_data_dir()
    trips = pipeline.list_trips(str(data_dir))
    return {
        "status": "ok",
        "data_dir": str(data_dir),
        "n_trips": len(trips),
    }


@router.get("/trips")
def list_trips() -> Dict[str, List[str]]:
    data_dir = get_data_dir()
    trips = pipeline.list_trips(str(data_dir))
    return {"trips": trips}


@router.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    tid = req.trip_id
    data_dir = get_data_dir()
    available = pipeline.list_trips(str(data_dir))
    if tid not in available:
        raise HTTPException(status_code=404, detail=f"Unknown trip {tid} in {data_dir}")

    log.info("POST /predict trip_id=%s -> running pipeline", tid)
    res = pipeline.predict(str(data_dir), tid)
    sc = evaluation_service.score_prediction(str(data_dir), tid, res)

    response = {
        "trip_id": tid,
        "n_gps": res["n"],
        "inference_seconds": round(res["secs"], 3),
        "gps": [[float(a), float(b)] for a, b in res["tr"][:, [1, 0]]],  # [lat, lon]
        "predicted_route": pipeline.route_polyline(res),
        "predicted_arc_per_point": [int(a) for a in res["pred"]],
    }

    if sc is not None:
        response["metrics"] = {
            "classifier_only": sc["clf"],
            "classifier_plus_hmm": sc["pred"],
            "route_arc_precision": sc["route_arc_precision"],
            "route_arc_recall": sc["route_arc_recall"],
            "route_arc_f1": sc["route_arc_f1"],
        }
    return response


@router.post("/evaluate/{trip_id}")
def evaluate(trip_id: str) -> Dict[str, Any]:
    data_dir = get_data_dir()
    available = pipeline.list_trips(str(data_dir))
    if trip_id not in available:
        raise HTTPException(status_code=404, detail=f"Unknown trip {trip_id} in {data_dir}")

    log.info("POST /evaluate/%s", trip_id)
    res = pipeline.predict(str(data_dir), trip_id)
    sc = evaluation_service.score_prediction(str(data_dir), trip_id, res)

    return {
        "trip_id": trip_id,
        "n_gps": res["n"],
        "inference_seconds": round(res["secs"], 3),
        "predicted_arc_per_point": [int(a) for a in res["pred"]],
        "metrics": sc,
    }
