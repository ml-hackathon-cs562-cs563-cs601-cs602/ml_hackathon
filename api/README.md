# PathSnap API Service

FastAPI inference and evaluation backend (running on Laptop 1).
Owner: Member A & C

## Starting the API
To run the server:
```bash
# Optional: specify custom evaluation trips folder
# export EVAL_DATA_DIR=/path/to/eval_trips (Linux/macOS)
# $env:EVAL_DATA_DIR="path/to/eval_trips" (PowerShell)

uvicorn api.main:app --host 0.0.0.0 --port 8000
# or
python api/main.py
```

## Endpoints
- `GET /health`: Health status, active data directory, and number of available trips.
- `GET /trips`: List of available trip IDs (no ground-truth labels exposed).
- `POST /predict`: Takes JSON `{"trip_id": "<id>"}`, runs the candidate + feature + XGBoost + HMM/Viterbi pipeline, and returns predicted route polyline, predicted arc per point, and metrics if ground-truth is available on the server.
- `POST /evaluate/{trip_id}`: Evaluates the prediction against server-side ground truth in `api/server_data/` and returns precision, recall, F1, and top-1 accuracy.

## Request Logs
All incoming requests and inference durations are logged to standard output:
```
2026-11-18 12:50:00 [INFO] pathsnap.routes: POST /predict trip_id=00000007 -> running pipeline
```
