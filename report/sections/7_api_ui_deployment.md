# 7. System Deployment & Evaluation Protocol

## 7.1 Two-Laptop Distributed Architecture
For the final live evaluation, the system is deployed across two independent machines communicating over local Wi-Fi:

```
[ Laptop 2: Evaluator Client UI ]
  |-- Web Browser (http://localhost:8080)
  |-- Leaflet.js Interactive Map
  |-- Calls POST /predict {trip_id}
          |
     (Local Wi-Fi / HTTP JSON)
          |
          v
[ Laptop 1: PathSnap API & Evaluation Service ]
  |-- FastAPI Server (http://<laptop1-ip>:8000)
  |-- Candidate Generator + Tuned XGBoost + HMM/Viterbi
  |-- Evaluator ground truth isolated in api/server_data/
  |-- Terminal displays real-time request logs and timings
```

### Laptop 1 (Inference Backend)
- **Startup Command:**
  ```bash
  EVAL_DATA_DIR="/path/to/evaluator_trips" uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
- **Terminal Request Logging:**
  ```
  2026-11-18 12:50:01 [INFO] pathsnap.routes: POST /predict trip_id=00000007 -> running pipeline
  2026-11-18 12:50:02 [INFO] pathsnap.pipeline: trip 00000007: 154 GPS pts, 0.082s inference
  ```

### Laptop 2 (Evaluator Client UI)
- **Startup Command:**
  ```bash
  python ui/app.py
  ```
- Serves the static Leaflet map client at `http://localhost:8080`.
- Input field connects to `http://<laptop1-ip>:8000`.
- "Load trips" button queries `GET /trips` and populates evaluator trip buttons.
- Selecting any trip triggers `POST /predict`, dynamically drawing raw GPS points (blue markers) and the predicted reconstructed polyline (red line).

## 7.2 Strict Data Isolation Compliance
1. **Zero Ground-Truth Leakage:** The prediction pipeline in `api/pipeline.py` loads `load_trip(..., with_route=False)`. Reference route files (`.route`) are never read during prediction.
2. **Server-Side Evaluation:** Evaluator ground-truth routes reside strictly on Laptop 1 in `api/server_data/`. Only `api/evaluation_service.py` reads ground truth to compute evaluation metrics.
3. **Reproducibility Confirmation:** All code in this monorepo matches the demonstrated system running during the evaluation viva.
