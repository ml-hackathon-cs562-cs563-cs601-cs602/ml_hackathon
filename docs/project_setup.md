# PathSnap Project Setup Guide

**Complete guide for running the backend (FastAPI) and frontend (UI) on two laptops connected via WiFi**

**Last Updated:** November 2026  
**Author:** Member D

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Preparing Model Artifacts](#4-preparing-model-artifacts)
5. [Laptop 1: Backend Server Setup](#5-laptop-1-backend-server-setup)
6. [Laptop 2: Frontend Client Setup](#6-laptop-2-frontend-client-setup)
7. [Connecting Both Laptops](#7-connecting-both-laptops)
8. [Running the Application](#8-running-the-application)
9. [Testing the Setup](#9-testing-the-setup)
10. [Troubleshooting](#10-troubleshooting)
11. [API Endpoints Reference](#11-api-endpoints-reference)
12. [Model Selection Guide](#12-model-selection-guide)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Requirement |
|-----------|------------|
| Laptops | 2 laptops (can be different OS) |
| Network | Same WiFi network (both laptops connected) |
| RAM | Minimum 8GB (16GB recommended) |
| Storage | 2GB free space |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime environment |
| Git | Latest | Version control |
| Web Browser | Chrome/Firefox/Edge | UI access |

---

## 2. Project Structure

```
ml_hackathon/
├── api/                          # FastAPI backend
│   ├── main.py                   # Application entry point
│   ├── routes.py                 # API routes
│   ├── routes_model_selection.py # Routes with model selection
│   ├── pipeline.py               # Inference pipeline
│   ├── pipeline_multi_model.py   # Multi-model pipeline
│   └── evaluation_service.py     # Ground truth evaluation
│
├── ui/                           # Frontend
│   ├── index.html                # Basic UI
│   └── app_model_selection.html  # UI with model selection
│
├── src/pathsnap/                 # Core library
│   ├── candidates/               # Candidate generation
│   ├── features/                 # Feature extraction
│   ├── hmm/                      # HMM/Viterbi decoder
│   ├── models/                   # Model wrappers
│   └── evaluation/               # Metrics
│
├── models/                       # Model code
│   ├── lgbm/                     # LightGBM
│   ├── mlp/                      # MLP
│   ├── rf/                       # Random Forest
│   ├── logreg/                   # Logistic Regression
│   └── dtree/                    # Decision Tree
│
├── results/                      # Training results
│   ├── model4/                   # LightGBM outputs
│   │   └── outputs_lgbm/
│   │       └── artifacts/
│   │           ├── lgbm_untuned.txt
│   │           └── lgbm_feature_names.json
│   └── model5/                   # MLP outputs
│       └── outputs_mlp.zip       # Extract this!
│
├── artifacts/                    # Final model artifacts
│   └── (place trained models here)
│
├── data/raw/                     # Raw trip data
│   ├── student_test/             # 10 student test trips
│   └── train/                    # 80 training trips
│
├── configs/                      # Configuration files
│   ├── partitions.yaml           # Trip partitions
│   └── models/                   # Model configs
│
├── requirements.txt              # Python dependencies
└── docs/                         # Documentation
```

---

## 3. Environment Setup

### Step 1: Clone the Repository

```bash
# On both laptops
git clone https://github.com/YOUR_TEAM/ml_hackathon.git
cd ml_hackathon
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Additional packages for LightGBM (if needed)
pip install lightgbm

# Verify installation
python -c "import fastapi; import lightgbm; import sklearn; print('All packages installed!')"
```

---

## 4. Preparing Model Artifacts

### Step 1: Extract MLP Model

The MLP model is in a zip file. Extract it:

```bash
cd ml_hackathon
unzip results/model5/outputs_mlp.zip -d results/model5/
```

This should create:
- `results/model5/outputs_mlp/artifacts/mlp_untuned.joblib`
- `results/model5/outputs_mlp/artifacts/scaler.joblib`

### Step 2: Verify Model Files

Check that all model files exist:

```bash
# LightGBM
ls -la results/model4/outputs_lgbm/artifacts/lgbm_untuned.txt

# MLP
ls -la results/model5/outputs_mlp/artifacts/mlp_untuned.joblib
ls -la results/model5/outputs_mlp/artifacts/scaler.joblib
```

### Step 3: Copy to Artifacts Directory (Optional)

```bash
# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Copy models
cp results/model4/outputs_lgbm/artifacts/lgbm_untuned.txt artifacts/
cp results/model5/outputs_mlp/artifacts/mlp_untuned.joblib artifacts/
cp results/model5/outputs_mlp/artifacts/scaler.joblib artifacts/
```

---

## 5. Laptop 1: Backend Server Setup

Laptop 1 will run the FastAPI backend with access to models and data.

### Step 1: Get Laptop 1's IP Address

```bash
# On Linux/Mac
hostname -I
# or
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows
ipconfig
# Look for "IPv4 Address" under your WiFi adapter
```

**Note down this IP address.** Example: `192.168.1.100`

### Step 2: Configure Environment Variables (Optional)

```bash
# Set the data directory (if using evaluator data)
export EVAL_DATA_DIR=/path/to/evaluator/data

# Set API host and port
export API_HOST=0.0.0.0
export API_PORT=8000
```

### Step 3: Start the FastAPI Server

**Option A: Using uvicorn directly**

```bash
cd ml_hackathon
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Option B: Using Python script**

```bash
cd ml_hackathon
python api/main.py
```

**Option C: With auto-reload (for development)**

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Verify Server is Running

Open a browser on Laptop 1 and go to:
- `http://localhost:8000/health`
- `http://localhost:8000/models`

You should see JSON responses.

---

## 6. Laptop 2: Frontend Client Setup

Laptop 2 will run the UI that connects to Laptop 1's API.

### Step 1: Open the UI

**Option A: Direct file open**

1. Navigate to `ml_hackathon/ui/`
2. Open `app_model_selection.html` in a web browser

**Option B: Using a simple HTTP server (recommended)**

```bash
cd ml_hackathon/ui
python -m http.server 8080
```

Then open `http://localhost:8080/app_model_selection.html` in browser.

### Step 2: Configure API Address

In the UI:
1. Find the "API Server Address" field
2. Enter Laptop 1's IP address with port 8000:
   ```
   http://192.168.1.100:8000
   ```
   (Replace `192.168.1.100` with your actual Laptop 1 IP)

3. Click "Connect"

---

## 7. Connecting Both Laptops

### Network Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         WiFi Router                          │
│                    (192.168.1.1)                             │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
              │                           │
    ┌─────────▼─────────┐       ┌─────────▼─────────┐
    │     Laptop 1      │       │     Laptop 2      │
    │   (Server)        │       │   (Client)        │
    │                   │       │                   │
    │ IP: 192.168.1.100 │       │ IP: 192.168.1.101 │
    │ Port: 8000        │◄──────┤                   │
    │                   │  API  │                   │
    │ - FastAPI         │  calls│ - Browser UI      │
    │ - Models          │       │ - Model selection │
    │ - HMM/Viterbi     │       │ - Trip selection  │
    │ - Ground truth    │       │ - Visualization   │
    └───────────────────┘       └───────────────────┘
```

### Step 1: Verify Both Laptops on Same Network

**On Laptop 1:**
```bash
# Check if you can ping Laptop 2
ping 192.168.1.101
```

**On Laptop 2:**
```bash
# Check if you can ping Laptop 1
ping 192.168.1.100
```

Both should respond successfully.

### Step 2: Test API Connection from Laptop 2

**On Laptop 2, open terminal:**
```bash
# Test API health endpoint
curl http://192.168.1.100:8000/health

# Or use browser
# Open: http://192.168.1.100:8000/health
```

### Step 3: Firewall Configuration

If the connection fails, check firewall settings:

**On Linux (Ubuntu/Debian):**
```bash
# Allow port 8000
sudo ufw allow 8000/tcp

# Check status
sudo ufw status
```

**On Windows:**
1. Open "Windows Defender Firewall"
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Click "Allow another app"
4. Browse to Python executable and add it
5. Ensure "Private" is checked

**On macOS:**
1. Go to "System Preferences" > "Security & Privacy" > "Firewall"
2. Click "Firewall Options"
3. Add Python to allowed apps

---

## 8. Running the Application

### Complete Startup Sequence

#### On Laptop 1 (Server):

```bash
# 1. Open terminal
# 2. Navigate to project
cd ml_hackathon

# 3. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 4. Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Look for startup messages:
# INFO:     Started server process
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### On Laptop 2 (Client):

```bash
# 1. Open terminal (optional, for HTTP server)
# 2. Navigate to UI directory
cd ml_hackathon/ui

# 3. Start HTTP server (optional)
python -m http.server 8080

# 4. Open browser to:
# http://localhost:8080/app_model_selection.html
# Or directly open the file in browser
```

### Using the UI

1. **Connect to API**
   - Enter Laptop 1's IP: `http://192.168.1.100:8000`
   - Click "Connect"
   - Status should show "Connected"

2. **Select a Model**
   - Available models are displayed with metrics
   - Click on a model card to select it
   - Recommended: **LightGBM** (best performance) or **MLP** (lowest log loss)

3. **Select a Trip**
   - Click on any trip ID from the list
   - Prediction starts automatically

4. **View Results**
   - Map shows GPS points (blue) and predicted route (red)
   - Metrics panel shows Top-1, Macro F1, etc.

---

## 9. Testing the Setup

### Test 1: API Health Check

```bash
# On Laptop 1
curl http://localhost:8000/health

# Expected response:
{
  "status": "ok",
  "data_dir": "/path/to/data",
  "n_trips": 10,
  "selected_model": "lightgbm"
}
```

### Test 2: List Models

```bash
curl http://localhost:8000/models

# Expected response:
{
  "models": [
    {
      "name": "lightgbm",
      "display_name": "LightGBM",
      "family": "Gradient Boosting",
      "metrics": {...}
    },
    ...
  ]
}
```

### Test 3: List Trips

```bash
curl http://localhost:8000/trips

# Expected response:
{
  "trips": ["00000000", "00000007", "00000009", ...]
}
```

### Test 4: Run Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_id": "00000000", "model": "lightgbm"}'

# Expected response:
{
  "trip_id": "00000000",
  "model_name": "LightGBM",
  "n_gps": 2600,
  "inference_seconds": 1.234,
  "gps": [[lat, lon], ...],
  "predicted_route": [[[lat, lon], [lat, lon]], ...],
  "metrics": {...}
}
```

### Test 5: Cross-Machine Connection

**On Laptop 2:**
```bash
# Replace with actual IP
curl http://192.168.1.100:8000/health
```

---

## 10. Troubleshooting

### Problem 1: "Connection refused" or "Cannot reach API"

**Causes:**
- API server not running
- Wrong IP address
- Firewall blocking port 8000

**Solutions:**
```bash
# Check if API is running
ps aux | grep uvicorn

# Check if port is open
netstat -an | grep 8000

# Test locally first
curl http://localhost:8000/health

# Check firewall
sudo ufw status  # Linux
```

### Problem 2: "Model not found" error

**Causes:**
- Model files not in expected location
- MLP zip not extracted

**Solutions:**
```bash
# Verify LightGBM model
ls -la results/model4/outputs_lgbm/artifacts/

# Extract MLP if needed
unzip results/model5/outputs_mlp.zip -d results/model5/
```

### Problem 3: "Unknown trip" error

**Causes:**
- Trip data not in `data/raw/` directory
- Wrong data directory

**Solutions:**
```bash
# Check available trips
ls -la data/raw/student_test/

# Verify data directory in API
curl http://localhost:8000/health
# Check "data_dir" field
```

### Problem 4: CORS error in browser

**Causes:**
- CORS middleware not configured

**Solutions:**

The API already has CORS enabled. If you still see errors:

```python
# In api/main.py, verify this is present:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Problem 5: Slow predictions

**Causes:**
- Large trip (many GPS points)
- Running on CPU

**Solutions:**
- Use LightGBM (fastest inference)
- Consider subsampling for testing
- Check available RAM

### Problem 6: ImportError for LightGBM

**Solution:**
```bash
pip install lightgbm

# If installation fails on Linux:
sudo apt-get install -y libgomp1
pip install lightgbm
```

---

## 11. API Endpoints Reference

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List all available models |
| POST | `/select_model` | Select active model |
| GET | `/selected_model` | Get currently selected model |
| GET | `/trips` | List available trip IDs |
| POST | `/predict` | Run prediction on a trip |
| POST | `/evaluate/{trip_id}` | Evaluate prediction against ground truth |

### Endpoint Details

#### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "data_dir": "/path/to/data",
  "n_trips": 10,
  "selected_model": "lightgbm"
}
```

#### GET /models

```bash
curl http://localhost:8000/models
```

Response:
```json
{
  "models": [
    {
      "name": "lightgbm",
      "display_name": "LightGBM",
      "family": "Gradient Boosting",
      "metrics": {
        "top1": 0.848,
        "f1": 0.904,
        "log_loss": 0.104
      }
    },
    ...
  ]
}
```

#### POST /select_model

```bash
curl -X POST http://localhost:8000/select_model \
  -H "Content-Type: application/json" \
  -d '{"model": "mlp"}'
```

Response:
```json
{
  "status": "ok",
  "selected_model": "mlp",
  "message": "Model 'mlp' selected for predictions"
}
```

#### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_id": "00000000", "model": "lightgbm"}'
```

Response:
```json
{
  "trip_id": "00000000",
  "model_name": "LightGBM",
  "n_gps": 2600,
  "inference_seconds": 1.234,
  "gps": [[52.96, -1.86], ...],
  "predicted_route": [[[lat, lon], [lat, lon]], ...],
  "metrics": {
    "top1_classifier": 0.82,
    "top1_hmm": 0.91,
    "macro_f1": 0.88,
    "route_continuity": 0.95
  }
}
```

---

## 12. Model Selection Guide

### Available Models

| Model | Family | Top-1 | F1 | Log Loss | Inference Speed | Recommended |
|-------|--------|-------|-----|----------|-----------------|-------------|
| **LightGBM** | Boosting | 84.8% | 90.4% | 0.104 | ~1.2s | ⭐ **Best overall** |
| **MLP** | Neural Net | 84.8% | 91.1% | **0.098** | ~0.25s | ⭐ Best log loss |
| Random Forest | Bagging | 69.0% | 81.5% | 0.222 | ~0.3s | ✓ Good baseline |
| Decision Tree | Tree | 67.6% | 78.8% | 0.397 | ~0.03s | Fast but overfits |
| Logistic Reg | Linear | 67.2% | 73.3% | 0.348 | ~0.04s | Simple baseline |

### Recommendations

**For best accuracy:**
- Use **LightGBM** or **MLP** (both have similar Top-1, MLP has slightly better log loss)

**For fastest inference:**
- Use **Decision Tree** or **Logistic Regression** (but lower accuracy)

**For Viterbi decoder:**
- Use **MLP** (lowest log loss = best-calibrated probabilities)

**For production deployment:**
- Use **LightGBM** (fast inference, no scaler needed, excellent accuracy)

---

## Quick Reference Card

### Server Commands (Laptop 1)

```bash
# Start server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Check health
curl http://localhost:8000/health

# Get IP address
hostname -I
```

### Client Commands (Laptop 2)

```bash
# Start HTTP server for UI
cd ui && python -m http.server 8080

# Test API connection
curl http://192.168.1.100:8000/health

# Open UI
# Browser: http://localhost:8080/app_model_selection.html
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export EVAL_DATA_DIR=/path/to/data
```

---

## Checklist Before Demo

- [ ] Both laptops connected to same WiFi
- [ ] Laptop 1 IP address noted
- [ ] Firewall allows port 8000
- [ ] Virtual environment activated on both laptops
- [ ] All dependencies installed
- [ ] Model files present and extracted
- [ ] Trip data in `data/raw/student_test/`
- [ ] API server running on Laptop 1
- [ ] API health check returns OK
- [ ] UI opens on Laptop 2
- [ ] UI connects to API successfully
- [ ] Model selection works
- [ ] Prediction runs and shows route on map

---

**End of Setup Guide**

For questions or issues, contact Member D or refer to the project documentation in `docs/`.
