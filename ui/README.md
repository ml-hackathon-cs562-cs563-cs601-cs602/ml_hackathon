# PathSnap UI

Web-based interface for PathSnap GPS map-matching system.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Basic UI (original evaluator interface) |
| `app_model_selection.html` | Enhanced UI with model selection |

## Quick Start

### Option 1: Open directly in browser

1. Navigate to `ui/` folder
2. Double-click `app_model_selection.html`
3. Enter API server address (e.g., `http://192.168.1.100:8000`)
4. Click Connect

### Option 2: Serve via HTTP server (recommended)

```bash
# From the ui/ directory
python -m http.server 8080

# Open browser to:
# http://localhost:8080/app_model_selection.html
```

## Features

### app_model_selection.html

- **Model Selection**: Choose from 5 trained models
- **Trip Selection**: Select from available trips
- **Real-time Prediction**: View GPS trace and predicted route
- **Metrics Display**: Top-1 accuracy, Macro F1, Route continuity
- **Dark/Light Mode**: Automatic based on system preference
- **Responsive Design**: Works on different screen sizes

### Available Models

| Model | Top-1 | F1 | Log Loss |
|-------|-------|-----|----------|
| LightGBM | 84.8% | 90.4% | 0.104 |
| MLP | 84.8% | 91.1% | 0.098 |
| Random Forest | 69.0% | 81.5% | 0.222 |
| Decision Tree | 67.6% | 78.8% | 0.397 |
| Logistic Regression | 67.2% | 73.3% | 0.348 |

## Network Setup

### Laptop 1 (Server)

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Laptop 2 (Client)

1. Open `app_model_selection.html` in browser
2. Enter Laptop 1's IP address: `http://<LAPTOP_1_IP>:8000`
3. Click Connect

## Troubleshooting

### Cannot connect to API

1. Verify API is running: `curl http://localhost:8000/health`
2. Check firewall allows port 8000
3. Use IP address (not `localhost`) when connecting from another machine

### Models not showing

1. Verify model files exist in `results/model4/` and `results/model5/`
2. Check API health endpoint returns `"available_models": 5`

### Prediction fails

1. Check trip data exists in `data/raw/student_test/`
2. Verify selected model is valid
3. Check API logs for errors

## Screenshots

*(Add screenshots here after running the application)*

## Notes

- The UI uses Leaflet.js for map rendering
- GPS observations shown in blue
- Predicted route shown in red
- Supports both light and dark themes automatically
