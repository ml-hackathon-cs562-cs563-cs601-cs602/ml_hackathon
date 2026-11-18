# PathSnap Evaluator UI

Static Leaflet map UI designed for Laptop 2.
Owner: Member A & C

## Running the UI
1. Set the API URL in `index.html` or in the browser input box (default: `http://localhost:8000`).
2. Run:
   ```bash
   python ui/app.py
   # or
   python ui/serve_ui.py
   ```
3. Open `http://localhost:8080` in a browser.
4. Click **Load trips**, select an evaluator trip, and inspect the predicted route, GPS observations, and metrics.
