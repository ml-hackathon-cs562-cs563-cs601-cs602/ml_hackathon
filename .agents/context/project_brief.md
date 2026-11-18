# Project brief

## Challenge
GPS tracking is noisy in urban areas (buildings, multipath, dropouts, low sampling).
A vehicle can appear to jump onto a parallel street or across buildings. PathSnap is a
distributed ML system that matches noisy GPS observations to a road network and
reconstructs realistic trajectories.

## Deployment (two laptops)
- Laptop 1: trained model, HMM/Viterbi, road network, FastAPI server, evaluator
  ground truth. Runs the API.
- Laptop 2: evaluator UI. Lists the 10 evaluator trips, sends the selected trip ID to
  the API, displays the predicted route and results.
- The UI must work for all ten evaluator trips, not one demo trip.

## Data (as reported in the planning document, NOT yet verified)
The team's planning document reports the following. Verify every item in
`pipelines/01_inspect.py` before using any of it in the report.
- Each trip has its own road network. Node counts range from about 300 to 155,000,
  arc counts from about 650 to 307,000.
- About 80 training trips (about 2,600 GPS points each on average, up to about 19,000),
  10 student-test trips (about 1,900 points each on average).
- Coordinates appear to lie around lon -1.86, lat 52.96. That places them in UTM zone 30N.
- Expected training candidate rows: roughly 1 to 3 million.

## Data files (from the dataset guide)
- `.track`: lon, lat, timestamp in seconds. No header. Nominally 1 Hz (about 97% of steps are 1 s apart).
- `.nodes`: lon, lat. No header. Row index is the node index.
- `.arcs`: from_node_id, to_node_id. No header. Row index is `arc_id`.
- `.route`: ordered sequence of ground-truth `arc_id` values. Supplied as reference
  information, not verified physical truth.
