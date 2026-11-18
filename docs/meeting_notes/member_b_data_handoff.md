# PathSnap – Data Loader and EDA Handoff

## Dataset

- Training partition: 80 trips
- Student-test partition: 10 trips
- Total: 90 trips

## File formats

| File | Columns | Meaning |
|---|---|---|
| `.track` | longitude, latitude, timestamp | GPS observations |
| `.nodes` | longitude, latitude | Road-network nodes; row index is node ID |
| `.arcs` | from_node_id, to_node_id | Directed arcs; row index is arc ID |
| `.route` | arc_id | Chronological ground-truth arc sequence |

## Data quality findings

- No missing GPS observations.
- No invalid GPS coordinates.
- No duplicate timestamps.
- No negative timestamp gaps.
- Median timestamp interval is 1 second.
- Training partition: 590 gaps above 10 seconds.
- Training partition: 16 gaps above 5 minutes.
- Training partition: 2 gaps above 1 hour.
- Student-test partition: 27 gaps above 10 seconds.

## Route integrity findings

- Zero invalid route arc IDs.
- Training: 31,243 connected pairs and 13 disconnected pairs.
- Student-test: 3,466 connected pairs and 1 disconnected pair.
- Fourteen disconnected transitions were investigated.
- Endpoint gaps ranged from 10.62 m to 595.81 m.

## Data loader interface

File: `data_processing/data_loader.py`

Available functions:

- `list_trips(partition)`
- `load_track(trip_id, partition)`
- `load_nodes(trip_id, partition)`
- `load_arcs(trip_id, partition)`
- `load_route(trip_id, partition)`
- `load_trip(trip_id, partition, include_route=False)`

Example:

```python
from data_processing.data_loader import load_trip

trip = load_trip("00000001", "train")

track = trip["track"]
nodes = trip["nodes"]
arcs = trip["arcs"]
```

Ground-truth route is excluded by default.

## Important modelling caution

GPS observations and route entries do not have a one-to-one relationship established yet. Do not label every GPS observation using simple membership of an arc in the route sequence.

Investigate and validate GPS-to-route alignment before supervised feature labelling.

Long timestamp gaps and disconnected route transitions should also be considered in candidate generation and HMM/Viterbi decoding.

## Generated EDA outputs

- `results/dataset_inventory.csv`
- `results/timestamp_gap_report.csv`
- `results/route_validation.csv`
- `viz/maps/00000001_map.html`