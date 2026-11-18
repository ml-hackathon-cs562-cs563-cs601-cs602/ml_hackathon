# Data contract

## Raw files per trip
| File | Columns (no header) | Notes |
|---|---|---|
| `.track` | lon, lat, timestamp_s | GPS observations. Nominally 1 Hz. |
| `.nodes` | lon, lat | Row index = node index (0-based). |
| `.arcs` | from_node_id, to_node_id | Row index = `arc_id` (0-based). |
| `.route` | arc_id (one per row, ordered) | Reference route. Not verified physical truth. |

Do not assume a `.track` row matches a `.nodes` or `.arcs` row. Each file has its own indexing.

## Partitions (configs/partitions.yaml)
- Train: 80 trips. Fitting, CV, Bayesian optimisation.
- Student test: 10 trips. Development comparisons, decoder tuning.
- Evaluator test: 10 trips. Final evaluation only. IDs appear in configs; labels never do.
- No trip's observations may appear in more than one partition.

## Candidate table (handoff from B to C and D)
Written to `data/features/features.parquet`. One row per (trip, GPS point, candidate arc).

| Column | Meaning |
|---|---|
| trip_id | Trip identifier |
| partition | train / student |
| point_idx | GPS observation index within the trip |
| candidate_arc_id | Candidate arc within that trip's network |
| label | 1 if candidate_arc_id is in the trip's `.route`, else 0 (train and student only) |
| dist_m | Perpendicular distance from point to arc, metres |
| ...features | Defined in `src/pathsnap/features/build.py`, versioned by `features_version` |

Schema changes require a team decision and a new `features_version`. Announce them in the group.

## Coordinates
- Input lon/lat are angular. Use the metric CRS from `configs/data.yaml` for distances.
- Check coordinate order (lon first) before any projection.
