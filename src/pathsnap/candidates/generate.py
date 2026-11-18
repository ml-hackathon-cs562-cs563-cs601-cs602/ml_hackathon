# For each GPS point, return the K nearest arcs within radius R (both from configs/data.yaml).
# Output: candidate rows keyed by (trip_id, point_idx, candidate_arc_id).
# Also compute candidate miss rate (true arc absent from candidates) to choose K and R.
# Owner: B. Critical path: others are blocked until v0 is ready.
