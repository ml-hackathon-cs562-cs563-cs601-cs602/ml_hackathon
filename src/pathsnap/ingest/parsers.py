# Parse one trip's raw files: .track (lon, lat, t), .nodes (lon, lat),
# .arcs (from_node_id, to_node_id; arc_id = row index), .route (ordered arc_ids).
# Must not assume a GPS row matches a node row. Validate column counts and types.
# Owner: A.
