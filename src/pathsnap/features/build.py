
"""Build numerical GPS-to-road candidate features."""

import numpy as np
import pandas as pd

from data_processing.data_loader import (
    load_arcs,
    project_trip_coordinates,
)
from pathsnap.geo.distances import (
    along_segment_position,
    heading_difference,
    segment_heading,
)


FEATURE_COLUMNS = [
    "dist_m",
    "heading_diff",
    "along_arc_position",
    "speed_mps",
    "time_gap_s",
    "arc_length_m",
    "candidate_rank",
    "prev_point_dist_m",
    "next_point_dist_m",
    "node_degree",
]


def build_features(
    candidates: pd.DataFrame,
    trip_id: str,
    partition: str = "train",
    features_version: str = "v0",
) -> pd.DataFrame:
    """Build candidate-level features without reading route labels."""
    required = {
        "trip_id", "partition", "point_idx",
        "candidate_arc_id", "candidate_rank", "dist_m",
    }
    if not required.issubset(candidates.columns):
        raise ValueError(
            f"Missing candidate columns: {required - set(candidates.columns)}"
        )

    if candidates.empty:
        result = candidates.copy()
        for name in FEATURE_COLUMNS:
            if name not in result.columns:
                result[name] = pd.Series(dtype=float)
        result["features_version"] = features_version
        return result

    if not (
        (candidates["trip_id"] == trip_id).all()
        and (candidates["partition"] == partition).all()
    ):
        raise ValueError("Candidate rows must belong to the requested trip")

    projected = project_trip_coordinates(trip_id, partition)
    track = projected["track"]
    nodes = projected["nodes"]
    arcs = load_arcs(trip_id, partition)

    if not candidates["point_idx"].isin(track.index).all():
        raise ValueError("Candidate references an unknown GPS point")
    if not candidates["candidate_arc_id"].isin(arcs.index).all():
        raise ValueError("Candidate references an unknown road arc")

    # GPS movement and timing features.
    xy = track[["x", "y"]].to_numpy(dtype=float)
    timestamps = track["timestamp"].to_numpy(dtype=float)
    n = len(track)

    prev_dist = np.full(n, np.nan)
    next_dist = np.full(n, np.nan)
    time_gap = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    gps_heading = np.full(n, np.nan)

    if n > 1:
        delta_xy = np.diff(xy, axis=0)
        distances = np.linalg.norm(delta_xy, axis=1)
        dt = np.diff(timestamps)

        prev_dist[1:] = distances
        next_dist[:-1] = distances
        time_gap[1:] = dt

        valid_dt = dt > 0
        speed[1:] = np.divide(
            distances,
            dt,
            out=np.full(len(dt), np.nan),
            where=valid_dt,
        )

        headings = (
            np.degrees(
                np.arctan2(delta_xy[:, 0], delta_xy[:, 1])
            ) % 360.0
        )
        moving = distances > 0
        gps_heading[1:] = np.where(moving, headings, np.nan)
        gps_heading[0] = headings[0] if moving[0] else np.nan

    # Incident arc count for each road node.
    degrees = pd.concat(
        [arcs["from_node_id"], arcs["to_node_id"]]
    ).value_counts()

    result = candidates.copy()
    point_ids = result["point_idx"].to_numpy(dtype=int)
    arc_ids = result["candidate_arc_id"].to_numpy(dtype=int)

    result["prev_point_dist_m"] = prev_dist[point_ids]
    result["next_point_dist_m"] = next_dist[point_ids]
    result["time_gap_s"] = time_gap[point_ids]
    result["speed_mps"] = speed[point_ids]

    selected_arcs = arcs.loc[arc_ids]
    start_ids = selected_arcs["from_node_id"].to_numpy(dtype=int)
    end_ids = selected_arcs["to_node_id"].to_numpy(dtype=int)

    start_xy = nodes.loc[start_ids, ["x", "y"]].to_numpy(dtype=float)
    end_xy = nodes.loc[end_ids, ["x", "y"]].to_numpy(dtype=float)
    point_xy = xy[point_ids]

    arc_vectors = end_xy - start_xy
    arc_lengths = np.linalg.norm(arc_vectors, axis=1)
    length_squared = np.sum(arc_vectors ** 2, axis=1)

    relative = point_xy - start_xy
    along = np.divide(
        np.sum(relative * arc_vectors, axis=1),
        length_squared,
        out=np.zeros(len(result), dtype=float),
        where=length_squared > 0,
    )

    result["along_arc_position"] = np.clip(along, 0.0, 1.0)
    result["arc_length_m"] = arc_lengths

    road_headings = (
        np.degrees(
            np.arctan2(arc_vectors[:, 0], arc_vectors[:, 1])
        ) % 360.0
    )

    gps_headings = gps_heading[point_ids]
    differences = np.abs(gps_headings - road_headings) % 360.0
    result["heading_diff"] = np.minimum(
        differences, 360.0 - differences
    )
    result.loc[arc_lengths == 0, "heading_diff"] = np.nan

    result["node_degree"] = (
        degrees.reindex(start_ids, fill_value=0).to_numpy()
        + degrees.reindex(end_ids, fill_value=0).to_numpy()
    )

    result["features_version"] = features_version
    return result
