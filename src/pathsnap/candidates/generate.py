
"""Spatially indexed road-arc candidate generation."""

from typing import Any

import pandas as pd
from shapely import LineString, Point
from shapely.strtree import STRtree

from data_processing.data_loader import (
    load_arcs,
    project_trip_coordinates,
)


def generate_candidates(
    trip_id: str,
    partition: str,
    k: int,
    radius_m: float,
) -> pd.DataFrame:
    """
    Generate the K nearest road-arc candidates within a radius
    for every GPS observation in a trip.

    Coordinates are projected to a metric CRS before distance
    calculations.

    Args:
        trip_id: Unique trip identifier.
        partition: Dataset partition (train or student_test).
        k: Maximum number of candidates per GPS observation.
        radius_m: Maximum point-to-road distance in metres.

    Returns:
        DataFrame with one row per GPS-point/candidate-arc pair.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    projected = project_trip_coordinates(trip_id, partition)
    track = projected["track"]
    nodes = projected["nodes"]
    arcs = load_arcs(trip_id, partition)

    if track.empty or arcs.empty:
        return pd.DataFrame(
            columns=[
                "trip_id",
                "partition",
                "point_idx",
                "candidate_arc_id",
                "candidate_rank",
                "dist_m",
            ]
        )

    # Construct one geometry for each road arc.
    arc_ids: list[int] = []
    geometries = []

    for arc_id, arc in arcs.iterrows():
        start_id = int(arc["from_node_id"])
        end_id = int(arc["to_node_id"])

        if start_id not in nodes.index or end_id not in nodes.index:
            raise ValueError(
                f"Arc {arc_id} references a missing road node"
            )

        start = nodes.loc[start_id]
        end = nodes.loc[end_id]

        geometry = LineString(
            [
                (float(start["x"]), float(start["y"])),
                (float(end["x"]), float(end["y"])),
            ]
        )

        arc_ids.append(int(arc_id))
        geometries.append(geometry)

    # Build the spatial index once per trip.
    tree = STRtree(geometries)

    results: list[dict[str, Any]] = []

    # Find nearby arcs for each GPS observation.
    for point_idx, observation in track.iterrows():
        gps_point = Point(
            float(observation["x"]),
            float(observation["y"]),
        )

        nearby_indices = tree.query(
            gps_point,
            predicate="dwithin",
            distance=radius_m,
        )

        # Calculate exact distances and sort nearest first.
        nearby = []

        for geometry_idx in nearby_indices:
            geometry_idx = int(geometry_idx)
            distance = float(
                gps_point.distance(geometries[geometry_idx])
            )

            nearby.append(
                (
                    distance,
                    arc_ids[geometry_idx],
                )
            )

        nearby.sort(key=lambda item: (item[0], item[1]))

        # Retain only the K nearest candidates.
        for rank, (distance, arc_id) in enumerate(nearby[:k], start=1):
            results.append(
                {
                    "trip_id": trip_id,
                    "partition": partition,
                    "point_idx": int(point_idx),
                    "candidate_arc_id": arc_id,
                    "candidate_rank": rank,
                    "dist_m": distance,
                }
            )

    return pd.DataFrame(
        results,
        columns=[
            "trip_id",
            "partition",
            "point_idx",
            "candidate_arc_id",
            "candidate_rank",
            "dist_m",
        ],
    )
