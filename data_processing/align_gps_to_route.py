"""
PathSnap GPS-to-route monotonic alignment prototype.

Usage from project root:
    python data_processing/align_gps_to_route.py \
        --data-root "data/raw/Student folder" \
        --partition train \
        --trips 00000001 00000013 00000021 00000042 00000054 00000069 00000078 00000090 00000097 \
        --output-dir results/alignment

Requires: numpy, pandas, pyproj, shapely>=2.0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point


def read_trip(data_root: Path, partition: str, trip_id: str):
    trip_dir = data_root / partition / trip_id
    track = pd.read_csv(
        trip_dir / f"{trip_id}.track", sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"]
    )
    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes", sep=r"\s+", header=None,
        names=["longitude", "latitude"]
    )
    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs", sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"]
    )
    route = pd.read_csv(
        trip_dir / f"{trip_id}.route", sep=r"\s+", header=None,
        names=["arc_id"]
    )["arc_id"].astype(int).to_numpy()
    return track, nodes, arcs, route


def make_trip_transformer(track: pd.DataFrame, nodes: pd.DataFrame):
    """Use per-trip UTM, or local AEQD if the trip spans UTM zones."""
    coords = pd.concat(
        [track[["longitude", "latitude"]], nodes[["longitude", "latitude"]]],
        ignore_index=True
    )
    lon = float(coords["longitude"].median())
    lat = float(coords["latitude"].median())
    zones = np.clip(
        ((coords["longitude"] + 180) // 6 + 1), 1, 60
    ).astype(int).unique()
    outside_utm = (
        coords["latitude"].min() < -80 or coords["latitude"].max() > 84
    )

    if len(zones) == 1 and not outside_utm:
        zone = int(np.clip(math.floor((lon + 180) / 6) + 1, 1, 60))
        epsg = (32600 if lat >= 0 else 32700) + zone
        crs = f"EPSG:{epsg}"
        projection_type = "UTM"
    else:
        epsg = None
        crs = (
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
            "+datum=WGS84 +units=m +no_defs"
        )
        projection_type = "AEQD"

    return (
        Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        projection_type,
        epsg,
    )


def viterbi_monotonic(emission: np.ndarray, timestamps: np.ndarray,
                      base_skip_penalty: float = 0.8):
    """
    Monotonic DP over route positions.

    Allowed transitions:
      - stay on the same route arc (repeated assignment),
      - advance to the next route arc,
      - skip one or more route arcs with a time-adaptive penalty.

    A larger timestamp gap reduces the penalty for skipping route arcs.
    The recurrence is O(number_of_GPS_points * number_of_route_arcs).
    """
    n, m = emission.shape
    dp = np.empty((n, m), dtype=np.float32)
    back = np.full((n, m), -1, dtype=np.int32)
    dp[0] = emission[0]
    route_idx = np.arange(m, dtype=np.float32)

    gaps = np.r_[1.0, np.maximum(1.0, np.diff(timestamps))]

    for i in range(1, n):
        previous = dp[i - 1]
        # Gap-aware penalty: ordinary 1-second sampling keeps skips costly;
        # long observation gaps permit more route progress.
        penalty = max(
            0.05,
            base_skip_penalty / np.sqrt(min(float(gaps[i]), 3600.0))
        )

        adjusted = previous - penalty * route_idx
        prefix_min = np.minimum.accumulate(adjusted)

        prefix_argmin = np.empty(m, dtype=np.int32)
        best_k = 0
        for k in range(m):
            if adjusted[k] < adjusted[best_k]:
                best_k = k
            prefix_argmin[k] = best_k

        best_cost = previous.copy()       # stay on same route arc
        best_prev = np.arange(m, dtype=np.int32)

        # Advance exactly one route position (no skipped arc).
        if m > 1:
            improve = previous[:-1] < best_cost[1:]
            best_cost[1:] = np.minimum(best_cost[1:], previous[:-1])
            best_prev[1:] = np.where(
                improve, np.arange(m - 1, dtype=np.int32), best_prev[1:]
            )

        # Advance by >=2 positions; skipped intermediate arcs are penalised.
        if m > 2:
            destinations = np.arange(2, m)
            jump_cost = prefix_min[:m - 2] + penalty * (destinations - 1)
            improve = jump_cost < best_cost[2:]
            best_cost[2:] = np.minimum(best_cost[2:], jump_cost)
            best_prev[2:] = np.where(
                improve, prefix_argmin[:m - 2], best_prev[2:]
            )

        dp[i] = emission[i] + best_cost
        back[i] = best_prev

    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    return path, gaps


def align_one_trip(
    data_root: Path,
    partition: str,
    trip_id: str,
    distance_threshold_m: float = 35.0,
    emission_scale_m: float = 25.0,
    base_skip_penalty: float = 0.8,
    confidence_threshold: float = 0.35,
    confidence_distance_scale_m: float = 30.0,
    confidence_margin_scale_m: float = 10.0,
):
    track, nodes, arcs, route = read_trip(data_root, partition, trip_id)
    if len(track) == 0 or len(nodes) == 0 or len(route) == 0:
        raise ValueError(f"{trip_id}: empty track, nodes, or route")
    if route.min() < 0 or route.max() >= len(arcs):
        raise ValueError(f"{trip_id}: route arc ID outside .arcs range")

    transformer, projection_type, epsg = make_trip_transformer(track, nodes)
    gps_x, gps_y = transformer.transform(
        track["longitude"].to_numpy(), track["latitude"].to_numpy()
    )
    node_x, node_y = transformer.transform(
        nodes["longitude"].to_numpy(), nodes["latitude"].to_numpy()
    )

    route_arcs = arcs.iloc[route].reset_index(drop=True)
    starts = route_arcs["from_node_id"].to_numpy(dtype=int)
    ends = route_arcs["to_node_id"].to_numpy(dtype=int)
    if (
        starts.min() < 0 or ends.min() < 0
        or starts.max() >= len(nodes) or ends.max() >= len(nodes)
    ):
        raise ValueError(f"{trip_id}: arc endpoint node ID outside .nodes range")

    # Directed route geometry: each route position keeps its from->to direction.
    route_lines = np.asarray([
        LineString([(node_x[a], node_y[a]), (node_x[b], node_y[b])])
        for a, b in zip(starts, ends)
    ], dtype=object)

    gps_points = np.asarray(
        [Point(x, y) for x, y in zip(gps_x, gps_y)], dtype=object
    )

    n, m = len(track), len(route)
    distance_matrix = np.empty((n, m), dtype=np.float32)
    # Chunk rows to avoid allocating a very large temporary geometry array.
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        distance_matrix[start:stop] = shapely.distance(
            gps_points[start:stop, None], route_lines[None, :]
        )

    # Huber-like robust emission cost from point-to-directed-arc distance.
    z = distance_matrix / emission_scale_m
    emission = np.where(z <= 3.0, 0.5 * z * z, 3.0 * z - 4.5).astype(np.float32)

    timestamps = track["timestamp"].to_numpy(dtype=float)
    route_position, time_gaps = viterbi_monotonic(
        emission, timestamps, base_skip_penalty
    )
    aligned_distance = distance_matrix[np.arange(n), route_position]
    aligned_arc_ids = route[route_position]

    # Heuristic confidence, NOT a calibrated posterior probability:
    # combines absolute distance with the local distance margin to alternatives.
    confidence = np.zeros(n, dtype=float)
    local_margin = np.zeros(n, dtype=float)
    for i, j in enumerate(route_position):
        lo, hi = max(0, j - 5), min(m, j + 6)
        local = distance_matrix[i, lo:hi]
        if len(local) > 1:
            second_best = float(np.partition(local, 1)[1])
        else:
            second_best = float(aligned_distance[i] + 100.0)
        local_margin[i] = max(0.0, second_best - float(aligned_distance[i]))
        confidence[i] = (
            np.exp(-float(aligned_distance[i]) / confidence_distance_scale_m)
            * (1.0 - np.exp(-local_margin[i] / confidence_margin_scale_m))
        )

    status = np.where(
        (aligned_distance <= distance_threshold_m)
        & (confidence >= confidence_threshold),
        "aligned",
        "uncertain",
    ).astype(object)

    # Mark the observation after a long GPS gap, but do not automatically
    # discard it if its geometry and sequence evidence remain strong.
    long_gap = time_gaps > 60.0
    status[(status == "aligned") & long_gap] = "aligned_after_gap"

    # Route entries may contain disconnected consecutive arcs. Do not create
    # labels at either side of such a break without a separate bridge decision.
    disconnected_positions = np.where(ends[:-1] != starts[1:])[0]
    if len(disconnected_positions):
        break_positions = np.unique(np.r_[
            disconnected_positions, disconnected_positions + 1
        ])
        near_break = np.isin(route_position, break_positions)
        status[near_break] = "uncertain_route_discontinuity"

    labels = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": np.arange(n),
        "aligned_arc_id": aligned_arc_ids,
        "alignment_distance_m": aligned_distance,
        "confidence": confidence,
        "alignment_status": status,
        # Additional audit columns:
        "route_position": route_position,
        "local_margin_m": local_margin,
        "time_gap_s": time_gaps,
        "projection_type": projection_type,
        "projection_epsg": epsg,
    })

    route_steps = np.diff(route_position)
    skipped_arcs = int(np.maximum(route_steps - 1, 0).sum())
    repeated_assignments = int((route_steps == 0).sum())
    label_eligible = labels["alignment_status"].eq("aligned")
    geometry_accepted = labels["alignment_status"].isin(
        ["aligned", "aligned_after_gap"]
    )
    accepted_distances = labels.loc[
        label_eligible, "alignment_distance_m"
    ].to_numpy()

    metrics = {
        "trip_id": trip_id,
        "partition": partition,
        "gps_points": n,
        "route_entries": m,
        "projection_type": projection_type,
        "projection_epsg": epsg,
        "gps_alignment_coverage_pct": 100.0 * label_eligible.mean(),
        "geometry_accepted_including_postgap_pct": 100.0 * geometry_accepted.mean(),
        "uncertain_excluded_pct": 100.0 * (~label_eligible).mean(),
        "post_gap_flagged_points": int(
            labels["alignment_status"].eq("aligned_after_gap").sum()
        ),
        "route_discontinuity_excluded_points": int(
            labels["alignment_status"].eq("uncertain_route_discontinuity").sum()
        ),
        "all_path_distance_median": float(np.median(aligned_distance)),
        "all_path_distance_p90": float(np.percentile(aligned_distance, 90)),
        "all_path_distance_max": float(np.max(aligned_distance)),
        "accepted_distance_median": (
            float(np.median(accepted_distances)) if len(accepted_distances) else np.nan
        ),
        "accepted_distance_p90": (
            float(np.percentile(accepted_distances, 90)) if len(accepted_distances) else np.nan
        ),
        "accepted_distance_max": (
            float(np.max(accepted_distances)) if len(accepted_distances) else np.nan
        ),
        "skipped_route_arcs": skipped_arcs,
        "repeated_assignments": repeated_assignments,
        "route_order_consistency_pct": (
            100.0 if np.all(route_steps >= 0) else
            100.0 * np.mean(route_steps >= 0)
        ),
        "disconnected_route_transitions": int(len(disconnected_positions)),
        "gps_gaps_over_60s": int(np.sum(time_gaps > 60.0)),
        "max_gps_gap_s": float(np.max(time_gaps)),
    }
    return labels, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/Student folder")
    parser.add_argument("--partition", default="train")
    parser.add_argument("--trips", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/alignment")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_labels, all_metrics = [], []
    for trip_id in args.trips:
        labels, metrics = align_one_trip(
            data_root, args.partition, trip_id
        )
        all_labels.append(labels)
        all_metrics.append(metrics)
        print(
            f"{trip_id}: coverage={metrics['gps_alignment_coverage_pct']:.1f}% "
            f"uncertain={metrics['uncertain_excluded_pct']:.1f}% "
            f"median={metrics['all_path_distance_median']:.1f}m "
            f"p90={metrics['all_path_distance_p90']:.1f}m"
        )

    labels_df = pd.concat(all_labels, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    labels_df.to_csv(output_dir / "point_alignment_labels.csv", index=False)
    metrics_df.to_csv(output_dir / "alignment_validation_metrics.csv", index=False)
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
