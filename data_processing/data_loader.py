
from pathlib import Path
import pandas as pd


# Dataset location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "Student folder"

VALID_PARTITIONS = {"train", "student_test"}


def list_trips(partition="train"):
    """Return available trip IDs in a partition."""

    if partition not in VALID_PARTITIONS:
        raise ValueError(f"Invalid partition: {partition}")

    partition_dir = DATA_ROOT / partition

    if not partition_dir.exists():
        raise FileNotFoundError(partition_dir)

    return sorted(
        folder.name
        for folder in partition_dir.iterdir()
        if folder.is_dir()
    )


def load_track(trip_id, partition="train"):
    """Load GPS observations."""

    trip_dir = _get_trip_directory(trip_id, partition)

    return pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )


def load_nodes(trip_id, partition="train"):
    """Load road-network nodes. Row index is node ID."""

    trip_dir = _get_trip_directory(trip_id, partition)

    return pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )


def load_arcs(trip_id, partition="train"):
    """Load directed road arcs. Row index is arc ID."""

    trip_dir = _get_trip_directory(trip_id, partition)

    return pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )


def load_route(trip_id, partition="train"):
    """
    Load chronological ground-truth arc IDs.

    Keep separate from GPS observations:
    route entries are NOT assumed to correspond
    one-to-one with GPS observations.
    """

    trip_dir = _get_trip_directory(trip_id, partition)

    return pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )


def load_trip(trip_id, partition="train", include_route=False):
    """
    Load a complete trip.

    Route labels are excluded by default to reduce
    accidental label leakage into prediction pipelines.
    """

    data = {
        "trip_id": trip_id,
        "partition": partition,
        "track": load_track(trip_id, partition),
        "nodes": load_nodes(trip_id, partition),
        "arcs": load_arcs(trip_id, partition)
    }

    if include_route:
        data["route"] = load_route(trip_id, partition)

    return data


def _get_trip_directory(trip_id, partition):
    """Validate partition and locate trip directory."""

    if partition not in VALID_PARTITIONS:
        raise ValueError(f"Invalid partition: {partition}")

    trip_dir = DATA_ROOT / partition / str(trip_id)

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip {trip_id} not found in {partition}"
        )

    return trip_dir
