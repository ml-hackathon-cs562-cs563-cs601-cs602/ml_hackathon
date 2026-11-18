
from pathlib import Path
import pandas as pd
from pyproj import Transformer

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


def get_utm_epsg(longitude, latitude):
    """
    Determine the UTM EPSG code for a geographic coordinate.

    Supports standard UTM coverage:
    longitude: -180 to 180
    latitude: -80 to 84
    """

    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    if not (-80 <= latitude <= 84):
        raise ValueError(
            "UTM is not suitable for latitudes outside -80 to 84 degrees."
        )

    zone = int((longitude + 180) // 6) + 1
    zone = min(max(zone, 1), 60)

    # Northern hemisphere: EPSG 326xx
    # Southern hemisphere: EPSG 327xx
    if latitude >= 0:
        return 32600 + zone

    return 32700 + zone



def project_trip_coordinates(trip_id, partition="train"):
    """
    Project GPS observations and road nodes into metres.

    Uses:
    - UTM when all trip coordinates fall within one UTM zone.
    - Local AEQD when coordinates cross UTM zones.

    Both GPS observations and road nodes use the same CRS.
    """

    track = load_track(trip_id, partition)
    nodes = load_nodes(trip_id, partition)

    # Combine coordinates to select a CRS covering the entire trip.
    coordinates = pd.concat([
        track[["longitude", "latitude"]],
        nodes[["longitude", "latitude"]]
    ], ignore_index=True)

    longitude = float(coordinates["longitude"].median())
    latitude = float(coordinates["latitude"].median())

    zones = (
        ((coordinates["longitude"] + 180) // 6 + 1)
        .clip(1, 60)
        .astype(int)
        .unique()
    )

    outside_utm = (
        coordinates["latitude"].min() < -80
        or coordinates["latitude"].max() > 84
    )

    if len(zones) == 1 and not outside_utm:
        epsg = get_utm_epsg(longitude, latitude)
        target_crs = f"EPSG:{epsg}"
        projection_type = "UTM"
    else:
        # Local projection centred on the trip.
        epsg = None
        target_crs = (
            f"+proj=aeqd +lat_0={latitude} "
            f"+lon_0={longitude} +datum=WGS84 "
            "+units=m +no_defs"
        )
        projection_type = "AEQD"

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True
    )

    track = track.copy()
    track["x"], track["y"] = transformer.transform(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy()
    )

    nodes = nodes.copy()
    nodes["x"], nodes["y"] = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )

    return {
        "trip_id": trip_id,
        "partition": partition,
        "epsg": epsg,
        "projection_type": projection_type,
        "target_crs": target_crs,
        "track": track,
        "nodes": nodes
    }

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
