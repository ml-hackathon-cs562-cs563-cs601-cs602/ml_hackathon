
from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path("data/raw/Student folder")

def load_trip(trip_dir):
    trip_id = trip_dir.name

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


def inspect_trip(trip_dir, partition):
    trip_id = trip_dir.name

    track, nodes, arcs, route = load_trip(trip_dir)

    time_gaps = track["timestamp"].diff()

    return {
        "trip_id": trip_id,
        "partition": partition,
        "gps_points": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_entries": len(route),
        "missing_gps": int(
            track[["longitude", "latitude", "timestamp"]]
            .isna().any(axis=1).sum()
        ),
        "invalid_coordinates": int(
            (
                ~track["longitude"].between(-180, 180)
                | ~track["latitude"].between(-90, 90)
            ).sum()
        ),
        "duplicate_timestamps": int(
            track["timestamp"].duplicated().sum()
        ),
        "negative_time_gaps": int((time_gaps < 0).sum()),
        "max_time_gap": (
            float(time_gaps.max())
            if len(time_gaps.dropna()) else np.nan
        ),
        "longitude_min": track["longitude"].min(),
        "longitude_max": track["longitude"].max(),
        "latitude_min": track["latitude"].min(),
        "latitude_max": track["latitude"].max(),
    }


def main():
    records = []

    for partition in ["train", "student_test"]:
        partition_dir = DATA_ROOT / partition

        for trip_dir in sorted(partition_dir.iterdir()):
            if not trip_dir.is_dir():
                continue

            try:
                records.append(
                    inspect_trip(trip_dir, partition)
                )
                print(f"Inspected trip {trip_dir.name}")

            except Exception as error:
                print(f"ERROR in {trip_dir.name}: {error}")

    df = pd.DataFrame(records)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "dataset_inventory.csv"
    df.to_csv(output_file, index=False)

    print("\n========== DATASET SUMMARY ==========")
    print(df.groupby("partition").agg({
        "trip_id": "count",
        "gps_points": ["mean", "min", "max"],
        "nodes": ["mean", "min", "max"],
        "arcs": ["mean", "min", "max"],
        "route_entries": ["mean", "min", "max"]
    }))

    print(f"\nSaved inventory to: {output_file}")


if __name__ == "__main__":
    main()
