
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("data/raw/Student folder")

def haversine(lon1, lat1, lon2, lat2):
    earth_radius = 6371000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


for partition in ["train", "student_test"]:
    for trip_dir in sorted((ROOT / partition).iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

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
            names=["from_node", "to_node"]
        )

        route = pd.read_csv(
            trip_dir / f"{trip_id}.route",
            sep=r"\s+",
            header=None,
            names=["arc_id"]
        )["arc_id"].astype(int).tolist()

        for i in range(len(route) - 1):
            arc_a = arcs.iloc[route[i]]
            arc_b = arcs.iloc[route[i + 1]]

            end_node = int(arc_a["to_node"])
            start_node = int(arc_b["from_node"])

            if end_node == start_node:
                continue

            end_coord = nodes.iloc[end_node]
            start_coord = nodes.iloc[start_node]

            distance = haversine(
                end_coord["longitude"],
                end_coord["latitude"],
                start_coord["longitude"],
                start_coord["latitude"]
            )

            print(
                f"{trip_id} ({partition}) | "
                f"route positions {i}->{i+1} | "
                f"arcs {route[i]}->{route[i+1]} | "
                f"endpoint gap: {distance:.2f} metres"
            )

print("\nDistance analysis complete.")
