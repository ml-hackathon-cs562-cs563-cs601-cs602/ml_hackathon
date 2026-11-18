
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/Student folder")

for partition in ["train", "student_test"]:
    for trip_dir in sorted((ROOT / partition).iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

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

            if arc_a["to_node"] != arc_b["from_node"]:
                print(f"\nTrip: {trip_id} ({partition})")
                print(f"Route positions: {i} and {i + 1}")
                print(f"Arc IDs: {route[i]} -> {route[i + 1]}")
                print(
                    f"First arc: {arc_a['from_node']} -> "
                    f"{arc_a['to_node']}"
                )
                print(
                    f"Second arc: {arc_b['from_node']} -> "
                    f"{arc_b['to_node']}"
                )

print("\nInspection complete.")
