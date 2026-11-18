
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/Student folder")
records = []

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
        )

        route_ids = route["arc_id"].astype(int)

        # Check whether route arc IDs exist in the arcs table
        valid_ids = route_ids.between(0, len(arcs) - 1)
        invalid_id_count = int((~valid_ids).sum())

        # Check connectivity between consecutive valid route arcs
        valid_route = route_ids[valid_ids].to_numpy()
        valid_sequence = len(valid_route) == len(route_ids)

        connected_pairs = 0
        disconnected_pairs = 0

        if valid_sequence and len(valid_route) > 1:
            selected_arcs = arcs.iloc[valid_route].reset_index(drop=True)

            for i in range(len(selected_arcs) - 1):
                current_to = selected_arcs.loc[i, "to_node"]
                next_from = selected_arcs.loc[i + 1, "from_node"]

                if current_to == next_from:
                    connected_pairs += 1
                else:
                    disconnected_pairs += 1

        total_pairs = connected_pairs + disconnected_pairs

        records.append({
            "trip_id": trip_id,
            "partition": partition,
            "total_arcs": len(arcs),
            "route_entries": len(route_ids),
            "invalid_arc_ids": invalid_id_count,
            "connected_pairs": connected_pairs,
            "disconnected_pairs": disconnected_pairs,
            "connectivity_percent": (
                connected_pairs / total_pairs * 100
                if total_pairs else None
            )
        })

df = pd.DataFrame(records)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/route_validation.csv", index=False)

print("\n========== ROUTE VALIDATION SUMMARY ==========")
print(df.groupby("partition").agg({
    "trip_id": "count",
    "route_entries": "sum",
    "invalid_arc_ids": "sum",
    "connected_pairs": "sum",
    "disconnected_pairs": "sum"
}))

print("\n========== TRIPS WITH INVALID IDS OR DISCONNECTED ROUTES ==========")
print(
    df[
        (df["invalid_arc_ids"] > 0) |
        (df["disconnected_pairs"] > 0)
    ].to_string(index=False)
)

print("\nSaved: results/route_validation.csv")
