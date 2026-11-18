
import pandas as pd

from data_processing.data_loader import (
    list_trips,
    load_track,
    load_nodes,
    project_trip_coordinates
)


def main():
    results = []

    for partition in ["train", "student_test"]:
        for trip_id in list_trips(partition):
            track = load_track(trip_id, partition)
            nodes = load_nodes(trip_id, partition)

            projected = project_trip_coordinates(trip_id, partition)

            coordinates = pd.concat([
                track[["longitude", "latitude"]],
                nodes[["longitude", "latitude"]]
            ], ignore_index=True)

            zones = (
                ((coordinates["longitude"] + 180) // 6 + 1)
                .clip(1, 60)
                .astype(int)
                .unique()
            )

            results.append({
                "trip_id": trip_id,
                "partition": partition,
                "projection_type": projected["projection_type"],
                "epsg": projected["epsg"],
                "utm_zone_count": len(zones),
                "gps_points": len(track),
                "road_nodes": len(nodes)
            })

    report = pd.DataFrame(results)

    report.to_csv(
        "results/projection_audit.csv",
        index=False
    )

    print("\n========== FINAL PROJECTION AUDIT ==========")
    print(report["projection_type"].value_counts())
    print("\nTotal trips:", len(report))
    print("\nProjection exceptions:")
    print(
        report[report["projection_type"] == "AEQD"].to_string(index=False)
    )
    print("\nSaved: results/projection_audit.csv")


if __name__ == "__main__":
    main()
