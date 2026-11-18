
from data_processing.data_loader import project_trip_coordinates


def main():
    trip_ids = [
        "00000001",
        "00000011",
        "00000021",
        "00000048",
        "00000097"
    ]

    for trip_id in trip_ids:
        result = project_trip_coordinates(trip_id, "train")

        print("\nTrip:", trip_id)
        print("Projection:", result["projection_type"])
        print("EPSG:", result["epsg"])
        print("GPS points:", len(result["track"]))
        print("Road nodes:", len(result["nodes"]))

        print("GPS projected coordinate sample:")
        print(result["track"][["x", "y"]].head(2))


if __name__ == "__main__":
    main()
