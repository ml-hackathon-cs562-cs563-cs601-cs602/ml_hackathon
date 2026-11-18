
from data_processing.data_loader import (
    list_trips,
    load_trip
)

def main():
    trips = list_trips("train")

    print("Training trips:", len(trips))
    print("First five trips:", trips[:5])

    trip = load_trip("00000001", "train")

    print("\nTrip ID:", trip["trip_id"])
    print("GPS observations:", len(trip["track"]))
    print("Road nodes:", len(trip["nodes"]))
    print("Road arcs:", len(trip["arcs"]))

    print("\nRoute excluded by default:",
          "route" not in trip)

    labelled_trip = load_trip(
        "00000001",
        "train",
        include_route=True
    )

    print("Route entries:", len(labelled_trip["route"]))

    print("\nData loader test completed.")


if __name__ == "__main__":
    main()
