
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/Student folder")

for partition in ["train", "student_test"]:
    for trip_dir in sorted((ROOT / partition).iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        gaps = track["timestamp"].diff().dropna()

        large_gaps = gaps[gaps > 10]

        if not large_gaps.empty:
            print(f"\nTrip: {trip_id} ({partition})")
            print(f"Total GPS points: {len(track)}")
            print(f"Gaps > 10 seconds: {len(large_gaps)}")
            print(f"Maximum gap: {gaps.max()} seconds")
            print(f"Median gap: {gaps.median()} seconds")
