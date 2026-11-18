
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/Student folder")
records = []

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

        gaps = track["timestamp"].diff()

        records.append({
            "trip_id": trip_id,
            "partition": partition,
            "gps_points": len(track),
            "gaps_over_10s": int((gaps > 10).sum()),
            "gaps_over_60s": int((gaps > 60).sum()),
            "gaps_over_5min": int((gaps > 300).sum()),
            "gaps_over_1hr": int((gaps > 3600).sum()),
            "max_gap_seconds": gaps.max()
        })

df = pd.DataFrame(records)

Path("results").mkdir(exist_ok=True)
df.to_csv("results/timestamp_gap_report.csv", index=False)

print("\n========== TIMESTAMP GAP SUMMARY ==========")
print(df.groupby("partition").agg({
    "trip_id": "count",
    "gps_points": "sum",
    "gaps_over_10s": "sum",
    "gaps_over_60s": "sum",
    "gaps_over_5min": "sum",
    "gaps_over_1hr": "sum"
}))

print("\n========== TRIPS WITH GAPS OVER 5 MINUTES ==========")
print(
    df[df["gaps_over_5min"] > 0][
        ["trip_id", "partition", "gaps_over_5min", "max_gap_seconds"]
    ].to_string(index=False)
)

print("\nSaved: results/timestamp_gap_report.csv")
