
"""Stage 2: Build road-arc candidates for GPS observations."""

from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_processing.data_loader import list_trips
from pathsnap.candidates.generate import generate_candidates


def main() -> None:
    """Generate and save candidate rows for configured trips."""
    config_path = PROJECT_ROOT / "configs" / "data.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    k = int(config["candidate_k"])
    radius_m = float(config["candidate_radius_m"])
    trip_limit = config.get("candidate_trip_limit")

    if k < 1 or radius_m <= 0:
        raise ValueError("Invalid candidate K or radius configuration")

    trips = list_trips("train")

    if trip_limit is not None:
        trips = trips[:int(trip_limit)]

    if not trips:
        raise ValueError("No training trips found")

    candidate_frames = []

    print(f"Processing {len(trips)} training trips")
    print(f"K = {k}, radius = {radius_m} metres")

    for index, trip_id in enumerate(trips, start=1):
        print(f"[{index}/{len(trips)}] Processing trip {trip_id}")

        candidates = generate_candidates(
            trip_id=trip_id,
            partition="train",
            k=k,
            radius_m=radius_m,
        )

        candidate_frames.append(candidates)

        print(f"    Candidate rows: {len(candidates)}")

    combined = pd.concat(candidate_frames, ignore_index=True)

    output_dir = PROJECT_ROOT / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "candidates_v0.parquet"
    combined.to_parquet(output_path, index=False)

    print("\nCandidate generation completed")
    print(f"Trips processed: {len(trips)}")
    print(f"Total candidate rows: {len(combined)}")
    print(f"Unique GPS points: {combined[['trip_id', 'point_idx']].drop_duplicates().shape[0]}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
