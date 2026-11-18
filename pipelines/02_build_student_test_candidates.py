
"""Generate road-arc candidates for student-test GPS observations."""

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
    """Generate and save candidates for all student-test trips."""

    config_path = PROJECT_ROOT / "configs" / "data.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    k = int(config["candidate_k"])
    radius_m = float(config["candidate_radius_m"])

    if k < 1 or radius_m <= 0:
        raise ValueError("Invalid candidate K or radius configuration")

    partition = "student_test"
    trips = list_trips(partition)

    if not trips:
        raise ValueError("No student-test trips found")

    candidate_frames = []

    print(f"Processing {len(trips)} student-test trips")
    print(f"K = {k}, radius = {radius_m} metres")

    for index, trip_id in enumerate(trips, start=1):
        print(f"[{index}/{len(trips)}] Processing trip {trip_id}")

        candidates = generate_candidates(
            trip_id=trip_id,
            partition=partition,
            k=k,
            radius_m=radius_m,
        )

        candidate_frames.append(candidates)

        print(f"    Candidate rows: {len(candidates)}")

    combined = pd.concat(candidate_frames, ignore_index=True)

    if combined.empty:
        raise ValueError("No student-test candidates were generated")

    if combined["partition"].ne(partition).any():
        raise ValueError("Unexpected partition in candidate output")

    key_columns = [
        "trip_id",
        "point_idx",
        "candidate_arc_id",
    ]

    if combined.duplicated(key_columns).any():
        raise ValueError("Duplicate student-test candidates detected")

    output_dir = PROJECT_ROOT / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "candidates_student_test_v0.parquet"
    combined.to_parquet(output_path, index=False)

    unique_points = combined[
        ["trip_id", "point_idx"]
    ].drop_duplicates().shape[0]

    print("\nStudent-test candidate generation completed")
    print(f"Trips processed: {len(trips)}")
    print(f"Total candidate rows: {len(combined)}")
    print(f"Unique GPS points: {unique_points}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
