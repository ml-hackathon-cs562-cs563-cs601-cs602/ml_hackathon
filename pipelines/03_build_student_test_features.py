
"""Build and preprocess student-test candidate features."""

from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.features.build import build_features
from pathsnap.features.preprocess import FeaturePreprocessor


def main() -> None:
    """Build student-test features and transform using training state."""

    config_path = PROJECT_ROOT / "configs" / "features.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    version = str(config["features_version"])

    candidate_path = (
        PROJECT_ROOT / config["student_test_candidate_input_path"]
    )
    raw_output_path = (
        PROJECT_ROOT / config["student_test_features_output_path"]
    )
    scaled_output_path = (
        PROJECT_ROOT / config["student_test_preprocessed_output_path"]
    )
    artifact_path = (
        PROJECT_ROOT / config["preprocessor_artifact_path"]
    )

    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Student-test candidates not found: {candidate_path}"
        )

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Training preprocessor not found: {artifact_path}"
        )

    candidates = pd.read_parquet(candidate_path)

    if candidates.empty:
        raise ValueError("Student-test candidate dataset is empty")

    if set(candidates["partition"].unique()) != {"student_test"}:
        raise ValueError(
            "Expected student_test candidates only"
        )

    feature_frames = []

    groups = candidates.groupby(
        ["partition", "trip_id"],
        sort=True,
    )

    print(f"Candidate rows: {len(candidates)}")
    print(f"Student-test trips: {len(groups)}")
    print(f"Feature version: {version}")

    for index, ((partition, trip_id), trip_candidates) in enumerate(
        groups,
        start=1,
    ):
        print(
            f"[{index}/{len(groups)}] Processing trip {trip_id}"
        )

        features = build_features(
            candidates=trip_candidates,
            trip_id=str(trip_id),
            partition=str(partition),
            features_version=version,
        )

        feature_frames.append(features)

        print(f"    Feature rows: {len(features)}")

    combined = pd.concat(feature_frames, ignore_index=True)

    if len(combined) != len(candidates):
        raise ValueError(
            "Feature generation changed the candidate row count"
        )

    key_columns = [
        "trip_id",
        "point_idx",
        "candidate_arc_id",
    ]

    if combined.duplicated(key_columns).any():
        raise ValueError(
            "Duplicate student-test feature rows detected"
        )

    # Save original student-test feature values.
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(raw_output_path, index=False)

    # Load the preprocessor fitted on all 80 training trips.
    processor = FeaturePreprocessor.load(artifact_path)

    # Transform only. Never fit on student-test data.
    transformed = processor.transform(combined)

    if transformed.isna()[processor.feature_columns].any().any():
        raise ValueError(
            "Missing values remain after student-test preprocessing"
        )

    scaled_output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(scaled_output_path, index=False)

    print("\nStudent-test feature generation completed")
    print(f"Trips processed: {len(groups)}")
    print(f"Raw feature rows: {len(combined)}")
    print(f"Transformed rows: {len(transformed)}")
    print(f"Raw features: {raw_output_path.relative_to(PROJECT_ROOT)}")
    print(
        "Scaled features: "
        f"{scaled_output_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Preprocessor reused: "
        f"{artifact_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
