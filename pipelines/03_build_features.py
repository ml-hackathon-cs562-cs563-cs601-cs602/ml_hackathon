
"""Stage 3: Build candidate-level features and preprocess them."""

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
    """Build features, fit on training data, and save transformed data."""
    config_path = PROJECT_ROOT / "configs" / "features.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    version = str(config["features_version"])

    input_path = PROJECT_ROOT / config["candidate_input_path"]
    raw_output_path = PROJECT_ROOT / config["features_output_path"]
    scaled_output_path = (
        PROJECT_ROOT / config["preprocessed_output_path"]
    )
    artifact_path = (
        PROJECT_ROOT / config["preprocessor_artifact_path"]
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Candidate dataset not found: {input_path}"
        )

    candidates = pd.read_parquet(input_path)

    if candidates.empty:
        raise ValueError("Candidate dataset is empty")

    feature_frames = []

    groups = candidates.groupby(
        ["partition", "trip_id"],
        sort=True,
    )

    print(f"Candidate rows: {len(candidates)}")
    print(f"Trips to process: {len(groups)}")
    print(f"Feature version: {version}")

    for index, ((partition, trip_id), trip_candidates) in enumerate(
        groups,
        start=1,
    ):
        print(
            f"[{index}/{len(groups)}] "
            f"Processing {partition} trip {trip_id}"
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

    key_columns = [
        "trip_id",
        "point_idx",
        "candidate_arc_id",
    ]

    if combined.duplicated(key_columns).any():
        raise ValueError("Duplicate candidate feature rows detected")

    if len(combined) != len(candidates):
        raise ValueError("Feature generation changed the row count")

    # Save original, unscaled feature values.
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(raw_output_path, index=False)

    # Fit preprocessing only on training rows.
    training_rows = combined[
        combined["partition"] == "train"
    ].copy()

    if training_rows.empty:
        raise ValueError("No training rows available for preprocessing")

    processor = FeaturePreprocessor()
    processor.fit(training_rows)

    # Transform the complete dataset using training-fitted parameters.
    transformed = processor.transform(combined)

    scaled_output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(scaled_output_path, index=False)

    # Save fitted preprocessing state for future reuse.
    processor.save(artifact_path)

    print("\nFeature generation and preprocessing completed")
    print(f"Raw feature rows: {len(combined)}")
    print(f"Transformed rows: {len(transformed)}")
    print(f"Training rows used for fitting: {len(training_rows)}")
    print(f"Raw features: {raw_output_path.relative_to(PROJECT_ROOT)}")
    print(
        "Scaled features: "
        f"{scaled_output_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Preprocessor artifact: {artifact_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
