
import pandas as pd
import pytest

from pathsnap.features.preprocess import FeaturePreprocessor
from pathsnap.features.build import FEATURE_COLUMNS


@pytest.fixture
def training_data():
    data = {
        "trip_id": ["trip1", "trip1", "trip2"],
        "partition": ["train", "train", "train"],
        "point_idx": [0, 1, 0],
        "candidate_arc_id": [10, 11, 12],
    }

    for column in FEATURE_COLUMNS:
        data[column] = [1.0, 2.0, 3.0]

    data["speed_mps"] = [None, 10.0, 20.0]

    return pd.DataFrame(data)


def test_fit_and_transform(training_data):
    processor = FeaturePreprocessor()
    processor.fit(training_data)

    result = processor.transform(training_data)

    assert result[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert len(result) == len(training_data)


def test_preserves_metadata(training_data):
    processor = FeaturePreprocessor()
    result = processor.fit_transform(training_data)

    for column in [
        "trip_id",
        "partition",
        "point_idx",
        "candidate_arc_id",
    ]:
        assert result[column].tolist() == training_data[column].tolist()


def test_rejects_student_test_during_fit(training_data):
    invalid = training_data.copy()
    invalid.loc[0, "partition"] = "student_test"

    processor = FeaturePreprocessor()

    with pytest.raises(ValueError, match="training data only"):
        processor.fit(invalid)


def test_transform_requires_fit(training_data):
    processor = FeaturePreprocessor()

    with pytest.raises(RuntimeError, match="not been fitted"):
        processor.transform(training_data)


def test_save_and_load(training_data, tmp_path):
    processor = FeaturePreprocessor()
    processor.fit(training_data)

    artifact_path = tmp_path / "preprocessor.joblib"
    processor.save(artifact_path)

    loaded = FeaturePreprocessor.load(artifact_path)

    original_result = processor.transform(training_data)
    loaded_result = loaded.transform(training_data)

    pd.testing.assert_frame_equal(
        original_result,
        loaded_result,
    )


def test_rejects_missing_feature_column(training_data):
    invalid = training_data.drop(columns=[FEATURE_COLUMNS[0]])

    processor = FeaturePreprocessor()

    with pytest.raises(ValueError, match="Missing required"):
        processor.fit(invalid)


def test_rejects_empty_training_data(training_data):
    processor = FeaturePreprocessor()

    with pytest.raises(ValueError, match="empty data"):
        processor.fit(training_data.iloc[0:0])
