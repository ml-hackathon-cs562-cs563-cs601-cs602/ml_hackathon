
import numpy as np
import pandas as pd
import pytest

from pathsnap.features import build as feature_module


@pytest.fixture
def synthetic_feature_data(monkeypatch):
    """Create a small GPS track and road network."""

    track = pd.DataFrame({
        "x": [5.0, 15.0, 25.0],
        "y": [1.0, 1.0, 1.0],
        "timestamp": [0.0, 1.0, 2.0],
    })

    nodes = pd.DataFrame({
        "x": [0.0, 10.0, 20.0],
        "y": [0.0, 0.0, 0.0],
    })

    arcs = pd.DataFrame({
        "from_node_id": [0, 1],
        "to_node_id": [1, 2],
    })

    candidates = pd.DataFrame({
        "trip_id": ["test_trip"] * 3,
        "partition": ["train"] * 3,
        "point_idx": [0, 1, 2],
        "candidate_arc_id": [0, 1, 1],
        "candidate_rank": [1, 1, 1],
        "dist_m": [1.0, 1.0, 5.0],
    })

    monkeypatch.setattr(
        feature_module,
        "project_trip_coordinates",
        lambda trip_id, partition: {
            "track": track,
            "nodes": nodes,
        },
    )

    monkeypatch.setattr(
        feature_module,
        "load_arcs",
        lambda trip_id, partition: arcs,
    )

    return candidates


def test_generates_all_feature_columns(synthetic_feature_data):
    result = feature_module.build_features(
        synthetic_feature_data, "test_trip"
    )

    for column in feature_module.FEATURE_COLUMNS:
        assert column in result.columns

    assert result["features_version"].eq("v0").all()


def test_preserves_candidate_rows(synthetic_feature_data):
    result = feature_module.build_features(
        synthetic_feature_data, "test_trip"
    )

    assert len(result) == len(synthetic_feature_data)
    assert result["candidate_arc_id"].tolist() == [0, 1, 1]
    assert result["point_idx"].tolist() == [0, 1, 2]


def test_heading_and_position_ranges(synthetic_feature_data):
    result = feature_module.build_features(
        synthetic_feature_data, "test_trip"
    )

    assert result["heading_diff"].dropna().between(0, 180).all()
    assert result["along_arc_position"].between(0, 1).all()


def test_movement_features(synthetic_feature_data):
    result = feature_module.build_features(
        synthetic_feature_data, "test_trip"
    )

    assert result.loc[1, "speed_mps"] == pytest.approx(10.0)
    assert result.loc[2, "speed_mps"] == pytest.approx(10.0)
    assert result.loc[1, "time_gap_s"] == pytest.approx(1.0)


def test_trip_boundary_missing_values(synthetic_feature_data):
    result = feature_module.build_features(
        synthetic_feature_data, "test_trip"
    )

    assert pd.isna(result.loc[0, "prev_point_dist_m"])
    assert pd.isna(result.loc[2, "next_point_dist_m"])
    assert pd.isna(result.loc[0, "time_gap_s"])


def test_arc_length_and_node_degree(synthetic_feature_data):
    result = feature_module.build_features(
        synthetic_feature_data, "test_trip"
    )

    assert result["arc_length_m"].tolist() == pytest.approx(
        [10.0, 10.0, 10.0]
    )
    assert result["node_degree"].tolist() == [3, 3, 3]


def test_rejects_wrong_trip(synthetic_feature_data):
    with pytest.raises(ValueError, match="belong"):
        feature_module.build_features(
            synthetic_feature_data, "another_trip"
        )


def test_rejects_unknown_candidate_arc(synthetic_feature_data):
    invalid = synthetic_feature_data.copy()
    invalid.loc[0, "candidate_arc_id"] = 999

    with pytest.raises(ValueError, match="unknown road arc"):
        feature_module.build_features(invalid, "test_trip")
