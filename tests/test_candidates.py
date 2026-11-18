
import pandas as pd
import pytest

from pathsnap.candidates import generate as candidate_module


@pytest.fixture
def synthetic_trip(monkeypatch):
    """Create a small projected road network for testing."""

    track = pd.DataFrame({
        "x": [5.0, 15.0, 50.0],
        "y": [1.0, 1.0, 1.0],
        "timestamp": [0, 1, 2],
    })

    nodes = pd.DataFrame({
        "x": [0.0, 10.0, 20.0, 30.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    })

    arcs = pd.DataFrame({
        "from_node_id": [0, 1, 2],
        "to_node_id": [1, 2, 3],
    })

    monkeypatch.setattr(
        candidate_module,
        "project_trip_coordinates",
        lambda trip_id, partition: {
            "track": track,
            "nodes": nodes,
        },
    )

    monkeypatch.setattr(
        candidate_module,
        "load_arcs",
        lambda trip_id, partition: arcs,
    )


def test_generates_nearest_candidates(synthetic_trip):
    result = candidate_module.generate_candidates(
        "test_trip", "train", k=2, radius_m=100
    )

    first_point = result[result["point_idx"] == 0]

    assert len(first_point) == 2
    assert first_point.iloc[0]["candidate_arc_id"] == 0
    assert first_point.iloc[0]["dist_m"] == pytest.approx(1.0)


def test_respects_candidate_limit(synthetic_trip):
    result = candidate_module.generate_candidates(
        "test_trip", "train", k=1, radius_m=100
    )

    assert result.groupby("point_idx").size().max() <= 1


def test_excludes_candidates_outside_radius(synthetic_trip):
    result = candidate_module.generate_candidates(
        "test_trip", "train", k=10, radius_m=2
    )

    assert (result["dist_m"] <= 2).all()


def test_candidate_ranks_are_sequential(synthetic_trip):
    result = candidate_module.generate_candidates(
        "test_trip", "train", k=3, radius_m=100
    )

    for _, group in result.groupby("point_idx"):
        assert group["candidate_rank"].tolist() == list(
            range(1, len(group) + 1)
        )


@pytest.mark.parametrize(
    "k,radius",
    [(0, 100), (-1, 100), (10, 0), (10, -5)],
)
def test_rejects_invalid_parameters(synthetic_trip, k, radius):
    with pytest.raises(ValueError):
        candidate_module.generate_candidates(
            "test_trip", "train", k, radius
        )

def test_empty_road_network(monkeypatch):
    """Return an empty result when no road arcs exist."""
    track = pd.DataFrame({
        "x": [5.0],
        "y": [1.0],
        "timestamp": [0],
    })
    nodes = pd.DataFrame({
        "x": [0.0],
        "y": [0.0],
    })
    arcs = pd.DataFrame(
        columns=["from_node_id", "to_node_id"]
    )

    monkeypatch.setattr(
        candidate_module,
        "project_trip_coordinates",
        lambda trip_id, partition: {
            "track": track,
            "nodes": nodes,
        },
    )
    monkeypatch.setattr(
        candidate_module,
        "load_arcs",
        lambda trip_id, partition: arcs,
    )

    result = candidate_module.generate_candidates(
        "empty_trip", "train", k=10, radius_m=100
    )

    assert result.empty
    assert "candidate_arc_id" in result.columns


def test_missing_node_reference(monkeypatch):
    """Raise a clear error when an arc references a missing node."""
    track = pd.DataFrame({
        "x": [5.0],
        "y": [1.0],
        "timestamp": [0],
    })
    nodes = pd.DataFrame({
        "x": [0.0],
        "y": [0.0],
    })
    arcs = pd.DataFrame({
        "from_node_id": [0],
        "to_node_id": [99],
    })

    monkeypatch.setattr(
        candidate_module,
        "project_trip_coordinates",
        lambda trip_id, partition: {
            "track": track,
            "nodes": nodes,
        },
    )
    monkeypatch.setattr(
        candidate_module,
        "load_arcs",
        lambda trip_id, partition: arcs,
    )

    with pytest.raises(ValueError, match="missing road node"):
        candidate_module.generate_candidates(
            "invalid_trip", "train", k=10, radius_m=100
        )
