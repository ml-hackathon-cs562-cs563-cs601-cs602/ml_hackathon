
"""
Coordinate projection utilities for PathSnap.

Reuses the tested data loader projection implementation.
Each trip receives an appropriate metric CRS.
"""

from typing import Any

from data_processing.data_loader import project_trip_coordinates


def project_trip(
    trip_id: str,
    partition: str = "train"
) -> dict[str, Any]:
    """
    Load and project GPS observations and road nodes.

    Args:
        trip_id: Unique trip identifier.
        partition: Either train or student_test.

    Returns:
        Dictionary containing projected track, nodes,
        projection type and target CRS.
    """

    return project_trip_coordinates(
        trip_id=trip_id,
        partition=partition
    )
