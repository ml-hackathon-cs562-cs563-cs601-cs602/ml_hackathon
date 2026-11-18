
import math

import pytest

from pathsnap.geo.distances import (
    point_to_segment_distance,
    along_segment_position,
    heading_difference,
    segment_heading,
)


def test_point_to_segment_distance():
    distance = point_to_segment_distance(
        (5, 3), (0, 0), (10, 0)
    )
    assert distance == pytest.approx(3.0)


def test_point_beyond_segment_endpoint():
    distance = point_to_segment_distance(
        (12, 0), (0, 0), (10, 0)
    )
    assert distance == pytest.approx(2.0)


def test_zero_length_segment():
    distance = point_to_segment_distance(
        (3, 4), (0, 0), (0, 0)
    )
    assert distance == pytest.approx(5.0)


def test_along_segment_position():
    position = along_segment_position(
        (5, 3), (0, 0), (10, 0)
    )
    assert position == pytest.approx(0.5)


def test_position_clamped_to_segment():
    position = along_segment_position(
        (15, 0), (0, 0), (10, 0)
    )
    assert position == pytest.approx(1.0)


def test_zero_length_segment_position():
    position = along_segment_position(
        (5, 5), (0, 0), (0, 0)
    )
    assert position == 0.0


def test_heading_difference_wraparound():
    assert heading_difference(350, 10) == pytest.approx(20.0)


def test_heading_difference_opposite_directions():
    assert heading_difference(0, 180) == pytest.approx(180.0)


def test_eastward_segment_heading():
    heading = segment_heading((0, 0), (10, 0))
    assert heading == pytest.approx(90.0)


def test_zero_length_segment_heading():
    heading = segment_heading((0, 0), (0, 0))
    assert heading == 0.0
