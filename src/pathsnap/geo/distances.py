
"""Metric geometry utilities for GPS-to-road candidate features."""

import math

import numpy as np


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return the shortest distance from a point to a segment, in metres."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)

    segment = b - a
    length_squared = float(np.dot(segment, segment))

    if length_squared == 0:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, segment) / length_squared)
    t = max(0.0, min(1.0, t))

    closest = a + t * segment
    return float(np.linalg.norm(p - closest))


def along_segment_position(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return the projected position along a segment, between 0 and 1."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)

    segment = b - a
    length_squared = float(np.dot(segment, segment))

    if length_squared == 0:
        return 0.0

    t = float(np.dot(p - a, segment) / length_squared)
    return max(0.0, min(1.0, t))


def heading_difference(
    gps_heading: float,
    road_heading: float,
) -> float:
    """Return the smallest absolute angular difference in degrees."""
    difference = abs(gps_heading - road_heading) % 360.0
    return min(difference, 360.0 - difference)


def segment_heading(
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return segment heading in degrees clockwise from north."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if dx == 0 and dy == 0:
        return 0.0

    return math.degrees(math.atan2(dx, dy)) % 360.0
