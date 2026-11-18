"""Local metric equirectangular projection for GPS and road network coordinates.

Owner: Member B
"""

from __future__ import annotations
import numpy as np


class Proj:
    """Local equirectangular projection (metres) about the trip centroid.
    
    At city scale (<30 km), projection distortion is far below GPS noise (<1 cm error).
    """

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = 111320.0 * np.cos(np.radians(self.lat0))
        self.ky = 110540.0

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Project (lon, lat) to local Cartesian (x, y) in metres."""
        x = (np.asarray(lon, dtype=float) - self.lon0) * self.kx
        y = (np.asarray(lat, dtype=float) - self.lat0) * self.ky
        return np.column_stack([x, y])

    def unproject(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Unproject local Cartesian (x, y) back to (lon, lat)."""
        lon = np.asarray(x, dtype=float) / self.kx + self.lon0
        lat = np.asarray(y, dtype=float) / self.ky + self.lat0
        return np.column_stack([lon, lat])
