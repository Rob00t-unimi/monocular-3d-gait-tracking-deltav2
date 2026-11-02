"""Unit tests for 3D trajectory smoothness and jerk metric."""

import unittest
import numpy as np

from src.metrics.geometry.smoothness import calculate_3d_jerk, SmoothnessResult


class TestSmoothnessMetric(unittest.TestCase):
    def test_constant_velocity_zero_jerk(self):
        """Constant velocity trajectory has zero acceleration and zero jerk."""
        t = np.linspace(0, 2, 60)
        # Position: p(t) = v * t -> jerk = 0
        pos = np.column_stack([2.0 * t, -1.0 * t, 0.5 * t])
        res = calculate_3d_jerk(pos, fps=30.0)

        self.assertIsInstance(res, SmoothnessResult)
        self.assertAlmostEqual(res.rms_jerk, 0.0, places=5)
        self.assertAlmostEqual(res.peak_jerk, 0.0, places=5)

    def test_3d_points_array_shape(self):
        """Accepts (T, N, 3) and computes centroid jerk."""
        t = np.linspace(0, 1, 30)
        # Create 3 points orbiting or moving
        pts = np.zeros((30, 3, 3))
        for i in range(3):
            pts[:, i, 0] = np.sin(2 * np.pi * 2 * t) + i * 0.1
            pts[:, i, 1] = np.cos(2 * np.pi * 2 * t)
            pts[:, i, 2] = t

        res = calculate_3d_jerk(pts, fps=30.0)
        self.assertGreater(res.rms_jerk, 0.0)
        self.assertGreaterEqual(res.peak_jerk, res.rms_jerk)

    def test_invalid_shapes_and_values(self):
        """Raises ValueError on invalid dimensions, too short sequences, or NaN values."""
        with self.assertRaises(ValueError):
            calculate_3d_jerk(np.zeros((3, 3)))  # Too short (< 4 frames)

        with self.assertRaises(ValueError):
            calculate_3d_jerk(np.zeros((10, 2)))  # 2D instead of 3D

        with self.assertRaises(ValueError):
            arr = np.zeros((10, 3))
            arr[2, 0] = np.nan
            calculate_3d_jerk(arr)


if __name__ == "__main__":
    unittest.main()
