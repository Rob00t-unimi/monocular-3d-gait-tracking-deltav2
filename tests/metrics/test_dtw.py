"""
Unit tests for Dynamic Time Warping (DTW) Metric.
"""

import unittest
import numpy as np
from src.metrics.geometry.dtw import calculate_dtw, DTWResult


class TestDTWMetric(unittest.TestCase):
    def test_identical_1d_signals(self):
        y = np.sin(np.linspace(0, 4 * np.pi, 100))
        res = calculate_dtw(y, y)
        self.assertIsInstance(res, DTWResult)
        self.assertAlmostEqual(res.total_distance, 0.0, places=6)
        self.assertAlmostEqual(res.normalized_distance, 0.0, places=6)
        self.assertEqual(res.path_length, 100)

    def test_identical_3d_trajectories(self):
        t = np.linspace(0, 2 * np.pi, 80)
        traj = np.stack([np.sin(t), np.cos(t), t], axis=-1)
        res = calculate_dtw(traj, traj)
        self.assertAlmostEqual(res.total_distance, 0.0, places=6)
        self.assertAlmostEqual(res.normalized_distance, 0.0, places=6)
        self.assertEqual(res.path_length, 80)

    def test_time_warped_signal(self):
        # Two signals with same shape but non-linear time stretching
        t1 = np.linspace(0, 2 * np.pi, 100)
        t2 = t1 ** 1.3  # Non-linear time warp
        t2 = t2 / t2[-1] * (2 * np.pi)

        y1 = np.sin(t1)
        y2 = np.sin(t2)

        # Standard Euclidean MSE between y1 and y2 would be large
        raw_rmse = np.sqrt(np.mean((y1 - y2) ** 2))
        self.assertGreater(raw_rmse, 0.2)

        # DTW aligns the waveforms with very small normalized distance
        res = calculate_dtw(y1, y2)
        self.assertLess(res.normalized_distance, 0.05)

    def test_different_sequence_lengths(self):
        # 60 samples vs 100 samples of the same underlying gait cycle
        t1 = np.linspace(0, 2 * np.pi, 60)
        t2 = np.linspace(0, 2 * np.pi, 100)
        y1 = np.sin(t1)
        y2 = np.sin(t2)

        res = calculate_dtw(y1, y2)
        self.assertLess(res.normalized_distance, 0.05)
        self.assertGreaterEqual(res.path_length, 100)

    def test_normalization_option(self):
        t = np.linspace(0, 2 * np.pi, 50)
        y1 = np.sin(t)
        # Scaled and shifted in amplitude
        y2 = 5.0 * y1 + 10.0

        # Without normalization, distance reflects scale difference
        res_raw = calculate_dtw(y1, y2, normalize=False)
        self.assertGreater(res_raw.normalized_distance, 5.0)

        # With normalization, distance drops to near zero
        res_norm = calculate_dtw(y1, y2, normalize=True)
        self.assertAlmostEqual(res_norm.normalized_distance, 0.0, places=5)

    def test_window_constraint(self):
        t = np.linspace(0, 2 * np.pi, 60)
        y = np.sin(t)
        res_windowed = calculate_dtw(y, y, window=10)
        self.assertAlmostEqual(res_windowed.total_distance, 0.0, places=6)

    def test_dimension_mismatch_raises_error(self):
        traj_3d = np.zeros((20, 3))
        traj_2d = np.zeros((20, 2))
        with self.assertRaises(ValueError):
            calculate_dtw(traj_3d, traj_2d)

    def test_empty_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_dtw([], [1.0, 2.0])

    def test_nan_values_raise_error(self):
        with self.assertRaises(ValueError):
            calculate_dtw([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
