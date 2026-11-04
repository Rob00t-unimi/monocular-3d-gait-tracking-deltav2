"""
Unit tests for 3D Rigidity and Geometric Stability Metric.
"""

import unittest
import numpy as np
from src.metrics.geometry.rigidity import (
    calculate_rigidity_index,
    RigidityResult,
)


class TestRigidityMetric(unittest.TestCase):
    def setUp(self):
        # Initial 3 points forming a right triangle in 3D:
        # P0 at origin, P1 at (3, 0, 0), P2 at (0, 4, 0).
        # Distances: d(0,1) = 3.0, d(0,2) = 4.0, d(1,2) = 5.0
        self.p_init = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0]
        ])
        self.n_frames = 50

    def test_pure_rigid_translation(self):
        # Move the entire triangle arbitrarily across space
        trajs = np.zeros((self.n_frames, 3, 3))
        for t in range(self.n_frames):
            shift = np.array([t * 1.5, np.sin(t) * 10.0, np.cos(t) * 5.0])
            trajs[t] = self.p_init + shift

        res = calculate_rigidity_index(trajs)
        self.assertIsInstance(res, RigidityResult)
        self.assertAlmostEqual(res.sigma_rigidity, 0.0, places=6)
        self.assertAlmostEqual(res.relative_deformation_percent, 0.0, places=6)
        self.assertAlmostEqual(res.max_pairwise_deviation, 0.0, places=6)

        # Initial distances check
        self.assertAlmostEqual(res.pairwise_initial_distances[(0, 1)], 3.0, places=6)
        self.assertAlmostEqual(res.pairwise_initial_distances[(0, 2)], 4.0, places=6)
        self.assertAlmostEqual(res.pairwise_initial_distances[(1, 2)], 5.0, places=6)

    def test_pure_rigid_3d_rotation(self):
        # Rotate triangle around Z axis by angle theta(t)
        trajs = np.zeros((self.n_frames, 3, 3))
        for t in range(self.n_frames):
            theta = np.radians(t * 5.0)
            Rz = np.array([
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta),  np.cos(theta), 0.0],
                [0.0,            0.0,           1.0]
            ])
            trajs[t] = self.p_init @ Rz.T

        res = calculate_rigidity_index(trajs)
        self.assertAlmostEqual(res.sigma_rigidity, 0.0, places=6)
        self.assertAlmostEqual(res.relative_deformation_percent, 0.0, places=6)
        self.assertAlmostEqual(res.max_pairwise_deviation, 0.0, places=6)

    def test_known_deformation(self):
        # Distance between P0 and P1 oscillates sinusoidally between 2.0 and 4.0 (mean 3.0, amp 1.0)
        trajs = np.zeros((self.n_frames, 2, 3))
        t_arr = np.linspace(0, 4 * np.pi, self.n_frames)
        d_t = 3.0 + 1.0 * np.sin(t_arr)

        trajs[:, 0, :] = 0.0
        trajs[:, 1, 0] = d_t

        res = calculate_rigidity_index(trajs)
        # Expected std of a sine wave with amplitude 1.0 is ~ 1.0 / sqrt(2) ≈ 0.707
        expected_std = float(np.std(d_t))
        self.assertAlmostEqual(res.sigma_rigidity, expected_std, places=5)
        self.assertGreater(res.relative_deformation_percent, 20.0)
        self.assertAlmostEqual(res.max_pairwise_deviation, 1.0, delta=0.005)

    def test_dense_cluster_supported(self):
        # 10 points
        np.random.seed(42)
        n_pts = 10
        pts = np.random.randn(n_pts, 3)
        trajs = np.tile(pts[None, :, :], (self.n_frames, 1, 1))

        res = calculate_rigidity_index(trajs)
        # 10 points -> 45 pairs
        self.assertEqual(len(res.pairwise_stds), 45)
        self.assertAlmostEqual(res.sigma_rigidity, 0.0, places=6)

    def test_2d_points_supported(self):
        pts_2d = self.p_init[:, :2]
        trajs_2d = np.tile(pts_2d[None, :, :], (20, 1, 1))
        res = calculate_rigidity_index(trajs_2d)
        self.assertAlmostEqual(res.sigma_rigidity, 0.0, places=6)

    def test_invalid_shapes_raise_error(self):
        # 2D instead of 3D tensor
        with self.assertRaises(ValueError):
            calculate_rigidity_index(np.zeros((10, 3)))
        # Only 1 point
        with self.assertRaises(ValueError):
            calculate_rigidity_index(np.zeros((10, 1, 3)))
        # Only 1 frame
        with self.assertRaises(ValueError):
            calculate_rigidity_index(np.zeros((1, 3, 3)))
        # NaN values
        arr = np.zeros((10, 3, 3))
        arr[2, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            calculate_rigidity_index(arr)


if __name__ == "__main__":
    unittest.main()
