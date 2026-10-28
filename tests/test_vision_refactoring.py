"""
Unit tests for the refactored vision and coordinate system modules.
Verifies vector math, mesh generation, and line fitting logic.
"""

import unittest
import numpy as np
from src.preprocessing.vision.geometry import VectorMath
from src.preprocessing.vision.line_fitting import RobustLineFitter

class TestVisionProcessing(unittest.TestCase):
    def test_femur_basis_vertical(self):
        # Knee at (10, 10), Hip at (10, 20) -> Vector (0, 10)
        p_knee = np.array([10, 10])
        p_hip = np.array([10, 20])
        
        v_mag, u, u_perp, v_femur = VectorMath.calculate_femur_basis(p_knee, p_hip)
        
        self.assertEqual(v_mag, 10.0)
        np.testing.assert_array_equal(u, [0, 1])
        np.testing.assert_array_equal(u_perp, [-1, 0])
        np.testing.assert_array_equal(v_femur, [0, 10])

    def test_relative_mesh(self):
        # 10x10 frame, basis at (5, 5) pointing right
        p_origin = np.array([5, 5])
        u = np.array([1, 0])
        u_perp = np.array([0, 1])
        
        long_dist, trans_proj, trans_dist = VectorMath.create_relative_mesh(10, 10, p_origin, u, u_perp)
        
        # Point (6, 5) should have long_dist=1, trans_proj=0
        self.assertEqual(long_dist[5, 6], 1.0)
        self.assertEqual(trans_proj[5, 6], 0.0)
        
        # Point (5, 6) should have long_dist=0, trans_proj=1
        self.assertEqual(long_dist[6, 5], 0.0)
        self.assertEqual(trans_proj[6, 5], 1.0)

    def test_robust_line_fitting_constrained(self):
        fitter = RobustLineFitter()
        # Perfect vertical femur
        femur_u = np.array([0, 1])
        # Skewed edge points (45 degrees)
        pts = [[10, 10], [11, 11], [12, 12]]
        
        # Fit with 20 deg constraint
        line = fitter.fit_bounded_line(pts, femur_u, max_angle_deg=20.0)
        
        vx, vy, x0, y0 = line
        # The angle must be constrained to 20 degrees from vertical
        angle = np.degrees(np.arccos(np.dot([vx, vy], femur_u)))
        self.assertAlmostEqual(angle, 20.0, places=2)

if __name__ == '__main__':
    unittest.main()
