"""
Unit tests for ThighPointSelector.
Verifies the shift of the knee point (P1) and triangle formation.
"""

import unittest
import numpy as np
from src.preprocessing.vision.thigh_point_selector import ThighPointSelector
from src.core.constants import KNEE_OFFSET_FACTOR

class TestThighPointSelector(unittest.TestCase):
    def setUp(self):
        self.selector = ThighPointSelector()

    def test_p1_shift(self):
        # Vertical femur: Knee at (10, 10), Hip at (10, 20) -> Length 10, u=(0, 1)
        p1 = np.array([10.0, 10.0])
        u = np.array([0.0, 1.0])
        u_perp = np.array([-1.0, 0.0])
        v_mag = 10.0
        v_femur = np.array([0.0, 10.0])
        
        # Mask: all ones for simplicity
        h, w = 100, 100
        inner_mask = np.ones((h, w), dtype=np.uint8)
        
        pts = self.selector.select_points(p1, u, u_perp, v_mag, v_femur, inner_mask, h, w)
        
        # P1 shifted should be p1 + u * (v_mag * KNEE_OFFSET_FACTOR)
        expected_p1 = p1 + u * (v_mag * KNEE_OFFSET_FACTOR)
        
        self.assertAlmostEqual(pts[0][0], expected_p1[0])
        self.assertAlmostEqual(pts[0][1], expected_p1[1])
        
        # P2 should be based on original p1. 
        # Longitudinal position: 10 + (10 * 0.3) = 13.0
        # If it was based on p1_shifted, it would be 10.5 + 3.0 = 13.5
        self.assertAlmostEqual(pts[1][1], 13.0)
        
        # P2 and P3 should be different from P1
        self.assertNotEqual(pts[0], pts[1])
        self.assertNotEqual(pts[0], pts[2])
        self.assertNotEqual(pts[1], pts[2])

if __name__ == '__main__':
    unittest.main()
