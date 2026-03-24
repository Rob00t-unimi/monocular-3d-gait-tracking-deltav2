"""
Unit tests for the refactored FemurVisionOrchestrator.
"""

import unittest
from src.preprocessing.vision.vision_orchestrator import FemurVisionOrchestrator
import numpy as np
from unittest.mock import patch, MagicMock

class TestFemurVisionOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = FemurVisionOrchestrator()

    def test_compose_empty_points(self):
        # Should return a default triangle if points are empty
        pts, mask1, mask2 = self.orchestrator.compose({})
        self.assertEqual(len(pts), 3)
        self.assertIsNone(mask1)
        self.assertIsNone(mask2)

    @patch('src.preprocessing.vision.vision_orchestrator.VectorMath.calculate_femur_basis')
    def test_compose_no_frame(self, mock_math):
        # Knee at (10, 10), Hip at (10, 20) -> length 10, u=(0, 1)
        points = {"RHip": {"x": 10, "y": 20}, "RKnee": {"x": 10, "y": 10}}
        mock_math.return_value = (10.0, np.array([0, 1]), np.array([-1, 0]), np.array([0, 10]))
        
        from src.core.constants import KNEE_OFFSET_FACTOR
        
        pts, mask1, mask2 = self.orchestrator.compose(points, frame=None)
        self.assertEqual(len(pts), 3)
        
        # Expected shifted P1: (10, 10) + (0, 1) * (10 * 0.05) = (10, 10.5)
        expected_y = 10.0 + 10.0 * KNEE_OFFSET_FACTOR
        self.assertAlmostEqual(pts[0][1], expected_y)
        self.assertIsNone(mask1)
        self.assertIsNone(mask2)

if __name__ == '__main__':
    unittest.main()
