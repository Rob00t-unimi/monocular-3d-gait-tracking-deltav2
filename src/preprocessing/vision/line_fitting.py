"""
Module for robust line fitting to edge points.
Uses Huber M-estimators for outlier rejection.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional
from src.core.constants import HUBER_MAX_ANGLE_DEG

logger = logging.getLogger(__name__)

class RobustLineFitter:
    """
    Fits geometric lines to noisy edge data while handling outliers.
    """

    def fit_bounded_line(self, points: List[List[int]], femur_u: np.ndarray, 
                        max_angle_deg: float = HUBER_MAX_ANGLE_DEG) -> Optional[np.ndarray]:
        """
        Fits a line using Huber loss and constrains the deviation from femur axis.
        
        Args:
            points: List of [x, y] edge coordinates.
            femur_u: Longitudinal unit vector of the femur.
            max_angle_deg: Maximum allowed deviation in degrees.
            
        Returns:
            np.ndarray containing [vx, vy, x0, y0] or None if fitting fails.
        """
        if len(points) < 2:
            return None

        pts_arr = np.array(points, dtype=np.float32)
        # Huber M-estimator is less sensitive to outliers than Least Squares
        line = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]
        v_line = np.array([vx, vy])

        # Ensure line vector points in same general direction as femur
        if np.dot(v_line, femur_u) < 0:
            v_line = -v_line

        # Calculate deviation from femur axis
        dot = np.clip(np.dot(v_line, femur_u), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))

        # If line is too skewed, cap the angle
        if angle_deg > max_angle_deg:
            v_line = self._constrain_angle(femur_u, v_line, max_angle_deg)
            vx, vy = v_line[0], v_line[1]

        return np.array([vx, vy, x0, y0])

    def _constrain_angle(self, base_u: np.ndarray, current_v: np.ndarray, max_deg: float) -> np.ndarray:
        """Rotates base vector to cap the deviation angle."""
        cross = base_u[0] * current_v[1] - base_u[1] * current_v[0]
        sign = 1 if cross >= 0 else -1
        rad = np.radians(max_deg) * sign
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        new_vx = base_u[0] * cos_a - base_u[1] * sin_a
        new_vy = base_u[0] * sin_a + base_u[1] * cos_a
        return np.array([new_vx, new_vy])
