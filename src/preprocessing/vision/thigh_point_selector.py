"""
Module for selecting the final 3 tracking points on the thigh mask.
Handles obtuse angle logic and geometric constraints.
"""

import numpy as np
from typing import List, Tuple
from src.core.constants import KNEE_OFFSET_FACTOR

class ThighPointSelector:
    """
    Calculates P1 (shifted knee), P2, and P3 points within the identified leg boundaries.
    """

    def select_points(self, p1: np.ndarray, u: np.ndarray, u_perp: np.ndarray, 
                     v_mag: float, v_femur: np.ndarray, 
                     inner_mask: np.ndarray, h: int, w: int) -> List[List[float]]:
        """
        Calculates the sensor triangle points.
        
        Args:
            p1: Original knee joint [x, y].
            u: Longitudinal axis unit vector.
            u_perp: Transverse axis unit vector.
            v_mag: Femur length.
            v_femur: Full femur vector.
            inner_mask: Refined binary leg mask.
            h, w: Frame dimensions.
            
        Returns:
            List of three [x, y] coordinate pairs (P1 shifted up, P2, P3).
        """
        # 1. Determine if the femur orientation is obtuse (relative to frame axes)
        femur_angle_deg = np.degrees(np.arctan2(-v_femur[1], v_femur[0]))
        if femur_angle_deg < 0: femur_angle_deg += 360
        is_obtuse = 90 < femur_angle_deg < 270

        # 2. Define anchor and search areas (lower third of thigh)
        # We use the ORIGINAL p1 to ensure P2 and P3 fall within the mask (0 to 0.35 v_mag)
        anchor_p2 = p1 + u * (v_mag * 0.3)
        search_center = (p1 + anchor_p2) / 2.0
        offset_val = 0.1 * v_mag

        # 3. Find P3 (Transverse offset from center)
        p3 = search_center.astype(int)
        chosen_side = 1

        for side in [1, -1]:
            tmp = side * u_perp * offset_val
            p3_candidate = (search_center + (-tmp if is_obtuse else tmp)).astype(int)

            if self._is_inside(p3_candidate, inner_mask, h, w):
                p3 = p3_candidate
                chosen_side = side
                break

        # 4. Find P2 (Transverse offset from anchor)
        tmp = chosen_side * u_perp * offset_val
        p2_candidate = (anchor_p2 + (tmp if is_obtuse else -tmp)).astype(int)

        p2 = p2_candidate if self._is_inside(p2_candidate, inner_mask, h, w) else anchor_p2.astype(int)

        # 5. Shift ONLY the first point (P1) slightly higher to avoid the joint
        p1_shifted = p1 + u * (v_mag * KNEE_OFFSET_FACTOR)

        return [
            [float(p1_shifted[0]), float(p1_shifted[1])],
            [float(p2[0]), float(p2[1])],
            [float(p3[0]), float(p3[1])]
        ]

    def _is_inside(self, p: np.ndarray, mask: np.ndarray, h: int, w: int) -> bool:
        """Utility to check if a point is within frame bounds and mask."""
        return 0 <= p[1] < h and 0 <= p[0] < w and mask[p[1], p[0]] == 1
