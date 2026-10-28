"""
Module for Sobel-based edge detection and thigh boundary extraction.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple
from src.core.constants import GAUSSIAN_BLUR_KSIZE, SOBEL_KSIZE, GRADIENT_THRESHOLD, NUM_THIGH_SLICES, THIGH_LENGTH_FACTOR

logger = logging.getLogger(__name__)

class ThighEdgeDetector:
    """
    Finds the left and right boundary points of the thigh in a video frame.
    Uses gradient analysis along the perpendicular axis of the femur.
    """

    def detect_edges(self, frame: np.ndarray, u_perp: np.ndarray, long_dist: np.ndarray, 
                    trans_proj: np.ndarray, trans_dist: np.ndarray, 
                    width_limit: float, v_mag: float) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Extracts sets of edge points for both sides of the thigh.
        
        Args:
            frame: Video frame (BGR).
            u_perp: Unit vector perpendicular to the femur.
            long_dist: Mesh of longitudinal distances.
            trans_proj: Mesh of transverse projections.
            trans_dist: Mesh of absolute transverse distances.
            width_limit: Max transverse search distance.
            v_mag: Femur length in pixels.
            
        Returns:
            Tuple containing lists of [x, y] coordinates for left and right edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSIZE, 0)

        # Sobel gradients to detect intensity changes
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
        
        # Project gradient onto the transverse axis
        grad_perp = np.abs(gx * u_perp[0] + gy * u_perp[1])

        left_pts, right_pts = [], []
        slice_step = (v_mag * THIGH_LENGTH_FACTOR) / NUM_THIGH_SLICES

        for i in range(NUM_THIGH_SLICES):
            l_min = i * slice_step
            l_max = (i + 1) * slice_step
            slice_mask = (long_dist >= l_min) & (long_dist < l_max)

            # Detect Left Edge (transverse projection > 0)
            left_mask = slice_mask & (trans_proj > 0) & (trans_dist <= width_limit)
            if np.any(left_mask):
                y_max, x_max = self._find_max_gradient(grad_perp, left_mask)
                if grad_perp[y_max, x_max] > GRADIENT_THRESHOLD:
                    left_pts.append([int(x_max), int(y_max)])

            # Detect Right Edge (transverse projection < 0)
            right_mask = slice_mask & (trans_proj < 0) & (trans_dist <= width_limit)
            if np.any(right_mask):
                y_max, x_max = self._find_max_gradient(grad_perp, right_mask)
                if grad_perp[y_max, x_max] > GRADIENT_THRESHOLD:
                    right_pts.append([int(x_max), int(y_max)])

        return left_pts, right_pts

    def _find_max_gradient(self, grad_img: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
        """Utility to find the pixel with the highest gradient within a mask."""
        g = np.where(mask, grad_img, 0)
        return np.unravel_index(np.argmax(g), g.shape)
