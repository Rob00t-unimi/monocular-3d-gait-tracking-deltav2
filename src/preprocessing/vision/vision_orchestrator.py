"""
Orchestrator for the computer vision pipeline (thigh masking and triangulation).
Implements IPointComposer protocol.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from src.core.interfaces import IPointComposer
from src.core.constants import THIGH_WIDTH_FACTOR, SAFETY_MARGIN_PX, DEFAULT_TRIANGLE_SIZE, KNEE_OFFSET_FACTOR
from src.preprocessing.vision.geometry import VectorMath
from src.preprocessing.vision.edge_processing import ThighEdgeDetector
from src.preprocessing.vision.line_fitting import RobustLineFitter
from src.preprocessing.vision.thigh_point_selector import ThighPointSelector

logger = logging.getLogger(__name__)

class FemurVisionOrchestrator(IPointComposer):
    """
    Coordinates geometry, edge detection, and robust fitting to generate the femur triangle.
    """

    def __init__(self):
        self.edge_detector = ThighEdgeDetector()
        self.line_fitter = RobustLineFitter()
        self.point_selector = ThighPointSelector()

    def compose(self, points: Dict[str, Any], frame: Optional[np.ndarray] = None) -> Tuple[List[List[float]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculates tracking points and generates visualization masks.
        """
        if not points or "RHip" not in points or "RKnee" not in points:
            return self._fallback_triangle(points), None, None

        p_hip = np.array([points["RHip"]["x"], points["RHip"]["y"]])
        p_knee = np.array([points["RKnee"]["x"], points["RKnee"]["y"]])

        # 1. Base Geometry
        v_mag, u, u_perp, v_femur = VectorMath.calculate_femur_basis(p_knee, p_hip)
        if v_mag == 0: return self._fallback_triangle(points), None, None

        # Fallback to pure math if no frame provided
        if frame is None:
            pts = self._math_only_triangle(p_knee, p_hip, v_mag)
            return pts, None, None

        h, w = frame.shape[:2]
        long_dist, trans_proj, trans_dist = VectorMath.create_relative_mesh(h, w, p_knee, u, u_perp)

        # 2. Edge Processing
        width_limit = THIGH_WIDTH_FACTOR * v_mag
        left_pts, right_pts = self.edge_detector.detect_edges(frame, u_perp, long_dist, trans_proj, trans_dist, width_limit, v_mag)

        line_l = self.line_fitter.fit_bounded_line(left_pts, u)
        line_r = self.line_fitter.fit_bounded_line(right_pts, u)

        # 3. Mask Generation
        base_mask_bool = (long_dist >= 0) & (long_dist <= v_mag * 0.35) & (trans_dist <= width_limit)
        inner_mask = self._trim_mask(base_mask_bool, line_l, line_r, u, u_perp, p_knee, long_dist, trans_proj, SAFETY_MARGIN_PX)
        outer_mask = self._trim_mask(base_mask_bool, line_l, line_r, u, u_perp, p_knee, long_dist, trans_proj, 0.0)

        # 4. Point Selection
        triangle = self.point_selector.select_points(p_knee, u, u_perp, v_mag, v_femur, inner_mask, h, w)

        return triangle, (outer_mask * 255).astype(np.uint8), (inner_mask * 255).astype(np.uint8)

    def _trim_mask(self, base_mask, l_left, l_right, u, u_perp, p1, long_d, trans_p, margin):
        """Internal logic to slice the mask using fitted lines."""
        mask = base_mask.copy()
        for line, side in [(l_left, 1), (l_right, -1)]:
            if line is not None:
                vx, vy, x0, y0 = line
                V = np.array([vx, vy])
                P0 = np.array([x0, y0])
                dL = np.dot(V, u)
                if abs(dL) > 1e-5:
                    m = np.dot(V, u_perp) / dL
                    L0 = np.dot(P0 - p1, u)
                    T0 = np.dot(P0 - p1, u_perp)
                    T_exact = T0 + m * (long_d - L0)
                    if side == 1: mask = mask & (trans_p <= (T_exact - margin))
                    else: mask = mask & (trans_p >= (T_exact + margin))
        return mask.astype(np.uint8)

    def _fallback_triangle(self, points):
        """Small default triangle when points are missing."""
        base = [50.0, 50.0]
        if points:
            k = next(iter(points.values()))
            base = [float(k["x"]), float(k["y"])]
        s = DEFAULT_TRIANGLE_SIZE
        return [base, [base[0] + s, base[1]], [base[0], base[1] + s]]

    def _math_only_triangle(self, p1, p_hip, v_mag):
        """Geometric calculation when image analysis is impossible."""
        v = (p_hip - p1) / v_mag
        v_p = np.array([-v[1], v[0]])
        # Use original p1 for search points P2 and P3
        p2 = p1 + v * (v_mag * 0.1) + v_p * (v_mag * 0.1)
        p3 = p1 + v * (v_mag * 0.1) - v_p * (v_mag * 0.1)
        # Shift ONLY P1
        p1_shifted = p1 + v * (v_mag * KNEE_OFFSET_FACTOR)
        return [p1_shifted.tolist(), p2.tolist(), p3.tolist()]
