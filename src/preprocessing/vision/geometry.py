"""
Module for coordinate systems, vector math, and grid transformations.
Essential for mask generation and point placement in 3D-to-2D space.
"""

import numpy as np
from typing import Tuple, List

class VectorMath:
    """
    Static utilities for femur-aligned coordinate systems.
    """

    @staticmethod
    def calculate_femur_basis(p_knee: np.ndarray, p_hip: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a local 2D basis aligned with the femur (p_knee -> p_hip).
        
        Args:
            p_knee: Coordinates of the knee joint.
            p_hip: Coordinates of the hip joint.
            
        Returns:
            A tuple (length, unit_u, unit_u_perp, full_vector).
        """
        v_femur = p_hip - p_knee
        v_mag = np.linalg.norm(v_femur)
        
        if v_mag == 0:
            return 0.0, np.array([1, 0]), np.array([0, 1]), v_femur
            
        u = v_femur / v_mag
        # Perpendicular vector (rotating 90 degrees CCW)
        u_perp = np.array([-u[1], u[0]])
        
        return float(v_mag), u, u_perp, v_femur

    @staticmethod
    def create_relative_mesh(h: int, w: int, p_origin: np.ndarray, u: np.ndarray, u_perp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates longitudinal and transverse projections for every pixel in the frame.
        
        Args:
            h, w: Frame dimensions.
            p_origin: Origin point (e.g. p_knee).
            u: Longitudinal axis vector.
            u_perp: Transverse axis vector.
            
        Returns:
            Tuple (longitudinal_dist, transverse_projection, absolute_transverse_dist).
        """
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        # Vector from origin to each pixel
        dx = X - p_origin[0]
        dy = Y - p_origin[1]

        # Dot product with basis vectors
        long_dist = dx * u[0] + dy * u[1]
        trans_proj = dx * u_perp[0] + dy * u_perp[1]
        
        return long_dist, trans_proj, np.abs(trans_proj)
