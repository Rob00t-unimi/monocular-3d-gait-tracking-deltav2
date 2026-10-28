# -*- coding: utf-8 -*-
"""
Postprocessing package for 3D Gait Analysis Point Tracking.
Provides geometric pruning, temporal persistence filtering, and optimal triplet selection.
"""

from .geometric_pruner import (
    RefinedTrackingResult,
    GeometricPruner,
    calc_area_2d,
    calc_area_3d,
    get_convex_hull_area_2d,
    apply_convex_hull_pruning,
    apply_adaptive_pca_dbscan,
    select_optimal_triplet,
)

__all__ = [
    "RefinedTrackingResult",
    "GeometricPruner",
    "calc_area_2d",
    "calc_area_3d",
    "get_convex_hull_area_2d",
    "apply_convex_hull_pruning",
    "apply_adaptive_pca_dbscan",
    "select_optimal_triplet",
]
