# -*- coding: utf-8 -*-
"""
Unit tests for the Geometric Pruning and Triplet Selection Module.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.postprocessing.geometric_pruner import (
    RefinedTrackingResult,
    GeometricPruner,
    calc_area_2d,
    calc_area_3d,
    get_convex_hull_area_2d,
    apply_convex_hull_pruning,
    select_optimal_triplet,
)


def test_calc_area_2d_right_triangle():
    """Area of a right triangle with base 3 and height 4 should be 6.0."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 0.0])
    p3 = np.array([0.0, 4.0])
    area = calc_area_2d(p1, p2, p3)
    assert np.isclose(area, 6.0)


def test_calc_area_3d_perpendicular_vectors():
    """Area of a 3D right triangle formed by vectors of length 3 and 4 should be 6.0."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 0.0, 0.0])
    p3 = np.array([0.0, 4.0, 0.0])
    area = calc_area_3d(p1, p2, p3)
    assert np.isclose(area, 6.0)


def test_get_convex_hull_area_square():
    """Convex hull of a unit square should be 1.0."""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5]  # interior point
    ])
    area = get_convex_hull_area_2d(square)
    assert np.isclose(area, 1.0)


def test_convex_hull_pruning_removes_outliers():
    """
    Simulates T=10 frames with 4 stable points and 1 outlier that drifts far away at frame t=5.
    The outlier must be eliminated by 100% persistence filter.
    """
    T = 10
    N = 5
    points_3d = np.zeros((T, N, 3))
    base_square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    for t in range(T):
        points_3d[t, :4, :2] = base_square + 0.01 * np.sin(t)
        points_3d[t, :4, 2] = 1.0
        if t == 5:
            points_3d[t, 4, :2] = [50.0, 50.0]
        else:
            points_3d[t, 4, :2] = [0.5, 0.5]
        points_3d[t, 4, 2] = 1.0

    survived = apply_convex_hull_pruning(points_3d, ref_idx_rel=0, max_area_ratio=2.0)

    assert 4 not in survived
    assert len(survived) >= 3
    for idx in range(4):
        assert idx in survived


def test_select_optimal_triplet_rejects_collinear():
    """
    Ensures that a degenerate collinear triangle (area near 0) is rejected
    in favor of a non-collinear triangle with sufficient 3D area.
    """
    T = 5
    pts_2d = np.zeros((T, 4, 2))
    pts_3d = np.zeros((T, 4, 3))

    for t in range(T):
        pts_2d[t] = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0]
        ])
        pts_3d[t, :, :2] = pts_2d[t]
        pts_3d[t, :, 2] = 1.0

    original_idx = (0, 1, 3)
    cand_idx = np.array([0, 1, 2, 3])

    best = select_optimal_triplet(
        points_2d=pts_2d,
        points_3d=pts_3d,
        candidate_indices=cand_idx,
        ref_idx_rel=0,
        original_indices=original_idx,
        min_3d_ratio=0.05
    )

    assert 3 in best
    assert set(best) != {0, 1, 2}


def test_refined_result_npz_roundtrip():
    """Tests saving and loading RefinedTrackingResult to ensure complete tensor integrity."""
    T, M = 30, 15
    triplet_3d = np.random.randn(T, 3, 3).astype(np.float32)
    triplet_2d = np.random.randn(T, 3, 2).astype(np.float32)
    triplet_idx = np.array([0, 5, 9], dtype=np.int32)
    survived_3d = np.random.randn(T, M, 3).astype(np.float32)
    survived_2d = np.random.randn(T, M, 2).astype(np.float32)
    survived_idx = np.arange(M, dtype=np.int32)

    result = RefinedTrackingResult(
        triplet_track3d=triplet_3d,
        triplet_track2d=triplet_2d,
        triplet_indices=triplet_idx,
        survived_track3d=survived_3d,
        survived_track2d=survived_2d,
        survived_indices=survived_idx,
        video_name="test_video.mp4",
        ref_frame_idx=10,
        start_idx=0,
        fps=30.0,
        mode="test_mode"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_refined.npz"
        result.save_npz(out_path)
        assert out_path.exists()

        loaded = RefinedTrackingResult.load_npz(out_path)
        assert loaded.video_name == "test_video.mp4"
        assert loaded.ref_frame_idx == 10
        assert loaded.start_idx == 0
        assert np.allclose(loaded.triplet_track3d, triplet_3d)
        assert np.allclose(loaded.survived_track3d, survived_3d)
        assert np.array_equal(loaded.triplet_indices, triplet_idx)
        assert np.array_equal(loaded.survived_indices, survived_idx)


def test_geometric_pruner_end_to_end():
    """Tests full execution of GeometricPruner.refine() on a synthetic tracking dataset."""
    T, N = 40, 20
    np.random.seed(42)

    t = np.linspace(0, 4 * np.pi, T)
    track2d = np.zeros((T, N, 2), dtype=np.float32)
    track3d = np.zeros((T, N, 3), dtype=np.float32)

    for i in range(N):
        track2d[:, i, 0] = 500.0 + 30.0 * np.sin(t) + i * 5.0
        track2d[:, i, 1] = 400.0 + 20.0 * np.cos(t) + i * 5.0
        track3d[:, i, 0] = 0.5 * np.sin(t) + i * 0.02
        track3d[:, i, 1] = 0.3 * np.cos(t) + i * 0.02
        track3d[:, i, 2] = 2.0 + 0.1 * np.sin(t)

    pruner = GeometricPruner(cutoff_freq=3.0, max_area_ratio=5.0, num_thigh_points=N)
    refined = pruner.refine(
        track2d=track2d,
        track3d_world=track3d,
        ref_frame_idx=10,
        start_idx=0,
        video_name="synth_video.mp4",
        fps=30.0
    )

    assert isinstance(refined, RefinedTrackingResult)
    assert refined.triplet_track3d.shape == (T, 3, 3)
    assert refined.triplet_track2d.shape == (T, 3, 2)
    assert len(refined.triplet_indices) == 3
    assert refined.survived_track3d.shape[0] == T
    assert refined.survived_track3d.shape[2] == 3
    assert len(refined.survived_indices) >= 3
    assert not np.isnan(refined.triplet_track3d).any()
    assert not np.isnan(refined.survived_track3d).any()


class TestGeometricPruning(unittest.TestCase):
    """Unittest-compatible test suite wrapper for GeometricPruner."""

    def test_calc_area_2d_right_triangle(self):
        test_calc_area_2d_right_triangle()

    def test_calc_area_3d_perpendicular_vectors(self):
        test_calc_area_3d_perpendicular_vectors()

    def test_get_convex_hull_area_square(self):
        test_get_convex_hull_area_square()

    def test_convex_hull_pruning_removes_outliers(self):
        test_convex_hull_pruning_removes_outliers()

    def test_select_optimal_triplet_rejects_collinear(self):
        test_select_optimal_triplet_rejects_collinear()

    def test_refined_result_npz_roundtrip(self):
        test_refined_result_npz_roundtrip()

    def test_geometric_pruner_end_to_end(self):
        test_geometric_pruner_end_to_end()


if __name__ == "__main__":
    unittest.main()
