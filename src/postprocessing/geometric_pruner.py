# -*- coding: utf-8 -*-
"""
Geometric Pruning and Optimal Triplet Selection Pipeline for 3D Gait Tracking.

This module refines raw point tracking tensors by enforcing physical and biomechanical
consistency constraints:
1. Low-pass zero-phase filtering aligned to gait frequency cutoff.
2. Isolation of thigh points from contextual body anchor points.
3. Convex Hull Area-based outlier rejection on the 2D planar projection with 100% temporal persistence.
4. Optimal Triplet Selection (v4.1) guaranteeing non-collinear 3D area (>= 5% of reference)
   and maximizing area coverage and temporal stability.
"""

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.spatial import ConvexHull
try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger("geometric_pruner")


# --- DATA STRUCTURES ---

@dataclass
class RefinedTrackingResult:
    """
    Container for geometrically pruned and refined 3D tracking outputs.

    Attributes:
        triplet_track3d: (T, 3, 3) Trajectory of the optimal rigid 3D triangle (P1, P2, P3).
        triplet_track2d: (T, 3, 2) Corresponding pixel trajectory of the optimal triplet.
        triplet_indices: (3,) Original indices of the selected optimal triplet.
        survived_track3d: (T, M, 3) Trajectory of all M points that survived geometric pruning.
        survived_track2d: (T, M, 2) Pixel trajectory of all M survived points.
        survived_indices: (M,) Original indices of all survived points.
        video_name: Name of the video file.
        ref_frame_idx: Absolute reference frame index.
        start_idx: Absolute start frame index in the video window.
        fps: Frame rate (default: 30.0).
        mode: Description string of the refinement mode.
        metadata: Optional dictionary with supplementary information.
    """
    triplet_track3d: np.ndarray
    triplet_track2d: np.ndarray
    triplet_indices: np.ndarray
    survived_track3d: np.ndarray
    survived_track2d: np.ndarray
    survived_indices: np.ndarray
    video_name: str
    ref_frame_idx: int
    start_idx: int
    fps: float = 30.0
    mode: str = "geometric_pruning_v4.1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_npz(self, output_path: Path) -> Path:
        """
        Saves refined tracking results to NPZ format.
        Maintains backward compatibility with analysis dashboards by saving:
        - 'track3d' and 'track2d' (optimal triplet)
        - 'survived_track3d' and 'survived_track2d' (entire persistent cluster)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            track3d=self.triplet_track3d,
            track2d=self.triplet_track2d,
            triplet_track3d=self.triplet_track3d,
            triplet_track2d=self.triplet_track2d,
            indices=self.triplet_indices,
            triplet_indices=self.triplet_indices,
            survived_track3d=self.survived_track3d,
            survived_track2d=self.survived_track2d,
            survived_indices=self.survived_indices,
            video_name=self.video_name,
            ref_frame_idx=self.ref_frame_idx,
            start_idx=self.start_idx,
            fps=self.fps,
            mode=self.mode
        )
        return output_path

    @classmethod
    def load_npz(cls, input_path: Path) -> "RefinedTrackingResult":
        """Loads refined tracking results from an NPZ file."""
        data = np.load(input_path, allow_pickle=True)
        triplet_3d = data["track3d"] if "track3d" in data else data["triplet_track3d"]
        triplet_2d = data["track2d"] if "track2d" in data else data["triplet_track2d"]
        indices = data["indices"] if "indices" in data else data["triplet_indices"]
        survived_3d = data.get("survived_track3d", triplet_3d)
        survived_2d = data.get("survived_track2d", triplet_2d)
        survived_idx = data.get("survived_indices", indices)

        return cls(
            triplet_track3d=triplet_3d,
            triplet_track2d=triplet_2d,
            triplet_indices=indices,
            survived_track3d=survived_3d,
            survived_track2d=survived_2d,
            survived_indices=survived_idx,
            video_name=str(data.get("video_name", "Unknown")),
            ref_frame_idx=int(data.get("ref_frame_idx", 0)),
            start_idx=int(data.get("start_idx", 0)),
            fps=float(data.get("fps", 30.0)),
            mode=str(data.get("mode", "geometric_pruning_v4.1"))
        )


# --- GEOMETRIC COMPUTATIONS ---

def calc_area_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Computes vectorized 2D triangle area using cross-product determinant formula."""
    return 0.5 * np.abs(
        p1[..., 0] * (p2[..., 1] - p3[..., 1]) +
        p2[..., 0] * (p3[..., 1] - p1[..., 1]) +
        p3[..., 0] * (p1[..., 1] - p2[..., 1])
    )


def calc_area_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Computes vectorized 3D triangle surface area using vector cross product norm."""
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross, axis=-1)


def get_convex_hull_area_2d(pts_2d: np.ndarray) -> float:
    """
    Computes the area of the 2D Convex Hull of a point set.
    In 2D, scipy.spatial.ConvexHull.volume represents the planar area.
    """
    if len(pts_2d) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts_2d).volume)
    except Exception:
        return 0.0


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float = 30.0, order: int = 4) -> np.ndarray:
    """Applies a zero-phase forward-backward Butterworth lowpass filter along axis 0."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data, axis=0)


# --- PRUNING ALGORITHMS ---

def apply_convex_hull_pruning(
    points_3d: np.ndarray,
    ref_idx_rel: int,
    max_area_ratio: float = 5.0
) -> np.ndarray:
    """
    Custom Geometric Pruning based on 2D Convex Hull area expansion and 100% persistence.

    Methodology:
    1. Computes the initial 2D Convex Hull area (ref_area) of the cluster at the reference frame (X-Y plane).
    2. Defines max_allowed_area = ref_area * max_area_ratio.
    3. For every frame t: if the hull area exceeds max_allowed_area, iteratively removes the point
       furthest from the cluster centroid until area <= threshold or <= 3 points remain.
    4. 100% Persistence Filter: only retains points that survive across all frames.

    Args:
        points_3d: (T, N, 3) 3D coordinate trajectories.
        ref_idx_rel: Local index of the reference frame in the window.
        max_area_ratio: Maximum allowable area expansion multiplier over reference area.

    Returns:
        1D array of indices corresponding to survived points.
    """
    T, N, _ = points_3d.shape
    if N < 3:
        return np.arange(N)

    # Reference area on X-Y plane
    pts_ref = points_3d[ref_idx_rel, :, :2]
    ref_area = get_convex_hull_area_2d(pts_ref)

    if ref_area <= 1e-9:
        logger.warning("Reference frame convex hull area is near zero; skipping area pruning.")
        return np.arange(N)

    max_allowed_area = ref_area * max_area_ratio
    global_valid_mask = np.ones(N, dtype=bool)

    for t in range(T):
        pts_t = points_3d[t, :, :2]
        current_indices = np.arange(N)

        while True:
            current_pts = pts_t[current_indices]
            current_area = get_convex_hull_area_2d(current_pts)

            if current_area <= max_allowed_area or len(current_indices) <= 3:
                break

            # Remove point furthest from cluster centroid
            centroid = np.mean(current_pts, axis=0)
            dists = np.linalg.norm(current_pts - centroid, axis=1)
            idx_to_remove_local = np.argmax(dists)
            current_indices = np.delete(current_indices, idx_to_remove_local)

        frame_mask = np.zeros(N, dtype=bool)
        frame_mask[current_indices] = True
        global_valid_mask &= frame_mask

    survived = np.where(global_valid_mask)[0]
    return survived


def apply_adaptive_pca_dbscan(
    points_3d: np.ndarray,
    n_components: int = 5,
    initial_eps: float = 10.0
) -> np.ndarray:
    """
    Kinematic velocity clustering via PCA and adaptive DBSCAN to isolate coherent leg points.

    Args:
        points_3d: (T, N, 3) 3D coordinate trajectories.
        n_components: Number of PCA components for velocity features.
        initial_eps: Initial DBSCAN epsilon.

    Returns:
        1D array of indices in the dominant coherent cluster.
    """
    T, N, _ = points_3d.shape
    if not HAS_SKLEARN or N < 4 or T < 2:
        return np.arange(N)

    velocities = np.diff(points_3d, axis=0)
    vel_features = velocities.transpose(1, 0, 2).reshape(N, -1)

    scaler = StandardScaler()
    vel_scaled = scaler.fit_transform(vel_features)

    n_comp = min(n_components, vel_scaled.shape[1], N)
    pca = PCA(n_components=n_comp)
    vel_pca = pca.fit_transform(vel_scaled)

    eps = initial_eps
    for _ in range(15):
        clustering = DBSCAN(eps=eps, min_samples=max(3, N // 50)).fit(vel_pca)
        labels = clustering.labels_
        u, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(u) > 0:
            main_label = u[np.argmax(counts)]
            indices = np.where(labels == main_label)[0]
            if len(indices) >= max(4, int(0.5 * N)):
                return indices
        eps *= 1.3

    return np.arange(N)


def select_optimal_triplet(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    candidate_indices: np.ndarray,
    ref_idx_rel: int,
    original_indices: Tuple[int, int, int] = (0, 1, 2),
    min_3d_ratio: float = 0.05,
    max_search_candidates: int = 80,
    top_combos_eval: int = 3000,
    z_penalty_weight: float = 0.0
) -> Tuple[int, int, int]:
    """
    Optimal Triplet Selection System (v4.1) with Vectorized Pre-filtering.

    Evaluates candidate triplets of survived points and selects the one that maximizes
    area coverage and physical stability while strictly enforcing non-collinear 3D area.

    Score formulation:
        Score = ln(1 + Area_2D_ref) * ln(1 + MinArea_2D) * Stability_2D * Stability_3D
        where Stability = MeanArea / (StdArea + 1e-6)

    Constraint:
        Min(Area_3D(t)) >= min_3d_ratio * Area_3D_ref(original_query)
        Tested across adaptive degradation thresholds [min_3d_ratio, 0.02, 0.01, 0.005, 0.001].
    """
    if len(candidate_indices) < 3:
        raise ValueError(f"Fewer than 3 candidate points available for triplet selection ({len(candidate_indices)})")

    # 1. Benchmark 3D reference area for physical non-collinearity
    pts_orig_ref_3d = points_3d[ref_idx_rel, list(original_indices)]
    area_orig_ref_3d = float(calc_area_3d(pts_orig_ref_3d[0], pts_orig_ref_3d[1], pts_orig_ref_3d[2]))

    # 2. Candidate pool subsampling if pool is very large
    cand_indices = candidate_indices
    if len(cand_indices) > max_search_candidates:
        top = [i for i in original_indices if i in cand_indices]
        rest = [i for i in cand_indices if i not in top]
        step = max(1, len(rest) // (max_search_candidates - len(top)))
        cand_indices = np.array(top + rest[::step])

    combos_arr = np.array(list(combinations(range(len(cand_indices)), 3)))

    # 3. Vectorized pre-filtering based on 3D area regularized against monocular Z spikes
    pts_ref_3d = points_3d[ref_idx_rel, cand_indices]
    v1_ref = pts_ref_3d[combos_arr[:, 1]] - pts_ref_3d[combos_arr[:, 0]]
    v2_ref = pts_ref_3d[combos_arr[:, 2]] - pts_ref_3d[combos_arr[:, 0]]
    cross_ref = np.cross(v1_ref, v2_ref)
    raw_areas_3d = 0.5 * np.linalg.norm(cross_ref, axis=1)
    if z_penalty_weight > 0.0:
        xy_span = np.maximum(np.linalg.norm(v1_ref[:, :2], axis=1), np.linalg.norm(v2_ref[:, :2], axis=1)) + 1e-6
        dz_span = np.maximum(np.abs(v1_ref[:, 2]), np.abs(v2_ref[:, 2]))
        z_Aspect_penalty = 1.0 / (1.0 + z_penalty_weight * (dz_span / xy_span) ** 2)
        ref_areas_3d = raw_areas_3d * z_Aspect_penalty
    else:
        ref_areas_3d = raw_areas_3d

    # Sort descending by reference area and take top K combinations
    eval_k = min(top_combos_eval, len(combos_arr))
    top_combo_idx = np.argsort(ref_areas_3d)[-eval_k:][::-1]
    combos_to_eval = combos_arr[top_combo_idx]

    pts_seq_2d = points_2d[:, cand_indices, :2]
    pts_seq_3d = points_3d[:, cand_indices, :3]

    # 4. Multi-level search with adaptive floor thresholds
    thresholds = [min_3d_ratio, 0.02, 0.01, 0.005, 0.001]
    # Remove duplicates and sort descending
    thresholds = sorted(list({t for t in thresholds if t <= min_3d_ratio}), reverse=True)

    for thresh in thresholds:
        abs_min_3d_floor = thresh * area_orig_ref_3d
        best_score = -1.0
        best_triplet = None

        for c in combos_to_eval:
            tp1_3d = pts_seq_3d[:, c[0]]
            tp2_3d = pts_seq_3d[:, c[1]]
            tp3_3d = pts_seq_3d[:, c[2]]

            v1_t = tp2_3d - tp1_3d
            v2_t = tp3_3d - tp1_3d
            areas_3d = 0.5 * np.linalg.norm(np.cross(v1_t, v2_t), axis=-1)

            if np.min(areas_3d) < abs_min_3d_floor:
                continue

            tp1_2d = pts_seq_2d[:, c[0]]
            tp2_2d = pts_seq_2d[:, c[1]]
            tp3_2d = pts_seq_2d[:, c[2]]
            areas_2d = calc_area_2d(tp1_2d, tp2_2d, tp3_2d)

            area_ref_2d = float(areas_2d[ref_idx_rel])
            min_area_2d = float(np.min(areas_2d))
            mean_area_2d = float(np.mean(areas_2d))

            stability_2d = mean_area_2d / (float(np.std(areas_2d)) + 1e-6)
            stability_3d = float(np.mean(areas_3d)) / (float(np.std(areas_3d)) + 1e-6)

            score = np.log1p(area_ref_2d) * np.log1p(min_area_2d) * stability_2d * stability_3d

            if score > best_score:
                best_score = score
                best_triplet = (int(cand_indices[c[0]]), int(cand_indices[c[1]]), int(cand_indices[c[2]]))

        if best_triplet is not None:
            return best_triplet

    # 5. Fallback to candidate with maximum mean 3D area
    logger.warning("No triplet satisfied adaptive 3D non-collinearity floors. Selecting highest mean area triplet.")
    max_m_area = -1.0
    fallback_triplet = None
    for c in combos_to_eval[:100]:
        tp1_3d = pts_seq_3d[:, c[0]]
        tp2_3d = pts_seq_3d[:, c[1]]
        tp3_3d = pts_seq_3d[:, c[2]]
        m_a = float(np.mean(0.5 * np.linalg.norm(np.cross(tp2_3d - tp1_3d, tp3_3d - tp1_3d), axis=-1)))
        if m_a > max_m_area:
            max_m_area = m_a
            fallback_triplet = (int(cand_indices[c[0]]), int(cand_indices[c[1]]), int(cand_indices[c[2]]))

    if fallback_triplet is not None:
        return fallback_triplet

    return (int(cand_indices[0]), int(cand_indices[1]), int(cand_indices[2]))


# --- MAIN PIPELINE CLASS ---

class GeometricPruner:
    """
    High-level orchestrator for post-processing and geometric pruning of raw tracking results.
    """

    def __init__(
        self,
        cutoff_freq: Optional[float] = None,
        max_area_ratio: float = 5.0,
        min_3d_ratio: float = 0.05,
        num_thigh_points: int = 203,
        use_pca_dbscan_refinement: bool = False,
        z_penalty_weight: float = 0.0
    ):
        self.cutoff_freq = cutoff_freq if cutoff_freq is not None else self._load_default_cutoff()
        self.max_area_ratio = max_area_ratio
        self.min_3d_ratio = min_3d_ratio
        self.num_thigh_points = num_thigh_points
        self.use_pca_dbscan_refinement = use_pca_dbscan_refinement
        self.z_penalty_weight = z_penalty_weight

    @staticmethod
    def _load_default_cutoff() -> float:
        """Loads highcut frequency from cutoff_freqs.json if present, else defaults to 2.93 Hz."""
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_path = project_root / "data" / "gt_cache" / "cutoff_freqs.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    return float(data.get("highcut", 2.93))
            except Exception:
                pass
        return 2.93

    def refine(
        self,
        track2d: np.ndarray,
        track3d_world: np.ndarray,
        ref_frame_idx: int,
        start_idx: int,
        video_name: str = "Unknown",
        fps: float = 30.0
    ) -> RefinedTrackingResult:
        """
        Executes full geometric refinement on raw tracking tensors.

        Args:
            track2d: (T, N, 2) raw pixel tracks.
            track3d_world: (T, N, 3) raw metric world tracks.
            ref_frame_idx: Absolute reference frame index.
            start_idx: Absolute window start index.
            video_name: Name of video file.
            fps: Video frame rate.

        Returns:
            RefinedTrackingResult containing optimal triplet and persistent cluster.
        """
        ref_idx_rel = max(0, min(len(track3d_world) - 1, ref_frame_idx - start_idx))
        t2d = track2d.copy()
        t3d = track3d_world.copy()

        # 1. Zero-phase lowpass filtering on all points
        total_pts = t3d.shape[1]
        for i in range(total_pts):
            for d in range(2):
                t2d[:, i, d] = butter_lowpass_filter(t2d[:, i, d], self.cutoff_freq, fs=fps)
            for d in range(3):
                t3d[:, i, d] = butter_lowpass_filter(t3d[:, i, d], self.cutoff_freq, fs=fps)

        # 2. Isolate thigh points (first num_thigh_points)
        num_thigh = min(self.num_thigh_points, total_pts)
        thigh_indices = np.arange(num_thigh)
        thigh_t3d = t3d[:, thigh_indices, :]
        thigh_t2d = t2d[:, thigh_indices, :]

        # 3. Geometric Convex Hull Area Pruning (100% persistence)
        survived_local = apply_convex_hull_pruning(
            points_3d=thigh_t3d,
            ref_idx_rel=ref_idx_rel,
            max_area_ratio=self.max_area_ratio
        )

        # 4. Optional PCA-DBSCAN velocity coherence filter
        if self.use_pca_dbscan_refinement and len(survived_local) >= 6:
            pca_survived = apply_adaptive_pca_dbscan(thigh_t3d[:, survived_local, :])
            if len(pca_survived) >= 3:
                survived_local = survived_local[pca_survived]

        if len(survived_local) < 3:
            logger.warning(f"Fewer than 3 points survived pruning ({len(survived_local)}); falling back to initial query triplet.")
            survived_local = np.array([0, 1, 2])

        survived_global = thigh_indices[survived_local].astype(int)

        # 5. Optimal Triplet Selection (v4.1)
        best_triplet = select_optimal_triplet(
            points_2d=t2d,
            points_3d=t3d,
            candidate_indices=survived_global,
            ref_idx_rel=ref_idx_rel,
            original_indices=(0, 1, 2),
            min_3d_ratio=self.min_3d_ratio,
            z_penalty_weight=self.z_penalty_weight
        )

        triplet_idx_arr = np.array(best_triplet, dtype=np.int32)

        return RefinedTrackingResult(
            triplet_track3d=t3d[:, triplet_idx_arr, :],
            triplet_track2d=t2d[:, triplet_idx_arr, :],
            triplet_indices=triplet_idx_arr,
            survived_track3d=t3d[:, survived_global, :],
            survived_track2d=t2d[:, survived_global, :],
            survived_indices=survived_global,
            video_name=video_name,
            ref_frame_idx=ref_frame_idx,
            start_idx=start_idx,
            fps=fps,
            mode="deltav2_geometric_pruning_v4.1"
        )
