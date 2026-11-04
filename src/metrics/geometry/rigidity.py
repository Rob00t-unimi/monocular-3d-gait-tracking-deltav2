"""
3D Rigidity and Geometric Stability Metric.

Quantifies adherence to the rigid body hypothesis for tracked 3D points on the thigh.
Calculates temporal variance, standard deviation, and relative percentage deformation
of pairwise inter-point distances across the gait cycle.
"""

from dataclasses import dataclass
from typing import Union, Dict, Tuple
import itertools
import numpy as np


@dataclass(frozen=True)
class RigidityResult:
    """
    Container for 3D rigid body deformation analysis.

    Attributes:
        sigma_rigidity: Mean standard deviation of pairwise Euclidean distances over time.
                        Ideal rigid body: sigma_rigidity -> 0.
        relative_deformation_percent: Mean coefficient of variation (std / mean_dist * 100).
                                      Scale-invariant indicator of structural deformation.
        max_pairwise_deviation: Maximum absolute deviation from the reference (initial) distance.
        pairwise_stds: Dictionary mapping (point_i, point_j) to its temporal distance std.
        pairwise_initial_distances: Dictionary mapping (point_i, point_j) to distance at frame 0.
    """
    sigma_rigidity: float
    relative_deformation_percent: float
    max_pairwise_deviation: float
    pairwise_stds: Dict[Tuple[int, int], float]
    pairwise_initial_distances: Dict[Tuple[int, int], float]


def calculate_rigidity_index(
    track3d: Union[np.ndarray, list]
) -> RigidityResult:
    """
    Computes rigidity and geometric deformation metrics for tracked 3D points.

    Args:
        track3d: Array of shape (T, N, 3) or (T, N, 2) where T is time frames and N is number of points.

    Returns:
        RigidityResult dataclass.

    Raises:
        ValueError: If input dimensions are invalid (requires T >= 2, N >= 2, 2D or 3D points)
                    or if data contains non-finite values.
    """
    arr = np.asarray(track3d, dtype=np.float64)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array of shape (T, N, D), got shape {arr.shape}.")

    n_frames, n_pts, n_coords = arr.shape

    if n_frames < 2:
        raise ValueError(f"Requires at least 2 temporal frames, got {n_frames}.")

    if n_pts < 2:
        raise ValueError(f"Requires at least 2 tracked points to form distances, got {n_pts}.")

    if n_coords not in (2, 3):
        raise ValueError(f"Point coordinate dimension must be 2 or 3, got {n_coords}.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Trajectory contains NaN or Inf values.")

    if n_pts > 15:
        # Vectorized path for larger point sets (e.g. point cloud)
        iu = np.triu_indices(n_pts, k=1)
        diff = arr[:, iu[0], :] - arr[:, iu[1], :]  # (T, num_pairs, n_coords)
        dists = np.linalg.norm(diff, axis=-1)       # (T, num_pairs)

        stds_arr = np.std(dists, axis=0)
        means_arr = np.mean(dists, axis=0)
        d_inits = dists[0, :]

        rel_def_arr = np.where(means_arr > 1e-12, (stds_arr / means_arr) * 100.0, 0.0)
        max_dev = float(np.max(np.abs(dists - d_inits[None, :])))
        mean_sigma = float(np.mean(stds_arr))
        mean_rel_def = float(np.mean(rel_def_arr))

        pair_tuples = list(zip(iu[0].tolist(), iu[1].tolist()))
        stds = {(int(i), int(j)): float(s) for (i, j), s in zip(pair_tuples, stds_arr)}
        init_dists = {(int(i), int(j)): float(d) for (i, j), d in zip(pair_tuples, d_inits)}
    else:
        pairs = list(itertools.combinations(range(n_pts), 2))
        stds = {}
        init_dists = {}
        rel_deformations = []
        max_dev = 0.0

        for i, j in pairs:
            # Pairwise distance vector over all time frames: shape (T,)
            diff = arr[:, i, :] - arr[:, j, :]
            dists = np.linalg.norm(diff, axis=-1)

            d_std = float(np.std(dists))
            d_mean = float(np.mean(dists))
            d_init = float(dists[0])

            stds[(i, j)] = d_std
            init_dists[(i, j)] = d_init

            # Relative deformation (std / mean)
            if d_mean > 1e-12:
                rel_deformations.append((d_std / d_mean) * 100.0)
            else:
                rel_deformations.append(0.0)

            # Deviation from frame 0
            pair_max_dev = float(np.max(np.abs(dists - d_init)))
            if pair_max_dev > max_dev:
                max_dev = pair_max_dev

        mean_sigma = float(np.mean(list(stds.values())))
        mean_rel_def = float(np.mean(rel_deformations))

    return RigidityResult(
        sigma_rigidity=mean_sigma,
        relative_deformation_percent=mean_rel_def,
        max_pairwise_deviation=max_dev,
        pairwise_stds=stds,
        pairwise_initial_distances=init_dists
    )
