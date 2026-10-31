"""
Dynamic Time Warping (DTW) Distance Metric.

Measures the morphological distance between two temporal trajectories (1D or multi-dimensional 3D)
allowing for non-linear temporal stretching, phase misalignment, and speed variations.
Supports Sakoe-Chiba band window constraints and z-score normalization.
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class DTWResult:
    """
    Container for Dynamic Time Warping alignment analysis.

    Attributes:
        total_distance: Total accumulated Euclidean distance along optimal warping path.
        normalized_distance: Total distance divided by warping path length.
                             Represents the average spatial deviation per aligned time step.
        path_length: Number of aligned pairs in the optimal warping path.
    """
    total_distance: float
    normalized_distance: float
    path_length: int


def calculate_dtw(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list],
    window: Optional[int] = None,
    normalize: bool = False
) -> DTWResult:
    """
    Computes Dynamic Time Warping (DTW) distance between two trajectories.

    Args:
        y_pred: Predicted trajectory array of shape (N,) or (N, D).
        y_true: Ground Truth trajectory array of shape (M,) or (M, D).
        window: Optional Sakoe-Chiba band window size (int >= 0).
                Restricts alignment to |i - j| <= window.
        normalize: If True, z-score normalizes each feature dimension before alignment
                   to compare trajectory shape independently of scale.

    Returns:
        DTWResult dataclass with total_distance, normalized_distance, and path_length.

    Raises:
        ValueError: If input dimensions are invalid, empty, or contain non-finite values.
    """
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    arr_true = np.asarray(y_true, dtype=np.float64)

    # Standardize to 2D shape (T, D)
    if arr_pred.ndim == 1:
        arr_pred = arr_pred[:, None]
    if arr_true.ndim == 1:
        arr_true = arr_true[:, None]

    if arr_pred.ndim != 2 or arr_true.ndim != 2:
        raise ValueError(
            f"Trajectories must be 1D or 2D (T, D). Got shapes {arr_pred.shape} and {arr_true.shape}."
        )

    n_pred, d_pred = arr_pred.shape
    n_true, d_true = arr_true.shape

    if d_pred != d_true:
        raise ValueError(
            f"Coordinate dimension mismatch: y_pred has {d_pred} features, "
            f"y_true has {d_true} features."
        )

    if n_pred < 1 or n_true < 1:
        raise ValueError("Trajectories must not be empty.")

    if not np.all(np.isfinite(arr_pred)) or not np.all(np.isfinite(arr_true)):
        raise ValueError("Trajectories contain NaN or Inf values.")

    # Optional Z-score normalization along time
    if normalize:
        arr_pred = _normalize_trajectory(arr_pred)
        arr_true = _normalize_trajectory(arr_true)

    # Pairwise point-to-point Euclidean distances
    dist_matrix = cdist(arr_pred, arr_true, metric="euclidean")

    # Dynamic Programming cost accumulation
    dp = np.full((n_pred + 1, n_true + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    if window is not None:
        if window < 0:
            raise ValueError(f"Window size must be non-negative, got {window}.")
        window_size = int(window)
        # Ensure window is large enough to cover the length difference |n_pred - n_true|
        window_size = max(window_size, abs(n_pred - n_true))
    else:
        window_size = max(n_pred, n_true)

    for i in range(1, n_pred + 1):
        j_start = max(1, i - window_size)
        j_end = min(n_true + 1, i + window_size + 1)
        for j in range(j_start, j_end):
            cost = dist_matrix[i - 1, j - 1]
            min_prev = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = cost + min_prev

    total_cost = float(dp[n_pred, n_true])

    if np.isinf(total_cost):
        raise ValueError("DTW alignment failed to reach end of sequences (window may be too narrow).")

    # Backtrack optimal warping path length
    path_len = _backtrack_path_length(dp, n_pred, n_true)
    norm_dist = float(total_cost / max(path_len, 1))

    return DTWResult(
        total_distance=total_cost,
        normalized_distance=norm_dist,
        path_length=path_len
    )


def _normalize_trajectory(arr: np.ndarray) -> np.ndarray:
    """Z-score normalizes each feature dimension across time."""
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    # Avoid zero division
    std[std < 1e-12] = 1.0
    return (arr - mean) / std


def _backtrack_path_length(dp: np.ndarray, n: int, m: int) -> int:
    """Calculates the number of steps in the optimal warping path."""
    i, j = n, m
    length = 0
    while i > 0 and j > 0:
        length += 1
        # Step back to the minimum neighbor
        prev_diag = dp[i - 1, j - 1]
        prev_up = dp[i - 1, j]
        prev_left = dp[i, j - 1]

        m_cost = min(prev_diag, prev_up, prev_left)
        if m_cost == prev_diag:
            i -= 1
            j -= 1
        elif m_cost == prev_up:
            i -= 1
        else:
            j -= 1

    # Add remaining steps if edge reached
    length += i + j
    return length
