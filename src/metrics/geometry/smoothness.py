"""
3D Trajectory Smoothness and Jerk Metric.

Quantifies high-frequency jitter, depth popping, and kinematic plausibility of 3D point tracking trajectories.
Computes the third time derivative (jerk) of the 3D centroid of tracked points.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class SmoothnessResult:
    """
    Container for 3D trajectory smoothness metrics.

    Attributes:
        rms_jerk: Root Mean Square of the 3D jerk vector norm. Lower = smoother.
        peak_jerk: Maximum jerk vector norm observed.
        mean_jerk: Mean jerk vector norm across time.
    """
    rms_jerk: float
    peak_jerk: float
    mean_jerk: float


def calculate_3d_jerk(
    track3d: Union[np.ndarray, list],
    fps: float = 30.0
) -> SmoothnessResult:
    """
    Computes 3D jerk (time derivative of acceleration) of the centroid of tracked 3D points.

    Args:
        track3d: Array of shape (T, N, 3) or (T, 3).
        fps: Sampling rate in Hz (default 30.0).

    Returns:
        SmoothnessResult dataclass.

    Raises:
        ValueError: If input dimensions are invalid, non-finite, or length < 4.
    """
    arr = np.asarray(track3d, dtype=np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError("Trajectory contains NaN or Inf values.")

    if arr.ndim == 3:
        centroid = np.mean(arr, axis=1)  # (T, 3)
    elif arr.ndim == 2:
        centroid = arr                   # (T, 3)
    else:
        raise ValueError(f"Expected (T, N, 3) or (T, 3), got shape {arr.shape}.")

    n_frames, n_coords = centroid.shape
    if n_coords != 3:
        raise ValueError(f"Expected 3 spatial coordinates, got {n_coords}.")

    if n_frames < 4:
        raise ValueError(f"Requires at least 4 temporal frames to compute jerk, got {n_frames}.")

    dt = 1.0 / max(float(fps), 1e-6)
    vel = np.gradient(centroid, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)  # (T, 3)

    jerk_norms = np.linalg.norm(jerk, axis=-1)  # (T,)
    rms = float(np.sqrt(np.mean(jerk_norms ** 2)))
    peak = float(np.max(jerk_norms))
    mean = float(np.mean(jerk_norms))

    return SmoothnessResult(
        rms_jerk=rms,
        peak_jerk=peak,
        mean_jerk=mean
    )
