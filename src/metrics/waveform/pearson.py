"""
Pearson Correlation Coefficient Metric.

Measures linear correlation and phase synchrony between predicted kinematic signals
(e.g., vision-derived Pitch or AccX) and Ground Truth signals (IMU).
Invariance to scaling factor and constant offset makes it ideal for gait waveform comparison.
"""

from typing import Union
import numpy as np


def calculate_pearson(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list]
) -> float:
    """
    Computes Pearson Correlation Coefficient (r) between two 1D signals.

    Args:
        y_pred: Predicted signal array (e.g. vision tracking).
        y_true: Ground Truth signal array (e.g. IMU).

    Returns:
        r: Pearson correlation coefficient in [-1.0, 1.0].
           Returns 0.0 if either signal has zero variance.

    Raises:
        ValueError: If signal lengths differ, are empty, or contain non-finite values.
    """
    arr_pred = np.asarray(y_pred, dtype=np.float64).squeeze()
    arr_true = np.asarray(y_true, dtype=np.float64).squeeze()

    if arr_pred.ndim != 1 or arr_true.ndim != 1:
        raise ValueError(f"Signals must be 1D. Got shapes {arr_pred.shape} and {arr_true.shape}.")

    if len(arr_pred) != len(arr_true):
        raise ValueError(
            f"Signal length mismatch: y_pred has length {len(arr_pred)}, "
            f"y_true has length {len(arr_true)}."
        )

    if len(arr_pred) < 2:
        raise ValueError("Signals must contain at least 2 samples.")

    if not np.all(np.isfinite(arr_pred)) or not np.all(np.isfinite(arr_true)):
        raise ValueError("Signals contain NaN or Inf values.")

    # Centered signals
    diff_pred = arr_pred - np.mean(arr_pred)
    diff_true = arr_true - np.mean(arr_true)

    std_pred = np.linalg.norm(diff_pred)
    std_true = np.linalg.norm(diff_true)

    # If any signal is constant (zero variance), correlation is undefined -> 0.0
    if std_pred < 1e-12 or std_true < 1e-12:
        return 0.0

    r = float(np.dot(diff_pred, diff_true) / (std_pred * std_true))
    return float(np.clip(r, -1.0, 1.0))
