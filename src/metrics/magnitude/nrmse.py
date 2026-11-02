"""
Normalized Root Mean Square Error (nRMSE) Metric.

Evaluates the amplitude error between predicted signals and Ground Truth signals.
In gait analysis, normalizing by the Range of Motion (ROM_GT = max - min)
allows meaningful error comparison across different participants, speeds, and trials.
"""

from typing import Union
import numpy as np


def calculate_rmse(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list]
) -> float:
    """
    Computes absolute Root Mean Square Error (RMSE).

    Args:
        y_pred: Predicted signal array.
        y_true: Ground Truth signal array.

    Returns:
        rmse: Absolute root mean square error (same units as input).

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

    if len(arr_pred) == 0:
        raise ValueError("Signals must not be empty.")

    if not np.all(np.isfinite(arr_pred)) or not np.all(np.isfinite(arr_true)):
        raise ValueError("Signals contain NaN or Inf values.")

    mse = np.mean((arr_pred - arr_true) ** 2)
    return float(np.sqrt(mse))


def calculate_nrmse(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list],
    normalization: str = "rom",
    as_percentage: bool = True
) -> float:
    """
    Computes Normalized Root Mean Square Error (nRMSE).

    Args:
        y_pred: Predicted signal array.
        y_true: Ground Truth signal array.
        normalization: Normalization strategy:
            - "rom": Range of Motion (max(y_true) - min(y_true)) [Default in gait analysis]
            - "std": Standard deviation of y_true
            - "mean": Mean absolute value of y_true
        as_percentage: If True, returns value multiplied by 100 (e.g. 15.2 for 15.2%).
                       If False, returns fractional value in [0, +inf).

    Returns:
        nrmse: Normalized error value (percentage or fraction).

    Raises:
        ValueError: If normalization factor is zero, or inputs are invalid.
    """
    arr_true = np.asarray(y_true, dtype=np.float64).squeeze()
    rmse = calculate_rmse(y_pred, arr_true)

    if normalization == "rom":
        norm_factor = float(np.max(arr_true) - np.min(arr_true))
    elif normalization == "std":
        norm_factor = float(np.std(arr_true))
    elif normalization == "mean":
        norm_factor = float(np.mean(np.abs(arr_true)))
    else:
        raise ValueError(
            f"Unknown normalization mode '{normalization}'. Supported: 'rom', 'std', 'mean'."
        )

    if norm_factor < 1e-12:
        raise ValueError(
            f"Normalization factor for '{normalization}' is zero or too small ({norm_factor})."
        )

    nrmse = rmse / norm_factor
    return float(nrmse * 100.0 if as_percentage else nrmse)
