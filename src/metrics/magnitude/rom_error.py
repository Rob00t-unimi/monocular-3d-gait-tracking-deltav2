"""
Range of Motion (ROM) Error Metric.

Quantifies discrepancies in total kinematic excursion (peak-to-peak amplitude)
between predicted tracking and Ground Truth (IMU).
Crucial for identifying perspective foreshortening (underestimation of ROM)
or spurious tracking spikes (overestimation of ROM).
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class ROMResult:
    """
    Container for Range of Motion analysis.

    Attributes:
        rom_pred: Peak-to-peak excursion of predicted signal.
        rom_true: Peak-to-peak excursion of Ground Truth signal.
        absolute_error: Absolute difference |rom_pred - rom_true|.
        signed_error: Signed difference (rom_pred - rom_true).
                      Negative indicates underestimation (e.g. foreshortening).
                      Positive indicates overestimation.
        relative_error_percent: Relative error ((rom_pred - rom_true) / rom_true) * 100.
    """
    rom_pred: float
    rom_true: float
    absolute_error: float
    signed_error: float
    relative_error_percent: float


def calculate_rom(y: Union[np.ndarray, list]) -> float:
    """
    Computes peak-to-peak Range of Motion (max - min) for a 1D signal.

    Args:
        y: Signal array or sequence.

    Returns:
        rom: Range of motion value.

    Raises:
        ValueError: If signal is empty, not 1D, or contains non-finite values.
    """
    arr = np.asarray(y, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {arr.shape}.")
    if len(arr) == 0:
        raise ValueError("Signal must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains NaN or Inf values.")

    return float(np.max(arr) - np.min(arr))


def calculate_rom_error(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list]
) -> ROMResult:
    """
    Computes Range of Motion (ROM) errors between predicted and ground truth signals.

    Args:
        y_pred: Predicted signal array.
        y_true: Ground Truth signal array.

    Returns:
        ROMResult dataclass with rom_pred, rom_true, absolute_error, signed_error,
        and relative_error_percent.

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

    rom_pred = calculate_rom(arr_pred)
    rom_true = calculate_rom(arr_true)

    signed_diff = rom_pred - rom_true
    abs_diff = abs(signed_diff)

    if rom_true < 1e-12:
        rel_percent = 0.0 if abs_diff < 1e-12 else float("inf")
    else:
        rel_percent = (signed_diff / rom_true) * 100.0

    return ROMResult(
        rom_pred=rom_pred,
        rom_true=rom_true,
        absolute_error=float(abs_diff),
        signed_error=float(signed_diff),
        relative_error_percent=float(rel_percent)
    )
