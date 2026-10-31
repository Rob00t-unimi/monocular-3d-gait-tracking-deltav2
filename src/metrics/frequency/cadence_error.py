"""
Cadence and Dominant Frequency Error Metric.

Extracts the dominant gait frequency (in Hz) and cadence (in Strides/Steps Per Minute, SPM/BPM)
from kinematic signals (Pitch or AccX) using high-resolution windowed spectral analysis.
Computes the relative cadence error and diagnoses Identity Swap (double-frequency artifact).
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class CadenceResult:
    """
    Container for gait cadence and frequency analysis.

    Attributes:
        freq_pred: Dominant frequency of predicted signal (Hz).
        freq_true: Dominant frequency of Ground Truth signal (Hz).
        cadence_pred_bpm: Predicted cadence in cycles per minute (Hz * 60).
        cadence_true_bpm: Ground Truth cadence in cycles per minute (Hz * 60).
        relative_error_percent: Percentage relative error |freq_pred - freq_true| / freq_true * 100.
        frequency_ratio: Ratio freq_pred / freq_true.
        has_identity_swap: Boolean flag indicating if frequency_ratio >= swap_threshold
                            (typically ~2.0 when left/right leg identity is confused).
    """
    freq_pred: float
    freq_true: float
    cadence_pred_bpm: float
    cadence_true_bpm: float
    relative_error_percent: float
    frequency_ratio: float
    has_identity_swap: bool


def extract_dominant_frequency(
    y: Union[np.ndarray, list],
    fs: float = 30.0,
    freq_range: Tuple[float, float] = (0.4, 4.0),
    n_fft: int = 4096
) -> float:
    """
    Extracts the dominant gait frequency from a 1D signal using high-resolution zero-padded FFT.

    Args:
        y: Signal array (e.g. Pitch or AccX).
        fs: Sampling rate in Hz (> 0).
        freq_range: Tuple (f_min, f_max) restricting the search to physiological gait frequencies.
        n_fft: Number of FFT points for frequency interpolation.

    Returns:
        f_dom: Dominant frequency in Hz.

    Raises:
        ValueError: If signal is empty, not 1D, non-finite, or fs <= 0.
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")

    arr = np.asarray(y, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {arr.shape}.")
    if len(arr) < 4:
        raise ValueError("Signal must contain at least 4 samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains NaN or Inf values.")

    f_min, f_max = freq_range
    if f_min < 0 or f_max <= f_min:
        raise ValueError(f"Invalid frequency range: ({f_min}, {f_max}).")

    # Detrend and windowing to suppress spectral leakage
    n_samples = len(arr)
    centered = arr - np.mean(arr)

    # If signal has virtually zero variance, fallback to f_min
    if np.linalg.norm(centered) < 1e-12:
        return float(f_min)

    windowed = centered * np.hanning(n_samples)
    pad_len = max(n_fft, n_samples)

    fft_vals = np.abs(np.fft.rfft(windowed, n=pad_len))
    freqs = np.fft.rfftfreq(pad_len, d=1.0 / fs)

    # Restrict to search band
    in_range = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(in_range):
        return float(f_min)

    best_idx = np.argmax(fft_vals[in_range])
    dom_freq = freqs[in_range][best_idx]
    return float(dom_freq)


def calculate_cadence_error(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list],
    fs_pred: float = 30.0,
    fs_true: Optional[float] = None,
    freq_range: Tuple[float, float] = (0.4, 4.0),
    swap_ratio_threshold: float = 1.6
) -> CadenceResult:
    """
    Computes cadence discrepancy and checks for leg identity swap.

    Args:
        y_pred: Predicted signal array (e.g. vision tracking Pitch/AccX).
        y_true: Ground Truth signal array (e.g. IMU Pitch/AccX).
        fs_pred: Sampling rate of predicted signal in Hz (default 30.0).
        fs_true: Sampling rate of GT signal in Hz (defaults to fs_pred).
        freq_range: Tuple (f_min, f_max) in Hz.
        swap_ratio_threshold: Ratio threshold above which an Identity Swap is flagged (default 1.6).

    Returns:
        CadenceResult dataclass.
    """
    fs_gt = fs_true if fs_true is not None else fs_pred

    f_pred = extract_dominant_frequency(y_pred, fs=fs_pred, freq_range=freq_range)
    f_true = extract_dominant_frequency(y_true, fs=fs_gt, freq_range=freq_range)

    cad_pred = f_pred * 60.0
    cad_true = f_true * 60.0

    if f_true < 1e-6:
        rel_error = 0.0 if abs(f_pred - f_true) < 1e-6 else float("inf")
        ratio = 1.0 if abs(f_pred - f_true) < 1e-6 else float("inf")
    else:
        rel_error = abs(f_pred - f_true) / f_true * 100.0
        ratio = f_pred / f_true

    is_swap = bool(ratio >= swap_ratio_threshold)

    return CadenceResult(
        freq_pred=float(f_pred),
        freq_true=float(f_true),
        cadence_pred_bpm=float(cad_pred),
        cadence_true_bpm=float(cad_true),
        relative_error_percent=float(rel_error),
        frequency_ratio=float(ratio),
        has_identity_swap=is_swap
    )
