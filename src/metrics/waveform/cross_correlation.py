"""
Cross-Correlation and Temporal Lag Metric.

Computes the maximum cross-correlation and optimal temporal time lag (tau)
between predicted signals and Ground Truth signals (IMU).
Quantifies latency, phase shift, and synchronization accuracy.
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class CrossCorrelationResult:
    """
    Result container for cross-correlation analysis.

    Attributes:
        peak_correlation: Maximum normalized correlation coefficient in [-1.0, 1.0].
        lag_samples: Lag in discrete samples at maximum correlation.
                     Positive lag means y_pred is delayed with respect to y_true.
        lag_seconds: Lag in seconds (lag_samples / fps).
    """
    peak_correlation: float
    lag_samples: int
    lag_seconds: float


def calculate_cross_correlation(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list],
    fps: float = 30.0,
    max_lag_seconds: Optional[float] = None
) -> CrossCorrelationResult:
    """
    Calculates normalized cross-correlation and optimal time lag.

    Args:
        y_pred: Predicted signal array (e.g. vision tracking).
        y_true: Ground Truth signal array (e.g. IMU).
        fps: Sampling frequency in Hertz (must be > 0).
        max_lag_seconds: Optional upper bound on search window in seconds (|lag| <= max_lag).

    Returns:
        CrossCorrelationResult containing peak_correlation, lag_samples, and lag_seconds.

    Raises:
        ValueError: If signal lengths mismatch, are empty, non-finite, or fps <= 0.
    """
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}.")

    arr_pred = np.asarray(y_pred, dtype=np.float64).squeeze()
    arr_true = np.asarray(y_true, dtype=np.float64).squeeze()

    if arr_pred.ndim != 1 or arr_true.ndim != 1:
        raise ValueError(f"Signals must be 1D. Got shapes {arr_pred.shape} and {arr_true.shape}.")

    if len(arr_pred) != len(arr_true):
        raise ValueError(
            f"Signal length mismatch: y_pred has length {len(arr_pred)}, "
            f"y_true has length {len(arr_true)}."
        )

    n_samples = len(arr_pred)
    if n_samples < 2:
        raise ValueError("Signals must contain at least 2 samples.")

    if not np.all(np.isfinite(arr_pred)) or not np.all(np.isfinite(arr_true)):
        raise ValueError("Signals contain NaN or Inf values.")

    # Zero-mean signals
    p_zero = arr_pred - np.mean(arr_pred)
    t_zero = arr_true - np.mean(arr_true)

    norm_p = np.linalg.norm(p_zero)
    norm_t = np.linalg.norm(t_zero)

    if norm_p < 1e-12 or norm_t < 1e-12:
        return CrossCorrelationResult(peak_correlation=0.0, lag_samples=0, lag_seconds=0.0)

    # Cross-correlation: scipy.signal.correlate(in1, in2)
    # in1 = p_zero, in2 = t_zero.
    # Lags range from -(n - 1) to +(n - 1).
    corr = signal.correlate(p_zero, t_zero, mode="full", method="auto")
    corr_normalized = corr / (norm_p * norm_t)
    lags = signal.correlation_lags(n_samples, n_samples, mode="full")

    # Filter by max_lag_seconds if specified
    if max_lag_seconds is not None:
        if max_lag_seconds <= 0:
            raise ValueError(f"max_lag_seconds must be positive, got {max_lag_seconds}.")
        max_samples = int(np.round(max_lag_seconds * fps))
        valid_mask = np.abs(lags) <= max_samples
        if not np.any(valid_mask):
            valid_mask = np.ones_like(lags, dtype=bool)
    else:
        valid_mask = np.ones_like(lags, dtype=bool)

    valid_corr = corr_normalized[valid_mask]
    valid_lags = lags[valid_mask]

    best_idx = int(np.argmax(valid_corr))
    peak_r = float(np.clip(valid_corr[best_idx], -1.0, 1.0))
    best_lag_samples = int(valid_lags[best_idx])
    best_lag_seconds = float(best_lag_samples / fps)

    return CrossCorrelationResult(
        peak_correlation=peak_r,
        lag_samples=best_lag_samples,
        lag_seconds=best_lag_seconds
    )


def is_axial_view(angle: Optional[Union[str, int, float]]) -> bool:
    """
    Determines whether a viewpoint is axial (aligned with the camera's optical axis Z),
    such as Frontal (0°) or Posterior (180°), where legs never cross or occlude in 2D.

    In axial views, limbs remain horizontally separated by the pelvis width throughout gait.
    Therefore, physical limb identity swap (jumping between legs due to occlusion) is geometrically
    impossible; any persistent antiphase is an artifact of monocular depth estimation / foreshortening.
    """
    if angle is None:
        return False
    ang_str = str(angle).strip().lower()
    return (
        ang_str in {"0", "180", "front", "rear", "frontal", "posterior"}
        or "front" in ang_str
        or "rear" in ang_str
        or "posterior" in ang_str
    )


@dataclass(frozen=True)
class IdentitySwapDetectionResult:
    """
    Result container for biomechanical identity swap detection via phase cross-correlation.

    Attributes:
        has_identity_swap: True if genuine leg identity swap occurs (requires crossover visibility: 90° or 45°).
        is_depth_inversion_artifact: True if persistent antiphase occurs in axial views (0°/180°) due to
                                     monocular depth estimation / optical axis foreshortening distortion.
        antiphase_percentage: Percentage of rolling stride windows where correlation r < antiphase_threshold.
        min_rolling_corr: Minimum correlation across rolling windows.
        mean_rolling_corr: Mean correlation across rolling windows.
        max_rolling_corr: Maximum correlation across rolling windows.
        is_axial_view: True if camera angle is axial (0° / 180° / Front / Rear).
        diagnostic: Human-readable diagnostic description of the tracking status.
    """
    has_identity_swap: bool
    antiphase_percentage: float
    min_rolling_corr: float
    mean_rolling_corr: float
    max_rolling_corr: float
    is_depth_inversion_artifact: bool = False
    is_axial_view: bool = False
    diagnostic: str = "None"


def detect_identity_swap_cross_correlation(
    y_pred: Union[np.ndarray, list],
    y_true: Union[np.ndarray, list],
    fps: float = 30.0,
    window_seconds: float = 1.2,
    step_seconds: float = 0.2,
    antiphase_threshold: float = -0.35,
    duration_threshold_percent: float = 35.0,
    angle: Optional[Union[str, int, float]] = None,
) -> IdentitySwapDetectionResult:
    """
    Detects Leg Identity Swap vs Depth Inversion Artifacts via rolling window cross-correlation.
    
    Biomechanical Foundation:
    - In lateral and oblique views (90°, 45°), legs cross and occlude each other in the 2D image plane.
      A tracker erroneously latching onto the contralateral limb produces persistent negative correlation
      (antiphase, ~180° out of phase) relative to the unilateral Ground Truth sensor -> True Identity Swap.
    - In axial views (0° Frontal, 180° Posterior), legs are permanently separated along the camera X axis
      and NEVER occlude each other. Sagittal gait motion (Pitch) is aligned with camera optical axis Z.
      Persistent negative correlation in axial views is NOT an identity swap, but a Depth Inversion /
      Foreshortening Artifact from monocular depth estimation.

    Args:
        y_pred: Predicted signal array (e.g. vision tracking Pitch).
        y_true: Ground Truth signal array (e.g. IMU Pitch).
        fps: Sampling rate in Hz (> 0).
        window_seconds: Duration of rolling window in seconds (~1 gait stride, default 1.2s).
        step_seconds: Step size between evaluation windows in seconds (default 0.2s).
        antiphase_threshold: Threshold below which local correlation is classified as antiphase (default -0.35).
        duration_threshold_percent: Percentage of duration in antiphase required to flag persistent anomaly (default 35.0%).
        angle: Viewpoint angle ('90', '45', 'Front'/'0', 'Rear'/'180', etc.).

    Returns:
        IdentitySwapDetectionResult.
    """
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}.")

    arr_pred = np.asarray(y_pred, dtype=np.float64).squeeze()
    arr_true = np.asarray(y_true, dtype=np.float64).squeeze()

    if arr_pred.ndim != 1 or arr_true.ndim != 1:
        raise ValueError("Signals must be 1D.")
    if len(arr_pred) != len(arr_true):
        raise ValueError("Signal lengths mismatch.")

    is_axial = is_axial_view(angle)
    n_samples = len(arr_pred)
    w_len = max(int(np.round(fps * window_seconds)), 4)
    step = max(int(np.round(fps * step_seconds)), 1)

    if n_samples <= w_len:
        # Fallback for ultra-short clips
        std_p = np.std(arr_pred)
        std_t = np.std(arr_true)
        if std_p > 1e-5 and std_t > 1e-5:
            r = float(np.corrcoef(arr_pred, arr_true)[0, 1])
        else:
            r = 0.0
        has_persistent_antiphase = bool(r < antiphase_threshold)
        pct = 100.0 if has_persistent_antiphase else 0.0
        has_swap = has_persistent_antiphase and not is_axial
        is_depth_artifact = has_persistent_antiphase and is_axial
        diag = (
            "Depth/Foreshortening Artifact (Axial View)" if is_depth_artifact
            else "True Identity Swap (Occlusion Crossover)" if has_swap
            else "None (In-Phase)"
        )
        return IdentitySwapDetectionResult(
            has_identity_swap=has_swap,
            antiphase_percentage=pct,
            min_rolling_corr=r,
            mean_rolling_corr=r,
            max_rolling_corr=r,
            is_depth_inversion_artifact=is_depth_artifact,
            is_axial_view=is_axial,
            diagnostic=diag
        )

    corrs = []
    for start in range(0, n_samples - w_len + 1, step):
        seg_p = arr_pred[start : start + w_len]
        seg_t = arr_true[start : start + w_len]
        if np.std(seg_p) > 1e-5 and np.std(seg_t) > 1e-5:
            r = np.corrcoef(seg_p, seg_t)[0, 1]
            if np.isfinite(r):
                corrs.append(float(r))

    if not corrs:
        return IdentitySwapDetectionResult(
            has_identity_swap=False,
            antiphase_percentage=0.0,
            min_rolling_corr=0.0,
            mean_rolling_corr=0.0,
            max_rolling_corr=0.0,
            is_depth_inversion_artifact=False,
            is_axial_view=is_axial,
            diagnostic="None (No Signal Variance)"
        )

    corrs_arr = np.array(corrs)
    antiphase_pct = float(np.mean(corrs_arr < antiphase_threshold) * 100.0)
    has_persistent_antiphase = bool(antiphase_pct >= duration_threshold_percent)
    has_swap = has_persistent_antiphase and not is_axial
    is_depth_artifact = has_persistent_antiphase and is_axial

    if is_depth_artifact:
        diag = "Depth/Foreshortening Artifact (Axial View)"
    elif has_swap:
        diag = "True Identity Swap (Occlusion Crossover)"
    else:
        diag = "None (In-Phase)"

    return IdentitySwapDetectionResult(
        has_identity_swap=has_swap,
        antiphase_percentage=antiphase_pct,
        min_rolling_corr=float(np.min(corrs_arr)),
        mean_rolling_corr=float(np.mean(corrs_arr)),
        max_rolling_corr=float(np.max(corrs_arr)),
        is_depth_inversion_artifact=is_depth_artifact,
        is_axial_view=is_axial,
        diagnostic=diag
    )

