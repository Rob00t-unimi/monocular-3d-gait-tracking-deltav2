"""
Unified Gait Metrics Calculator.

Aggregates all individual kinematic evaluation metrics into a comprehensive report:
- Waveform temporal synchrony (Pearson r, Cross-correlation lag)
- Amplitude and Magnitude errors (nRMSE %, Range of Motion error)
- Frequency and gait cadence (dominant frequency, cadence SPM, identity swap detection)
- 3D Geometric integrity (Rigidity deformation index, Dynamic Time Warping distance)
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np

from src.metrics.waveform.pearson import calculate_pearson
from src.metrics.waveform.cross_correlation import (
    calculate_cross_correlation,
    CrossCorrelationResult,
    detect_identity_swap_cross_correlation,
    IdentitySwapDetectionResult,
)
from src.metrics.magnitude.nrmse import calculate_nrmse, calculate_rmse
from src.metrics.magnitude.rom_error import calculate_rom_error, ROMResult
from src.metrics.frequency.cadence_error import (
    calculate_cadence_error,
    CadenceResult,
)
from src.metrics.geometry.rigidity import (
    calculate_rigidity_index,
    RigidityResult,
)
from src.metrics.geometry.dtw import calculate_dtw, DTWResult
from src.metrics.geometry.smoothness import (
    calculate_3d_jerk,
    SmoothnessResult,
)


@dataclass(frozen=True)
class GaitEvaluationReport:
    """
    Comprehensive gait analysis evaluation report.
    """
    # 1. Waveform Metrics
    pearson_pitch: float
    pearson_acc: Optional[float]
    cross_corr_pitch: CrossCorrelationResult
    cross_corr_acc: Optional[CrossCorrelationResult]

    # 2. Magnitude & Amplitude Metrics
    nrmse_pitch_percent: float
    nrmse_acc_percent: Optional[float]
    rom_pitch: ROMResult
    rom_acc: Optional[ROMResult]

    # 3. Frequency & Cadence
    cadence_pitch: CadenceResult

    # 4. 3D Geometric Consistency
    dtw_pitch: DTWResult
    dtw_acc3d: Optional[DTWResult]
    rigidity: Optional[RigidityResult]
    smoothness: Optional[SmoothnessResult] = None
    swap_detection: Optional[IdentitySwapDetectionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flattens all metrics into a dictionary."""
        return {
            "pearson_pitch": self.pearson_pitch,
            "pearson_acc": self.pearson_acc,
            "pitch_lag_seconds": self.cross_corr_pitch.lag_seconds,
            "pitch_peak_cross_corr": self.cross_corr_pitch.peak_correlation,
            "acc_lag_seconds": self.cross_corr_acc.lag_seconds if self.cross_corr_acc else None,
            "nrmse_pitch_percent": self.nrmse_pitch_percent,
            "nrmse_acc_percent": self.nrmse_acc_percent,
            "rom_error_pitch": self.rom_pitch.absolute_error,
            "rom_relative_error_pitch_percent": self.rom_pitch.relative_error_percent,
            "cadence_pred_bpm": self.cadence_pitch.cadence_pred_bpm,
            "cadence_true_bpm": self.cadence_pitch.cadence_true_bpm,
            "cadence_error_percent": self.cadence_pitch.relative_error_percent,
            "cadence_freq_ratio": self.cadence_pitch.frequency_ratio,
            "has_identity_swap": (
                self.swap_detection.has_identity_swap
                if self.swap_detection is not None
                else self.cadence_pitch.has_identity_swap
            ),
            "is_depth_inversion_artifact": (
                self.swap_detection.is_depth_inversion_artifact
                if self.swap_detection is not None
                else False
            ),
            "antiphase_percentage": (
                self.swap_detection.antiphase_percentage
                if self.swap_detection is not None
                else 0.0
            ),
            "swap_diagnostic": (
                self.swap_detection.diagnostic
                if self.swap_detection is not None
                else "None"
            ),
            "dtw_pitch_normalized_distance": self.dtw_pitch.normalized_distance,
            "dtw_acc3d_normalized_distance": (
                self.dtw_acc3d.normalized_distance if self.dtw_acc3d else None
            ),
            "sigma_rigidity": self.rigidity.sigma_rigidity if self.rigidity else None,
            "relative_deformation_percent": (
                self.rigidity.relative_deformation_percent if self.rigidity else None
            ),
            "rms_jerk": self.smoothness.rms_jerk if self.smoothness else None,
        }

    def to_summary_table_row(self, method_name: str = "Proposed") -> Dict[str, Any]:
        """
        Formats metrics into the standard benchmark summary row for thesis reports.
        """
        if self.swap_detection is not None:
            if self.swap_detection.has_identity_swap:
                swap_label = "YES"
            elif self.swap_detection.is_depth_inversion_artifact:
                swap_label = "NO (Depth Artifact)"
            else:
                swap_label = "NO"
        else:
            swap_label = "YES" if self.cadence_pitch.has_identity_swap else "NO"

        return {
            "Method": method_name,
            "Pearson Pitch (r)": round(self.pearson_pitch, 3),
            "Pearson Acc (r)": round(self.pearson_acc, 3) if self.pearson_acc is not None else "N/A",
            "nRMSE Pitch (%)": f"{round(self.nrmse_pitch_percent, 1)}%",
            "Cadence Error (%)": f"{round(self.cadence_pitch.relative_error_percent, 1)}%",
            "Identity Swap": swap_label,
            "Rigidity 3D (sigma)": (
                round(self.rigidity.sigma_rigidity, 4) if self.rigidity else "N/A"
            ),
            "3D Rel Def (%)": (
                f"{round(self.rigidity.relative_deformation_percent, 1)}%" if self.rigidity else "N/A"
            ),
            "3D Jerk (RMS)": (
                round(self.smoothness.rms_jerk, 1) if self.smoothness else "N/A"
            ),
            "DTW Pitch (norm)": round(self.dtw_pitch.normalized_distance, 3),
            "3D Acc DTW (norm)": (
                round(self.dtw_acc3d.normalized_distance, 3) if self.dtw_acc3d else "N/A"
            ),
        }


def calculate_all_metrics(
    pitch_pred: Union[np.ndarray, list],
    pitch_true: Union[np.ndarray, list],
    acc_pred: Optional[Union[np.ndarray, list]] = None,
    acc_true: Optional[Union[np.ndarray, list]] = None,
    track3d: Optional[np.ndarray] = None,
    acc3d_pred: Optional[np.ndarray] = None,
    acc3d_true: Optional[np.ndarray] = None,
    fps_pred: float = 30.0,
    fps_true: Optional[float] = None,
    angle: Optional[Union[str, int, float]] = None,
) -> GaitEvaluationReport:
    """
    Computes all gait kinematics evaluation metrics in a single coordinated pass.

    Args:
        pitch_pred: Predicted thigh Pitch trajectory.
        pitch_true: Ground Truth thigh Pitch trajectory.
        acc_pred: Optional predicted principal acceleration (e.g. AccX).
        acc_true: Optional GT principal acceleration.
        track3d: Optional (T, N, 3) tracking points array for 3D rigidity analysis.
        acc3d_pred: Optional (T, 3) 3D acceleration vectors for 3D DTW.
        acc3d_true: Optional (T, 3) 3D acceleration vectors for 3D DTW.
        fps_pred: Sampling rate of predicted signal in Hz (default 30.0).
        fps_true: Sampling rate of GT signal in Hz (defaults to fps_pred).
        angle: Optional viewpoint angle ('90', '45', 'Front'/'0', 'Rear'/'180') for biomechanical swap validation.

    Returns:
        GaitEvaluationReport containing all computed metrics.
    """
    fs_gt = fps_true if fps_true is not None else fps_pred

    # 1. Pitch Waveform Metrics
    p_pitch = calculate_pearson(pitch_pred, pitch_true)
    cc_pitch = calculate_cross_correlation(pitch_pred, pitch_true, fps=fps_pred)
    nrmse_pitch = calculate_nrmse(pitch_pred, pitch_true, normalization="rom", as_percentage=True)
    rom_p = calculate_rom_error(pitch_pred, pitch_true)
    cad_p = calculate_cadence_error(pitch_pred, pitch_true, fs_pred=fps_pred, fs_true=fs_gt)
    dtw_p = calculate_dtw(pitch_pred, pitch_true, normalize=True)
    swap_det = detect_identity_swap_cross_correlation(
        pitch_pred, pitch_true, fps=fps_pred, angle=angle
    )

    # 2. Acceleration Metrics (optional)
    p_acc = None
    cc_acc = None
    nrmse_acc = None
    rom_a = None
    if acc_pred is not None and acc_true is not None:
        p_acc = calculate_pearson(acc_pred, acc_true)
        cc_acc = calculate_cross_correlation(acc_pred, acc_true, fps=fps_pred)
        nrmse_acc = calculate_nrmse(acc_pred, acc_true, normalization="rom", as_percentage=True)
        rom_a = calculate_rom_error(acc_pred, acc_true)

    # 3. 3D Acceleration DTW (optional)
    dtw_a3d = None
    if acc3d_pred is not None and acc3d_true is not None:
        dtw_a3d = calculate_dtw(acc3d_pred, acc3d_true, normalize=True)

    # 4. 3D Rigidity and Smoothness (optional)
    rig = None
    smooth = None
    if track3d is not None:
        rig = calculate_rigidity_index(track3d)
        try:
            smooth = calculate_3d_jerk(track3d, fps=fps_pred)
        except Exception:
            smooth = None

    return GaitEvaluationReport(
        pearson_pitch=p_pitch,
        pearson_acc=p_acc,
        cross_corr_pitch=cc_pitch,
        cross_corr_acc=cc_acc,
        nrmse_pitch_percent=nrmse_pitch,
        nrmse_acc_percent=nrmse_acc,
        rom_pitch=rom_p,
        rom_acc=rom_a,
        cadence_pitch=cad_p,
        dtw_pitch=dtw_p,
        dtw_acc3d=dtw_a3d,
        rigidity=rig,
        smoothness=smooth,
        swap_detection=swap_det,
    )
