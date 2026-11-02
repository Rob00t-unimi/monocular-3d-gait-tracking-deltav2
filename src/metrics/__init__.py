"""
Metrics package for gait kinematics and 3D tracking evaluation.
"""

from src.metrics.calculator import (
    calculate_all_metrics,
    GaitEvaluationReport,
)
from src.metrics.waveform.pearson import calculate_pearson
from src.metrics.waveform.cross_correlation import (
    calculate_cross_correlation,
    CrossCorrelationResult,
    detect_identity_swap_cross_correlation,
    IdentitySwapDetectionResult,
    is_axial_view,
)
from src.metrics.magnitude.nrmse import calculate_rmse, calculate_nrmse
from src.metrics.magnitude.rom_error import (
    calculate_rom,
    calculate_rom_error,
    ROMResult,
)
from src.metrics.frequency.cadence_error import (
    extract_dominant_frequency,
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

__all__ = [
    "calculate_all_metrics",
    "GaitEvaluationReport",
    "calculate_pearson",
    "calculate_cross_correlation",
    "CrossCorrelationResult",
    "detect_identity_swap_cross_correlation",
    "IdentitySwapDetectionResult",
    "is_axial_view",
    "calculate_rmse",
    "calculate_nrmse",
    "calculate_rom",
    "calculate_rom_error",
    "ROMResult",
    "extract_dominant_frequency",
    "calculate_cadence_error",
    "CadenceResult",
    "calculate_rigidity_index",
    "RigidityResult",
    "calculate_dtw",
    "DTWResult",
    "calculate_3d_jerk",
    "SmoothnessResult",
]
