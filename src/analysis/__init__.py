"""
Analysis package: signal extraction, GT loading, and metrics for the
Gait360 benchmark evaluation.
"""

from src.analysis.signal_extractor import (
    extract_signals,
    extract_raw_signals,
    extract_aligned_signals,
    load_cutoffs,
)
from src.analysis.gt_loader import load_gt, load_raw_imu, get_gt_csv_path
from src.analysis.kinematic_alignment import (
    extract_tracking_kinematics,
    convert_imu_to_world,
    filter_signals,
    temporal_align,
    align_camera_to_world,
    run_pipeline,
)

__all__ = [
    "extract_signals",
    "extract_raw_signals",
    "extract_aligned_signals",
    "load_cutoffs",
    "load_gt",
    "load_raw_imu",
    "get_gt_csv_path",
    "extract_tracking_kinematics",
    "convert_imu_to_world",
    "filter_signals",
    "temporal_align",
    "align_camera_to_world",
    "run_pipeline",
]
