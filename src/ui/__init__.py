"""Interactive graphical user interfaces for 2D keypoint validation, cohort selection, and 3D/kinematic inspection."""

from src.ui.keypoint_validator_gui import ValidatorGUI
from src.ui.cohort_selector_gui import InferenceVideoSelectorGUI, VideoSelectorApp
from src.ui.trajectory_3d_inspector import Unified3DViewer, Unified3DInspector, inspect_signals_3d
from src.ui.kinematic_signal_inspector import UnifiedSignalBrowser, UnifiedSignalInspector, inspect_signals
from src.ui.benchmark_dashboard import GaitBenchmarkDashboard

KeypointValidatorGUI = ValidatorGUI

__all__ = [
    "KeypointValidatorGUI",
    "ValidatorGUI",
    "InferenceVideoSelectorGUI",
    "VideoSelectorApp",
    "Unified3DViewer",
    "Unified3DInspector",
    "inspect_signals_3d",
    "UnifiedSignalBrowser",
    "UnifiedSignalInspector",
    "inspect_signals",
    "GaitBenchmarkDashboard",
]
