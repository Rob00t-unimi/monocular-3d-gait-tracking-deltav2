from typing import Protocol, List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd

from src.core.models import VideoInfo, VideoMetadata, ProcessingState

class ISignalLoader(Protocol):
    """Protocol for loading raw IMU signal data."""
    def load(self, file_path: Path) -> Optional[pd.DataFrame]: ...

class ISignalProcessor(Protocol):
    """Protocol for transforming and filtering IMU signals."""
    def process(self, df: pd.DataFrame, lowcut: float, highcut: float) -> pd.DataFrame: ...

class ISignalValidator(Protocol):
    """Protocol for validating processed IMU signals."""
    def is_valid(self, df: pd.DataFrame) -> bool: ...

class IFrequencyEstimator(Protocol):
    """Protocol for estimating dominant gait frequencies."""
    def estimate(self, sample_paths: List[Path]) -> Tuple[float, float]: ...

class IVideoProvider(Protocol):
    """Protocol for video frame extraction and metadata reading."""
    def get_metadata(self, video_path: Path) -> VideoMetadata: ...
    def get_frame(self, video_path: Path, frame_idx: int) -> Tuple[Optional[np.ndarray], int]: ...

class IPoseLoader(Protocol):
    """Protocol for loading 2D pose data (e.g. OpenPose)."""
    def load_frame_keypoints(self, json_path: Path) -> Dict[str, List[float]]: ...

class IPointComposer(Protocol):
    """Protocol for calculating 3D tracking points (e.g. femur triangle)."""
    def compose(self, points: Dict[str, Any], frame: Optional[np.ndarray] = None) -> Tuple[List[List[float]], Optional[np.ndarray], Optional[np.ndarray]]: ...

class IStateRepository(Protocol):
    """Protocol for persisting the processing state of the pipeline."""
    def load(self, name: str) -> ProcessingState: ...
    def save(self, name: str, state: ProcessingState) -> None: ...
