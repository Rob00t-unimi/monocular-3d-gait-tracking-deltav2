# -*- coding: utf-8 -*-
"""
Base 3D Point Tracker Interface and Data Container.

Follows SOLID Principles:
- Single Responsibility: Defines the contract and serialization for raw 3D tracking data.
- Open/Closed: New tracking models (e.g., DELTAv2, VGGT, SpaTracker) implement Base3DTracker
  without altering downstream runner scripts.
- Liskov Substitution: All trackers return a consistent TrackingResult container.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class TrackingResult:
    """
    Container for raw 3D/2D point tracking outputs before post-processing.

    Attributes:
        track2d: (T, N, 2) array of pixel coordinates (x, y)
        track3d_world: (T, N, 3) array of metric/world 3D coordinates (X, Y, Z)
        visibility: (T, N) array of visibility flags or probabilities
        confidence: (T, N) array of tracking confidence scores
        video_name: Name of the processed video file
        ref_frame_idx: Reference frame index where query points were sampled
        start_idx: Starting frame index of the processed video window
        fps: Video frame rate (default: 30.0)
        mode: Description of tracking configuration / model mode
        scales: Image scaling factors (default: [1.0, 1.0])
        metadata: Optional dictionary with supplementary runtime or model information
    """
    track2d: np.ndarray
    track3d_world: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    video_name: str
    ref_frame_idx: int
    start_idx: int
    fps: float = 30.0
    mode: str = "raw_inference"
    scales: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_npz(self, output_path: Path) -> Path:
        """
        Serializes raw tracking results to compressed NPZ format.

        Args:
            output_path: Target .npz file path.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            track2d=self.track2d,
            track3d_world=self.track3d_world,
            visibility=self.visibility,
            confidence=self.confidence,
            video_name=self.video_name,
            ref_frame_idx=self.ref_frame_idx,
            start_idx=self.start_idx,
            fps=self.fps,
            mode=self.mode,
            scales=self.scales
        )
        return output_path

    @classmethod
    def load_npz(cls, input_path: Path) -> "TrackingResult":
        """
        Loads raw tracking results from a compressed NPZ file.
        """
        data = np.load(input_path, allow_pickle=True)
        return cls(
            track2d=data["track2d"],
            track3d_world=data["track3d_world"],
            visibility=data["visibility"],
            confidence=data["confidence"],
            video_name=str(data["video_name"]),
            ref_frame_idx=int(data["ref_frame_idx"]),
            start_idx=int(data["start_idx"]),
            fps=float(data["fps"]) if "fps" in data else 30.0,
            mode=str(data["mode"]) if "mode" in data else "raw_inference",
            scales=data["scales"] if "scales" in data else np.array([1.0, 1.0])
        )


class Base3DTracker(ABC):
    """
    Abstract Base Class for 3D Video Point Trackers.
    """

    @abstractmethod
    def track_video(
        self,
        video_path: Path,
        ref_frame_idx: int,
        specific_keypoints: np.ndarray,
        max_frames: int = 250
    ) -> TrackingResult:
        """
        Tracks 3D points throughout the video sequence initialized at ref_frame_idx.

        Args:
            video_path: Path to the target video file (.mp4).
            ref_frame_idx: Frame index where reference keypoints are defined.
            specific_keypoints: (3, 2) array of reference keypoints (P1, P2, P3).
            max_frames: Maximum number of frames to load and track.

        Returns:
            TrackingResult containing unprojected 3D coordinates and 2D pixel tracks.
        """
        pass
