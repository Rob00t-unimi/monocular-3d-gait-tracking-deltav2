"""
Data models for the 3D Gait Tracking Pipeline.
Defines immutable entities representing videos and their processing state.
"""

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class VideoInfo:
    """
    Immutable data representing a single video test and its associated sensors.

    Attributes:
        name: Filename of the video.
        video_path: Absolute or relative path to the MP4 file.
        skeleton_path: Path to the skeleton AVI file (if available).
        keypoints_map: Dictionary mapping frame indices (str) to OpenPose JSON paths.
        sensor_data_path: Path to the raw IMU CSV file (if available).
        person: Participant ID (e.g., 1 for P1).
        experiment: Experiment ID (e.g., 1 for E1).
        test: Test ID (e.g., 1 for T1).
        angle: Camera angle (e.g., '45', '90', 'Front', 'Rear').
        is_alt: Whether it uses alternative clothing.
        sensor_position: Body part where the IMU is located (e.g., 'leg').
        interests_keypoints: List of OpenPose keypoint names relevant for this sensor position.
    """
    name: str
    video_path: Path
    skeleton_path: Path
    keypoints_map: Dict[str, str]
    sensor_data_path: Optional[str]
    person: int
    experiment: int
    test: int
    angle: str
    is_alt: bool
    sensor_position: str
    interests_keypoints: List[str]

    def __post_init__(self):
        """Validates paths and identifiers."""
        if not self.name:
            raise ValueError("Video name cannot be empty.")
        if self.person < 0 or self.experiment < 0 or self.test < 0:
            raise ValueError("Person, experiment, and test IDs must be non-negative.")

    def get_frame_keypoints_path(self, frame_idx: int) -> str:
        """
        Retrieves the OpenPose JSON path for a specific frame.

        Args:
            frame_idx: The integer index of the requested frame.

        Returns:
            The file path as a string, or an empty string if not found.
        """
        idx = str(frame_idx)
        return self.keypoints_map.get(idx, "")

@dataclass(frozen=True)
class VideoMetadata:
    """
    Physical properties extracted from a video file.

    Attributes:
        fps: Frames per second.
        frames_num: Total number of frames in the video.
        width: Pixel width.
        height: Pixel height.
    """
    fps: float = 0.0
    frames_num: int = 0
    width: int = 0
    height: int = 0

    @property
    def duration_seconds(self) -> float:
        """Calculates total video duration in seconds."""
        if self.fps <= 0: return 0.0
        return self.frames_num / self.fps

@dataclass
class ProcessingState:
    """
    Mutable state representing the current tracking and validation progress.

    Attributes:
        fullbody_discarded: Flag if the fullbody keypoints are marked as invalid.
        specific_discarded: Flag if the specific points (e.g., thigh) are invalid.
        ref_frame_idx: Optimal frame index selected for tracking initialization.
        fullbody_keypoints: Dictionary of ALL detected keypoints for the ref frame.
        specific_keypoints: Coordinates of the 3 points (P1, P2, P3) for tracking.
    """
    fullbody_discarded: bool = False
    specific_discarded: bool = False
    ref_frame_idx: Optional[int] = None
    fullbody_keypoints: Dict[str, List[float]] = field(default_factory=dict)
    specific_keypoints: List[List[float]] = field(default_factory=list)

    def update_fullbody(self, ref_frame_idx: int, keypoints: Dict[str, List[float]]) -> None:
        """Updates fullbody data while ensuring deep copy for integrity."""
        self.ref_frame_idx = ref_frame_idx
        self.fullbody_keypoints = deepcopy(keypoints)
        self.fullbody_discarded = False

    def update_specific(self, ref_frame_idx: int, keypoints: List[List[float]]) -> None:
        """Updates specific points data while ensuring deep copy for integrity."""
        self.ref_frame_idx = ref_frame_idx
        self.specific_keypoints = deepcopy(keypoints)
        self.specific_discarded = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the state to a dictionary for JSON persistence."""
        return {
            "ref_frame_idx": self.ref_frame_idx,
            "specific": {
                "discarded": self.specific_discarded,
                "keypoints": self.specific_keypoints if not self.specific_discarded else []
            },
            "fullbody": {
                "discarded": self.fullbody_discarded,
                "keypoints": self.fullbody_keypoints if not self.fullbody_discarded else {}
            }
        }
