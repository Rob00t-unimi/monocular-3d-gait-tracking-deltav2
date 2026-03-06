"""
Module for extracting frames and metadata from video files.
Implements IVideoProvider protocol.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.core.interfaces import IVideoProvider
from src.core.models import VideoMetadata

logger = logging.getLogger(__name__)

class OpenCVVideoProvider(IVideoProvider):
    """
    Standard implementation using OpenCV for video handling.
    """

    def get_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extracts FPS, total frames, and resolution from a video.

        Args:
            video_path: Path to the MP4 file.

        Returns:
            A VideoMetadata object. Defaults to 0 for missing data.
        """
        try:
            path_str = str(video_path)
            cap = cv2.VideoCapture(path_str)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {path_str}")
                return VideoMetadata()

            fps = cap.get(cv2.CAP_PROP_FPS)
            frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return VideoMetadata(fps=fps, frames_num=frames_num, width=width, height=height)
            
        except Exception as e:
            logger.error(f"Error reading metadata from {video_path}: {e}")
            return VideoMetadata()

    def get_frame(self, video_path: Path, frame_idx: int) -> Tuple[Optional[np.ndarray], int]:
        """
        Retrieves a specific frame by index.

        Args:
            video_path: Path to the MP4 file.
            frame_idx: Zero-based frame index.

        Returns:
            A tuple containing the (Optional) frame image and the index itself.
        """
        try:
            path_str = str(video_path)
            cap = cv2.VideoCapture(path_str)
            
            if not cap.isOpened():
                return None, frame_idx

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            cap.release()

            if not success:
                logger.warning(f"Could not read frame {frame_idx} from {video_path}")
                return None, frame_idx
            
            return frame, frame_idx
            
        except Exception as e:
            logger.error(f"Error extracting frame {frame_idx} from {video_path}: {e}")
            return None, frame_idx

# Backward compatibility (optional instance)
video_provider = OpenCVVideoProvider()
