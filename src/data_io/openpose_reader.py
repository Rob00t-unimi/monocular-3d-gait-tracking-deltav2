"""
Module for parsing OpenPose JSON output files.
Implements IPoseLoader protocol.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from src.core.interfaces import IPoseLoader

logger = logging.getLogger(__name__)

# Body_25 mapping (moved to local to avoid circular import if needed, or referenced from constants)
BODY_25_MAP = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24
}

class OpenPoseLoader(IPoseLoader):
    """
    Handles JSON parsing for 2D keypoints detected by OpenPose.
    """

    def load_frame_keypoints(self, json_path: Path) -> Dict[str, List[float]]:
        """
        Loads the 2D keypoints for a single person from an OpenPose JSON file.
        
        Args:
            json_path: Path to the JSON file for a specific frame.
            
        Returns:
            A dictionary mapping keypoint names to [x, y, confidence].
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Use the first person detected
            if not data.get("people"):
                return {}

            keypoints_raw = data["people"][0].get("pose_keypoints_2d", [])
            
            # Map indices to names
            keypoints_map = {}
            for name, idx in BODY_25_MAP.items():
                start = idx * 3
                if start + 2 < len(keypoints_raw):
                    # Store as [x, y, confidence]
                    keypoints_map[name] = keypoints_raw[start : start + 3]
            
            return keypoints_map
            
        except Exception as e:
            logger.error(f"Error parsing keypoints from {json_path}: {e}")
            return {}

# Backward compatibility (optional instance)
openpose_reader = OpenPoseLoader()
OPENPOSE_MAP = BODY_25_MAP
