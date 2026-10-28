"""
High-level service for video preparation.
Orchestrates frame selection and tracking point initialization.
"""

import logging
import cv2
from pathlib import Path
from src.core.models import VideoInfo, ProcessingState, VideoMetadata
from src.data_io.video_reader import video_provider
from src.data_io.state_manager import videos_state_manager
from src.preprocessing.vision.frame_selection import ref_frame_finder
from src.preprocessing.vision.vision_orchestrator import FemurVisionOrchestrator

logger = logging.getLogger(__name__)

class VideoProcessingService:
    """
    Coordinates the video preparation phase, including reference frame selection
    and femur triangulation.
    """

    def __init__(self):
        self.vision_orchestrator = FemurVisionOrchestrator()

    def get_reference_frame(self, video_info: VideoInfo, metadata: VideoMetadata, skip_if_already_done: bool = False):
        """
        Retrieves or calculates the optimal reference frame and tracking points.
        
        Args:
            video_info: Domain model for the video.
            metadata: Physical metadata of the video.
            skip_if_already_done: If True, returns (None, state, None, None) if state exists.
        """
        state = videos_state_manager.load(video_info.name)
        
        # Check if we have ANY previous state for this video
        is_processed = state.ref_frame_idx is not None or state.specific_discarded or state.fullbody_discarded
        
        if is_processed:
            if skip_if_already_done:
                return None, state, None, None
            
            # If it's discarded, we still return it so the GUI can show it's discarded
            if state.ref_frame_idx is None:
                return None, state, None, None

            # Load the frame
            frame_bgr, _ = video_provider.get_frame(video_info.video_path, state.ref_frame_idx)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr is not None else None
            
            # Recalculate masks for the GUI to display correctly
            from src.data_io.openpose_reader import openpose_reader
            json_path = video_info.get_frame_keypoints_path(state.ref_frame_idx)
            raw_kps_dict = openpose_reader.load_frame_keypoints(Path(json_path))
            
            extracted = {}
            for kp_name in video_info.interests_keypoints:
                if kp_name in raw_kps_dict:
                    val = raw_kps_dict[kp_name]
                    if val[2] > 0.05:
                        extracted[kp_name] = {"x": val[0], "y": val[1], "confidence": val[2]}

            # We don't overwrite triangle/keypoints if they already exist in state
            _, m_base, m_inner = self.vision_orchestrator.compose(extracted, frame_bgr)
            
            return frame_rgb, state, m_base, m_inner

        # --- No previous state: Run Automatic Detection ---
        logger.info(f"Automatically calculating reference for {video_info.name}...")
        
        # 1. Find optimal frame index
        res = ref_frame_finder.find_specific_keypoints_ref(video_info, metadata)
        frame_idx = res["ref_frame_idx"]

        # 2. Extract fullbody keypoints for that frame
        from src.data_io.openpose_reader import BODY_25_MAP
        kp_array = ref_frame_finder._get_keypoints_array_for_frame(video_info, frame_idx)
        fullbody_kps = ref_frame_finder._extract_frame_kps(kp_array, list(BODY_25_MAP.keys()), 0.1)

        # 3. Compute tracking points (triangulation)
        img_bgr, _ = video_provider.get_frame(video_info.video_path, frame_idx)
        triangle, m_base, m_inner = self.vision_orchestrator.compose(res["keypoints"], img_bgr)

        # 4. Update and persist state
        state.update_specific(frame_idx, triangle)
        state.update_fullbody(frame_idx, fullbody_kps)
        videos_state_manager.save(video_info.name, state)

        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        return frame_rgb, state, m_base, m_inner

    def discard_fullbody(self, video_name: str):
        state = videos_state_manager.load(video_name)
        state.fullbody_discarded = True
        videos_state_manager.save(video_name, state)

    def discard_specific(self, video_name: str):
        state = videos_state_manager.load(video_name)
        state.specific_discarded = True
        videos_state_manager.save(video_name, state)

    def update_reference_frame(self, video_info: VideoInfo, frame_idx: int, specific_points: list):
        """Manual update of the reference frame from GUI."""
        state = videos_state_manager.load(video_info.name)
        from src.data_io.openpose_reader import BODY_25_MAP
        kp_array = ref_frame_finder._get_keypoints_array_for_frame(video_info, frame_idx)
        fullbody_kps = ref_frame_finder._extract_frame_kps(kp_array, list(BODY_25_MAP.keys()), 0.1)

        state.update_specific(frame_idx, specific_points)
        state.update_fullbody(frame_idx, fullbody_kps)
        videos_state_manager.save(video_info.name, state)

# Exported instance for the pipeline
video_service = VideoProcessingService()
