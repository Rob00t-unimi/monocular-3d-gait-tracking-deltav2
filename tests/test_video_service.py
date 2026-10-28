"""
Unit tests for the refactored VideoProcessingService.
Ensures correct interaction with StateManager and VisionOrchestrator.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.preprocessing.video_prepper import VideoProcessingService
from src.core.models import VideoInfo, VideoMetadata, ProcessingState

class TestVideoProcessingService(unittest.TestCase):
    def setUp(self):
        self.service = VideoProcessingService()
        self.video_info = VideoInfo(
            name="test_video",
            video_path=MagicMock(),
            skeleton_path=MagicMock(),
            keypoints_map={},
            sensor_data_path="test_sensor.csv",
            person=1,
            experiment=1,
            test=1,
            angle="90",
            is_alt=False,
            sensor_position="leg",
            interests_keypoints=["RHip", "RKnee"]
        )
        self.metadata = VideoMetadata(fps=30, frames_num=100, width=640, height=480)

    @patch('src.preprocessing.video_prepper.videos_state_manager')
    def test_discard_fullbody(self, mock_state_manager):
        # Setup: load returns a specific state object
        state = ProcessingState()
        mock_state_manager.load.return_value = state
        
        self.service.discard_fullbody("test_video")
        
        # Verify
        self.assertTrue(state.fullbody_discarded)
        mock_state_manager.save.assert_called_once()

    @patch('src.preprocessing.video_prepper.videos_state_manager')
    def test_discard_specific(self, mock_state_manager):
        state = ProcessingState()
        mock_state_manager.load.return_value = state
        
        self.service.discard_specific("test_video")
        
        self.assertTrue(state.specific_discarded)
        mock_state_manager.save.assert_called_once()

    @patch('src.preprocessing.video_prepper.videos_state_manager')
    @patch('src.preprocessing.video_prepper.video_provider')
    @patch('src.preprocessing.video_prepper.ref_frame_finder')
    @patch('cv2.cvtColor')
    def test_get_reference_frame_not_in_state(self, mock_cvtColor, mock_finder, mock_provider, mock_state_manager):
        # Setup: Empty state (fresh processing)
        state = ProcessingState()
        mock_state_manager.load.return_value = state
        
        # Mock dependencies
        mock_finder.find_specific_keypoints_ref.return_value = {"ref_frame_idx": 20, "keypoints": {"RHip": [0,0]}}
        mock_provider.get_frame.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), 20)
        mock_cvtColor.return_value = "rgb_frame"
        
        # Mock vision orchestrator inside service
        self.service.vision_orchestrator = MagicMock()
        self.service.vision_orchestrator.compose.return_value = ([[1.0, 2.0]], "mask_base", "mask_inner")

        # Act
        frame_rgb, out_state, m_base, m_inner = self.service.get_reference_frame(self.video_info, self.metadata)

        # Assert
        self.assertEqual(frame_rgb, "rgb_frame")
        self.assertEqual(m_base, "mask_base")
        self.assertEqual(m_inner, "mask_inner")
        self.assertEqual(out_state.ref_frame_idx, 20)
        mock_state_manager.save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
