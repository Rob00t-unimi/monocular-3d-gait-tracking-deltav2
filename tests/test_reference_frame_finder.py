"""
Unit tests for the refactored RefFrameFinder.
"""

import unittest
from unittest.mock import patch
from src.preprocessing.vision.frame_selection import RefFrameFinder, LegSearchStrategy
from src.core.models import VideoInfo, VideoMetadata

class TestRefFrameFinder(unittest.TestCase):
    def setUp(self):
        self.finder = RefFrameFinder()
        self.metadata = VideoMetadata(fps=30, frames_num=10, width=640, height=480)
        self.video_info = VideoInfo(
            name="test_video",
            video_path=None,
            skeleton_path=None,
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

    @patch('src.preprocessing.vision.frame_selection.RefFrameFinder._front_rear_case')
    def test_find_specific_keypoints_ref_front_rear_case(self, mock_front_rear):
        mock_front_rear.return_value = {"ref_frame_idx": 0, "keypoints": {"RHip": [0,0]}}
        res = self.finder.find_specific_keypoints_ref(self.video_info, self.metadata)
        self.assertEqual(res, {"ref_frame_idx": 0, "keypoints": {"RHip": [0,0]}})

    @patch('src.preprocessing.vision.frame_selection.LegSearchStrategy.execute')
    @patch('src.preprocessing.vision.frame_selection.RefFrameFinder._front_rear_case')
    def test_find_specific_keypoints_ref_strategy_exec(self, mock_front_rear, mock_strategy):
        mock_front_rear.return_value = False
        mock_strategy.return_value = {"ref_frame_idx": 5, "keypoints": {"RHip": [0,0]}}
        res = self.finder.find_specific_keypoints_ref(self.video_info, self.metadata)
        self.assertEqual(res, {"ref_frame_idx": 5, "keypoints": {"RHip": [0,0]}})

if __name__ == '__main__':
    unittest.main()
