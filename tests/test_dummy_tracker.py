"""
Unit tests for DummyTracker.
Verifies geometric point transfer, reference frame identity, missing frame interpolation,
and 3D lifting.
"""

import unittest
from pathlib import Path
import numpy as np
import tempfile
import shutil

from src.core.models import VideoInfo, ProcessingState, VideoMetadata
from src.inference.dummy_tracker import DummyTracker


class MockPoseLoader:
    def __init__(self, keypoints_by_path=None):
        self.keypoints_by_path = keypoints_by_path or {}

    def load_frame_keypoints(self, json_path: Path):
        return self.keypoints_by_path.get(str(json_path).replace("\\", "/"), {})


class MockVideoProvider:
    def __init__(self, width=1920, height=1080, fps=30.0):
        self.width = width
        self.height = height
        self.fps = fps

    def get_metadata(self, video_path: Path):
        return VideoMetadata(fps=self.fps, frames_num=100, width=self.width, height=self.height)

    def get_frame(self, video_path: Path, frame_idx: int):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8), frame_idx


class TestDummyTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.knee_ref = np.array([100.0, 500.0])
        self.hip_ref = np.array([100.0, 300.0])  # Length = 200, pointing upwards (u = [0, -1], u_perp = [1, 0])
        
        # 3 thigh points: P1 near knee, P2 and P3 laterally offset
        self.pts_init = [
            [100.0, 480.0],  # along axis
            [120.0, 450.0],  # lateral right
            [80.0, 450.0]    # lateral left
        ]
        
        self.state = ProcessingState(
            ref_frame_idx=10,
            specific_keypoints=self.pts_init,
            fullbody_keypoints={
                "RHip": {"x": self.hip_ref[0], "y": self.hip_ref[1], "confidence": 0.95},
                "RKnee": {"x": self.knee_ref[0], "y": self.knee_ref[1], "confidence": 0.92}
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_relative_coordinates_and_reconstruction(self):
        tracker = DummyTracker()
        pts = np.array(self.pts_init)
        v_mag, u, u_perp, alphas, betas = tracker.compute_relative_coordinates(
            pts, self.knee_ref, self.hip_ref
        )
        self.assertAlmostEqual(v_mag, 200.0)
        
        # Reconstruct on same femur
        pts_recon, _ = tracker.transfer_points_to_frame(alphas, betas, self.knee_ref, self.hip_ref)
        np.testing.assert_allclose(pts_recon, pts, atol=1e-5)

    def test_rigid_transformation_translation_and_scale(self):
        tracker = DummyTracker()
        pts = np.array(self.pts_init)
        _, _, _, alphas, betas = tracker.compute_relative_coordinates(
            pts, self.knee_ref, self.hip_ref
        )

        # Shift femur by (+50, +20) and double length
        new_knee = self.knee_ref + np.array([50.0, 20.0])
        new_hip = new_knee + np.array([0.0, -400.0])  # length = 400

        pts_trans, new_mag = tracker.transfer_points_to_frame(alphas, betas, new_knee, new_hip)
        self.assertAlmostEqual(new_mag, 400.0)

        # P1 was originally at 10% along femur (20px from knee in 200px length)
        # In new femur, it should be at 10% of 400px = 40px from new knee
        expected_p1 = new_knee + np.array([0.0, -40.0])
        np.testing.assert_allclose(pts_trans[0], expected_p1, atol=1e-5)

    def test_tracking_sequence_identity_at_ref_frame(self):
        # Setup mock data for 5 frames: 8, 9, 10 (ref), 11, 12
        keypoints_map = {}
        mock_kpts = {}
        for f in range(8, 13):
            p_str = f"mock_json/f_{f:03d}.json"
            keypoints_map[str(f)] = p_str
            # Small shift per frame
            dx = (f - 10) * 5.0
            mock_kpts[p_str] = {
                "RHip": [self.hip_ref[0] + dx, self.hip_ref[1], 0.9],
                "RKnee": [self.knee_ref[0] + dx, self.knee_ref[1], 0.9]
            }

        mock_loader = MockPoseLoader(mock_kpts)
        tracker = DummyTracker(pose_loader=mock_loader, video_reader=MockVideoProvider())

        video_info = VideoInfo(
            name="test_video.mp4",
            video_path=Path("dummy_test.mp4"),
            skeleton_path=Path("dummy_skel.avi"),
            keypoints_map=keypoints_map,
            sensor_data_path=None,
            person=1,
            experiment=1,
            test=1,
            angle="45",
            is_alt=False,
            sensor_position="leg",
            interests_keypoints=["RHip", "RKnee"]
        )

        res = tracker.track(video_info, self.state, start_frame=8, end_frame=12)

        # Shape checks
        self.assertEqual(res["track2d"].shape, (5, 3, 2))
        self.assertEqual(res["track3d"].shape, (5, 3, 3))
        self.assertEqual(res["visibility"].shape, (5, 3))
        self.assertEqual(res["confidence"].shape, (5, 3))

        # Check ref_frame_idx (index 2 in 8..12) exactly matches pts_init
        rel_ref_idx = 10 - 8
        np.testing.assert_allclose(res["track2d"][rel_ref_idx], np.array(self.pts_init), atol=1e-5)

    def test_missing_frame_interpolation(self):
        # Frame 9 is missing (empty detection)
        keypoints_map = {
            "8": "mock_json/f_008.json",
            "9": "mock_json/f_009.json",
            "10": "mock_json/f_010.json"
        }
        mock_kpts = {
            "mock_json/f_008.json": {
                "RHip": [100.0, 300.0, 0.9],
                "RKnee": [100.0, 500.0, 0.9]
            },
            "mock_json/f_009.json": {},  # Missing!
            "mock_json/f_010.json": {
                "RHip": [120.0, 300.0, 0.9],
                "RKnee": [120.0, 500.0, 0.9]
            }
        }
        mock_loader = MockPoseLoader(mock_kpts)
        tracker = DummyTracker(pose_loader=mock_loader, video_reader=MockVideoProvider())

        video_info = VideoInfo(
            name="test_missing.mp4",
            video_path=Path("dummy_test.mp4"),
            skeleton_path=Path("dummy_skel.avi"),
            keypoints_map=keypoints_map,
            sensor_data_path=None,
            person=1,
            experiment=1,
            test=1,
            angle="45",
            is_alt=False,
            sensor_position="leg",
            interests_keypoints=["RHip", "RKnee"]
        )

        res = tracker.track(video_info, self.state, start_frame=8, end_frame=10)

        # Frame 9 (index 1) should be interpolated to X = 110.0
        self.assertAlmostEqual(res["track2d"][1, 0, 0], 110.0, places=3)
        # Interpolated frame should have 0 visibility and 0 confidence
        self.assertEqual(res["visibility"][1, 0], 0.0)
        self.assertEqual(res["confidence"][1, 0], 0.0)
        # Valid frames should have visibility = 1.0
        self.assertEqual(res["visibility"][0, 0], 1.0)
        self.assertEqual(res["visibility"][2, 0], 1.0)

    def test_save_results(self):
        tracker = DummyTracker(video_reader=MockVideoProvider())
        keypoints_map = {"10": "mock_json/f_010.json"}
        mock_loader = MockPoseLoader({"mock_json/f_010.json": {
            "RHip": [100.0, 300.0, 0.9],
            "RKnee": [100.0, 500.0, 0.9]
        }})
        tracker.pose_loader = mock_loader

        video_info = VideoInfo(
            name="test_save.mp4",
            video_path=Path("dummy_test.mp4"),
            skeleton_path=Path("dummy_skel.avi"),
            keypoints_map=keypoints_map,
            sensor_data_path=None,
            person=1,
            experiment=1,
            test=1,
            angle="45",
            is_alt=False,
            sensor_position="leg",
            interests_keypoints=["RHip", "RKnee"]
        )

        res = tracker.track(video_info, self.state, start_frame=10, end_frame=10)
        out_path = Path(self.temp_dir) / "test_out.npz"
        saved = tracker.save_results(res, out_path)
        self.assertTrue(saved.exists())

        loaded = np.load(saved)
        self.assertIn("track2d", loaded)
        self.assertIn("track3d", loaded)
        self.assertIn("track3d_world", loaded)
        self.assertIn("survived_track3d", loaded)
        self.assertEqual(str(loaded["video_name"]), "test_save.mp4")


if __name__ == '__main__':
    unittest.main()
