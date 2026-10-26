"""
Dummy Tracking Baseline for 3D Gait Analysis.

This module implements a non-deep learning geometric baseline ("Dummy Method")
that leverages existing OpenPose 2D detections and the 3 specific thigh keypoints
(P1, P2, P3) defined at the reference frame (ref_frame_idx).

At ref_frame_idx, the 3 points are projected into a scale-invariant, 2D local
coordinate system aligned with the femur axis (RKnee -> RHip).
For every frame t, the points are rigidly transferred according to the detected
femur position and scale, and naively lifted to 3D via pinhole projection.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import argparse
import numpy as np

from src.core.models import VideoInfo, ProcessingState
from src.preprocessing.vision.geometry import VectorMath
from src.data_io.openpose_reader import openpose_reader, OpenPoseLoader
from src.data_io.video_reader import video_provider
from src.data_io.state_manager import videos_state_manager
from src.data_io.dataset_scanner import DatasetScanner

logger = logging.getLogger(__name__)


def _extract_coord_conf(entry: Any) -> Optional[Tuple[float, float, float]]:
    """
    Extracts (x, y, confidence) from various keypoint representations
    (dict, list, tuple, ndarray).
    """
    if entry is None:
        return None
    if isinstance(entry, dict):
        x = entry.get("x")
        y = entry.get("y")
        c = entry.get("confidence", 1.0)
        if x is not None and y is not None:
            return float(x), float(y), float(c)
    elif isinstance(entry, (list, tuple, np.ndarray)) and len(entry) >= 2:
        c = float(entry[2]) if len(entry) > 2 else 1.0
        return float(entry[0]), float(entry[1]), float(c)
    return None


class DummyTracker:
    """
    Geometric baseline tracker.
    
    Transfers initial thigh points across video frames using OpenPose skeleton detections
    and projects them to 3D using a naive pinhole lifting.
    """

    def __init__(
        self,
        pose_loader: Optional[OpenPoseLoader] = None,
        video_reader: Optional[Any] = None
    ):
        self.pose_loader = pose_loader or openpose_reader
        self.video_reader = video_reader or video_provider

    def compute_relative_coordinates(
        self,
        specific_points: np.ndarray,
        p_knee: np.ndarray,
        p_hip: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates scale-invariant local coordinates (alphas, betas) of points
        relative to the femur coordinate system (knee origin, longitudinal axis to hip).
        
        Args:
            specific_points: Array of shape (N, 2) containing 2D points.
            p_knee: 2D coordinates of knee [x, y].
            p_hip: 2D coordinates of hip [x, y].
            
        Returns:
            Tuple of:
                - v_mag: Femur length (pixels).
                - u: Longitudinal unit vector (knee -> hip).
                - u_perp: Perpendicular unit vector (transverse).
                - alphas: Longitudinal relative coordinates (N,).
                - betas: Transverse relative coordinates (N,).
        """
        v_mag, u, u_perp, _ = VectorMath.calculate_femur_basis(p_knee, p_hip)
        if v_mag <= 1e-6:
            raise ValueError("Reference femur length is zero or too small.")

        delta_p = specific_points - p_knee
        alphas = np.dot(delta_p, u) / v_mag
        betas = np.dot(delta_p, u_perp) / v_mag
        return v_mag, u, u_perp, alphas, betas

    def transfer_points_to_frame(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        p_knee: np.ndarray,
        p_hip: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Transfers points to target frame given current knee and hip positions.
        
        Args:
            alphas: Longitudinal relative coordinates (N,).
            betas: Transverse relative coordinates (N,).
            p_knee: Current knee joint [x, y].
            p_hip: Current hip joint [x, y].
            
        Returns:
            Tuple of:
                - transferred_points: Array of shape (N, 2).
                - v_mag: Current femur length in pixels.
        """
        v_mag, u, u_perp, _ = VectorMath.calculate_femur_basis(p_knee, p_hip)
        if v_mag <= 1e-6:
            v_mag = 1.0
            u = np.array([0.0, -1.0])
            u_perp = np.array([1.0, 0.0])

        transferred = p_knee + v_mag * (alphas[:, None] * u[None, :] + betas[:, None] * u_perp[None, :])
        return transferred, v_mag

    def lift_to_3d(
        self,
        track2d: np.ndarray,
        femur_lengths: np.ndarray,
        ref_femur_length: float,
        width: int,
        height: int,
        depth_mode: str = "constant",
        base_depth: float = 2.5,
        focal_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Naively lifts 2D trajectories to 3D camera coordinates using a pinhole camera model.
        
        Args:
            track2d: Array of shape (T, N, 2) with pixel coordinates (u, v).
            femur_lengths: Array of shape (T,) with femur pixel lengths.
            ref_femur_length: Femur length at ref frame (pixels).
            width: Image width in pixels.
            height: Image height in pixels.
            depth_mode: "constant" (planar) or "anthropometric" (scaled with 1/length).
            base_depth: Base depth in meters (default 2.5m).
            focal_length: Focal length in pixels. If None, defaults to 1.2 * max(width, height).
            
        Returns:
            track3d: Array of shape (T, N, 3) with (X, Y, Z) coordinates.
        """
        t_len, n_pts, _ = track2d.shape
        cx = width / 2.0
        cy = height / 2.0
        f = focal_length if focal_length is not None else 1.2 * max(width, height)

        if depth_mode == "anthropometric":
            scale_ratio = np.clip(femur_lengths / max(ref_femur_length, 1e-5), 0.2, 5.0)
            depth_per_frame = base_depth / scale_ratio
        else:
            # "constant" default
            depth_per_frame = np.full(t_len, base_depth, dtype=np.float64)

        track3d = np.zeros((t_len, n_pts, 3), dtype=np.float64)
        for t in range(t_len):
            zt = depth_per_frame[t]
            xt = (track2d[t, :, 0] - cx) * (zt / f)
            yt = (track2d[t, :, 1] - cy) * (zt / f)
            track3d[t, :, 0] = xt
            track3d[t, :, 1] = yt
            track3d[t, :, 2] = zt

        return track3d

    def track(
        self,
        video_info: VideoInfo,
        state: ProcessingState,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        depth_mode: str = "constant",
        base_depth: float = 2.5,
        focal_length: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes dummy tracking across the video sequence.
        
        Args:
            video_info: VideoInfo instance with video path and keypoints map.
            state: ProcessingState with ref_frame_idx and specific_keypoints.
            start_frame: Optional first frame to track.
            end_frame: Optional last frame to track.
            depth_mode: "constant" or "anthropometric".
            base_depth: Initial distance from camera in meters.
            focal_length: Optional camera focal length in pixels.
            
        Returns:
            Dictionary matching the tracking output format:
                - track2d: (T, N, 2)
                - track3d: (T, N, 3)
                - track3d_world: (T, N, 3)
                - survived_track3d: (T, N, 3)
                - visibility: (T, N)
                - confidence: (T, N)
                - ref_frame_idx: int
                - start_idx: int
                - fps: float
                - video_name: str
                - depth_mode: str
                - model_name: str
        """
        if state.ref_frame_idx is None:
            raise ValueError(f"State for {video_info.name} has no ref_frame_idx.")
        if not state.specific_keypoints or len(state.specific_keypoints) == 0:
            raise ValueError(f"State for {video_info.name} has no specific_keypoints.")

        ref_idx = int(state.ref_frame_idx)
        specific_points = np.array(state.specific_keypoints, dtype=np.float64)

        # 1. Resolve Reference Joints (RKnee and RHip)
        ref_hip, ref_knee = self._get_ref_joints(video_info, state, ref_idx)
        if ref_hip is None or ref_knee is None:
            raise ValueError(f"Could not resolve reference RKnee/RHip joints for {video_info.name} at frame {ref_idx}.")

        p_knee_ref = np.array([ref_knee[0], ref_knee[1]], dtype=np.float64)
        p_hip_ref = np.array([ref_hip[0], ref_hip[1]], dtype=np.float64)

        v_mag_ref, u_ref, u_perp_ref, alphas, betas = self.compute_relative_coordinates(
            specific_points, p_knee_ref, p_hip_ref
        )

        # 2. Determine Frame Range
        sorted_keys = sorted([int(k) for k in video_info.keypoints_map.keys()]) if video_info.keypoints_map else []
        s_idx = sorted_keys[0] if start_frame is None and sorted_keys else (start_frame if start_frame is not None else 0)
        e_idx = sorted_keys[-1] if end_frame is None and sorted_keys else (end_frame if end_frame is not None else s_idx)
        if s_idx > e_idx:
            raise ValueError(f"Invalid frame range [{s_idx}, {e_idx}].")

        frame_list = list(range(s_idx, e_idx + 1))
        t_len = len(frame_list)
        n_pts = len(specific_points)

        # 3. Collect Raw Joint Positions Across Frames
        raw_knees = []
        raw_hips = []
        is_valid = []
        raw_confs = []

        for frame_no in frame_list:
            knee, hip = self._load_frame_joints(video_info, frame_no)
            if knee is not None and hip is not None:
                raw_knees.append([knee[0], knee[1]])
                raw_hips.append([hip[0], hip[1]])
                raw_confs.append((knee[2] + hip[2]) / 2.0)
                is_valid.append(True)
            else:
                raw_knees.append([np.nan, np.nan])
                raw_hips.append([np.nan, np.nan])
                raw_confs.append(0.0)
                is_valid.append(False)

        raw_knees = np.array(raw_knees, dtype=np.float64)
        raw_hips = np.array(raw_hips, dtype=np.float64)
        raw_confs = np.array(raw_confs, dtype=np.float64)
        is_valid = np.array(is_valid, dtype=bool)

        if not np.any(is_valid):
            raise ValueError(f"No valid OpenPose hip/knee detections found for {video_info.name} in range [{s_idx}, {e_idx}].")

        # 4. Interpolate Missing Detections
        knees_interp, hips_interp = self._interpolate_joints(frame_list, raw_knees, raw_hips, is_valid)

        # 5. Transfer Points Frame-by-Frame (2D Tracking)
        track2d = np.zeros((t_len, n_pts, 2), dtype=np.float64)
        femur_lengths = np.zeros(t_len, dtype=np.float64)
        visibility = np.zeros((t_len, n_pts), dtype=np.float32)
        confidence = np.zeros((t_len, n_pts), dtype=np.float32)

        for i, frame_no in enumerate(frame_list):
            if frame_no == ref_idx:
                # Guarantee exact identity at reference frame
                track2d[i] = specific_points
                femur_lengths[i] = v_mag_ref
            else:
                pts_t, v_mag_t = self.transfer_points_to_frame(
                    alphas, betas, knees_interp[i], hips_interp[i]
                )
                track2d[i] = pts_t
                femur_lengths[i] = v_mag_t

            if is_valid[i]:
                visibility[i, :] = 1.0
                confidence[i, :] = float(raw_confs[i])
            else:
                visibility[i, :] = 0.0
                confidence[i, :] = 0.0

        # 6. Lift to 3D
        metadata = self.video_reader.get_metadata(video_info.video_path)
        width = metadata.width if metadata.width > 0 else 1920
        height = metadata.height if metadata.height > 0 else 1080
        fps = metadata.fps if metadata.fps > 0 else 30.0

        track3d = self.lift_to_3d(
            track2d=track2d,
            femur_lengths=femur_lengths,
            ref_femur_length=v_mag_ref,
            width=width,
            height=height,
            depth_mode=depth_mode,
            base_depth=base_depth,
            focal_length=focal_length
        )

        return {
            "track2d": track2d,
            "track3d": track3d,
            "track3d_world": track3d,
            "survived_track3d": track3d,
            "visibility": visibility,
            "confidence": confidence,
            "ref_frame_idx": ref_idx,
            "start_idx": s_idx,
            "fps": fps,
            "video_name": video_info.name,
            "depth_mode": depth_mode,
            "model_name": "dummy_rigid_transfer"
        }

    def save_results(self, results: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Saves tracking results to a compressed .npz archive."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            track2d=results["track2d"],
            track3d=results["track3d"],
            track3d_world=results["track3d_world"],
            survived_track3d=results["survived_track3d"],
            visibility=results["visibility"],
            confidence=results["confidence"],
            ref_frame_idx=results["ref_frame_idx"],
            start_idx=results["start_idx"],
            fps=results["fps"],
            video_name=results["video_name"],
            depth_mode=results.get("depth_mode", "constant"),
            model_name=results.get("model_name", "dummy_rigid_transfer")
        )
        logger.info(f"Saved dummy tracking results to {out}")
        return out

    # --- Internal Helpers ---

    def _get_ref_joints(
        self,
        video_info: VideoInfo,
        state: ProcessingState,
        ref_idx: int
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Resolves RHip and RKnee from state.fullbody_keypoints or from disk JSON."""
        hip = _extract_coord_conf(state.fullbody_keypoints.get("RHip"))
        knee = _extract_coord_conf(state.fullbody_keypoints.get("RKnee"))
        if hip is not None and knee is not None:
            return hip, knee

        # Fallback to loading json directly
        return self._load_frame_joints(video_info, ref_idx)

    def _load_frame_joints(
        self,
        video_info: VideoInfo,
        frame_idx: int
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Loads RHip and RKnee keypoints for a given frame index from JSON."""
        json_path_str = video_info.get_frame_keypoints_path(frame_idx)
        if not json_path_str:
            return None, None

        keypoints_map = self.pose_loader.load_frame_keypoints(Path(json_path_str))
        if not keypoints_map:
            return None, None

        hip = _extract_coord_conf(keypoints_map.get("RHip"))
        knee = _extract_coord_conf(keypoints_map.get("RKnee"))

        # Require reasonable confidence (> 0.05) and non-zero coords
        if hip and knee:
            if hip[2] > 0.05 and knee[2] > 0.05 and (hip[0] != 0 or hip[1] != 0):
                return hip, knee
        return None, None

    def _interpolate_joints(
        self,
        frames: List[int],
        knees: np.ndarray,
        hips: np.ndarray,
        is_valid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates missing knee and hip coordinates across the sequence."""
        valid_indices = np.where(is_valid)[0]
        if len(valid_indices) == len(frames):
            return knees.copy(), hips.copy()

        t_steps = np.arange(len(frames))
        valid_t = t_steps[valid_indices]

        knees_out = np.zeros_like(knees)
        hips_out = np.zeros_like(hips)

        for coord_idx in [0, 1]:
            # np.interp handles linear interpolation and constant forward/backward fill at bounds
            knees_out[:, coord_idx] = np.interp(t_steps, valid_t, knees[valid_indices, coord_idx])
            hips_out[:, coord_idx] = np.interp(t_steps, valid_t, hips[valid_indices, coord_idx])

        return knees_out, hips_out


def main():
    parser = argparse.ArgumentParser(description="Run Dummy Tracking Baseline on Gait Dataset.")
    parser.add_argument("--video", type=str, default="P1_E1T1_45_ALT.mp4", help="Video file name.")
    parser.add_argument("--data_dir", type=str, default="data", help="Data base directory.")
    parser.add_argument("--depth_mode", type=str, default="constant", choices=["constant", "anthropometric"])
    parser.add_argument("--output_dir", type=str, default="data/inference/dummy", help="Output directory.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames centered around ref_frame.")
    args = parser.parse_args()

    scanner = DatasetScanner(args.data_dir)
    target_video = None
    for v in scanner:
        if v.name == args.video:
            target_video = v
            break

    if not target_video:
        print(f"Video {args.video} not found in dataset scanner.")
        return

    state = videos_state_manager.load(target_video.name)
    if state.ref_frame_idx is None or len(state.specific_keypoints) < 3:
        print(f"Video {args.video} has invalid or incomplete state.")
        return

    s_idx, e_idx = None, None
    if args.max_frames is not None:
        ref_idx = state.ref_frame_idx
        half = (args.max_frames - 1) // 2
        s_idx = max(0, ref_idx - half)
        e_idx = s_idx + args.max_frames - 1

    tracker = DummyTracker()
    results = tracker.track(
        video_info=target_video,
        state=state,
        start_frame=s_idx,
        end_frame=e_idx,
        depth_mode=args.depth_mode
    )

    out_file = Path(args.output_dir) / f"dummy_results_{target_video.name}.npz"
    tracker.save_results(results, out_file)
    print(f"Tracking completed successfully: {out_file} (Frames: {len(results['track2d'])})")


if __name__ == "__main__":
    main()
