"""
Module for identifying the optimal reference frame for gait tracking initialization.
Uses anatomical distance peaks and strategy patterns for different body parts.
"""

import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from typing import Dict, List, Optional, Any

from src.core.models import VideoInfo, VideoMetadata
from src.data_io.openpose_reader import BODY_25_MAP, openpose_reader

logger = logging.getLogger(__name__)

class SearchStrategy:
    def execute(self, finder, video_info: VideoInfo, metadata: VideoMetadata, keypoint_names: list, min_conf: float, tolerance: float):
        raise NotImplementedError

class BasicPresenceStrategy(SearchStrategy):
    def execute(self, finder, video_info: VideoInfo, metadata: VideoMetadata, keypoint_names: list, min_conf: float, tolerance: float):
        return finder._strategy_basic_presence(video_info, metadata, keypoint_names, min_conf)

class LegSearchStrategy(SearchStrategy):
    def execute(self, finder, video_info: VideoInfo, metadata: VideoMetadata, keypoint_names: list, min_conf: float, tolerance: float):
        required_kps = list(set(keypoint_names + ["RHip", "LHip", "RKnee", "LKnee", "RWrist", "LWrist"]))
        signals = finder._get_smoothed_signals(video_info, metadata, required_kps, min_conf=min_conf, apply_smoothing=False)

        peaks = finder._find_leg_spread_peaks(signals, metadata.frames_num)
        best_frame = finder._evaluate_peaks_chronologically(peaks, signals, metadata.frames_num, metadata.width)

        if best_frame is not None:
            kp_array = finder._get_keypoints_array_for_frame(video_info, int(best_frame))
            frame_kps = finder._extract_frame_kps(kp_array, keypoint_names, 0.0)
            return {"ref_frame_idx": int(best_frame), "keypoints": frame_kps}

        return False

class RefFrameFinder:
    """
    Finds the best frame index to start tracking, based on visibility and gait phase.
    """
    def __init__(self):
        self._search_strategies = {
            "leg": LegSearchStrategy(),
            "arm": BasicPresenceStrategy(),
            "head": BasicPresenceStrategy()
        }

    def find_specific_keypoints_ref(self, video_info: VideoInfo, metadata: VideoMetadata,
                                    confidence_threshold=0.2, check_overlaps=True, overlap_tolerance=20) -> dict:
        """
        Public entry point to find the reference frame.
        """
        keypoint_names = video_info.interests_keypoints

        # 1. Special cases (Front/Rear)
        res = self._front_rear_case(metadata, video_info, keypoint_names, confidence_threshold)
        if res: return res

        # 2. Strategy execution
        strategy = self._search_strategies.get(video_info.sensor_position, BasicPresenceStrategy())
        if strategy and check_overlaps:
            res = strategy.execute(self, video_info, metadata, keypoint_names, confidence_threshold, overlap_tolerance)
            if res: return res
        
        # 3. Fallbacks
        res = self._a_visible_keypoint_fallback(metadata, video_info, keypoint_names, confidence_threshold)
        if res: return res

        return {"ref_frame_idx": 0, "keypoints": {}}

    def _strategy_basic_presence(self, video_info, metadata, keypoint_names, min_conf):
        for frame_idx in range(metadata.frames_num):
            kp_array = self._get_keypoints_array_for_frame(video_info, frame_idx)
            if kp_array is None: continue
            frame_kps = self._extract_frame_kps(kp_array, keypoint_names, min_conf)
            if len(frame_kps) == len(keypoint_names):
                return {"ref_frame_idx": frame_idx, "keypoints": frame_kps}
        return False

    def _find_leg_spread_peaks(self, signals, frames_num):
        stride_wave = np.zeros(frames_num)
        for i in range(frames_num):
            rk_x, lk_x = signals["RKnee"]['x'][i], signals["LKnee"]['x'][i]
            stride_wave[i] = abs(rk_x - lk_x) if not np.isnan(rk_x) and not np.isnan(lk_x) else 0.0
        peaks, _ = find_peaks(stride_wave, distance=15, prominence=10)
        return peaks

    def _evaluate_peaks_chronologically(self, peaks, signals, frames_num, width, min_margin=20.0):
        def is_frame_valid(frame_idx, check_arm):
            def get_keypoint_xy(kp):
                if kp not in signals or np.isnan(signals[kp]['x'][frame_idx]): return None
                return (signals[kp]['x'][frame_idx], signals[kp]['y'][frame_idx])

            rk, lk, rh, lh = get_keypoint_xy("RKnee"), get_keypoint_xy("LKnee"), get_keypoint_xy("RHip"), get_keypoint_xy("LHip")
            if not (rk and lk and rh and lh): return False
            if rk[0] < min_margin or rk[0] > (width - min_margin): return False
            if self._segments_intersect(rh, rk, lh, lk): return False
            return True

        for peak in peaks:
            if is_frame_valid(peak, check_arm=False): return peak
        return None

    def _get_smoothed_signals(self, video_info, metadata, keypoints_list, min_conf=0.4, apply_smoothing=False):
        frames_num = metadata.frames_num
        raw = {kp: {'x': np.zeros(frames_num), 'y': np.zeros(frames_num)} for kp in keypoints_list}
        for i in range(frames_num):
            arr = self._get_keypoints_array_for_frame(video_info, i)
            for kp in keypoints_list:
                val = self._get_kp(kp, arr, min_conf) if arr is not None else None
                raw[kp]['x'][i], raw[kp]['y'][i] = val if val else (np.nan, np.nan)
        return raw

    def _extract_frame_kps(self, kp_array, keypoint_names, min_conf):
        extracted = {}
        if kp_array is None: return extracted
        for name in keypoint_names:
            kp = self._get_kp(name, kp_array, min_conf)
            if kp:
                idx = BODY_25_MAP[name] * 3
                extracted[name] = {"x": float(kp[0]), "y": float(kp[1]), "confidence": float(kp_array[idx+2])}
        return extracted

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _segments_intersect(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)

    def _get_kp(self, name, array, threshold):
        if name not in BODY_25_MAP: return None
        idx = BODY_25_MAP[name] * 3
        if idx + 2 < len(array) and array[idx+2] > threshold:
            return (array[idx], array[idx+1])
        return None

    def _get_keypoints_array_for_frame(self, video_info, frame_idx):
        path = video_info.get_frame_keypoints_path(frame_idx)
        if not path: return None
        kps_dict = openpose_reader.load_frame_keypoints(Path(path))
        # Flatten back to array for internal methods if needed or refactor methods
        # For now, reconstruct a flat array to keep internal logic unchanged
        arr = np.zeros(25 * 3)
        for name, vals in kps_dict.items():
            idx = BODY_25_MAP[name] * 3
            arr[idx:idx+3] = vals
        return arr

    def _front_rear_case(self, metadata, video_info, keypoint_names, min_conf):
        angle = video_info.angle.lower()
        idx = (metadata.frames_num - 1) if "rear" in angle else 0 if "front" in angle else None
        if idx is not None:
            arr = self._get_keypoints_array_for_frame(video_info, idx)
            return {"ref_frame_idx": idx, "keypoints": self._extract_frame_kps(arr, keypoint_names, min_conf)}
        return False

    def _a_visible_keypoint_fallback(self, metadata, video_info, keypoint_names, min_conf):
        for i in range(metadata.frames_num):
            arr = self._get_keypoints_array_for_frame(video_info, i)
            extracted = self._extract_frame_kps(arr, keypoint_names, min_conf)
            if extracted: return {"ref_frame_idx": i, "keypoints": extracted}
        return False

ref_frame_finder = RefFrameFinder()
