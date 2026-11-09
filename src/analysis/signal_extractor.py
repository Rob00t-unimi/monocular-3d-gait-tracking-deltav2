"""
Signal extraction from refined (and raw) DELTAv2 NPZ tracking results.

Replicates the exact signal-processing pipeline used in
experiments/delta_test/vision_vs_gt_comparison.py so that the
feature/analysis branch produces identical numbers.

Output signals (all on the video timeline ``t_vis``):
  - accX, accY, accZ  : second-order derivative of centroid position
                        (Savitzky-Golay by default, or np.gradient).
  - Roll, Pitch, Yaw  : Best-Fit Plane orientation via SVD on the
                        survived-track point cloud.

All signals are band-pass filtered [lowcut, highcut] Hz (Butterworth
zero-phase, order 4) *after* derivation, matching the dashboard default
state ``filter_post='Both'``.
"""

from __future__ import annotations

import json
import logging
import numpy as np

logger = logging.getLogger(__name__)
from pathlib import Path
from scipy import signal
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers (ported verbatim from experiments/delta_test/vision_vs_gt_comparison.py)
# ---------------------------------------------------------------------------

def _extract_rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Extract Roll, Pitch, Yaw (degrees) from a 3x3 rotation matrix."""
    p = np.degrees(np.arcsin(np.clip(R[1, 2], -0.99, 0.99)))
    r = np.degrees(np.arctan2(-R[0, 2], R[2, 2]))
    y = np.degrees(np.arctan2(-R[1, 0], R[1, 1]))
    return r, p, y


def _calculate_triplet_rpy(triplet: np.ndarray) -> np.ndarray:
    """
    Compute Roll, Pitch, Yaw from a 3-point triplet (P1=hip/top, P2=knee_ext, P3=knee_int).
    Matches get_vision_rotation_matrices() from experiments.
    """
    p1, p2, p3 = triplet[:, 0, :], triplet[:, 1, :], triplet[:, 2, :]
    T = p1.shape[0]
    rpy = []
    for t in range(T):
        y_axis = (p2[t] + p3[t]) / 2.0 - p1[t]
        y_axis /= np.linalg.norm(y_axis) + 1e-6
        v_lat = p3[t] - p2[t]
        x_axis = v_lat - np.dot(v_lat, y_axis) * y_axis
        x_axis /= np.linalg.norm(x_axis) + 1e-6
        z_axis = np.cross(x_axis, y_axis)
        R = np.column_stack((x_axis, y_axis, z_axis))
        rpy.append(_extract_rpy_from_matrix(R))
    return np.array(rpy)


def _calculate_best_fit_rpy(points: np.ndarray) -> np.ndarray:
    """
    Compute Roll-Pitch-Yaw for each frame via SVD Best-Fit Plane on the
    survived-track point cloud.

    Args:
        points: (T, N, 3) array of 3-D point positions.

    Returns:
        rpy: (T, 3) array of [Roll, Pitch, Yaw] in degrees.
    """
    T = points.shape[0]
    rpy = []
    for t in range(T):
        pts = points[t]  # (N, 3)
        centroid = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[2, :]
        if normal[2] < 0:
            normal = -normal
        y_axis = vh[0, :]
        y_axis /= np.linalg.norm(y_axis) + 1e-6
        x_axis = np.cross(y_axis, normal)
        x_axis /= np.linalg.norm(x_axis) + 1e-6
        y_axis = np.cross(normal, x_axis)
        R = np.column_stack((x_axis, y_axis, normal))
        rpy.append(_extract_rpy_from_matrix(R))
    return np.array(rpy)  # (T, 3)


def _apply_bandpass(
    data: np.ndarray,
    fps: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter along axis 0.

    Args:
        data:    (T, ...) array.
        fps:     Sampling frequency (frames per second).
        lowcut:  Low-cut frequency in Hz.
        highcut: High-cut frequency in Hz.
        order:   Filter order (default 4).

    Returns:
        Filtered array, same shape as ``data``.
    """
    nyq = 0.5 * fps
    low = lowcut / nyq
    high = min(highcut, nyq * 0.99) / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = signal.filtfilt(b, a, data[:, i])
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_signals(
    npz_path: Path | str,
    lowcut: float = 0.5,
    highcut: float = 2.9296875,
    acc_method: str = "savgol",
    source: str = "triplet",
    filter_post: bool = True,
) -> dict:
    """
    Extract kinematic signals from a *refined* DELTAv2 NPZ file.

    The pipeline is:
      1. Load ``track3d`` (T, 3, 3) and ``survived_track3d`` (T, M, 3).
      2. High-pass filter at ``lowcut`` to remove drift.
      3. Compute centroid acceleration:
         - if source == 'triplet': mean of optimal 3 points (track3d).
         - if source == 'cloud': mean of all survived thigh points (survived_track3d).
      4. Compute RPY orientation:
         - if source == 'triplet': geometric triangulation via _calculate_triplet_rpy.
         - if source == 'cloud': SVD Best-Fit Plane via _calculate_best_fit_rpy.
      5. Band-pass [lowcut, highcut] both signals.

    Args:
        npz_path:    Path to ``delta_v2_refined_<video>.npz``.
        lowcut:      Low-cut frequency in Hz (default from cutoff_freqs.json).
        highcut:     High-cut frequency in Hz (default from cutoff_freqs.json).
        acc_method:  ``'savgol'`` (Savitzky-Golay 2nd derivative) or ``'gradient'``.
        source:      ``'triplet'`` (optimal 3 points, default) or ``'cloud'`` (all survived points).
        filter_post: Whether to apply the band-pass after derivation / RPY.

    Returns:
        dict with keys: ``t_vis``, ``accX``, ``accY``, ``accZ``,
        ``Roll``, ``Pitch``, ``Yaw``, ``fps``, ``start_idx``, ``ref_frame_idx``, ``T``.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    triplet = data["track3d"]             # (T, 3, 3)
    survived = data.get("survived_track3d", triplet)  # (T, M, 3)
    fps = float(data.get("fps", 30.0))
    ref_frame_idx = int(data.get("ref_frame_idx", 0))
    start_idx = int(data.get("start_idx", 0))

    T = triplet.shape[0]
    t_vis = (start_idx + np.arange(T)) / fps

    # --- Step 1: high-pass to remove slow drift ---
    nyq = 0.5 * fps
    b_hp, a_hp = signal.butter(4, lowcut / nyq, btype="high")

    def _hp(arr: np.ndarray) -> np.ndarray:
        """Filter each point×dimension independently along the time axis (axis 0)."""
        orig_shape = arr.shape
        flat = arr.reshape(orig_shape[0], -1)  # (T, N*3)
        out = np.empty_like(flat)
        for col in range(flat.shape[1]):
            out[:, col] = signal.filtfilt(b_hp, a_hp, flat[:, col])
        return out.reshape(orig_shape)

    surv_filt = _hp(survived)   # (T, M, 3)
    trip_filt = _hp(triplet)    # (T, 3, 3)

    # --- Step 2 & 3: centroid position & acceleration ---
    dt = 1.0 / fps
    if source == "triplet":
        centroid = np.mean(trip_filt, axis=1)
        rpy_raw = _calculate_triplet_rpy(trip_filt)
    else:
        centroid = np.mean(surv_filt, axis=1)
        rpy_raw = _calculate_best_fit_rpy(surv_filt)

    if acc_method == "savgol":
        acc_raw = signal.savgol_filter(centroid, 11, 3, deriv=2, delta=dt, axis=0)
    else:
        acc_raw = np.gradient(np.gradient(centroid, dt, axis=0), dt, axis=0)

    # Unwrap angles
    for i in range(3):
        rpy_raw[:, i] = np.degrees(np.unwrap(np.radians(rpy_raw[:, i])))

    # --- Step 5: band-pass both signals ---
    if filter_post:
        acc = _apply_bandpass(acc_raw, fps, lowcut, highcut)
        rpy = _apply_bandpass(rpy_raw, fps, lowcut, highcut)
    else:
        acc = acc_raw
        rpy = rpy_raw

    return {
        "t_vis": t_vis,
        "accX": acc[:, 0],
        "accY": acc[:, 1],
        "accZ": acc[:, 2],
        "Roll": rpy[:, 0],
        "Pitch": rpy[:, 1],
        "Yaw": rpy[:, 2],
        "fps": fps,
        "start_idx": start_idx,
        "ref_frame_idx": ref_frame_idx,
        "T": T,
        "points_3d": trip_filt if source == "triplet" else surv_filt,
    }


def extract_raw_signals(
    npz_path: Path | str,
    lowcut: float = 0.5,
    highcut: float = 2.9296875,
    acc_method: str = "savgol",
    source: str = "triplet",
    filter_post: bool = True,
) -> dict:
    """
    Extract kinematic signals from a *raw* DELTAv2 NPZ file.

    By default (source='triplet'), uses the original 3 reference points (P1, P2, P3)
    annotated in the reference frame (indices 0, 1, 2 of track3d_world).

    Args:
        npz_path:    Path to ``delta_v2_raw_<video>.npz``.
        lowcut:      Low-cut frequency in Hz.
        highcut:     High-cut frequency in Hz.
        acc_method:  ``'savgol'`` or ``'gradient'``.
        source:      ``'triplet'`` (original 3 points, default) or ``'cloud'`` (first 203 pts).
        filter_post: Whether to apply band-pass.

    Returns:
        dict with kinematic signals and metadata.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    track3d = data["track3d_world"]  # (T, N, 3)
    fps = float(data.get("fps", 30.0))
    ref_frame_idx = int(data.get("ref_frame_idx", 0))
    start_idx = int(data.get("start_idx", 0))

    T = track3d.shape[0]
    t_vis = (start_idx + np.arange(T)) / fps

    nyq = 0.5 * fps
    b_hp, a_hp = signal.butter(4, lowcut / nyq, btype="high")

    def _hp(arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        flat = arr.reshape(orig_shape[0], -1)
        out = np.empty_like(flat)
        for col in range(flat.shape[1]):
            out[:, col] = signal.filtfilt(b_hp, a_hp, flat[:, col])
        return out.reshape(orig_shape)

    if source == "triplet":
        pts = track3d[:, :3, :]  # Original 3 reference keypoints
        pts_filt = _hp(pts)
        centroid = np.mean(pts_filt, axis=1)
        rpy_raw = _calculate_triplet_rpy(pts_filt)
    else:
        n_thigh = min(203, track3d.shape[1])
        pts = track3d[:, :n_thigh, :]
        pts_filt = _hp(pts)
        centroid = np.mean(pts_filt, axis=1)
        rpy_raw = _calculate_best_fit_rpy(pts_filt)

    dt = 1.0 / fps
    if acc_method == "savgol":
        acc_raw = signal.savgol_filter(centroid, 11, 3, deriv=2, delta=dt, axis=0)
    else:
        acc_raw = np.gradient(np.gradient(centroid, dt, axis=0), dt, axis=0)

    for i in range(3):
        rpy_raw[:, i] = np.degrees(np.unwrap(np.radians(rpy_raw[:, i])))

    if filter_post:
        acc = _apply_bandpass(acc_raw, fps, lowcut, highcut)
        rpy = _apply_bandpass(rpy_raw, fps, lowcut, highcut)
    else:
        acc = acc_raw
        rpy = rpy_raw

    return {
        "t_vis": t_vis,
        "accX": acc[:, 0],
        "accY": acc[:, 1],
        "accZ": acc[:, 2],
        "Roll": rpy[:, 0],
        "Pitch": rpy[:, 1],
        "Yaw": rpy[:, 2],
        "fps": fps,
        "start_idx": start_idx,
        "ref_frame_idx": ref_frame_idx,
        "T": T,
        "points_3d": pts_filt,
    }


def load_cutoffs(project_root: Path | str | None = None) -> tuple[float, float]:
    """
    Load (lowcut, highcut) from ``data/gt_cache/cutoff_freqs.json``.

    Falls back to (0.5, 2.9296875) if the file is not found.

    Args:
        project_root: Root of the repository. If None, uses the parent of
                      ``src/`` (i.e. two levels up from this file).

    Returns:
        (lowcut, highcut) in Hz.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    cutoff_path = Path(project_root) / "data" / "gt_cache" / "cutoff_freqs.json"
    if cutoff_path.exists():
        with open(cutoff_path) as f:
            d = json.load(f)
        return float(d["lowcut"]), float(d["highcut"])
    return 0.5, 2.9296875


def extract_aligned_signals(
    npz_path: Path | str,
    lowcut: float = 0.5,
    highcut: float = 2.9296875,
    source: str = "triplet",
    project_root: Optional[Path | str] = None,
) -> Optional[dict]:
    """
    Extracts kinematic signals and aligns them directly to the IMU Ground Truth
    in the shared static World Coordinate System via kinematic_alignment.

    Args:
        npz_path: Path to refined or raw tracking NPZ file.
        lowcut: Butterworth low-cut frequency (Hz).
        highcut: Butterworth high-cut frequency (Hz).
        source: 'triplet' (optimal 3 points) or 'cloud' (all survived thigh points).
        project_root: Repository root path (optional).

    Returns:
        Dictionary with synchronized, world-aligned signals (accX, accY, accZ, Roll, Pitch, Yaw)
        and matching GT world signals, or None if loading fails.
    """
    from src.analysis.gt_loader import load_raw_imu
    from src.analysis.kinematic_alignment import run_pipeline

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    # 1. Select points
    if "track3d" in data:
        # Refined file
        triplet = data["track3d"]
        survived = data.get("survived_track3d", triplet)
        pts = triplet if source == "triplet" else survived
    elif "track3d_world" in data:
        # Raw file
        track = data["track3d_world"]
        n_pts = 3 if source == "triplet" else min(203, track.shape[1])
        pts = track[:, :n_pts, :]
    else:
        logger.error(f"Unrecognized NPZ format in {npz_path}")
        return None

    fps = float(data.get("fps", 30.0))
    start_idx = int(data.get("start_idx", 0))
    ref_frame_idx = int(data.get("ref_frame_idx", 0))
    ref_idx = max(0, ref_frame_idx - start_idx)

    T = pts.shape[0]
    t_vis = (start_idx + np.arange(T)) / fps

    video_name = str(data.get("video_name", npz_path.stem.replace("delta_v2_refined_", "").replace("delta_v2_raw_", "")))

    # 2. Load matching raw IMU data
    df_imu = load_raw_imu(video_name, project_root)
    if df_imu is None or len(df_imu) < 10:
        logger.warning(f"No valid raw IMU data found for {video_name}. Falling back to standard extraction.")
        base = extract_signals(npz_path, lowcut, highcut, source=source) if "track3d" in data else extract_raw_signals(npz_path, lowcut, highcut, source=source)
        return base

    # 3. Interpolate IMU to video rate across the video window (with 4s safety margin)
    t_margin = 4.0
    t_start_safe = max(0.0, t_vis[0] - t_margin)
    t_end_safe = min(df_imu["Timestamp"].iloc[-1], t_vis[-1] + t_margin)
    t_eval_imu = np.arange(t_start_safe, t_end_safe, 1.0 / fps)

    from scipy import interpolate
    f_acc = interpolate.interp1d(df_imu["Timestamp"], df_imu[["accX", "accY", "accZ"]].values, axis=0, fill_value="extrapolate")
    f_quat = interpolate.interp1d(df_imu["Timestamp"], df_imu[["q0", "q1", "q2", "q3"]].values, axis=0, fill_value="extrapolate")
    imu_acc_interp = f_acc(t_eval_imu)
    imu_quat_interp = f_quat(t_eval_imu)
    # Renormalize interpolated quaternions
    imu_quat_interp /= (np.linalg.norm(imu_quat_interp, axis=-1, keepdims=True) + 1e-9)

    # 4. Run kinematic alignment pipeline
    pipeline_res = run_pipeline(
        video_pts=pts,
        imu_acc=imu_acc_interp,
        imu_quat=imu_quat_interp,
        fs=fps,
        ref_idx=min(ref_idx, len(pts) - 1),
        scalar_first=True,
        lowcut=lowcut,
        highcut=highcut,
    )

    N_out = len(pipeline_res["acc_xyz_tracking"])
    sv = int(pipeline_res.get("start_vis", 0))
    ev = int(pipeline_res.get("end_vis", sv + N_out))
    t_vis_out = t_vis[sv:ev]

    return {
        "t_vis": t_vis_out,
        "accX": pipeline_res["acc_xyz_tracking"][:, 0],
        "accY": pipeline_res["acc_xyz_tracking"][:, 1],
        "accZ": pipeline_res["acc_xyz_tracking"][:, 2],
        "Roll": pipeline_res["rpy_tracking"][:, 0],
        "Pitch": pipeline_res["rpy_tracking"][:, 1],
        "Yaw": pipeline_res["rpy_tracking"][:, 2],
        "gt_accX": pipeline_res["acc_xyz_sensor"][:, 0],
        "gt_accY": pipeline_res["acc_xyz_sensor"][:, 1],
        "gt_accZ": pipeline_res["acc_xyz_sensor"][:, 2],
        "gt_Roll": pipeline_res["rpy_sensor"][:, 0],
        "gt_Pitch": pipeline_res["rpy_sensor"][:, 1],
        "gt_Yaw": pipeline_res["rpy_sensor"][:, 2],
        "fps": fps,
        "start_idx": start_idx,
        "ref_frame_idx": ref_frame_idx,
        "T": N_out,
        "points_3d": pts[sv:ev],
        "temporal_lag": pipeline_res["temporal_lag"],
        "correlations": pipeline_res["pearson_correlations"],
        "R_cam_to_world": pipeline_res["R_cam_to_world"],
        "R_axes_align": pipeline_res["R_axes_align"],
    }
