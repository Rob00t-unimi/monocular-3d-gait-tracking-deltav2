"""
Kinematic Alignment & Coordinate Frame Synchronization Module.

Implements the complete physical and mathematical pipeline for aligning 3D vision-based
point tracking (Camera Frame) with Inertial Measurement Unit (IMU) Ground Truth (Sensor Frame)
within a shared static World Coordinate System:

1. Extraction of 3D centroid translational acceleration and twist-stabilized 3D orientation
   (with local tangent-plane depth regularization for narrow-base thigh triplets and
   temporal sign continuity for SVD point cloud normals).
2. Projection of IMU specific force to the static World Frame and gravity depuration.
3. Zero-phase Butterworth bandpass filtering across translational and rotational channels.
4. Bounded, overlap-normalized multi-modal temporal synchronization.
5. Unsupervised Principal Component anatomical axis calibration for IMU orientation
   and reference-anchored Lie-algebra so(3) spatial alignment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import ndimage, signal
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# 1. Extraction of Video Kinematics (Camera Frame)
# ---------------------------------------------------------------------------

def extract_tracking_kinematics(
    pts: np.ndarray,
    fs: float,
    acc_method: str = "savgol",
    triplet_dz_scale: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes 3D centroid position, linear acceleration, and twist-stabilized 3D orientation
    from a tracked 3-point rigid triplet (T, 3, 3) or N-point cloud (T, N, 3) in Camera space.

    Mathematical Note on Narrow-Base Triplet Normal Stabilization:
        A 3-point patch on the thigh typically spans ~3-6 cm with a triangle altitude
        (lever arm h perpendicular to the primary edge u) of ~1.5-2.4 cm. Unregularized
        monocular depth jitter delta_z of 1.5-2.0 cm between vertices induces parasitic
        normal twist arctan(delta_z / h) ~ 45-90 deg. By attenuating only the internal
        relative depth offset (z_i - z_mean) via `triplet_dz_scale` while preserving 100%
        of the 3D centroid depth z_mean(t), translational 3D accelerations remain exact
        while out-of-plane normal twist is suppressed.

    Args:
        pts: (T, 3, 3) or (T, N, 3) array of 3D points in Camera coordinates.
        fs: Sampling frequency of the video (frames per second).
        acc_method: Acceleration derivative method ('savgol' or 'gradient').
        triplet_dz_scale: Attenuation factor for internal relative Z jitter in 3-point triplets.

    Returns:
        dict containing:
          - 'centroid': (T, 3) position of the centroid
          - 'acc_cam': (T, 3) raw acceleration in Camera frame
          - 'R_cam': scipy.spatial.transform.Rotation object representing orientation
          - 'quats_cam': (T, 4) quaternions [x, y, z, w]
          - 'rpy_cam': (T, 3) Euler angles (xyz, degrees)
    """
    T = pts.shape[0]
    dt = 1.0 / fs

    # 1. Centroid position (preserves 100% of full 3D metric coordinates)
    centroid = np.mean(pts, axis=1)  # (T, 3)

    # 2. Centroid acceleration (Camera frame)
    if acc_method == "savgol":
        w_len = min(11, T if T % 2 != 0 else T - 1)
        w_len = max(w_len, 5)
        acc_cam = signal.savgol_filter(
            centroid, window_length=w_len, polyorder=2, deriv=2, delta=dt, axis=0
        )
    else:
        acc_cam = np.gradient(np.gradient(centroid, dt, axis=0), dt, axis=0)

    # 3. Orientation Triangulation / Best-Fit Plane with Sign-Continuity & Twist Regularization
    pts_ori = pts.copy()
    if pts_ori.shape[1] == 3:
        z_mean = np.mean(pts_ori[:, :, 2], axis=1, keepdims=True)
        pts_ori[:, :, 2] = z_mean + triplet_dz_scale * (pts_ori[:, :, 2] - z_mean)

        p1, p2, p3 = pts_ori[:, 0, :], pts_ori[:, 1, :], pts_ori[:, 2, :]
        u = (p2 - p1) / (np.linalg.norm(p2 - p1, axis=-1, keepdims=True) + 1e-9)
        w_raw = np.cross(u, p3 - p1)
        w = w_raw / (np.linalg.norm(w_raw, axis=-1, keepdims=True) + 1e-9)
        for t in range(1, T):
            if np.dot(w[t], w[t - 1]) < 0:
                w[t] = -w[t]
        v = np.cross(w, u)
        R_matrices = np.stack([u, v, w], axis=-1)  # (T, 3, 3)
    else:
        R_matrices = np.empty((T, 3, 3))
        for t in range(T):
            p_t = pts_ori[t]
            c_t = np.mean(p_t, axis=0)
            _, _, vh = np.linalg.svd(p_t - c_t, full_matrices=False)
            normal = vh[2, :]
            if normal[2] < 0:
                normal = -normal
            y_axis = vh[0, :] / (np.linalg.norm(vh[0, :]) + 1e-9)
            x_axis = np.cross(y_axis, normal)
            x_axis /= (np.linalg.norm(x_axis) + 1e-9)
            y_axis = np.cross(normal, x_axis)
            R_matrices[t] = np.column_stack((x_axis, y_axis, normal))

    rot_cam = R.from_matrix(R_matrices)
    quats_cam = rot_cam.as_quat()
    rpy_cam = rot_cam.as_euler("xyz", degrees=True)

    return {
        "centroid": centroid,
        "acc_cam": acc_cam,
        "R_cam": rot_cam,
        "quats_cam": quats_cam,
        "rpy_cam": rpy_cam,
    }


# ---------------------------------------------------------------------------
# 2. Conversion of IMU to Static World Frame & Gravity Depuration
# ---------------------------------------------------------------------------

def convert_imu_to_world(
    acc_body: np.ndarray,
    quats: Union[np.ndarray, R],
    scalar_first: bool = False,
) -> Tuple[np.ndarray, R]:
    """
    Converts raw IMU body-frame acceleration into the static World frame and
    depurates the gravity component.

    Args:
        acc_body: (T, 3) accelerometer specific force in sensor body frame.
        quats: (T, 4) sensor orientation quaternions or scipy Rotation object.
        scalar_first: If True, quaternions are ordered [w, x, y, z]; otherwise [x, y, z, w].

    Returns:
        acc_world_free: (T, 3) gravity-free linear acceleration in World frame.
        rot_world: scipy.spatial.transform.Rotation object of sensor in World frame.
    """
    if isinstance(quats, R):
        rot_world = quats
    else:
        q = np.asarray(quats, dtype=np.float64)
        if scalar_first:
            q = q[:, [1, 2, 3, 0]]
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        q = q / norms
        rot_world = R.from_quat(q)

    # Rotate specific force from sensor body frame to static World frame
    acc_world_raw = rot_world.apply(acc_body)

    # Depurate static gravity vector (quasi-constant velocity assumption over trial)
    gravity_vec = np.mean(acc_world_raw, axis=0, keepdims=True)
    acc_world_free = acc_world_raw - gravity_vec

    return acc_world_free, rot_world


# ---------------------------------------------------------------------------
# 3. Zero-Phase Butterworth Bandpass Filtering
# ---------------------------------------------------------------------------

def filter_signals(
    signals_dict: Dict[str, np.ndarray],
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 3.0,
    order: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Applies a zero-phase forward-backward Butterworth bandpass filter (`filtfilt`)
    to all signals in `signals_dict`.

    Args:
        signals_dict: Mapping of signal names to (T, ...) numpy arrays.
        fs: Sampling rate in Hz.
        lowcut: High-pass cutoff frequency in Hz (removes DC drift).
        highcut: Low-pass cutoff frequency in Hz (removes high-frequency noise).
        order: Butterworth filter order.

    Returns:
        Dictionary with filtered signals of identical shapes.
    """
    nyq = 0.5 * fs
    low = max(0.001, min(lowcut / nyq, 0.98))
    high = max(low + 0.001, min(highcut / nyq, 0.99))
    b, a = signal.butter(order, [low, high], btype="band")

    filtered: Dict[str, np.ndarray] = {}
    padlen = 3 * max(len(a), len(b))
    for k, arr in signals_dict.items():
        if len(arr) > padlen:
            filtered[k] = signal.filtfilt(b, a, arr, axis=0)
        else:
            filtered[k] = arr - np.mean(arr, axis=0, keepdims=True)
    return filtered


# ---------------------------------------------------------------------------
# 4. Bounded Overlap-Normalized Multi-Modal Temporal Alignment
# ---------------------------------------------------------------------------

def temporal_align(
    gt_rot: R,
    tracking_rot: R,
    ref_idx: int = 0,
    max_lag_samples: Optional[int] = None,
    gt_acc: Optional[np.ndarray] = None,
    tracking_acc: Optional[np.ndarray] = None,
) -> int:
    """
    Computes optimal temporal lag between GT and Vision tracking using bounded,
    overlap-normalized cross-correlation of rotation-invariant angular excursion
    (and optional 3D acceleration magnitude).

    Args:
        gt_rot: scipy Rotation object of GT orientations (T_gt,).
        tracking_rot: scipy Rotation object of Video orientations (T_vis,).
        ref_idx: Reference frame index to anchor relative excursion.
        max_lag_samples: Maximum allowable lag in samples. If None, defaults to
                         min(90, 45% of min sequence length) to prevent cycle-skipping
                         and excessive frame truncation.
        gt_acc: Optional (T_gt, 3) filtered GT acceleration in World frame.
        tracking_acc: Optional (T_vis, 3) filtered tracking acceleration in Camera frame.

    Returns:
        best_lag: Integer sample shift. If positive, video lags GT.
    """
    n_vis = len(tracking_rot)
    n_gt = len(gt_rot)
    ref_idx_gt = max(0, min(ref_idx, n_gt - 1))
    ref_idx_vis = max(0, min(ref_idx, n_vis - 1))

    R_rel_gt = gt_rot[ref_idx_gt].inv() * gt_rot
    R_rel_vis = tracking_rot[ref_idx_vis].inv() * tracking_rot

    theta_gt = np.linalg.norm(R_rel_gt.as_rotvec(), axis=-1)
    theta_vis = np.linalg.norm(R_rel_vis.as_rotvec(), axis=-1)

    tg_c = (theta_gt - np.mean(theta_gt)) / (np.std(theta_gt) + 1e-9)
    tv_c = (theta_vis - np.mean(theta_vis)) / (np.std(theta_vis) + 1e-9)

    has_acc = (
        gt_acc is not None
        and tracking_acc is not None
        and len(gt_acc) == n_gt
        and len(tracking_acc) == n_vis
    )
    if has_acc:
        ag_norm = np.linalg.norm(gt_acc, axis=-1)
        av_norm = np.linalg.norm(tracking_acc, axis=-1)
        ag_c = (ag_norm - np.mean(ag_norm)) / (np.std(ag_norm) + 1e-9)
        av_c = (av_norm - np.mean(av_norm)) / (np.std(av_norm) + 1e-9)

    if max_lag_samples is None:
        max_lag = min(90, max(15, int(0.45 * min(n_vis, n_gt))))
    else:
        max_lag = int(max_lag_samples)

    min_overlap = min(45, max(10, min(n_vis, n_gt) // 2))
    best_score = -1e9
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        sv = max(0, lag)
        ev = min(n_vis, n_gt + lag)
        si = sv - lag
        ei = ev - lag
        if ev - sv < min_overlap:
            continue

        seg_tv = tv_c[sv:ev]
        seg_tg = tg_c[si:ei]
        if np.std(seg_tv) > 1e-6 and np.std(seg_tg) > 1e-6:
            r_th = float(np.corrcoef(seg_tv, seg_tg)[0, 1])
        else:
            r_th = 0.0

        if has_acc:
            seg_av = av_c[sv:ev]
            seg_ag = ag_c[si:ei]
            if np.std(seg_av) > 1e-6 and np.std(seg_ag) > 1e-6:
                r_ac = float(np.corrcoef(seg_av, seg_ag)[0, 1])
            else:
                r_ac = 0.0
            score = 0.70 * r_th + 0.30 * r_ac
        else:
            score = r_th

        if score > best_score:
            best_score = score
            best_lag = int(lag)

    return best_lag


# ---------------------------------------------------------------------------
# 5. Geometric Alignment (Camera -> World & Anatomical PCA Sensor Alignment)
# ---------------------------------------------------------------------------

def align_camera_to_world(
    acc_cam: np.ndarray,
    rot_cam: R,
    acc_gt_world: np.ndarray,
    rot_gt_world: R,
    ref_idx: int = 0,
    fs: float = 30.0,
    lowcut: float = 0.5,
    highcut: float = 3.0,
) -> Dict[str, Any]:
    """
    Computes 3D spatial alignment mapping Camera space onto World/IMU anatomical space.

    Key Mathematical Improvements:
      1. Despikes frame-to-frame angular velocity increments on `rotvec_cam` via a 5-point
         median filter and physical angular rate clipping (12 deg/frame = 360 deg/s) to
         prevent boundary occlusion glitches from inflating Range of Motion.
      2. Resolves arbitrary Shimmer IMU strap mounting orientation via unsupervised SVD
         Principal Component Analysis on `rotvec_gt`, placing the dominant anatomical
         sagittal flexion-extension axis onto Column 1 (`Pitch`), secondary onto Column 0
         (`Roll`), and tertiary onto Column 2 (`Yaw`).
      3. Anchors the 3D orientation alignment `R_axes_align` within a local window around
         `ref_idx` (where tracking is guaranteed to lie on the target limb), preventing
         global 180-degree sign inversions from masking subsequent Identity Swaps.

    Args:
        acc_cam: (T, 3) kinematic acceleration in Camera space.
        rot_cam: scipy Rotation object representing Camera tracking orientations.
        acc_gt_world: (T, 3) linear kinematic acceleration in World space.
        rot_gt_world: scipy Rotation object representing GT World orientations.
        ref_idx: Frame index of reference pose T_0.

    Returns:
        dict containing aligned World accelerations, anatomical RPY angles (degrees),
        rotation operators, and alignment residuals.
    """
    # 1. Acceleration alignment (Static Camera Frame -> Static World Frame)
    std_cam = np.std(acc_cam) + 1e-9
    std_gt = np.std(acc_gt_world) + 1e-9

    R_cam_to_world, rms_acc = R.align_vectors(acc_gt_world / std_gt, acc_cam / std_cam)
    acc_tracking_world = R_cam_to_world.apply(acc_cam)

    # 2. Orientation alignment in Lie Algebra so(3) with Anatomical PCA Basis
    n_samples = len(rot_cam)
    ref_idx_cam = max(0, min(ref_idx, n_samples - 1))
    ref_idx_gt = max(0, min(ref_idx, len(rot_gt_world) - 1))

    R_rel_cam = rot_cam[ref_idx_cam].inv() * rot_cam
    R_rel_gt = rot_gt_world[ref_idx_gt].inv() * rot_gt_world

    # Despike single-frame boundary occlusion glitches in camera angular velocity
    rotvec_cam_raw = R_rel_cam.as_rotvec()
    if len(rotvec_cam_raw) >= 5:
        drv = np.diff(rotvec_cam_raw, axis=0, prepend=rotvec_cam_raw[:1])
        drv = ndimage.median_filter(drv, size=(5, 1))
        max_rad_per_frame = np.radians(12.0)
        drv = np.clip(drv, -max_rad_per_frame, max_rad_per_frame)
        rotvec_cam = np.cumsum(drv, axis=0)
    else:
        rotvec_cam = rotvec_cam_raw

    rotvec_gt = R_rel_gt.as_rotvec()

    # Bandpass filter rotation vectors to isolate physiological gait band before SVD / alignment
    filt_rv = filter_signals(
        {"cam": rotvec_cam, "gt": rotvec_gt},
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=4,
    )
    rotvec_cam = filt_rv["cam"]
    rotvec_gt = filt_rv["gt"]

    # Unsupervised anatomical axis alignment of IMU via SVD (PC1 -> Pitch [col 1])
    rv_gt_centered = rotvec_gt - np.mean(rotvec_gt, axis=0, keepdims=True)
    if len(rv_gt_centered) >= 3 and np.linalg.norm(rv_gt_centered) > 1e-9:
        _, _, vh_gt = np.linalg.svd(rv_gt_centered, full_matrices=False)
        B_gt = vh_gt[[1, 0, 2], :]  # Row 0: PC2 (Roll), Row 1: PC1 (Pitch), Row 2: PC3 (Yaw)
        if np.linalg.det(B_gt) < 0:
            B_gt[2, :] = -B_gt[2, :]
        R_imu_anat = R.from_matrix(B_gt)
        rotvec_gt_anat = R_imu_anat.apply(rotvec_gt)
    else:
        rotvec_gt_anat = rotvec_gt

    # Anchor R_axes_align in the local neighborhood of ref_idx (+-45 frames)
    w_half = min(45, max(20, n_samples // 2))
    w_start = max(0, ref_idx_cam - w_half)
    w_end = min(n_samples, ref_idx_cam + w_half)
    if w_end - w_start < min(20, n_samples):
        w_start, w_end = 0, n_samples

    R_axes_align, rms_ori = R.align_vectors(
        rotvec_gt_anat[w_start:w_end],
        rotvec_cam[w_start:w_end],
    )
    rotvec_tracking_world = R_axes_align.apply(rotvec_cam)

    # 3. Express anatomical angular excursions in degrees (Roll=col 0, Pitch=col 1, Yaw=col 2)
    rpy_tracking_world = np.degrees(rotvec_tracking_world)
    rpy_gt_world = np.degrees(rotvec_gt_anat)

    return {
        "acc_tracking_world": acc_tracking_world,
        "rpy_tracking_world": rpy_tracking_world,
        "rpy_gt_world": rpy_gt_world,
        "R_cam_to_world": R_cam_to_world,
        "R_axes_align": R_axes_align,
        "rms_acc": float(rms_acc),
        "rms_ori": float(rms_ori),
    }


# ---------------------------------------------------------------------------
# 6. Main Orchestrator Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_pts: np.ndarray,
    imu_acc: np.ndarray,
    imu_quat: Union[np.ndarray, R],
    fs: float,
    ref_idx: int = 0,
    imu_gyr: Optional[np.ndarray] = None,
    scalar_first: bool = False,
    lowcut: float = 0.5,
    highcut: float = 3.0,
) -> Dict[str, Any]:
    """
    Executes the full end-to-end kinematic alignment pipeline.

    Args:
        video_pts: (T, 3, 3) or (T, N, 3) tracking points in Camera frame.
        imu_acc: (T_imu, 3) raw accelerometer readings in body frame.
        imu_quat: (T_imu, 4) quaternions or Rotation object.
        fs: Video frame rate (Hz).
        ref_idx: Reference pose index T_0.
        imu_gyr: Optional (T_imu, 3) gyroscope angular rates.
        scalar_first: True if IMU quaternions are [w, x, y, z].
        lowcut: High-pass cutoff for Butterworth bandpass (Hz).
        highcut: Low-pass cutoff for Butterworth bandpass (Hz).

    Returns:
        dict containing synchronized and aligned signals in the shared World reference frame.
    """
    # Step 1: Extract Video Kinematics
    vis_kin = extract_tracking_kinematics(video_pts, fs)

    # Step 2: Convert IMU to Static World Frame & Remove Gravity
    acc_imu_world_raw, rot_imu_world = convert_imu_to_world(
        imu_acc, imu_quat, scalar_first=scalar_first
    )

    # Step 3: Bandpass filter both signals (DC and high-frequency noise removal)
    filt_vis = filter_signals({"acc": vis_kin["acc_cam"]}, fs, lowcut, highcut)
    filt_imu = filter_signals({"acc": acc_imu_world_raw}, fs, lowcut, highcut)

    acc_cam_filt = filt_vis["acc"]
    acc_imu_filt = filt_imu["acc"]
    rot_cam = vis_kin["R_cam"]
    rot_imu = rot_imu_world

    # Step 4: Bounded Multi-Modal Temporal Alignment
    lag = temporal_align(
        rot_imu,
        rot_cam,
        ref_idx=min(ref_idx, len(rot_cam) - 1),
        gt_acc=acc_imu_filt,
        tracking_acc=acc_cam_filt,
    )

    # Slice signals to lock phase synchronization
    start_vis = max(0, lag)
    end_vis = min(len(acc_cam_filt), len(acc_imu_filt) + lag)
    start_imu = start_vis - lag
    end_imu = end_vis - lag

    if end_vis - start_vis < 30 or end_imu - start_imu < 30:
        min_len = min(len(acc_cam_filt), len(acc_imu_filt))
        start_vis, end_vis = 0, min_len
        start_imu, end_imu = 0, min_len

    a_cam_s = acc_cam_filt[start_vis:end_vis]
    r_cam_s = rot_cam[start_vis:end_vis]
    a_imu_s = acc_imu_filt[start_imu:end_imu]
    r_imu_s = rot_imu[start_imu:end_imu]

    # Adjust reference frame index to the sliced window!
    ref_idx_sliced = max(0, min(ref_idx - start_vis, len(a_cam_s) - 1))

    # Step 5: Spatial & Anatomical Alignment (Camera -> World)
    aligned_res = align_camera_to_world(
        acc_cam=a_cam_s,
        rot_cam=r_cam_s,
        acc_gt_world=a_imu_s,
        rot_gt_world=r_imu_s,
        ref_idx=ref_idx_sliced,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
    )

    acc_xyz_sensor = a_imu_s
    acc_xyz_tracking = aligned_res["acc_tracking_world"]

    # Step 6: Zero-phase Butterworth bandpass filter (order=4) RPY angles
    rpy_tracking = np.copy(aligned_res["rpy_tracking_world"])
    rpy_sensor = np.copy(aligned_res["rpy_gt_world"])

    filt_rpy = filter_signals(
        {"tracking": rpy_tracking, "sensor": rpy_sensor},
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=4,
    )
    rpy_tracking = filt_rpy["tracking"]
    rpy_sensor = filt_rpy["sensor"]

    # Filter aligned tracking acceleration to guarantee zero residual drift
    filt_acc = filter_signals(
        {"tracking": acc_xyz_tracking, "sensor": acc_xyz_sensor},
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=4,
    )
    acc_xyz_tracking = filt_acc["tracking"]
    acc_xyz_sensor = filt_acc["sensor"]

    # Calculate Pearson correlations across all 6 axes
    corr_ax = float(np.corrcoef(acc_xyz_tracking[:, 0], acc_xyz_sensor[:, 0])[0, 1])
    corr_ay = float(np.corrcoef(acc_xyz_tracking[:, 1], acc_xyz_sensor[:, 1])[0, 1])
    corr_az = float(np.corrcoef(acc_xyz_tracking[:, 2], acc_xyz_sensor[:, 2])[0, 1])
    corr_roll = float(np.corrcoef(rpy_tracking[:, 0], rpy_sensor[:, 0])[0, 1])
    corr_pitch = float(np.corrcoef(rpy_tracking[:, 1], rpy_sensor[:, 1])[0, 1])
    corr_yaw = float(np.corrcoef(rpy_tracking[:, 2], rpy_sensor[:, 2])[0, 1])

    correlations = {
        "accX": corr_ax,
        "accY": corr_ay,
        "accZ": corr_az,
        "Roll": corr_roll,
        "Pitch": corr_pitch,
        "Yaw": corr_yaw,
    }

    return {
        "acc_xyz_sensor": acc_xyz_sensor,
        "rpy_sensor": rpy_sensor,
        "acc_xyz_tracking": acc_xyz_tracking,
        "rpy_tracking": rpy_tracking,
        "temporal_lag": lag,
        "start_vis": start_vis,
        "end_vis": end_vis,
        "R_cam_to_world": aligned_res["R_cam_to_world"],
        "R_axes_align": aligned_res["R_axes_align"],
        "pearson_correlations": correlations,
    }
