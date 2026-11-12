"""
Comprehensive 40-Video Benchmark Evaluation Suite (World-Space Kinematic Alignment).

Evaluates 2D waveform, frequency, magnitude, and 3D geometric spatial tracking
metrics against IMU Ground Truth in the shared static World Coordinate System.

Evaluates 4 distinct methods across all 40 videos:
  1. dummy            - 2D Keypoint Lift Baseline
  2. raw              - DELTAv2 Raw Unpruned 3 Reference Keypoints
  3. refined_triplet  - DELTAv2 Geometric Pruning + Optimal Rigidity Triplet
  4. refined_cloud    - DELTAv2 Geometric Pruning + Point Cloud SVD Best-Fit Plane

Outputs comprehensive multi-dimensional results to results/benchmark_40_metrics.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import interpolate, signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.gt_loader import load_raw_imu
from src.analysis.signal_extractor import extract_aligned_signals, load_cutoffs
from src.analysis.kinematic_alignment import run_pipeline
from src.data_io.dataset_scanner import DatasetScanner
from src.data_io.state_manager import videos_state_manager
from src.inference.dummy_tracker import DummyTracker
from src.metrics.calculator import calculate_all_metrics
from src.metrics.magnitude.nrmse import calculate_nrmse, calculate_rmse
from src.metrics.waveform.cross_correlation import calculate_cross_correlation
from src.metrics.waveform.pearson import calculate_pearson

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("compute_all_benchmark_metrics")

MANIFEST_PATH = PROJECT_ROOT / "data" / "selected_inference_videos_40.csv"
RAW_DIR = PROJECT_ROOT / "data" / "inference_results" / "raw"
REFINED_DIR = PROJECT_ROOT / "data" / "inference_results" / "refined"
RESULTS_DIR = PROJECT_ROOT / "results"

CLEAN_CATEGORIES = [
    "90° Sagittale (Destra -> Sinistra)",
    "90° Sagittale (Sinistra -> Destra)",
    "0-deg Pure Frontal (Approaching)",
    "180-deg Pure Posterior (Receding)",
    "45-deg Oblique (Right -> Left) - Front-Oblique (Approaching)",
    "45-deg Oblique (Right -> Left) - Rear-Oblique (Receding)",
    "45-deg Oblique (Left -> Right) - Front-Oblique (Approaching)",
    "45-deg Oblique (Left -> Right) - Rear-Oblique (Receding)",
]


def get_category_name(raw_cat: str) -> str:
    """Deterministic index-based mapping from category_8class to clean category title."""
    try:
        idx = int(str(raw_cat).strip()[0]) - 1
        if 0 <= idx < len(CLEAN_CATEGORIES):
            return CLEAN_CATEGORIES[idx]
    except Exception:
        pass
    return str(raw_cat)


def compute_triangle_area_cv(pts: np.ndarray | None) -> float:
    """Computes Area Coefficient of Variation (std / mean * 100) for a 3-point 3D triangle."""
    if pts is None or pts.ndim != 3 or pts.shape[1] < 3:
        return 0.0
    v1 = pts[:, 1, :] - pts[:, 0, :]
    v2 = pts[:, 2, :] - pts[:, 0, :]
    c = np.cross(v1, v2)
    areas = 0.5 * np.linalg.norm(c, axis=-1)
    mean_a = float(np.mean(areas))
    if mean_a > 1e-12:
        return float(np.std(areas) / mean_a * 100.0)
    return 0.0


def align_track3d_to_gt(
    track3d: np.ndarray,
    video_name: str,
    fps: float,
    start_idx: int,
    ref_frame_idx: int,
    lowcut: float,
    highcut: float,
) -> Optional[Dict[str, Any]]:
    """Aligns arbitrary 3D tracking coordinates (e.g. Dummy) to World IMU space."""
    df_imu = load_raw_imu(video_name, PROJECT_ROOT)
    if df_imu is None or len(df_imu) < 10:
        return None

    T = track3d.shape[0]
    t_vis = (start_idx + np.arange(T)) / fps
    ref_idx = max(0, ref_frame_idx - start_idx)

    t_margin = 4.0
    t_start_safe = max(0.0, t_vis[0] - t_margin)
    t_end_safe = min(df_imu["Timestamp"].iloc[-1], t_vis[-1] + t_margin)
    t_eval_imu = np.arange(t_start_safe, t_end_safe, 1.0 / fps)

    f_acc = interpolate.interp1d(df_imu["Timestamp"], df_imu[["accX", "accY", "accZ"]].values, axis=0, fill_value="extrapolate")
    f_quat = interpolate.interp1d(df_imu["Timestamp"], df_imu[["q0", "q1", "q2", "q3"]].values, axis=0, fill_value="extrapolate")
    imu_acc_interp = f_acc(t_eval_imu)
    imu_quat_interp = f_quat(t_eval_imu)
    imu_quat_interp /= (np.linalg.norm(imu_quat_interp, axis=-1, keepdims=True) + 1e-9)

    pipeline_res = run_pipeline(
        video_pts=track3d,
        imu_acc=imu_acc_interp,
        imu_quat=imu_quat_interp,
        fs=fps,
        ref_idx=min(ref_idx, len(track3d) - 1),
        scalar_first=True,
        lowcut=lowcut,
        highcut=highcut,
    )

    N_out = len(pipeline_res["acc_xyz_tracking"])
    sv = int(pipeline_res.get("start_vis", 0))
    ev = int(pipeline_res.get("end_vis", sv + N_out))
    return {
        "t_vis": t_vis[sv:ev],
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
        "points_3d": track3d[sv:ev],
        "temporal_lag": pipeline_res["temporal_lag"],
        "correlations": pipeline_res["pearson_correlations"],
    }


def compute_comprehensive_metrics(
    signals: Dict[str, Any],
    fps: float,
    angle: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes all 2D waveform, frequency, and 3D spatial metrics against World-aligned GT.
    All incoming signals are already expressed in the shared static World Coordinate System.
    """
    pitch_pred = signals["Pitch"].copy()
    pitch_true = signals["gt_Pitch"].copy()
    acc_pred = signals["accX"].copy()
    acc_true = signals["gt_accX"].copy()
    track3d = signals.get("points_3d")

    acc3d_pred = np.column_stack([signals["accX"], signals["accY"], signals["accZ"]])
    acc3d_true = np.column_stack([signals["gt_accX"], signals["gt_accY"], signals["gt_accZ"]])

    # Core report via GaitEvaluationReport against synced World GT
    report = calculate_all_metrics(
        pitch_pred=pitch_pred,
        pitch_true=pitch_true,
        acc_pred=acc_pred,
        acc_true=acc_true,
        track3d=track3d,
        acc3d_pred=acc3d_pred,
        acc3d_true=acc3d_true,
        fps_pred=fps,
        fps_true=fps,
        angle=angle,
    )

    # Detailed individual metrics
    r_pitch = float(calculate_pearson(pitch_pred, pitch_true))
    r_accX = float(calculate_pearson(acc_pred, acc_true))
    r_accY = float(calculate_pearson(signals["accY"], signals["gt_accY"]))
    r_accZ = float(calculate_pearson(signals["accZ"], signals["gt_accZ"]))
    r_roll = float(calculate_pearson(signals["Roll"], signals["gt_Roll"]))
    r_yaw = float(calculate_pearson(signals["Yaw"], signals["gt_Yaw"]))

    rmse_p = float(calculate_rmse(pitch_pred, pitch_true))
    nrmse_p = float(report.nrmse_pitch_percent)

    rmse_a = float(calculate_rmse(acc_pred, acc_true))
    nrmse_a = float(report.nrmse_acc_percent) if report.nrmse_acc_percent is not None else 0.0

    rom_res = report.rom_pitch
    rom_p_pred = float(rom_res.rom_pred)
    rom_p_true = float(rom_res.rom_true)

    sig_rig = float(report.rigidity.sigma_rigidity) if report.rigidity else None
    rel_def = float(report.rigidity.relative_deformation_percent) if report.rigidity else None

    rms_jerk = float(report.smoothness.rms_jerk) if report.smoothness else None
    peak_jerk = float(report.smoothness.peak_jerk) if report.smoothness else None
    mean_jerk = float(report.smoothness.mean_jerk) if report.smoothness else None

    area_cv = compute_triangle_area_cv(track3d)

    dtw_acc3d_raw = float(report.dtw_acc3d.total_distance) if report.dtw_acc3d else 0.0
    dtw_acc3d_norm = float(report.dtw_acc3d.normalized_distance) if report.dtw_acc3d else 0.0

    lag_samples = int(signals.get("temporal_lag", 0))
    lag_sec = lag_samples / fps

    return {
        # --- Temporal Synchronization ---
        "temporal_shift_sec": lag_sec,
        "temporal_shift_samples": lag_samples,
        # --- 2D Waveform & Correlation (World Aligned) ---
        "pearson_pitch": r_pitch,
        "abs_pearson_pitch": abs(r_pitch),
        "pearson_accX": r_accX,
        "abs_pearson_accX": abs(r_accX),
        "pearson_accY": r_accY,
        "pearson_accZ": r_accZ,
        "pearson_roll": r_roll,
        "pearson_yaw": r_yaw,
        # --- Magnitude & Error (World Aligned) ---
        "rmse_pitch": rmse_p,
        "nrmse_pitch_percent": nrmse_p,
        "rmse_accX": rmse_a,
        "nrmse_accX_percent": nrmse_a,
        "rom_pitch_pred_deg": rom_p_pred,
        "rom_pitch_true_deg": rom_p_true,
        "rom_pitch_error_percent": float(rom_res.relative_error_percent),
        # --- Frequency & Cadence ---
        "cadence_pred_hz": float(report.cadence_pitch.freq_pred),
        "cadence_true_hz": float(report.cadence_pitch.freq_true),
        "cadence_pred_bpm": float(report.cadence_pitch.cadence_pred_bpm),
        "cadence_true_bpm": float(report.cadence_pitch.cadence_true_bpm),
        "cadence_error_percent": float(report.cadence_pitch.relative_error_percent),
        "cadence_freq_ratio": float(report.cadence_pitch.frequency_ratio),
        "has_identity_swap": bool(
            report.swap_detection.has_identity_swap
            if report.swap_detection is not None
            else report.cadence_pitch.has_identity_swap
        ),
        "is_depth_inversion_artifact": bool(
            report.swap_detection.is_depth_inversion_artifact
            if report.swap_detection is not None
            else False
        ),
        "antiphase_percentage": float(
            report.swap_detection.antiphase_percentage
            if report.swap_detection is not None
            else 0.0
        ),
        "swap_diagnostic": str(
            report.swap_detection.diagnostic
            if report.swap_detection is not None
            else "None"
        ),
        # --- Dynamic Time Warping (1D & 3D) ---
        "dtw_pitch_norm": float(report.dtw_pitch.normalized_distance),
        "dtw_pitch_raw": float(report.dtw_pitch.total_distance),
        "dtw_acc3d_norm": dtw_acc3d_norm,
        "dtw_acc3d_raw": dtw_acc3d_raw,
        # --- 3D Spatial Rigidity & Smoothness ---
        "sigma_rigidity_m": sig_rig,
        "relative_deformation_percent": rel_def,
        "rms_jerk": rms_jerk,
        "peak_jerk": peak_jerk,
        "mean_jerk": mean_jerk,
        "triangle_area_cv_percent": float(area_cv),
    }


def main(force: bool = False):
    json_path = RESULTS_DIR / "benchmark_40_metrics.json"
    if json_path.exists() and not force:
        with open(json_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        meta = cached.get("metadata", {})
        logger.info(
            f"Benchmark metrics cache already exists at {json_path.name} "
            f"({meta.get('videos_processed', 40)}/{meta.get('videos_total', 40)} videos). "
            "Loading from cache (use --force to recompute)..."
        )
        return cached

    start_total = time.time()
    logger.info("Initializing 40-video World-Aligned benchmark metric evaluation...")

    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(MANIFEST_PATH)
    total_videos = len(df_manifest)
    logger.info(f"Loaded manifest with {total_videos} videos.")

    lowcut, highcut = load_cutoffs(PROJECT_ROOT)
    logger.info(f"Loaded physiological bandpass cutoffs: [{lowcut:.3f} Hz - {highcut:.3f} Hz]")

    scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
    dummy_tracker = DummyTracker()

    # Data structures for results
    results_by_category: Dict[str, Dict[str, Dict[str, Any]]] = {cat: {} for cat in CLEAN_CATEGORIES}
    evaluations_by_video: Dict[str, Dict[str, Any]] = {}

    processed_count = 0

    for idx, row in df_manifest.iterrows():
        vname = row["video_name"]
        raw_cat = row["category_8class"]
        cat_name = get_category_name(raw_cat)

        person_str = str(row["person"]).replace("P", "")
        person_int = int(person_str)
        exp_int = int(str(row["experiment"]).replace("E", ""))
        test_int = int(str(row["test"]).replace("T", "").replace("ALT", ""))
        angle_str = str(row["angle"])
        is_alt = bool(row["condition"] != "Standard")

        print(f"\n[{idx+1:02d}/{total_videos:02d}] Processing: {vname} | Category: {cat_name}")

        raw_npz_path = RAW_DIR / f"delta_v2_raw_{vname}.npz"
        refined_npz_path = REFINED_DIR / f"delta_v2_refined_{vname}.npz"
        sensor_csv_path = Path(row["sensor_csv"])
        if not sensor_csv_path.is_absolute():
            sensor_csv_path = PROJECT_ROOT / sensor_csv_path

        if not raw_npz_path.exists() or not refined_npz_path.exists():
            logger.error(f"Missing NPZ for {vname}. Skipping.")
            continue
        if not sensor_csv_path.exists():
            logger.error(f"Missing GT CSV {sensor_csv_path} for {vname}. Skipping.")
            continue

        d_ref = np.load(refined_npz_path, allow_pickle=True)
        fps = float(d_ref.get("fps", 30.0))
        ref_frame_idx = int(d_ref.get("ref_frame_idx", 0))
        start_idx = int(d_ref.get("start_idx", 0))
        n_frames = len(d_ref["track3d"])

        # 1. Extract World-Aligned Signals for Refined Triplet
        sig_refined_trip = extract_aligned_signals(refined_npz_path, lowcut, highcut, source="triplet")
        if sig_refined_trip is None:
            logger.error(f"Failed to align refined triplet for {vname}. Skipping.")
            continue
        metrics_ref_trip = compute_comprehensive_metrics(sig_refined_trip, fps=fps, angle=angle_str)

        # 2. Extract World-Aligned Signals for Refined Cloud
        sig_refined_cloud = extract_aligned_signals(refined_npz_path, lowcut, highcut, source="cloud")
        metrics_ref_cloud = compute_comprehensive_metrics(sig_refined_cloud, fps=fps, angle=angle_str) if sig_refined_cloud else metrics_ref_trip

        # 3. Extract World-Aligned Signals for Raw Triplet
        sig_raw = extract_aligned_signals(raw_npz_path, lowcut, highcut, source="triplet")
        metrics_raw = compute_comprehensive_metrics(sig_raw, fps=fps, angle=angle_str) if sig_raw else metrics_ref_trip

        # 4. Run Dummy Tracker & Align
        v_info = scanner.get_video_info(person_int, exp_int, test_int, angle_str, is_alt)
        state = videos_state_manager.load(vname)
        metrics_dummy = metrics_ref_trip # fallback
        if state.ref_frame_idx is not None and len(state.specific_keypoints) >= 3:
            dummy_res = dummy_tracker.track(
                video_info=v_info,
                state=state,
                start_frame=start_idx,
                end_frame=start_idx + n_frames - 1,
                depth_mode="anthropometric",
            )
            track3d_dummy = dummy_res["track3d"]
            sig_dummy = align_track3d_to_gt(track3d_dummy, vname, fps, start_idx, ref_frame_idx, lowcut, highcut)
            if sig_dummy is not None:
                metrics_dummy = compute_comprehensive_metrics(sig_dummy, fps=fps, angle=angle_str)

        video_record = {
            "video_name": vname,
            "category": cat_name,
            "person": row["person"],
            "experiment": row["experiment"],
            "test": row["test"],
            "angle": row["angle"],
            "direction": row["direction"],
            "condition": row["condition"],
            "frames_num": n_frames,
            "fps": fps,
            "methods": {
                "dummy": metrics_dummy,
                "raw": metrics_raw,
                "refined_triplet": metrics_ref_trip,
                "refined_cloud": metrics_ref_cloud,
            },
        }

        # Store in both lookups
        results_by_category[cat_name][vname] = {
            "dummy": metrics_dummy,
            "raw": metrics_raw,
            "refined_triplet": metrics_ref_trip,
            "refined_cloud": metrics_ref_cloud,
        }
        evaluations_by_video[vname] = video_record

        processed_count += 1
        print(
            f"   [ALIGNED] Pitch r: Trip={metrics_ref_trip['pearson_pitch']:+.3f} | Cloud={metrics_ref_cloud['pearson_pitch']:+.3f} | "
            f"Raw={metrics_raw['pearson_pitch']:+.3f} | Dummy={metrics_dummy['pearson_pitch']:+.3f}  ||  "
            f"AccX r: Trip={metrics_ref_trip['pearson_accX']:+.3f} | Cloud={metrics_ref_cloud['pearson_accX']:+.3f}  ||  "
            f"3D DTW: Trip={metrics_ref_trip['dtw_acc3d_norm']:.3f} | Cloud={metrics_ref_cloud['dtw_acc3d_norm']:.3f}"
        )

    # 5. Export to JSON
    json_path = RESULTS_DIR / "benchmark_40_metrics.json"
    full_output = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos_total": total_videos,
            "videos_processed": processed_count,
            "cutoff_frequencies_hz": {"lowcut": lowcut, "highcut": highcut},
            "alignment_method": "world_space_kinematic_alignment",
            "methods": ["dummy", "raw", "refined_triplet", "refined_cloud"],
            "categories": CLEAN_CATEGORIES,
        },
        "results_by_category": results_by_category,
        "evaluations_by_video": evaluations_by_video,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=4)

    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPUTATION COMPLETE: {processed_count}/{total_videos} videos processed in {elapsed:.1f}s.")
    print(f"Full results saved to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
