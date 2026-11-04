"""
Ground Truth (IMU) loader and temporal alignment for the Gait360 benchmark.

Replicates the ``_load_gt`` logic from
experiments/delta_test/vision_vs_gt_comparison.py and aligns the GT signal
to the exact temporal window of the corresponding tracking NPZ.

Also provides ``load_raw_imu`` to load raw quaternions (q0..q3), accelerations,
and angular rates directly for the kinematic alignment pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal, interpolate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_timestamps(ts: np.ndarray) -> np.ndarray:
    """Normalise raw IMU timestamps to seconds starting at 0."""
    diff = np.median(np.diff(ts))
    if diff > 0.5:
        # Timestamps are in milliseconds
        return (ts - ts[0]) / 1000.0
    return ts - ts[0]


_CACHED_CSVS: list[str] | None = None


def get_gt_csv_path(video_name: str, project_root: Optional[Path | str] = None) -> Optional[Path]:
    """Locate the raw Gait360 sensor CSV for a given video name."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_io.dataset_scanner import DatasetScanner

    csv_stem = (
        video_name
        .replace(".mp4", "")
        .replace("_45", "")
        .replace("_90", "")
        .replace("_Front", "")
        .replace("_Rear", "")
        .replace("_ALT", "")
    )
    global _CACHED_CSVS
    if _CACHED_CSVS is None:
        scanner = DatasetScanner(str(project_root / "data"))
        _CACHED_CSVS = scanner.get_csvs()

    gt_path_str = next(
        (p for p in _CACHED_CSVS if csv_stem in Path(p).stem),
        None,
    )
    if gt_path_str is None:
        logger.warning(f"No GT CSV found for video '{video_name}' (stem='{csv_stem}')")
        return None
    res_p = Path(gt_path_str)
    return res_p if res_p.is_absolute() else (project_root / res_p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_imu(
    video_name: str,
    project_root: Optional[Path | str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load raw IMU data including quaternions (q0, q1, q2, q3), accelerations,
    gyroscopes, and Roll/Pitch/Yaw.
    
    Timestamps are converted to float seconds starting at 0.0.
    """
    csv_path = get_gt_csv_path(video_name, project_root)
    if csv_path is None or not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, skiprows=2).dropna(how="all")
        if "Timestamp" not in df.columns:
            return None
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)
        df["Timestamp"] = _fix_timestamps(df["Timestamp"].values)
        return df
    except Exception as e:
        logger.error(f"Failed to load raw IMU from {csv_path}: {e}")
        return None


def load_gt(
    video_name: str,
    t_vis: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 2.9296875,
    project_root: Optional[Path | str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load, filter, and align the GT IMU signal to the video timeline.

    Args:
        video_name:   e.g. ``"P27_E2T2_90.mp4"`` — used to find the
                      corresponding CSV file in the Gait360 dataset.
        t_vis:        (T,) array of absolute video timestamps in seconds
                      (output of ``signal_extractor.extract_signals``).
        lowcut:       Low-cut frequency in Hz.
        highcut:      High-cut frequency in Hz.
        project_root: Repository root. Autodetected if None.

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``['Timestamp', 'accX', 'accY', 'accZ', 'Roll', 'Pitch', 'Yaw']``
        where ``Timestamp`` equals ``t_vis``, or ``None`` if the CSV was
        not found.
    """
    csv_path = get_gt_csv_path(video_name, project_root)
    if csv_path is None:
        return None

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(project_root)

    from src.preprocessing.signals.signal_orchestrator import SignalOrchestrator

    orch = SignalOrchestrator()
    df = orch.loader.load(csv_path)

    df["Timestamp"] = _fix_timestamps(df["Timestamp"].values)
    df = orch.estimator._unwrap_angles(df)
    for col in ["Roll", "Pitch", "Yaw"]:
        df[col] -= df[col].mean()

    # Bandpass filter [lowcut, highcut]
    df_filt = orch.processor.process(df, lowcut, highcut)

    # --- Interpolate to video timeline ---
    result = pd.DataFrame({"Timestamp": t_vis})
    for col in ["accX", "accY", "accZ", "Roll", "Pitch", "Yaw"]:
        f = interpolate.interp1d(
            df_filt["Timestamp"].values,
            df_filt[col].values,
            fill_value="extrapolate",
        )
        result[col] = f(t_vis)

    return result
