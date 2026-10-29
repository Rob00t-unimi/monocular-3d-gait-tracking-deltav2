"""
Module for gait frequency estimation using Power Spectral Density (PSD).
Implements IFrequencyEstimator protocol.
"""

import numpy as np
import logging
from scipy import signal
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

from src.core.interfaces import IFrequencyEstimator, ISignalLoader
from src.core.constants import MIN_SAMPLES, HUMAN_RANGE_FREQ

logger = logging.getLogger(__name__)

class GaitFrequencyEstimator(IFrequencyEstimator):
    """
    Estimates dominant frequencies and adaptive cutoffs for human gait signals.
    Uses Welch's method for PSD estimation.
    """

    def __init__(self, loader: ISignalLoader):
        self.loader = loader

    def estimate(self, sample_paths: List[Path]) -> Tuple[float, float]:
        """
        Profiles a set of CSVs to find the typical frequency band.
        
        Args:
            sample_paths: List of paths to sensor CSV files.
            
        Returns:
            A tuple of (lowcut, highcut) frequencies in Hz.
        """
        dominant_freqs = []
        
        for path in sample_paths:
            df = self.loader.load(path)
            if df is None or len(df) < MIN_SAMPLES:
                continue

            fs = self._calculate_fs(df['Timestamp'].values)
            if not fs or fs <= 0:
                continue
                
            # Unwrap angles before analysis to avoid frequency spikes from jump points
            df = self._unwrap_angles(df)

            freq = self._extract_dominant_frequency(df, fs)
            if freq:
                dominant_freqs.append(freq)

        if not dominant_freqs:
            logger.warning("Could not profile frequencies, using default human gait range.")
            return HUMAN_RANGE_FREQ[0], HUMAN_RANGE_FREQ[1]

        median_freq = np.median(dominant_freqs)
        
        # Adaptive band: Median / 3 to Median * 3 (capturing main harmonics)
        lowcut = max(HUMAN_RANGE_FREQ[0], median_freq / 3.0)
        highcut = min(HUMAN_RANGE_FREQ[1], median_freq * 3.0)
        
        return float(lowcut), float(highcut)

    def _calculate_fs(self, timestamps: np.ndarray) -> Optional[float]:
        """Calculates sampling frequency dynamically from timestamps."""
        diffs = np.diff(timestamps)
        if len(diffs) == 0: return None
        dt_s = np.median(diffs)
        return 1.0 / dt_s if dt_s > 0 else None

    def _unwrap_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes 360-degree discontinuities in Euler angles."""
        df_copy = df.copy()
        for col in ['Roll', 'Pitch', 'Yaw']:
            if col in df_copy.columns:
                rads = np.deg2rad(df_copy[col])
                df_copy[col] = np.rad2deg(np.unwrap(rads))
        return df_copy

    def _extract_dominant_frequency(self, df: pd.DataFrame, fs: float) -> Optional[float]:
        """Identifies the peak frequency in the human gait range."""
        # Preference for Pitch as it's the primary axis for gait flexion/extension
        if 'Pitch' in df.columns:
            signal_data = df['Pitch'].values
        else:
            # Fallback to acceleration magnitude
            acc_cols = [c for c in ['accX', 'accY', 'accZ'] if c in df.columns]
            if not acc_cols: return None
            signal_data = np.linalg.norm(df[acc_cols].values, axis=1)

        # De-trend (remove DC component)
        signal_data = signal_data - np.mean(signal_data)

        # Welch PSD estimation
        nperseg = min(len(signal_data), 256)
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)

        # Filter for human gait range
        valid_idx = (freqs >= HUMAN_RANGE_FREQ[0]) & (freqs <= HUMAN_RANGE_FREQ[1])
        if not np.any(valid_idx):
            return None

        return freqs[valid_idx][np.argmax(psd[valid_idx])]
