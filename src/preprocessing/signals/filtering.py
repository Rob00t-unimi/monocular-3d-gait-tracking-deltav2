"""
Module for applying bandpass filtering to sensor signals.
Implements ISignalProcessor protocol.
"""

import logging
import pandas as pd
import numpy as np
from scipy import signal
from typing import Optional
from src.core.interfaces import ISignalProcessor

logger = logging.getLogger(__name__)

class ButterworthSignalProcessor(ISignalProcessor):
    """
    Applies a zero-phase Butterworth filter to IMU data.
    """

    def __init__(self, order: int = 4):
        self.order = order

    def process(self, df: pd.DataFrame, lowcut: float, highcut: float) -> pd.DataFrame:
        """
        Filters sensor columns while preserving the Timestamp.
        
        Args:
            df: Input DataFrame with raw signals.
            lowcut: Lower frequency bound in Hz.
            highcut: Upper frequency bound in Hz.
            
        Returns:
            A new DataFrame containing the filtered signals.
        """
        fs = self._calculate_fs(df['Timestamp'].values)
        if not fs or fs <= 0:
            logger.warning("Invalid sampling frequency detected, skipping filtering.")
            return df.copy()

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Guard against Nyquist violations
        if high >= 1.0: 
            high = 0.99
            logger.debug(f"Cap highcut to {high*nyq:.2f}Hz for Nyquist limit.")

        b, a = signal.butter(self.order, [low, high], btype='band')

        df_filtered = df.copy()
        # Ensure we only filter the signal columns
        signal_cols = ['accX', 'accY', 'accZ', 'Roll', 'Pitch', 'Yaw']
        
        for col in signal_cols:
            if col in df_filtered.columns:
                data = df_filtered[col].values
                # Ensure signal is long enough for the filter
                if len(data) > max(len(a), len(b)):
                    # zero-phase filter (filtfilt) is crucial for temporal alignment
                    padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
                    df_filtered[col] = signal.filtfilt(b, a, data, axis=0, padlen=padlen)
        
        return df_filtered

    def _calculate_fs(self, timestamps: np.ndarray) -> Optional[float]:
        """Calculates sampling frequency dynamically from timestamps."""
        diffs = np.diff(timestamps)
        if len(diffs) == 0: return None
        dt_s = np.median(diffs)
        return 1.0 / dt_s if dt_s > 0 else None
