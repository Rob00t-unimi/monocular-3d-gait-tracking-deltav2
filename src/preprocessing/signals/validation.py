"""
Module for validating processed signals against human gait constraints.
Implements ISignalValidator protocol.
"""

import numpy as np
import logging
import pandas as pd
from src.core.interfaces import ISignalValidator
from src.core.constants import HUMAN_MAX_DEGREE, MAX_ACC, MAX_ANGULAR_JUMP

logger = logging.getLogger(__name__)

class PhysiologicalSignalValidator(ISignalValidator):
    """
    Validates IMU data based on biomechanical and physical limits.
    Ensures sensors were active and data is physiologically plausible.
    """

    def is_valid(self, df: pd.DataFrame) -> bool:
        """
        Executes multiple validation checks on the signal.
        
        Args:
            df: Filtered sensor DataFrame.
            
        Returns:
            True if all checks pass, False otherwise.
        """
        checks = [
            self._check_activity,
            self._check_acceleration_peaks,
            self._check_range_of_motion,
            self._check_angular_continuity
        ]
        
        for check in checks:
            if not check(df):
                return False
                
        return True

    def _check_activity(self, df: pd.DataFrame) -> bool:
        """Checks if the sensor was actually moving (variance > threshold)."""
        for col in ['accX', 'accY', 'accZ']:
            if col in df.columns:
                variance = np.var(df[col])
                if variance < 1e-4:
                    logger.warning(f"Signal rejected: variance for {col} is near zero ({variance:.2e})")
                    return False
        return True

    def _check_acceleration_peaks(self, df: pd.DataFrame) -> bool:
        """Checks for extreme shock/drop events (> MAX_ACC)."""
        for col in ['accX', 'accY', 'accZ']:
            if col in df.columns:
                peak = np.max(np.abs(df[col]))
                if peak > MAX_ACC:
                    logger.warning(f"Signal rejected: unreasonable peak in {col} ({peak:.2f} m/s^2)")
                    return False
        return True

    def _check_range_of_motion(self, df: pd.DataFrame) -> bool:
        """Checks for peak-to-peak amplitude (Euler angles). Gait should not exceed ~180 deg."""
        for col in ['Roll', 'Pitch', 'Yaw']:
            if col in df.columns:
                rom = np.max(df[col]) - np.min(df[col])
                if rom > HUMAN_MAX_DEGREE:
                    logger.warning(f"Signal rejected: impossible Range of Motion in {col} ({rom:.1f} degrees)")
                    return False
        return True

    def _check_angular_continuity(self, df: pd.DataFrame) -> bool:
        """Checks for massive sudden jumps between frames (Gimbal lock/Glitches)."""
        for col in ['Roll', 'Pitch', 'Yaw']:
            if col in df.columns:
                max_jump = np.max(np.abs(np.diff(df[col])))
                if max_jump > MAX_ANGULAR_JUMP:
                    logger.warning(f"Signal rejected: sudden jump in {col} ({max_jump:.1f} degrees/frame)")
                    return False
        return True
