"""
Module for loading and cleaning IMU sensor data.
Implements ISignalLoader protocol.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List
from src.core.interfaces import ISignalLoader
from src.core.constants import SENSORS_COLUMNS

logger = logging.getLogger(__name__)

class CsvSignalLoader(ISignalLoader):
    """
    Handles CSV loading for IMU sensors, ensuring column mapping and numeric conversion.
    
    Attributes:
        columns_to_keep: List of required signal columns.
    """

    def __init__(self, columns_to_keep: List[str] = SENSORS_COLUMNS):
        self.columns_to_keep = columns_to_keep

    def load(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Loads raw IMU data from a CSV file.
        
        Args:
            file_path: Path to the sensor CSV file.
            
        Returns:
            A cleaned pandas DataFrame or None if loading fails.
        """
        try:
            # Gait360 CSVs typically have 2 metadata rows to skip
            df = pd.read_csv(file_path, skiprows=2)
            df = df.dropna(how='all')

            # Ensure Timestamp is numeric
            if 'Timestamp' not in df.columns:
                logger.error(f"Timestamp column missing in {file_path}")
                return None
                
            df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
            df = df.dropna(subset=['Timestamp'])

            # Keep only requested columns that are actually present
            available_cols = [c for c in self.columns_to_keep if c in df.columns]
            return df[available_cols].reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Failed to load signal data from {file_path}: {e}")
            return None
