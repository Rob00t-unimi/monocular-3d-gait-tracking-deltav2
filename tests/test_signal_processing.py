"""
Unit tests for the refactored signal processing modules.
Verifies frequency estimation, filtering, and physiological validation.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pathlib import Path

from src.preprocessing.signals.frequency_analysis import GaitFrequencyEstimator
from src.preprocessing.signals.filtering import ButterworthSignalProcessor
from src.preprocessing.signals.validation import PhysiologicalSignalValidator
from src.core.constants import HUMAN_MAX_DEGREE, MAX_ACC

class TestSignalProcessing(unittest.TestCase):
    def setUp(self):
        # Create a synthetic gait-like signal (2 Hz sine wave)
        self.fs = 100.0
        t = np.arange(0, 5, 1/self.fs)
        self.t = t
        # Simulated Pitch signal at 2Hz
        pitch = 30 * np.sin(2 * 2 * np.pi * t)
        self.sample_df = pd.DataFrame({
            'Timestamp': t,
            'accX': 1.0 + 0.5 * np.sin(2 * 2 * np.pi * t),
            'accY': 0.1 * np.random.randn(len(t)),
            'accZ': 9.8 + 0.2 * np.sin(2 * 2 * np.pi * t),
            'Roll': 5 * np.cos(2 * 2 * np.pi * t),
            'Pitch': pitch,
            'Yaw': 2 * np.sin(2 * 2 * np.pi * t)
        })

    def test_frequency_estimation(self):
        mock_loader = MagicMock()
        mock_loader.load.return_value = self.sample_df
        estimator = GaitFrequencyEstimator(mock_loader)
        
        # Should detect ~2Hz dominant frequency
        low, high = estimator.estimate([Path("fake.csv")])
        
        # 2Hz / 3 = 0.66. Due to Welch binning, it might be slightly different.
        self.assertAlmostEqual(low, 0.65, places=1)
        self.assertAlmostEqual(high, 5.0, places=1) # Capped at 5.0 by HUMAN_RANGE_FREQ

    def test_butterworth_filtering(self):
        processor = ButterworthSignalProcessor(order=4)
        # Filter between 0.5 and 3.0 Hz
        filtered_df = processor.process(self.sample_df, 0.5, 3.0)
        
        self.assertEqual(len(filtered_df), len(self.sample_df))
        # Ensure timestamp is preserved
        np.testing.assert_array_almost_equal(filtered_df['Timestamp'], self.sample_df['Timestamp'])
        # Signal should be smoother (less noise if we added any)
        self.assertTrue('Pitch' in filtered_df.columns)

    def test_physiological_validator_valid(self):
        validator = PhysiologicalSignalValidator()
        self.assertTrue(validator.is_valid(self.sample_df))

    def test_physiological_validator_invalid_rom(self):
        validator = PhysiologicalSignalValidator()
        # Create impossible ROM (> 180 degrees)
        invalid_df = self.sample_df.copy()
        invalid_df['Pitch'] = 200 * np.sin(2 * np.pi * self.t)
        self.assertFalse(validator.is_valid(invalid_df))

    def test_physiological_validator_invalid_peaks(self):
        validator = PhysiologicalSignalValidator()
        # Create impossible acceleration peak
        invalid_df = self.sample_df.copy()
        invalid_df.loc[0, 'accX'] = 500.0 
        self.assertFalse(validator.is_valid(invalid_df))

if __name__ == '__main__':
    unittest.main()
