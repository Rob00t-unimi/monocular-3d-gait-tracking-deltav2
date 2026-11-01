"""
Unit tests for Cadence and Dominant Frequency Error Metric.
"""

import unittest
import numpy as np
from src.metrics.frequency.cadence_error import (
    extract_dominant_frequency,
    calculate_cadence_error,
    CadenceResult,
)


class TestCadenceErrorMetric(unittest.TestCase):
    def test_extract_dominant_frequency_sine(self):
        fs = 30.0
        target_f = 1.35  # Typical human gait cadence in Hz (~81 steps/min)
        t = np.arange(150) / fs  # 5 seconds
        y = np.sin(2 * np.pi * target_f * t)

        detected_f = extract_dominant_frequency(y, fs=fs)
        self.assertAlmostEqual(detected_f, target_f, delta=0.03)

    def test_cadence_identical_signals(self):
        fs = 30.0
        t = np.arange(180) / fs
        y = np.sin(2 * np.pi * 1.2 * t)

        res = calculate_cadence_error(y, y, fs_pred=fs)
        self.assertIsInstance(res, CadenceResult)
        self.assertAlmostEqual(res.freq_pred, 1.2, delta=0.03)
        self.assertAlmostEqual(res.freq_true, 1.2, delta=0.03)
        self.assertAlmostEqual(res.cadence_pred_bpm, 72.0, delta=2.0)
        self.assertAlmostEqual(res.relative_error_percent, 0.0, places=3)
        self.assertAlmostEqual(res.frequency_ratio, 1.0, places=3)
        self.assertFalse(res.has_identity_swap)

    def test_identity_swap_detection(self):
        # GT signal at 1.0 Hz (single leg gait cycle)
        # Pred signal at 2.0 Hz (spurious double frequency from swapping between legs)
        fs = 30.0
        t = np.arange(180) / fs
        y_true = np.sin(2 * np.pi * 1.0 * t)
        y_pred = np.sin(2 * np.pi * 2.0 * t)

        res = calculate_cadence_error(y_pred, y_true, fs_pred=fs)
        self.assertAlmostEqual(res.freq_true, 1.0, delta=0.03)
        self.assertAlmostEqual(res.freq_pred, 2.0, delta=0.03)
        self.assertAlmostEqual(res.frequency_ratio, 2.0, delta=0.05)
        self.assertAlmostEqual(res.relative_error_percent, 100.0, delta=5.0)
        self.assertTrue(res.has_identity_swap)

    def test_different_sampling_rates(self):
        # Vision at 30.0 Hz, IMU at 100.0 Hz, both tracking a 1.25 Hz motion
        target_f = 1.25
        t_vis = np.arange(150) / 30.0   # 5 sec at 30 Hz
        t_imu = np.arange(500) / 100.0  # 5 sec at 100 Hz

        y_vis = np.sin(2 * np.pi * target_f * t_vis)
        y_imu = np.sin(2 * np.pi * target_f * t_imu)

        res = calculate_cadence_error(y_vis, y_imu, fs_pred=30.0, fs_true=100.0)
        self.assertAlmostEqual(res.freq_pred, target_f, delta=0.03)
        self.assertAlmostEqual(res.freq_true, target_f, delta=0.03)
        self.assertAlmostEqual(res.frequency_ratio, 1.0, delta=0.04)
        self.assertLess(res.relative_error_percent, 4.0)
        self.assertFalse(res.has_identity_swap)

    def test_invalid_fs_raises_error(self):
        with self.assertRaises(ValueError):
            extract_dominant_frequency([1, 2, 3, 4], fs=-5.0)

    def test_too_few_samples_raises_error(self):
        with self.assertRaises(ValueError):
            extract_dominant_frequency([1.0, 2.0], fs=30.0)

    def test_nan_values_raise_error(self):
        with self.assertRaises(ValueError):
            extract_dominant_frequency([1.0, np.nan, 3.0, 4.0], fs=30.0)


if __name__ == "__main__":
    unittest.main()
