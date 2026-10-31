"""
Unit tests for Cross-Correlation and Temporal Lag Metric.
"""

import unittest
import numpy as np
from src.metrics.waveform.cross_correlation import (
    calculate_cross_correlation,
    CrossCorrelationResult,
)


class TestCrossCorrelationMetric(unittest.TestCase):
    def setUp(self):
        self.fps = 30.0
        np.random.seed(42)
        # Transient gait-like signal with clear acceleration features
        t = np.linspace(0, 5, 300)
        self.base_sig = np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.2 * t) + 0.3 * np.random.randn(300)

    def test_zero_lag_identical_signals(self):
        res = calculate_cross_correlation(self.base_sig, self.base_sig, fps=self.fps)
        self.assertIsInstance(res, CrossCorrelationResult)
        self.assertAlmostEqual(res.peak_correlation, 1.0, places=5)
        self.assertEqual(res.lag_samples, 0)
        self.assertAlmostEqual(res.lag_seconds, 0.0, places=5)

    def test_positive_lag_delayed_pred(self):
        # y_pred is delayed by 6 samples relative to y_true
        shift = 6
        y_true = self.base_sig[shift:]
        y_pred = self.base_sig[:-shift]

        res = calculate_cross_correlation(y_pred, y_true, fps=self.fps)
        self.assertEqual(res.lag_samples, shift)
        self.assertAlmostEqual(res.lag_seconds, shift / self.fps, places=4)
        self.assertGreater(res.peak_correlation, 0.95)

    def test_negative_lag_advanced_pred(self):
        # y_pred is advanced by 4 samples relative to y_true
        shift = 4
        y_true = self.base_sig[:-shift]
        y_pred = self.base_sig[shift:]

        res = calculate_cross_correlation(y_pred, y_true, fps=self.fps)
        self.assertEqual(res.lag_samples, -shift)
        self.assertAlmostEqual(res.lag_seconds, -shift / self.fps, places=4)
        self.assertGreater(res.peak_correlation, 0.95)

    def test_max_lag_window_constraint(self):
        # Delayed by 15 samples = 0.5s at 30 fps
        shift = 15
        y_true = self.base_sig[shift:]
        y_pred = self.base_sig[:-shift]

        # Restrict max lag to 0.2s (6 samples)
        res = calculate_cross_correlation(
            y_pred, y_true, fps=self.fps, max_lag_seconds=0.2
        )
        self.assertLessEqual(abs(res.lag_samples), 6)

    def test_constant_signal_returns_zeros(self):
        const_sig = np.zeros_like(self.base_sig)
        res = calculate_cross_correlation(const_sig, self.base_sig, fps=self.fps)
        self.assertEqual(res.peak_correlation, 0.0)
        self.assertEqual(res.lag_samples, 0)
        self.assertEqual(res.lag_seconds, 0.0)

    def test_length_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_cross_correlation([1.0, 2.0], [1.0, 2.0, 3.0], fps=self.fps)

    def test_invalid_fps_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_cross_correlation(self.base_sig, self.base_sig, fps=-10.0)

    def test_nan_values_raise_error(self):
        y_nan = self.base_sig.copy()
        y_nan[5] = np.nan
        with self.assertRaises(ValueError):
            calculate_cross_correlation(y_nan, self.base_sig, fps=self.fps)


if __name__ == "__main__":
    unittest.main()
