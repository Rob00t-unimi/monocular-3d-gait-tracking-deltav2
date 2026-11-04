"""
Unit tests for Pearson Correlation Metric.
"""

import unittest
import numpy as np
from src.metrics.waveform.pearson import calculate_pearson


class TestPearsonMetric(unittest.TestCase):
    def test_identical_signals(self):
        t = np.linspace(0, 4 * np.pi, 100)
        sig = np.sin(t)
        r = calculate_pearson(sig, sig)
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_inverted_signals(self):
        t = np.linspace(0, 4 * np.pi, 100)
        sig = np.sin(t)
        r = calculate_pearson(sig, -sig)
        self.assertAlmostEqual(r, -1.0, places=6)

    def test_affine_transformation_invariance(self):
        # r should be exactly 1.0 under y_pred = a * y_true + b (a > 0)
        t = np.linspace(0, 2 * np.pi, 100)
        y_true = np.sin(t)
        y_pred = 3.5 * y_true + 12.0
        r = calculate_pearson(y_pred, y_true)
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_uncorrelated_signals(self):
        # Sin and Cos over integer number of periods are orthogonal
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        sin_sig = np.sin(t)
        cos_sig = np.cos(t)
        r = calculate_pearson(sin_sig, cos_sig)
        self.assertAlmostEqual(r, 0.0, places=5)

    def test_constant_signal_returns_zero(self):
        sig = np.sin(np.linspace(0, 10, 50))
        const_sig = np.ones(50) * 4.0
        r = calculate_pearson(sig, const_sig)
        self.assertEqual(r, 0.0)

    def test_squeezable_input_shapes(self):
        sig1 = np.array([[1.0], [2.0], [3.0], [4.0]])
        sig2 = np.array([1.0, 2.0, 3.0, 4.0])
        r = calculate_pearson(sig1, sig2)
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_length_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_pearson([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_nan_values_raise_error(self):
        with self.assertRaises(ValueError):
            calculate_pearson([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])

    def test_single_sample_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_pearson([1.0], [1.0])


if __name__ == "__main__":
    unittest.main()
