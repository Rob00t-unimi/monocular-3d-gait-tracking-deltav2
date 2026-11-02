"""
Unit tests for RMSE and Normalized RMSE (nRMSE) Metric.
"""

import unittest
import numpy as np
from src.metrics.magnitude.nrmse import calculate_rmse, calculate_nrmse


class TestNRMSEMetric(unittest.TestCase):
    def test_identical_signals(self):
        sig = np.sin(np.linspace(0, 10, 100))
        rmse = calculate_rmse(sig, sig)
        nrmse = calculate_nrmse(sig, sig)
        self.assertAlmostEqual(rmse, 0.0, places=6)
        self.assertAlmostEqual(nrmse, 0.0, places=6)

    def test_constant_offset_known_rom(self):
        # y_true from 0 to 10 (ROM = 10.0)
        y_true = np.linspace(0, 10, 101)
        # Shifted by +1.0
        y_pred = y_true + 1.0

        rmse = calculate_rmse(y_pred, y_true)
        self.assertAlmostEqual(rmse, 1.0, places=6)

        # nRMSE = (1.0 / 10.0) * 100 = 10.0%
        nrmse_percent = calculate_nrmse(y_pred, y_true, normalization="rom", as_percentage=True)
        self.assertAlmostEqual(nrmse_percent, 10.0, places=5)

        # Fraction mode
        nrmse_frac = calculate_nrmse(y_pred, y_true, normalization="rom", as_percentage=False)
        self.assertAlmostEqual(nrmse_frac, 0.1, places=6)

    def test_std_normalization(self):
        y_true = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        std_val = float(np.std(y_true))
        y_pred = y_true + 2.0  # RMSE = 2.0

        nrmse = calculate_nrmse(y_pred, y_true, normalization="std", as_percentage=False)
        expected = 2.0 / std_val
        self.assertAlmostEqual(nrmse, expected, places=5)

    def test_mean_normalization(self):
        y_true = np.array([10.0, 20.0, 30.0])
        mean_val = 20.0
        y_pred = y_true + 2.0  # RMSE = 2.0

        nrmse = calculate_nrmse(y_pred, y_true, normalization="mean", as_percentage=False)
        self.assertAlmostEqual(nrmse, 2.0 / mean_val, places=6)

    def test_zero_rom_raises_error(self):
        const_true = np.ones(50) * 5.0
        y_pred = np.sin(np.linspace(0, 10, 50))
        with self.assertRaises(ValueError):
            calculate_nrmse(y_pred, const_true, normalization="rom")

    def test_unknown_normalization_raises_error(self):
        y = np.linspace(0, 5, 20)
        with self.assertRaises(ValueError):
            calculate_nrmse(y, y, normalization="invalid_mode")

    def test_length_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_rmse([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_nan_values_raise_error(self):
        with self.assertRaises(ValueError):
            calculate_rmse([1.0, np.nan], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
