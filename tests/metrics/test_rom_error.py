"""
Unit tests for Range of Motion (ROM) Error Metric.
"""

import unittest
import numpy as np
from src.metrics.magnitude.rom_error import (
    calculate_rom,
    calculate_rom_error,
    ROMResult,
)


class TestROMErrorMetric(unittest.TestCase):
    def test_calculate_rom(self):
        y = np.array([-10.0, 0.0, 35.0, 15.0])
        self.assertEqual(calculate_rom(y), 45.0)

    def test_identical_signals(self):
        y = np.linspace(-25.0, 25.0, 101)  # Exact ROM = 50.0
        res = calculate_rom_error(y, y)
        self.assertIsInstance(res, ROMResult)
        self.assertAlmostEqual(res.rom_true, 50.0, places=5)
        self.assertAlmostEqual(res.rom_pred, 50.0, places=5)
        self.assertAlmostEqual(res.absolute_error, 0.0, places=6)
        self.assertAlmostEqual(res.signed_error, 0.0, places=6)
        self.assertAlmostEqual(res.relative_error_percent, 0.0, places=6)

    def test_offset_invariance(self):
        # Shifting the signal vertically does not alter its ROM
        t = np.linspace(0, 2 * np.pi, 100)
        y_true = np.sin(t) * 20.0  # ROM = 40.0
        y_pred = y_true + 100.0   # Same ROM

        res = calculate_rom_error(y_pred, y_true)
        self.assertAlmostEqual(res.absolute_error, 0.0, places=6)
        self.assertAlmostEqual(res.relative_error_percent, 0.0, places=6)

    def test_underestimation_foreshortening(self):
        # Perspective foreshortening: True ROM = 50, Pred ROM = 40
        y_true = np.array([0.0, 50.0, 10.0, 30.0])
        y_pred = np.array([0.0, 40.0, 10.0, 25.0])

        res = calculate_rom_error(y_pred, y_true)
        self.assertAlmostEqual(res.rom_true, 50.0)
        self.assertAlmostEqual(res.rom_pred, 40.0)
        self.assertAlmostEqual(res.absolute_error, 10.0)
        self.assertAlmostEqual(res.signed_error, -10.0)
        # (40 - 50) / 50 * 100 = -20%
        self.assertAlmostEqual(res.relative_error_percent, -20.0)

    def test_overestimation(self):
        # Spurious tracking peak: True ROM = 40, Pred ROM = 50
        y_true = np.array([0.0, 40.0, 10.0])
        y_pred = np.array([0.0, 50.0, 10.0])

        res = calculate_rom_error(y_pred, y_true)
        self.assertAlmostEqual(res.absolute_error, 10.0)
        self.assertAlmostEqual(res.signed_error, 10.0)
        # (50 - 40) / 40 * 100 = +25%
        self.assertAlmostEqual(res.relative_error_percent, 25.0)

    def test_zero_rom_handling(self):
        const_true = np.ones(50) * 10.0
        const_pred = np.ones(50) * 10.0
        res = calculate_rom_error(const_pred, const_true)
        self.assertEqual(res.absolute_error, 0.0)
        self.assertEqual(res.relative_error_percent, 0.0)

    def test_length_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_rom_error([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_nan_values_raise_error(self):
        with self.assertRaises(ValueError):
            calculate_rom([1.0, np.nan, 3.0])


if __name__ == "__main__":
    unittest.main()
