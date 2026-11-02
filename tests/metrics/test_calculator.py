"""
Unit tests for the Unified Gait Metrics Calculator.
"""

import unittest
import numpy as np
from src.metrics.calculator import (
    calculate_all_metrics,
    GaitEvaluationReport,
)


class TestGaitMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.fps = 30.0
        self.t = np.arange(150) / self.fps
        # Synthetic 1.25 Hz gait pitch (excursion from -20 to +30 deg)
        self.pitch_gt = 25.0 * np.sin(2 * np.pi * 1.25 * self.t) + 5.0
        # Synthetic AccX (gravity-adjusted swing acceleration)
        self.acc_gt = 8.0 * np.cos(2 * np.pi * 1.25 * self.t)

        # Synthetic 3D points (triplet) undergoing rigid movement
        p0 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        self.track3d = np.tile(p0[None, :, :], (150, 1, 1))

    def test_identical_signals_perfect_scores(self):
        report = calculate_all_metrics(
            pitch_pred=self.pitch_gt,
            pitch_true=self.pitch_gt,
            acc_pred=self.acc_gt,
            acc_true=self.acc_gt,
            track3d=self.track3d,
            fps_pred=self.fps,
        )

        self.assertIsInstance(report, GaitEvaluationReport)
        self.assertAlmostEqual(report.pearson_pitch, 1.0, places=5)
        self.assertAlmostEqual(report.pearson_acc, 1.0, places=5)
        self.assertAlmostEqual(report.nrmse_pitch_percent, 0.0, places=5)
        self.assertAlmostEqual(report.nrmse_acc_percent, 0.0, places=5)
        self.assertAlmostEqual(report.cadence_pitch.relative_error_percent, 0.0, places=3)
        self.assertFalse(report.cadence_pitch.has_identity_swap)
        self.assertAlmostEqual(report.dtw_pitch.normalized_distance, 0.0, places=5)
        self.assertAlmostEqual(report.rigidity.sigma_rigidity, 0.0, places=6)

        # Summary row check
        row = report.to_summary_table_row("TestBaseline")
        self.assertEqual(row["Method"], "TestBaseline")
        self.assertEqual(row["Pearson Pitch (r)"], 1.0)
        self.assertEqual(row["Identity Swap"], "NO")

    def test_pitch_only_inputs(self):
        # Testing optional omission of acceleration and 3D track
        report = calculate_all_metrics(
            pitch_pred=self.pitch_gt,
            pitch_true=self.pitch_gt,
            fps_pred=self.fps,
        )

        self.assertIsNotNone(report.pearson_pitch)
        self.assertIsNone(report.pearson_acc)
        self.assertIsNone(report.cross_corr_acc)
        self.assertIsNone(report.nrmse_acc_percent)
        self.assertIsNone(report.rom_acc)
        self.assertIsNone(report.rigidity)
        self.assertIsNone(report.dtw_acc3d)

        # Dictionary serialization
        d = report.to_dict()
        self.assertIn("pearson_pitch", d)
        self.assertIsNone(d["pearson_acc"])

        # Summary table formatting
        row = report.to_summary_table_row("DELTA")
        self.assertEqual(row["Pearson Acc (r)"], "N/A")
        self.assertEqual(row["Rigidity 3D (sigma)"], "N/A")

    def test_identity_swap_reporting(self):
        # 1. Antiphase signal on 90 deg lateral view -> True Identity Swap
        pitch_antiphase = -self.pitch_gt

        report_lateral = calculate_all_metrics(
            pitch_pred=pitch_antiphase,
            pitch_true=self.pitch_gt,
            fps_pred=self.fps,
            angle="90",
        )
        self.assertTrue(report_lateral.swap_detection.has_identity_swap)
        self.assertFalse(report_lateral.swap_detection.is_depth_inversion_artifact)
        row_lateral = report_lateral.to_summary_table_row("SwapModelLateral")
        self.assertEqual(row_lateral["Identity Swap"], "YES")

        # 2. Antiphase signal on 0 deg axial view -> Depth Artifact, NOT Swap
        report_axial = calculate_all_metrics(
            pitch_pred=pitch_antiphase,
            pitch_true=self.pitch_gt,
            fps_pred=self.fps,
            angle="0",
        )
        self.assertFalse(report_axial.swap_detection.has_identity_swap)
        self.assertTrue(report_axial.swap_detection.is_depth_inversion_artifact)
        row_axial = report_axial.to_summary_table_row("SwapModelAxial")
        self.assertEqual(row_axial["Identity Swap"], "NO (Depth Artifact)")

        # 3. Vision signal with double frequency still detected by cadence FFT
        pitch_double_freq = 25.0 * np.sin(2 * np.pi * 2.5 * self.t) + 5.0
        report_double = calculate_all_metrics(
            pitch_pred=pitch_double_freq,
            pitch_true=self.pitch_gt,
            fps_pred=self.fps,
        )
        self.assertTrue(report_double.cadence_pitch.has_identity_swap)


if __name__ == "__main__":
    unittest.main()
