"""Unit and integration tests for the unified CLI (`src/cli.py`), pipeline orchestrators (`src/pipeline/`), and benchmark modules (`src/benchmark/`)."""

import unittest
from pathlib import Path

from src.cli import build_parser
from src.pipeline import cohort_selector, dataset_pipeline, pruning_pipeline, tracking_pipeline, vision_pipeline
from src.benchmark import evaluator, report_generator
from src.ui import KeypointValidatorGUI, ValidatorGUI


class TestUnifiedCLIAndPipelines(unittest.TestCase):
    """Verify the unified parametric CLI subcommands and pipeline module contracts."""

    def setUp(self) -> None:
        self.parser = build_parser()

    def test_cli_align_dataset_subcommand(self) -> None:
        args = self.parser.parse_args(["align-dataset", "--manifest", "data/dataset_manifest.json", "--dir", "data/Gait_Dataset", "--dry-run"])
        self.assertEqual(args.command, "align-dataset")
        self.assertEqual(args.manifest, "data/dataset_manifest.json")
        self.assertEqual(args.dataset_dir, "data/Gait_Dataset")
        self.assertTrue(args.dry_run)

    def test_dataset_aligner_mock(self) -> None:
        import tempfile, json
        from src.data_io.dataset_aligner import DatasetAligner

        aligner = DatasetAligner()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            f1 = tmppath / "subdir" / "file1.txt"
            f1.parent.mkdir(parents=True)
            f1.write_text("sample content for hashing", encoding="utf-8")

            h1 = aligner.compute_file_hash(str(f1))
            self.assertTrue(len(h1) > 0)

            manifest_dict = {"canonical/file1.txt": h1}
            m_path = tmppath / "manifest.json"
            with open(m_path, "w", encoding="utf-8") as mf:
                json.dump(manifest_dict, mf)

            summary = aligner.align(str(m_path), str(tmppath), dry_run=True)
            self.assertEqual(summary["to_move_count"], 1)
            self.assertEqual(summary["missing_count"], 0)

    def test_cli_scan_subcommand(self) -> None:
        args = self.parser.parse_args(["scan", "--mode", "candidates", "--dataset-dir", "data/Gait_Dataset"])
        self.assertEqual(args.command, "scan")
        self.assertEqual(args.mode, "candidates")
        self.assertEqual(args.dataset_dir, "data/Gait_Dataset")

    def test_cli_select_cohort_subcommand(self) -> None:
        args = self.parser.parse_args(["select-cohort", "--gui"])
        self.assertEqual(args.command, "select-cohort")
        self.assertTrue(args.gui)

    def test_cli_preprocess_subcommand(self) -> None:
        args = self.parser.parse_args(["preprocess", "--config", "configs/default_config.yaml"])
        self.assertEqual(args.command, "preprocess")
        self.assertEqual(args.config, "configs/default_config.yaml")

    def test_cli_track_subcommand(self) -> None:
        args = self.parser.parse_args(["track", "--tracker", "deltav2", "--video", "P27_E2T2_90.mp4", "--force"])
        self.assertEqual(args.command, "track")
        self.assertEqual(args.tracker, "deltav2")
        self.assertEqual(args.video, "P27_E2T2_90.mp4")
        self.assertTrue(args.force)

    def test_cli_prune_subcommand(self) -> None:
        args = self.parser.parse_args(
            ["prune", "--z-weight", "0.25", "--min-altitude", "0.035", "--max-aspect-ratio", "8.0"]
        )
        self.assertEqual(args.command, "prune")
        self.assertAlmostEqual(args.z_weight, 0.25)
        self.assertAlmostEqual(args.min_altitude, 0.035)
        self.assertAlmostEqual(args.max_aspect_ratio, 8.0)

    def test_cli_evaluate_subcommand(self) -> None:
        args = self.parser.parse_args(["evaluate", "--reports-only"])
        self.assertEqual(args.command, "evaluate")
        self.assertTrue(args.reports_only)

    def test_cli_inspect_subcommand(self) -> None:
        for mode in ("keypoints", "cohort", "3d", "signals"):
            args = self.parser.parse_args(["inspect", "--mode", mode])
            self.assertEqual(args.command, "inspect")
            self.assertEqual(args.mode, mode)

    def test_pipeline_and_benchmark_modules_importable(self) -> None:
        self.assertTrue(hasattr(dataset_pipeline, "run_dataset_pipeline"))
        self.assertTrue(hasattr(cohort_selector, "main"))
        self.assertTrue(hasattr(vision_pipeline, "main"))
        self.assertTrue(hasattr(tracking_pipeline, "main"))
        self.assertTrue(hasattr(pruning_pipeline, "main"))
        self.assertTrue(hasattr(evaluator, "main"))
        self.assertTrue(hasattr(report_generator, "generate_reports"))
        self.assertIs(KeypointValidatorGUI, ValidatorGUI)


if __name__ == "__main__":
    unittest.main()
