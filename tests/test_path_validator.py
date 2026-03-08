"""
Unit tests for path validation, configuration resolution, and project root discovery.
"""

import tempfile
import unittest
from pathlib import Path

from src.utils.path_validator import (
    get_project_root,
    load_pipeline_config,
    resolve_config_path,
    resolve_project_path,
    validate_directory_exists,
    validate_file_exists,
)


class TestPathValidator(unittest.TestCase):
    """Tests for src.utils.path_validator functions."""

    def test_get_project_root_exists(self):
        root = get_project_root()
        self.assertTrue(root.is_dir())
        self.assertTrue((root / "src").is_dir())

    def test_resolve_config_path_default_and_legacy_fallback(self):
        default_cfg = resolve_config_path(None)
        self.assertEqual(default_cfg.name, "default_config.yaml")
        self.assertTrue(default_cfg.is_file())

        legacy_cfg = resolve_config_path("config.yaml")
        self.assertEqual(legacy_cfg.name, "default_config.yaml")
        self.assertTrue(legacy_cfg.is_file())

    def test_resolve_legacy_dataset_marker(self):
        legacy_str = r"C:\old_workspace\repo\data\Gait_Dataset\Participant_1\P1.csv"
        resolved = resolve_project_path(legacy_str)
        self.assertIn("Gait_Dataset", str(resolved))
        self.assertTrue(str(resolved).endswith("P1.csv"))

    def test_validate_file_and_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tmp_file = tmp_dir / "sample.txt"
            tmp_file.write_text("ok", encoding="utf-8")

            self.assertEqual(validate_directory_exists(tmp_dir), tmp_dir.resolve())
            self.assertEqual(validate_file_exists(tmp_file), tmp_file.resolve())

            with self.assertRaises(FileNotFoundError):
                validate_file_exists(tmp_dir / "missing.txt")

    def test_load_pipeline_config(self):
        cfg = load_pipeline_config("config.yaml")
        if cfg:
            self.assertIn("project", cfg)
            self.assertIn("tracking", cfg)


if __name__ == "__main__":
    unittest.main()
