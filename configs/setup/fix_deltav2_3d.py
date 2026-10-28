"""
Setup utility verifying DELTAv2 3D predictor configuration and MediaPipe Pose Landmarker assets.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from src.core.constants import MEDIAPIPE_POSE_MODEL_URL
from src.utils.path_validator import get_project_root


def verify_and_prepare_deltav2_assets() -> Path:
    """Verifies DELTAv2 submodule presence and caches the MediaPipe Pose Landmarker task file."""
    root = get_project_root()
    deltav2_dir = root / "models" / "DELTAv2"
    asset_path = root / "models" / "mediapipe" / "pose_landmarker_full.task"

    if not deltav2_dir.exists():
        print(f"[WARNING] DELTAv2 directory not found at {deltav2_dir}")
    else:
        print(f"[OK] DELTAv2 directory verified at {deltav2_dir}")

    if not asset_path.exists():
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading MediaPipe Pose Landmarker model to {asset_path}...")
        urllib.request.urlretrieve(MEDIAPIPE_POSE_MODEL_URL, asset_path)
    print(f"[OK] MediaPipe asset ready at {asset_path}")
    return asset_path


if __name__ == "__main__":
    verify_and_prepare_deltav2_assets()
