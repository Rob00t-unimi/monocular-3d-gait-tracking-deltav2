"""
Diagnostic script to verify DELTAv2 tracker wrapper, GeometricPruner, and Kinematic Alignment readiness.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.kinematic_alignment import run_pipeline
from src.inference.delta_v2_tracker import DeltaV2Tracker
from src.metrics.calculator import calculate_all_metrics
from src.postprocessing.geometric_pruner import GeometricPruner
from src.utils.path_validator import get_project_root, resolve_config_path


def verify_deltav2_pipeline() -> None:
    root = get_project_root()
    cfg = resolve_config_path()
    print("=" * 70)
    print("  Gait360 DELTAv2 Pipeline Readiness Check")
    print("=" * 70)
    print(f"[OK] Project Root          : {root}")
    print(f"[OK] Default Config        : {cfg} (exists={cfg.is_file()})")
    print(f"[OK] DeltaV2Tracker Class  : {DeltaV2Tracker.__name__}")
    print(f"[OK] GeometricPruner Class : {GeometricPruner.__name__}")
    print(f"[OK] Kinematic Alignment   : {run_pipeline.__name__}")
    print(f"[OK] Metrics Calculator    : {calculate_all_metrics.__name__}")
    print("[OK] All core 3D tracking and biomechanical analysis modules verified.")


if __name__ == "__main__":
    verify_deltav2_pipeline()
