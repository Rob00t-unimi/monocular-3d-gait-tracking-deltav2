"""Unified Parametric Command-Line Interface for monocular-3d-gait-tracking-deltav2 (Gait360).

Provides a single entrypoint (`gait360` or `python -m src.cli`) with seven subcommands:
  - `scan`: Scan dataset, align multi-view video/OpenPose/IMU trials, and build candidate quality cache.
  - `select-cohort`: Select stratified 40/60 benchmark video cohort across 8 viewing perspectives (or launch GUI).
  - `preprocess`: Run 2D vision thigh segmentation, reference frame selection, and 3-point anchor extraction.
  - `track`: Run dense 3D trajectory inference (`deltav2` or `dummy`) with automatic `.npz` resume/caching.
  - `prune`: Run the 3-Stage Geometric Pruner (anisotropic spatial DBSCAN, kinematic MAD, and rigid triplet).
  - `evaluate`: Compute 3D SVD-aligned kinematic/geometric metrics and regenerate JSON, CSV, and Markdown reports.
  - `inspect`: Launch interactive 2D/3D/waveform inspection GUIs (`keypoints`, `cohort`, `3d`, `signals`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Construct the unified CLI argument parser with all 7 stage subcommands."""
    parser = argparse.ArgumentParser(
        prog="gait360",
        description=(
            "monocular-3d-gait-tracking-deltav2 (Gait360): End-to-End 360-Degree Monocular "
            "3D Gait Analysis Pipeline via DELTAv2 and 3-Stage Geometric Post-Processing."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline stage subcommand")

    # 1. scan
    p_scan = subparsers.add_parser(
        "scan",
        help="Scan multi-view dataset, align video/OpenPose/IMU trials, and build candidate quality cache",
    )
    p_scan.add_argument(
        "--mode",
        choices=["all", "scan", "candidates"],
        default="all",
        help="Execution mode: 'scan' (dataset map only), 'candidates' (quality cache only), or 'all' (default)",
    )
    p_scan.add_argument(
        "--dataset-dir",
        default="data/Gait_Dataset",
        help="Relative or absolute path to the multi-view Gait Dataset directory",
    )
    p_scan.add_argument(
        "--cache-path",
        default="data/cached_dataset_map.json",
        help="Output path for the serialized dataset alignment map JSON",
    )

    # 2. select-cohort
    p_cohort = subparsers.add_parser(
        "select-cohort",
        help="Select deterministic stratified 40-video benchmark cohort across 8 viewing angles or launch GUI",
    )
    p_cohort.add_argument(
        "--gui",
        action="store_true",
        help="Launch the interactive Tkinter Stratified Cohort Selector GUI",
    )

    # 3. preprocess
    p_prep = subparsers.add_parser(
        "preprocess",
        help="Run 2D thigh segmentation, optimal reference frame selection, and 3-point anchor extraction",
    )
    p_prep.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to pipeline YAML configuration (default: configs/default_config.yaml)",
    )
    p_prep.add_argument(
        "--gui",
        action="store_true",
        help="Launch the interactive 2D Keypoint & Reference Frame Validator GUI after extraction",
    )

    # 4. track
    p_track = subparsers.add_parser(
        "track",
        help="Execute 3D point tracking inference (DELTAv2 or OpenPose Dummy baseline) with .npz caching",
    )
    p_track.add_argument(
        "--tracker",
        choices=["deltav2", "dummy"],
        default="deltav2",
        help="3D tracking architecture to execute (default: deltav2)",
    )
    p_track.add_argument(
        "--video",
        default=None,
        help="Optional single video filename to track (e.g., P27_E2T2_90.mp4); runs full cohort if omitted",
    )
    p_track.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if cached raw .npz trajectory tensor already exists",
    )

    # 5. prune
    p_prune = subparsers.add_parser(
        "prune",
        help="Run 3-Stage Geometric Pruner on raw DELTAv2 .npz tensors to extract pruned cloud and rigid triplet",
    )
    p_prune.add_argument(
        "--video",
        default=None,
        help="Optional single video filename to prune; processes all raw .npz tensors if omitted",
    )
    p_prune.add_argument(
        "--z-weight",
        type=float,
        default=0.25,
        help="Anisotropic monocular depth Z-axis weight for Stage 1-2 pruning (default: 0.25)",
    )
    p_prune.add_argument(
        "--min-altitude",
        type=float,
        default=0.035,
        help="Minimum physical triangle altitude in meters for Stage 3 normal-twist guard (default: 0.035)",
    )
    p_prune.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=8.0,
        help="Maximum triangle aspect ratio for Stage 3 normal-twist guard (default: 8.0)",
    )

    # 6. evaluate
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Compute 3D SVD-aligned kinematic/geometric metrics and regenerate JSON, CSV, and Markdown reports",
    )
    p_eval.add_argument(
        "--reports-only",
        action="store_true",
        help="Regenerate Markdown reports from existing results/benchmark_40_metrics.json without recomputing",
    )

    # 7. inspect
    p_insp = subparsers.add_parser(
        "inspect",
        help="Launch interactive graphical inspectors for 2D keypoints, cohort selection, 3D clouds, or waveforms",
    )
    p_insp.add_argument(
        "--mode",
        choices=["keypoints", "cohort", "3d", "signals"],
        default="3d",
        help="Interactive GUI mode to launch (default: 3d)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI arguments and dispatch to the corresponding pipeline module."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        from src.pipeline.dataset_pipeline import run_dataset_pipeline

        run_dataset_pipeline(
            mode=args.mode,
            dataset_dir=args.dataset_dir,
            cache_path=args.cache_path,
        )
        return 0

    if args.command == "select-cohort":
        if args.gui:
            import tkinter as tk
            from src.ui.cohort_selector_gui import VideoSelectorApp

            root = tk.Tk()
            app = VideoSelectorApp(root)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()
        else:
            from src.pipeline.cohort_selector import main as cohort_main

            cohort_main()
        return 0

    if args.command == "preprocess":
        from src.pipeline.vision_pipeline import main as vision_main

        sys.argv = ["gait360 preprocess", "--config", args.config]
        vision_main()
        return 0

    if args.command == "track":
        from src.pipeline.tracking_pipeline import main as track_main

        cli_args = ["gait360 track"]
        if args.video:
            cli_args.extend(["--video", args.video])
        if args.force:
            cli_args.append("--force")
        sys.argv = cli_args
        track_main()
        return 0

    if args.command == "prune":
        from src.pipeline.pruning_pipeline import main as prune_main

        sys.argv = ["gait360 prune"]
        prune_main()
        return 0

    if args.command == "evaluate":
        from src.benchmark.evaluator import main as eval_main
        from src.benchmark.report_generator import generate_reports

        if not args.reports_only:
            eval_main()
        generate_reports()
        return 0

    if args.command == "inspect":
        import tkinter as tk

        if args.mode == "cohort":
            from src.ui.cohort_selector_gui import VideoSelectorApp

            root = tk.Tk()
            app = VideoSelectorApp(root)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()
        elif args.mode == "3d":
            from src.ui.trajectory_3d_inspector import Unified3DInspector

            root = tk.Tk()
            Unified3DInspector(root)
            root.mainloop()
        elif args.mode == "signals":
            from src.ui.kinematic_signal_inspector import UnifiedSignalInspector

            root = tk.Tk()
            UnifiedSignalInspector(root)
            root.mainloop()
        elif args.mode == "keypoints":
            from src.pipeline.vision_pipeline import main as vision_main

            sys.argv = ["gait360 preprocess", "--config", "configs/default_config.yaml"]
            vision_main()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
