"""
Main CLI Entry Point for the Gait360 3D Tracking & Biomechanical Analysis Pipeline.
Executes linear pipeline stages:
  1. Dataset Scanning & Validation
  2. Ground Truth IMU Signal Processing (Physiological Bandpass & Welch Cutoffs)
  3. Optional 1D / 3D Signal Visualization
  4. Video Reference Frame & Femur Triangle Preparation (Manual GUI or Automated)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import DEFAULT_ARTICULATION
from src.data_io.dataset_scanner import DatasetScanner
from src.data_io.video_reader import video_provider
from src.preprocessing.signals.signal_orchestrator import SignalOrchestrator
from src.preprocessing.video_prepper import video_service
from src.ui.trajectory_3d_inspector import inspect_signals_3d
from src.ui.kinematic_signal_inspector import inspect_signals
from src.ui.keypoint_validator_gui import validate_first_frame_of_all_videos
from src.data_io.state_manager import videos_state_manager
from src.utils.path_validator import load_pipeline_config, resolve_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _get_valid_videos(scanner, invalid_csvs, articulation: str = DEFAULT_ARTICULATION):
    """Filters for videos with valid IMU data on the target articulation."""
    valid_list = []
    for v in scanner:
        if (
            v.sensor_data_path
            and v.sensor_position.lower() == articulation.lower()
            and v.sensor_data_path not in invalid_csvs
        ):
            valid_list.append(v)
    return valid_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Gait360 3D Tracking & Biomechanical Pipeline Runner")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to pipeline YAML configuration (defaults to configs/default_config.yaml; supports legacy config.yaml fallback)",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Disable Step 4: Video Preparation (Manual/Auto)",
    )
    parser.add_argument(
        "--view1D",
        nargs="?",
        const="all",
        choices=["valid", "invalid", "all"],
        help="Enable Step 3: 1D Signal Visualization (defaults to 'all' if flag is present)",
    )
    parser.add_argument(
        "--view3D",
        nargs="?",
        const="all",
        choices=["valid", "invalid", "all"],
        help="Enable Step 3: 3D Signal Visualization (defaults to 'all' if flag is present)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Change Step 4 to AUTOMATED mode (default is Manual GUI)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Base data directory override")
    parser.add_argument(
        "--force-gt",
        action="store_true",
        help="Force recalculation of cutoff frequencies and re-filtering (ignores cache)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="In Step 4, skip videos that are already validated in videos_state.json",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="In Step 4, skip all videos until this specific video name is reached",
    )

    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config)
    config = load_pipeline_config(cfg_path)
    data_dir = args.data_dir or config.get("paths", {}).get("data_dir", "data/")
    articulation = config.get("project", {}).get("default_articulation", DEFAULT_ARTICULATION)

    logger.info(f"Loaded configuration from {cfg_path}")

    # =========================================================================
    # STEP 1: Dataset Initialization
    # =========================================================================
    logger.info(f"--- Step 1: Initializing Dataset Scanner at {data_dir} ---")
    scanner = DatasetScanner(data_dir)

    # =========================================================================
    # STEP 2: Ground Truth Processing (Cache-aware)
    # =========================================================================
    logger.info("--- Step 2: Ground Truth Processing ---")
    orchestrator = SignalOrchestrator()
    invalid_csvs, _ = orchestrator.run_pipeline(scanner, force_recalc=args.force_gt)

    # =========================================================================
    # STEP 3: Optional Signal Visualization
    # =========================================================================
    if args.view1D:
        logger.info(f"--- Step 3: Visualizing 1D {args.view1D} signals ---")
        inspect_signals(mode=args.view1D, data_dir=data_dir)

    if args.view3D:
        logger.info(f"--- Step 3: Visualizing 3D {args.view3D} signals ---")
        inspect_signals_3d(mode=args.view3D, data_dir=data_dir)

    # =========================================================================
    # STEP 4: Video Evaluation (Enabled by Default: Manual)
    # =========================================================================
    if not args.skip_video:
        valid_videos = _get_valid_videos(scanner, invalid_csvs, articulation=articulation)

        if args.start_from:
            found = False
            filtered_list = []
            for v in valid_videos:
                if v.name == args.start_from:
                    found = True
                if found:
                    filtered_list.append(v)
            if not found:
                logger.warning(f"Video {args.start_from} not found in valid dataset. Starting from beginning.")
            else:
                logger.info(f"Skipping until {args.start_from}. {len(filtered_list)} videos remaining.")
                valid_videos = filtered_list

        if args.auto:
            if args.resume:
                all_states = videos_state_manager._load_full_cache()
                unprocessed = [
                    v for v in valid_videos
                    if v.name not in all_states or (
                        all_states[v.name].get('ref_frame_idx') is None and
                        not all_states[v.name].get('specific', {}).get('discarded') and
                        not all_states[v.name].get('fullbody', {}).get('discarded')
                    )
                ]
                logger.info(f"--- Step 4: Fast resume active: {len(valid_videos) - len(unprocessed)}/{len(valid_videos)} videos already completed in videos_state.json ---")
                valid_videos = unprocessed

            if valid_videos:
                logger.info(f"Starting AUTOMATED Evaluation for {len(valid_videos)} remaining videos...")
                with logging_redirect_tqdm():
                    for v_info in tqdm(valid_videos, desc="Auto-Evaluating", unit="vid"):
                        metadata = video_provider.get_metadata(v_info.video_path)
                        video_service.get_reference_frame(v_info, metadata, skip_if_already_done=args.resume)
            else:
                logger.info("All videos already validated in videos_state.json. Skipping automated loop.")
        else:
            logger.info("--- Step 4: Launching MANUAL Validation GUI ---")
            with logging_redirect_tqdm():
                validate_first_frame_of_all_videos(valid_videos, articulation, args.resume)
    else:
        logger.info("--- Step 4: Video Preparation Skipped ---")

    logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    main()
