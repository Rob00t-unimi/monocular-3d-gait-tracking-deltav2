# -*- coding: utf-8 -*-
"""
CLI Runner for DELTAv2 Raw 3D Point Tracking Inference.

Features:
- Single-video execution: `python scripts/run_inference_raw.py --video P27_E2T2_90.mp4`
- Full-batch execution: `python scripts/run_inference_raw.py --batch`
- Automatic resume capability: skips already processed videos.
- Saves raw unprojected 3D coordinates and 2D tracks into data/inference_results/raw/.
- Manages GPU memory cleanly across iterations.
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_io.state_manager import videos_state_manager
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_inference_raw")


def run_single_video(
    tracker: Any,
    video_name: str,
    video_path: Path,
    output_dir: Path,
    max_frames: int = 250,
    resume: bool = True
) -> bool:
    """
    Executes raw tracking for one video and saves its result to NPZ.
    Returns True if successfully processed, False if skipped or failed.
    """
    output_path = output_dir / f"delta_v2_raw_{video_name}.npz"
    if resume and output_path.exists():
        logger.info(f"Skipping {video_name} (already computed at {output_path.name})")
        return True

    state = videos_state_manager.load(video_name)
    if state is None or state.ref_frame_idx is None:
        logger.error(f"Cannot process {video_name}: No valid reference frame found in videos_state.json")
        return False

    ref_frame_idx = state.ref_frame_idx
    kps_init = np.array(state.specific_keypoints, dtype=np.float32)
    if len(kps_init) < 3:
        logger.error(f"Cannot process {video_name}: Fewer than 3 reference keypoints available")
        return False

    logger.info(f"Starting raw tracking for '{video_name}' (Ref Frame: {ref_frame_idx})...")
    start_t = time.time()

    try:
        result = tracker.track_video(
            video_path=video_path,
            ref_frame_idx=ref_frame_idx,
            specific_keypoints=kps_init,
            max_frames=max_frames
        )
        result.save_npz(output_path)
        elapsed = time.time() - start_t
        logger.info(f"Finished {video_name} in {elapsed:.1f}s -> Saved {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error tracking {video_name}: {e}", exc_info=True)
        return False
    finally:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run DELTAv2 Raw Point Tracking Inference")
    parser.add_argument("--video", type=str, help="Specific video name to process (e.g. P27_E2T2_90.mp4)")
    parser.add_argument("--batch", action="store_true", help="Run batch inference on the full benchmark dataset")
    parser.add_argument("--csv", type=str, default="data/selected_inference_videos_40.csv", help="Path to benchmark CSV")
    parser.add_argument("--output_dir", type=str, default="data/inference_results/raw", help="Output directory for raw NPZ")
    parser.add_argument("--max_frames", type=int, default=250, help="Maximum video frames to track")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already processed videos")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose model logs")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_ROOT / args.csv
    if not csv_path.exists():
        logger.error(f"Benchmark CSV not found at {csv_path}")
        sys.exit(1)

    df_benchmark = pd.read_csv(csv_path)

    # Initialize tracker lazily
    from src.inference.delta_v2_tracker import DeltaV2Tracker
    tracker = DeltaV2Tracker(
        device=args.device,
        verbose=args.verbose
    )

    if args.video:
        row = df_benchmark[df_benchmark["video_name"] == args.video]
        if row.empty:
            logger.error(f"Video {args.video} not found in {args.csv}")
            sys.exit(1)
        v_path = Path(row.iloc[0]["video_path"])
        success = run_single_video(
            tracker=tracker,
            video_name=args.video,
            video_path=v_path,
            output_dir=output_dir,
            max_frames=args.max_frames,
            resume=args.resume
        )
        sys.exit(0 if success else 1)

    elif args.batch:
        output_files = list(output_dir.glob("delta_v2_raw_*.npz"))
        existing_names = {f.name.replace("delta_v2_raw_", "").replace(".npz", "") for f in output_files}
        
        total_videos = len(df_benchmark)
        already_done = sum(1 for _, r in df_benchmark.iterrows() if r["video_name"] in existing_names)
        to_process = total_videos - already_done if args.resume else total_videos
        
        logger.info(f"=== DELTAv2 Batch Inference ===")
        logger.info(f"Total benchmark videos : {total_videos}")
        logger.info(f"Already completed      : {already_done}")
        logger.info(f"Remaining to compute   : {to_process}")
        logger.info(f"Output directory       : {output_dir}")
        logger.info("=================================")

        successful = already_done if args.resume else 0
        computed_times = []
        total_start = time.time()

        try:
            for idx, (_, row) in enumerate(df_benchmark.iterrows(), 1):
                v_name = row["video_name"]
                v_path = Path(row["video_path"])
                is_already_done = args.resume and (output_dir / f"delta_v2_raw_{v_name}.npz").exists()
                
                # Estimate remaining time based on processed videos
                if computed_times and not is_already_done:
                    avg_time = sum(computed_times) / len(computed_times)
                    remaining_vids = total_videos - idx + 1
                    eta_min = (avg_time * remaining_vids) / 60.0
                    eta_str = f" | ETA: ~{eta_min:.1f} min"
                else:
                    eta_str = ""

                logger.info(f"[{idx}/{total_videos}] ({idx/total_videos*100:.1f}%){eta_str} -> {v_name}")
                
                v_start = time.time()
                ok = run_single_video(
                    tracker=tracker,
                    video_name=v_name,
                    video_path=v_path,
                    output_dir=output_dir,
                    max_frames=args.max_frames,
                    resume=args.resume
                )
                
                if not is_already_done and ok:
                    computed_times.append(time.time() - v_start)
                    successful += 1

            total_time = time.time() - total_start
            logger.info(f"Batch completed successfully: {successful}/{total_videos} videos available in {output_dir.name} (Elapsed: {total_time/60:.1f} min).")
        except KeyboardInterrupt:
            current_done = len(list(output_dir.glob("delta_v2_raw_*.npz")))
            logger.warning("\n[INTERRUPTED] Execution interrupted by user.")
            logger.warning(f"Completed videos ({current_done}/{total_videos}) have been safely saved to disk.")
            logger.warning("You can resume at any time by re-running the same command: only remaining videos will be processed.")
            sys.exit(0)
    else:
        logger.info("No action specified. Use --video <name> for single test or --batch to run the full benchmark.")


if __name__ == "__main__":
    main()
