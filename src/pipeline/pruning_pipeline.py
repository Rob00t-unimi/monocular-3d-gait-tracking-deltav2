# -*- coding: utf-8 -*-
"""
CLI Runner for Geometric Pruning and Triplet Selection.

Features:
- Single-video execution: `python scripts/run_geometric_pruning.py --video P27_E2T2_90.mp4`
- Full-batch execution: `python scripts/run_geometric_pruning.py --batch`
- Automatic resume capability: skips already refined videos.
- Progress reporting with ETA and clean KeyboardInterrupt handling.
- Reads from data/inference_results/raw/ and writes to data/inference_results/refined/.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.postprocessing.geometric_pruner import GeometricPruner, RefinedTrackingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_geometric_pruning")


def prune_single_video(
    pruner: GeometricPruner,
    raw_npz_path: Path,
    output_dir: Path,
    resume: bool = True
) -> bool:
    """
    Executes geometric pruning on a single raw NPZ file and saves the refined result.
    Returns True if successfully processed or skipped, False if failed.
    """
    raw_npz_path = Path(raw_npz_path)
    if not raw_npz_path.exists():
        logger.error(f"Raw NPZ not found: {raw_npz_path}")
        return False

    v_name = raw_npz_path.name.replace("delta_v2_raw_", "").replace(".npz", "")
    output_path = output_dir / f"delta_v2_refined_{v_name}.npz"

    if resume and output_path.exists():
        logger.info(f"Skipping {v_name} (already refined at {output_path.name})")
        return True

    logger.info(f"Refining '{v_name}' (from {raw_npz_path.name})...")
    start_t = time.time()

    try:
        data = np.load(raw_npz_path, allow_pickle=True)
        track2d = data["track2d"]
        track3d_world = data["track3d_world"]
        ref_frame_idx = int(data.get("ref_frame_idx", 0))
        start_idx = int(data.get("start_idx", 0))
        fps = float(data.get("fps", 30.0))

        refined = pruner.refine(
            track2d=track2d,
            track3d_world=track3d_world,
            ref_frame_idx=ref_frame_idx,
            start_idx=start_idx,
            video_name=v_name,
            fps=fps
        )

        refined.save_npz(output_path)
        elapsed = time.time() - start_t
        logger.info(
            f"Finished {v_name} in {elapsed:.2f}s -> Saved {output_path.name} "
            f"(Survived: {len(refined.survived_indices)} pts, Triplet: {list(refined.triplet_indices)})"
        )
        return True
    except Exception as e:
        logger.error(f"Error refining {v_name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Geometric Pruning and Triplet Selection on Raw Tracking Outputs")
    parser.add_argument("--video", type=str, help="Specific video name to process (e.g. P27_E2T2_90.mp4)")
    parser.add_argument("--batch", action="store_true", help="Process all available raw NPZ files")
    parser.add_argument("--raw_dir", type=str, default="data/inference_results/raw", help="Directory containing raw NPZ files")
    parser.add_argument("--output_dir", type=str, default="data/inference_results/refined", help="Output directory for refined NPZ")
    parser.add_argument("--max_area_ratio", type=float, default=5.0, help="Maximum allowed convex hull area ratio over reference")
    parser.add_argument("--min_3d_ratio", type=float, default=0.05, help="Minimum 3D triangle area ratio over reference")
    parser.add_argument("--num_thigh_points", type=int, default=203, help="Number of initial thigh points to evaluate")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already refined files")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        sys.exit(1)

    pruner = GeometricPruner(
        max_area_ratio=args.max_area_ratio,
        min_3d_ratio=args.min_3d_ratio,
        num_thigh_points=args.num_thigh_points
    )

    if args.video:
        v_name = args.video
        target_npz = raw_dir / f"delta_v2_raw_{v_name}.npz"
        if not target_npz.exists():
            # Try finding without prefix or directly
            candidates = list(raw_dir.glob(f"*{v_name}*.npz"))
            if candidates:
                target_npz = candidates[0]
            else:
                logger.error(f"Raw NPZ not found for video '{v_name}' in {raw_dir}")
                sys.exit(1)

        ok = prune_single_video(
            pruner=pruner,
            raw_npz_path=target_npz,
            output_dir=output_dir,
            resume=args.resume
        )
        sys.exit(0 if ok else 1)

    elif args.batch:
        raw_files = sorted(list(raw_dir.glob("delta_v2_raw_*.npz")))
        if not raw_files:
            logger.warning(f"No raw NPZ files found in {raw_dir}")
            sys.exit(0)

        existing_refined = {f.name.replace("delta_v2_refined_", "").replace(".npz", "") for f in output_dir.glob("delta_v2_refined_*.npz")}
        total_files = len(raw_files)
        already_done = sum(1 for f in raw_files if f.name.replace("delta_v2_raw_", "").replace(".npz", "") in existing_refined)
        to_process = total_files - already_done if args.resume else total_files

        logger.info("=== Geometric Pruning Batch ===")
        logger.info(f"Total raw files       : {total_files}")
        logger.info(f"Already refined       : {already_done}")
        logger.info(f"Remaining to process  : {to_process}")
        logger.info(f"Output directory      : {output_dir}")
        logger.info("===============================")

        successful = already_done if args.resume else 0
        computed_times = []
        total_start = time.time()

        try:
            for idx, raw_path in enumerate(raw_files, 1):
                v_name = raw_path.name.replace("delta_v2_raw_", "").replace(".npz", "")
                is_already_done = args.resume and (output_dir / f"delta_v2_refined_{v_name}.npz").exists()

                if computed_times and not is_already_done:
                    avg_time = sum(computed_times) / len(computed_times)
                    remaining = total_files - idx + 1
                    eta_sec = avg_time * remaining
                    eta_str = f" | ETA: ~{eta_sec:.1f}s"
                else:
                    eta_str = ""

                logger.info(f"[{idx}/{total_files}] ({idx/total_files*100:.1f}%){eta_str} -> {v_name}")

                v_start = time.time()
                ok = prune_single_video(
                    pruner=pruner,
                    raw_npz_path=raw_path,
                    output_dir=output_dir,
                    resume=args.resume
                )

                if not is_already_done and ok:
                    computed_times.append(time.time() - v_start)
                    successful += 1

            total_elapsed = time.time() - total_start
            logger.info(f"Pruning batch completed: {successful}/{total_files} files available in {output_dir.name} (Elapsed: {total_elapsed:.1f}s).")
        except KeyboardInterrupt:
            logger.warning("\n[INTERRUPTED] Pruning batch interrupted by user.")
            sys.exit(0)
    else:
        logger.info("No action specified. Use --video <name> for single test or --batch to process all available raw files.")


if __name__ == "__main__":
    main()
