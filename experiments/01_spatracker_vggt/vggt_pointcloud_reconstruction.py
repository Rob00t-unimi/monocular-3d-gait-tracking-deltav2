"""
Exploratory experiment script for standalone VGGT (Visual Geometry Grounded Transformer) 2D/3D tracking.
Evaluates dense grid and specific femur keypoint tracking on short frame windows.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "vggt"))
sys.path.insert(0, str(PROJECT_ROOT))

from vggt.models.vggt import VGGT
from src.data_io.state_manager import videos_state_manager
from src.data_io.video_reader import video_provider


def create_dense_grid(h: int, w: int, stride: int = 16) -> torch.Tensor:
    """Creates a uniform 2D grid of (x, y) query points for dense tracking."""
    x = torch.linspace(0, w - 1, w // stride)
    y = torch.linspace(0, h - 1, h // stride)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def run_vggt_experiment(video_path_str: str, stride_grid: int = 16) -> None:
    """Runs VGGT inference on a validated video sequence and exports a debug overlay video."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"[INFO] Using device: {device} with dtype: {dtype}")

    video_path = Path(video_path_str)
    video_name = video_path.name
    state = videos_state_manager.load(video_name)

    if state.ref_frame_idx is None:
        print(f"[ERROR] Reference frame not found for {video_name}. Validate the video first.")
        return

    ref_idx = state.ref_frame_idx
    if not state.specific_keypoints:
        print(f"[ERROR] Specific femur keypoints not found for {video_name}.")
        return

    specific_points = torch.tensor(state.specific_keypoints, dtype=torch.float32)
    print(f"[INFO] Reference frame: {ref_idx}, Specific keypoints: {len(specific_points)}")

    metadata = video_provider.get_metadata(video_path)
    if metadata.frames_num == 0:
        print(f"[ERROR] Could not read video metadata: {video_path}")
        return

    frames = []
    print(f"[INFO] Reading {metadata.frames_num} frames...")
    for i in range(metadata.frames_num):
        frame, _ = video_provider.get_frame(video_path, i)
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not frames:
        print("[ERROR] No frames extracted from video.")
        return

    H, W = metadata.height, metadata.width
    patch_size = 14
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size

    if new_H != H or new_W != W:
        print(f"[INFO] Resizing from {W}x{H} to {new_W}x{new_H} for ViT patch compatibility")

    max_frames = 10
    ordered_indices = ([ref_idx] + [i for i in range(len(frames)) if i != ref_idx])[:max_frames]

    processed_frames = []
    for idx in ordered_indices:
        f = frames[idx]
        if new_H != H or new_W != W:
            f = cv2.resize(f, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        processed_frames.append(f)

    input_frames = torch.from_numpy(np.stack(processed_frames)).permute(0, 3, 1, 2).float() / 255.0
    input_frames = input_frames.to(device)

    scale_x = new_W / W
    scale_y = new_H / H
    scaled_specific = specific_points.clone()
    scaled_specific[:, 0] *= scale_x
    scaled_specific[:, 1] *= scale_y

    grid_points = create_dense_grid(new_H, new_W, stride=stride_grid)
    all_queries = torch.cat([scaled_specific, grid_points], dim=0).to(device)

    print(f"[INFO] Running VGGT inference on {len(processed_frames)} frames...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
            predictions = model(input_frames.unsqueeze(0), all_queries.unsqueeze(0))

    tracks = predictions["track"].squeeze(0).cpu().float().numpy()
    tracks[:, :, 0] /= scale_x
    tracks[:, :, 1] /= scale_y

    output_filename = f"vggt_debug_{video_path.stem}.mp4"
    out_path = str(PROJECT_ROOT / "results" / output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_path, fourcc, 5.0, (W, H))

    print(f"[INFO] Generating debug overlay video: {output_filename}")
    for s in tqdm(range(len(processed_frames))):
        img_bgr = cv2.cvtColor(frames[ordered_indices[s]], cv2.COLOR_RGB2BGR)

        for pt in tracks[s, len(specific_points) :]:
            cv2.circle(img_bgr, (int(round(pt[0])), int(round(pt[1]))), 1, (255, 255, 0), -1)

        for i in range(len(specific_points)):
            pt = tracks[s, i]
            pos = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(img_bgr, pos, 6, (255, 255, 255), -1)
            cv2.circle(img_bgr, pos, 4, (0, 0, 255), -1)

        label = f"Frame: {ordered_indices[s]} (REF: {ref_idx})"
        cv2.putText(img_bgr, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        video_writer.write(img_bgr)

    video_writer.release()
    print(f"\n[OK] Debug video saved: {out_path}")


if __name__ == "__main__":
    from src.data_io.dataset_scanner import DatasetScanner

    data_dir = str(PROJECT_ROOT / "data")
    scanner = DatasetScanner(data_dir)
    video_infos = list(scanner)

    if video_infos:
        target_video = None
        for info in video_infos:
            v_name = Path(info.video_path).name
            state = videos_state_manager.load(v_name)
            if state.ref_frame_idx is not None and state.specific_keypoints:
                target_video = str(PROJECT_ROOT / str(info.video_path))
                break

        if target_video:
            print(f"[START] Analyzing video: {target_video}")
            run_vggt_experiment(target_video, stride_grid=16)
        else:
            print("[ERROR] No validated video found in videos_state.json.")
    else:
        print("[ERROR] No videos discovered by DatasetScanner.")
