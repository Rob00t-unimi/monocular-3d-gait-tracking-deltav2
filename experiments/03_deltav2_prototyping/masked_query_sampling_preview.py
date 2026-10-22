import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Nuova API di MediaPipe Tasks (Compatibile con 0.10.x)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("\n[!] Error: MediaPipe not found. Run: pip install mediapipe\n")
    sys.exit(1)

# ==========================================================
# CONFIGURAZIONE CAMPIONAMENTO MASCHERATO (MediaPipe Pose)
# ==========================================================
TARGET_NUM = 200 # Points inside thigh triangle
BODY_NUM = 800   # Total points on subject silhouette
# ==========================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_io.dataset_scanner import DatasetScanner
from src.data_io.video_reader import video_provider
from src.data_io.state_manager import videos_state_manager

def get_person_mask_mediapipe(frame_rgb):
    """Rileva la sagoma usando Pose Landmarker (Immagine Originale)"""
    model_path = SCRIPT_DIR / "pose_landmarker_full.task"
    if not model_path.exists():
        import urllib.request
        print("Downloading Robust Pose Model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect(mp_image)
        if results.segmentation_masks:
            return results.segmentation_masks[0].numpy_view() > 0.5
        return None

def sample_full_silhouette(mask, num_samples):
    """Sample random points inside the segmented mask"""
    if mask is None: return np.zeros((0, 2))
    y_coords, x_coords = np.where(mask.squeeze())
    if len(x_coords) == 0: return np.zeros((0, 2))
    all_pix = np.stack([x_coords, y_coords], axis=1)
    if len(all_pix) > num_samples:
        np.random.seed(42)
        idx = np.random.choice(len(all_pix), num_samples, replace=False)
        return all_pix[idx]
    return all_pix

def sample_triangle(p1, p2, p3, n=TARGET_NUM):
    np.random.seed(42); u, v = np.random.rand(n), np.random.rand(n)
    mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
    target = next((v for v in scanner if v.name == args.video), None)
    state = videos_state_manager.load(target.name)
    frame, _ = video_provider.get_frame(target.video_path, state.ref_frame_idx)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("Running MediaPipe Segmentation on original image...")
    mask = get_person_mask_mediapipe(frame_rgb)
    
    print(f"Sampling {BODY_NUM} points from silhouette...")
    body_pts = sample_full_silhouette(mask, BODY_NUM)
    
    kps = np.array(state.specific_keypoints)
    thigh_pts = sample_triangle(kps[0], kps[1], kps[2])

    plt.figure(figsize=(12, 12))
    plt.imshow(frame_rgb)
    
    if mask is not None:
        overlay = np.zeros((*mask.squeeze().shape, 4))
        overlay[mask.squeeze(), 0] = 1.0; overlay[mask.squeeze(), 3] = 0.3
        plt.imshow(overlay)
    
    if len(body_pts) > 0:
        plt.scatter(body_pts[:, 0], body_pts[:, 1], c='yellow', s=2, alpha=0.8, label='Silhouette Context')
    plt.scatter(thigh_pts[:, 0], thigh_pts[:, 1], c='cyan', s=10, alpha=0.8, label='Target Thigh')
    plt.scatter(kps[:, 0], kps[:, 1], c='red', marker='X', s=100, label='Manual Joints')
    
    plt.title(f"MediaPipe Pure Silhouette Sampling: {args.video}")
    plt.legend(); plt.axis('off'); plt.show()

if __name__ == "__main__":
    main()
