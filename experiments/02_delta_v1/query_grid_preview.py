import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Configurazione Percorsi
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.data_io.dataset_scanner import DatasetScanner
from src.data_io.video_reader import video_provider
from src.data_io.state_manager import videos_state_manager

def get_person_body_grid(keypoints, W, H, num_points=400):
    """Logica 'sartoriale' per avvolgere la persona senza troppa aria"""
    x_min, x_max = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
    y_min, y_max = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])
    
    dy = y_max - y_min
    dx = x_max - x_min
    
    # Euristiche basate su proporzioni umane (Thigh as unit)
    pad_y_up = dy * 17    # Busto + Testa
    pad_y_down = dy * 9.0  # Tibia + Piedi
    pad_x = max(dx * 9.0, dy * 1.8) # Shoulder width / stride span
    
    person_y_min = max(0, y_min - pad_y_up)
    person_y_max = min(H, y_max + pad_y_down)
    person_x_min = max(0, x_min - pad_x)
    person_x_max = min(W, x_max + pad_x)
    
    # Genera griglia regolare
    aspect_ratio = (person_y_max - person_y_min) / (person_x_max - person_x_min + 1e-6)
    grid_h = int(np.sqrt(num_points * aspect_ratio))
    grid_w = num_points // grid_h
    
    xs = np.linspace(person_x_min, person_x_max, grid_w)
    ys = np.linspace(person_y_min, person_y_max, grid_h)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv.flatten(), yv.flatten()], axis=1)

def sample_triangle(p1, p2, p3, n=200):
    np.random.seed(42); u, v = np.random.rand(n), np.random.rand(n)
    mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3

def main():
    parser = argparse.ArgumentParser(description="Preview DELTA Query Points")
    parser.add_argument("--video", type=str, help="Video name")
    args = parser.parse_args()

    scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
    target_video = None
    for v in scanner:
        if args.video and v.name != args.video: continue
        state = videos_state_manager.load(v.name); 
        if state.ref_frame_idx is not None: target_video, target_state = v, state; break
    
    if not target_video: print("No calibrated video found."); return

    f, _ = video_provider.get_frame(target_video.video_path, target_state.ref_frame_idx)
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    kps = np.array(target_state.specific_keypoints)
    thigh_pts = sample_triangle(kps[0], kps[1], kps[2], 200)
    body_grid = get_person_body_grid(kps, W, H, 600)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    # Plot Points
    plt.scatter(body_grid[:, 0], body_grid[:, 1], c='cyan', s=10, alpha=0.5, label='Body Context (300 pts)')
    plt.scatter(thigh_pts[:, 0], thigh_pts[:, 1], c='red', s=10, alpha=0.7, label='Thigh Surface (200 pts)')
    plt.scatter(kps[:, 0], kps[:, 1], c='yellow', s=50, edgecolors='black', marker='X', label='Target Keypoints (3 pts)')
    
    plt.title(f"DELTA Query Preview - {target_video.name}\nFrame: {target_state.ref_frame_idx}")
    plt.legend()
    plt.axis('off')
    
    print(f"Showing preview for {target_video.name}. Total points: {len(kps)+len(thigh_pts)+len(body_grid)}")
    plt.show()

if __name__ == "__main__":
    main()
