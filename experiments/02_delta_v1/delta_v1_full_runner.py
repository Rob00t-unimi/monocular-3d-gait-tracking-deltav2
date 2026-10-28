import sys
import os

# --- PARAMETRI DI CONFIGURAZIONE (Modificabili) ---
MAX_FRAMES = 250         # Numero massimo di frame da processare
N_ITERS = 16             # Iterazioni di raffinamento (Standard: 12, High-Stability: 16-24)
SUPPORT_GRID_SIZE = 12   # Dimensione griglia di supporto (Standard: 8, High-Stability: 10-14)
DEPTH_BATCH_SIZE = 4     # Batch size per UniDepth (Aumentare se VRAM > 8GB)

# Anatomical Anchor Parameters (Anchor points)
NUM_THIGH_SAMPLES = 200  # Sampled points on the thigh
NUM_PELVIS_POINTS = 300  # Anchor points on the pelvis (Anti-Swap)
NUM_TORSO_POINTS = 500   # Anchor points on the torso (Context)
# --------------------------------------------------

# Bypass OMP duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import argparse
import json

# Configurazione Percorsi
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()
DELTA_ROOT = (PROJECT_ROOT / "models/DELTA").resolve()

# Aggiunta percorsi DELTA e sottomoduli
sys.path.append(str(DELTA_ROOT))
sys.path.append(str(DELTA_ROOT / "submodules" / "UniDepth"))
sys.path.append(str(PROJECT_ROOT))

from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.predictor.predictor import Predictor3D
from unidepth.models import UniDepthV2

from src.data_io.dataset_scanner import DatasetScanner
from src.data_io.video_reader import video_provider
from src.data_io.state_manager import videos_state_manager

@torch.inference_mode()
def predict_unidepth(video_np, model, device, batch_size=DEPTH_BATCH_SIZE):
    video_torch = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(device).float()
    depth_pred = []
    for i in range(0, video_torch.shape[0], batch_size):
        chunk = video_torch[i : i + batch_size]
        predictions = model.infer(chunk)
        depth_pred_ = predictions["depth"].squeeze(1).cpu().numpy()
        depth_pred.append(depth_pred_)
    return np.concatenate(depth_pred, axis=0)

def get_anatomical_armor(keypoints, W, H):
    x_min, x_max = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
    y_min, y_max = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])
    dy, dx = y_max - y_min, x_max - x_min
    
    # Pelvi
    p_y_min, p_y_max = max(0, y_min - dy * 2), y_min
    p_x_min, p_x_max = max(0, x_min - dx * 3), min(W, x_max + dx * 3)
    
    # Torso
    t_y_min, t_y_max = max(0, y_min - dy * 15), p_y_min
    t_x_min, t_x_max = max(0, x_min - dx * 5), min(W, x_max + dx * 5)

    def grid(x1, x2, y1, y2, n):
        xs, ys = np.linspace(x1, x2, int(np.sqrt(n))), np.linspace(y1, y2, int(np.sqrt(n)))
        xv, yv = np.meshgrid(xs, ys); return np.stack([xv.flatten(), yv.flatten()], axis=1)

    return np.concatenate([grid(p_x_min, p_x_max, p_y_min, p_y_max, NUM_PELVIS_POINTS), 
                           grid(t_x_min, t_x_max, t_y_min, t_y_max, NUM_TORSO_POINTS)], axis=0)

def main():
    parser = argparse.ArgumentParser(description="High-Stability Single-Pass DELTA Inference")
    parser.add_argument("--video", type=str, help="Video name")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--ckpt", type=str, default=str(DELTA_ROOT / "checkpoints/densetrack3d.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"STABILITY MODE - Single Pass DELTA (V1) - Device: {device}")

    scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
    target_video = None
    for video in scanner:
        if args.video and video.name != args.video: continue
        state = videos_state_manager.load(video.name)
        if state.ref_frame_idx is not None: target_video, target_state = video, state; break
    if not target_video: return

    metadata = video_provider.get_metadata(target_video.video_path)
    ref_idx = target_state.ref_frame_idx
    s = max(0, ref_idx - (args.max_frames // 2))
    e = min(metadata.frames_num - 1, s + args.max_frames - 1)
    
    frames = []
    for i in range(s, e + 1):
        f, _ = video_provider.get_frame(target_video.video_path, i); frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    video_np = np.stack(frames); H_orig, W_orig = video_np.shape[1:3]
    
    delta_model = DenseTrack3D(stride=4, window_len=16, add_space_attn=True, num_virtual_tracks=64,
                               model_resolution=(384, 512), upsample_factor=4)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    delta_model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict, strict=False)
    
    predictor = Predictor3D(model=delta_model).eval().to(device)
    predictor.n_iters = N_ITERS
    predictor.support_grid_size = SUPPORT_GRID_SIZE

    unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().to(device)

    print(f"Predicting Depth for {len(video_np)} frames...")
    videodepth_np = predict_unidepth(video_np, unidepth_model, device)
    
    kps_init = np.array(target_state.specific_keypoints)
    def sample_triangle(p1, p2, p3, n=NUM_THIGH_SAMPLES):
        np.random.seed(42); u, v = np.random.rand(n), np.random.rand(n)
        mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3

    full_queries = np.concatenate([kps_init, sample_triangle(kps_init[0], kps_init[1], kps_init[2]), 
                                   get_anatomical_armor(kps_init, W_orig, H_orig)], axis=0)
    
    queries_tensor = np.zeros((full_queries.shape[0], 3))
    queries_tensor[:, 0] = ref_idx - s
    queries_tensor[:, 1:] = full_queries
    queries_tensor = torch.from_numpy(queries_tensor).float().to(device)[None]

    print(f"Running Bidirectional Tracking on {len(full_queries)} points...")
    video_t = torch.from_numpy(video_np).permute(0, 3, 1, 2).cuda()[None].float()
    vdepth_t = torch.from_numpy(videodepth_np).unsqueeze(1).cuda()[None].float()
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = predictor(video_t, vdepth_t, queries=queries_tensor, grid_query_frame=ref_idx - s, backward_tracking=True)

    output_path = SCRIPT_DIR / "test_res" / f"delta_results_{target_video.name}.npz"
    np.savez(output_path, track2d=out["trajs_uv"][0].cpu().numpy(), track3d_world=out["trajs_3d_dict"]["coords"][0].cpu().numpy(),
             visibility=out["vis"][0].cpu().numpy(), confidence=out["conf"][0].cpu().numpy(), frames=video_np, scales=np.array([1.0, 1.0]),
             video_name=target_video.name, ref_frame_idx=ref_idx, start_idx=s, fps=30.0, mode="single_pass_armor_v1_configurable")
    print(f"DONE. Saved to {output_path}")

if __name__ == "__main__":
    main()
