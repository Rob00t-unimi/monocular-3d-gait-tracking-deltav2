import sys
import os

# --- PARAMETRI DI CONFIGURAZIONE (Modificabili) ---
CHUNK_SIZE = 60          
OVERLAP = 25             
N_ITERS = 12             
SUPPORT_GRID_SIZE = 8    
DEPTH_BATCH_SIZE = 4     
DEPTH_WARNING_THRESHOLD = 0.30  
COARSE_TO_FINE = True    

# Parametri di Campionamento (Sincronizzati con preview_v2_sampling.py)
TARGET_NUM = 200
BODY_W_MUL = 20.0    
BODY_H_MUL = 27.0    
BODY_Y_OFFSET = -17.0 
BODY_NUM = 800       
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
DELTA_ROOT = (PROJECT_ROOT / "models/DELTAv2").resolve()

sys.path.append(str(DELTA_ROOT))
sys.path.append(str(DELTA_ROOT / "submodules" / "UniDepth"))
sys.path.append(str(PROJECT_ROOT))

from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
from densetrack3d.models.predictor.predictor import Predictor3D
from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2

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

def kabsch_umeyama(P, Q):
    P_centroid = np.mean(P, axis=0); Q_centroid = np.mean(Q, axis=0)
    P_centered = P - P_centroid; Q_centered = Q - Q_centroid
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[2, :] *= -1; R = Vt.T @ U.T
    T = Q_centroid - R @ P_centroid; return R, T

def run_delta_on_chunk(video_np, videodepth_np, queries_uv, query_frame_idx, predictor, device, use_fp16=True):
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(device)[None].float()
    videodepth_tensor = torch.from_numpy(videodepth_np).unsqueeze(1).to(device)[None].float()
    full_queries = np.zeros((queries_uv.shape[0], 3))
    full_queries[:, 0] = query_frame_idx; full_queries[:, 1:] = queries_uv
    queries_tensor = torch.from_numpy(full_queries).float().to(device)[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_fp16):
        out_dict = predictor(video_tensor, videodepth_tensor, queries=queries_tensor, 
                             grid_query_frame=query_frame_idx, backward_tracking=True)
    return {"track2d": out_dict["trajs_uv"][0].cpu().numpy(), "track3d": out_dict["trajs_3d_dict"]["coords"][0].cpu().numpy(), "conf": out_dict["conf"][0].cpu().numpy()}

def get_grid(x1, x2, y1, y2, n, W_img, H_img):
    x1, x2 = max(0, x1), min(W_img, x2)
    y1, y2 = max(0, y1), min(H_img, y2)
    if x1 >= x2 or y1 >= y2: return np.zeros((0, 2))
    n_side = int(np.sqrt(n))
    if n_side < 1: return np.zeros((0, 2))
    xs, ys = np.linspace(x1, x2, n_side), np.linspace(y1, y2, n_side)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv.flatten(), yv.flatten()], axis=1)

def get_body_context_box(keypoints, W, H):
    x_min, x_max = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
    y_min, y_max = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])
    dx, dy = x_max - x_min, y_max - y_min
    mid_x = (x_min + x_max) / 2
    b_y1 = y_min + (BODY_Y_OFFSET * dy)
    b_y2 = b_y1 + (BODY_H_MUL * dy)
    b_w = BODY_W_MUL * dx
    return get_grid(mid_x - b_w/2, mid_x + b_w/2, b_y1, b_y2, BODY_NUM, W, H)

def main():
    parser = argparse.ArgumentParser(description="Configurable Chunked DELTAv2 Tracking")
    parser.add_argument("--video", type=str, help="Video name")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--ckpt", type=str, default=str(DELTA_ROOT / "checkpoints/densetrack3dv2.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
    target_video = None
    for v in scanner:
        if args.video and v.name != args.video: continue
        state = videos_state_manager.load(v.name)
        if state.ref_frame_idx is not None: target_video, target_state = v, state; break
    if not target_video: return

    metadata = video_provider.get_metadata(target_video.video_path); total_frames = metadata.frames_num; ref_idx = target_state.ref_frame_idx

    delta_model = DenseTrack3DV2(stride=4, window_len=16, add_space_attn=True, num_virtual_tracks=64, 
                                 model_resolution=(384, 512), upsample_factor=4, coarse_to_fine_dense=COARSE_TO_FINE)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    delta_model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict, strict=False)
    predictor = Predictor3D(model=delta_model).eval().to(device)
    predictor.n_iters = N_ITERS; predictor.support_grid_size = SUPPORT_GRID_SIZE
    unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().to(device)

    # 1. PIVOT
    p_start, p_end = max(0, ref_idx - args.chunk_size // 2), min(total_frames, ref_idx + args.chunk_size // 2)
    frames_p = []
    for i in range(p_start, p_end):
        f, _ = video_provider.get_frame(target_video.video_path, i); frames_p.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    video_p = np.stack(frames_p); kps_init = np.array(target_state.specific_keypoints)
    def sample_triangle(p1, p2, p3, n=TARGET_NUM):
        np.random.seed(42); u, v = np.random.rand(n), np.random.rand(n); mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3
    
    full_query_uv = np.concatenate([kps_init, sample_triangle(kps_init[0], kps_init[1], kps_init[2]), 
                                   get_body_context_box(kps_init, video_p.shape[2], video_p.shape[1])], axis=0)
    
    depth_p = predict_unidepth(video_p, unidepth_model, device)
    res_pivot = run_delta_on_chunk(video_p, depth_p, full_query_uv, ref_idx - p_start, predictor, device)
    global_2d, global_3d, global_conf = {}, {}, {}
    for i in range(p_start, p_end):
        global_2d[i], global_3d[i], global_conf[i] = res_pivot["track2d"][i-p_start], res_pivot["track3d"][i-p_start], res_pivot["conf"][i-p_start]

    # 2. PROPAGATION
    def process_chunks(direction="forward"):
        curr_ptr = p_end if direction == "forward" else p_start
        while (curr_ptr < total_frames if direction == "forward" else curr_ptr > 0):
            overlap_rng = range(max(0, curr_ptr - args.overlap), curr_ptr) if direction == "forward" else range(curr_ptr, min(total_frames, curr_ptr + args.overlap))
            if len(overlap_rng) == 0: break
            confs = np.array([global_conf[i] for i in overlap_rng])
            abs_safe_idx = list(overlap_rng)[np.argmax(np.mean(confs[:, :TARGET_NUM+3], axis=1))]
            
            # Monitor Depth Jump
            current_z = np.mean(global_3d[abs_safe_idx][:TARGET_NUM+3, 2])
            prev_idx = abs_safe_idx - 1 if direction == "forward" else abs_safe_idx + 1
            if prev_idx in global_3d:
                z_jump = np.abs(current_z - np.mean(global_3d[prev_idx][:TARGET_NUM+3, 2]))
                if z_jump > DEPTH_WARNING_THRESHOLD: print(f"--- Warning: Depth Jump at {abs_safe_idx} ({z_jump*100:.1f}cm) ---")

            c_start, c_end = (abs_safe_idx, min(total_frames, abs_safe_idx + args.chunk_size)) if direction == "forward" else (max(0, abs_safe_idx - args.chunk_size + 1), abs_safe_idx + 1)
            q_idx = 0 if direction == "forward" else (c_end - c_start - 1)
            print(f"--- {direction.upper()}: [{c_start}:{c_end}] (Handover at {abs_safe_idx}) ---")
            
            frames_c = []
            for i in range(c_start, c_end):
                f, _ = video_provider.get_frame(target_video.video_path, i); frames_c.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            depth_c = predict_unidepth(np.stack(frames_c), unidepth_model, device)
            res_c = run_delta_on_chunk(np.stack(frames_c), depth_c, global_2d[abs_safe_idx], q_idx, predictor, device)
            for i in range(c_start, c_end): global_2d[i], global_3d[i], global_conf[i] = res_c["track2d"][i-c_start], res_c["track3d"][i-c_start], res_c["conf"][i-c_start]
            curr_ptr = c_end if direction == "forward" else c_start

    process_chunks("forward"); process_chunks("backward")

    all_idx = sorted(global_2d.keys()); out_path = SCRIPT_DIR / "test_res" / f"delta_v2_results_{target_video.name}.npz"
    all_f = []
    for i in all_idx:
        f, _ = video_provider.get_frame(target_video.video_path, i); all_f.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    np.savez(out_path, track2d=np.array([global_2d[i] for i in all_idx]), track3d_world=np.array([global_3d[i] for i in all_idx]), visibility=(np.array([global_conf[i] for i in all_idx]) > 0.5),
             frames=np.stack(all_f), scales=np.array([1.0, 1.0]), video_name=target_video.name, ref_frame_idx=ref_idx, start_idx=0, fps=30.0, mode="chunked_v2_body_box")
    print(f"\nDONE. Saved to {out_path}")

if __name__ == "__main__":
    main()
