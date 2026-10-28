import sys
import os

# --- PARAMETRI DI CONFIGURAZIONE (Modificabili) ---
CHUNK_SIZE = 60          # Lunghezza di ogni blocco di frame
OVERLAP = 25             # Sovrapposizione tra i blocchi
N_ITERS = 12             # Iterazioni di raffinamento (Standard: 12)
SUPPORT_GRID_SIZE = 8    # Dimensione griglia di supporto (Standard: 8)
DEPTH_BATCH_SIZE = 4     # Batch size per UniDepth
DEPTH_WARNING_THRESHOLD = 0.30  # Soglia per segnalare swap (in metri)
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

def get_person_body_grid(keypoints, W, H, num_points=600):
    x_min, x_max = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
    y_min, y_max = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])
    dy, dx = y_max - y_min, x_max - x_min
    pad_y_up, pad_y_down, pad_x = dy * 17, dy * 9.0, max(dx * 9.0, dy * 1.8) 
    p_y_min, p_y_max = max(0, y_min - pad_y_up), min(H, y_max + pad_y_down)
    p_x_min, p_x_max = max(0, x_min - pad_x), min(W, x_max + pad_x)
    aspect = (p_y_max - p_y_min) / (p_x_max - p_x_min + 1e-6)
    gh = int(np.sqrt(num_points * aspect)); gw = num_points // gh
    xs, ys = np.linspace(p_x_min, p_x_max, gw), np.linspace(p_y_min, p_y_max, gh)
    xv, yv = np.meshgrid(xs, ys); return np.stack([xv.flatten(), yv.flatten()], axis=1)

def main():
    parser = argparse.ArgumentParser(description="Configurable Chunked DELTA Tracking")
    parser.add_argument("--video", type=str, help="Video name")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
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

    delta_model = DenseTrack3D(stride=4, window_len=16, add_space_attn=True, num_virtual_tracks=64, model_resolution=(384, 512), upsample_factor=4)
    state_dict = torch.load(str(DELTA_ROOT / "checkpoints/densetrack3d.pth"), map_location="cpu")
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
    def sample_triangle(p1, p2, p3, n=200):
        np.random.seed(42); u, v = np.random.rand(n), np.random.rand(n); mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3
    full_query_uv = np.concatenate([kps_init, sample_triangle(kps_init[0], kps_init[1], kps_init[2], 200), get_person_body_grid(kps_init, video_p.shape[2], video_p.shape[1], 600)], axis=0)
    
    depth_p = predict_unidepth(video_p, unidepth_model, device)
    res_pivot = run_delta_on_chunk(video_p, depth_p, full_query_uv, ref_idx - p_start, predictor, device)
    global_2d, global_3d, global_conf = {}, {}, {}
    for i in range(p_start, p_end):
        global_2d[i], global_3d[i], global_conf[i] = res_pivot["track2d"][i-p_start], res_pivot["track3d"][i-p_start], res_pivot["conf"][i-p_start]

    # 2. PROPAGATION
    def process_chunks(direction="forward"):
        curr_ptr = p_end if direction == "forward" else p_start
        while (curr_ptr < total_frames if direction == "forward" else curr_ptr > 0):
            overlap_rng = range(curr_ptr - args.overlap, curr_ptr) if direction == "forward" else range(curr_ptr, curr_ptr + args.overlap)
            confs = np.array([global_conf[i] for i in overlap_rng])
            abs_safe_idx = list(overlap_rng)[np.argmax(np.mean(confs[:, :203], axis=1))]
            
            # Monitor Depth Jump
            current_z = np.mean(global_3d[abs_safe_idx][:203, 2])
            prev_idx = abs_safe_idx - 1 if direction == "forward" else abs_safe_idx + 1
            if prev_idx in global_3d:
                z_jump = np.abs(current_z - np.mean(global_3d[prev_idx][:203, 2]))
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

    all_idx = sorted(global_2d.keys()); out_path = SCRIPT_DIR / "test_res" / f"delta_results_{target_video.name}.npz"
    all_f = []
    for i in all_idx:
        f, _ = video_provider.get_frame(target_video.video_path, i); all_f.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    np.savez(out_path, track2d=np.array([global_2d[i] for i in all_idx]), track3d_world=np.array([global_3d[i] for i in all_idx]), visibility=(np.array([global_conf[i] for i in all_idx]) > 0.5),
             frames=np.stack(all_f), scales=np.array([1.0, 1.0]), video_name=target_video.name, ref_frame_idx=ref_idx, start_idx=0, fps=30.0, mode="chunked_v1_configurable")
    print(f"\nDONE. Saved to {out_path}")

if __name__ == "__main__":
    main()
