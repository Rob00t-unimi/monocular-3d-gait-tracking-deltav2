import sys
import os

# Bypass OMP duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import cv2
from pathlib import Path

# Configurazione Percorsi
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()
SPA_ROOT = (PROJECT_ROOT / "models/SpaTrackerV2").resolve()
sys.path.append(str(SPA_ROOT))
sys.path.append(str(SPA_ROOT / "models"))
sys.path.append(str(PROJECT_ROOT))

from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

from src.data_io.dataset_scanner import DatasetScanner
from src.data_io.video_reader import video_provider
from src.data_io.state_manager import videos_state_manager

import argparse

def main():
    parser = argparse.ArgumentParser(description="Multi-Scenario SpaTrackerV2 Inference")
    parser.add_argument("--video", type=str, help="Specific video name to process (e.g. P1_E1T1_45_ALT.mp4)")
    parser.add_argument("--max_frames", type=int, default=89, help="Max total frames to process")
    args = parser.parse_args()

    print(f"Initializing SpaTrackerV2 - Max Frames: {args.max_frames}")

    # 1. Caricamento Dataset
    scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
    target_video = None
    for video in scanner:
        if args.video and video.name != args.video:
            continue
        state = videos_state_manager.load(video.name)
        if state.ref_frame_idx is not None and len(state.specific_keypoints) == 3:
            target_video, target_state = video, state
            break

    if not target_video:
        print(f"No valid video state found {'for ' + args.video if args.video else ''}."); return

    # 2. Determinazione Finestra Temporale con Auto-Pivot
    metadata = video_provider.get_metadata(target_video.video_path)
    total_frames_in_video = metadata.frames_num
    ref_idx = target_state.ref_frame_idx

    MIN_WINDOW = 12 # Soglia minima per giustificare il tracking bidirezionale
    half_window = (args.max_frames - 1) // 2

    # Initial attempt: center the reference frame
    s = max(0, ref_idx - half_window)
    e = min(total_frames_in_video - 1, s + args.max_frames - 1)

    # Bilanciamento se siamo alla fine del video
    if (e - s + 1) < args.max_frames:
        s = max(0, e - args.max_frames + 1)

    # Pivot logic: if one window side is too short, collapse it into a single pass
    # to run a single longer and more stable inference window.
    if (ref_idx - s) < MIN_WINDOW and ref_idx < (total_frames_in_video - 1):
        print(f"Prefix too short ({ref_idx - s} < {MIN_WINDOW}). Pivoting to Pure Forward.")
        s = ref_idx
        e = min(total_frames_in_video - 1, s + args.max_frames - 1)
    elif (e - ref_idx) < MIN_WINDOW and ref_idx > 0:
        print(f"Suffix too short ({e - ref_idx} < {MIN_WINDOW}). Pivoting to Pure Backward.")
        e = ref_idx
        s = max(0, e - args.max_frames + 1)

    start_idx, end_idx = s, e
    print(f"Video: {target_video.name} | Total: {total_frames_in_video} frames")
    print(f"Processing window: [{start_idx} : {end_idx}] (Ref: {ref_idx}, Length: {end_idx - start_idx + 1})")

    full_frames = []
    for i in range(start_idx, end_idx + 1):
        f, _ = video_provider.get_frame(target_video.video_path, i)
        if f is None: break
        full_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    
    idx_ref_in_clip = ref_idx - start_idx
    video_tensor = torch.from_numpy(np.stack(full_frames)).permute(0, 3, 1, 2).float()

    # 3. Preprocessing
    video_tensor_processed = preprocess_image(video_tensor)[None].cuda()
    B, T, C, H_p, W_p = video_tensor_processed.shape
    H_orig, W_orig = full_frames[0].shape[:2]
    
    scale_x, scale_y = W_p / W_orig, H_p / H_orig

    # 4. Preparazione Query
    kps_init = np.array(target_state.specific_keypoints)
    kps_scaled = kps_init.copy()
    kps_scaled[:, 0] *= scale_x
    kps_scaled[:, 1] *= scale_y

    def sample_triangle(p1, p2, p3, n=200):
        np.random.seed(42)
        u, v = np.random.rand(n), np.random.rand(n)
        mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3

    triangle_pts = sample_triangle(kps_scaled[0], kps_scaled[1], kps_scaled[2], n=200)
    
    q_target = np.zeros((3, 3)); q_target[:, 1:] = kps_scaled
    q_tri = np.zeros((200, 3)); q_tri[:, 1:] = triangle_pts
    n_sparse = 300; np.random.seed(42); sparse_pts = np.random.rand(n_sparse, 2) * np.array([W_p, H_p])
    q_sparse = np.zeros((n_sparse, 3)); q_sparse[:, 1:] = sparse_pts
    
    full_queries = np.concatenate([q_target, q_tri, q_sparse], axis=0)
    full_queries[:, 0] = 0 # Query Frame is always 0 relative to the input sub-video

    # 5. Models
    vggt_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").cuda().eval()
    spatrack_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline").cuda().eval()
    spatrack_model.spatrack.track_num = 768

    # 6. UNICO PROCESSING VGGT
    print("Running VGGT (Geometry extraction)...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            geo = vggt_model(video_tensor_processed) 
            points_map = geo["points_map"].squeeze(0)
            poses = geo["poses_pred"].squeeze(0)
            intrs = geo["intrs"].squeeze(0)
            unc = geo["unc_metric"].squeeze(0)
            video_input_tracker = video_tensor_processed.squeeze(0)

    # 7. TRACKING FUNCTION
    def run_tracking(frames, p_map, ps, ins, un, queries, direction_name):
        print(f"Running tracking: {direction_name} ({len(frames)} frames)...")
        if len(frames) <= 1:
            # Return dummy tensor if only the reference frame is present
            T_f = len(frames)
            N_q = queries.shape[0]
            return {
                "track2d": np.repeat(queries[None, :, 1:], T_f, axis=0),
                "track3d": np.zeros((T_f, N_q, 3)), # Non-critical, merged downstream
                "vis": np.ones((T_f, N_q)),
                "conf": np.ones((T_f, N_q))
            }

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                (
                    c2w_traj, intrinsics, _, _,
                    track3d_cam, track2d_pred, vis_pred, conf_pred, _
                ) = spatrack_model.forward(
                    frames,
                    depth=p_map[..., 2].cpu().numpy(),
                    intrs=ins.cpu().numpy(),
                    extrs=ps.cpu().numpy(),
                    queries=queries,
                    fps=1, iters_track=12,
                    query_no_BA=False, fixed_cam=True, stage=1,
                    unc_metric=(un > 0.5).cpu().numpy(), 
                    support_frame=0,
                    replace_ratio=0.2
                )
                
                xyz_cam = track3d_cam[..., :3]
                device = xyz_cam.device
                R_c2w = c2w_traj[:, :3, :3].to(device)
                T_c2w = c2w_traj[:, :3, 3].to(device)
                track3d_world = torch.einsum("tij,tnj->tni", R_c2w, xyz_cam) + T_c2w[:, None, :]
                
                return {
                    "track2d": track2d_pred.cpu().numpy(),
                    "track3d": track3d_world.cpu().numpy(),
                    "vis": vis_pred.cpu().numpy(),
                    "conf": conf_pred.cpu().numpy()
                }

    # 8. ESECUZIONE SCENARI
    has_fwd = idx_ref_in_clip < (len(full_frames) - 1)
    has_bwd = idx_ref_in_clip > 0

    res_fwd = None
    if has_fwd:
        res_fwd = run_tracking(
            video_input_tracker[idx_ref_in_clip:], points_map[idx_ref_in_clip:], 
            poses[idx_ref_in_clip:], intrs[idx_ref_in_clip:], 
            unc[idx_ref_in_clip:], full_queries, "FORWARD"
        )

    res_bwd_sorted = None
    if has_bwd:
        res_bwd = run_tracking(
            torch.flip(video_input_tracker[:idx_ref_in_clip+1], dims=[0]),
            torch.flip(points_map[:idx_ref_in_clip+1], dims=[0]),
            torch.flip(poses[:idx_ref_in_clip+1], dims=[0]),
            torch.flip(intrs[:idx_ref_in_clip+1], dims=[0]),
            torch.flip(unc[:idx_ref_in_clip+1], dims=[0]),
            full_queries, "BACKWARD"
        )
        # Raddrizza il risultato backward
        res_bwd_sorted = {k: np.flip(v, axis=0) for k, v in res_bwd.items()}

    # 9. UNIFICAZIONE
    if has_fwd and has_bwd:
        # Scenario Bidirezionale
        final_track3d = np.concatenate([res_bwd_sorted["track3d"][:-1], res_fwd["track3d"]], axis=0)
        final_track2d = np.concatenate([res_bwd_sorted["track2d"][:-1], res_fwd["track2d"]], axis=0)
        final_vis = np.concatenate([res_bwd_sorted["vis"][:-1], res_fwd["vis"]], axis=0)
        final_conf = np.concatenate([res_bwd_sorted["conf"][:-1], res_fwd["conf"]], axis=0)
    elif has_fwd:
        # Scenario Solo Forward (Ref all'inizio)
        final_track3d = res_fwd["track3d"]
        final_track2d = res_fwd["track2d"]
        final_vis = res_fwd["vis"]
        final_conf = res_fwd["conf"]
    elif has_bwd:
        # Scenario Solo Backward (Ref alla fine)
        final_track3d = res_bwd_sorted["track3d"]
        final_track2d = res_bwd_sorted["track2d"]
        final_vis = res_bwd_sorted["vis"]
        final_conf = res_bwd_sorted["conf"]
    else:
        print("Error: No frames to process."); return

    # 10. SALVATAGGIO
    output_path = SCRIPT_DIR / "test_res" / f"tracking_results_unified_{target_video.name}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path,
             track2d=final_track2d,
             track3d_world=final_track3d,
             visibility=final_vis,
             confidence=final_conf,
             frames=np.stack(full_frames),
             scales=np.array([scale_x, scale_y]),
             video_name=target_video.name,
             ref_frame_idx=ref_idx,
             start_idx=start_idx)

    print(f"DONE. Results saved to {output_path}")

if __name__ == "__main__":
    main()
