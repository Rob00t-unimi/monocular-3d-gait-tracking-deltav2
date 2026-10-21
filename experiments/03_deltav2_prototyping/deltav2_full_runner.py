import os
import sys

# --- ABSOLUTE NOISE SUPPRESSION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["GLOG_minloglevel"] = "3" 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
import logging
import contextlib

# 1. Silenzia subito tutto il logging globale in modo radicale
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger().setLevel(logging.CRITICAL)

for logger_name in ["huggingface_hub", "transformers", "urllib3", "timm"]:
    l = logging.getLogger(logger_name)
    l.setLevel(logging.ERROR)
    l.propagate = False

# 2. Crea un logger "blindato" per l'applicazione
app_log = logging.getLogger("app")
app_log.setLevel(logging.INFO)
app_log.propagate = False
_orig_stdout = sys.stdout 

try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN
    RESET = colorama.Style.RESET_ALL
except:
    GREEN = ""
    RESET = ""

class CustomFormatter(logging.Formatter):
    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        return f"{timestamp} - {record.getMessage()}"

handler = logging.StreamHandler(_orig_stdout)
handler.setFormatter(CustomFormatter(datefmt='%H:%M:%S'))
app_log.addHandler(handler)

# 3. Silenzia i warnings di Python
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def silence_all(enabled=True):
    if not enabled:
        yield
        return
    devnull_file = open(os.devnull, 'w')
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        sys.stdout = devnull_file
        sys.stderr = devnull_file
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull_file.close()

# Import pesanti nel silenzio
with silence_all():
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from pathlib import Path
    import argparse
    import json
    from tqdm import tqdm
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        HAS_MEDIAPIPE = True
    except:
        HAS_MEDIAPIPE = False

# --- PARAMETRI DI CONFIGURAZIONE ---
MAX_FRAMES = 250         
N_ITERS = 6              # Allineato al paper (4-6). Evita drifting da sovra-iterazione.
SUPPORT_GRID_SIZE = 12   # Default standard
DEPTH_BATCH_SIZE = 8
COARSE_TO_FINE = True    
TARGET_NUM = 200
BODY_NUM = 800

# Percorsi
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()
DELTA_ROOT = (PROJECT_ROOT / "models/DELTAv2").resolve()
sys.path.extend([str(DELTA_ROOT), str(DELTA_ROOT / "submodules" / "UniDepth"), str(PROJECT_ROOT)])

with silence_all():
    from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
    from densetrack3d.models.predictor.predictor import Predictor3D
    from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
    from src.data_io.dataset_scanner import DatasetScanner
    from src.data_io.video_reader import video_provider
    from src.data_io.state_manager import videos_state_manager

@torch.inference_mode()
def predict_unidepth(video_np, model, device, batch_size=DEPTH_BATCH_SIZE, verbose=False):
    video_torch = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(device).float()
    depth_pred = []
    pbar = tqdm(total=len(video_np), 
                desc=f"{GREEN}[1/3]{RESET} Estimating depth maps", 
                disable=verbose, leave=True, file=_orig_stdout,
                bar_format='{l_bar}{bar:30}{r_bar}',
                colour='cyan',
                unit="fr")
    for i in range(0, video_torch.shape[0], batch_size):
        chunk = video_torch[i : i + batch_size]
        with silence_all(not verbose):
            predictions = model.infer(chunk)
        depth_pred.append(predictions["depth"].squeeze(1).cpu().numpy())
        pbar.update(len(chunk))
    pbar.close()
    return np.concatenate(depth_pred, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--ckpt", type=str, default=str(DELTA_ROOT / "checkpoints/densetrack3dv2.pth"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        app_log.setLevel(logging.DEBUG)
        warnings.resetwarnings()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_log.info(f"System ready. Device: {device}")

    with silence_all(not args.verbose):
        scanner = DatasetScanner(str(PROJECT_ROOT / "data"))
        target_video = None
        for video in scanner:
            if args.video and video.name != args.video: continue
            state = videos_state_manager.load(video.name)
            if state.ref_frame_idx is not None: target_video, target_state = video, state; break
    
    if not target_video: 
        app_log.error(f"Video {args.video} not found or no reference frame.")
        return

    metadata = video_provider.get_metadata(target_video.video_path)
    ref_idx = target_state.ref_frame_idx
    
    if metadata.frames_num <= args.max_frames:
        s, e = 0, metadata.frames_num - 1
    else:
        s = max(0, ref_idx - (args.max_frames // 2))
        e = min(metadata.frames_num - 1, s + args.max_frames - 1)
        if e == metadata.frames_num - 1:
            s = max(0, e - args.max_frames + 1)
    
    app_log.info(f"Loading video '{target_video.name}' (Loaded: {s}-{e}, Total: {metadata.frames_num}, Ref: {ref_idx})...")
    frames = []
    for i in range(s, e + 1):
        f, _ = video_provider.get_frame(target_video.video_path, i)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    video_np = np.stack(frames)
    
    app_log.info("Loading models (DELTAv2 + UniDepth)...")
    with silence_all(not args.verbose):
        delta_model = DenseTrack3DV2(stride=4, 
                                     window_len=16, # Riportato a 16: deve corrispondere ai pesi del checkpoint
                                     add_space_attn=True, 
                                     num_virtual_tracks=64, 
                                     model_resolution=(384, 512), 
                                     upsample_factor=4, 
                                     coarse_to_fine_dense=COARSE_TO_FINE)
        delta_model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"], strict=False)
        predictor = Predictor3D(model=delta_model).eval().to(device)
        predictor.n_iters = N_ITERS
        predictor.support_grid_size = SUPPORT_GRID_SIZE 
        unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().to(device)

    videodepth_np = predict_unidepth(video_np, unidepth_model, device, verbose=args.verbose)
    
    kps_init = np.array(target_state.specific_keypoints)
    def sample_triangle(p1, p2, p3, n=TARGET_NUM):
        np.random.seed(42); u, v = np.random.rand(n), np.random.rand(n)
        mask = u + v > 1; u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3

    body_pts = np.zeros((0, 2))
    if HAS_MEDIAPIPE:
        try:
            app_log.info(f"{GREEN}[2/3]{RESET} Segmenting frame {ref_idx} with MediaPipe...")
            with silence_all(not args.verbose):
                model_path = SCRIPT_DIR / "pose_landmarker_full.task"
                if not model_path.exists():
                    import urllib.request
                    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
                    urllib.request.urlretrieve(url, model_path)
                
                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, output_segmentation_masks=True)
                with vision.PoseLandmarker.create_from_options(options) as landmarker:
                    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=video_np[ref_idx-s]))
                    if res.segmentation_masks:
                        mask = res.segmentation_masks[0].numpy_view() > 0.5
                        y, x = np.where(mask.squeeze())
                        if len(x) > 0:
                            all_pix = np.stack([x, y], axis=1)
                            body_pts = all_pix[np.random.choice(len(all_pix), min(len(all_pix), BODY_NUM), replace=False)]
            
            if len(body_pts) == 0:
                app_log.info("MediaPipe found no silhouette. Using Bounding Box fallback.")
        except Exception:
            app_log.info("MediaPipe failed. Using Bounding Box fallback.")
    else:
        app_log.info("MediaPipe not installed. Using Bounding Box for context points.")
    
    if len(body_pts) == 0:
        x_min, x_max = np.min(kps_init[:, 0]), np.max(kps_init[:, 0])
        y_min, y_max = np.min(kps_init[:, 1]), np.max(kps_init[:, 1])
        dx, dy = x_max - x_min, y_max - y_min
        mid_x = (x_min + x_max) / 2
        b_y1, b_y2 = y_min + (-17.0 * dy), y_min + (10.0 * dy)
        b_w = 20.0 * dx
        n_side = int(np.sqrt(BODY_NUM))
        xs, ys = np.linspace(max(0, mid_x - b_w/2), min(video_np.shape[2], mid_x + b_w/2), n_side), np.linspace(max(0, b_y1), min(video_np.shape[1], b_y2), n_side)
        xv, yv = np.meshgrid(xs, ys)
        body_pts = np.stack([xv.flatten(), yv.flatten()], axis=1)

    full_q = np.concatenate([kps_init, sample_triangle(kps_init[0], kps_init[1], kps_init[2]), body_pts], axis=0)
    queries = torch.from_numpy(np.concatenate([np.full((len(full_q), 1), ref_idx - s), full_q], axis=1)).float().to(device)[None]

    app_log.info(f"{GREEN}[3/3]{RESET} Tracking {len(full_q)} points with DELTAv2...")
    v_t = torch.from_numpy(video_np).permute(0, 3, 1, 2).cuda()[None].float()
    d_t = torch.from_numpy(videodepth_np).unsqueeze(1).cuda()[None].float()
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        with silence_all(not args.verbose):
            out = predictor(v_t, d_t, queries=queries, grid_query_frame=ref_idx - s, backward_tracking=True)

    out_p = SCRIPT_DIR / "test_res" / f"delta_v2_results_{target_video.name}.npz"
    np.savez(out_p, track2d=out["trajs_uv"][0].cpu().numpy(), track3d_world=out["trajs_3d_dict"]["coords"][0].cpu().numpy(),
             visibility=out["vis"][0].cpu().numpy(), confidence=out["conf"][0].cpu().numpy(), frames=video_np, scales=np.array([1.0, 1.0]),
             video_name=target_video.name, ref_frame_idx=ref_idx, start_idx=s, fps=30.0, mode="v2_single_pass_body_box")
    app_log.info(f"Done! Result: {out_p.name}")

if __name__ == "__main__":
    main()
