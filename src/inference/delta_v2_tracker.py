# -*- coding: utf-8 -*-
"""
DELTAv2 3D Point Tracker Implementation.

Implements Base3DTracker using:
- UniDepthV2 for metric monocular depth estimation.
- DenseTrack3DV2 (Coarse-to-Fine) for robust 3D point tracking.
- MediaPipe Pose Landmarker for body silhouette context sampling (200 thigh + 800 context).
- Bidirectional tracking starting from the reference frame.

Strictly preserves the exact hyperparameter and model configurations from experiments/delta_test/delta_v2_test.py.
"""

import os
import sys
import contextlib
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Suppress noise before third-party imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DELTA_ROOT = (PROJECT_ROOT / "models" / "DELTAv2").resolve()
UNIDEPTH_ROOT = (DELTA_ROOT / "submodules" / "UniDepth").resolve()

# Add submodule paths
for p in [str(DELTA_ROOT), str(UNIDEPTH_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

@contextlib.contextmanager
def silence_all(enabled=True):
    """Context manager to silence stdout/stderr during verbose model inference."""
    if not enabled:
        yield
        return
    devnull_file = open(os.devnull, "w")
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_stdout_fd, old_stderr_fd = os.dup(1), os.dup(2)
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

with silence_all():
    import torch
    import torch.nn.functional as F
    from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
    from densetrack3d.models.predictor.predictor import Predictor3D
    from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
    from src.data_io.video_reader import video_provider
    from src.inference.base_tracker import Base3DTracker, TrackingResult

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        HAS_MEDIAPIPE = True
    except Exception:
        HAS_MEDIAPIPE = False


class DeltaV2Tracker(Base3DTracker):
    """
    Production DELTAv2 3D tracker encapsulating UniDepthV2, DenseTrack3DV2, and MediaPipe silhouette masking.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        depth_batch_size: int = 8,
        n_iters: int = 6,
        support_grid_size: int = 12,
        target_thigh_points: int = 200,
        body_context_points: int = 800,
        coarse_to_fine: bool = True,
        verbose: bool = False
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (DELTA_ROOT / "checkpoints" / "densetrack3dv2.pth")
        self.depth_batch_size = depth_batch_size
        self.n_iters = n_iters
        self.support_grid_size = support_grid_size
        self.target_thigh_points = target_thigh_points
        self.body_context_points = body_context_points
        self.coarse_to_fine = coarse_to_fine
        self.verbose = verbose

        self.predictor = None
        self.unidepth_model = None
        self.mediapipe_model_path = DELTA_ROOT / "pose_landmarker_full.task"

    def _ensure_models_loaded(self):
        """Lazy-loads neural network models onto the target device."""
        if self.predictor is not None and self.unidepth_model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DELTAv2 checkpoint not found at: {self.checkpoint_path}")

        logger.info(f"Loading DELTAv2 model on {self.device}...")
        with silence_all(not self.verbose):
            delta_model = DenseTrack3DV2(
                stride=4,
                window_len=16,
                add_space_attn=True,
                num_virtual_tracks=64,
                model_resolution=(384, 512),
                upsample_factor=4,
                coarse_to_fine_dense=self.coarse_to_fine
            )
            ckpt_dict = torch.load(self.checkpoint_path, map_location="cpu")
            weights = ckpt_dict["model"] if "model" in ckpt_dict else ckpt_dict
            delta_model.load_state_dict(weights, strict=False)

            self.predictor = Predictor3D(model=delta_model).eval().to(self.device)
            self.predictor.n_iters = self.n_iters
            self.predictor.support_grid_size = self.support_grid_size

            logger.info("Loading UniDepthV2 model (lpiccinelli/unidepth-v2-vitl14)...")
            self.unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().to(self.device)

    def _ensure_mediapipe_asset(self) -> Path:
        """Ensures the MediaPipe pose landmarker task file is downloaded and cached."""
        if self.mediapipe_model_path.exists():
            return self.mediapipe_model_path
        
        alt_path = PROJECT_ROOT / "experiments" / "delta_test" / "pose_landmarker_full.task"
        if alt_path.exists():
            return alt_path

        self.mediapipe_model_path.parent.mkdir(parents=True, exist_ok=True)
        url = MEDIAPIPE_POSE_MODEL_URL
        logger.info(f"Downloading MediaPipe task asset to {self.mediapipe_model_path}...")
        urllib.request.urlretrieve(url, self.mediapipe_model_path)
        return self.mediapipe_model_path

    @torch.inference_mode()
    def _predict_depth(self, video_np: np.ndarray) -> np.ndarray:
        """Estimates metric depth maps frame-by-frame in chunks."""
        video_torch = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(self.device).float()
        depth_pred = []
        for i in range(0, video_torch.shape[0], self.depth_batch_size):
            chunk = video_torch[i : i + self.depth_batch_size]
            with silence_all(not self.verbose):
                predictions = self.unidepth_model.infer(chunk)
            depth_pred.append(predictions["depth"].squeeze(1).cpu().numpy())
        return np.concatenate(depth_pred, axis=0)

    @staticmethod
    def sample_triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 200) -> np.ndarray:
        """Samples n points uniformly within the triangle formed by (p1, p2, p3)."""
        np.random.seed(42)
        u, v = np.random.rand(n), np.random.rand(n)
        mask = u + v > 1
        u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        return u[:, None] * p1 + v[:, None] * p2 + (1 - u - v)[:, None] * p3

    def _sample_body_context_points(self, ref_frame_rgb: np.ndarray, kps_init: np.ndarray) -> np.ndarray:
        """Samples context points from the body silhouette via MediaPipe (or bounding box fallback)."""
        body_pts = np.zeros((0, 2))
        if HAS_MEDIAPIPE:
            try:
                task_path = self._ensure_mediapipe_asset()
                base_options = python.BaseOptions(model_asset_path=str(task_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    output_segmentation_masks=True
                )
                with vision.PoseLandmarker.create_from_options(options) as landmarker:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_frame_rgb)
                    res = landmarker.detect(mp_img)
                    if res.segmentation_masks:
                        mask = res.segmentation_masks[0].numpy_view() > 0.5
                        y, x = np.where(mask.squeeze())
                        if len(x) > 0:
                            all_pix = np.stack([x, y], axis=1)
                            body_pts = all_pix[np.random.choice(len(all_pix), min(len(all_pix), self.body_context_points), replace=False)]
            except Exception as e:
                logger.warning(f"MediaPipe segmentation fallback: {e}")

        # Fallback to bounding box grid if MediaPipe produced no points
        if len(body_pts) == 0:
            x_min, x_max = np.min(kps_init[:, 0]), np.max(kps_init[:, 0])
            y_min, y_max = np.min(kps_init[:, 1]), np.max(kps_init[:, 1])
            dx, dy = x_max - x_min, y_max - y_min
            mid_x = (x_min + x_max) / 2.0
            b_y1, b_y2 = y_min + (-17.0 * dy), y_min + (10.0 * dy)
            b_w = 20.0 * dx
            n_side = int(np.sqrt(self.body_context_points))
            xs = np.linspace(max(0, mid_x - b_w / 2.0), min(ref_frame_rgb.shape[1], mid_x + b_w / 2.0), n_side)
            ys = np.linspace(max(0, b_y1), min(ref_frame_rgb.shape[0], b_y2), n_side)
            xv, yv = np.meshgrid(xs, ys)
            body_pts = np.stack([xv.flatten(), yv.flatten()], axis=1)

        return body_pts

    def track_video(
        self,
        video_path: Path,
        ref_frame_idx: int,
        specific_keypoints: np.ndarray,
        max_frames: int = 250
    ) -> TrackingResult:
        """
        Executes raw bidirectional 3D point tracking on the specified video sequence.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._ensure_models_loaded()

        # 1. Determine frame window
        metadata = video_provider.get_metadata(video_path)
        total_frames = metadata.frames_num

        if total_frames <= max_frames:
            s, e = 0, total_frames - 1
        else:
            s = max(0, ref_frame_idx - (max_frames // 2))
            e = min(total_frames - 1, s + max_frames - 1)
            if e == total_frames - 1:
                s = max(0, e - max_frames + 1)

        # 2. Load frames in RGB format (fast sequential capture with fallback)
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            for i in range(s, e + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        if len(frames) != (e - s + 1):
            frames = []
            for i in range(s, e + 1):
                f, _ = video_provider.get_frame(video_path, i)
                if f is not None:
                    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                else:
                    raise RuntimeError(f"Could not read frame {i} from {video_path}")
        video_np = np.stack(frames)

        # 3. Predict depth
        logger.info(f"Predicting depth for {len(video_np)} frames (window: {s}-{e})...")
        videodepth_np = self._predict_depth(video_np)

        # 4. Sample query points (3 keypoints + 200 triangle + 800 body context = 1003 points)
        kps_init = np.array(specific_keypoints, dtype=np.float32)
        thigh_samples = self.sample_triangle(kps_init[0], kps_init[1], kps_init[2], n=self.target_thigh_points)
        ref_rel = ref_frame_idx - s
        body_samples = self._sample_body_context_points(video_np[ref_rel], kps_init)

        full_q = np.concatenate([kps_init, thigh_samples, body_samples], axis=0)
        t_indices = np.full((len(full_q), 1), ref_rel)
        queries = torch.from_numpy(np.concatenate([t_indices, full_q], axis=1)).float().to(self.device)[None]

        # 5. Execute 3D tracking
        logger.info(f"Tracking {len(full_q)} points with DELTAv2 (bidirectional)...")
        v_t = torch.from_numpy(video_np).permute(0, 3, 1, 2).to(self.device)[None].float()
        d_t = torch.from_numpy(videodepth_np).unsqueeze(1).to(self.device)[None].float()

        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
            with silence_all(not self.verbose):
                out = self.predictor(
                    v_t,
                    d_t,
                    queries=queries,
                    grid_query_frame=ref_rel,
                    backward_tracking=True
                )

        # 6. Extract numpy outputs
        t2d = out["trajs_uv"][0].cpu().numpy()
        t3d = out["trajs_3d_dict"]["coords"][0].cpu().numpy()
        vis = out["vis"][0].cpu().numpy()
        conf = out["conf"][0].cpu().numpy()

        return TrackingResult(
            track2d=t2d,
            track3d_world=t3d,
            visibility=vis,
            confidence=conf,
            video_name=video_path.name,
            ref_frame_idx=ref_frame_idx,
            start_idx=s,
            fps=30.0,
            mode="v2_single_pass_body_box",
            scales=np.array([1.0, 1.0]),
            metadata={
                "total_frames": total_frames,
                "window_start": s,
                "window_end": e,
                "num_queries": len(full_q),
                "num_thigh_queries": 3 + self.target_thigh_points,
                "num_body_context": len(body_samples)
            }
        )
