"""
Interactive Signal and 3D Trajectory Visualizer for Gait360 Benchmark.

High-performance, reactive GUI with persistent artists, focus management,
and robust event handling across all 40 benchmark videos.

Features:
    - Phase-Synchronization (cross-correlation lag alignment against IMU GT)
    - Optimal Triplet Refinement (Geometry Pruning v4.1)
    - 2D Kinematics and 3D Trajectory Trails in State Space

Usage:
    python -m src.cli inspect --mode dashboard [--start INDEX]

Controls:
    ←  /  →  arrow keys / A / D : Previous / Next benchmark video
    Prev / Next buttons          : Same as arrow keys
    Time slider / ↑ / ↓          : Scrub timeline, synchronizing 2D and 3D plots
    G  key / GT button           : Toggle visibility of GT (IMU)
    R  key / Raw button          : Toggle visibility of Raw tracking (Orig 3pts)
    F  key / Ref button          : Toggle visibility of Refined tracking (Optimal Triplet)
    S  key / Sync button         : Toggle Phase Synchronization (aligned vs unaligned GT)
    N  key / Norm button         : Toggle normalization to [-1, 1]

Layout:
    Left (2 cols × 3 rows) : 2D signals (AccX/Pitch, AccY/Roll, AccZ/Yaw)
    Right-Top (3D)          : 3D Acceleration Trail [accX, accY, accZ]
    Right-Bottom (3D)       : 3D Orientation Trail [Roll, Pitch, Yaw]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # Interactive desktop GUI backend

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.signal_extractor import (
    extract_signals,
    extract_raw_signals,
    extract_aligned_signals,
    load_cutoffs,
)
from src.analysis.gt_loader import load_gt
from src.metrics.waveform.cross_correlation import calculate_cross_correlation
from src.metrics.waveform.pearson import calculate_pearson

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("visualize_signals")

# ---------------------------------------------------------------------------
# Constants & Configurations
# ---------------------------------------------------------------------------

MANIFEST_PATH = PROJECT_ROOT / "data" / "selected_inference_videos_40.csv"
RAW_DIR = PROJECT_ROOT / "data" / "inference_results" / "raw"
REFINED_DIR = PROJECT_ROOT / "data" / "inference_results" / "refined"

SIGNAL_LAYOUT = [
    ("accX",  0, 0, "AccX"),
    ("Pitch", 0, 1, "Pitch (°)"),
    ("accY",  1, 0, "AccY"),
    ("Roll",  1, 1, "Roll (°)"),
    ("accZ",  2, 0, "AccZ"),
    ("Yaw",   2, 1, "Yaw (°)"),
]

SPATRACKER_VGGT_HASHES = {
    "P1_E1T1_45_ALT.mp4": "82af1dfc8521d3048f368abdf55721c99e62a5eb045e1f413874b1e443dd1ee1",
    "P2_E1T1_90.mp4":     "938e7241d84ad70538e590c959f7a3adbab4b622fe5756ca5bf0bec6852159f6",
    "P3_E1T2_45.mp4":     "bb2311fb5985d5f00039f429c8287bc787830953aac93af4dbd2a10522c769f6",
    "P7_E1T1_90.mp4":     "f4dfa3a0ee57921699b150ffc5de65118cb85e00b18719cd07bda697684946ed",
    "P8_E3T1_Rear.mp4":   "bc874576978e951ba64aa39f5984fff9a13891f02d78bb722eef536f28c1b887",
    "P11_E3T1_Front.mp4": "3a6ac6f474b067e94c5abe6bb627ac9f468823aa733a2c908ab768b782a0ecb5",
}

DELTAV2_MEDIAPIPE_HASHES = {
    "P1_E1T1_45_ALT.mp4": "d5020d0903d5e250e9ba318a706ff597c5a3f8ab00cd535e4502b2e142044037",
    "P2_E1T1_90.mp4":     "0dfe15c4be899c908c6310ac3da334540baa6e0869eaa49319bf81a8d6ef3562",
    "P3_E1T2_45.mp4":     "4e481fc84af99c83ce8181bb52e307845855a593b04c4f3ab637e1d7d2d1ed45",
    "P7_E1T1_90.mp4":     "c6bcfa0792656edec7afbe3b4744e0f527e2d769e616430026db9dbb252eda87",
    "P8_E3T1_Rear.mp4":   "c6f0541e42953a8d5b83b5d7aec87103dc887b239bd110fdcb723f2cfc6ff984",
    "P11_E3T1_Front.mp4": "8cf33f05633e775a535b321f6b6bbc2a20cfed705a4ce29368a0caede798a774",
}

BENCHMARK_CATEGORIES_6 = {
    "P1_E1T1_45_ALT.mp4": "45° Oblique (Alternative Clothing)",
    "P2_E1T1_90.mp4":     "90° Sagittal View",
    "P3_E1T2_45.mp4":     "45° Oblique View",
    "P7_E1T1_90.mp4":     "90° Sagittal View",
    "P8_E3T1_Rear.mp4":   "Posterior (Rear) View",
    "P11_E3T1_Front.mp4": "Frontal View",
}

COLORS = {
    "gt":      ("#777777", "GT (IMU)",          0.65, "--", 1.3),
    "raw":     ("#2196F3", "Raw (Orig 3pts)",    0.80, "-",  1.4),
    "refined": ("#FF6D00", "Refined (Opt 3pts)", 0.95, "-",  1.7),
}

# Strong module-level reference preventing Python garbage collection
_ACTIVE_DASHBOARDS: list[GaitBenchmarkDashboard] = []


def _norm(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [-1, 1] range."""
    m = np.max(np.abs(arr))
    return arr / m if m > 1e-9 else arr.copy()


# ---------------------------------------------------------------------------
# Interactive Dashboard Class
# ---------------------------------------------------------------------------

class GaitBenchmarkDashboard:
    """Synchronized, responsive 2D and 3D visualizer for the 40 benchmark videos."""

    def __init__(
        self,
        video_names: list[str],
        categories: dict[str, str],
        lowcut: float,
        highcut: float,
        start_idx: int = 0,
        model_mode: str = "deltav2",
    ):
        self.video_names = video_names
        self.categories = categories
        self.lowcut = lowcut
        self.highcut = highcut
        self.idx = max(0, min(start_idx, len(video_names) - 1))
        self.model_mode = model_mode

        # UI state: individual visibility & sync toggles
        self.show_gt = True
        self.show_raw = True
        self.show_refined = True
        self.sync_gt = True
        self.normalize = True
        self.source = "triplet"
        self.current_time = 0.0

        # Memory cache for loaded videos
        self._cache: dict[str, dict] = {}

        self._build_figure()
        self._load_current_video(rescale=True)

    # -----------------------------------------------------------------------
    # Figure Construction & Layout
    # -----------------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(22, 11))
        # Keep strong reference on figure
        self.fig._dashboard_ref = self

        gs = self.fig.add_gridspec(
            6, 3,
            width_ratios=[1.0, 1.0, 1.25],
            hspace=0.65,
            wspace=0.25,
            left=0.05,
            right=0.96,
            top=0.92,
            bottom=0.14,
        )

        # 6 2D subplots with persistent artists
        self.axes_2d = {}
        self.lines_2d = {}
        self.markers_2d = {}

        grid_positions = {
            "accX":  gs[0:2, 0],
            "Pitch": gs[0:2, 1],
            "accY":  gs[2:4, 0],
            "Roll":  gs[2:4, 1],
            "accZ":  gs[4:6, 0],
            "Yaw":   gs[4:6, 1],
        }

        for col_name, row_span in grid_positions.items():
            ax = self.fig.add_subplot(row_span)
            ylabel = col_name if not col_name.endswith(tuple("XYZ")) else f"Acc{col_name[-1]}"
            if col_name in ("Roll", "Pitch", "Yaw"):
                ylabel = f"{col_name} (°)"

            ax.set_title(ylabel, fontsize=9, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.grid(True, alpha=0.18)

            c_gt, l_gt, a_gt, ls_gt, lw_gt = COLORS["gt"]
            c_rw, l_rw, a_rw, ls_rw, lw_rw = COLORS["raw"]
            c_rf, l_rf, a_rf, ls_rf, lw_rf = COLORS["refined"]

            if self.model_mode == "compare":
                l_rw = "DELTAv2"
                l_rf = "SpaTracker"
            elif self.model_mode == "spatracker":
                l_rw = "SpaTracker (Raw)"
                l_rf = "SpaTracker"

            line_gt, = ax.plot([], [], color=c_gt, label=l_gt, alpha=a_gt, linestyle=ls_gt, linewidth=lw_gt)
            line_raw, = ax.plot([], [], color=c_rw, label=l_rw, alpha=a_rw, linestyle=ls_rw, linewidth=lw_rw)
            line_ref, = ax.plot([], [], color=c_rf, label=l_rf, alpha=a_rf, linestyle=ls_rf, linewidth=lw_rf)
            marker = ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1.2)

            self.axes_2d[col_name] = ax
            self.lines_2d[col_name] = {"gt": line_gt, "raw": line_raw, "ref": line_ref}
            self.markers_2d[col_name] = marker
            ax.legend(fontsize=7, loc="upper right")

        # 2 3D subplots
        self.ax3d_acc = self.fig.add_subplot(gs[0:3, 2], projection="3d")
        self.ax3d_ori = self.fig.add_subplot(gs[3:6, 2], projection="3d")

        self._init_3d_axes(self.ax3d_acc, "3D Acceleration Space [accX, accY, accZ]", ["accX", "accY", "accZ"])
        self._init_3d_axes(self.ax3d_ori, "3D Orientation Space [Roll, Pitch, Yaw]", ["Roll", "Pitch", "Yaw"])

        # Keyboard & Mouse listeners (ensuring keyboard focus is always maintained)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", lambda _: self._refocus_canvas())
        self.fig.canvas.mpl_connect("button_release_event", lambda _: self._refocus_canvas())

        # Bottom UI controls
        bg = "#efefef"
        ax_prev = self.fig.add_axes([0.015, 0.03, 0.045, 0.04])
        ax_next = self.fig.add_axes([0.065, 0.03, 0.045, 0.04])
        ax_gt   = self.fig.add_axes([0.115, 0.03, 0.055, 0.04])
        ax_raw  = self.fig.add_axes([0.175, 0.03, 0.055, 0.04])
        ax_ref  = self.fig.add_axes([0.235, 0.03, 0.055, 0.04])
        ax_mode = self.fig.add_axes([0.295, 0.03, 0.075, 0.04])
        ax_sync = self.fig.add_axes([0.375, 0.03, 0.065, 0.04])
        ax_norm = self.fig.add_axes([0.445, 0.03, 0.06, 0.04])
        ax_slider = self.fig.add_axes([0.515, 0.035, 0.44, 0.03])

        btn_rw_label = "DELTA: ON" if self.model_mode == "compare" else "Raw: ON"
        btn_rf_label = "SpaTrk: ON" if self.model_mode in ["spatracker", "compare"] else "Ref: ON"
        self.btn_prev = Button(ax_prev, "◀ Prev", color=bg)
        self.btn_next = Button(ax_next, "Next ▶", color=bg)
        self.btn_gt   = Button(ax_gt,   "GT: ON",  color=bg)
        self.btn_raw  = Button(ax_raw,  btn_rw_label, color=bg)
        self.btn_ref  = Button(ax_ref,  btn_rf_label, color=bg)
        self.btn_mode = Button(ax_mode, "Pts: Triplet", color=bg)
        self.btn_sync = Button(ax_sync, "Sync: ON", color=bg)
        self.btn_norm = Button(ax_norm, "Norm: ON", color=bg)

        self.btn_prev.on_clicked(lambda _: self._prev())
        self.btn_next.on_clicked(lambda _: self._next())
        self.btn_gt.on_clicked(lambda _: self._toggle_gt())
        self.btn_raw.on_clicked(lambda _: self._toggle_raw())
        self.btn_ref.on_clicked(lambda _: self._toggle_refined())
        self.btn_mode.on_clicked(lambda _: self._toggle_mode())
        self.btn_sync.on_clicked(lambda _: self._toggle_sync())
        self.btn_norm.on_clicked(lambda _: self._toggle_norm())

        self.slider = Slider(ax_slider, "Time (s)", 0.0, 1.0, valinit=0.0, valstep=0.01)
        self.slider.on_changed(self._on_slider_move)

        # Retain explicit references on figure to prevent garbage collection
        self.fig._widgets = [
            self.btn_prev, self.btn_next, self.btn_gt, self.btn_raw,
            self.btn_ref, self.btn_mode, self.btn_sync, self.btn_norm, self.slider
        ]

    def _init_3d_axes(self, ax, title: str, labels: list[str]):
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(labels[0], fontsize=7)
        ax.set_ylabel(labels[1], fontsize=7)
        ax.set_zlabel(labels[2], fontsize=7)

    def _refocus_canvas(self):
        """Restores keyboard focus to the main Matplotlib canvas after button clicks."""
        try:
            if hasattr(self.fig.canvas, "get_tk_widget"):
                self.fig.canvas.get_tk_widget().focus_set()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Data Loading (World Space Aligned via Kinematic Pipeline)
    # -----------------------------------------------------------------------

    def _load_video_entry(self, video_name: str) -> dict:
        """Loads and extracts GT (nominal & synced), Raw, and Refined signals in World frame."""
        ref_sig = None
        raw_sig = None

        if self.model_mode == "spatracker":
            h = SPATRACKER_VGGT_HASHES.get(video_name)
            if h:
                lfs_p = PROJECT_ROOT / ".git" / "lfs" / "objects" / h[:2] / h[2:4] / h
                if lfs_p.exists():
                    try:
                        ref_sig = extract_aligned_signals(lfs_p, self.lowcut, self.highcut, source=self.source)
                    except Exception as e:
                        logger.warning(f"[{video_name}] SpaTracker aligned extraction error: {e}")
        elif self.model_mode == "compare":
            h_spa = SPATRACKER_VGGT_HASHES.get(video_name)
            h_mp = DELTAV2_MEDIAPIPE_HASHES.get(video_name)
            if h_spa:
                lfs_spa = PROJECT_ROOT / ".git" / "lfs" / "objects" / h_spa[:2] / h_spa[2:4] / h_spa
                if lfs_spa.exists():
                    try:
                        ref_sig = extract_aligned_signals(lfs_spa, self.lowcut, self.highcut, source=self.source)
                    except Exception as e:
                        logger.warning(f"[{video_name}] SpaTracker compare extraction error: {e}")
            if h_mp:
                lfs_mp = PROJECT_ROOT / ".git" / "lfs" / "objects" / h_mp[:2] / h_mp[2:4] / h_mp
                if lfs_mp.exists():
                    try:
                        raw_sig = extract_aligned_signals(lfs_mp, self.lowcut, self.highcut, source=self.source)
                    except Exception as e:
                        logger.warning(f"[{video_name}] DELTAv2 compare extraction error: {e}")
        else:
            refined_npz = REFINED_DIR / f"delta_v2_refined_{video_name}.npz"
            raw_npz = RAW_DIR / f"delta_v2_raw_{video_name}.npz"
            if refined_npz.exists():
                try:
                    ref_sig = extract_aligned_signals(refined_npz, self.lowcut, self.highcut, source=self.source)
                except Exception as e:
                    logger.warning(f"[{video_name}] Refined aligned extraction error: {e}")
            if raw_npz.exists():
                try:
                    raw_sig = extract_aligned_signals(raw_npz, self.lowcut, self.highcut, source=self.source)
                except Exception as e:
                    logger.warning(f"[{video_name}] Raw aligned extraction error: {e}")

        ref_ref = ref_sig if ref_sig is not None else raw_sig

        gt_nominal = None
        gt_synced = None
        lag_sec = 0.0
        lag_samples = 0

        if ref_ref is not None:
            t_vis = ref_ref["t_vis"]
            fps = ref_ref["fps"]
            lag_samples = ref_ref.get("temporal_lag", 0)
            lag_sec = lag_samples / fps

            # World-aligned Synced GT
            if "gt_accX" in ref_ref:
                gt_synced = pd.DataFrame({
                    "Timestamp": t_vis,
                    "accX": ref_ref["gt_accX"],
                    "accY": ref_ref["gt_accY"],
                    "accZ": ref_ref["gt_accZ"],
                    "Roll": ref_ref["gt_Roll"],
                    "Pitch": ref_ref["gt_Pitch"],
                    "Yaw": ref_ref["gt_Yaw"],
                })

            # Nominal GT (for unaligned comparison)
            try:
                gt_nominal = load_gt(video_name, t_vis, self.lowcut, self.highcut)
            except Exception:
                gt_nominal = gt_synced

        return {
            "gt_nominal": gt_nominal,
            "gt_synced": gt_synced,
            "lag_sec": lag_sec,
            "lag_samples": lag_samples,
            "raw": raw_sig,
            "refined": ref_sig,
            "t_vis": ref_ref["t_vis"] if ref_ref is not None else np.array([]),
            "fps": ref_ref["fps"] if ref_ref is not None else 30.0,
            "correlations": ref_ref.get("correlations", {}) if ref_ref is not None else {},
        }

    # -----------------------------------------------------------------------
    # Navigation & Actions
    # -----------------------------------------------------------------------

    def _prev(self):
        try:
            if self.idx > 0:
                self.idx -= 1
                self._load_current_video(rescale=True)
        finally:
            self._refocus_canvas()

    def _next(self):
        try:
            if self.idx < len(self.video_names) - 1:
                self.idx += 1
                self._load_current_video(rescale=True)
        finally:
            self._refocus_canvas()

    def _toggle_gt(self):
        try:
            self.show_gt = not self.show_gt
            self.btn_gt.label.set_text("GT: ON" if self.show_gt else "GT: OFF")
            self._update_all_views(rescale=False)
        finally:
            self._refocus_canvas()

    def _toggle_raw(self):
        try:
            self.show_raw = not self.show_raw
            prefix = "DELTA" if self.model_mode == "compare" else "Raw"
            self.btn_raw.label.set_text(f"{prefix}: {'ON' if self.show_raw else 'OFF'}")
            self._update_all_views(rescale=False)
        finally:
            self._refocus_canvas()

    def _toggle_refined(self):
        try:
            self.show_refined = not self.show_refined
            prefix = "SpaTrk" if self.model_mode in ["spatracker", "compare"] else "Ref"
            self.btn_ref.label.set_text(f"{prefix}: {'ON' if self.show_refined else 'OFF'}")
            self._update_all_views(rescale=False)
        finally:
            self._refocus_canvas()

    def _toggle_mode(self):
        try:
            self.source = "cloud" if self.source == "triplet" else "triplet"
            self.btn_mode.label.set_text(f"Pts: {self.source.capitalize()}")
            vname = self.video_names[self.idx]
            self._cache.pop(vname, None)
            self._load_current_video(rescale=True)
        finally:
            self._refocus_canvas()

    def _toggle_sync(self):
        try:
            self.sync_gt = not self.sync_gt
            self.btn_sync.label.set_text("Sync: ON" if self.sync_gt else "Sync: OFF")
            self._update_all_views(rescale=False)
        finally:
            self._refocus_canvas()

    def _toggle_norm(self):
        try:
            self.normalize = not self.normalize
            self.btn_norm.label.set_text("Norm: ON" if self.normalize else "Norm: OFF")
            self._update_all_views(rescale=True)
        finally:
            self._refocus_canvas()

    def _on_key(self, event):
        try:
            if event.key in ("left", "a"):
                self._prev()
            elif event.key in ("right", "d"):
                self._next()
            elif event.key in ("g", "G"):
                self._toggle_gt()
            elif event.key in ("r", "R"):
                self._toggle_raw()
            elif event.key in ("f", "F"):
                self._toggle_refined()
            elif event.key in ("m", "M"):
                self._toggle_mode()
            elif event.key in ("s", "S"):
                self._toggle_sync()
            elif event.key in ("n", "N"):
                self._toggle_norm()
            elif event.key == "up":
                val = min(self.slider.valmax, self.slider.val + 0.1)
                self.slider.set_val(val)
            elif event.key == "down":
                val = max(self.slider.valmin, self.slider.val - 0.1)
                self.slider.set_val(val)
        finally:
            self._refocus_canvas()

    def _on_slider_move(self, val):
        try:
            self.current_time = float(val)
            # Fast update of vertical lines
            for m in self.markers_2d.values():
                m.set_xdata([self.current_time])
            self._update_3d_plots()
            self.fig.canvas.draw_idle()
        finally:
            self._refocus_canvas()

    # -----------------------------------------------------------------------
    # Rendering (Smooth & Fast)
    # -----------------------------------------------------------------------

    def _load_current_video(self, rescale: bool = True):
        vname = self.video_names[self.idx]
        if vname not in self._cache:
            print(f"[{self.idx+1}/{len(self.video_names)}] Loading '{vname}' ...", end=" ", flush=True)
            self._cache[vname] = self._load_video_entry(vname)
            print("done.")

        data = self._cache[vname]
        t_vis = data["t_vis"]
        if len(t_vis) > 0:
            # Temporarily disable slider events to prevent recursive draws during setup
            self.slider.eventson = False
            self.slider.valmin = float(t_vis[0])
            self.slider.valmax = float(t_vis[-1])
            self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
            self.current_time = float(t_vis[0])
            self.slider.set_val(self.current_time)
            self.slider.eventson = True

        self._update_all_views(rescale=rescale)

    def _update_all_views(self, rescale: bool = False):
        vname = self.video_names[self.idx]
        data = self._cache[vname]
        t_vis = data["t_vis"]
        gt = data["gt_synced"] if self.sync_gt else data["gt_nominal"]
        raw = data["raw"]
        refined = data["refined"]

        # Update 2D lines in-place
        for col_name, lines in self.lines_2d.items():
            ax = self.axes_2d[col_name]

            # GT
            if self.show_gt and gt is not None and col_name in gt:
                y_gt = gt[col_name].values
                t_gt = gt["Timestamp"].values if "Timestamp" in gt else t_vis
                n_gt = min(len(t_gt), len(y_gt))
                if self.normalize:
                    y_gt = _norm(y_gt)
                lines["gt"].set_data(t_gt[:n_gt], y_gt[:n_gt])
                lines["gt"].set_visible(True)
            else:
                lines["gt"].set_visible(False)

            # Raw
            if self.show_raw and raw is not None and col_name in raw:
                y_raw = raw[col_name]
                t_raw = raw["t_vis"] if "t_vis" in raw else t_vis
                n_raw = min(len(t_raw), len(y_raw))
                if self.normalize:
                    y_raw = _norm(y_raw)
                lines["raw"].set_data(t_raw[:n_raw], y_raw[:n_raw])
                lines["raw"].set_visible(True)
            else:
                lines["raw"].set_visible(False)

            # Refined
            if self.show_refined and refined is not None and col_name in refined:
                y_ref = refined[col_name]
                t_ref = refined["t_vis"] if "t_vis" in refined else t_vis
                n_ref = min(len(t_ref), len(y_ref))
                if self.normalize:
                    y_ref = _norm(y_ref)
                lines["ref"].set_data(t_ref[:n_ref], y_ref[:n_ref])
                lines["ref"].set_visible(True)
            else:
                lines["ref"].set_visible(False)

            self.markers_2d[col_name].set_xdata([self.current_time])

            if rescale:
                ax.relim()
                ax.autoscale_view()
                ax.legend(fontsize=7, loc="upper right")

        # Update 3D plots
        self._update_3d_plots()

        # Update Title
        cat = self.categories.get(vname, "")
        gt_s  = "GT: ON" if self.show_gt else "GT: OFF"
        raw_s = "Raw: ON" if self.show_raw else "Raw: OFF"
        ref_s = "Ref: ON" if self.show_refined else "Ref: OFF"
        sync_s = f"Sync: ON ({data['lag_samples']:+d}fr)" if self.sync_gt else "Sync: OFF"
        norm_s = "Norm: ON" if self.normalize else "Norm: OFF"
        pts_s  = f"Pts: {self.source.capitalize()}"
        corrs = data.get("correlations", {})
        corr_s = f"Pitch r={corrs.get('Pitch', 0):+.2f} | AccX r={corrs.get('accX', 0):+.2f}" if corrs else ""

        self.fig.suptitle(
            f"[{self.idx+1}/{len(self.video_names)}]  {vname}  |  {cat}  |  {pts_s}  |  {corr_s}  |  {sync_s}  |  {norm_s}",
            fontsize=11,
            fontweight="bold",
        )
        self.fig.canvas.draw_idle()

    def _update_3d_plots(self):
        """Draws 3D acceleration and orientation trajectory trails without ax.cla()."""
        vname = self.video_names[self.idx]
        data = self._cache[vname]
        t_vis = data["t_vis"]
        if len(t_vis) == 0:
            return

        fps = data["fps"]
        idx_t = int(np.clip((self.current_time - t_vis[0]) * fps, 0, len(t_vis) - 1))

        gt = (data["gt_synced"] if self.sync_gt else data["gt_nominal"]) if self.show_gt else None
        raw = data["raw"] if self.show_raw else None
        refined = data["refined"] if self.show_refined else None

        def _get_mat(src, cols):
            if src is None:
                return None
            vals = np.column_stack([src[c].values if isinstance(src, pd.DataFrame) else src[c] for c in cols])
            if self.normalize:
                for j in range(3):
                    vals[:, j] = _norm(vals[:, j])
            return vals

        self._render_3d_fast(
            ax=self.ax3d_acc,
            cols=["accX", "accY", "accZ"],
            gt_mat=_get_mat(gt, ["accX", "accY", "accZ"]),
            raw_mat=_get_mat(raw, ["accX", "accY", "accZ"]),
            ref_mat=_get_mat(refined, ["accX", "accY", "accZ"]),
            cmap="viridis",
            idx_t=idx_t,
        )

        self._render_3d_fast(
            ax=self.ax3d_ori,
            cols=["Roll", "Pitch", "Yaw"],
            gt_mat=_get_mat(gt, ["Roll", "Pitch", "Yaw"]),
            raw_mat=_get_mat(raw, ["Roll", "Pitch", "Yaw"]),
            ref_mat=_get_mat(refined, ["Roll", "Pitch", "Yaw"]),
            cmap="plasma",
            idx_t=idx_t,
        )

    def _render_3d_fast(self, ax, cols, gt_mat, raw_mat, ref_mat, cmap, idx_t):
        # Remove only collections and lines (preserves 3D camera and axes, ultra-fast & safe)
        for item in list(ax.collections) + list(ax.lines):
            try:
                item.remove()
            except Exception:
                pass

        # GT trail
        if gt_mat is not None and len(gt_mat) > 0:
            ax.plot(gt_mat[:, 0], gt_mat[:, 1], gt_mat[:, 2], color="#888888", alpha=0.35, linewidth=1.0, linestyle="--")

        # Raw trail
        if raw_mat is not None and len(raw_mat) > 0:
            raw_sub = raw_mat[:idx_t + 1]
            if len(raw_sub) > 1:
                ax.plot(raw_sub[:, 0], raw_sub[:, 1], raw_sub[:, 2], color="#2196F3", alpha=0.6, linewidth=1.2)
                ax.scatter(raw_sub[-1, 0], raw_sub[-1, 1], raw_sub[-1, 2], color="#1976D2", s=20)

        # Refined trail with colormap
        if ref_mat is not None and len(ref_mat) > 0:
            ref_sub = ref_mat[:idx_t + 1]
            if len(ref_sub) > 1:
                segs = np.concatenate([ref_sub[:-1].reshape(-1, 1, 3), ref_sub[1:].reshape(-1, 1, 3)], axis=1)
                lc = Line3DCollection(segs, cmap=cmap, linewidth=2.2, alpha=0.9)
                lc.set_array(np.linspace(0, 1, len(ref_sub) - 1))
                ax.add_collection(lc)
                ax.scatter(ref_sub[-1, 0], ref_sub[-1, 1], ref_sub[-1, 2], color="red", s=40, zorder=10)

        # Set limits
        valid_mats = [m for m in (gt_mat, raw_mat, ref_mat) if m is not None and len(m) > 0]
        if valid_mats:
            all_pts = np.vstack(valid_mats)
            ax.set_xlim(all_pts[:, 0].min() - 0.1, all_pts[:, 0].max() + 0.1)
            ax.set_ylim(all_pts[:, 1].min() - 0.1, all_pts[:, 1].max() + 0.1)
            ax.set_zlim(all_pts[:, 2].min() - 0.1, all_pts[:, 2].max() + 0.1)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive 2D/3D Gait Benchmark Visualizer")
    parser.add_argument("--start", type=int, default=0, help="Initial video index")
    parser.add_argument(
        "--model",
        choices=["deltav2", "spatracker", "compare"],
        default="deltav2",
        help="Model to visualize: 'deltav2' (40 videos), 'spatracker' (6 benchmark videos), or 'compare' (DELTAv2 vs SpaTracker vs GT side-by-side on 6 videos)",
    )
    args = parser.parse_args()

    lowcut, highcut = load_cutoffs(PROJECT_ROOT)
    print(f"Frequency band: [{lowcut:.3f} Hz - {highcut:.3f} Hz]")

    if args.model in ["spatracker", "compare"]:
        video_names = list(SPATRACKER_VGGT_HASHES.keys())
        categories = BENCHMARK_CATEGORIES_6
        print(f"Loaded {len(video_names)} benchmark videos for mode '{args.model}'.")
    else:
        if not MANIFEST_PATH.exists():
            print(f"ERROR: Benchmark manifest not found at {MANIFEST_PATH}")
            sys.exit(1)

        df = pd.read_csv(MANIFEST_PATH)
        video_names = df["video_name"].tolist()
        categories = dict(zip(df["video_name"], df["category_name"])) if "category_name" in df.columns else {}
        print(f"Loaded benchmark manifest: {len(video_names)} videos for mode '{args.model}'.")

    dashboard = GaitBenchmarkDashboard(
        video_names=video_names,
        categories=categories,
        lowcut=lowcut,
        highcut=highcut,
        start_idx=args.start,
        model_mode=args.model,
    )
    _ACTIVE_DASHBOARDS.append(dashboard)

    plt.show()


if __name__ == "__main__":
    main()
