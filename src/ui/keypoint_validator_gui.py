import logging
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import numpy as np

from src.core.models import VideoInfo, ProcessingState, VideoMetadata
from src.data_io.openpose_reader import openpose_reader, OPENPOSE_MAP
from src.data_io.video_reader import video_provider
from src.preprocessing.vision.vision_orchestrator import FemurVisionOrchestrator
from src.preprocessing.video_prepper import video_service

logger = logging.getLogger(__name__)

class ValidatorGUI:
    def __init__(self, videos_iterator, video_types, skip_if_already_done=True):
        self.videos_iterator = videos_iterator
        self.video_types = video_types if isinstance(video_types, list) else [video_types]
        self.skip_if_already_done = skip_if_already_done
        self.videos_processed = []
        self.is_loaded_from_state = False

        # State variables
        self.focus_mode = 0  # 0: None, 1: Person, 2: Thigh
        self.show_mask = True
        self.show_skeleton = True

        self.clear()
        self._setup_plot()

        # Start the loop
        self.next_video()

    def _setup_plot(self):
        """Initializes the matplotlib figure and event listeners."""
        self.fig, self.ax_rgb = plt.subplots(1, 1, figsize=(10, 7))
        plt.subplots_adjust(bottom=0.2)

        # Connect mouse events for drag and drop
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.buttons()
        self.keyboard_keys()

    def clear(self):
        self.video_info: VideoInfo = None
        self.metadata: VideoMetadata = None
        self.img = None
        self.frame_idx = 0
        self.keypoints = None
        self.fullbody_keypoints = None
        self.mask_base = None
        self.mask_inner = None
        self.scatter = None
        self.polygon_patch = None
        self.dragging_idx = None

    def discard_all(self):
        if self.video_info:
            logger.info(f"Discarding video {self.video_info.name} (Both)")
            video_service.discard_specific(self.video_info.name)
            video_service.discard_fullbody(self.video_info.name)
        self.next_video(skip_save=True)

    def discard_specific(self):
        if self.video_info:
            logger.info(f"Discarding video {self.video_info.name} (Specific only)")
            video_service.discard_specific(self.video_info.name)
        self.next_video(skip_save=True)

    def discard_fullbody(self):
        if self.video_info:
            logger.info(f"Discarding video {self.video_info.name} (Fullbody only)")
            video_service.discard_fullbody(self.video_info.name)
        self.next_video(skip_save=True)

    def first_frame(self):
        if self.metadata and self.frame_idx > 0:
            self.frame_idx = 0
            self._fetch_and_render_frame()

    def last_frame(self):
        if self.metadata and self.frame_idx < self.metadata.frames_num - 1:
            self.frame_idx = self.metadata.frames_num - 1
            self._fetch_and_render_frame()

    def next_frame(self):
        if self.metadata and self.frame_idx < self.metadata.frames_num - 1:
            self.frame_idx += 1
            self._fetch_and_render_frame()

    def prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self._fetch_and_render_frame()

    def _fetch_and_render_frame(self):
        # We don't need the skeleton path anymore, just the frame
        frame_bgr, _ = video_provider.get_frame(self.video_info.video_path, self.frame_idx)
        import cv2
        self.img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr is not None else None

        json_path = self.video_info.get_frame_keypoints_path(self.frame_idx)
        raw_kps_dict = openpose_reader.load_frame_keypoints(Path(json_path))
        
        # Adapt to internal format for GUI drawing (reconstruct flat array logic if needed or use dict)
        # The GUI logic expects dicts now in many places
        
        # Simple extraction logic to adapt to vision_orchestrator
        extracted = {}
        self.fullbody_keypoints = {}
        if raw_kps_dict:
            for kp_name in self.video_info.interests_keypoints:
                if kp_name in raw_kps_dict:
                    val = raw_kps_dict[kp_name]
                    if val[2] > 0.05:
                        extracted[kp_name] = {"x": val[0], "y": val[1], "confidence": val[2]}

            # Extract fullbody for skeleton and focus
            for kp_name, val in raw_kps_dict.items():
                if val[2] > 0.1: # 0.1 confidence threshold for drawing
                    self.fullbody_keypoints[kp_name] = {"x": val[0], "y": val[1], "confidence": val[2]}

        if not hasattr(self, 'vision_orchestrator'):
            self.vision_orchestrator = FemurVisionOrchestrator()
            
        new_triangle, m_base, m_inner = self.vision_orchestrator.compose(extracted, frame_bgr)

        if new_triangle: self.keypoints = new_triangle
        self.mask_base, self.mask_inner = m_base, m_inner
        self.visualize()

    def visualize(self):
        if self.img is None: return

        self.ax_rgb.clear()

        self.ax_rgb.imshow(self.img)
        title_prefix = "[LOADED] " if self.is_loaded_from_state else "[AUTO] "
        self.ax_rgb.set_title(f"{title_prefix}{self.video_info.name} | Frame: {self.frame_idx}\n(Drag points to adjust)")
        self.ax_rgb.axis('off')

        # Draw overlays if masks exist and show_mask is true
        if self.show_mask and hasattr(self, 'mask_inner') and self.mask_inner is not None:
            is_inner = self.mask_inner > 0
            is_base = self.mask_base > 0
            is_base_only = is_base & (~is_inner)

            overlay_inner = np.zeros((*self.mask_inner.shape, 4), dtype=np.float32)
            overlay_inner[is_inner] = [0.0, 1.0, 0.0, 0.4]

            overlay_base = np.zeros((*self.mask_base.shape, 4), dtype=np.float32)
            overlay_base[is_base_only] = [1.0, 0.65, 0.0, 0.4]

            self.ax_rgb.imshow(overlay_inner)
            self.ax_rgb.imshow(overlay_base)

        # Draw triangle and points
        if self.keypoints and len(self.keypoints) == 3:
            xs = [p[0] for p in self.keypoints]
            ys = [p[1] for p in self.keypoints]

            if all(x > 0 for x in xs):
                poly_coords = np.column_stack((xs, ys))
                self.polygon_patch = patches.Polygon(poly_coords, closed=True, fill=True, color='cyan', alpha=0.3, zorder=4)
                self.ax_rgb.add_patch(self.polygon_patch)
                self.scatter = self.ax_rgb.scatter(xs, ys, c=['blue', 'red', 'red'], s=40, edgecolors='white', zorder=5)

        # Draw fullbody skeleton
        if self.show_skeleton and self.fullbody_keypoints:
            self._draw_skeleton()

        self.focus()

        self.fig.canvas.draw_idle()

    def _draw_skeleton(self):
        """Draws the OpenPose fullbody skeleton directly on the ax_rgb plot."""
        # Standard OpenPose BODY_25 pairs
        skeleton_pairs = [
            ("Nose", "Neck"), ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
            ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("Neck", "MidHip"),
            ("MidHip", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("MidHip", "LHip"),
            ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("Nose", "REye"), ("REye", "REar"),
            ("Nose", "LEye"), ("LEye", "LEar"), ("LAnkle", "LBigToe"), ("LAnkle", "LSmallToe"),
            ("LAnkle", "LHeel"), ("RAnkle", "RBigToe"), ("RAnkle", "RSmallToe"), ("RAnkle", "RHeel")
        ]

        for kp1_name, kp2_name in skeleton_pairs:
            if kp1_name in self.fullbody_keypoints and kp2_name in self.fullbody_keypoints:
                p1 = self.fullbody_keypoints[kp1_name]
                p2 = self.fullbody_keypoints[kp2_name]
                if p1["confidence"] > 0.1 and p2["confidence"] > 0.1:
                    line = plt.Line2D([p1["x"], p2["x"]], [p1["y"], p2["y"]], color='yellow', linewidth=2, alpha=0.6, zorder=3)
                    self.ax_rgb.add_line(line)

        # Scatter for skeleton joints
        kx = [p["x"] for p in self.fullbody_keypoints.values() if p["confidence"] > 0.1]
        ky = [p["y"] for p in self.fullbody_keypoints.values() if p["confidence"] > 0.1]
        if kx and ky:
            self.ax_rgb.scatter(kx, ky, c='yellow', s=10, zorder=4, alpha=0.6)

    def next_video(self, skip_save=False):
        # Save current state before moving on
        if self.video_info:
            if self.video_info.sensor_data_path is None or not self.video_info.sensor_data_path:    # Excluded videos where we have not sensors data
                self.next_video(skip_save=True)
                return
            if not skip_save:
                self.update_video_state()
                self.videos_processed.append(self.video_info)

        self.clear()

        try:
            self.video_info = next(self.videos_iterator)
            # Skip invalid video types if handled by specific subclass logic
            while not self._is_valid_video_type():
                self.video_info = next(self.videos_iterator)

            self.metadata = video_provider.get_metadata(str(self.video_info.video_path))
            self.fig.canvas.manager.set_window_title(f"Validating: {self.video_info.name}")
            self.get_start_frame()

        except StopIteration:
            logger.info("All videos validated! Closing GUI.")
            plt.close(self.fig)

    def _is_valid_video_type(self):
        return self.video_info.sensor_position in self.video_types

    def keyboard_keys(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        if event.key == 'left': self.prev_frame()
        elif event.key == 'right': self.next_frame()
        elif event.key == 'up': self.first_frame()
        elif event.key == 'down': self.last_frame()
        elif event.key == 'enter': self.next_video()
        elif event.key in ['delete', 'backspace']: self.discard_all()
        elif event.key.lower() == 'f': self.toggle_focus()
        elif event.key.lower() == 'm': self.toggle_mask()
        elif event.key.lower() == 's': self.toggle_skeleton()

    def toggle_focus(self):
        self.focus_mode = (self.focus_mode + 1) % 3
        self.visualize()

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.visualize()

    def toggle_skeleton(self):
        self.show_skeleton = not self.show_skeleton
        self.visualize()

    def buttons(self):
        # We adjust the button layout for a single plot
        y_pos = 0.05
        btn_height = 0.075

        # Navigation
        self.ax_btn_first = plt.axes([0.02, y_pos, 0.06, btn_height])
        self.btn_first = Button(self.ax_btn_first, '<< First')
        self.btn_first.on_clicked(lambda e: self.first_frame())

        self.ax_btn_prev = plt.axes([0.09, y_pos, 0.06, btn_height])
        self.btn_prev = Button(self.ax_btn_prev, '< Prev')
        self.btn_prev.on_clicked(lambda e: self.prev_frame())

        self.ax_btn_next = plt.axes([0.16, y_pos, 0.06, btn_height])
        self.btn_next = Button(self.ax_btn_next, 'Next >')
        self.btn_next.on_clicked(lambda e: self.next_frame())

        self.ax_btn_last = plt.axes([0.23, y_pos, 0.06, btn_height])
        self.btn_last = Button(self.ax_btn_last, 'Last >>')
        self.btn_last.on_clicked(lambda e: self.last_frame())

        # Toggles
        self.ax_btn_focus = plt.axes([0.31, y_pos, 0.06, btn_height])
        self.btn_focus = Button(self.ax_btn_focus, 'Focus')
        self.btn_focus.on_clicked(lambda e: self.toggle_focus())

        self.ax_btn_mask = plt.axes([0.38, y_pos, 0.06, btn_height])
        self.btn_mask = Button(self.ax_btn_mask, 'Mask')
        self.btn_mask.on_clicked(lambda e: self.toggle_mask())

        self.ax_btn_skel = plt.axes([0.45, y_pos, 0.07, btn_height])
        self.btn_skel = Button(self.ax_btn_skel, 'Skeleton')
        self.btn_skel.on_clicked(lambda e: self.toggle_skeleton())

        # Actions
        self.ax_btn_discard_all = plt.axes([0.54, y_pos, 0.09, btn_height])
        self.btn_discard_all = Button(self.ax_btn_discard_all, 'Discard All', color='lightcoral')
        self.btn_discard_all.on_clicked(lambda e: self.discard_all())

        self.ax_btn_discard_spec = plt.axes([0.64, y_pos, 0.10, btn_height])
        self.btn_discard_spec = Button(self.ax_btn_discard_spec, 'Discard Spec', color='lightcoral')
        self.btn_discard_spec.on_clicked(lambda e: self.discard_specific())

        self.ax_btn_discard_full = plt.axes([0.75, y_pos, 0.10, btn_height])
        self.btn_discard_full = Button(self.ax_btn_discard_full, 'Discard Full', color='lightcoral')
        self.btn_discard_full.on_clicked(lambda e: self.discard_fullbody())

        self.ax_btn_save = plt.axes([0.87, y_pos, 0.11, btn_height])
        self.btn_save = Button(self.ax_btn_save, 'Save & Next', color='lightgreen')
        self.btn_save.on_clicked(lambda e: self.next_video())

    def update_video_state(self):
        logger.info(f"Saving Reference updates for {self.video_info.name} at Frame {self.frame_idx}...")
        video_service.update_reference_frame(self.video_info, self.frame_idx, self.keypoints)

    def get_start_frame(self):
        result = video_service.get_reference_frame(self.video_info, self.metadata, self.skip_if_already_done)
        if not result:
            return self.next_video(skip_save=True)

        frame_rgb, state, m_base, m_inner = result
        
        # Check if we actually loaded something pre-existing
        self.is_loaded_from_state = (state.ref_frame_idx is not None)

        if state.specific_discarded and state.fullbody_discarded:
            return self.next_video(skip_save=True)
        if frame_rgb is None:
            return self.next_video(skip_save=True)

        self.img = frame_rgb
        self.frame_idx = state.ref_frame_idx if state.ref_frame_idx is not None else 0
        self.keypoints = state.specific_keypoints
        self.fullbody_keypoints = state.fullbody_keypoints
        self.mask_base, self.mask_inner = m_base, m_inner
        self.visualize()

    def focus(self):
        if self.img is None: return
        h, w = self.img.shape[:2]

        if self.focus_mode == 0:
            self.ax_rgb.set_xlim(0, w)
            self.ax_rgb.set_ylim(h, 0)
            return

        xs, ys = [], []
        padding = 100

        if self.focus_mode == 1:  # PERSON FOCUS (Triangle + Skeleton)
            if self.keypoints:
                xs.extend([p[0] for p in self.keypoints if p[0] > 0])
                ys.extend([p[1] for p in self.keypoints if p[1] > 0])

            if self.fullbody_keypoints:
                xs.extend([p["x"] for p in self.fullbody_keypoints.values() if p["x"] > 0])
                ys.extend([p["y"] for p in self.fullbody_keypoints.values() if p["y"] > 0])
            padding = 100
        elif self.focus_mode == 2:  # THIGH FOCUS (Only triangle)
            if self.keypoints:
                xs.extend([p[0] for p in self.keypoints if p[0] > 0])
                ys.extend([p[1] for p in self.keypoints if p[1] > 0])
            padding = 50

        if xs and ys:
            xlim_min, xlim_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
            ylim_min, ylim_max = max(0, min(ys) - padding), min(h, max(ys) + padding)

            self.ax_rgb.set_xlim(xlim_min, xlim_max)
            self.ax_rgb.set_ylim(ylim_max, ylim_min)

    # --- Mouse Events for Dragging ---
    def on_press(self, event):
        if event.inaxes != self.ax_rgb or self.scatter is None or not self.keypoints: return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None: return

        min_dist = float('inf')
        closest_idx = None

        for i, (px, py) in enumerate(self.keypoints):
            dist = math.hypot(click_x - px, click_y - py)
            if dist < min_dist and dist < 15.0:  # 15.0 is tolerance
                min_dist = dist
                closest_idx = i

        if closest_idx is not None:
            self.dragging_idx = closest_idx

    def on_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax_rgb: return

        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None: return

        self.keypoints[self.dragging_idx] = [mouse_x, mouse_y]

        xs = [p[0] for p in self.keypoints]
        ys = [p[1] for p in self.keypoints]

        self.scatter.set_offsets(np.c_[xs, ys])
        if self.polygon_patch is not None:
            self.polygon_patch.set_xy(np.column_stack((xs, ys)))

        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging_idx is not None:
            self.dragging_idx = None


def validate_first_frame_of_all_videos(videos_iterator, considered_video_types, skip_if_already_done=True):
    videos_iterator = iter(videos_iterator)
    gui = ValidatorGUI(videos_iterator, considered_video_types, skip_if_already_done)

    plt.show()
    return gui.videos_processed