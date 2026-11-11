import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from src.preprocessing.signals.signal_orchestrator import SignalOrchestrator
from src.data_io.dataset_scanner import DatasetScanner

class Unified3DViewer:
    def __init__(self, all_signals, invalid_set, lowcut, highcut):
        self.file_list = all_signals
        self.invalid_set = invalid_set
        self.lowcut = lowcut
        self.highcut = highcut
        self.index = 0
        self.num_files = len(all_signals)
        self.orchestrator = SignalOrchestrator()
        
        # Animation & Data State
        self.current_frame = 0.0
        self.is_playing = False
        self.timer = None
        self.df_filt = None 
        self.speed_mode = "1x" # Default to real speed
        self.fs = 83.33
        self.last_time = 0

        # Setup Figure
        self.fig = plt.figure(figsize=(16, 10))
        plt.subplots_adjust(bottom=0.25)
        
        self.ax_acc = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_ang = self.fig.add_subplot(1, 2, 2, projection='3d')

        # Widgets
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Time', 0, 100, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        ax_play = plt.axes([0.40, 0.03, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_play.on_clicked(self.toggle_play)

        ax_speed = plt.axes([0.52, 0.03, 0.1, 0.04])
        self.btn_speed = Button(ax_speed, f'Speed: {self.speed_mode}')
        self.btn_speed.on_clicked(self.toggle_speed)

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.update_file()

        print(f"Unified 3D Signal Inspector (Real-Time Sync) started.")
        print(f"Controls: [Arrows] File | [Space] Play/Pause | [S] Speed | [Q] Quit")

    def load_and_process(self):
        csv_path = Path(self.file_list[self.index])
        if not csv_path.is_absolute():
            csv_path = Path(__file__).resolve().parents[2] / csv_path
        df_raw = self.orchestrator.loader.load(csv_path)
        if df_raw is None:
            return None, None
        
        self.fs = self._calculate_fs(df_raw['Timestamp'].values)
        df_unwrapped = self.orchestrator.estimator._unwrap_angles(df_raw.copy())
        df_filtered = self.orchestrator.processor.process(df_unwrapped, self.lowcut, self.highcut)
        return df_unwrapped, df_filtered

    def _calculate_fs(self, ts):
        diffs = np.diff(ts)
        if len(diffs) == 0: return 83.33
        dt = np.median(diffs)
        return 1.0 / dt if dt > 0 else 83.33

    def update_file(self):
        df_raw, df_filt = self.load_and_process()
        if df_raw is None: return

        self.df_filt = df_filt
        self.max_frames = len(df_filt)
        
        # Cache centered raw data
        self.x_raw = df_raw.iloc[:, 1].values - df_raw.iloc[:, 1].mean()
        self.y_raw = df_raw.iloc[:, 2].values - df_raw.iloc[:, 2].mean()
        self.z_raw = df_raw.iloc[:, 3].values - df_raw.iloc[:, 3].mean()
        
        self.roll_raw = df_raw['Roll'].values - df_raw['Roll'].mean()
        self.pitch_raw = df_raw['Pitch'].values - df_raw['Pitch'].mean()
        self.yaw_raw = df_raw['Yaw'].values - df_raw['Yaw'].mean()

        # Set static limits once per file to speed up rendering
        self._set_static_limits()

        self.slider.valmax = self.max_frames - 1
        self.slider.ax.set_xlim(0, self.max_frames - 1)
        
        self.current_frame = 0.0
        self.slider.set_val(0) 
        self.render_frame()

    def _set_static_limits(self):
        # Calculate limits once to avoid expensive calls during animation
        for ax, cols in [(self.ax_acc, ['accX', 'accY', 'accZ']), 
                        (self.ax_ang, ['Roll', 'Pitch', 'Yaw'])]:
            data = self.df_filt[cols]
            ax.set_xlim(data.iloc[:, 0].min()*1.2, data.iloc[:, 0].max()*1.2)
            ax.set_ylim(data.iloc[:, 1].min()*1.2, data.iloc[:, 1].max()*1.2)
            ax.set_zlim(data.iloc[:, 2].min()*1.2, data.iloc[:, 2].max()*1.2)

    def on_slider_change(self, val):
        if self.df_filt is None: return
        self.current_frame = float(val)
        self.render_frame()

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_time = time.time()
            self.start_animation()
        else:
            self.stop_animation()

    def toggle_speed(self, event):
        self.speed_mode = "1x" if self.speed_mode == "rallenty" else "rallenty"
        self.btn_speed.label.set_text(f'Speed: {self.speed_mode}')
        self.fig.canvas.draw_idle()

    def start_animation(self):
        if self.timer is None:
            self.timer = self.fig.canvas.new_timer(interval=20) # High frequency tick
            self.timer.add_callback(self.advance_frame)
        self.timer.start()

    def stop_animation(self):
        if self.timer:
            self.timer.stop()

    def advance_frame(self):
        if not self.is_playing: return
        
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Calculate real-time step
        multiplier = 1.0 if self.speed_mode == "1x" else 0.2
        step = self.fs * dt * multiplier
            
        self.current_frame = (self.current_frame + step) % self.max_frames
        self.slider.set_val(int(self.current_frame))

    def render_frame(self):
        if self.df_filt is None: return
        frame_idx = int(self.current_frame)

        # Save view
        elev_acc, azim_acc = self.ax_acc.elev, self.ax_acc.azim
        elev_ang, azim_ang = self.ax_ang.elev, self.ax_ang.azim

        self._plot_dynamic(self.ax_acc, 'accX', 'accY', 'accZ', '3D Acceleration', 'viridis', frame_idx)
        self._plot_dynamic(self.ax_ang, 'Roll', 'Pitch', 'Yaw', '3D Orientation', 'plasma', frame_idx)

        # Restore view
        if elev_acc is not None:
            self.ax_acc.view_init(elev=elev_acc, azim=azim_acc)
            self.ax_ang.view_init(elev=elev_ang, azim=azim_ang)

        csv_path_str = self.file_list[self.index]
        self.fig.suptitle(f"3D View - {Path(csv_path_str).name}\nSpeed: {self.speed_mode} | {self.fs:.1f} Hz",
                          fontsize=12, fontweight='bold')
        self.fig.canvas.draw_idle()

    def _plot_dynamic(self, ax, cx, cy, cz, title, cmap_name, frame_idx):
        # Only clear data lines, not labels/limits (Saves CPU)
        for line in list(ax.collections) + list(ax.lines):
            line.remove()
        
        idx = frame_idx + 1
        if cx == 'accX':
            xr, yr, zr = self.x_raw[:idx], self.y_raw[:idx], self.z_raw[:idx]
        else:
            xr, yr, zr = self.roll_raw[:idx], self.pitch_raw[:idx], self.yaw_raw[:idx]
        
        xf, yf, zf = self.df_filt[cx].values[:idx], self.df_filt[cy].values[:idx], self.df_filt[cz].values[:idx]

        if len(xf) > 1:
            ax.plot(xr, yr, zr, color='#888888', alpha=0.4, linewidth=0.8)
            points = np.array([xf, yf, zf]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t = np.linspace(0, 1, len(segments))
            lc = Line3DCollection(segments, cmap=cmap_name, linewidth=2.0, alpha=0.9)
            lc.set_array(t)
            ax.add_collection(lc)
            ax.scatter(xf[-1], yf[-1], zf[-1], color='red', s=30, zorder=12)

    def on_mouse_move(self, event):
        if event.inaxes == self.ax_acc:
            self.ax_ang.view_init(elev=self.ax_acc.elev, azim=self.ax_acc.azim)
        elif event.inaxes == self.ax_ang:
            self.ax_acc.view_init(elev=self.ax_ang.elev, azim=self.ax_ang.azim)

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % self.num_files
            self.update_file()
        elif event.key == 'left':
            self.index = (self.index - 1) % self.num_files
            self.update_file()
        elif event.key == ' ': self.toggle_play(None)
        elif event.key == 's': self.toggle_speed(None)
        elif event.key == 'q': plt.close()

def inspect_signals_3d(mode="all", data_dir="data"):
    orchestrator = SignalOrchestrator()
    invalid_set = set()
    if orchestrator.invalid_cache_path.exists():
        with open(orchestrator.invalid_cache_path, "r") as f:
            invalid_set = set(json.load(f))
    lowcut, highcut = 0.5, 10.0
    if orchestrator.cutoff_cache_path.exists():
        with open(orchestrator.cutoff_cache_path, "r") as f:
            data = json.load(f)
            lowcut, highcut = data["lowcut"], data["highcut"]
    scanner = DatasetScanner(data_dir)
    all_csvs = scanner.get_csvs()
    invalid_list = sorted([c for c in all_csvs if c in invalid_set])
    valid_list = sorted([c for c in all_csvs if c not in invalid_set])
    if mode == "invalid": signals_to_show = invalid_list
    elif mode == "valid": signals_to_show = valid_list
    else: signals_to_show = invalid_list + valid_list
    if not signals_to_show: return
    browser = Unified3DViewer(signals_to_show, invalid_set, lowcut, highcut)
    plt.show()


Unified3DInspector = Unified3DViewer
