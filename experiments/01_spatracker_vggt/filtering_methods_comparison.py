"""
Tool to compare different filtering methods for 3D tracking data.
Usage:
    python compare_filtering_methods_plot.py [--video VIDEO_NAME]

Arguments:
    --video: Video name to analyze (e.g. P1_E1T1_45_ALT.mp4). Default: P1_E1T1_45_ALT.mp4.
"""

import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.link_analysis import hits_alg
from scipy import signal
import os
import json
from pathlib import Path
from matplotlib.widgets import RadioButtons, Slider

# Configurazione Percorsi
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()

# --- FUNZIONI DI FILTRAGGIO ---

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0: normal_cutoff = 0.99
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def savgol_filt(data, window=11, poly=3):
    # Assicura finestra dispari
    if window % 2 == 0: window += 1
    return signal.savgol_filter(data, window, poly, axis=0)

def load_dynamic_cutoffs():
    print("loading cutoff freqs..")
    cache_path = PROJECT_ROOT / "data" / "gt_cache" / "cutoff_freqs.json"
    print(cache_path)
    lowcut, highcut = 0.5, 10.0
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
            lowcut, highcut = data["lowcut"], data["highcut"]
        print(f"cutoff freqs: {lowcut}, {highcut}")
    except: print("Not possible to load Cutoff Frequencies")
    return lowcut, highcut

class FilterComparisonTool:
    def __init__(self, file_path):
        print(f"Initializing Filter Comparison Tool for {file_path}...")
        data = np.load(file_path)
        
        # Carichiamo solo il Ginocchio (P1) per chiarezza, o la tripletta intera?
        # Use P1 only for stability analysis
        self.raw_data = data['track3d'][:, 0, :] 
        self.T = self.raw_data.shape[0]
        self.video_name = str(data.get('video_name', 'Unknown'))
        self.fs = 30.0
        self.junction_idx = int(data.get('ref_frame_idx', 44))
        
        _, self.base_hc = load_dynamic_cutoffs()
        
        # --- PRE-COMPUTE FILTERS ---
        self.filters = {
            "RAW": self.raw_data,
            "Standard (Butt 4th)": np.array([butter_lowpass(self.raw_data[:, d], self.base_hc, self.fs, 4) for d in range(3)]).T,
            "Steep (Butt 8th)": np.array([butter_lowpass(self.raw_data[:, d], self.base_hc, self.fs, 8) for d in range(3)]).T,
            "Aggressive (Butt 4th, 0.66*HC)": np.array([butter_lowpass(self.raw_data[:, d], self.base_hc*0.66, self.fs, 4) for d in range(3)]).T,
            "Savitzky-Golay": np.array([savgol_filt(self.raw_data[:, d], window=11, poly=3) for d in range(3)]).T
        }
        
        self.current_filter_name = "Standard (Butt 4th)"
        self.colors = ['black', 'blue', 'red', 'green', 'purple']
        
        # Setup Figure
        self.fig = plt.figure(figsize=(20, 12))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
        
        # --- 1D SUBPLOTS (Left Column) ---
        self.ax1d = [self.fig.add_subplot(3, 2, i*2+1) for i in range(3)]
        self.labels = ['X (Progress)', 'Y (Height)', 'Z (Depth)']
        
        # --- 3D/2D SUBPLOTS (Right Column) ---
        self.ax3d = self.fig.add_subplot(2, 2, 2, projection='3d')
        self.ax2d = self.fig.add_subplot(2, 2, 4) # Side View
        
        # Widgets
        ax_radio = plt.axes([0.4, 0.02, 0.2, 0.06], facecolor='#f0f0f0')
        self.radio = RadioButtons(ax_radio, list(self.filters.keys()), active=1)
        self.radio.on_clicked(self.on_filter_change)
        
        ax_slider = plt.axes([0.1, 0.02, 0.2, 0.02])
        self.slider = Slider(ax_slider, 'Frame', 0, self.T-1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_plots)

        # Inizializzazione limiti
        self.lim_xy = np.max(np.abs(self.raw_data[:, :2] - self.raw_data[self.junction_idx, :2])) * 1.2
        self.lim_z = np.max(np.abs(self.raw_data[:, 2] - self.raw_data[self.junction_idx, 2])) * 1.5

        self.update_plots(0)
        print("Comparison Tool ready. Select filter to update 3D/2D views.")
        plt.show()

    def on_filter_change(self, label):
        self.current_filter_name = label
        self.update_plots(self.slider.val)

    def update_plots(self, val):
        t = int(self.slider.val)
        current_data = self.filters[self.current_filter_name]
        origin = self.raw_data[self.junction_idx]
        
        # 1. Update 1D Comparison (Sovrapposti)
        for d in range(3):
            ax = self.ax1d[d]
            ax.cla()
            for i, (name, f_data) in enumerate(self.filters.items()):
                # Usiamo dati relativi all'origine per i grafici 1D? No, assoluti ma centrati visivamente
                sig = f_data[:, d] - origin[d]
                alpha = 1.0 if name == self.current_filter_name else 0.3
                lw = 2.0 if name == self.current_filter_name else 1.0
                ax.plot(sig, label=name, color=self.colors[i], alpha=alpha, linewidth=lw)
            
            ax.axvline(x=self.junction_idx, color='gray', linestyle='--', alpha=0.5)
            ax.scatter(t, current_data[t, d] - origin[d], color='red', s=40, zorder=10)
            ax.set_ylabel(self.labels[d])
            if d == 0: ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.2)

        # 2. Update 3D Perspective
        self._draw_3d(self.ax3d, current_data - origin, t)
        
        # 3. Update 2D Side View
        self._draw_2d(self.ax2d, current_data - origin, t)
        
        self.fig.suptitle(f"Filter Comparison: {self.video_name} | Active: {self.current_filter_name}\n"
                          f"Base Highcut: {self.base_hc:.2f}Hz", fontweight='bold', fontsize=14)
        self.fig.canvas.draw_idle()

    def _draw_3d(self, ax, data, t):
        elev, azim = ax.elev, ax.azim
        ax.cla()
        # X=X, Y=Z(Depth), Z=Y(Height)
        ax.plot(data[:t+1, 0], data[:t+1, 2], data[:t+1, 1], color='blue', alpha=0.7)
        ax.scatter(data[self.junction_idx, 0], data[self.junction_idx, 2], data[self.junction_idx, 1], color='yellow', s=100, edgecolors='black')
        ax.scatter(data[t, 0], data[t, 2], data[t, 1], color='red', s=50)
        
        ax.set_xlim(-self.lim_xy, self.lim_xy)
        ax.set_ylim(-self.lim_z, self.lim_z)
        ax.set_zlim(-self.lim_xy, self.lim_xy)
        ax.set_xlabel('X (Lat)')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Height)')
        ax.set_title(f"3D View: {self.current_filter_name}", fontsize=10)
        ax.view_init(elev=elev, azim=azim)

    def _draw_2d(self, ax, data, t):
        ax.cla()
        ax.plot(data[:t+1, 0], data[:t+1, 1], color='blue', alpha=0.7)
        ax.scatter(data[self.junction_idx, 0], data[self.junction_idx, 1], color='yellow', s=100, edgecolors='black', zorder=10)
        ax.scatter(data[t, 0], data[t, 1], color='red', s=50, zorder=11)
        
        ax.set_xlabel('X (Progress)')
        ax.set_ylabel('Y (Height)')
        ax.set_title(f"Side View (X-Y): {self.current_filter_name}", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-self.lim_xy, self.lim_xy)
        ax.set_ylim(-self.lim_xy, self.lim_xy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter Comparison Tool")
    parser.add_argument("--video", type=str, default="P1_E1T1_45_ALT.mp4", help="Video name (e.g. P1_E1T1_45_ALT.mp4)")
    args = parser.parse_args()

    video_name = args.video
    file = SCRIPT_DIR / "test_res" / f"triplet_signals_{video_name}.npz"
    if os.path.exists(file):
        FilterComparisonTool(file)
    else:
        print(f"File {file} not found.")
