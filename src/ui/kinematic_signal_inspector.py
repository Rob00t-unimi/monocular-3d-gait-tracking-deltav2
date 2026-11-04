import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.preprocessing.signals.signal_orchestrator import SignalOrchestrator
from src.data_io.dataset_scanner import DatasetScanner

class UnifiedSignalBrowser:
    def __init__(self, all_signals, invalid_set, lowcut, highcut):
        self.file_list = all_signals
        self.invalid_set = invalid_set
        self.lowcut = lowcut
        self.highcut = highcut
        self.index = 0
        self.num_files = len(all_signals)
        self.orchestrator = SignalOrchestrator()

        # Setup 3x2 grid: accX, Pitch, accY, Roll, accZ, Yaw
        self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
        self.axes = self.axes.flatten() 

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

        print(f"Unified Signal Inspector started.")
        print(f"Controls: [Right Arrow] Next | [Left Arrow] Previous | [Q] Quit")

    def load_and_process(self):
        csv_path = Path(self.file_list[self.index])
        df_raw = self.orchestrator.loader.load(csv_path)
        if df_raw is None:
            return None, None

        # Unwrap angles
        df_unwrapped = self.orchestrator.estimator._unwrap_angles(df_raw.copy())
        # Filter signals
        df_filtered = self.orchestrator.processor.process(df_unwrapped, self.lowcut, self.highcut)
        
        return df_unwrapped, df_filtered

    def update_plot(self):
        df_raw, df_filtered = self.load_and_process()
        if df_raw is None:
            return

        csv_path_str = self.file_list[self.index]
        filename = Path(csv_path_str).name
        is_invalid = csv_path_str in self.invalid_set
        status_str = "INVALID" if is_invalid else "VALID"
        status_color = "darkred" if is_invalid else "darkgreen"

        self.fig.suptitle(f"[{status_str}] Signal [{self.index + 1}/{self.num_files}]: {filename}",
                          fontsize=14, fontweight='bold', color=status_color)

        # Map signals to specific axes (matching the notebook example order)
        plot_map = [
            ('accX', 0, 'tab:blue'), ('Pitch', 1, 'tab:orange'),
            ('accY', 2, 'tab:green'), ('Roll', 3, 'tab:red'),
            ('accZ', 4, 'tab:purple'), ('Yaw', 5, 'tab:brown')
        ]

        for col, ax_idx, color in plot_map:
            ax = self.axes[ax_idx]
            ax.clear()

            if col in df_raw.columns and col in df_filtered.columns:
                # Plot raw/unwrapped in gray
                ax.plot(df_raw[col].values, color='gray', alpha=0.3, label='Raw/Unwrapped')
                # Plot filtered in color
                ax.plot(df_filtered[col].values, color=color, linewidth=1.5, label='Filtered')
                
                ax.set_title(f"Signal: {col}", fontsize=10)
                ax.legend(loc='upper right', fontsize=8)

            ax.grid(True, linestyle='--', alpha=0.5)
            if ax_idx >= 4: ax.set_xlabel("Samples")
            if ax_idx % 2 == 0: ax.set_ylabel("Value")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % self.num_files
            self.update_plot()
        elif event.key == 'left':
            self.index = (self.index - 1) % self.num_files
            self.update_plot()
        elif event.key == 'q':
            plt.close()

def inspect_signals(mode="all", data_dir="data"):
    orchestrator = SignalOrchestrator()
    
    # Load invalid set
    invalid_set = set()
    if orchestrator.invalid_cache_path.exists():
        with open(orchestrator.invalid_cache_path, "r") as f:
            invalid_set = set(json.load(f))

    # Load cutoff frequencies
    lowcut, highcut = 0.5, 10.0 # defaults
    if orchestrator.cutoff_cache_path.exists():
        with open(orchestrator.cutoff_cache_path, "r") as f:
            data = json.load(f)
            lowcut, highcut = data["lowcut"], data["highcut"]

    # Get all CSVs
    scanner = DatasetScanner(data_dir)
    all_csvs = scanner.get_csvs()
    
    invalid_list = sorted([c for c in all_csvs if c in invalid_set])
    valid_list = sorted([c for c in all_csvs if c not in invalid_set])

    if mode == "invalid":
        signals_to_show = invalid_list
    elif mode == "valid":
        signals_to_show = valid_list
    else: # all
        signals_to_show = invalid_list + valid_list

    if not signals_to_show:
        print(f"No signals to display for mode: {mode}")
        return

    browser = UnifiedSignalBrowser(signals_to_show, invalid_set, lowcut, highcut)
    plt.show()
