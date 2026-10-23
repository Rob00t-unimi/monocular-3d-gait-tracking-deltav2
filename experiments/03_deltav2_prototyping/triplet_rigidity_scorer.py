"""
Refined Triplet Selection Script for DELTA - Version 4.1
Logic:
1) PCA Clustering on 3D velocities to isolate leg points.
2) 3D Non-Collinearity Check: Area 3D must be > 1% of original to ensure a valid plane.
3) Absolute 2D Floor: Area must be > 10% of ORIGINAL triplet area at Ref Frame.
4) Scoring System: Maximize Area coverage and Stability in both 2D and 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from pathlib import Path
from scipy import signal
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import imageio
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Configurazione Percorsi
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = (SCRIPT_DIR / "../../").resolve()

# --- UTILITIES ---

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0: normal_cutoff = 0.99
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def load_dynamic_cutoffs():
    cache_path = PROJECT_ROOT / "data" / "gt_cache" / "cutoff_freqs.json"
    lowcut, highcut = 0.5, 3.0
    try:
        with open(cache_path, "r") as f:
            data = json.load(f); lowcut, highcut = data["lowcut"], data["highcut"]
    except: pass
    return lowcut, highcut

def calc_area_2d(p1, p2, p3):
    """Compute 2D triangle area (vectorized)."""
    return 0.5 * np.abs(p1[..., 0]*(p2[..., 1]-p3[..., 1]) +
                        p2[..., 0]*(p3[..., 1]-p1[..., 1]) +
                        p3[..., 0]*(p1[..., 1]-p2[..., 1]))

def calc_area_3d(p1, p2, p3):
    """Compute 3D triangle area using the cross product."""
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross, axis=-1)

# --- CLUSTERING LOGIC ---

def apply_adaptive_pca_dbscan(points_3d, n_components=5, initial_eps=10):
    T, N, D = points_3d.shape
    velocities = np.diff(points_3d, axis=0)
    vel_features = velocities.transpose(1, 0, 2).reshape(N, -1)

    scaler = StandardScaler()
    vel_scaled = scaler.fit_transform(vel_features)

    pca = PCA(n_components=n_components)
    vel_pca = pca.fit_transform(vel_scaled)

    eps = initial_eps
    for attempt in range(15):
        clustering = DBSCAN(eps=eps, min_samples=max(3, N // 50)).fit(vel_pca)
        labels = clustering.labels_
        u, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(u) > 0:
            main_label = u[np.argmax(counts)]
            indices = np.where(labels == main_label)[0]
            if len(indices) >= 4:
                return indices
        eps *= 1.3
    return np.arange(N)

# --- OPTIMAL SELECTION ---

def select_optimal_triplet(points_2d, points_3d, survived_indices, ref_idx, original_indices=[0, 1, 2]):
    if len(survived_indices) < 3: return None

    # 1. Benchmark Area 2D
    pts_orig_ref_2d = points_2d[ref_idx, original_indices]
    area_orig_ref_2d = calc_area_2d(pts_orig_ref_2d[0], pts_orig_ref_2d[1], pts_orig_ref_2d[2])

    # 2. 3D Area Benchmark (Physical non-collinearity)
    pts_orig_ref_3d = points_3d[ref_idx, original_indices]
    area_orig_ref_3d = calc_area_3d(pts_orig_ref_3d[0], pts_orig_ref_3d[1], pts_orig_ref_3d[2])

    # Safety floor for 3D non-collinearity
    abs_min_3d_floor = 0.05 * area_orig_ref_3d # Must retain at least 5% of the initial 3D surface area

    # thresholds = [0.05, 0.02, 0.01, 0.005] # Parametri 2D disabilitati
    cand_indices = survived_indices
    # if len(cand_indices) > 100:
    #     print(f"Downsampling candidates from {len(cand_indices)} to 100...")
    #     top = [i for i in original_indices if i in cand_indices]
    #     rest = [i for i in cand_indices if i not in top]
    #     step = len(rest) // (100 - len(top))
    #     cand_indices = np.array(top + rest[::max(1, step)])

    pts_seq_2d = points_2d[:, cand_indices, :2]
    pts_seq_3d = points_3d[:, cand_indices, :3]
    combos = list(combinations(range(len(cand_indices)), 3))
    
    print(f"Searching for optimal triplet (Floor 3D: {abs_min_3d_floor:.2f})...")
    
    best_score = -1
    best_triplet = None
    
    for c in combos:
        # 1. Check 2D Floor (DISABILITATO)
        # tp1_2d, tp2_2d, tp3_2d = pts_seq_2d[:, c[0]], pts_seq_2d[:, c[1]], pts_seq_2d[:, c[2]]
        # areas_2d = calc_area_2d(tp1_2d, tp2_2d, tp3_2d)
        
        # 2. Check 3D Non-Collinearity (Must define a plane)
        tp1_3d, tp2_3d, tp3_3d = pts_seq_3d[:, c[0]], pts_seq_3d[:, c[1]], pts_seq_3d[:, c[2]]
        areas_3d = calc_area_3d(tp1_3d, tp2_3d, tp3_3d)
        if np.min(areas_3d) < abs_min_3d_floor: continue
            
        # 3. SCORING (Use areas and stability to select the optimal triplet)
        tp1_2d, tp2_2d, tp3_2d = pts_seq_2d[:, c[0]], pts_seq_2d[:, c[1]], pts_seq_2d[:, c[2]]
        areas_2d = calc_area_2d(tp1_2d, tp2_2d, tp3_2d)
        
        area_ref_2d = areas_2d[ref_idx]
        min_area_2d = np.min(areas_2d)
        mean_area_2d = np.mean(areas_2d)
        
        stability_2d = mean_area_2d / (np.std(areas_2d) + 1e-6)
        stability_3d = np.mean(areas_3d) / (np.std(areas_3d) + 1e-6)
        
        # Point scoring rewards wide and rigid triangles
        score = np.log1p(area_ref_2d) * np.log1p(min_area_2d) * stability_2d * stability_3d
        
        if score > best_score:
            best_score = score
            best_triplet = (int(cand_indices[c[0]]), int(cand_indices[c[1]]), int(cand_indices[c[2]]))

    if best_triplet:
        print(f"Success! Optimal triplet found via scoring system.")
        return best_triplet

    print("Warning: Falling back to first available combination.")
    return (int(cand_indices[0]), int(cand_indices[1]), int(cand_indices[2]))

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Optimal Triplet Selection System")
    parser.add_argument("--video", type=str, default="P1_E1T1_45_ALT.mp4")
    parser.add_argument("--gif", action="store_true", help="Save a GIF of the result")
    args = parser.parse_args()

    input_file = SCRIPT_DIR / "test_res" / f"delta_v2_results_{args.video}.npz"
    if not input_file.exists():
        print(f"Error: {input_file} not found."); return

    data = np.load(input_file)
    frames, track2d, track3d = data['frames'], data['track2d'], data['track3d_world']
    ref_idx_abs, start_idx_abs = int(data.get('ref_frame_idx', 0)), int(data.get('start_idx', 0))
    ref_idx_rel = ref_idx_abs - start_idx_abs
    scale_x, scale_y = data['scales']

    total_pts = track2d.shape[1]
    num_leg_pts = 203

    t2d, t3d = track2d.copy(), track3d.copy()

    # 1) Filtering
    print(f"Filtering {total_pts} points...")
    _, highcut = load_dynamic_cutoffs()
    for i in range(total_pts):
        for d in range(2): t2d[:, i, d] = butter_lowpass(t2d[:, i, d], highcut, 30.0)
        for d in range(3): t3d[:, i, d] = butter_lowpass(t3d[:, i, d], highcut, 30.0)

    # 2) PCA Cluster
    print("Extracting leg cluster via PCA...")
    leg_indices = np.arange(num_leg_pts)
    survived_indices_rel = apply_adaptive_pca_dbscan(t3d[:, leg_indices, :])
    survived_indices = leg_indices[survived_indices_rel].astype(int)

    # 3) Optimal Selection
    best_triplet = select_optimal_triplet(t2d, t3d, survived_indices, ref_idx_rel)

    if best_triplet:
        out_npz = SCRIPT_DIR / "test_res" / f"triplet_signals_{args.video}_pca_exhaustive.npz"
        np.savez(out_npz, track3d=t3d[:, best_triplet, :], track2d=t2d[:, best_triplet, :],
                 indices=np.array(best_triplet), survived_track3d=t3d[:, survived_indices, :],
                 survived_indices=survived_indices, video_name=args.video, mode="point_system_delta_v4.1",
                 ref_frame_idx=ref_idx_abs, start_idx=start_idx_abs, fps=30.0)
        print(f"Saved to {out_npz}")

    # --- GIF GENERATION ---
    if args.gif:
        out_gif = SCRIPT_DIR / "test_res" / f"triplet_viz_{args.video}.gif"
        print(f"Generating GIF: {out_gif}")
        
        # Compute bounding box for cropping (based on leg points only)
        all_pts_2d = t2d[:, survived_indices, :2] / [scale_x, scale_y]
        x_min, x_max = np.min(all_pts_2d[..., 0]), np.max(all_pts_2d[..., 0])
        y_min, y_max = np.min(all_pts_2d[..., 1]), np.max(all_pts_2d[..., 1])
        
        pad = 50
        H, W = frames.shape[1:3]
        x_min, x_max = max(0, int(x_min - pad)), min(W, int(x_max + pad))
        y_min, y_max = max(0, int(y_min - pad)), min(H, int(y_max + pad))
        
        crop_w, crop_h = x_max - x_min, y_max - y_min
        aspect_ratio = crop_w / crop_h
        base_w = 8
        fig_gif = plt.figure(figsize=(base_w, base_w / aspect_ratio))
        canvas_gif = FigureCanvasAgg(fig_gif)
        ax_gif = fig_gif.add_axes([0, 0, 1, 1])
        
        gif_frames = []
        step = 1 if len(frames) < 250 else 2
        
        # Indices for point groups
        grid_indices = np.arange(num_leg_pts, total_pts).astype(int)
        rejected_indices = np.array([i for i in leg_indices if i not in survived_indices]).astype(int)
        orig_idx = [0, 1, 2]
        
        for t in tqdm(range(0, len(frames), step), desc="Generating GIF Frames"):
            ax_gif.clear()
            ax_gif.imshow(frames[t])
            
            # 1. Background Grid (Gray)
            ax_gif.scatter(t2d[t, grid_indices, 0]/scale_x, t2d[t, grid_indices, 1]/scale_y,
                           c='gray', s=3, alpha=0.1, edgecolors='none')
            
            # 2. Rejected Leg Points (Red)
            if len(rejected_indices) > 0:
                ax_gif.scatter(t2d[t, rejected_indices, 0]/scale_x, t2d[t, rejected_indices, 1]/scale_y, 
                               c='red', s=5, alpha=0.3, edgecolors='none')
            
            # 3. Leg Cluster (Cyan)
            ax_gif.scatter(t2d[t, survived_indices, 0]/scale_x, t2d[t, survived_indices, 1]/scale_y, 
                           c='cyan', s=15, edgecolors='black', linewidth=0.3, alpha=0.8)
            
            # 4. Original Query (Yellow X)
            ax_gif.scatter(t2d[t, orig_idx, 0]/scale_x, t2d[t, orig_idx, 1]/scale_y, 
                           c='yellow', s=30, edgecolors='black', marker='X', linewidth=0.5)
            
            # 5. Optimal Triplet (Magenta)
            if best_triplet:
                o_xy = t2d[t, list(best_triplet), :2] / [scale_x, scale_y]
                ax_gif.plot(o_xy[:, 0], o_xy[:, 1], 'mo', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
                ax_gif.plot(np.append(o_xy[:, 0], o_xy[0,0]), np.append(o_xy[:, 1], o_xy[0,1]), 'm-', linewidth=1.5)
            
            ax_gif.set_xlim(x_min, x_max); ax_gif.set_ylim(y_max, y_min)
            ax_gif.axis('off')
            
            canvas_gif.draw()
            frame_img = np.array(canvas_gif.buffer_rgba())[:, :, :3]
            gif_frames.append(frame_img)
            
        imageio.mimsave(out_gif, gif_frames, fps=15, loop=0)
        plt.close(fig_gif)
        plt.ion()
        print(f"GIF saved successfully.")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    img = ax.imshow(frames[ref_idx_rel])

    # 1. Griglia di Sfondo (Grigio)
    grid_indices = np.arange(num_leg_pts, total_pts).astype(int)
    pts_grid = ax.scatter(t2d[ref_idx_rel, grid_indices, 0]/scale_x, t2d[ref_idx_rel, grid_indices, 1]/scale_y,
                          c='gray', s=5, alpha=0.2, label='Background Grid')

    # 2. Discarded leg points (Red)
    rejected_indices = np.array([i for i in leg_indices if i not in survived_indices]).astype(int)
    pts_rej = None
    if len(rejected_indices) > 0:
        pts_rej = ax.scatter(t2d[ref_idx_rel, rejected_indices, 0]/scale_x, t2d[ref_idx_rel, rejected_indices, 1]/scale_y,
                             c='red', s=10, alpha=0.4, label='Rejected Leg Points')

    # 3. Cluster points (Cyan)
    pts_all = ax.scatter(t2d[ref_idx_rel, survived_indices, 0]/scale_x, t2d[ref_idx_rel, survived_indices, 1]/scale_y,
                         c='cyan', s=20, edgecolors='black', linewidth=0.5, label='Leg Cluster')

    # 4. Query originale (Yellow)
    orig_idx = [0, 1, 2]
    pts_orig = ax.scatter(t2d[ref_idx_rel, orig_idx, 0]/scale_x, t2d[ref_idx_rel, orig_idx, 1]/scale_y,
                          c='yellow', s=40, edgecolors='black', marker='X', label='Original Query')

    tri_pts, tri_line = None, None
    if best_triplet:
        o_xy = t2d[ref_idx_rel, list(best_triplet), :2] / [scale_x, scale_y]
        tri_pts, = ax.plot(o_xy[:, 0], o_xy[:, 1], 'mo', markersize=10, markeredgecolor='white', label='Optimal Triplet')
        tri_line, = ax.plot(np.append(o_xy[:, 0], o_xy[0,0]), np.append(o_xy[:, 1], o_xy[0,1]), 'm-', linewidth=2)

    ax.legend(); curr_t = [ref_idx_rel]
    def update(t):
        img.set_data(frames[t])

        # Update positions for ALL point groups
        pts_grid.set_offsets(t2d[t, grid_indices, :2] / [scale_x, scale_y])
        if pts_rej: pts_rej.set_offsets(t2d[t, rejected_indices, :2] / [scale_x, scale_y])
        pts_all.set_offsets(t2d[t, survived_indices, :2] / [scale_x, scale_y])
        pts_orig.set_offsets(t2d[t, orig_idx, :2] / [scale_x, scale_y])

        if best_triplet:
            o_xy = t2d[t, list(best_triplet), :2] / [scale_x, scale_y]
            tri_pts.set_data(o_xy[:, 0], o_xy[:, 1])
            tri_line.set_data(np.append(o_xy[:, 0], o_xy[0,0]), np.append(o_xy[:, 1], o_xy[0,1]))

        ax.set_title(f"Point System v4.1 | Frame {t} | Video: {args.video}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right': curr_t[0] = (curr_t[0] + 1) % len(frames)
        elif event.key == 'left': curr_t[0] = (curr_t[0] - 1) % len(frames)
        update(curr_t[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    main()
