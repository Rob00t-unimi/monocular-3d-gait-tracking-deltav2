"""
Refined Triplet Selection Script - Version 3.2 (Exhaustive PCA)
Workflow: 
1) Frequency Filtering 
2) PCA on Velocities (Rhythm detection)
3) Adaptive DBSCAN 
4) Exhaustive Search: checks ALL combinations to find the largest stable triangle.
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

# --- CLUSTERING LOGIC ---

def apply_adaptive_pca_dbscan(points_3d, n_components=5, initial_eps=10):
    T, N, D = points_3d.shape
    # Velocity profile to ignore absolute scale offsets
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
                print(f"Cluster found (eps={eps:.2f}): {len(indices)} points.")
                return indices
        eps *= 1.3
    return np.arange(N)

# --- EXHAUSTIVE SELECTION ---

def select_exhaustive_triplet(points_2d, indices, ref_idx, min_area_ratio=0.10):
    """
    Evaluate ALL candidate point combinations inside the cluster.
    Find the largest triangle that never collapses across the video.
    """
    if len(indices) < 3: return None
    
    # Cap cluster size at 120 points to prevent combinatorial explosion
    # (120^3 / 6 = ~280k triplette, gestibile in pochi secondi)
    if len(indices) > 120:
        print(f"Limiting search to 120 most confident points (from {len(indices)})")
        # Note: select leading query points ranked by stability
        indices = indices[:120]

    # Pre-compute 2D pixel coordinates
    pts_seq = points_2d[:, indices, :2]
    pts_ref = pts_seq[ref_idx]
    
    print(f"Generating all combinations for {len(indices)} points...")
    combos = list(combinations(range(len(indices)), 3))
    
    # Compute areas in reference frame (Vectorized)
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    tri_pts = pts_ref[np.array(combos)] # [N_combos, 3, 2]
    p1, p2, p3 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    areas_ref = 0.5 * np.abs(p1[:, 0]*(p2[:, 1]-p3[:, 1]) + p2[:, 0]*(p3[:, 1]-p1[:, 1]) + p3[:, 0]*(p1[:, 1]-p2[:, 1]))
    
    # Ordiniamo le combinazioni per area decrescente
    sorted_idx = np.argsort(areas_ref)[::-1]
    
    print(f"Checking stability for {len(combos)} triplet candidates...")
    for rank, idx in enumerate(sorted_idx):
        c = combos[idx]
        area_ref = areas_ref[idx]
        if area_ref < 1.0: continue # Salta triangoli degeneri
        
        # Extract 2D triplet trajectory
        tp1, tp2, tp3 = pts_seq[:, c[0]], pts_seq[:, c[1]], pts_seq[:, c[2]]
        
        # Formula corretta dell'area nel tempo: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        areas_t = 0.5 * np.abs(tp1[:, 0]*(tp2[:, 1]-tp3[:, 1]) + tp2[:, 0]*(tp3[:, 1]-tp1[:, 1]) + tp3[:, 0]*(tp1[:, 1]-tp2[:, 1]))
        
        # Test di persistenza: l'area non deve mai scendere sotto la soglia
        if np.min(areas_t) > (area_ref * min_area_ratio):
            print(f"Found Optimal Stable Triplet (Rank {rank+1})!")
            print(f"Ref Area: {area_ref:.1f} px, Min Area in video: {np.min(areas_t):.1f} px")
            return (int(indices[c[0]]), int(indices[c[1]]), int(indices[c[2]]))
            
    print("Warning: No stable triplet found in exhaustive search. Falling back to largest ref area.")
    return (int(indices[combos[sorted_idx[0]][0]]), int(indices[combos[sorted_idx[0]][1]]), int(indices[combos[sorted_idx[0]][2]]))

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Exhaustive PCA Triplet Selection")
    parser.add_argument("--video", type=str, default="P1_E1T1_45_ALT.mp4")
    parser.add_argument("--gif", action="store_true", help="Save a GIF of the result")
    args = parser.parse_args()

    input_file = SCRIPT_DIR / "test_res" / f"tracking_results_unified_{args.video}.npz"
    if not input_file.exists():
        print(f"Error: {input_file} not found."); return

    data = np.load(input_file)
    frames, track2d, track3d = data['frames'], data['track2d'], data['track3d_world']
    ref_idx_abs, start_idx_abs = int(data.get('ref_frame_idx', 0)), int(data.get('start_idx', 0))
    ref_idx_rel = ref_idx_abs - start_idx_abs
    scale_x, scale_y = data['scales']
    
    num_pts = 203
    t2d, t3d = track2d[:, :num_pts].copy(), track3d[:, :num_pts].copy()
    
    # 1) Filtering
    print("Filtering noise...")
    _, highcut = load_dynamic_cutoffs()
    for i in range(num_pts):
        for d in range(2): t2d[:, i, d] = butter_lowpass(t2d[:, i, d], highcut, 30.0)
        for d in range(3): t3d[:, i, d] = butter_lowpass(t3d[:, i, d], highcut, 30.0)
        
    # 2) PCA Cluster
    print("Extracting movement cluster...")
    survived_indices = apply_adaptive_pca_dbscan(t3d, n_components=10, initial_eps=10)
    
    # 3) Exhaustive Search
    best_triplet = select_exhaustive_triplet(t2d, survived_indices, ref_idx_rel)
    
    # Save Results
    if best_triplet:
        out_npz = SCRIPT_DIR / "test_res" / f"triplet_signals_{args.video}_pca_exhaustive.npz"
        np.savez(out_npz, track3d=t3d[:, best_triplet, :], track2d=t2d[:, best_triplet, :],
                 indices=np.array(best_triplet), survived_track3d=t3d[:, survived_indices, :],
                 survived_indices=survived_indices, video_name=args.video, mode="pca_exhaustive_v3.2",
                 ref_frame_idx=ref_idx_abs, start_idx=start_idx_abs, fps=30.0)
        print(f"Saved NPZ to {out_npz}")

    # --- GIF GENERATION ---
    if args.gif:
        out_gif = SCRIPT_DIR / "test_res" / f"triplet_viz_{args.video}.gif"
        print(f"Generating GIF: {out_gif}")
        
        # Compute global crop bounding box from cluster points
        all_pts_2d = t2d[:, survived_indices, :2] / [scale_x, scale_y]
        x_min, x_max = np.min(all_pts_2d[..., 0]), np.max(all_pts_2d[..., 0])
        y_min, y_max = np.min(all_pts_2d[..., 1]), np.max(all_pts_2d[..., 1])
        
        # Aggiunta padding (50px)
        H, W = frames.shape[1:3]
        pad = 50
        x_min, x_max = max(0, int(x_min - pad)), min(W, int(x_max + pad))
        y_min, y_max = max(0, int(y_min - pad)), min(H, int(y_max + pad))
        
        # Compute crop dimensions to configure figure aspect ratio
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        aspect_ratio = crop_w / crop_h
        
        # Impostiamo una dimensione base (es. 8 pollici di larghezza) e calcoliamo l'altezza proporzionale
        base_w = 8
        fig_h = base_w / aspect_ratio
        
        # Creiamo una figura esplicitamente con il backend Agg per evitare conflitti con il backend interattivo
        fig_gif = plt.figure(figsize=(base_w, fig_h))
        canvas_gif = FigureCanvasAgg(fig_gif)
        # Use add_axes([0,0,1,1]) so the plot fills the entire figure without white margins
        ax_gif = fig_gif.add_axes([0, 0, 1, 1])
        
        gif_frames = []
        step = 1 if len(frames) < 250 else 2
        
        # Per la GIF, definiamo gli indici scartati e la query originale
        rejected_indices = np.array([i for i in range(num_pts) if i not in survived_indices])
        orig_idx = [0, 1, 2]
        
        for t in tqdm(range(0, len(frames), step), desc="Generating GIF Frames"):
            ax_gif.clear()
            ax_gif.imshow(frames[t])
            
            # 1. Plot Rejected Points (Red)
            if len(rejected_indices) > 0:
                ax_gif.scatter(t2d[t, rejected_indices, 0]/scale_x, t2d[t, rejected_indices, 1]/scale_y, 
                               c='red', s=5, alpha=0.3, edgecolors='none')
            
            # 2. Plot Cluster Points (Cyan)
            ax_gif.scatter(t2d[t, survived_indices, 0]/scale_x, t2d[t, survived_indices, 1]/scale_y, 
                           c='cyan', s=15, edgecolors='black', linewidth=0.3, alpha=0.8)
            
            # 3. Plot Original Query (Yellow X)
            ax_gif.scatter(t2d[t, orig_idx, 0]/scale_x, t2d[t, orig_idx, 1]/scale_y, 
                           c='yellow', s=30, edgecolors='black', marker='X', linewidth=0.5)
            
            # 4. Plot Optimal Triplet (Magenta)
            if best_triplet:
                o_xy = t2d[t, list(best_triplet), :2] / [scale_x, scale_y]
                ax_gif.plot(o_xy[:, 0], o_xy[:, 1], 'mo', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
                ax_gif.plot(np.append(o_xy[:, 0], o_xy[0,0]), np.append(o_xy[:, 1], o_xy[0,1]), 'm-', linewidth=1.5)
            
            ax_gif.set_xlim(x_min, x_max); ax_gif.set_ylim(y_max, y_min)
            ax_gif.axis('off')
            
            # Rendering forzato sulla canvas Agg
            canvas_gif.draw()
            # Metodo compatibile con Matplotlib 3.8+ (buffer_rgba)
            frame_img = np.array(canvas_gif.buffer_rgba())[:, :, :3] # Prendi solo RGB, scarta Alpha
            gif_frames.append(frame_img)
            
        imageio.mimsave(out_gif, gif_frames, fps=15, loop=0)
        plt.close(fig_gif)
        
        plt.ion()
        print(f"GIF saved successfully.")

    # Interactive Visualization (Invariata)
    fig, ax = plt.subplots(figsize=(12, 10))
    img = ax.imshow(frames[ref_idx_rel])
    
    # 1. Discarded points (Red)
    rejected_indices = np.array([i for i in range(num_pts) if i not in survived_indices])
    pts_rej = ax.scatter(t2d[ref_idx_rel, rejected_indices, 0]/scale_x, t2d[ref_idx_rel, rejected_indices, 1]/scale_y, 
                         c='red', s=10, alpha=0.4, label='Rejected Points')
    
    # 2. Cluster points (Cyan)
    pts_all = ax.scatter(t2d[ref_idx_rel, survived_indices, 0]/scale_x, t2d[ref_idx_rel, survived_indices, 1]/scale_y, 
                         c='cyan', s=20, edgecolors='black', linewidth=0.5, label='Leg Points')
    
    # 3. Query originale (Yellow)
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
        pts_rej.set_offsets(t2d[t, rejected_indices, :2] / [scale_x, scale_y])
        pts_all.set_offsets(t2d[t, survived_indices, :2] / [scale_x, scale_y])
        pts_orig.set_offsets(t2d[t, orig_idx, :2] / [scale_x, scale_y])
        
        if best_triplet:
            o_xy = t2d[t, list(best_triplet), :2] / [scale_x, scale_y]
            tri_pts.set_data(o_xy[:, 0], o_xy[:, 1])
            tri_line.set_data(np.append(o_xy[:, 0], o_xy[0,0]), np.append(o_xy[:, 1], o_xy[0,1]))
        ax.set_title(f"Exhaustive Search v3.2 | Frame {t} | Video: {args.video}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right': curr_t[0] = (curr_t[0] + 1) % len(frames)
        elif event.key == 'left': curr_t[0] = (curr_t[0] - 1) % len(frames)
        elif event.key == 'r': curr_t[0] = ref_idx_rel
        update(curr_t[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    main()
