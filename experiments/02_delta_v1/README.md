# 🔬 Technical Decision Report: Exploratory Evaluation of DELTA v1 (Single-Pass & Chunked SVD)

- **Experiment Branch:** `experiment/delta-v1`
- **Evaluated Models:** `DenseTrack3D` (DELTA v1) + `UniDepthV2` (`lpiccinelli/unidepth-v2-vitl14`)
- **Decision Verdict:** **Superseded by DELTAv2 (`DenseTrack3Dv2` + MediaPipe Silhouette)**

---

## 1. Experimental Hypothesis & Architecture

To overcome the 90-frame memory ceiling and bundle-adjustment elasticity of SpatialTrackerV2 + VGGT, **DELTA v1 (`DenseTrack3D`)** coupled with **UniDepthV2** metric depth maps was evaluated in two operational modes (`experiments/delta_test/`):
1. **Anatomical Context Armature (`delta_test.py`):** Replaced diffuse background grids with a structured 1003-point anatomical armature comprising 3 target femur vertices, 200 interior thigh triangle points, 300 pelvis anti-swap anchor points, and 500 torso context points.
2. **Chunked Kabsch-Umeyama Stitching (`delta_test_chunked.py`):** Partitioned long videos into 60-frame chunks with a 25-frame overlap, aligning consecutive chunks via rigid SVD registration ($\mathbf{R} = \mathbf{V}\mathbf{U}^T, \mathbf{T} = \bar{\mathbf{Q}} - \mathbf{R}\bar{\mathbf{P}}$).

---

## 2. Quantitative Findings (6-Video Matched-Window Benchmark)

| Metric | DELTA v1 (Chunked SVD - Matched) | DELTAv2 (Bounding Box - Matched) | DELTAv2 (MediaPipe - Matched) |
| :--- | :---: | :---: | :---: |
| **AccX Pearson $r$** | $+0.482 \pm 0.117$ | $+0.497 \pm 0.218$ | $+0.449 \pm 0.255$ |
| **Pitch Pearson $r$** | $+0.535 \pm 0.335$ | $+0.232 \pm 0.298$ | $+0.430 \pm 0.398$ |
| **Pitch DTW (norm)** | $0.235 \pm 0.053$ | $0.312 \pm 0.099$ | **$0.221 \pm 0.078$** |
| **3D Rigid Deformation (%)** | $46.4\% \pm 31.0\%$ | $24.5\% \pm 7.8\%$ | **$19.6\% \pm 7.6\%$** |
| **RMS Jerk ($\text{m/s}^3$)** | $476.3 \pm 133.8$ | $176.7 \pm 46.7$ | **$30.7 \pm 13.0$** |

---

## 3. Technical Reasons for Upgrading to DELTAv2

1. **Chunk-Boundary Acceleration Discontinuities (RMS Jerk = 476.3 m/s³):**
   Stitching 60-frame windows via Kabsch-Umeyama SVD introduces sub-millimeter positional steps at chunk boundaries that explode under second-order differentiation, increasing RMS Jerk by **15.5×** compared to single-pass DELTAv2 (`476.3` vs `30.7 m/s³`).
2. **Rectangular Armature Background Contamination:**
   Fixed rectangular pelvis/torso boxes spill outside the human body contour during oblique and sagittal stride swings, sampling static wall pixels that distort local correlation volumes.
3. **Cumulative Drift at High Iteration Counts ($N_{\text{iters}}=16$):**
   Running 12–16 iterative updates in DELTA v1 overfits to transient clothing wrinkles. Transitioning to **DELTAv2 (`DenseTrack3Dv2`)** with `coarse_to_fine_dense=True`, $N_{\text{iters}}=6$, and **MediaPipe Pose Landmarker silhouette segmentation** enables single-pass full-video inference (240+ frames) with **19.6%** deformation and **30.7 m/s³** jerk.
