# 02 — Step-by-Step Pipeline Methodology

This document details the complete five-stage workflow executed from raw multi-view video and WitMotion IMU recordings to validated 3D thigh kinematics.

---

## Stage 1: Dataset Scanning, Synchronization & Stratified Cohort Selection

1. **Subject & Trial Discovery (`src/data_io/dataset_scanner.py`):** Scans `data/Gait_Dataset` across 59 subjects, matching RGB camera views (`90°` lateral, `45°` oblique, `0°` frontal, `180°` posterior) with OpenPose 2D JSON detections and WitMotion IMU CSV logs.
2. **Ground-Truth IMU Quality Grading (`src/pipeline/dataset_pipeline.py`):** Filters IMU pitch and linear acceleration signals via a zero-phase 4th-order Butterworth filter and grades each candidate trial (`Optimal`, `Good`, `Imprecise`) based on physiological ROM ($15^\circ$–$65^\circ$) and harmonic cadence regularity.
3. **Balanced 8-Category Stratified Selection (`src/pipeline/cohort_selector.py`):** Selects 40 benchmark videos (5 videos per perspective across 8 directional categories, 39 unique subjects) saved in `data/selected_inference_videos_40.csv`.

---

## Stage 2: 2D Vision Preprocessing & Anatomical Thigh Anchor Extraction

1. **Optimal Reference Frame Selection (`src/preprocessing/vision/frame_selection.py`):** Evaluates horizontal inter-knee separation peaks (`stride_wave = |RKnee_x - LKnee_x|`, corresponding to maximum stride extension where feet are farthest apart and limb crossover occlusion is minimal) and joint detection confidence to select the cleanest reference frame $t_{\text{ref}}$ after `walk_start`.
2. **Canny Edge & RANSAC Boundary Fitting (`src/preprocessing/vision/edge_processing.py`, `line_fitting.py`):** Crops the thigh region of interest (ROI), extracts morphological Canny edges, and fits anterior and posterior thigh boundary lines.
3. **3-Point Anatomical Thigh Triplet (`src/preprocessing/vision/thigh_point_selector.py`):** Constructs a well-conditioned 2D triangle ($P_1$ proximal-anterior, $P_2$ distal-anterior, $P_3$ mid-posterior) inside the thigh silhouette, validated or modified via `KeypointValidatorGUI`.

---

## Stage 3: Dense 3D Point Tracking with DELTAv2 & Body Anchors

1. **Anatomical Thigh + Body Context Sampling (`src/inference/delta_v2_tracker.py`):** Samples `3` anchor points plus `200` dense query points inside the thigh triangle together with global context points: designed for `800` MediaPipe body silhouette points (`1003` total), with a `28×28 = 784`-point bounding-box grid fallback (`987` total). *(Note: in the v1.0.0 40-video benchmark, all 40 runs used the `987`-point grid fallback due to a silent MediaPipe asset/import fallback.)*
2. **Metric 3D Trajectory Unprojection:** Executes UniDepthV2 metric depth estimation coupled with DELTAv2 transformer spatio-temporal attention, unprojecting 2D+depth tracks into metric camera coordinates $(X, Y, Z)$ in meters (`data/inference_results/raw/delta_v2_raw_*.npz`).

---

## Stage 4: 3-Stage Geometric Post-Processing & Rigid Triplet Extraction

1. **Stage 1 — Spatial DBSCAN with Anisotropic Monocular Depth Scaling (`src/postprocessing/geometric_pruner.py`):** Attenuates monocular depth ray-shooting noise by applying $z_{\text{weight}} = 0.25$ before running frame-by-frame 3D DBSCAN clustering (`eps=0.12`).
2. **Stage 2 — Kinematic Velocity & Acceleration MAD Consensus:** Computes frame-to-frame velocity and acceleration profiles with $z_{\text{weight}}=0.25$ and removes points whose median residual exceeds $2.5 \times \text{MAD}$ (guarded by a physical floor of $0.015\text{ m/frame}$).
3. **Stage 3 — Non-Degenerate Rigid Triplet Extraction:** Evaluates candidate triplets inside the surviving thigh cluster, enforcing a minimum triangle altitude $h_{\min} \ge 3.5\text{ cm}$ and maximum aspect ratio $\text{AR} \le 8.0$ to prevent normal-vector twist while minimizing relative edge-length standard deviation (`3D Rigid Deformation = 28.4% ± 9.9%`).

---

## Stage 5: 3D Kinematic Alignment & Multi-Tier Benchmark Evaluation

1. **Kabsch SVD 3D Rigid-Body Alignment (`src/analysis/kinematic_alignment.py`):** Aligns the 3D vision acceleration vector (second temporal derivative of the centroid minus gravity) with the 3D IMU accelerometer vector in the anatomical thigh frame (`use_pca_basis=False`) via optimal $\text{SO}(3)$ SVD rotation and single-pass zero-mean velocity integration.
2. **Multi-Metric Evaluation (`src/benchmark/evaluator.py`):** Computes waveform similarity (Pearson $r$, signed Peak Cross-Correlation with $\pm 0.5\text{ s}$ lag cap), magnitude accuracy (nRMSE, ROM Error), spectral step cadence error, and 3D geometric stability (DTW, Rigid Deformation, RMS Jerk, Identity Swaps, and Axial Depth Artifacts).
