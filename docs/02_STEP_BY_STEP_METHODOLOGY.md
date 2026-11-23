# 02 — Step-by-Step Pipeline Methodology

This document details the complete five-stage workflow executed from raw multi-view video and WitMotion IMU recordings to validated 3D thigh kinematics, aligned with `src/`, `configs/default_config.yaml`, and the Technical Report.

---

## Stage 1: Dataset Scanning, Synchronization & Stratified Cohort Selection

1. **Subject & Trial Discovery (`src/data_io/dataset_scanner.py`):** Scans `data/Gait_Dataset` (Gait360: 64 participants, 2,942 videos at 30 Hz), matching RGB camera views (`90°` lateral, `45°` oblique, `0°` frontal, `180°` posterior) with OpenPose `BODY_25` 2D keypoint JSONs and synchronized 9-DoF WitMotion IMU CSV logs (nominal 83.33 Hz).
2. **Adaptive Welch PSD Profiling & IMU Quality Grading (`src/pipeline/dataset_pipeline.py`):** Estimates the dataset-wide physiological gait passband (`0.50–2.93 Hz`, default highcut `2.9296875 Hz`) via Welch PSD, filters IMU pitch and linear acceleration signals via a zero-phase 4th-order Butterworth bandpass filter, and grades each candidate trial (`Optimal`, `Good`, `Imprecise`) based on physiological ROM and harmonic cadence regularity.
3. **Balanced 8-Category Stratified Selection (`src/pipeline/cohort_selector.py`):** Deterministically selects **40 benchmark videos** (5 videos per perspective across 8 directional categories, **40 unique subjects**, all with `sensor_position == "leg"`) saved in `data/selected_inference_videos_40.csv`.

---

## Stage 2: 2D Vision Preprocessing & Anatomical Thigh Anchor Extraction

1. **Optimal Reference Frame Selection (`src/preprocessing/vision/frame_selection.py`):** Evaluates horizontal inter-knee separation peaks (`stride_wave = |RKnee_x - LKnee_x|`, corresponding to maximum stride extension where contralateral occlusion is minimal), femur segment non-intersection, and joint detection confidence to select the cleanest reference frame $t_{\text{ref}}$ after `walk_start`.
2. **Directional Sobel Edge Detection & Huber M-Estimator Fitting (`src/preprocessing/vision/edge_processing.py`, `line_fitting.py`):** Constructs the local femur orthonormal basis $(\mathbf{u}, \mathbf{u}_\perp)$ in the lower-third thigh fascia ($0 \le d_\parallel \le 0.35\,L_{\text{femur}}$, $|d_\perp| \le 0.40\,L_{\text{femur}}$), extracts directional Sobel gradients (`ksize=3`, Gaussian blur `7×7`, threshold `15`), and fits robust anterior/posterior thigh boundary lines via Huber regression (`max_angle_deg = 20°`).
3. **Inward Safety Erosion & 3-Point Anatomical Thigh Triplet (`src/preprocessing/vision/thigh_point_selector.py`):** Erodes the segmented thigh polygon inward by `2.0 px` (`safety_margin_px = 2.0`) to avoid background depth cliffs, and places a well-conditioned 2D anchor triangle $(P_1, P_2, P_3)$ inside the thigh silhouette, inspectable via `KeypointValidatorGUI`.

---

## Stage 3: Dense 3D Point Tracking with DELTAv2 & Context Anchors

1. **Thigh + Global Context Query Sampling (`src/inference/delta_v2_tracker.py`):** Samples `3` anatomical anchor vertices plus `200` dense interior thigh points (`203` thigh queries, indices `0..202`), paired with global context queries in two configurable modes:
   - **`bbox_grid` (Primary benchmark, `results/`):** `28×28 = 784` uniform bounding-box grid points (`987` total queries). Static background grid points anchor global scene geometry during sagittal/oblique crossovers.
   - **`mediapipe` (Silhouette ablation, `results_mediapipe/`):** `800` MediaPipe `PoseLandmarker` body-silhouette points (`1003` total queries) with confidence cascade (`0.5 -> 0.2 -> 0.1 -> 0.05 -> 0.02 -> 0.01 -> 0.0`) and fail-fast validation prior to depth inference.
2. **Metric 3D Trajectory Unprojection:** Executes batched **UniDepthV2** (`lpiccinelli/unidepth-v2-vitl14`, batch size `8`) metric depth estimation coupled with **DELTAv2** (`coarse_to_fine_dense=True`, `window_len=16`, `n_iters=6`), tracking bidirectionally in a single pass and unprojecting tracks into metric camera coordinates $(X, Y, Z)$ in meters.

---

## Stage 4: 3-Stage Geometric Post-Processing & Rigid Triplet Extraction

Implemented in `src/postprocessing/geometric_pruner.py` (`GeometricPruner` v4.1) on the `203` thigh trajectories:
1. **Stage 0 — Zero-Phase Butterworth Temporal Smoothing:** Applies a 4th-order zero-phase low-pass filter (`filtfilt`) at the dataset physiological cutoff ($f_c \approx 2.93\text{ Hz}$).
2. **Stage 1 — 2D Convex Hull Area Persistence Pruning:** Iteratively discards drifting boundary outliers whenever the 2D thigh cluster convex hull expands beyond `max_area_ratio = 5.0` times its reference area, enforcing **100% temporal persistence** across all $T$ frames.
3. **Stage 2 — Kinematic Velocity PCA + Adaptive DBSCAN (optional/ablation) & Coherent Cluster Isolation:** Projects 3D velocity profiles into a 5-component PCA embedding and applies adaptive DBSCAN (`initial_eps = 0.5`, multiplier `1.3`, up to `15` steps) to isolate the dominant rigid-body femoral cluster.
4. **Stage 3 — Combinatorial Optimal Rigid Triplet Selection:** Pre-filters candidate triplets using $Z$-spike-penalized 3D reference area ($w_z = 2.5$ when enabled), enforces adaptive 3D non-collinearity floors (`[0.05, 0.02, 0.01, 0.005, 0.001]` of the reference anchor 3D area), and maximizes the composite area-stability score $\mathcal{S}(\tau)$ (reducing 3D Rigid Deformation to **`28.4% ± 9.9%`**).

---

## Stage 5: 3D Kinematic Alignment & Multi-Tier Benchmark Evaluation

1. **Camera-Space Kinematics Extraction (`src/analysis/kinematic_alignment.py`):** Computes 3D centroid linear acceleration via **Savitzky-Golay** second-order differentiation (`window_length = 11`, `polyorder = 2`, `deriv = 2`). For 3-point triplets, applies **tangent-plane twist regularization** ($\alpha_z = 0.05$) around the instantaneous mean depth $\bar{Z}_t$ prior to Gram–Schmidt orthogonalization to suppress parasitic normal twist while preserving $\bar{Z}_t$ intact; for multi-point clouds, extracts the best-fit SVD frame with sign continuity.
2. **IMU World Projection, Kabsch Alignment & $\mathfrak{so}(3)$ Calibration:** Rotates IMU specific force into the static World frame and removes static gravity ($\mathbf{a}_{\text{world}}(t) = \mathbf{R}_{\text{imu}}(t)\mathbf{a}_{\text{body}}(t) - \mathbf{g}$), synchronizes temporal lag via overlap-normalized cross-correlation bounded to $|\tau| \le 2.5\text{ s}$, estimates the static camera-to-world rotation $\mathbf{R}_{\text{cam}\to\text{world}} \in \text{SO}(3)$ via Kabsch SVD (`Rotation.align_vectors`), and calibrates uncalibrated IMU mounting axes via anatomical PCA (`use_anatomical_pca_basis = True`) and Lie-algebra $\mathfrak{so}(3)$ relative orientation alignment.
3. **Multi-Metric Evaluation (`src/benchmark/evaluator.py`):** Evaluates 4 tiers (`Dummy`, `DELTAv2 Raw`, `DELTAv2 Refined Cloud`, `DELTAv2 Refined Triplet`) across waveform correlation (Pearson $r$), DTW, nRMSE, Pitch ROM error, spectral step cadence error, 3D Rigid Deformation, RMS Jerk, True Identity Swaps (`90°`/`45°`), and Axial Depth Artifacts (`0°`/`180°`).
