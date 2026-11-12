# 03 — Experimental Chronicle & 3D Tracking Model Selection

This chapter documents the chronological progression of architectures evaluated during the research phase (`experiments/01_spatracker_vggt/`, `experiments/02_delta_v1/`, `experiments/03_deltav2_prototyping/`).

---

## 1. Phase 1: Exploration of VGGT and SpaTrackerV2 (`experiments/01_spatracker_vggt/`)

### 1.1 VGGT (Visual Geometry Grounded Transformer)
- **Hypothesis:** Reconstruct dense 3D scene point clouds directly from monocular video frames.
- **Failure Mode on Gait Sequences:** VGGT assumes a predominantly static scene observed by a moving camera. In clinical gait laboratory recordings with a **static camera** and a **non-rigidly moving human subject**, background walls dominate the global geometry while the walking subject suffers severe depth flattening and temporal incoherence.

### 1.2 SpaTrackerV2 (Bidirectional Forward-Backward Point Tracking)
- **Hypothesis:** Track the 3 thigh query points forward and backward from the optimal reference frame $t_{\text{ref}}$ (selected at maximum horizontal knee separation).
- **Empirical Findings (16-Video Matched Cohort):**
  - **High Identity-Swap Rate (31.2%, 5/16 videos):** During contralateral leg crossover in $90^\circ$ lateral and $45^\circ$ oblique views, sparse queries frequently latch onto the swinging opposite leg.
  - **Low Smoothness & Rigidity:** Produces an average RMS Jerk of `98.4 ± 41.2 m/s³` and 3D Rigid Deformation of `68.4% ± 24.5%`.
- **Decision:** Abandon sparse-only tracking in favor of dense 3D tracking with anatomical context anchors.

---

## 2. Phase 2: Transition to Dense 3D Tracking with DELTA v1 (`experiments/02_delta_v1/`)

- **Innovation:** Introduced **Anatomical Body Anchors** (`200` dense thigh samples + `300` pelvis anti-swap points + `500` torso context points), forcing the transformer cross-attention to disambiguate the ipsi-lateral thigh from the contralateral leg.
- **Empirical Findings (16-Video Matched Cohort):**
  - Reduced identity swaps from `31.2%` down to **`12.5%` (2/16 videos)** and improved Pitch Pearson $r$ from `0.142` to `0.218`.
  - **Limitation:** DELTA v1 exhibited high-frequency monocular depth jitter (`RMS Jerk = 64.2 ± 28.7 m/s³`) and required sliding-window chunking on long sequences (`>250` frames), introducing boundary stitching discontinuities.

---

## 3. Phase 3: Adoption of DELTAv2 + 3-Stage Geometric Pruning (`experiments/03_deltav2_prototyping/`)

- **Breakthrough:** Upgrading to **DELTAv2** coupled with **UniDepthV2** metric depth and the **3-Stage Geometric Pruner** (`z_weight=0.25`, `h_min >= 3.5 cm`, `AR <= 8.0`) eliminated chunk boundary artifacts and stabilized 3D thigh geometry across all 360° viewing angles.

| Architecture (16-Video Matched Cohort) | Pitch Pearson $r$ ($\uparrow$) | Pitch nRMSE (%) ($\downarrow$) | AccX Pearson $r$ ($\uparrow$) | 3D Acc DTW ($\downarrow$) | 3D Rigid Def. (%) ($\downarrow$) | RMS Jerk ($\text{m/s}^3$) ($\downarrow$) | Identity Swaps ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Dummy Tracker (OpenPose)** | `0.211 ± 0.312` | `32.8 ± 9.4` | `0.214 ± 0.158` | `1.332 ± 0.310` | `15.8 ± 8.9` *(2D)* | `1418.5 ± 560.2` | `5 / 16 (31.2%)` |
| **2. SpaTrackerV2 (Forward-Backward)** | `0.142 ± 0.285` | `36.4 ± 11.2` | `0.168 ± 0.149` | `1.385 ± 0.295` | `68.4 ± 24.5` | `98.4 ± 41.2` | `5 / 16 (31.2%)` |
| **3. DELTA v1 (Dense 3D)** | `0.218 ± 0.294` | `33.1 ± 8.8` | `0.229 ± 0.161` | `1.278 ± 0.194` | `52.1 ± 18.7` | `64.2 ± 28.7` | `2 / 16 (12.5%)` |
| **4. DELTAv2 Raw (Unfiltered)** | `0.198 ± 0.264` | `30.2 ± 6.5` | `0.251 ± 0.165` | `1.235 ± 0.162` | `108.4 ± 25.1` | `174.6 ± 60.4` | `2 / 16 (12.5%)` |
| **5. DELTAv2 Refined Triplet (Ours)** | **`0.312 ± 0.334`** | **`29.8 ± 9.1`** | **`0.268 ± 0.159`** | **`1.218 ± 0.168`** | **`27.9 ± 9.6`** | **`35.8 ± 13.9`** | **`1 / 16 (6.2%)`** |
