# 🔬 Technical Decision Report: Exploratory Evaluation of SpatialTrackerV2 + VGGT

- **Experiment Branch:** `experiment/spatracker-vggt`
- **Evaluated Models:** `VGGT4Track` (`Yuxihenry/SpatialTrackerV2_Front`) + `SpatialTrackerV2-Offline` (`Predictor`)
- **Decision Verdict:** **Discarded as Primary Production Backbone** (Retained strictly as Matched-Window Benchmark Baseline)

---

## 1. Experimental Hypothesis & Architecture

At project inception, **SpatialTrackerV2** coupled with **VGGT (Visual Geometry Grounded Transformer)** was investigated as a joint camera-pose and 3D point tracking architecture for uncalibrated monocular gait videos from the **Gait360** dataset:
1. **Geometric Initialization (`VGGT4Track`):** Estimates dense 3D point maps (`points_map`), camera extrinsics ($\mathbf{R}_{c2w}, \mathbf{T}_{c2w}$), intrinsics (`intrs`), and depth uncertainty (`unc_metric`).
2. **Bidirectional Auto-Pivot Tracking (`spatrack_test.py`):** Because SpatialTrackerV2 operates exclusively forward in time, a bidirectional wrapper was engineered around the validated reference frame $t_{\text{ref}}$:
   - The prefix segment $[0, t_{\text{ref}}]$ is reversed via `torch.flip`, tracked forward from $t_{\text{ref}}$, and flipped back via `np.flip`.
   - The suffix segment $[t_{\text{ref}}, T_{\text{end}}]$ is tracked forward natively.
   - Both half-trajectories are projected into world coordinates via $\mathbf{X}_{\text{world}}(t) = \mathbf{R}_{c2w}(t)\mathbf{X}_{\text{cam}}(t) + \mathbf{T}_{c2w}(t)$ and stitched at $t_{\text{ref}}$.

---

## 2. Quantitative Findings (Strict Matched Window: 60–90 Frames)

| Metric | SpatialTrackerV2 + VGGT (Matched 60–90 fr) | DELTAv2 + MediaPipe (Matched 60–90 fr) |
| :--- | :---: | :---: |
| **AccX Pearson $r$** | $+0.360 \pm 0.255$ | **$+0.449 \pm 0.255$** |
| **Pitch Pearson $r$** | **$+0.546 \pm 0.250$** | $+0.430 \pm 0.398$ |
| **Pitch DTW (norm)** | $0.269 \pm 0.075$ | **$0.221 \pm 0.078$** |
| **3D Rigid Deformation (%)** | $56.7\% \pm 13.3\%$ | **$19.6\% \pm 7.6\%$** |
| **RMS Jerk ($\text{m/s}^3$)** | **$30.4 \pm 2.0$** | $30.7 \pm 13.0$ |

---

## 3. Technical Reasons for Deprecation

1. **Severe Non-Rigid Spatial Elasticity (56.7% Deformation):**
   Global bundle adjustment in SpatialTrackerV2 optimizes 2D reprojection consistency without enforcing pairwise Euclidean distance constraints on anatomical segments. Consequently, the tracked femur triangle stretches and compresses by **56.7% ± 13.3%** (nearly 3× worse than DELTAv2's **19.6% ± 7.6%**).
2. **Prohibitive GPU Memory Scaling (60–90 Frame Ceiling):**
   Jointly holding VGGT multi-frame attention maps and bundle adjustment gradients exhausts 24 GB VRAM beyond ~90 frames, making full-sequence gait analysis (180–300 frames, 5–8 gait cycles) infeasible in a single pass.
3. **Diffuse Background Query Drift:**
   Uniform background grid queries required to stabilize VGGT camera pose estimation frequently latch onto moving shadows or contralateral limbs during sagittal cross-overs, collapsing the tracking.

**Conclusion:** We transition to **DELTA / DELTAv2** with metric depth estimation (`UniDepthV2`) and anatomical query masking.
