# 03 — Experimental Chronicle & 3D Tracking Model Selection

This chapter documents the chronological progression of 3D tracking architectures evaluated during the research phase (`experiments/01_spatracker_vggt/`, `experiments/02_delta_v1/`, `experiments/03_deltav2_prototyping/`), aligned with the authoritative matched-window benchmark (`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`) and Technical Report Table I.

---

## 1. Phase 1: Exploration of VGGT and SpatialTrackerV2 (`experiments/01_spatracker_vggt/`)

### 1.1 VGGT (Visual Geometry Grounded Transformer)
- **Reference:** J. Wang et al., *"VGGT: Visual Geometry Grounded Transformer"*, *CVPR* 2025 (`arXiv:2503.11651`).
- **Hypothesis:** Estimate dense 3D point maps, camera intrinsics, and camera extrinsics directly from monocular gait sequences (`VGGT4Track`).
- **Failure Mode on Gait Sequences:** VGGT is designed primarily for static-scene multi-view geometry. In gait laboratory recordings with a **static camera** and a **non-rigidly moving subject**, holding multi-frame attention maps across long sequences exhausts 24 GB GPU VRAM beyond ~60–90 frames.

### 1.2 SpatialTrackerV2 (Bidirectional Forward-Backward Point Tracking)
- **Reference:** Y. Xiao et al., *"SpatialTrackerV2: 3D Point Tracking Made Easy"*, *ICCV* 2025 (`arXiv:2507.12462`).
- **Hypothesis:** Track thigh query points bidirectionally around the reference frame $t_{\text{ref}}$ via a forward/flipped-prefix wrapper coupled with VGGT4Track.
- **Empirical Findings (6-Video Strict Matched Window, 60–90 frames):**
  - Achieves high short-window Pitch correlation (`+0.546 ± 0.250`) and low RMS Jerk (`30.4 ± 2.0 m/s³`), **but suffers severe non-rigid spatial elasticity (`3D Rigid Deformation = 56.7% ± 13.3%`, nearly 3× worse than DELTAv2)** because global bundle adjustment lacks pairwise anatomical rigidity constraints.
  - **Scalability Bottleneck:** Cannot process full gait trials (164–238+ frames) in a single pass due to VRAM limits.
- **Decision:** Discarded as primary production backbone; retained as a short-window benchmark reference.

---

## 2. Phase 2: Transition to Dense 3D Tracking with DELTA v1 (`experiments/02_delta_v1/`)

- **Reference:** T. D. Ngo et al., *"DELTA: Dense Efficient Long-range 3D Tracking for Any Video"*, *ICLR* 2025 (`arXiv:2410.24211`).
- **Innovation:** Replaced diffuse grids with a structured 1003-point anatomical armature (`3` anchors + `200` thigh + `300` pelvis + `500` torso) coupled with **UniDepthV2** metric depth and 60-frame chunked Kabsch-Umeyama SVD stitching (25-frame overlap).
- **Empirical Findings (6-Video Matched Window):**
  - Competitive short-window Pitch correlation (`+0.535 ± 0.335`) and AccX correlation (`+0.482 ± 0.117`).
  - **Limitation:** Chunk-boundary SVD stitching introduces sub-millimeter positional discontinuities that explode under second-order differentiation (**`RMS Jerk = 476.3 ± 133.8 m/s³`**, 15.5× noisier than DELTAv2) and still yields high deformation (`46.4% ± 31.0%`).

---

## 3. Phase 3: Adoption of DELTAv2 + 3-Stage Geometric Pruning (`experiments/03_deltav2_prototyping/`)

- **Reference:** T. D. Ngo et al., *"DELTAv2: Accelerating Dense 3D Point Tracking"*, `arXiv:2602.18432`, 2026; paired with **UniDepthV2** (`L. Piccinelli et al., arXiv:2502.20110, 2025`).
- **Breakthrough:** Single-pass full-sequence tracking (240+ frames without chunking, `coarse_to_fine_dense=True`, `n_iters=6`) paired with **GeometricPruner v4.1** eliminated chunk-boundary spikes and achieved the lowest 3D rigid deformation (`19.6%` matched-window, `28.4%` full-sequence 40-video cohort).

### Strict Matched-Window Multi-Model Comparison (6 Benchmark Trials, 60–90 Frames)

| Architecture / Configuration | Temporal Window | AccX $r$ ($\uparrow$) | Pitch $r$ ($\uparrow$) | Pitch DTW ($\downarrow$) | 3D Acc DTW ($\downarrow$) | 3D Rigid Def. (%) ($\downarrow$) | RMS Jerk ($\text{m/s}^3$) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpatialTrackerV2 + VGGT** | Matched (60–90 fr) | `+0.360 ± 0.255` | **`+0.546 ± 0.250`** | `0.269 ± 0.075` | `1.167 ± 0.211` | `56.7 ± 13.3` | **`30.4 ± 2.0`** |
| **DELTAv2 (MediaPipe — Matched)** | Matched (60–90 fr) | `+0.449 ± 0.255` | `+0.430 ± 0.398` | **`0.221 ± 0.078`** | `1.309 ± 0.077` | **`19.6 ± 7.6`** | `30.7 ± 13.0` |
| **DELTAv2 (BBox Grid — Matched)** | Matched (60–90 fr) | **`+0.497 ± 0.218`** | `+0.232 ± 0.298` | `0.312 ± 0.099` | `1.291 ± 0.212` | `24.5 ± 7.8` | `176.7 ± 46.7` |
| **DELTA v1 (Chunked SVD — Matched)** | Matched (60–90 fr) | `+0.482 ± 0.117` | `+0.535 ± 0.335` | `0.235 ± 0.053` | `1.242 ± 0.176` | `46.4 ± 31.0` | `476.3 ± 133.8` |
| **Dummy (2D Lift — Matched)** | Matched (60–90 fr) | `+0.394 ± 0.271` | `+0.227 ± 0.181` | `0.278 ± 0.055` | **`1.067 ± 0.138`** | `0.0 ± 0.0` *(2D)* | `1236.6 ± 387.6` |
| **DELTAv2 (MediaPipe — Full Seq.)** | Full (164–238 fr) | `+0.401 ± 0.121` | `+0.184 ± 0.217` | `0.234 ± 0.020` | `1.166 ± 0.100` | `26.9 ± 5.6` | `35.6 ± 6.9` |

Full per-video tables are available in [`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md) and the full 40-video production evaluation in [`docs/05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md`](05_BENCHMARK_RESULTS_AND_CONCLUSIONS.md).
