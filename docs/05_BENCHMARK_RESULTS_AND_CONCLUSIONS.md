# 05 — 40-Video Benchmark Results & Ablation Study

This chapter summarizes the final quantitative evaluation across all **40 stratified gait videos** (8 viewing perspectives, 5 videos per perspective) generated deterministically via `gait360 evaluate`.

---

## 1. Global 4-Tier Performance Summary (40 Videos)

| Metric | 1. Dummy Tracker | 2. DELTAv2 Raw | 3. DELTAv2 Pruned Cloud | 4. DELTAv2 Refined Triplet |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** ($\uparrow$) | `0.199 ± 0.305` | `0.185 ± 0.258` | `0.276 ± 0.278` | **`0.295 ± 0.329`** |
| **Pitch Peak XCorr** ($\uparrow$) | `0.369 ± 0.175` | `0.331 ± 0.156` | `0.391 ± 0.169` | **`0.415 ± 0.195`** |
| **Pitch nRMSE (%)** ($\downarrow$) | `33.2 ± 9.7` | `30.8 ± 6.7` | `34.4 ± 10.6` | **`30.5 ± 9.3`** |
| **Pitch ROM Error (%)** ($0\%$) | `-31.7% ± 88.2%` | `-15.5% ± 66.2%` | `+45.2% ± 112.9%` | **`+4.1% ± 70.8%`** |
| **Pitch Cadence Err (spm)** ($\downarrow$) | `29.6 ± 27.6` | `31.6 ± 24.5` | `17.9 ± 20.5` | **`16.5 ± 20.6`** |
| **AccX Pearson $r$** ($\uparrow$) | `0.205 ± 0.163` | `0.243 ± 0.168` | **`0.273 ± 0.152`** | `0.255 ± 0.162` |
| **AccX Peak XCorr** ($\uparrow$) | `0.354 ± 0.119` | `0.371 ± 0.129` | **`0.383 ± 0.117`** | `0.376 ± 0.123` |
| **3D Acc DTW (norm)** ($\downarrow$) | `1.345 ± 0.328` | `1.242 ± 0.168` | `1.239 ± 0.160` | **`1.230 ± 0.173`** |
| **3D Rigid Deformation (%)** ($\downarrow$) | `16.4 ± 9.4` *(2D)* | `110.9 ± 26.8` | `41.4 ± 20.1` | **`28.4 ± 9.9`** |
| **RMS Jerk ($\text{m/s}^3$)** ($\downarrow$) | `1452.0 ± 592.3` | `179.9 ± 62.7` | **`28.9 ± 11.4`** | `36.6 ± 14.3` |
| **True Identity Swaps ($90^\circ$ & $45^\circ$)** | `9 / 30 (30.0%)` | `4 / 30 (13.3%)` | `3 / 30 (10.0%)` | **`3 / 30 (10.0%)`** |
| **Axial Depth Artifacts ($0^\circ$ & $180^\circ$)** | `4 / 10 (40.0%)` | `1 / 10 (10.0%)` | `1 / 10 (10.0%)` | **`1 / 10 (10.0%)`** |

---

## 2. Head-to-Head Ablation: Refined Rigid Triplet vs. Pruned Point Cloud

- **3D Rigid-Body Stability (`28.4%` vs `41.4%`):** The Refined Triplet wins on **39 out of 40 videos (`97.5%`)**, reducing internal non-rigid deformation by `13.0` percentage points.
- **Sagittal Pitch Waveform & Amplitude (`r = 0.295` vs `0.276`, ROM Error `+4.1%` vs `+45.2%`):** The Refined Triplet wins on **21 out of 40 videos** for Pitch Pearson $r$ and eliminates the `+45.2%` ROM inflation caused by peripheral soft-tissue motion in the full cloud.
- **Complementary Centroid Smoothing for Linear Acceleration (`AccX r = 0.273` Cloud vs `0.255` Triplet):** Spatial averaging across ~150 points in the Pruned Cloud acts as a natural spatial low-pass filter against second-derivative finite-difference noise (`RMS Jerk = 28.9` vs `36.6 m/s³`), whereas the 3-point Rigid Triplet is superior for angular kinematics and rigid-body fidelity.

Full per-category tables are available in:
- [`results/REPORT_40_VIDEOS_BENCHMARK.md`](../results/REPORT_40_VIDEOS_BENCHMARK.md)
- [`results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`](../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md)
- [`results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md)
- [`results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md`](../results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md)

---

## 3. Context Sampling Ablation: `bbox_grid` vs. `mediapipe` (40 Videos)

To isolate the effect of the context anchor distribution (`987`-point `28×28` bounding-box grid including background vs. `1003`-point pure MediaPipe human silhouette), the full 40-video pipeline was re-executed with `--context-mode mediapipe` (`results_mediapipe/`):

| Configuration (40 Videos) | Pitch Pearson $r$ ($\uparrow$) | AccX Pearson $r$ ($\uparrow$) | 3D Rigid Def (%) ($\downarrow$) | Pitch ROM Error (%) | True Swaps ($90^\circ$/$45^\circ$) ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bbox_grid` — DELTAv2 Raw** | `0.185 ± 0.316` | `0.243 ± 0.171` | `33.3 ± 12.3` | `-15.5 ± 48.5` | `4 / 30 (13.3%)` |
| **`mediapipe` — DELTAv2 Raw** | `0.265 ± 0.344` | **`0.289 ± 0.152`** | `35.8 ± 18.9` | `+5.1 ± 65.8` | `6 / 30 (20.0%)` |
| **`bbox_grid` — Refined Cloud (SVD)** | `0.276 ± 0.288` | `0.288 ± 0.145` | `41.4 ± 8.3` | `+44.1 ± 54.6` | `5 / 30 (16.7%)` |
| **`mediapipe` — Refined Cloud (SVD)** | `0.293 ± 0.295` | `0.268 ± 0.134` | `45.1 ± 13.7` | `+36.6 ± 46.2` | `6 / 30 (20.0%)` |
| **`mediapipe` — Refined Triplet** | `0.209 ± 0.385` | `0.258 ± 0.160` | `34.3 ± 18.5` | `+9.9 ± 60.8` | `8 / 30 (26.7%)` |
| **`bbox_grid` — Refined Triplet (Best Overall)** | **`0.295 ± 0.329`** | `0.255 ± 0.162` | **`28.4 ± 9.9`** | **`+4.1 ± 70.8`** | **`3 / 30 (10.0%)`** |

- **Within the `mediapipe` run:** `Refined Cloud (SVD)` (`Pitch r = 0.293`, `6/30` swaps) and `Raw` (`Pitch r = 0.265`, `AccX r = 0.289`) clearly outperform `mediapipe Refined Triplet` (`Pitch r = 0.209`, `8/30` swaps), and improve over `bbox_grid` `Raw` (`0.185`) and `Cloud` (`0.276`).
- **Overall winner (`bbox_grid` + `Refined Triplet`):** Across all tested combinations, **`bbox_grid` + `Refined Triplet`** remains the best method overall — matching `mediapipe Cloud` on `Pitch r` (`0.295` vs `0.293`) while halving True Identity Swaps (`10.0%` vs `20.0%`–`26.7%`) and achieving the lowest 3D rigid deformation (`28.4%`) and Pitch ROM bias (`+4.1%`), as static background grid points stabilize global scene geometry during limb crossovers.
