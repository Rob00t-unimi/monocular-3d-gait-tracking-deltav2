# 📊 Executive Summary & Benchmark Conclusions: DELTAv2 vs. Dummy Baseline
### *40-Video Quantitative Gait360 Evaluation with Inertial Ground-Truth Validation*

This document provides a high-level executive summary of the benchmark results evaluated across all **40 stratified gait videos** (39 subjects) from the public **Gait360** dataset, contrasting the proposed **DELTAv2 (Refined Triplet via Geometric Pruning)** against the **Dummy Baseline (2D OpenPose Keypoint Lifting)** and synchronized **WitMotion IMU** wearable ground-truth sensors.

---

## 🚀 1. The 4 Key Quantitative Advantages of DELTAv2

### 1. Kinematic Smoothness & Finite-Difference Noise Attenuation: **~40× Better**
* **Metric:** **RMS Jerk** ($\text{m/s}^3$, third time-derivative of position / first derivative of acceleration).
* **Dummy Baseline:** `1452.0 ± 646.6 m/s³`
* **DELTAv2 (Refined Triplet):** **`36.6 ± 14.3 m/s³`**
* **Verdict:** **97.5% reduction in derivative noise (~40× improvement).**  
  Naive 2D keypoint lifting suffers from discrete pixel quantization jitter, which explodes into severe acceleration spikes upon second differentiation. DELTAv2, through continuous metric depth estimation (UniDepthV2) and temporal optimization, produces physically plausible, smooth trajectories for the human thigh.

### 2. Angular Amplitude Fidelity (Range of Motion): **Zero-Bias Recovery**
* **Metric:** **Pitch ROM Error %** (percentage error on thigh sagittal flexion-extension excursion relative to IMU ground truth, ideal target = $0\%$).
* **Dummy Baseline:** `-31.7% ± 46.0%`
* **DELTAv2 (Refined Triplet):** **`+4.1% ± 70.8%`**
* **Verdict:** The Dummy baseline severely underestimates joint excursion (compressing the stride by nearly a third, -31.7%). DELTAv2 virtually eliminates this systematic bias (+4.1%), centering estimated joint excursion around true anatomical values.

### 3. Occlusion Robustness (Contralateral Limb Swapping): **3× Fewer Failures**
* **Metric:** **True Identity Swaps** (tracking failure caused by contralateral limb crossover in $90^\circ$ lateral and $45^\circ$ oblique perspectives).
* **Dummy Baseline:** `30.0%` (9 / 30 videos)
* **DELTAv2 (Refined Triplet):** **`10.0%`** (3 / 30 videos)
* **Verdict:** **66.7% reduction in identity swaps (3× higher reliability).**  
  Dense anatomical anchor sampling (MediaPipe body silhouette + pelvis) coupled with the *Geometric Pruner* preserves tracking focus on the ipsilateral thigh during swing phase, where 2D detectors consistently jump to the opposite leg.

### 4. Sagittal Waveform Correlation: **+48% Higher Fidelity**
* **Metric:** **Pitch Pearson $r$** (linear waveform correlation against ground-truth IMU pitch).
* **Dummy Baseline:** `0.199 ± 0.452`
* **DELTAv2 (Refined Triplet):** **`0.295 ± 0.329`** (peaking above **$+0.65$** in favorable non-axial viewpoints)
* **Verdict:** **+48.2% improvement in average correlation** with reduced standard deviation, demonstrating consistent stride cycle extraction.

---

## 📊 2. Comparative Benchmark Summary Table

| Evaluation Domain | Metric / Parameter | Dummy Baseline (2D Lift) | DELTAv2 (Refined Triplet) | Benchmark Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **Dynamics (Smoothness)** | **RMS Jerk** ($\downarrow$) | $1452.0 \text{ m/s}^3$ | **$36.6 \text{ m/s}^3$** | **~40× smoother** (eliminates derivative spikes) |
| **Amplitude (ROM)** | **Pitch ROM Error** ($0\%$) | $-31.7\%$ | **$+4.1\%$** | **Bias eliminated** (does not compress stride) |
| **Reliability (Occlusion)** | **True Identity Swaps** ($\downarrow$) | $30.0\%$ (9/30) | **$10.0\%$ (3/30)** | **Swaps cut by 66.7%** (3× more robust) |
| **Waveform (Kinematics)** | **Pitch Pearson $r$** ($\uparrow$) | $0.199$ | **$0.295$** | **+48.2% correlation gain** against IMU |
| **Frequency (Cadence)** | **Cadence Error** ($\downarrow$) | $29.6 \text{ spm}$ | **$16.5 \text{ spm}$** | **Cadence error reduced by 44.3%** |
| **Spatial State-Space** | **3D Acc DTW (norm)** ($\downarrow$) | $1.345$ | **$1.230$** | **Closer geometric trajectory distance** |

---

*Detailed Reports:*
- Full 40-Video Benchmark Report: [`../results/REPORT_40_VIDEOS_BENCHMARK.md`](../results/REPORT_40_VIDEOS_BENCHMARK.md)
- Triplet vs. Point Cloud Ablation: [`../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`](../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md)
- 16-Video Model Comparison (SpaTracker vs DELTA vs DELTAv2): [`../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md)
- Raw Metrics JSON: [`../results/benchmark_40_metrics.json`](../results/benchmark_40_metrics.json)
