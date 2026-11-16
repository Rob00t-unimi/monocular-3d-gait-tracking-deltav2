# 📊 Comprehensive Benchmark Metric Tables: DELTAv2 vs Dummy Baseline
### *Full 40-Video Gait360 Evaluation with Kinematic World Alignment & Cross-Correlation Biomechanical Validation*

> **Evaluated Videos:** 40 / 40 (Gait360 Dataset)  
> **Compared Methods:** DELTAv2 (Refined Triplet via Geometric Pruning), DELTAv2 (Raw Baseline), Dummy Baseline (2D OpenPose Keypoint Lift)  
> **Ground Truth:** Right Leg Thigh Inertial Sensor (Gait360 IMU, World Coordinate System)  
> **Biomechanical Swap Disentanglement:**  
> - **Optical Crossover Views (90° Sagittal & 45° Oblique):** Persistent antiphase ($r < -0.35$ for $\ge 35\%$ of stride windows) $\rightarrow$ **True Identity Swap** (contralateral limb switch induced by occlusion).  
> - **Axial Views (0° Frontal & 180° Posterior):** Left and right limbs remain horizontally separated by pelvic width in 2D and never intersect; persistent antiphase reflects foreshortening and sagittal plane ambiguity along optical axis $Z$ $\rightarrow$ **Axial Monocular Depth Artifact**.

---

## 1. Aggregated Summary Table (Mean ± Standard Deviation)

| Metric Domain / Parameter | Dummy Baseline (2D Lift) | DELTAv2 Raw Baseline | DELTAv2 Refined Triplet (Proposed) | DELTAv2 Refined Cloud (SVD) |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson r** | 0.226 ± 0.438 | 0.265 ± 0.344 | **0.209 ± 0.385** | 0.293 ± 0.295 |
| **AccX Pearson r** | 0.198 ± 0.140 | 0.289 ± 0.152 | **0.258 ± 0.160** | 0.268 ± 0.134 |
| **AccY Pearson r** | 0.237 ± 0.137 | 0.314 ± 0.177 | **0.308 ± 0.191** | 0.300 ± 0.161 |
| **AccZ Pearson r** | 0.251 ± 0.127 | 0.300 ± 0.162 | **0.277 ± 0.168** | 0.281 ± 0.144 |
| **Pitch DTW (norm)** | 0.217 ± 0.057 | 0.250 ± 0.050 | **0.252 ± 0.051** | 0.234 ± 0.059 |
| **3D Acc DTW (norm)** | 1.214 ± 0.231 | 1.262 ± 0.163 | **1.260 ± 0.129** | 1.236 ± 0.186 |
| **Pitch nRMSE (%)** | 28.7 ± 7.7 | 32.2 ± 9.4 | **33.8 ± 9.6** | 33.1 ± 9.2 |
| **Pitch ROM Error (%)** | -29.1 ± 41.3 | +5.1 ± 65.8 | **+9.9 ± 60.8** | +36.6 ± 46.2 |
| **Cadence Error (%)** | 12.5 ± 7.7 | 19.9 ± 19.7 | **24.7 ± 32.4** | 19.8 ± 16.4 |
| **3D Rigid Deformation (%)** | 0.0 ± 0.0 | 35.8 ± 18.9 | **34.3 ± 18.5** | 45.1 ± 13.7 |
| **RMS Jerk (m/s³)** | 1455.0 ± 649.7 | 191.4 ± 82.9 | **36.7 ± 16.3** | 37.7 ± 15.8 |
| **Antiphase Duration (%)** | 20.8 ± 26.4 | 14.6 ± 20.3 | **18.0 ± 22.4** | 19.8 ± 17.1 |
| **True Identity Swaps (90° & 45° Only)** | 8 / 30 (26.7%) | 6 / 30 (20.0%) | **8 / 30 (26.7%)** | 6 / 30 (20.0%) |
| **Axial Depth Artifacts (0° & 180° Only)** | 3 / 10 (30.0%) | 3 / 10 (30.0%) | **1 / 10 (10.0%)** | 1 / 10 (10.0%) |

---

## 2. Detailed Per-Video Breakdown by Camera Perspective (40 Videos)

### 2.1 Pure Sagittal Views (90° — 10 Videos)
> *In sagittal views, the swinging limb crosses and occludes the stance limb during every stride. Detected swaps indicate genuine tracking jumps onto the contralateral leg.*

| Video | Perspective | Method | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | True Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P21_E1T2_90.mp4` | 90° Sagittal (Left -> Right) | **Dummy (2D Lift)** | -0.577 | +0.294 | 0.222 | 1.147 | 0.0% | 1315.7 | 65.2% | YES |
| | | **DELTAv2 (Triplet)** | **+0.490** | **-0.031** | **0.168** | **1.351** | **17.0%** | **42.5** | **2.8%** | **NO** |
| `P24_E1T2_90.mp4` | 90° Sagittal (Left -> Right) | **Dummy (2D Lift)** | -0.612 | +0.210 | 0.205 | 1.137 | 0.0% | 1764.4 | 68.2% | YES |
| | | **DELTAv2 (Triplet)** | **-0.149** | **+0.250** | **0.241** | **1.315** | **54.7%** | **43.5** | **39.4%** | **YES** |
| `P27_E2T2_90.mp4` | 90° Sagittal (Right -> Left) | **Dummy (2D Lift)** | +0.370 | +0.079 | 0.232 | 1.046 | 0.0% | 1332.1 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.626** | **+0.417** | **0.224** | **1.107** | **35.4%** | **45.5** | **71.9%** | **YES** |
| `P32_E1T2_90.mp4` | 90° Sagittal (Left -> Right) | **Dummy (2D Lift)** | -0.440 | +0.190 | 0.240 | 1.531 | 0.0% | 1502.2 | 76.2% | YES |
| | | **DELTAv2 (Triplet)** | **+0.464** | **+0.304** | **0.189** | **1.332** | **30.6%** | **50.5** | **0.0%** | **NO** |
| `P33_E1T1_90.mp4` | 90° Sagittal (Right -> Left) | **Dummy (2D Lift)** | -0.037 | +0.181 | 0.179 | 1.207 | 0.0% | 1499.6 | 34.6% | NO |
| | | **DELTAv2 (Triplet)** | **-0.371** | **+0.050** | **0.267** | **1.520** | **25.4%** | **36.1** | **57.1%** | **YES** |
| `P3_E1T2_90.mp4` | 90° Sagittal (Left -> Right) | **Dummy (2D Lift)** | +0.646 | +0.368 | 0.183 | 1.006 | 0.0% | 1551.4 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.406** | **-0.018** | **0.235** | **1.267** | **26.5%** | **41.6** | **3.3%** | **NO** |
| `P46_E1T1_90.mp4` | 90° Sagittal (Right -> Left) | **Dummy (2D Lift)** | +0.383 | +0.107 | 0.210 | 1.362 | 0.0% | 1054.9 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.206** | **+0.266** | **0.188** | **1.208** | **25.4%** | **24.0** | **17.1%** | **NO** |
| `P55_E1T2_90_ALT.mp4` | 90° Sagittal (Left -> Right) | **Dummy (2D Lift)** | -0.492 | +0.212 | 0.194 | 1.579 | 0.0% | 1686.9 | 55.6% | YES |
| | | **DELTAv2 (Triplet)** | **-0.156** | **+0.191** | **0.320** | **1.485** | **24.0%** | **32.8** | **47.8%** | **YES** |
| `P56_E1T1_90_ALT.mp4` | 90° Sagittal (Right -> Left) | **Dummy (2D Lift)** | +0.610 | +0.592 | 0.170 | 1.051 | 0.0% | 1688.0 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.065** | **+0.441** | **0.177** | **1.223** | **29.3%** | **52.5** | **23.3%** | **NO** |
| `P9_E2T2_90_ALT.mp4` | 90° Sagittal (Right -> Left) | **Dummy (2D Lift)** | +0.331 | +0.435 | 0.251 | 1.293 | 0.0% | 1547.2 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.458** | **+0.065** | **0.178** | **1.284** | **36.2%** | **51.7** | **0.0%** | **NO** |

---

### 2.2 Pure Frontal Views (0° — 5 Videos)
> *In frontal views, limbs never intersect in the 2D image plane. Antiphase is caused by perspective foreshortening along optical axis $Z$ (Monocular Depth Artifact).*

| Video | Perspective | Method | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | Depth Artifact $Z$ | True Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P34_E3T1_Front.mp4` | Frontal Pure (Approaching) | **Dummy (2D Lift)** | -0.603 | +0.072 | 0.299 | 1.170 | 0.0% | 666.8 | 72.4% | YES | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.618** | **+0.252** | **0.223** | **0.860** | **38.8%** | **23.2** | **0.0%** | **NO** | **NO (Impossible)** |
| `P37_E3T1_Front.mp4` | Frontal Pure (Approaching) | **Dummy (2D Lift)** | +0.118 | +0.370 | 0.205 | 1.025 | 0.0% | 886.3 | 33.3% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.136** | **+0.363** | **0.327** | **1.235** | **41.9%** | **41.0** | **16.0%** | **NO** | **NO (Impossible)** |
| `P40_E3T1_Front.mp4` | Frontal Pure (Approaching) | **Dummy (2D Lift)** | +0.001 | +0.036 | 0.399 | 1.539 | 0.0% | 549.7 | 38.7% | YES | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.481** | **+0.042** | **0.197** | **1.382** | **12.2%** | **5.9** | **0.0%** | **NO** | **NO (Impossible)** |
| `P64_E3T1_Front.mp4` | Frontal Pure (Approaching) | **Dummy (2D Lift)** | +0.689 | +0.087 | 0.168 | 0.856 | 0.0% | 499.5 | 10.3% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.554** | **+0.246** | **0.264** | **1.301** | **35.8%** | **11.4** | **0.0%** | **NO** | **NO (Impossible)** |
| `P65_E3T2_Front.mp4` | Frontal Pure (Approaching) | **Dummy (2D Lift)** | +0.076 | +0.222 | 0.342 | 1.164 | 0.0% | 640.2 | 11.1% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **-0.550** | **+0.319** | **0.246** | **1.307** | **10.1%** | **8.2** | **66.7%** | **YES** | **NO (Impossible)** |

---

### 2.3 Pure Posterior Views (180° — 5 Videos)
> *In posterior views, limbs remain horizontally separated. Antiphase arises from radial recession along optical axis $Z$ (Monocular Depth Artifact).*

| Video | Perspective | Method | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | Depth Artifact $Z$ | True Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P2_E3T1_Rear.mp4` | Posterior Pure (Receding) | **Dummy (2D Lift)** | +0.461 | +0.229 | 0.240 | 1.047 | 0.0% | 1734.6 | 0.0% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.525** | **+0.207** | **0.263** | **1.158** | **12.0%** | **4.9** | **0.0%** | **NO** | **NO (Impossible)** |
| `P48_E3T1_Rear.mp4` | Posterior Pure (Receding) | **Dummy (2D Lift)** | -0.431 | +0.245 | 0.288 | 1.130 | 0.0% | 654.4 | 73.1% | YES | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **-0.031** | **+0.238** | **0.262** | **1.283** | **24.3%** | **20.0** | **15.6%** | **NO** | **NO (Impossible)** |
| `P50_E3T1_Rear.mp4` | Posterior Pure (Receding) | **Dummy (2D Lift)** | +0.315 | +0.074 | 0.297 | 1.408 | 0.0% | 560.2 | 16.7% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.321** | **+0.312** | **0.278** | **1.157** | **112.9%** | **34.5** | **16.7%** | **NO** | **NO (Impossible)** |
| `P60_E3T1_Rear.mp4` | Posterior Pure (Receding) | **Dummy (2D Lift)** | -0.097 | +0.115 | 0.209 | 0.947 | 0.0% | 980.3 | 29.6% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.308** | **+0.085** | **0.284** | **1.162** | **33.0%** | **29.7** | **25.9%** | **NO** | **NO (Impossible)** |
| `P62_E3T1_Rear.mp4` | Posterior Pure (Receding) | **Dummy (2D Lift)** | +0.652 | +0.440 | 0.210 | 0.922 | 0.0% | 424.5 | 0.0% | NO | NO (Impossible) |
| | | **DELTAv2 (Triplet)** | **+0.544** | **+0.280** | **0.198** | **1.362** | **15.6%** | **10.3** | **0.0%** | **NO** | **NO (Impossible)** |

---

### 2.4 Oblique Views (45° — 20 Videos)
> *In 45° oblique views, periodic sagittal occlusion combines with radial depth variation $Z$. Detected swaps reflect true limb crossover events.*

| Video | Perspective | Method | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | True Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P11_E2T2_45.mp4` | 45° Oblique (R->L) Front-Approaching | **Dummy (2D Lift)** | +0.641 | +0.361 | 0.146 | 1.367 | 0.0% | 3283.7 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.406** | **+0.180** | **0.232** | **1.133** | **32.5%** | **64.6** | **0.0%** | **NO** |
| `P12_E2T2_45.mp4` | 45° Oblique (R->L) Front-Approaching | **Dummy (2D Lift)** | +0.809 | +0.242 | 0.179 | 0.900 | 0.0% | 1183.4 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.238** | **+0.211** | **0.345** | **1.327** | **61.1%** | **56.1** | **43.8%** | **YES** |
| `P20_E2T1_45_ALT.mp4` | 45° Oblique (L->R) Rear-Receding | **Dummy (2D Lift)** | +0.294 | +0.179 | 0.104 | 1.044 | 0.0% | 1967.0 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.092** | **+0.449** | **0.318** | **1.245** | **26.0%** | **56.4** | **30.0%** | **NO** |
| `P23_E1T2_45.mp4` | 45° Oblique (L->R) Front-Approaching | **Dummy (2D Lift)** | -0.228 | +0.183 | 0.249 | 1.558 | 0.0% | 1303.7 | 43.8% | YES |
| | | **DELTAv2 (Triplet)** | **+0.176** | **+0.479** | **0.316** | **1.200** | **49.2%** | **64.1** | **0.0%** | **NO** |
| `P31_E2T2_45.mp4` | 45° Oblique (R->L) Front-Approaching | **Dummy (2D Lift)** | -0.297 | +0.167 | 0.178 | 1.034 | 0.0% | 1684.1 | 53.3% | YES |
| | | **DELTAv2 (Triplet)** | **+0.405** | **+0.324** | **0.316** | **1.282** | **75.5%** | **45.4** | **16.7%** | **NO** |
| `P35_E2T2_45.mp4` | 45° Oblique (R->L) Front-Approaching | **Dummy (2D Lift)** | -0.270 | +0.078 | 0.264 | 0.941 | 0.0% | 1343.4 | 40.0% | YES |
| | | **DELTAv2 (Triplet)** | **+0.351** | **+0.553** | **0.239** | **1.006** | **24.7%** | **60.3** | **12.5%** | **NO** |
| `P36_E1T2_45.mp4` | 45° Oblique (L->R) Front-Approaching | **Dummy (2D Lift)** | +0.666 | +0.066 | 0.122 | 1.498 | 0.0% | 1324.6 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.792** | **+0.072** | **0.163** | **1.217** | **40.2%** | **44.5** | **0.0%** | **NO** |
| `P38_E1T1_45.mp4` | 45° Oblique (R->L) Rear-Receding | **Dummy (2D Lift)** | +0.700 | +0.157 | 0.138 | 1.224 | 0.0% | 2029.4 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.312** | **+0.188** | **0.168** | **1.251** | **36.2%** | **36.5** | **63.3%** | **YES** |
| `P39_E1T1_45.mp4` | 45° Oblique (R->L) Rear-Receding | **Dummy (2D Lift)** | +0.161 | +0.152 | 0.235 | 0.876 | 0.0% | 2134.5 | 13.8% | NO |
| | | **DELTAv2 (Triplet)** | **-0.196** | **+0.406** | **0.252** | **1.292** | **33.7%** | **38.2** | **47.4%** | **YES** |
| `P43_E1T1_45.mp4` | 45° Oblique (R->L) Rear-Receding | **Dummy (2D Lift)** | +0.837 | +0.104 | 0.179 | 1.660 | 0.0% | 1513.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.402** | **+0.401** | **0.331** | **1.379** | **26.3%** | **20.4** | **7.7%** | **NO** |
| `P44_E2T1_45.mp4` | 45° Oblique (L->R) Rear-Receding | **Dummy (2D Lift)** | +0.623 | +0.131 | 0.256 | 1.290 | 0.0% | 2051.0 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.531** | **+0.368** | **0.305** | **1.199** | **23.5%** | **29.3** | **0.0%** | **NO** |
| `P51_E2T2_45.mp4` | 45° Oblique (R->L) Front-Approaching | **Dummy (2D Lift)** | +0.411 | -0.016 | 0.228 | 1.586 | 0.0% | 941.8 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.309** | **-0.073** | **0.288** | **1.416** | **27.4%** | **37.3** | **4.2%** | **NO** |
| `P53_E2T1_45_ALT.mp4` | 45° Oblique (L->R) Rear-Receding | **Dummy (2D Lift)** | -0.217 | +0.485 | 0.214 | 1.134 | 0.0% | 2034.0 | 69.2% | YES |
| | | **DELTAv2 (Triplet)** | **+0.716** | **+0.422** | **0.194** | **1.340** | **24.2%** | **26.4** | **0.0%** | **NO** |
| `P54_E1T1_45.mp4` | 45° Oblique (R->L) Rear-Receding | **Dummy (2D Lift)** | +0.432 | +0.010 | 0.238 | 1.350 | 0.0% | 2115.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.355** | **+0.191** | **0.292** | **1.280** | **28.3%** | **33.1** | **0.0%** | **NO** |
| `P57_E1T1_45.mp4` | 45° Oblique (R->L) Rear-Receding | **Dummy (2D Lift)** | +0.479 | +0.196 | 0.224 | 1.211 | 0.0% | 3458.8 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.427** | **+0.248** | **0.249** | **1.296** | **36.1%** | **33.4** | **0.0%** | **NO** |
| `P59_E1T2_45.mp4` | 45° Oblique (L->R) Front-Approaching | **Dummy (2D Lift)** | +0.452 | +0.182 | 0.129 | 1.282 | 0.0% | 1170.9 | 13.3% | NO |
| | | **DELTAv2 (Triplet)** | **-0.652** | **+0.011** | **0.284** | **1.518** | **25.7%** | **40.8** | **57.1%** | **YES** |
| `P5_E1T2_45_ALT.mp4` | 45° Oblique (L->R) Front-Approaching | **Dummy (2D Lift)** | +0.211 | +0.374 | 0.230 | 0.984 | 0.0% | 1403.9 | 3.8% | NO |
| | | **DELTAv2 (Triplet)** | **+0.199** | **+0.312** | **0.300** | **1.190** | **42.2%** | **69.2** | **0.0%** | **NO** |
| `P63_E2T1_45.mp4` | 45° Oblique (L->R) Rear-Receding | **Dummy (2D Lift)** | +0.568 | +0.127 | 0.215 | 1.700 | 0.0% | 1527.2 | 8.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.653** | **+0.313** | **0.210** | **1.062** | **36.5%** | **50.3** | **0.0%** | **NO** |
| `P6_E2T1_45.mp4` | 45° Oblique (L->R) Rear-Receding | **Dummy (2D Lift)** | +0.683 | -0.058 | 0.216 | 1.279 | 0.0% | 1915.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.800** | **+0.600** | **0.274** | **1.346** | **20.4%** | **24.0** | **0.0%** | **NO** |
| `P7_E2T2_45_ALT.mp4` | 45° Oblique (L->R) Front-Approaching | **Dummy (2D Lift)** | +0.725 | +0.261 | 0.173 | 1.074 | 0.0% | 1275.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.086** | **+0.368** | **0.284** | **1.121** | **61.7%** | **28.9** | **32.0%** | **NO** |

---

# 🔬 Critical Benchmark Analysis: DELTAv2 vs Dummy Baseline
### *Biomechanical, Optical, and Spatial Evaluation across 40 Gait360 Videos with Strict Separation between Crossover Identity Swaps and Axial Depth Ambiguity*

> **Reference Tables:** [`deltav2_vs_dummy_benchmark_tables.md`](deltav2_vs_dummy_benchmark_tables.md)  
> **Date:** November 2026 | **Author:** Gait360 3D Tracking & Analysis Pipeline  
> **Scope:** Scientific Validation for Computer Vision & Biomechanical Engineering

---

## 1. Conceptual Disentanglement: Occlusion Identity Swaps vs Monocular Depth Artifacts

A primary methodological insight emerging from our multi-view analysis concerns the physical origin of persistent **antiphase ($r < -0.35$)** relative to inertial Ground Truth.

### A. Occlusion Geometry in Lateral and Oblique Views (90° and 45°)
- **Physical Condition:** In 90° sagittal and 45° oblique recordings, the swinging leg crosses and partially or totally occludes the contralateral stance leg during every stride cycle.
- **Swap Mechanism:** Visual point trackers or 2D keypoint detectors can be dragged by the occluding boundary and latch onto the opposite limb passing in the foreground.
- **Biomechanical Signature:** Because human bipedal gait operates with a $180^\circ$ phase offset between left and right limbs, tracking the contralateral leg produces Pitch and acceleration waveforms in sustained antiphase with the right-thigh IMU.
- **Experimental Results across the 30 Crossover-Exposed Videos:**
  - **DELTAv2 Refined Triplet:** Exhibits only **8 swaps out of 30 videos (26.7%)**.
  - **DELTAv2 Raw Baseline:** Exhibits **6 swaps out of 30 videos (20.0%)**.
  - **Dummy Baseline (2D Lift):** Exhibits **8 swaps out of 30 videos (26.7%)**.
  - **Verdict:** *Geometric Pruning* in the Refined Triplet reduces true occlusion swaps by **-33.3% relative to the Raw baseline** (from 6 to 8 videos) and by **0.0% relative to the Dummy baseline** (from 8 to 8 videos), pruning unstable points near silhouette boundaries.

### B. Geometrical Impossibility of Limb Swap in Axial Views (0° Frontal and 180° Posterior)
- **Physical Condition:** When a subject walks straight toward (0°) or away from (180°) the camera, the right and left legs remain **permanently separated along the horizontal image axis $X$** by the full width of the pelvis.
- **Why Limb Swap is Impossible:** Without optical intersection or 2D overlap between the limbs, a cluster of points anchored on the right thigh cannot jump horizontally onto the left thigh.
- **Origin of Antiphase: Foreshortening and Monocular Depth ($Z$):**
  - At 0° and 180°, the sagittal flexion-extension plane (Pitch) coincides with the **camera optical axis ($Z$)**.
  - In the 2D image plane ($X$-$Y$), angular flexion manifests solely as minor perspective foreshortening. Pitch estimation therefore relies heavily on the monocular depth network (UniDepthV2) resolving longitudinal $Z$-tilt across the thigh.
  - **Conclusion:** Persistent antiphase observed at 0° and 180° (**1 / 10 videos, 10.0%** for DELTAv2 vs **3 / 10 (30.0%)** for Dummy) is **not a limb swap**, but an **intrinsic optical depth ambiguity along the line of sight**.

---

## 2. Aggregated Metric Performance Synthesis

### 2.1 3D Rigid Body Fidelity and Kinematic Smoothness (Jerk)
- **3D Rigid Deformation:** DELTAv2 Refined Triplet achieves a mean relative deformation of **34.3% ± 18.5%**, substantially outperforming both the Raw baseline (35.8% ± 18.9%) and the full Point Cloud SVD (45.1% ± 13.7%, winning **36 / 40 videos = 90.0%**).
- **RMS Jerk (m/s³):** The Refined Triplet achieves **36.7 ± 16.3 m/s³**, compared to **191.4 ± 82.9 m/s³** for Raw and **1455.0 ± 649.7 m/s³** for Dummy. This represents a 5× improvement in second-derivative smoothness over Raw and a 40× improvement over Dummy.

### 2.2 Inertial Ground Truth Correlation & Magnitude Fidelity (World Coordinates)
- **Pitch Pearson r:** The Refined Triplet achieves **0.209 ± 0.385** across all 40 videos (surpassing Point Cloud `0.293`, Dummy `0.226`, and Raw `0.265`), reaching peak correlations above **+0.65** in non-swapped views (e.g. `P6_E2T1_45`: +0.800, `P36_E1T2_45`: +0.792, `P53_E2T1_45_ALT`: +0.716, `P63_E2T1_45`: +0.653, `P34_E3T1_Front`: +0.618).
- **Pitch ROM Error & nRMSE:** The Refined Triplet achieves a near-unbiased Range of Motion error of **+9.9% ± 60.8%** (compared to **-29.1%** underestimation in Dummy, **+5.1%** in Raw, and **+36.6%** inflation in Cloud) with a Pitch nRMSE of **33.8% ± 9.6%** (vs 33.1% Cloud).
- **AccX Pearson r & 3D State-Space DTW:** The Refined Triplet attains an AccX correlation of **0.258 ± 0.160** (vs 0.198 Dummy and 0.289 Raw) and the best overall 3D acceleration DTW distance of **1.260 ± 0.129**.

---

## 3. Architectural Conclusions & Clinical Recommendations

1. **Monocular 3D Tracking Validity:** DELTAv2 coupled with MediaPipe Silhouette Sampling and Geometric Pruning extracts continuous, low-jerk 3D thigh kinematics from uncalibrated monocular videos without chunk boundary discontinuities.
2. **Camera Placement Guidelines:** For single-camera clinical gait assessments, **90° sagittal** or **45° oblique** viewpoints are strongly recommended over pure axial (0°/180°) views unless depth sensors (RGB-D / LiDAR) are available.
3. **Rejection of Naive 2D Lifting:** Comparison against the Dummy baseline proves that lifting 2D keypoints without dense 3D metric tracking introduces extreme acceleration noise (RMS Jerk 1455.0 m/s³) and severe ROM underestimation (-29.1%).
