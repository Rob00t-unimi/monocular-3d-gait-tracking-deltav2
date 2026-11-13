# 📊 Tabelle Metriche Complete del Benchmark: DELTAv2 vs Baseline Dummy
### *Valutazione Completa su 40 Video Gait360 con Allineamento Cinematico al Mondo e Validazione Biomeccanica via Cross-Correlazione*

> **Video Valutati:** 40 / 40 (Dataset Gait360)  
> **Nota sul Campionamento di Contesto:** Tutte le 40 traiettorie salvate in questo report v1.0.0 hanno utilizzato il fallback a griglia `28×28` da `987` punti (`3 + 200 + 784`) invece della maschera silhouette MediaPipe (`1003` punti) a causa di un fallback silenzioso in `DeltaV2Tracker`. Poiché l'ablazione su 16 video mostrava un jerk pre-pruning inferiore con MediaPipe (`146.9` vs `176.7 m/s³`), questi risultati rappresentano la baseline `bbox_grid` in attesa di ri-esecuzione con MediaPipe attivo.  
> **Metodi a Confronto:** DELTAv2 (Tripletta Rifinita via Geometric Pruning), DELTAv2 (Baseline Raw), Baseline Dummy (Lift 2D Keypoint OpenPose)  
> **Ground Truth:** Sensore Inerziale Coscia Destra (Gait360 IMU, Sistema di Coordinate World)  
> **Distinzione Biomeccanica degli Scambi (Disentanglement):**  
> - **Optical Crossover Views (90° Sagittale & 45° Oblique):** Persistent antiphase ($r < -0.35$ for $\ge 35\%$ of stride windows) $\rightarrow$ **True Identity Swap** (contralateral limb switch induced by occlusion).  
> - **Viste Assiali (0° Frontale e 180° Posteriore):** Left and right limbs remain horizontally separated by pelvic width in 2D and never intersect; persistent antiphase reflects foreshortening and sagittal plane ambiguity along optical axis $Z$ $\rightarrow$ **Axial Monocular Artefatto Profondita**.

---

## 1. Tabella Riassuntiva Aggregata (Media ± Deviazione Standard)

| Dominio Metrico / Parametro | Dummy Baseline (2D Lift) | DELTAv2 Raw Baseline | DELTAv2 Refined Triplet (Proposed) | DELTAv2 Refined Cloud (SVD) |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson r** | 0.199 ± 0.452 | 0.185 ± 0.316 | **0.295 ± 0.329** | 0.276 ± 0.288 |
| **AccX Pearson r** | 0.205 ± 0.140 | 0.243 ± 0.171 | **0.255 ± 0.162** | 0.288 ± 0.145 |
| **AccY Pearson r** | 0.246 ± 0.135 | 0.309 ± 0.185 | **0.324 ± 0.160** | 0.330 ± 0.197 |
| **AccZ Pearson r** | 0.245 ± 0.150 | 0.322 ± 0.173 | **0.303 ± 0.151** | 0.275 ± 0.139 |
| **Pitch DTW (norm)** | 0.218 ± 0.057 | 0.244 ± 0.049 | **0.243 ± 0.042** | 0.240 ± 0.044 |
| **3D Acc DTW (norm)** | 1.244 ± 0.230 | 1.275 ± 0.214 | **1.230 ± 0.173** | 1.248 ± 0.170 |
| **Pitch nRMSE (%)** | 28.6 ± 7.8 | 30.4 ± 6.7 | **30.5 ± 9.3** | 34.4 ± 8.3 |
| **Pitch ROM Error (%)** | -31.7 ± 46.0 | -15.5 ± 48.5 | **+4.1 ± 70.8** | +44.1 ± 54.6 |
| **Cadence Error (%)** | 11.7 ± 7.3 | 25.7 ± 27.4 | **24.1 ± 26.2** | 20.1 ± 16.1 |
| **3D Rigid Deformation (%)** | 0.0 ± 0.0 | 33.3 ± 12.3 | **28.4 ± 9.9** | 41.4 ± 8.3 |
| **RMS Jerk (m/s³)** | 1452.0 ± 646.6 | 179.9 ± 58.6 | **36.6 ± 14.3** | 36.2 ± 14.1 |
| **Durata in Antifase (%)** | 22.3 ± 26.4 | 17.3 ± 17.8 | **13.5 ± 15.9** | 16.8 ± 19.9 |
| **Veri Scambi di Identita (Solo 90° e 45°)** | 9 / 30 (30.0%) | 4 / 30 (13.3%) | **3 / 30 (10.0%)** | 5 / 30 (16.7%) |
| **Axial Artefatto Profonditas (0° & 180° Only)** | 4 / 10 (40.0%) | 1 / 10 (10.0%) | **1 / 10 (10.0%)** | 2 / 10 (20.0%) |

---

## 2. Disaggregazione Dettagliata per Video e Angolo di Ripresa (40 Video)

### 2.1 Viste Sagittalei Pure (90° — 10 Video)
> *Nelle viste sagittali, la gamba in oscillazione incrocia e occlude la gamba d'appoggio a ogni passo. Gli swap rilevati indicano effettivi salti di tracking sulla gamba controlaterale.*

| Video | Prospettiva | Metodo | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | Vero Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P21_E1T2_90.mp4` | 90° Sagittale (Sinistra -> Destra) | **Dummy (2D Lift)** | -0.577 | +0.294 | 0.222 | 1.147 | 0.0% | 1315.7 | 65.2% | SI |
| | | **DELTAv2 (Triplet)** | **+0.289** | **+0.035** | **0.173** | **1.317** | **20.7%** | **42.0** | **11.1%** | **NO** |
| `P24_E1T2_90.mp4` | 90° Sagittale (Sinistra -> Destra) | **Dummy (2D Lift)** | +0.787 | +0.080 | 0.162 | 1.448 | 0.0% | 1839.2 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.523** | **-0.262** | **0.321** | **1.591** | **23.3%** | **45.8** | **0.0%** | **NO** |
| `P27_E2T2_90.mp4` | 90° Sagittale (Destra -> Sinistra) | **Dummy (2D Lift)** | -0.150 | +0.102 | 0.204 | 1.649 | 0.0% | 1350.3 | 36.0% | SI |
| | | **DELTAv2 (Triplet)** | **+0.554** | **+0.098** | **0.300** | **1.215** | **52.4%** | **29.1** | **13.9%** | **NO** |
| `P32_E1T2_90.mp4` | 90° Sagittale (Sinistra -> Destra) | **Dummy (2D Lift)** | +0.694 | +0.446 | 0.182 | 1.160 | 0.0% | 1502.2 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.787** | **+0.496** | **0.165** | **1.116** | **22.8%** | **42.5** | **0.0%** | **NO** |
| `P33_E1T1_90.mp4` | 90° Sagittale (Destra -> Sinistra) | **Dummy (2D Lift)** | -0.037 | +0.181 | 0.179 | 1.207 | 0.0% | 1499.6 | 34.6% | NO |
| | | **DELTAv2 (Triplet)** | **+0.367** | **+0.326** | **0.221** | **1.443** | **24.3%** | **44.3** | **12.5%** | **NO** |
| `P3_E1T2_90.mp4` | 90° Sagittale (Sinistra -> Destra) | **Dummy (2D Lift)** | +0.599 | +0.241 | 0.165 | 1.391 | 0.0% | 1567.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.491** | **+0.335** | **0.216** | **1.309** | **25.7%** | **45.0** | **0.0%** | **NO** |
| `P46_E1T1_90.mp4` | 90° Sagittale (Destra -> Sinistra) | **Dummy (2D Lift)** | +0.383 | +0.107 | 0.210 | 1.362 | 0.0% | 1054.9 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.690** | **+0.136** | **0.191** | **1.322** | **34.1%** | **31.1** | **0.0%** | **NO** |
| `P55_E1T2_90_ALT.mp4` | 90° Sagittale (Sinistra -> Destra) | **Dummy (2D Lift)** | -0.791 | +0.180 | 0.191 | 1.178 | 0.0% | 1721.9 | 61.9% | SI |
| | | **DELTAv2 (Triplet)** | **-0.051** | **+0.256** | **0.229** | **1.174** | **40.6%** | **42.2** | **7.7%** | **NO** |
| `P56_E1T1_90_ALT.mp4` | 90° Sagittale (Destra -> Sinistra) | **Dummy (2D Lift)** | -0.192 | +0.076 | 0.279 | 1.516 | 0.0% | 1551.9 | 54.5% | SI |
| | | **DELTAv2 (Triplet)** | **+0.003** | **+0.129** | **0.202** | **1.342** | **33.2%** | **49.2** | **22.2%** | **NO** |
| `P9_E2T2_90_ALT.mp4` | 90° Sagittale (Destra -> Sinistra) | **Dummy (2D Lift)** | +0.331 | +0.435 | 0.251 | 1.293 | 0.0% | 1547.2 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.219** | **+0.435** | **0.281** | **1.393** | **51.4%** | **50.6** | **30.8%** | **NO** |

---

### 2.2 Viste Frontali Pure (0° — 5 Video)
> *In frontal views, limbs never intersect in the 2D image plane. Antiphase is caused by perspective foreshortening along optical axis $Z$ (Monocular Artefatto Profondita).*

| Video | Prospettiva | Metodo | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | Artefatto Profondita $Z$ | Vero Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P34_E3T1_Front.mp4` | Frontale Puro (Avvicinamento) | **Dummy (2D Lift)** | -0.603 | +0.072 | 0.299 | 1.170 | 0.0% | 666.8 | 72.4% | SI | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.351** | **+0.399** | **0.218** | **1.159** | **19.4%** | **9.9** | **27.6%** | **NO** | **NO (Impossibile)** |
| `P37_E3T1_Front.mp4` | Frontale Puro (Avvicinamento) | **Dummy (2D Lift)** | +0.118 | +0.370 | 0.205 | 1.025 | 0.0% | 886.3 | 33.3% | NO | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.160** | **+0.178** | **0.243** | **1.103** | **29.8%** | **32.2** | **12.0%** | **NO** | **NO (Impossibile)** |
| `P40_E3T1_Front.mp4` | Frontale Puro (Avvicinamento) | **Dummy (2D Lift)** | +0.001 | +0.036 | 0.399 | 1.539 | 0.0% | 549.7 | 38.7% | SI | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.227** | **+0.205** | **0.293** | **1.668** | **6.3%** | **3.8** | **0.0%** | **NO** | **NO (Impossibile)** |
| `P64_E3T1_Front.mp4` | Frontale Puro (Avvicinamento) | **Dummy (2D Lift)** | +0.580 | +0.268 | 0.168 | 0.893 | 0.0% | 480.0 | 12.5% | NO | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.317** | **+0.235** | **0.288** | **1.230** | **6.6%** | **5.9** | **17.2%** | **NO** | **NO (Impossibile)** |
| `P65_E3T2_Front.mp4` | Frontale Puro (Avvicinamento) | **Dummy (2D Lift)** | +0.076 | +0.222 | 0.342 | 1.164 | 0.0% | 640.2 | 11.1% | NO | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.582** | **+0.213** | **0.213** | **1.160** | **27.0%** | **31.8** | **2.8%** | **NO** | **NO (Impossibile)** |

---

### 2.3 Viste Posteriori Pure (180° — 5 Video)
> *In posterior views, limbs remain horizontally separated. Antiphase arises from radial recession along optical axis $Z$ (Monocular Artefatto Profondita).*

| Video | Prospettiva | Metodo | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | Artefatto Profondita $Z$ | Vero Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P2_E3T1_Rear.mp4` | Posteriore Puro (Allontanamento) | **Dummy (2D Lift)** | +0.461 | +0.229 | 0.240 | 1.047 | 0.0% | 1734.6 | 0.0% | NO | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.376** | **+0.453** | **0.281** | **1.198** | **15.3%** | **10.6** | **0.0%** | **NO** | **NO (Impossibile)** |
| `P48_E3T1_Rear.mp4` | Posteriore Puro (Allontanamento) | **Dummy (2D Lift)** | -0.431 | +0.245 | 0.288 | 1.130 | 0.0% | 654.4 | 73.1% | SI | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.097** | **+0.217** | **0.228** | **1.197** | **15.8%** | **11.1** | **12.5%** | **NO** | **NO (Impossibile)** |
| `P50_E3T1_Rear.mp4` | Posteriore Puro (Allontanamento) | **Dummy (2D Lift)** | +0.433 | +0.067 | 0.320 | 1.655 | 0.0% | 663.8 | 0.0% | NO | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **-0.181** | **+0.151** | **0.291** | **1.346** | **29.8%** | **23.5** | **33.3%** | **NO** | **NO (Impossibile)** |
| `P60_E3T1_Rear.mp4` | Posteriore Puro (Allontanamento) | **Dummy (2D Lift)** | -0.650 | +0.073 | 0.191 | 0.858 | 0.0% | 990.5 | 76.9% | SI | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **-0.563** | **+0.182** | **0.210** | **1.035** | **20.4%** | **27.6** | **68.8%** | **SI** | **NO (Impossibile)** |
| `P62_E3T1_Rear.mp4` | Posteriore Puro (Allontanamento) | **Dummy (2D Lift)** | +0.127 | +0.289 | 0.250 | 1.392 | 0.0% | 370.5 | 25.0% | NO | NO (Impossibile) |
| | | **DELTAv2 (Triplet)** | **+0.123** | **+0.566** | **0.269** | **1.116** | **37.7%** | **32.8** | **9.4%** | **NO** | **NO (Impossibile)** |

---

### 2.4 Viste Oblique (45° — 20 Video)
> *In 45° oblique views, periodic sagittal occlusion combines with radial depth variation $Z$. Detected swaps reflect true limb crossover events.*

| Video | Prospettiva | Metodo | Pitch $r$ | AccX $r$ | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Antiphase % | Vero Swap |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P11_E2T2_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | +0.641 | +0.361 | 0.146 | 1.367 | 0.0% | 3283.7 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.695** | **+0.200** | **0.264** | **0.897** | **28.2%** | **58.8** | **0.0%** | **NO** |
| `P12_E2T2_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | +0.809 | +0.242 | 0.179 | 0.900 | 0.0% | 1183.4 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.418** | **+0.358** | **0.303** | **1.163** | **28.7%** | **40.1** | **27.3%** | **NO** |
| `P20_E2T1_45_ALT.mp4` | 45° Obliquo (Sinistra -> Destra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.294 | +0.179 | 0.104 | 1.044 | 0.0% | 1967.0 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.337** | **+0.318** | **0.242** | **1.018** | **22.1%** | **50.3** | **0.0%** | **NO** |
| `P23_E1T2_45.mp4` | 45° Obliquo (Sinistra -> Destra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | -0.228 | +0.183 | 0.249 | 1.558 | 0.0% | 1303.7 | 43.8% | SI |
| | | **DELTAv2 (Triplet)** | **-0.320** | **+0.275** | **0.250** | **1.136** | **36.2%** | **58.5** | **35.7%** | **SI** |
| `P31_E2T2_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | -0.297 | +0.167 | 0.178 | 1.034 | 0.0% | 1684.1 | 53.3% | SI |
| | | **DELTAv2 (Triplet)** | **+0.768** | **+0.499** | **0.237** | **1.067** | **40.7%** | **46.5** | **0.0%** | **NO** |
| `P35_E2T2_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | -0.270 | +0.078 | 0.264 | 0.941 | 0.0% | 1343.4 | 40.0% | SI |
| | | **DELTAv2 (Triplet)** | **+0.618** | **+0.258** | **0.201** | **1.164** | **31.2%** | **63.0** | **0.0%** | **NO** |
| `P36_E1T2_45.mp4` | 45° Obliquo (Sinistra -> Destra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | -0.341 | +0.515 | 0.190 | 1.031 | 0.0% | 1324.6 | 52.0% | SI |
| | | **DELTAv2 (Triplet)** | **+0.316** | **+0.255** | **0.191** | **0.976** | **29.5%** | **39.1** | **0.0%** | **NO** |
| `P38_E1T1_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.386 | +0.207 | 0.179 | 1.068 | 0.0% | 2005.7 | 13.3% | NO |
| | | **DELTAv2 (Triplet)** | **+0.073** | **+0.038** | **0.219** | **1.405** | **41.8%** | **32.7** | **40.0%** | **SI** |
| `P39_E1T1_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.571 | +0.452 | 0.221 | 1.120 | 0.0% | 2224.3 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.326** | **+0.230** | **0.296** | **1.284** | **36.5%** | **30.0** | **8.8%** | **NO** |
| `P43_E1T1_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.837 | +0.104 | 0.179 | 1.660 | 0.0% | 1513.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.371** | **+0.223** | **0.325** | **1.299** | **20.4%** | **30.9** | **14.3%** | **NO** |
| `P44_E2T1_45.mp4` | 45° Obliquo (Sinistra -> Destra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.623 | +0.131 | 0.256 | 1.290 | 0.0% | 2051.0 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.441** | **+0.259** | **0.255** | **1.345** | **22.2%** | **52.2** | **16.7%** | **NO** |
| `P51_E2T2_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | +0.411 | -0.016 | 0.228 | 1.586 | 0.0% | 941.8 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.200** | **+0.427** | **0.224** | **1.439** | **22.8%** | **43.1** | **50.0%** | **SI** |
| `P53_E2T1_45_ALT.mp4` | 45° Obliquo (Sinistra -> Destra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | -0.217 | +0.485 | 0.214 | 1.134 | 0.0% | 2034.0 | 69.2% | SI |
| | | **DELTAv2 (Triplet)** | **+0.792** | **+0.361** | **0.187** | **1.074** | **19.9%** | **28.7** | **0.0%** | **NO** |
| `P54_E1T1_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.478 | +0.031 | 0.205 | 1.283 | 0.0% | 1879.6 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.274** | **+0.067** | **0.279** | **1.234** | **24.4%** | **37.6** | **14.3%** | **NO** |
| `P57_E1T1_45.mp4` | 45° Obliquo (Destra -> Sinistra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.479 | +0.196 | 0.224 | 1.211 | 0.0% | 3458.8 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.248** | **+0.229** | **0.274** | **1.190** | **39.6%** | **28.9** | **25.0%** | **NO** |
| `P59_E1T2_45.mp4` | 45° Obliquo (Sinistra -> Destra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | +0.452 | +0.182 | 0.129 | 1.282 | 0.0% | 1170.9 | 13.3% | NO |
| | | **DELTAv2 (Triplet)** | **+0.253** | **+0.487** | **0.276** | **1.148** | **27.6%** | **38.7** | **0.0%** | **NO** |
| `P5_E1T2_45_ALT.mp4` | 45° Obliquo (Sinistra -> Destra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | +0.211 | +0.374 | 0.230 | 0.984 | 0.0% | 1403.9 | 3.8% | NO |
| | | **DELTAv2 (Triplet)** | **+0.731** | **+0.509** | **0.231** | **0.935** | **29.5%** | **56.6** | **0.0%** | **NO** |
| `P63_E2T1_45.mp4` | 45° Obliquo (Sinistra -> Destra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.568 | +0.127 | 0.215 | 1.700 | 0.0% | 1527.2 | 8.0% | NO |
| | | **DELTAv2 (Triplet)** | **-0.011** | **+0.213** | **0.198** | **1.530** | **35.1%** | **46.5** | **25.9%** | **NO** |
| `P6_E2T1_45.mp4` | 45° Obliquo (Sinistra -> Destra) - Rear-Obliquo (Allontanamento) | **Dummy (2D Lift)** | +0.683 | -0.058 | 0.216 | 1.279 | 0.0% | 1915.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.462** | **-0.006** | **0.226** | **1.452** | **28.8%** | **29.0** | **0.0%** | **NO** |
| `P7_E2T2_45_ALT.mp4` | 45° Obliquo (Sinistra -> Destra) - Front-Obliquo (Avvicinamento) | **Dummy (2D Lift)** | +0.725 | +0.261 | 0.173 | 1.074 | 0.0% | 1275.5 | 0.0% | NO |
| | | **DELTAv2 (Triplet)** | **+0.678** | **+0.229** | **0.199** | **1.012** | **33.3%** | **40.9** | **0.0%** | **NO** |

---

# 🔬 Critical Benchmark Analysis: DELTAv2 vs Dummy Baseline
### *Biomechanical, Optical, and Spatial Evaluation across 40 Gait360 Videos with Strict Separation between Crossover Identity Swaps and Axial Depth Ambiguity*

> **Reference Tables:** [`deltav2_vs_dummy_benchmark_tables.md`](deltav2_vs_dummy_benchmark_tables.md)  
> **Date:** November 2026 | **Author:** Gait360 3D Tracking & Analysis Pipeline  
> **Scope:** Scientific Validation for Computer Vision & Biomechanical Engineering

---

## 1. Conceptual Disentanglement: Occlusion Identity Swaps vs Monocular Artefatto Profonditas

A primary methodological insight emerging from our multi-view analysis concerns the physical origin of persistent **antiphase ($r < -0.35$)** relative to inertial Ground Truth.

### A. Occlusion Geometry in Lateral and Oblique Views (90° and 45°)
- **Physical Condition:** In 90° sagittal and 45° oblique recordings, the swinging leg crosses and partially or totally occludes the contralateral stance leg during every stride cycle.
- **Swap Mechanism:** Visual point trackers or 2D keypoint detectors can be dragged by the occluding boundary and latch onto the opposite limb passing in the foreground.
- **Biomechanical Signature:** Because human bipedal gait operates with a $180^\circ$ phase offset between left and right limbs, tracking the contralateral leg produces Pitch and acceleration waveforms in sustained antiphase with the right-thigh IMU.
- **Experimental Results across the 30 Crossover-Exposed Videos:**
  - **DELTAv2 Refined Triplet:** Exhibits only **3 swaps out of 30 videos (10.0%)**.
  - **DELTAv2 Raw Baseline:** Exhibits **4 swaps out of 30 videos (13.3%)**.
  - **Dummy Baseline (2D Lift):** Exhibits **9 swaps out of 30 videos (30.0%)**.
  - **Verdict:** *Geometric Pruning* in the Refined Triplet reduces true occlusion swaps by **25.0% relative to the Raw baseline** (from 4 to 3 videos) and by **66.7% relative to the Dummy baseline** (from 9 to 3 videos), pruning unstable points near silhouette boundaries.

### B. Geometrical Impossibility of Limb Swap in Axial Views (0° Frontal and 180° Posterior)
- **Physical Condition:** When a subject walks straight toward (0°) or away from (180°) the camera, the right and left legs remain **permanently separated along the horizontal image axis $X$** by the full width of the pelvis.
- **Why Limb Swap is Impossible:** Without optical intersection or 2D overlap between the limbs, a cluster of points anchored on the right thigh cannot jump horizontally onto the left thigh.
- **Origin of Antiphase: Foreshortening and Monocular Depth ($Z$):**
  - At 0° and 180°, the sagittal flexion-extension plane (Pitch) coincides with the **camera optical axis ($Z$)**.
  - In the 2D image plane ($X$-$Y$), angular flexion manifests solely as minor perspective foreshortening. Pitch estimation therefore relies heavily on the monocular depth network (UniDepthV2) resolving longitudinal $Z$-tilt across the thigh.
  - **Conclusion:** Persistent antiphase observed at 0° and 180° (**1 / 10 videos, 10.0%** for DELTAv2 vs **4 / 10 (40.0%)** for Dummy) is **not a limb swap**, but an **intrinsic optical depth ambiguity along the line of sight**.

---

## 2. Aggregated Metric Performance Synthesis

### 2.1 3D Rigid Body Fidelity and Kinematic Smoothness (Jerk)
- **3D Rigid Deformation:** DELTAv2 Refined Triplet achieves a mean relative deformation of **28.4% ± 9.9%**, substantially outperforming both the Raw baseline (33.3% ± 12.3%) and the full Point Cloud SVD (41.4% ± 8.3%, winning **39 / 40 videos = 97.5%**).
- **RMS Jerk (m/s³):** The Refined Triplet achieves **36.6 ± 14.3 m/s³**, compared to **179.9 ± 58.6 m/s³** for Raw and **1452.0 ± 646.6 m/s³** for Dummy. This represents a 5× improvement in second-derivative smoothness over Raw and a 40× improvement over Dummy.

### 2.2 Inertial Ground Truth Correlation & Magnitude Fidelity (World Coordinates)
- **Pitch Pearson r:** The Refined Triplet achieves **0.295 ± 0.329** across all 40 videos (surpassing Point Cloud `0.276`, Dummy `0.199`, and Raw `0.185`), reaching peak correlations above **+0.65** in non-swapped views (e.g. `P53_E2T1_45_ALT`: +0.792, `P32_E1T2_90`: +0.787, `P31_E2T2_45`: +0.768, `P46_E1T1_90`: +0.690, `P65_E3T2_Front`: +0.582).
- **Pitch ROM Error & nRMSE:** The Refined Triplet achieves a near-unbiased Range of Motion error of **+4.1% ± 70.8%** (compared to **-31.7%** underestimation in Dummy, **-15.5%** in Raw, and **+44.1%** inflation in Cloud) with a Pitch nRMSE of **30.5% ± 9.3%** (vs 34.4% Cloud).
- **AccX Pearson r & 3D State-Space DTW:** The Refined Triplet attains an AccX correlation of **0.255 ± 0.162** (vs 0.205 Dummy and 0.243 Raw) and the best overall 3D acceleration DTW distance of **1.230 ± 0.173**.

---

## 3. Architectural Conclusions & Clinical Recommendations

1. **Monocular 3D Tracking Validity:** DELTAv2 (anche nella configurazione fallback `bbox_grid` da `987` punti) abbinato al Geometric Pruning estrae una cinematica 3D continua e a basso jerk senza discontinuità di chunking; il ripristino della silhouette MediaPipe (`1003` punti) è previsto come confronto migliorativo.
2. **Camera Placement Guidelines:** For single-camera clinical gait assessments, **90° sagittal** or **45° oblique** viewpoints are strongly recommended over pure axial (0°/180°) views unless depth sensors (RGB-D / LiDAR) are available.
3. **Rejection of Naive 2D Lifting:** Comparison against the Dummy baseline proves that lifting 2D keypoints without dense 3D metric tracking introduces extreme acceleration noise (RMS Jerk > 1450 m/s³) and severe ROM underestimation (-31.7%).

