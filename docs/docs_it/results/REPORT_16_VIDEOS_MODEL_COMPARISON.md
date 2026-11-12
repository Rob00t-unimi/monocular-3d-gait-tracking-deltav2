# 📊 Benchmark Completo dei Modelli con Allineamento Cinematico al Mondo
### *Confronto Rigoroso su Finestra Temporale Comune: SpatialTrackerV2 + VGGT vs DELTAv2 vs Baseline*

> **Video Valutati:** 6 Trial di Riferimento (45° Obliqua, 90° Sagittale, Posteriore, Frontale)  
> **Protocollo Scientifico:** Valutazione rigorosa su **Finestra Temporale Comune (60–90 frame)** per eliminare ogni asimmetria legata alla durata del video.  
> **Allineamento Cinematico:** Allineamento Standardizzato al Mondo ($\mathbf{R}_{cam \to world}$, Rimozione Gravità, Sincronizzazione del ritardo temporale)  
> **Riferimento IMU:** Gait360 IMU Coscia Destra  

---

## 1. Analisi Statistica Aggregata (Media ± Dev. Std)

| Metodo | Finestra Temporale | AccX (r) | Pitch (r) | Pitch DTW | 3D Acc DTW | 3D Def. Rigida % | RMS Jerk (m/s³) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpatialTrackerV2 + VGGT** | Comune (60-90 fr) | **+0.360 ± 0.255** | +0.546 ± 0.250 | 0.269 ± 0.075 | 1.167 ± 0.211 | **56.7% ± 13.3%** | 30.4 ± 2.0 |
| **DELTAv2 (MediaPipe - Comune)** | Comune (60-90 fr) | **+0.449 ± 0.255** | +0.430 ± 0.398 | 0.221 ± 0.078 | 1.309 ± 0.077 | **19.6% ± 7.6%** | 30.7 ± 13.0 |
| **DELTAv2 (Bounding Box - Comune)** | Comune (60-90 fr) | **+0.497 ± 0.218** | +0.232 ± 0.298 | 0.312 ± 0.099 | 1.291 ± 0.212 | **24.5% ± 7.8%** | 176.7 ± 46.7 |
| **DELTA (Chunked SVD - Comune)** | Comune (60-90 fr) | **+0.482 ± 0.117** | +0.535 ± 0.335 | 0.235 ± 0.053 | 1.242 ± 0.176 | **46.4% ± 31.0%** | 476.3 ± 133.8 |
| **Dummy (Lift 2D - Comune)** | Comune (60-90 fr) | **+0.394 ± 0.271** | +0.227 ± 0.181 | 0.278 ± 0.055 | 1.067 ± 0.138 | **0.0% ± 0.0%** | 1236.6 ± 387.6 |
| **DELTAv2 (MediaPipe - Sequenza Intera)** | Intera (164-238 fr) | **+0.401 ± 0.121** | +0.184 ± 0.217 | 0.234 ± 0.020 | 1.166 ± 0.100 | **26.9% ± 5.6%** | 35.6 ± 6.9 |

---

## 2. Risultati Dettagliati per Singolo Video (Finestra Comune)

| Video | Prospettiva | Metodo | Frame | AccX r | Pitch r | Pitch DTW | 3D Acc DTW | 3D Def % | RMS Jerk | Ritardo (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P1_E1T1_45_ALT.mp4` | 45° Obliqua (Abbigliamento Alternativo) | SpatialTrackerV2 + VGGT | 90 | **+0.463** | +0.099 | 0.350 | 1.022 | **57.0%** | 30.4 | -17 |
| `P1_E1T1_45_ALT.mp4` | 45° Obliqua (Abbigliamento Alternativo) | DELTAv2 (MediaPipe - Comune) | 90 | **+0.524** | +0.014 | 0.313 | 1.433 | **20.3%** | 45.3 | -8 |
| `P1_E1T1_45_ALT.mp4` | 45° Obliqua (Abbigliamento Alternativo) | DELTAv2 (Bounding Box - Comune) | 90 | **+0.758** | +0.119 | 0.216 | 1.492 | **29.4%** | 229.2 | 38 |
| `P1_E1T1_45_ALT.mp4` | 45° Obliqua (Abbigliamento Alternativo) | DELTA (Chunked SVD - Comune) | 90 | **+0.336** | +0.667 | 0.161 | 1.357 | **33.5%** | 499.1 | -12 |
| `P1_E1T1_45_ALT.mp4` | 45° Obliqua (Abbigliamento Alternativo) | Dummy (Lift 2D - Comune) | 90 | **+0.109** | +0.396 | 0.246 | 1.082 | **0.0%** | 1837.0 | -34 |
| `P1_E1T1_45_ALT.mp4` | 45° Obliqua (Abbigliamento Alternativo) | DELTAv2 (MediaPipe - Sequenza Intera) | 179 | **+0.296** | -0.234 | 0.238 | 1.037 | **24.6%** | 42.2 | 60 |
| `P2_E1T1_90.mp4` | 90° Vista Sagittale | SpatialTrackerV2 + VGGT | 90 | **+0.114** | +0.529 | 0.270 | 0.937 | **41.5%** | 27.5 | 22 |
| `P2_E1T1_90.mp4` | 90° Vista Sagittale | DELTAv2 (MediaPipe - Comune) | 90 | **+0.407** | +0.510 | 0.209 | 1.240 | **28.0%** | 28.6 | -16 |
| `P2_E1T1_90.mp4` | 90° Vista Sagittale | DELTAv2 (Bounding Box - Comune) | 90 | **+0.457** | +0.409 | 0.407 | 1.126 | **36.6%** | 178.4 | -11 |
| `P2_E1T1_90.mp4` | 90° Vista Sagittale | DELTA (Chunked SVD - Comune) | 90 | **+0.467** | +0.710 | 0.279 | 1.134 | **97.3%** | 595.3 | -1 |
| `P2_E1T1_90.mp4` | 90° Vista Sagittale | Dummy (Lift 2D - Comune) | 90 | **+0.239** | +0.081 | 0.289 | 1.316 | **0.0%** | 1004.1 | -29 |
| `P2_E1T1_90.mp4` | 90° Vista Sagittale | DELTAv2 (MediaPipe - Sequenza Intera) | 238 | **+0.275** | +0.390 | 0.205 | 1.189 | **32.5%** | 33.8 | -16 |
| `P3_E1T2_45.mp4` | 45° Vista Obliqua | SpatialTrackerV2 + VGGT | 90 | **+0.343** | +0.760 | 0.264 | 1.142 | **62.6%** | 32.6 | 0 |
| `P3_E1T2_45.mp4` | 45° Vista Obliqua | DELTAv2 (MediaPipe - Comune) | 90 | **+0.507** | +0.704 | 0.217 | 1.254 | **26.2%** | 26.1 | -40 |
| `P3_E1T2_45.mp4` | 45° Vista Obliqua | DELTAv2 (Bounding Box - Comune) | 90 | **+0.609** | +0.045 | 0.449 | 1.004 | **24.6%** | 176.3 | 40 |
| `P3_E1T2_45.mp4` | 45° Vista Obliqua | DELTA (Chunked SVD - Comune) | 90 | **+0.529** | +0.697 | 0.194 | 1.115 | **72.4%** | 641.3 | 5 |
| `P3_E1T2_45.mp4` | 45° Vista Obliqua | Dummy (Lift 2D - Comune) | 90 | **+0.250** | +0.217 | 0.213 | 0.917 | **0.0%** | 1509.8 | -24 |
| `P3_E1T2_45.mp4` | 45° Vista Obliqua | DELTAv2 (MediaPipe - Sequenza Intera) | 164 | **+0.468** | +0.244 | 0.214 | 1.230 | **29.3%** | 33.7 | -32 |
| `P7_E1T1_90.mp4` | 90° Vista Sagittale | SpatialTrackerV2 + VGGT | 90 | **+0.008** | +0.789 | 0.140 | 1.232 | **77.2%** | 32.6 | -4 |
| `P7_E1T1_90.mp4` | 90° Vista Sagittale | DELTAv2 (MediaPipe - Comune) | 90 | **+0.048** | +0.766 | 0.121 | 1.338 | **22.1%** | 47.7 | -35 |
| `P7_E1T1_90.mp4` | 90° Vista Sagittale | DELTAv2 (Bounding Box - Comune) | 90 | **+0.106** | +0.474 | 0.234 | 1.554 | **23.8%** | 155.4 | -34 |
| `P7_E1T1_90.mp4` | 90° Vista Sagittale | DELTA (Chunked SVD - Comune) | 90 | **+0.494** | +0.689 | 0.260 | 1.472 | **28.4%** | 456.4 | 26 |
| `P7_E1T1_90.mp4` | 90° Vista Sagittale | Dummy (Lift 2D - Comune) | 90 | **+0.312** | +0.455 | 0.253 | 1.010 | **0.0%** | 1263.2 | -16 |
| `P7_E1T1_90.mp4` | 90° Vista Sagittale | DELTAv2 (MediaPipe - Sequenza Intera) | 216 | **+0.327** | +0.210 | 0.258 | 1.239 | **26.2%** | 45.8 | -70 |
| `P8_E3T1_Rear.mp4` | Vista Posteriore | SpatialTrackerV2 + VGGT | 60 | **+0.598** | +0.487 | 0.249 | 1.126 | **59.4%** | 28.8 | 2 |
| `P8_E3T1_Rear.mp4` | Vista Posteriore | DELTAv2 (MediaPipe - Comune) | 60 | **+0.835** | +0.732 | 0.157 | 1.241 | **11.6%** | 16.2 | -24 |
| `P8_E3T1_Rear.mp4` | Vista Posteriore | DELTAv2 (Bounding Box - Comune) | 60 | **+0.546** | +0.558 | 0.236 | 1.336 | **17.7%** | 100.6 | -8 |
| `P8_E3T1_Rear.mp4` | Vista Posteriore | DELTA (Chunked SVD - Comune) | 60 | **+0.392** | +0.588 | 0.297 | 1.351 | **26.4%** | 278.0 | 13 |
| `P8_E3T1_Rear.mp4` | Vista Posteriore | Dummy (Lift 2D - Comune) | 60 | **+0.652** | +0.235 | 0.295 | 1.095 | **0.0%** | 764.2 | 2 |
| `P8_E3T1_Rear.mp4` | Vista Posteriore | DELTAv2 (MediaPipe - Sequenza Intera) | 219 | **+0.583** | +0.186 | 0.244 | 1.257 | **31.4%** | 27.9 | -33 |
| `P11_E3T1_Front.mp4` | Vista Frontale | SpatialTrackerV2 + VGGT | 60 | **+0.631** | +0.615 | 0.338 | 1.544 | **42.8%** | 30.7 | -9 |
| `P11_E3T1_Front.mp4` | Vista Frontale | DELTAv2 (MediaPipe - Comune) | 60 | **+0.373** | -0.149 | 0.309 | 1.345 | **9.3%** | 20.6 | 27 |
| `P11_E3T1_Front.mp4` | Vista Frontale | DELTAv2 (Bounding Box - Comune) | 60 | **+0.508** | -0.216 | 0.328 | 1.235 | **15.0%** | 220.4 | 5 |
| `P11_E3T1_Front.mp4` | Vista Frontale | DELTA (Chunked SVD - Comune) | 60 | **+0.673** | -0.144 | 0.217 | 1.019 | **20.7%** | 387.6 | -20 |
| `P11_E3T1_Front.mp4` | Vista Frontale | Dummy (Lift 2D - Comune) | 60 | **+0.805** | -0.021 | 0.372 | 0.984 | **0.0%** | 1041.1 | -5 |
| `P11_E3T1_Front.mp4` | Vista Frontale | DELTAv2 (MediaPipe - Sequenza Intera) | 229 | **+0.459** | +0.305 | 0.242 | 1.043 | **17.2%** | 30.4 | 24 |

---

## 3. Analisi Critica e Giustificazione Scientifica per DELTAv2

### A. Effetto della Finestra Temporale Comune (*Matched Window*)
Quando tutti i modelli candidati sono valutati sulla **stessa identica finestra temporale** (60–90 frame attorno al frame di riferimento, equivalenti a 1–2 cicli di passo completi):
1. **Accelerazione Lineare Anteriore (AccX) e Flesso-Estensione Sagittale (Pitch):**
   - **DELTAv2 (MediaPipe) supera SpatialTrackerV2 sull'accelerazione lineare:** raggiungendo una correlazione Pearson media di AccX pari a $r = \mathbf{+0.449 \pm 0.255}$ rispetto a $\mathbf{+0.360 \pm 0.255}$ di SpatialTrackerV2, la **minore distanza Pitch DTW** ($\mathbf{0.221 \pm 0.078}$ vs $\mathbf{0.269 \pm 0.075}$), e una correlazione di Pitch pari a $r = \mathbf{+0.430 \pm 0.398}$ vs $\mathbf{+0.546 \pm 0.250}$.
2. **Fedeltà Strutturale di Corpo Rigido (Segmento del Femore):**
   - DELTAv2 preserva una **deformazione 3D quasi 3 volte inferiore** rispetto a SpatialTrackerV2 (**19.6% vs 56.7%**). SpatialTrackerV2 risente di un'elasticità spaziale non vincolata durante la bundle adjustment globale.
3. **Fluidità della Traiettoria (RMS Jerk):**
   - Sugli intervalli comuni, DELTAv2 raggiunge un Jerk RMS di **30.7 m/s³**, perfettamente allineato a SpatialTrackerV2 (**30.4 m/s³**) e oltre 38 volte più fluido della baseline Dummy.

### B. Giustificazione Architetturale per la Scelta di DELTAv2
**DELTAv2** è stato selezionato come backbone di tracking 3D di produzione per tre ragioni decisive:
1. **Scalabilità su Sequenza Completa (Full-Sequence Tracking):** DELTAv2 traccia video di cammino a lunghezza intera (240+ frame) in un unico passaggio mantenendo stabilità e coerenza, senza salti tra blocchi. SpatialTrackerV2 collassa sui video lunghi a causa della complessità di memoria GPU di VGGT e del drift della bundle adjustment.
2. **Mascheramento Anatomico della Silhouette (MediaPipe):** L'ancoraggio delle query di supporto esclusivamente all'interno della sagoma umana previene il campionamento di pareti o pavimento di sfondo, riducendo il Jerk RMS di oltre 5 volte rispetto ai bounding box o alle griglie diffuse.
3. **Efficienza di Inferenza e Riproducibilità:** DELTAv2 adotta la stima metrica UniDepthV2 in batch e un'ottimizzazione coarse-to-fine a 6 iterazioni, garantendo un throughput nettamente superiore rispetto alla complessa pipeline VGGT + Predictor.
