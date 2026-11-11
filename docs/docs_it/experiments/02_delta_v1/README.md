# 🔬 Technical Decision Report: Valutazione Esplorativa di DELTA v1 (Single-Pass & Chunked SVD)

- **Branch di Esperimento:** `experiment/delta-v1`
- **Modelli Valutati:** `DenseTrack3D` (DELTA v1) + `UniDepthV2` (`lpiccinelli/unidepth-v2-vitl14`)
- **Verdetto Decisionale:** **Superato da DELTAv2 (`DenseTrack3Dv2` + Mascheramento Silhouette MediaPipe)**

---

## 1. Ipotesi Sperimentale e Architettura

Per superare il limite di memoria a 90 frame e l'eccessiva elasticità da bundle-adjustment riscontrati in SpatialTrackerV2 + VGGT, **DELTA v1 (`DenseTrack3D`)** accoppiato alle mappe di profondità metrica di **UniDepthV2** è stato valutato in due modalità operative (`experiments/delta_test/`):
1. **Armatura di Contesto Anatomico (`delta_test.py`):** Sostituzione delle griglie casuali di sfondo con una struttura anatomica di 1003 punti composta da 3 vertici target della coscia, 200 punti interni al triangolo coscia, 300 punti bacino anti-swap e 500 punti contesto sul tronco.
2. **Cucitura a Blocchi via Kabsch-Umeyama SVD (`delta_test_chunked.py`):** Suddivisione dei video lunghi in blocchi da 60 frame con sovrapposizione di 25 frame, allineando i blocchi consecutivi tramite registrazione SVD rigida ($\mathbf{R} = \mathbf{V}\mathbf{U}^T, \mathbf{T} = \bar{\mathbf{Q}} - \mathbf{R}\bar{\mathbf{P}}$).

---

## 2. Riscontri Quantitativi (Benchmark su Finestra Comune di 6 Video)

| Metrica | DELTA v1 (Chunked SVD - Comune) | DELTAv2 (Bounding Box - Comune) | DELTAv2 (MediaPipe - Comune) |
| :--- | :---: | :---: | :---: |
| **AccX Pearson $r$** | $+0.482 \pm 0.117$ | $+0.497 \pm 0.218$ | $+0.449 \pm 0.255$ |
| **Pitch Pearson $r$** | $+0.535 \pm 0.335$ | $+0.232 \pm 0.298$ | $+0.430 \pm 0.398$ |
| **Pitch DTW (norm)** | $0.235 \pm 0.053$ | $0.312 \pm 0.099$ | **$0.221 \pm 0.078$** |
| **Deformazione Rigida 3D (%)** | $46.4\% \pm 31.0\%$ | $24.5\% \pm 7.8\%$ | **$19.6\% \pm 7.6\%$** |
| **RMS Jerk ($\text{m/s}^3$)** | $476.3 \pm 133.8$ | $176.7 \pm 46.7$ | **$30.7 \pm 13.0$** |

---

## 3. Motivazioni Tecniche per il Passaggio a DELTAv2

1. **Discontinuità di Accelerazione ai Bordi dei Blocchi (Jerk RMS = 476.3 m/s³):**
   La cucitura di finestre temporali da 60 frame tramite Kabsch-Umeyama SVD introduce minuscoli gradini sub-millimetrici alle giunzioni che esplodono sotto la derivazione seconda temporale, aumentando il Jerk RMS di ben **15.5 volte** rispetto al passaggio singolo di DELTAv2 (`476.3` vs `30.7 m/s³`).
2. **Contaminazione dello Sfondo con Bounding Box Rettangolari:**
   I rettangoli rigidi per bacino e tronco fuoriescono dalla silhouette umana durante l'oscillazione degli arti, campionando pixel statici delle pareti che distorcono i volumi di correlazione.
3. **Drift Cumulativo con Alto Numero di Iterazioni ($N_{\text{iters}}=16$):**
   L'impiego di 12–16 iterazioni di raffinamento in DELTA v1 causa overfitting sulle pieghe transitorie dei vestiti. Il passaggio a **DELTAv2 (`DenseTrack3Dv2`)** con `coarse_to_fine_dense=True`, $N_{\text{iters}}=6$ e la **segmentazione della silhouette tramite MediaPipe Pose Landmarker** consente un'inferenza fluida in singolo passaggio sull'intero video (240+ frame), raggiungendo il **19.6%** di deformazione e **30.7 m/s³** di jerk.
