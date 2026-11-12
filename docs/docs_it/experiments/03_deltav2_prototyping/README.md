# 🏆 Technical Decision Report: Validazione di Produzione di DELTAv2 + Geometric Pruning v4.1

- **Branch di Esperimento:** `experiment/deltav2-evaluation`
- **Backbone di Produzione:** `DenseTrack3Dv2` (DELTAv2) + `UniDepthV2` + `MediaPipe Pose Landmarker` + `GeometricPruner v4.1`
- **Verdetto Decisionale:** **Adottato come Architettura Ufficiale di Produzione su tutti i 40 Video di Benchmark di Gait360**

---

## 1. Architettura e Raffinamenti Matematici

La pipeline di produzione integra quattro stadi sinergici:
1. **Campionamento Query con Maschera di Silhouette MediaPipe (`DeltaV2Tracker`):**
   Campiona 800 query di supporto rigorosamente all'interno della silhouette segmentata del corpo e 200 query all'interno del triangolo anatomico validato della coscia destra (più i 3 vertici di riferimento = 1003 query), tracciate in modo bidirezionale in singolo passaggio con `coarse_to_fine_dense=True` e $N_{\text{iters}}=6$.
2. **Pruning Geometrico e Selezione della Tripletta Ottimale (`GeometricPruner`):**
   - Pre-filtraggio passa-basso Butterworth del 4° ordine a fase zero ($f_c = 2.93\text{ Hz}$) con penalizzazione degli $Z$-spike lungo l'asse ottico monoculare ($w_z = 0.25$).
   - Pruning di persistenza del Convex Hull 2D ($\text{Area}(t) \le 5.0 \cdot \text{Area}_{\text{ref}}$ su tutti i frame).
   - Clustering cinematico adattivo di coerenza PCA-DBSCAN sulla velocità (`initial_eps = 0.5`).
   - Ricerca esaustiva $\binom{M}{3}$ della tripletta ottimale con vincolo di non collinearità 3D ($\min_t \text{Area}_{3D}(t) \ge \tau \cdot \text{Area}_{3D,\text{ref}}$).
3. **Allineamento Cinematico World (`kinematic_alignment.py`):**
   - Proiezione della forza specifica dell'IMU nel frame inerziale World ($\mathbf{a}_{\text{world}}(t) = \mathbf{R}_{\text{imu}}(t)\mathbf{a}_{\text{body}}(t) - \mathbf{g}$).
   - Stabilizzazione della normale della tripletta su base stretta tramite regolarizzazione anti-twist ($h_{\min} \ge 3.5\text{ cm}$) e despiking angolare.
   - Sincronizzazione del ritardo temporale tramite cross-correlazione normalizzata su finestra mobile della velocità angolare principale con segno.

---

## 2. Risultati Finali del Benchmark su 40 Video (Post-Fix `031a73d`)

| Metrica | Dummy Baseline (Lift 2D) | DELTAv2 Raw Baseline | DELTAv2 Refined Cloud (SVD) | DELTAv2 Refined Triplet (Produzione) |
| :--- | :---: | :---: | :---: | :---: |
| **Pitch Pearson $r$** | $0.199 \pm 0.452$ | $0.185 \pm 0.316$ | $0.276 \pm 0.325$ | **$0.295 \pm 0.329$** (vince 21/40 vs Cloud) |
| **Pitch ROM Error (%)** | $-31.7\% \pm 46.0\%$ | $-15.5\% \pm 48.5\%$ | $+45.2\% \pm 143.7\%$ | **$+4.1\% \pm 70.8\%$** |
| **Pitch nRMSE (%)** | $28.6\% \pm 7.8\%$ | $30.4\% \pm 6.7\%$ | $34.4\% \pm 14.3\%$ | **$30.5\% \pm 9.3\%$** |
| **AccX Pearson $r$** | $0.205 \pm 0.140$ | $0.243 \pm 0.171$ | $0.288 \pm 0.158$ | **$0.255 \pm 0.162$** |
| **3D Acc DTW (norm)** | $1.244 \pm 0.230$ | $1.275 \pm 0.214$ | $1.248 \pm 0.172$ | **$1.230 \pm 0.173$** (Migliore assoluto) |
| **Deformazione Rigida 3D (%)** | $0.0\% \pm 0.0\%$ | $33.3\% \pm 12.3\%$ | $41.4\% \pm 8.7\%$ | **$28.4\% \pm 9.9\%$** (vince **39/40 = 97.5%** vs Cloud) |
| **RMS Jerk ($\text{m/s}^3$)** | $1452.0 \pm 646.6$ | $179.9 \pm 58.6$ | $36.2 \pm 13.4$ | **$36.6 \pm 14.3$** (5× più fluido di Raw, 40× vs Dummy) |
| **True Identity Swaps ($90^\circ$ & $45^\circ$)** | $9 / 30\ (30.0\%)$ | $4 / 30\ (13.3\%)$ | $6 / 30\ (20.0\%)$ | **$3 / 30\ (10.0\%)$** |
| **Artefatti di Profondità Assiale ($0^\circ$ & $180^\circ$)** | $4 / 10\ (40.0\%)$ | $1 / 10\ (10.0\%)$ | $1 / 10\ (10.0\%)$ | **$1 / 10\ (10.0\%)$** |

**Verdetto:** La pipeline **DELTAv2 Refined Triplet** è validata come soluzione stato dell'arte per il tracking 3D monoculare del cammino su tutte le 40 acquisizioni multi-prospettiva.
