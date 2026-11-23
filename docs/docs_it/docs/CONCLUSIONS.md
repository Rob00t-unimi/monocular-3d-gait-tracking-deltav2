# 📊 Sintesi Esecutiva e Conclusioni del Benchmark: DELTAv2 vs. Dummy Baseline
### *Valutazione Quantitativa su 40 Video Gait360 con Validazione Ground-Truth Inerziale*

Questo documento fornisce una sintesi esecutiva dei risultati del benchmark su tutti i **40 video stratificati del cammino** (40 soggetti unici) del dataset pubblico **Gait360**, mettendo a confronto la pipeline proposta **DELTAv2 (Refined Triplet tramite Pruning Geometrico)** contro la **Dummy Baseline (Lifting 2D OpenPose)** e i sensori indossabili sincronizzati **WitMotion IMU**.

---

## 🚀 1. I 4 Vantaggi Quantitativi Chiave di DELTAv2

### 1. Fluidità Cinematica e Attenuazione del Rumore di Derivazione: **~40× Migliore**
* **Metrica:** **RMS Jerk** ($\text{m/s}^3$, derivata terza della posizione / derivata prima dell'accelerazione).
* **Dummy Baseline:** `1452.0 ± 646.6 m/s³`
* **DELTAv2 (Refined Triplet):** **`36.6 ± 14.3 m/s³`**
* **Verdetto:** **Riduzione del 97.5% del rumore di derivazione (~40× più fluido).**  
  Il lifting 2D soffre di quantizzazione a pixel discreti che esplode in picchi di accelerazione alla derivazione seconda. DELTAv2, grazie alla profondità metrica continua (UniDepthV2) e all'ottimizzazione temporale, genera traiettorie fisicamente plausibili e fluide.

### 2. Fedeltà d'Ampiezza Angolare (Range of Motion): **Recupero a Bias Quasi Nullo**
* **Metrica:** **Errore ROM Pitch %** (errore percentuale sull'escursione di flesso-estensione rispetto all'IMU, target ideale = $0\%$).
* **Dummy Baseline:** `-31.7% ± 46.0%`
* **DELTAv2 (Refined Triplet):** **`+4.1% ± 70.8%`**
* **Verdetto:** La baseline Dummy sottostima sistematicamente l'escursione articolare di quasi un terzo (`-31.7%`). DELTAv2 elimina questo bias sistematico (`+4.1%`), centrando l'escursione sui valori anatomici reali.

### 3. Robustezza alle Occlusioni (Scambi d'Arto Controlaterali): **3× Meno Fallimenti**
* **Metrica:** **Scambi d'Identità Veri** (fallimento per incrocio dell'arto controlaterale nelle viste a $90^\circ$ e $45^\circ$).
* **Dummy Baseline:** `30.0%` (9 / 30 video)
* **DELTAv2 (Refined Triplet):** **`10.0%`** (3 / 30 video)
* **Verdetto:** **Riduzione del 66.7% degli identity swap (affidabilità 3× superiore).**  
  Il campionamento denso coscia + contesto (griglia `bbox_grid` da `987` punti nel run primario `results/`; valutato anche a `40/40` con silhouette MediaPipe pura da `1003` punti in `results_mediapipe/`) abbinato al *Geometric Pruner* mantiene il tracciamento ancorato alla coscia corretta durante l'incrocio.

### 4. Correlazione della Forma d'Onda Sagittale: **+48% di Fedeltà**
* **Metrica:** **Pitch Pearson $r$** (correlazione lineare con il Pitch dell'IMU).
* **Dummy Baseline:** `0.199 ± 0.452`
* **DELTAv2 (Refined Triplet):** **`0.295 ± 0.329`** (media **$0.387$** sulle viste sagittali a $90^\circ$ con picchi tra **$+0.69$ e $+0.79$**)
* **Verdetto:** **Miglioramento del +48.2% nella correlazione media** con deviazione standard ridotta.

---

## 📊 2. Tabella Riepilogativa Comparativa (`bbox_grid`, N = 40)

| Dominio di Valutazione | Metrica / Parametro | Dummy Baseline (2D Lift) | DELTAv2 (Refined Triplet) | Verdetto Benchmark |
| :--- | :--- | :---: | :---: | :--- |
| **Dinamica (Fluidità)** | **RMS Jerk** ($\downarrow$) | $1452.0 \text{ m/s}^3$ | **$36.6 \text{ m/s}^3$** | **~40× più fluido** (elimina i picchi di derivazione) |
| **Ampiezza (ROM)** | **Errore ROM Pitch** ($0\%$) | $-31.7\%$ | **$+4.1\%$** | **Bias sistematico eliminato** |
| **Affidabilità (Occlusione)** | **Scambi d'Identità Veri ($90^\circ$/$45^\circ$)** ($\downarrow$) | $30.0\%$ (9/30) | **$10.0\%$ (3/30)** | **Swap ridotti del 66.7%** (3× più robusto) |
| **Stabilità Assiale** | **Artefatti Profondità Assiale ($0^\circ$/$180^\circ$)** ($\downarrow$) | $40.0\%$ (4/10) | **$10.0\%$ (1/10)** | **4× meno inversioni assiali** |
| **Forma d'Onda (Cinematica)** | **Pitch Pearson $r$** ($\uparrow$) | $0.199$ | **$0.295$** | **+48.2% di guadagno in correlazione** |
| **Spazio degli Stati 3D** | **3D Acc DTW (norm)** ($\downarrow$) | $1.244$ | **$1.230$** | **Minima distanza traiettoria 3D** |
| **Coerenza di Fase** | **Durata Antifase (%)** ($\downarrow$) | $22.3\%$ | **$13.5\%$** | **-39.5% di deriva in antifase** |

> **Nota di Trasparenza sulla Baseline Dummy:** Poiché Dummy usa un trasferimento rigido 2D a profondità costante ($Z_0 = 2.5\text{ m}$), la sua `Deformazione Rigida 3D` è tautologicamente `0.0%`, e il ritmo 2D ginocchio-anca produce un errore di cadenza (`11.7%` vs `24.1%`) e Pitch nRMSE (`28.6%` vs `30.5%`) inferiori, al prezzo di un Jerk 40× peggiore, 3× più identity swap, 4× più artefatti assiali e una compressione ROM del `-31.7%`.

---

*Report Dettagliati:*
- Report Completo 40 Video (`bbox_grid`): [`../results/REPORT_40_VIDEOS_BENCHMARK.md`](../results/REPORT_40_VIDEOS_BENCHMARK.md)
- Ablazione Tripletta vs Nuvola SVD: [`../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`](../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md)
- Confronto Multi-Modello Finestra Appaiata (SpatialTrackerV2 vs DELTA vs DELTAv2): [`../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md)
- Metriche Grezze JSON: [`../../../results/benchmark_40_metrics.json`](../../../results/benchmark_40_metrics.json)
- Report 40 Video MediaPipe: [`../../../results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md`](../../../results_mediapipe/REPORT_40_VIDEOS_BENCHMARK.md)

---

## 3. Confronto Modalità di Contesto (`bbox_grid` vs `mediapipe` — 40/40 Video)

Rieseguendo tutti i 40 video con `--context-mode mediapipe` (silhouette corporea da `1003` punti), MediaPipe migliora `Raw` (`Pitch r`: `0.185 -> 0.265`, `AccX r`: `0.243 -> 0.289`) e `Refined Cloud (SVD)` (`Pitch r`: `0.276 -> 0.293`), e dentro il run MediaPipe `Cloud (SVD)` supera `mediapipe Triplet` (`0.293` vs `0.209`). Tuttavia, **`bbox_grid` + `Refined Triplet` rimane il metodo migliore in assoluto** (`Pitch r = 0.295`, `3/30 = 10.0%` swap vs `20.0%`–`26.7%`, `28.4%` deformazione vs `34.3%`–`45.1%`, `+4.1%` errore ROM Pitch).
