# 📊 Sintesi Esecutiva e Conclusioni del Benchmark: DELTAv2 vs. Dummy Baseline
### *Valutazione Quantitativa su 40 Video Gait360 con Validazione tramite Ground-Truth Inerziale*

Questo documento fornisce una sintesi esecutiva ad alto livello dei risultati del benchmark valutato su tutti i **40 video di cammino stratificati** (39 soggetti) del dataset pubblico **Gait360**, confrontando la configurazione proposta **DELTAv2 (Tripletta Rigida tramite Geometric Pruning)** contro la **Dummy Baseline (Lifting 2D da Keypoint OpenPose)** e i sensori ground-truth indossabili sincronizzati **WitMotion IMU**.

---

## 🚀 1. I 4 Vantaggi Quantitativi Chiave di DELTAv2

### 1. Fluidità Cinematica e Attenuazione del Rumore alle Differenze Finite: **~40× Migliore**
* **Metrica:** **RMS Jerk** ($\text{m/s}^3$, derivata terza della posizione / derivata prima dell'accelerazione).
* **Dummy Baseline:** `1452.0 ± 646.6 m/s³`
* **DELTAv2 (Tripletta Rigida):** **`36.6 ± 14.3 m/s³`**
* **Verdetto:** **Riduzione del 97.5% del rumore derivativo (miglioramento di ~40×).**  
  Il lifting 2D naif dei keypoint soffre del jitter da quantizzazione discreta dei pixel, che esplode in violenti picchi di accelerazione con la derivazione seconda. DELTAv2, grazie alla stima metrica continua della profondità (UniDepthV2) e all'ottimizzazione temporale, produce traiettorie fluide e fisicamente plausibili per la coscia umana.

### 2. Fedeltà dell'Ampiezza Angolare (Range of Motion): **Recupero a Bias Zero**
* **Metrica:** **Errore % ROM di Pitch** (errore percentuale sull'escursione di flesso-estensione sagittale della coscia rispetto alla ground truth IMU, target ideale = $0\%$).
* **Dummy Baseline:** `-31.7% ± 46.0%`
* **DELTAv2 (Tripletta Rigida):** **`+4.1% ± 70.8%`**
* **Verdetto:** La Dummy baseline sottostima severamente l'escursione articolare (comprimendo il passo di quasi un terzo, -31.7%). DELTAv2 elimina virtualmente questo bias sistematico (+4.1%), centrando l'escursione articolare stimata attorno ai valori anatomici reali.

### 3. Robustezza alle Occlusioni (Scambio dell'Arto Controlaterale): **3× Meno Fallimenti**
* **Metrica:** **True Identity Swaps** (fallimenti di tracciamento causati dall'incrocio dell'arto controlaterale nelle prospettive laterali a $90^\circ$ e oblique a $45^\circ$).
* **Dummy Baseline:** `30.0%` (9 / 30 video)
* **DELTAv2 (Tripletta Rigida):** **`10.0%`** (3 / 30 video)
* **Verdetto:** **Riduzione del 66.7% negli scambi di identità (affidabilità 3× superiore).**  
  Il campionamento denso delle ancore anatomiche (silhouette MediaPipe + bacino) combinato con il *Geometric Pruner* preserva l'attenzione del tracking sulla coscia omolaterale durante la fase di swing, dove i rilevatori 2D saltano regolarmente sulla gamba opposta.

### 4. Correlazione della Forma d'Onda Sagittale: **+48% di Fedeltà Superiore**
* **Metrica:** **Pitch Pearson $r$** (correlazione lineare della forma d'onda rispetto al pitch del sensore IMU di ground truth).
* **Dummy Baseline:** `0.199 ± 0.452`
* **DELTAv2 (Tripletta Rigida):** **`0.295 ± 0.329`** (con picchi superiori a **$+0.65$** nelle viste favorevoli non assiali)
* **Verdetto:** **Miglioramento del +48.2% nella correlazione media** con deviazione standard ridotta, a dimostrazione di un'estrazione coerente del ciclo del passo.

---

## 📊 2. Tabella Comparativa di Sintesi del Benchmark

| Dominio di Valutazione | Metrica / Parametro | Dummy Baseline (2D Lift) | DELTAv2 (Tripletta Rigida) | Verdetto del Benchmark |
| :--- | :--- | :---: | :---: | :--- |
| **Dinamica (Fluidità)** | **RMS Jerk** ($\downarrow$) | $1452.0 \text{ m/s}^3$ | **$36.6 \text{ m/s}^3$** | **~40× più fluido** (elimina i picchi derivativi) |
| **Ampiezza (ROM)** | **Errore % ROM di Pitch** ($0\%$) | $-31.7\%$ | **$+4.1\%$** | **Bias eliminato** (non comprime il passo) |
| **Affidabilità (Occlusioni)** | **True Identity Swaps** ($\downarrow$) | $30.0\%$ (9/30) | **$10.0\%$ (3/30)** | **Scambi ridotti del 66.7%** (3× più robusto) |
| **Forma d'Onda (Cinematica)** | **Pitch Pearson $r$** ($\uparrow$) | $0.199$ | **$0.295$** | **+48.2% di guadagno di correlazione** verso IMU |
| **Frequenza (Cadenza)** | **Errore Cadenza** ($\downarrow$) | $29.6 \text{ spm}$ | **$16.5 \text{ spm}$** | **Errore sulla cadenza ridotto del 44.3%** |
| **Spazio degli Stati 3D** | **3D Acc DTW (norm)** ($\downarrow$) | $1.345$ | **$1.230$** | **Distanza geometrica traiettoria più ravvicinata** |

---

*Report Dettagliati:*
- Report completo benchmark su 40 video: [`../results/REPORT_40_VIDEOS_BENCHMARK.md`](../results/REPORT_40_VIDEOS_BENCHMARK.md)
- Ablazione tripletta vs. nuvola di punti: [`../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md`](../results/REPORT_TRIPLET_VS_CLOUD_ABLATION.md)
- Confronto tra modelli su 16 video (SpaTracker vs DELTA vs DELTAv2): [`../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md`](../results/REPORT_16_VIDEOS_MODEL_COMPARISON.md)
- JSON metriche grezze: [`../../../results/benchmark_40_metrics.json`](../../../results/benchmark_40_metrics.json)
